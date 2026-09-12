//! A Python frontend for Zyntax.
//!
//! Python source is parsed by `ruff_python_parser` and its AST is
//! rewritten into a [`TypedProgram`], which the runtime compiles the
//! same way it compiles anything else. No grammar of our own: Python's
//! surface (indentation, soft keywords, nested f-strings) is a poor fit
//! for a PEG and a solved problem in Ruff's hand-written parser.
//!
//! ## What this compiles
//!
//! The typed subset: the shape Codon, mypyc and RPython compile.
//! Annotated parameters, static classes, `Any` where a type is not
//! stated. What full Python adds on top of that is a runtime object
//! model, which lives above this layer rather than in it.
//!
//! ## What a Python type becomes
//!
//! `int` is `i64`, `float` is `f64`, `bool` is `bool`, `str` is
//! `String`, `None` is `Unit`. An unannotated parameter or return is
//! `Unknown` and left to inference. An integer literal is `i64` rather
//! than the provisional `i32` other frontends use, because that is what
//! `int` means here and it saves a widening at every call.

use ruff_python_ast as py;
use ruff_text_size::Ranged;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedCast, TypedDeclaration, TypedExpression, TypedFor,
    TypedFunction, TypedIf, TypedIfExpr, TypedLet, TypedLiteral, TypedParameter, TypedPattern,
    TypedRange, TypedStatement, TypedUnary, TypedWhile,
};
use zyntax_typed_ast::{
    BinaryOp, InternedString, Mutability, ParamOwnership, ParameterKind, PrimitiveType, Type,
    TypedNode, TypedProgram, UnaryOp, Visibility,
};

mod runtime;

/// Why a program could not be turned into a `TypedProgram`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Python syntax error: {0}")]
    Syntax(String),
    /// Something Python allows that this subset does not compile yet.
    /// Says what it was and where, so the gap is a fact rather than a
    /// guess.
    #[error("{what} is not supported yet (at byte offset {at})")]
    Unsupported { what: String, at: usize },
}

type Result<T> = std::result::Result<T, Error>;

/// The function a module's top-level statements become. A host runs a
/// Python program by calling this.
pub const ENTRY: &str = "__main__";

/// Python's builtins: the name a program calls, the runtime symbol it
/// links to, and what it returns. Each takes one value of any type; the
/// runtime reads the type from the box it arrives in. `print` here is
/// the no-newline half; the frontend spells Python's `print(...)` in
/// terms of both.
const BUILTINS: &[(&str, &str, PrimitiveType)] = &[
    ("println", "$Py$println", PrimitiveType::Unit),
    ("print", "$Py$print", PrimitiveType::Unit),
    ("str", "$Py$str", PrimitiveType::String),
    ("int", "$Py$int", PrimitiveType::I64),
    ("float", "$Py$float", PrimitiveType::F64),
    ("bool", "$Py$bool", PrimitiveType::Bool),
    ("len", "$Py$len", PrimitiveType::I64),
];

/// Give a runtime what a compiled Python program links against: the IO
/// plugin for string operations and the natives behind [`BUILTINS`]. A
/// host calls this once before compiling a program.
pub fn register_runtime(
    runtime: &mut zyntax_embed::TieredRuntime,
) -> std::result::Result<(), zyntax_embed::RuntimeError> {
    runtime.register_static_plugins([zrtl_io::static_plugin(), runtime::plugin()])
}

/// Parse Python source and rewrite it into a `TypedProgram`.
pub fn parse_program(source: &str) -> Result<TypedProgram> {
    let parsed = ruff_python_parser::parse_module(source)
        .map_err(|e| Error::Syntax(format!("{} at {:?}", e.error, e.location)))?;
    if let Some(first) = parsed.errors().first() {
        return Err(Error::Syntax(format!(
            "{} at {:?}",
            first.error, first.location
        )));
    }
    let module = parsed.into_syntax();

    let mut declarations = Vec::new();
    // A module's body is the program. Statements outside any `def` run
    // top to bottom when the module is executed, so they become the
    // body of the entry point, in order. A program that is nothing but
    // `def`s has no entry point of its own and calls whichever one the
    // host names.
    let mut top_level: Vec<&py::Stmt> = Vec::new();
    for stmt in &module.body {
        match stmt {
            py::Stmt::FunctionDef(f) => {
                if f.name.as_str() == ENTRY {
                    return Err(Error::Unsupported {
                        what: format!(
                            "a function named `{ENTRY}`; the module body is the program's entry"
                        ),
                        at: f.range().start().to_usize(),
                    });
                }
                let func = Lowerer::new().function(f)?;
                declarations.push(TypedNode::new(
                    TypedDeclaration::Function(func),
                    Type::Unknown,
                    span_of(f),
                ));
            }
            // A module docstring declares nothing and runs nothing.
            py::Stmt::Expr(e) if matches!(*e.value, py::Expr::StringLiteral(_)) => {}
            py::Stmt::Pass(_) => {}
            other => top_level.push(other),
        }
    }
    if !top_level.is_empty() {
        let mut lowerer = Lowerer::new();
        let mut statements = Vec::new();
        for s in &top_level {
            lowerer.stmt(s, &mut statements)?;
        }
        let span = Span::new(
            top_level[0].range().start().to_usize(),
            top_level[top_level.len() - 1].range().end().to_usize(),
        );
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(TypedFunction {
                name: intern(ENTRY),
                annotations: Vec::new(),
                effects: Vec::new(),
                with_handlers: Vec::new(),
                type_params: Vec::new(),
                params: Vec::new(),
                return_type: prim(PrimitiveType::Unit),
                body: Some(TypedBlock { statements, span }),
                visibility: Visibility::Public,
                is_async: false,
                is_fiber: false,
                is_pure: false,
                is_external: false,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                link_name: None,
                module: None,
            }),
            Type::Unknown,
            span,
        ));
    }

    // Python's builtins, declared here rather than borrowed from another
    // language's prelude.
    for (name, symbol, returns) in BUILTINS {
        declarations.push(TypedNode::new(
            TypedDeclaration::Function(TypedFunction {
                name: intern(name),
                annotations: Vec::new(),
                effects: Vec::new(),
                with_handlers: Vec::new(),
                type_params: Vec::new(),
                params: vec![TypedParameter {
                    name: intern("value"),
                    ty: Type::Any,
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: Vec::new(),
                    ownership: ParamOwnership::Copied,
                    span: Span::new(0, 0),
                }],
                return_type: prim(*returns),
                body: None,
                visibility: Visibility::Public,
                is_async: false,
                is_fiber: false,
                is_pure: false,
                is_external: true,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                link_name: Some(intern(symbol)),
                module: None,
            }),
            prim(*returns),
            Span::new(0, 0),
        ));
    }

    Ok(TypedProgram {
        declarations,
        language: Some(intern("python")),
        span: Span::new(0, source.len()),
        source_files: Vec::new(),
        type_registry: zyntax_typed_ast::TypeRegistry::new(),
    })
}

fn intern(s: &str) -> InternedString {
    InternedString::new_global(s)
}

fn span_of<N: Ranged>(node: &N) -> Span {
    let r = node.range();
    Span::new(r.start().to_usize(), r.end().to_usize())
}

fn prim(p: PrimitiveType) -> Type {
    Type::Primitive(p)
}

/// One function's worth of state: which names have been bound, so a
/// second assignment to a name is an assignment rather than a second
/// declaration.
///
/// Python has no declaration form; the first assignment in a scope is
/// the binding. The TypedAST does, so the first sight of a name becomes
/// a `let mut` and every later one an `Assign`.
struct Lowerer {
    bound: Vec<InternedString>,
    /// Hidden locals introduced so far, for naming the next one.
    temps: usize,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            bound: Vec::new(),
            temps: 0,
        }
    }

    /// A local no Python program can spell, for a value that must be
    /// evaluated once and read more than once.
    fn temp(&mut self) -> InternedString {
        self.temps += 1;
        intern(&format!("__tmp{}", self.temps))
    }

    fn function(&mut self, f: &py::StmtFunctionDef) -> Result<TypedFunction> {
        if !f.decorator_list.is_empty() {
            return Err(Error::Unsupported {
                what: "decorators".into(),
                at: f.range().start().to_usize(),
            });
        }
        if f.is_async {
            return Err(Error::Unsupported {
                what: "async def".into(),
                at: f.range().start().to_usize(),
            });
        }

        let mut params = Vec::new();
        for p in f.parameters.iter_non_variadic_params() {
            let name = intern(p.parameter.name.as_str());
            self.bound.push(name);
            let ty = match &p.parameter.annotation {
                Some(a) => self.annotation(a)?,
                None => Type::Unknown,
            };
            let default_value = match &p.default {
                Some(d) => Some(Box::new(self.expr(d)?)),
                None => None,
            };
            params.push(TypedParameter {
                name,
                ty,
                mutability: Mutability::Mutable,
                kind: if default_value.is_some() {
                    ParameterKind::Optional
                } else {
                    ParameterKind::Regular
                },
                default_value,
                attributes: Vec::new(),
                ownership: ParamOwnership::Copied,
                span: span_of(p),
            });
        }
        if f.parameters.vararg.is_some() || f.parameters.kwarg.is_some() {
            return Err(Error::Unsupported {
                what: "*args / **kwargs".into(),
                at: f.parameters.range().start().to_usize(),
            });
        }

        let return_type = match &f.returns {
            Some(r) => self.annotation(r)?,
            None => Type::Unknown,
        };

        let body = self.block(&f.body, span_of(f))?;

        Ok(TypedFunction {
            name: intern(f.name.as_str()),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params,
            return_type,
            body: Some(body),
            visibility: Visibility::Public,
            is_async: false,
            is_fiber: false,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
            link_name: None,
            module: None,
        })
    }

    /// A type annotation. Only the names this subset knows; anything
    /// else is an `Unresolved` for the resolver to find, so a
    /// user-declared class works without this file knowing about it.
    fn annotation(&self, e: &py::Expr) -> Result<Type> {
        match e {
            py::Expr::Name(n) => Ok(match n.id.as_str() {
                "int" => prim(PrimitiveType::I64),
                "float" => prim(PrimitiveType::F64),
                "bool" => prim(PrimitiveType::Bool),
                "str" => prim(PrimitiveType::String),
                "None" => prim(PrimitiveType::Unit),
                "Any" => Type::Any,
                other => Type::Unresolved(intern(other)),
            }),
            py::Expr::NoneLiteral(_) => Ok(prim(PrimitiveType::Unit)),
            other => Err(Error::Unsupported {
                what: format!("type annotation `{}`", expr_kind(other)),
                at: other.range().start().to_usize(),
            }),
        }
    }

    fn block(&mut self, stmts: &[py::Stmt], span: Span) -> Result<TypedBlock> {
        let mut statements = Vec::new();
        for s in stmts {
            self.stmt(s, &mut statements)?;
        }
        Ok(TypedBlock { statements, span })
    }

    fn stmt(&mut self, s: &py::Stmt, out: &mut Vec<TypedNode<TypedStatement>>) -> Result<()> {
        let span = span_of(s);
        let push = |out: &mut Vec<_>, st: TypedStatement| {
            out.push(TypedNode::new(st, Type::Unknown, span));
        };
        match s {
            py::Stmt::Pass(_) => {}
            py::Stmt::Return(r) => {
                let value = match &r.value {
                    Some(v) => Some(Box::new(self.expr(v)?)),
                    None => None,
                };
                push(out, TypedStatement::Return(value));
            }
            py::Stmt::Expr(e) => {
                let v = self.expr(&e.value)?;
                push(out, TypedStatement::Expression(Box::new(v)));
            }
            py::Stmt::Assign(a) => {
                if a.targets.len() != 1 {
                    return Err(Error::Unsupported {
                        what: "chained assignment".into(),
                        at: a.range().start().to_usize(),
                    });
                }
                let value = self.expr(&a.value)?;
                self.bind(&a.targets[0], None, value, span, out)?;
            }
            py::Stmt::AnnAssign(a) => {
                let ty = self.annotation(&a.annotation)?;
                let Some(v) = &a.value else {
                    return Err(Error::Unsupported {
                        what: "annotation without a value".into(),
                        at: a.range().start().to_usize(),
                    });
                };
                let value = self.expr(v)?;
                // An annotation is a hint, not a conversion: `x: float = 3`
                // binds the int 3. A declared binding type would convert,
                // so a primitive annotation is not passed on; a class name
                // is, since it only helps resolution.
                let ty = match ty {
                    Type::Primitive(_) => None,
                    other => Some(other),
                };
                self.bind(&a.target, ty, value, span, out)?;
            }
            py::Stmt::AugAssign(a) => {
                // `x += e` is `x = x + e`. Read the target once as the
                // left operand and once as the destination.
                let target = self.expr(&a.target)?;
                let rhs = self.expr(&a.value)?;
                let combined = arithmetic(a.op, target.clone(), rhs);
                let assign = TypedNode::new(
                    TypedExpression::Binary(TypedBinary {
                        op: BinaryOp::Assign,
                        left: Box::new(target),
                        right: Box::new(combined),
                    }),
                    Type::Unknown,
                    span,
                );
                push(out, TypedStatement::Expression(Box::new(assign)));
            }
            py::Stmt::If(i) => {
                let st = self.if_chain(&i.test, &i.body, &i.elif_else_clauses, span)?;
                push(out, st);
            }
            py::Stmt::While(w) => {
                if !w.orelse.is_empty() {
                    return Err(Error::Unsupported {
                        what: "while/else".into(),
                        at: w.range().start().to_usize(),
                    });
                }
                let condition = Box::new(self.expr(&w.test)?);
                let body = self.block(&w.body, span)?;
                push(
                    out,
                    TypedStatement::While(TypedWhile {
                        condition,
                        body,
                        span,
                    }),
                );
            }
            py::Stmt::For(f) => {
                let st = self.for_loop(f, span)?;
                push(out, st);
            }
            py::Stmt::Break(_) => push(out, TypedStatement::Break(None)),
            py::Stmt::Continue(_) => push(out, TypedStatement::Continue),
            other => {
                return Err(Error::Unsupported {
                    what: stmt_kind(other).to_string(),
                    at: other.range().start().to_usize(),
                })
            }
        }
        Ok(())
    }

    /// `target = value`: a `let mut` the first time a name is seen in
    /// this function, an assignment afterwards.
    fn bind(
        &mut self,
        target: &py::Expr,
        ty: Option<Type>,
        value: TypedNode<TypedExpression>,
        span: Span,
        out: &mut Vec<TypedNode<TypedStatement>>,
    ) -> Result<()> {
        let py::Expr::Name(n) = target else {
            return Err(Error::Unsupported {
                what: format!("assignment to {}", expr_kind(target)),
                at: target.range().start().to_usize(),
            });
        };
        let name = intern(n.id.as_str());
        if !self.bound.contains(&name) {
            self.bound.push(name);
            out.push(TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: ty.unwrap_or(Type::Unknown),
                    mutability: Mutability::Mutable,
                    initializer: Some(Box::new(value)),
                    span,
                }),
                Type::Unknown,
                span,
            ));
            return Ok(());
        }
        let lhs = TypedNode::new(TypedExpression::Variable(name), Type::Unknown, span);
        let assign = TypedNode::new(
            TypedExpression::Binary(TypedBinary {
                op: BinaryOp::Assign,
                left: Box::new(lhs),
                right: Box::new(value),
            }),
            Type::Unknown,
            span,
        );
        out.push(TypedNode::new(
            TypedStatement::Expression(Box::new(assign)),
            Type::Unknown,
            span,
        ));
        Ok(())
    }

    /// `if / elif / else` as nested `If`s, one per clause.
    fn if_chain(
        &mut self,
        test: &py::Expr,
        body: &[py::Stmt],
        rest: &[py::ElifElseClause],
        span: Span,
    ) -> Result<TypedStatement> {
        let condition = Box::new(self.expr(test)?);
        let then_block = self.block(body, span)?;
        let else_block = match rest.split_first() {
            None => None,
            Some((clause, tail)) => {
                let clause_span = span_of(clause);
                match &clause.test {
                    // `elif`: another If, carrying the remaining clauses.
                    Some(t) => {
                        let inner = self.if_chain(t, &clause.body, tail, clause_span)?;
                        Some(TypedBlock {
                            statements: vec![TypedNode::new(inner, Type::Unknown, clause_span)],
                            span: clause_span,
                        })
                    }
                    // `else`: the final block.
                    None => Some(self.block(&clause.body, clause_span)?),
                }
            }
        };
        Ok(TypedStatement::If(TypedIf {
            condition,
            then_block,
            else_block,
            span,
        }))
    }

    /// `for x in range(...)`, the one iterable this subset knows so far.
    fn for_loop(&mut self, f: &py::StmtFor, span: Span) -> Result<TypedStatement> {
        if f.is_async || !f.orelse.is_empty() {
            return Err(Error::Unsupported {
                what: "async for / for-else".into(),
                at: f.range().start().to_usize(),
            });
        }
        let py::Expr::Name(target) = &*f.target else {
            return Err(Error::Unsupported {
                what: "destructuring for target".into(),
                at: f.target.range().start().to_usize(),
            });
        };
        let py::Expr::Call(call) = &*f.iter else {
            return Err(Error::Unsupported {
                what: "for over anything but range()".into(),
                at: f.iter.range().start().to_usize(),
            });
        };
        let is_range = matches!(&*call.func, py::Expr::Name(n) if n.id.as_str() == "range");
        if !is_range || !call.arguments.keywords.is_empty() {
            return Err(Error::Unsupported {
                what: "for over anything but range()".into(),
                at: f.iter.range().start().to_usize(),
            });
        }
        let zero = || {
            TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(0)),
                prim(PrimitiveType::I64),
                span,
            )
        };
        let (start, end, step) = match call.arguments.args.as_ref() {
            [end] => (zero(), self.expr(end)?, None),
            [start, end] => (self.expr(start)?, self.expr(end)?, None),
            [start, end, step] => (self.expr(start)?, self.expr(end)?, Some(self.expr(step)?)),
            _ => {
                return Err(Error::Unsupported {
                    what: "range() with more than three arguments".into(),
                    at: f.iter.range().start().to_usize(),
                })
            }
        };
        let name = intern(target.id.as_str());
        self.bound.push(name);
        let body = self.block(&f.body, span)?;
        Ok(TypedStatement::For(TypedFor {
            pattern: Box::new(TypedNode::new(
                TypedPattern::Identifier {
                    name,
                    mutability: Mutability::Mutable,
                },
                prim(PrimitiveType::I64),
                span_of(target),
            )),
            // The IR's range has no step; a stepped `range(...)` is kept
            // as the call, which the counted-loop lowering reads as well.
            iterator: Box::new(TypedNode::new(
                match step {
                    None => TypedExpression::Range(TypedRange {
                        start: Some(Box::new(start)),
                        end: Some(Box::new(end)),
                        inclusive: false,
                    }),
                    Some(step) => TypedExpression::Call(TypedCall {
                        callee: Box::new(TypedNode::new(
                            TypedExpression::Variable(intern("range")),
                            Type::Unknown,
                            span_of(&*call.func),
                        )),
                        positional_args: vec![start, end, step],
                        named_args: Vec::new(),
                        type_args: Vec::new(),
                    }),
                },
                Type::Unknown,
                span_of(&*f.iter),
            )),
            body,
        }))
    }

    /// `print(a, b, ...)`: each argument, a space between, a newline
    /// after. `print()` alone is a bare newline.
    fn print_call(&mut self, c: &py::ExprCall, span: Span) -> Result<TypedNode<TypedExpression>> {
        let call = |name: &str, arg: TypedNode<TypedExpression>| {
            TypedNode::new(
                TypedExpression::Call(TypedCall {
                    callee: Box::new(TypedNode::new(
                        TypedExpression::Variable(intern(name)),
                        Type::Unknown,
                        span,
                    )),
                    positional_args: vec![arg],
                    named_args: Vec::new(),
                    type_args: Vec::new(),
                }),
                prim(PrimitiveType::Unit),
                span,
            )
        };
        let text = |s: &str| {
            TypedNode::new(
                TypedExpression::Literal(TypedLiteral::String(intern(s))),
                prim(PrimitiveType::String),
                span,
            )
        };
        let args = &c.arguments.args;
        if args.is_empty() {
            return Ok(call("println", text("")));
        }
        if args.len() == 1 {
            let a = self.expr(&args[0])?;
            return Ok(call("println", a));
        }
        // Several: a block of prints, the last ending the line.
        let mut statements = Vec::new();
        for (i, a) in args.iter().enumerate() {
            let v = self.expr(a)?;
            let is_last = i + 1 == args.len();
            let stmt = if is_last {
                call("println", v)
            } else {
                call("print", v)
            };
            statements.push(TypedNode::new(
                TypedStatement::Expression(Box::new(stmt)),
                Type::Unknown,
                span,
            ));
            if !is_last {
                statements.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(call("print", text(" ")))),
                    Type::Unknown,
                    span,
                ));
            }
        }
        Ok(TypedNode::new(
            TypedExpression::Block(TypedBlock { statements, span }),
            prim(PrimitiveType::Unit),
            span,
        ))
    }

    fn expr(&mut self, e: &py::Expr) -> Result<TypedNode<TypedExpression>> {
        let span = span_of(e);
        let node = |x: TypedExpression, ty: Type| TypedNode::new(x, ty, span);
        Ok(match e {
            py::Expr::NumberLiteral(n) => match &n.value {
                py::Number::Int(i) => {
                    let v = i.as_i64().ok_or_else(|| Error::Unsupported {
                        what: "integer literal wider than 64 bits".into(),
                        at: n.range().start().to_usize(),
                    })?;
                    node(
                        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
                        prim(PrimitiveType::I64),
                    )
                }
                py::Number::Float(f) => node(
                    TypedExpression::Literal(TypedLiteral::Float(*f)),
                    prim(PrimitiveType::F64),
                ),
                py::Number::Complex { .. } => {
                    return Err(Error::Unsupported {
                        what: "complex literal".into(),
                        at: n.range().start().to_usize(),
                    })
                }
            },
            py::Expr::BooleanLiteral(b) => node(
                TypedExpression::Literal(TypedLiteral::Bool(b.value)),
                prim(PrimitiveType::Bool),
            ),
            py::Expr::NoneLiteral(_) => node(
                TypedExpression::Literal(TypedLiteral::Null),
                prim(PrimitiveType::Unit),
            ),
            py::Expr::StringLiteral(s) => node(
                TypedExpression::Literal(TypedLiteral::String(intern(s.value.to_str()))),
                prim(PrimitiveType::String),
            ),
            py::Expr::Name(n) => node(
                TypedExpression::Variable(intern(n.id.as_str())),
                Type::Unknown,
            ),
            py::Expr::BinOp(b) => arithmetic(b.op, self.expr(&b.left)?, self.expr(&b.right)?),
            py::Expr::UnaryOp(u) => node(
                TypedExpression::Unary(TypedUnary {
                    op: match u.op {
                        py::UnaryOp::UAdd => UnaryOp::Plus,
                        py::UnaryOp::USub => UnaryOp::Minus,
                        py::UnaryOp::Not => UnaryOp::Not,
                        py::UnaryOp::Invert => UnaryOp::BitNot,
                    },
                    operand: Box::new(self.expr(&u.operand)?),
                }),
                Type::Unknown,
            ),
            // `a < b < c` is `a < b and b < c` with `b` evaluated once.
            // A middle operand that is not a name or a literal is bound
            // to a hidden local first, and the whole comparison becomes
            // a block whose value is the chain.
            py::Expr::Compare(c) => {
                let span = span_of(c);
                let mut operands = vec![self.expr(&c.left)?];
                for x in c.comparators.iter() {
                    operands.push(self.expr(x)?);
                }
                let mut bindings = Vec::new();
                for operand in operands.iter_mut().take(c.ops.len()).skip(1) {
                    if matches!(
                        operand.node,
                        TypedExpression::Variable(_) | TypedExpression::Literal(_)
                    ) {
                        continue;
                    }
                    let name = self.temp();
                    let value = std::mem::replace(
                        operand,
                        TypedNode::new(TypedExpression::Variable(name), Type::Unknown, span),
                    );
                    operand.ty = value.ty.clone();
                    bindings.push(TypedNode::new(
                        TypedStatement::Let(TypedLet {
                            name,
                            ty: value.ty.clone(),
                            mutability: Mutability::Immutable,
                            initializer: Some(Box::new(value)),
                            span,
                        }),
                        Type::Unknown,
                        span,
                    ));
                }
                let mut acc: Option<TypedNode<TypedExpression>> = None;
                for (i, op) in c.ops.iter().enumerate() {
                    let cmp = node(
                        TypedExpression::Binary(TypedBinary {
                            op: compare_op(*op, c.range().start().to_usize())?,
                            left: Box::new(operands[i].clone()),
                            right: Box::new(operands[i + 1].clone()),
                        }),
                        prim(PrimitiveType::Bool),
                    );
                    acc = Some(match acc {
                        None => cmp,
                        Some(prev) => node(
                            TypedExpression::Binary(TypedBinary {
                                op: BinaryOp::And,
                                left: Box::new(prev),
                                right: Box::new(cmp),
                            }),
                            prim(PrimitiveType::Bool),
                        ),
                    });
                }
                let chain = acc.expect("a comparison has at least one operator");
                if bindings.is_empty() {
                    chain
                } else {
                    let mut statements = bindings;
                    statements.push(TypedNode::new(
                        TypedStatement::Expression(Box::new(chain)),
                        Type::Unknown,
                        span,
                    ));
                    node(
                        TypedExpression::Block(TypedBlock { statements, span }),
                        prim(PrimitiveType::Bool),
                    )
                }
            }
            py::Expr::BoolOp(b) => {
                let op = match b.op {
                    py::BoolOp::And => BinaryOp::And,
                    py::BoolOp::Or => BinaryOp::Or,
                };
                let mut it = b.values.iter();
                let mut acc = self.expr(it.next().expect("a bool op has operands"))?;
                for v in it {
                    let rhs = self.expr(v)?;
                    acc = node(
                        TypedExpression::Binary(TypedBinary {
                            op,
                            left: Box::new(acc),
                            right: Box::new(rhs),
                        }),
                        prim(PrimitiveType::Bool),
                    );
                }
                acc
            }
            py::Expr::If(i) => node(
                TypedExpression::If(TypedIfExpr {
                    condition: Box::new(self.expr(&i.test)?),
                    then_branch: Box::new(self.expr(&i.body)?),
                    else_branch: Box::new(self.expr(&i.orelse)?),
                }),
                Type::Unknown,
            ),
            py::Expr::Call(c) => {
                if !c.arguments.keywords.is_empty() {
                    return Err(Error::Unsupported {
                        what: "keyword arguments".into(),
                        at: c.range().start().to_usize(),
                    });
                }
                // `print` writes its arguments separated by spaces and
                // ends the line. One argument is the runtime's println
                // directly; several are printed in turn with a space
                // between, then the line is ended.
                if matches!(&*c.func, py::Expr::Name(n) if n.id.as_str() == "print") {
                    return self.print_call(c, span);
                }
                let callee = self.expr(&c.func)?;
                let mut positional_args = Vec::with_capacity(c.arguments.args.len());
                for a in c.arguments.args.iter() {
                    positional_args.push(self.expr(a)?);
                }
                node(
                    TypedExpression::Call(TypedCall {
                        callee: Box::new(callee),
                        positional_args,
                        named_args: Vec::new(),
                        type_args: Vec::new(),
                    }),
                    Type::Unknown,
                )
            }
            other => {
                return Err(Error::Unsupported {
                    what: expr_kind(other).to_string(),
                    at: other.range().start().to_usize(),
                })
            }
        })
    }
}

/// `left op right`. Python's `/` always divides as floats, so both
/// sides are converted first; a float operand converts to itself.
fn arithmetic(
    op: py::Operator,
    left: TypedNode<TypedExpression>,
    right: TypedNode<TypedExpression>,
) -> TypedNode<TypedExpression> {
    let to_float = |e: TypedNode<TypedExpression>| {
        let span = e.span;
        TypedNode::new(
            TypedExpression::Cast(TypedCast {
                expr: Box::new(e),
                target_type: prim(PrimitiveType::F64),
            }),
            prim(PrimitiveType::F64),
            span,
        )
    };
    let (left, right, ty) = match op {
        py::Operator::Div => (to_float(left), to_float(right), prim(PrimitiveType::F64)),
        _ => (left, right, Type::Unknown),
    };
    let span = Span::new(left.span.start, right.span.end);
    TypedNode::new(
        TypedExpression::Binary(TypedBinary {
            op: operator(op),
            left: Box::new(left),
            right: Box::new(right),
        }),
        ty,
        span,
    )
}

fn operator(op: py::Operator) -> BinaryOp {
    match op {
        py::Operator::Add => BinaryOp::Add,
        py::Operator::Sub => BinaryOp::Sub,
        py::Operator::Mult => BinaryOp::Mul,
        py::Operator::MatMult => BinaryOp::MatMul,
        py::Operator::Div => BinaryOp::Div,
        py::Operator::FloorDiv => BinaryOp::FloorDiv,
        py::Operator::Mod => BinaryOp::FloorRem,
        py::Operator::Pow => BinaryOp::Pow,
        py::Operator::LShift => BinaryOp::Shl,
        py::Operator::RShift => BinaryOp::Shr,
        py::Operator::BitOr => BinaryOp::BitOr,
        py::Operator::BitXor => BinaryOp::BitXor,
        py::Operator::BitAnd => BinaryOp::BitAnd,
    }
}

fn compare_op(op: py::CmpOp, at: usize) -> Result<BinaryOp> {
    Ok(match op {
        py::CmpOp::Eq => BinaryOp::Eq,
        py::CmpOp::NotEq => BinaryOp::Ne,
        py::CmpOp::Lt => BinaryOp::Lt,
        py::CmpOp::LtE => BinaryOp::Le,
        py::CmpOp::Gt => BinaryOp::Gt,
        py::CmpOp::GtE => BinaryOp::Ge,
        py::CmpOp::Is | py::CmpOp::IsNot | py::CmpOp::In | py::CmpOp::NotIn => {
            return Err(Error::Unsupported {
                what: format!("`{}` comparison", op.as_str()),
                at,
            })
        }
    })
}

fn stmt_kind(s: &py::Stmt) -> &'static str {
    match s {
        py::Stmt::FunctionDef(_) => "def",
        py::Stmt::ClassDef(_) => "class",
        py::Stmt::Return(_) => "return",
        py::Stmt::Delete(_) => "del",
        py::Stmt::TypeAlias(_) => "type alias",
        py::Stmt::Assign(_) => "assignment",
        py::Stmt::AugAssign(_) => "augmented assignment",
        py::Stmt::AnnAssign(_) => "annotated assignment",
        py::Stmt::For(_) => "for",
        py::Stmt::While(_) => "while",
        py::Stmt::If(_) => "if",
        py::Stmt::With(_) => "with",
        py::Stmt::Match(_) => "match",
        py::Stmt::Raise(_) => "raise",
        py::Stmt::Try(_) => "try",
        py::Stmt::Assert(_) => "assert",
        py::Stmt::Import(_) | py::Stmt::ImportFrom(_) => "import",
        py::Stmt::Global(_) => "global",
        py::Stmt::Nonlocal(_) => "nonlocal",
        py::Stmt::Expr(_) => "expression statement",
        py::Stmt::Pass(_) => "pass",
        py::Stmt::Break(_) => "break",
        py::Stmt::Continue(_) => "continue",
        py::Stmt::IpyEscapeCommand(_) => "IPython escape",
    }
}

fn expr_kind(e: &py::Expr) -> &'static str {
    match e {
        py::Expr::BoolOp(_) => "boolean operator",
        py::Expr::Named(_) => "walrus",
        py::Expr::BinOp(_) => "binary operator",
        py::Expr::UnaryOp(_) => "unary operator",
        py::Expr::Lambda(_) => "lambda",
        py::Expr::If(_) => "conditional expression",
        py::Expr::Dict(_) => "dict literal",
        py::Expr::Set(_) => "set literal",
        py::Expr::ListComp(_) => "list comprehension",
        py::Expr::SetComp(_) => "set comprehension",
        py::Expr::DictComp(_) => "dict comprehension",
        py::Expr::Generator(_) => "generator expression",
        py::Expr::Await(_) => "await",
        py::Expr::Yield(_) | py::Expr::YieldFrom(_) => "yield",
        py::Expr::Compare(_) => "comparison",
        py::Expr::Call(_) => "call",
        py::Expr::FString(_) => "f-string",
        py::Expr::TString(_) => "t-string",
        py::Expr::StringLiteral(_) => "string",
        py::Expr::BytesLiteral(_) => "bytes",
        py::Expr::NumberLiteral(_) => "number",
        py::Expr::BooleanLiteral(_) => "bool",
        py::Expr::NoneLiteral(_) => "None",
        py::Expr::EllipsisLiteral(_) => "...",
        py::Expr::Attribute(_) => "attribute access",
        py::Expr::Subscript(_) => "subscript",
        py::Expr::Starred(_) => "starred expression",
        py::Expr::Name(_) => "name",
        py::Expr::List(_) => "list literal",
        py::Expr::Tuple(_) => "tuple",
        py::Expr::Slice(_) => "slice",
        py::Expr::IpyEscapeCommand(_) => "IPython escape",
    }
}
