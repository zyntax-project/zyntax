//! Python statements and expressions to typed nodes.
//!
//! Every node this emits carries the IR type the inference in
//! [`crate::types`] assigned, and every place a value crosses between
//! two types gets an explicit conversion: a cast between numbers, a box
//! into `Any`, a checked unbox out of it. Nothing is left for the
//! compiler to guess. Python's own semantics that the IR does not have
//! (how a bool prints, what `"a" * 3` is) are calls into the prelude.

use crate::types::{self, Locals, Module, Sig, Ty, Typer};
use crate::{intern, prim, span_of, Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedCast, TypedExpression, TypedFor, TypedFunction,
    TypedIf, TypedIfExpr, TypedLet, TypedLiteral, TypedParameter, TypedPattern, TypedRange,
    TypedStatement, TypedUnary, TypedWhile,
};
use zyntax_typed_ast::{
    BinaryOp, InternedString, Mutability, ParamOwnership, ParameterKind, PrimitiveType, Type,
    TypedNode, UnaryOp, Visibility,
};

type Node = TypedNode<TypedExpression>;
type Stmt = TypedNode<TypedStatement>;

/// The IR type a static type is carried as.
pub(crate) fn ir(ty: Ty) -> Type {
    match ty {
        Ty::Int => prim(PrimitiveType::I64),
        Ty::Float => prim(PrimitiveType::F64),
        Ty::Bool => prim(PrimitiveType::Bool),
        Ty::Str => prim(PrimitiveType::String),
        Ty::None => prim(PrimitiveType::Unit),
        Ty::Object | Ty::Unknown => Type::Any,
    }
}

/// A lowered expression and the static type it has.
pub(crate) struct Val {
    pub(crate) node: Node,
    pub(crate) ty: Ty,
}

fn node(x: TypedExpression, ty: Ty, span: Span) -> Node {
    TypedNode::new(x, ir(ty), span)
}

fn var(name: InternedString, ty: Ty, span: Span) -> Node {
    node(TypedExpression::Variable(name), ty, span)
}

fn int_lit(v: i64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        Ty::Int,
        span,
    )
}

fn str_lit(s: &str, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::String(intern(s))),
        Ty::Str,
        span,
    )
}

fn binary(op: BinaryOp, left: Node, right: Node, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::Binary(TypedBinary {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }),
        ty,
        span,
    )
}

fn call(name: &str, args: Vec<Node>, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::Call(TypedCall {
            callee: Box::new(node(
                TypedExpression::Variable(intern(name)),
                Ty::Unknown,
                span,
            )),
            positional_args: args,
            named_args: Vec::new(),
            type_args: Vec::new(),
        }),
        ty,
        span,
    )
}

fn cast(value: Node, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: ir(ty),
        }),
        ty,
        span,
    )
}

fn unsupported<T>(what: impl Into<String>, at: &impl Ranged) -> Result<T> {
    Err(Error::Unsupported {
        what: what.into(),
        at: at.range().start().to_usize(),
    })
}

/// One function's lowering state.
pub(crate) struct Lowerer<'m> {
    module: &'m Module,
    sig: Sig,
    locals: Locals,
    /// Names already bound in emission order, so the first sight of a
    /// name is its `let` and later ones are assignments.
    bound: Vec<InternedString>,
    temps: usize,
}

impl<'m> Lowerer<'m> {
    pub(crate) fn new(module: &'m Module, sig: Sig, locals: Locals) -> Self {
        let bound = sig.params.iter().map(|(n, _)| intern(n)).collect();
        Self {
            module,
            sig,
            locals,
            bound,
            temps: 0,
        }
    }

    fn typer(&self) -> Typer<'_> {
        Typer {
            module: self.module,
            vars: &self.locals.vars,
        }
    }

    fn ty_of(&self, e: &py::Expr) -> Ty {
        self.typer().expr(e)
    }

    fn var_ty(&self, name: &str) -> Ty {
        self.locals.vars.get(name).copied().unwrap_or(Ty::Object)
    }

    /// A local no Python program can spell.
    fn temp(&mut self) -> InternedString {
        self.temps += 1;
        intern(&format!("__tmp{}", self.temps))
    }

    // ─── Functions ──────────────────────────────────────────────────

    pub(crate) fn function(&mut self, f: &py::StmtFunctionDef) -> Result<TypedFunction> {
        if !f.decorator_list.is_empty() {
            return unsupported("decorators", f);
        }
        if f.is_async {
            return unsupported("async def", f);
        }
        if f.parameters.vararg.is_some() || f.parameters.kwarg.is_some() {
            return unsupported("*args / **kwargs", &*f.parameters);
        }
        let mut params = Vec::new();
        let mut prologue = Vec::new();
        for (p, (_, declared)) in f
            .parameters
            .iter_non_variadic_params()
            .zip(self.sig.params.clone())
        {
            let name = intern(p.parameter.name.as_str());
            let default_value = match &p.default {
                Some(d) => {
                    let v = self.expr(d)?;
                    Some(Box::new(self.coerce(v, declared)))
                }
                None => None,
            };
            params.push(TypedParameter {
                name,
                ty: ir(declared),
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
            // A parameter the body also assigns another type to lives as
            // an object from the start.
            let local = self.var_ty(p.parameter.name.as_str());
            if local != declared {
                let span = span_of(p);
                let value = self.coerce(
                    Val {
                        node: var(name, declared, span),
                        ty: declared,
                    },
                    local,
                );
                prologue.push(TypedNode::new(
                    TypedStatement::Let(TypedLet {
                        name,
                        ty: ir(local),
                        mutability: Mutability::Mutable,
                        initializer: Some(Box::new(value)),
                        span,
                    }),
                    Type::Unknown,
                    span,
                ));
            }
        }

        let span = span_of(f);
        let mut statements = prologue;
        for s in &f.body {
            self.stmt(s, &mut statements)?;
        }
        Ok(TypedFunction {
            name: intern(f.name.as_str()),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params,
            return_type: ir(self.sig.ret),
            body: Some(TypedBlock { statements, span }),
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

    /// The module body as the entry function's statements.
    pub(crate) fn body(&mut self, stmts: &[&py::Stmt]) -> Result<Vec<Stmt>> {
        let mut out = Vec::new();
        for s in stmts {
            self.stmt(s, &mut out)?;
        }
        Ok(out)
    }

    // ─── Conversions ────────────────────────────────────────────────

    /// `v` as a value of type `target`, converting where the two differ.
    pub(crate) fn coerce(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        match (v.ty, target) {
            (a, b) if a == b => v.node,
            (_, Ty::Unknown) => v.node,
            (Ty::Int | Ty::Bool, Ty::Float) => cast(v.node, Ty::Float, span),
            (Ty::Bool, Ty::Int) => cast(v.node, Ty::Int, span),
            (Ty::Int, Ty::Bool) => binary(BinaryOp::Ne, v.node, int_lit(0, span), Ty::Bool, span),
            // Into the dynamic world: a box. Out of it: a checked read.
            (_, Ty::Object) => cast(v.node, Ty::Object, span),
            (Ty::Object, _) => cast(v.node, target, span),
            // Two primitives with nothing between them. Type inference
            // never asks for this; a Python program that does is
            // treating the value dynamically, so it goes through a box.
            (_, _) => cast(cast(v.node, Ty::Object, span), target, span),
        }
    }

    fn expr_as(&mut self, e: &py::Expr, target: Ty) -> Result<Node> {
        let v = self.expr(e)?;
        Ok(self.coerce(v, target))
    }

    /// `bool(v)`: the value as a condition.
    fn truthy(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Bool => v.node,
            Ty::Int => binary(BinaryOp::Ne, v.node, int_lit(0, span), Ty::Bool, span),
            Ty::Float => binary(
                BinaryOp::Ne,
                v.node,
                node(
                    TypedExpression::Literal(TypedLiteral::Float(0.0)),
                    Ty::Float,
                    span,
                ),
                Ty::Bool,
                span,
            ),
            Ty::Str => call("__py_truthy_str", vec![v.node], Ty::Bool, span),
            Ty::None => node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                Ty::Bool,
                span,
            ),
            Ty::Object | Ty::Unknown => call("__py_truthy_any", vec![v.node], Ty::Bool, span),
        }
    }

    /// `str(v)`.
    fn str_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Int => call("__py_str_int", vec![v.node], Ty::Str, span),
            Ty::Float => call("__py_str_float", vec![v.node], Ty::Str, span),
            Ty::Bool => call("__py_str_bool", vec![v.node], Ty::Str, span),
            Ty::Str => v.node,
            Ty::None => call("__py_str_none", vec![], Ty::Str, span),
            Ty::Object | Ty::Unknown => call("__py_str_any", vec![v.node], Ty::Str, span),
        }
    }

    // ─── Statements ─────────────────────────────────────────────────

    fn block(&mut self, stmts: &[py::Stmt], span: Span) -> Result<TypedBlock> {
        let mut statements = Vec::new();
        for s in stmts {
            self.stmt(s, &mut statements)?;
        }
        Ok(TypedBlock { statements, span })
    }

    fn stmt(&mut self, s: &py::Stmt, out: &mut Vec<Stmt>) -> Result<()> {
        let span = span_of(s);
        let push = |out: &mut Vec<Stmt>, st: TypedStatement| {
            out.push(TypedNode::new(st, Type::Unknown, span));
        };
        match s {
            py::Stmt::Pass(_) => {}
            py::Stmt::Return(r) => {
                let value = match &r.value {
                    Some(v) => {
                        let ret = self.sig.ret;
                        Some(Box::new(self.expr_as(v, ret)?))
                    }
                    None if self.sig.ret == Ty::None => None,
                    None => {
                        let none = Val {
                            node: node(
                                TypedExpression::Literal(TypedLiteral::Null),
                                Ty::None,
                                span,
                            ),
                            ty: Ty::None,
                        };
                        let ret = self.sig.ret;
                        Some(Box::new(self.coerce(none, ret)))
                    }
                };
                push(out, TypedStatement::Return(value));
            }
            py::Stmt::Expr(e) => {
                let v = self.expr(&e.value)?;
                push(out, TypedStatement::Expression(Box::new(v.node)));
            }
            py::Stmt::Assign(a) => {
                if a.targets.len() != 1 {
                    return unsupported("chained assignment", a);
                }
                let value = self.expr(&a.value)?;
                self.bind(&a.targets[0], value, span, out)?;
            }
            py::Stmt::AnnAssign(a) => {
                let Some(v) = &a.value else {
                    return unsupported("annotation without a value", a);
                };
                // An annotation is a hint, not a conversion: `x: float = 3`
                // binds the int 3.
                let value = self.expr(v)?;
                self.bind(&a.target, value, span, out)?;
            }
            py::Stmt::AugAssign(a) => {
                let lhs = self.expr(&a.target)?;
                let rhs = self.expr(&a.value)?;
                let combined = self.arithmetic(a.op, lhs, rhs, &a.value, span)?;
                self.bind(&a.target, combined, span, out)?;
            }
            py::Stmt::If(i) => {
                let st = self.if_chain(&i.test, &i.body, &i.elif_else_clauses, span)?;
                push(out, st);
            }
            py::Stmt::While(w) => {
                if !w.orelse.is_empty() {
                    return unsupported("while/else", w);
                }
                let cond = self.expr(&w.test)?;
                let condition = Box::new(self.truthy(cond));
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
            other => return unsupported(types::stmt_kind(other), other),
        }
        Ok(())
    }

    /// `target = value`: a `let` the first time a name is seen in this
    /// function, an assignment afterwards; the value converted to the
    /// type inference gave the name.
    fn bind(
        &mut self,
        target: &py::Expr,
        value: Val,
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Result<()> {
        let py::Expr::Name(n) = target else {
            return unsupported(
                format!("assignment to {}", types::expr_kind(target)),
                target,
            );
        };
        let ty = self.var_ty(n.id.as_str());
        let value = self.coerce(value, ty);
        let name = intern(n.id.as_str());
        if !self.bound.contains(&name) {
            self.bound.push(name);
            out.push(TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: ir(ty),
                    mutability: Mutability::Mutable,
                    initializer: Some(Box::new(value)),
                    span,
                }),
                Type::Unknown,
                span,
            ));
            return Ok(());
        }
        let assign = binary(BinaryOp::Assign, var(name, ty, span), value, Ty::None, span);
        out.push(TypedNode::new(
            TypedStatement::Expression(Box::new(assign)),
            Type::Unknown,
            span,
        ));
        Ok(())
    }

    fn if_chain(
        &mut self,
        test: &py::Expr,
        body: &[py::Stmt],
        rest: &[py::ElifElseClause],
        span: Span,
    ) -> Result<TypedStatement> {
        let cond = self.expr(test)?;
        let condition = Box::new(self.truthy(cond));
        let then_block = self.block(body, span)?;
        let else_block = match rest.split_first() {
            None => None,
            Some((clause, tail)) => {
                let clause_span = span_of(clause);
                match &clause.test {
                    Some(t) => {
                        let inner = self.if_chain(t, &clause.body, tail, clause_span)?;
                        Some(TypedBlock {
                            statements: vec![TypedNode::new(inner, Type::Unknown, clause_span)],
                            span: clause_span,
                        })
                    }
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
            return unsupported("async for / for-else", f);
        }
        let py::Expr::Name(target) = &*f.target else {
            return unsupported("destructuring for target", &*f.target);
        };
        let py::Expr::Call(c) = &*f.iter else {
            return unsupported("for over anything but range()", &*f.iter);
        };
        if !types::is_name(&c.func, "range") || !c.arguments.keywords.is_empty() {
            return unsupported("for over anything but range()", &*f.iter);
        }
        let (start, end, step) = match c.arguments.args.as_ref() {
            [end] => (int_lit(0, span), self.expr_as(end, Ty::Int)?, None),
            [start, end] => (
                self.expr_as(start, Ty::Int)?,
                self.expr_as(end, Ty::Int)?,
                None,
            ),
            [start, end, step] => (
                self.expr_as(start, Ty::Int)?,
                self.expr_as(end, Ty::Int)?,
                Some(self.expr_as(step, Ty::Int)?),
            ),
            _ => return unsupported("range() with more than three arguments", &*f.iter),
        };
        let name = intern(target.id.as_str());
        if !self.bound.contains(&name) {
            self.bound.push(name);
        }
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
                        callee: Box::new(var(intern("range"), Ty::Unknown, span_of(&*c.func))),
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

    // ─── Expressions ────────────────────────────────────────────────

    pub(crate) fn expr(&mut self, e: &py::Expr) -> Result<Val> {
        let span = span_of(e);
        let ty = self.ty_of(e);
        let lit = |x: TypedExpression, ty: Ty| Val {
            node: node(x, ty, span),
            ty,
        };
        Ok(match e {
            py::Expr::NumberLiteral(n) => match &n.value {
                py::Number::Int(i) => {
                    let Some(v) = i.as_i64() else {
                        return unsupported("integer literal wider than 64 bits", n);
                    };
                    lit(
                        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
                        Ty::Int,
                    )
                }
                py::Number::Float(f) => {
                    lit(TypedExpression::Literal(TypedLiteral::Float(*f)), Ty::Float)
                }
                py::Number::Complex { .. } => return unsupported("complex literal", n),
            },
            py::Expr::BooleanLiteral(b) => lit(
                TypedExpression::Literal(TypedLiteral::Bool(b.value)),
                Ty::Bool,
            ),
            py::Expr::NoneLiteral(_) => lit(TypedExpression::Literal(TypedLiteral::Null), Ty::None),
            py::Expr::StringLiteral(s) => lit(
                TypedExpression::Literal(TypedLiteral::String(intern(s.value.to_str()))),
                Ty::Str,
            ),
            py::Expr::Name(n) => Val {
                node: var(intern(n.id.as_str()), ty, span),
                ty,
            },
            py::Expr::BinOp(b) => {
                let l = self.expr(&b.left)?;
                let r = self.expr(&b.right)?;
                self.arithmetic(b.op, l, r, &b.right, span)?
            }
            py::Expr::UnaryOp(u) => self.unary(u, span)?,
            py::Expr::Compare(c) => self.compare(c, span)?,
            py::Expr::BoolOp(b) => self.bool_op(b, ty, span)?,
            py::Expr::If(i) => {
                let cond = self.expr(&i.test)?;
                let condition = self.truthy(cond);
                let then_branch = self.expr_as(&i.body, ty)?;
                let else_branch = self.expr_as(&i.orelse, ty)?;
                Val {
                    node: node(
                        TypedExpression::If(TypedIfExpr {
                            condition: Box::new(condition),
                            then_branch: Box::new(then_branch),
                            else_branch: Box::new(else_branch),
                        }),
                        ty,
                        span,
                    ),
                    ty,
                }
            }
            py::Expr::Call(c) => self.call(c, ty, span)?,
            other => return unsupported(types::expr_kind(other), other),
        })
    }

    fn unary(&mut self, u: &py::ExprUnaryOp, span: Span) -> Result<Val> {
        let operand = self.expr(&u.operand)?;
        Ok(match u.op {
            py::UnaryOp::Not => {
                let cond = self.truthy(operand);
                Val {
                    node: node(
                        TypedExpression::Unary(TypedUnary {
                            op: UnaryOp::Not,
                            operand: Box::new(cond),
                        }),
                        Ty::Bool,
                        span,
                    ),
                    ty: Ty::Bool,
                }
            }
            py::UnaryOp::UAdd | py::UnaryOp::USub | py::UnaryOp::Invert => {
                let op = match u.op {
                    py::UnaryOp::USub => UnaryOp::Minus,
                    py::UnaryOp::Invert => UnaryOp::BitNot,
                    _ => UnaryOp::Plus,
                };
                match operand.ty {
                    Ty::Object | Ty::Unknown => {
                        let name = match op {
                            UnaryOp::Minus => "__py_neg_any",
                            UnaryOp::BitNot => "__py_invert_any",
                            _ => "__py_pos_any",
                        };
                        Val {
                            node: call(name, vec![operand.node], Ty::Object, span),
                            ty: Ty::Object,
                        }
                    }
                    _ => {
                        // Bools count as ints under arithmetic.
                        let ty = if operand.ty == Ty::Bool {
                            Ty::Int
                        } else {
                            operand.ty
                        };
                        let inner = self.coerce(operand, ty);
                        Val {
                            node: node(
                                TypedExpression::Unary(TypedUnary {
                                    op,
                                    operand: Box::new(inner),
                                }),
                                ty,
                                span,
                            ),
                            ty,
                        }
                    }
                }
            }
        })
    }

    /// `left op right`, on the operands' types.
    fn arithmetic(
        &mut self,
        op: py::Operator,
        left: Val,
        right: Val,
        right_expr: &py::Expr,
        span: Span,
    ) -> Result<Val> {
        let ty = types::binop(op, left.ty, right.ty, right_expr);
        // Strings have their own operators.
        if left.ty == Ty::Str || right.ty == Ty::Str {
            match (op, left.ty, right.ty) {
                (py::Operator::Add, Ty::Str, Ty::Str) => {
                    return Ok(Val {
                        node: binary(BinaryOp::Add, left.node, right.node, Ty::Str, span),
                        ty: Ty::Str,
                    })
                }
                (py::Operator::Mult, Ty::Str, Ty::Int | Ty::Bool) => {
                    let n = self.coerce(right, Ty::Int);
                    return Ok(Val {
                        node: call("__py_str_mul", vec![left.node, n], Ty::Str, span),
                        ty: Ty::Str,
                    });
                }
                (py::Operator::Mult, Ty::Int | Ty::Bool, Ty::Str) => {
                    let n = self.coerce(left, Ty::Int);
                    return Ok(Val {
                        node: call("__py_str_mul", vec![right.node, n], Ty::Str, span),
                        ty: Ty::Str,
                    });
                }
                _ => {}
            }
        }
        if ty == Ty::Object {
            // At least one side is dynamic: the runtime picks the
            // operation from the tags.
            let l = self.coerce(left, Ty::Object);
            let r = self.coerce(right, Ty::Object);
            let code = int_lit(types::arith_code(op), span);
            return Ok(Val {
                node: call("__py_arith_any", vec![code, l, r], Ty::Object, span),
                ty: Ty::Object,
            });
        }
        // Numbers. Operands meet at the result's type, except that a
        // comparison-free `/` computes in float regardless.
        let operand_ty = match op {
            py::Operator::Div => Ty::Float,
            _ if ty == Ty::Float => Ty::Float,
            _ => Ty::Int,
        };
        let l = self.coerce(left, operand_ty);
        let r = self.coerce(right, operand_ty);
        let bin = match op {
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
        };
        let result = binary(bin, l, r, operand_ty, span);
        // `**` with a negative exponent is a float in Python even for
        // integer operands; the lowering computes it as one.
        let result = if ty == Ty::Float && operand_ty == Ty::Int {
            cast(result, Ty::Float, span)
        } else {
            result
        };
        Ok(Val { node: result, ty })
    }

    /// One comparison between two typed values.
    fn compare_one(&mut self, op: py::CmpOp, left: Val, right: Val, span: Span) -> Result<Node> {
        let negate = |n: Node| {
            node(
                TypedExpression::Unary(TypedUnary {
                    op: UnaryOp::Not,
                    operand: Box::new(n),
                }),
                Ty::Bool,
                span,
            )
        };
        match op {
            py::CmpOp::In | py::CmpOp::NotIn => {
                let n = if left.ty == Ty::Str && right.ty == Ty::Str {
                    call(
                        "__py_str_contains",
                        vec![right.node, left.node],
                        Ty::Bool,
                        span,
                    )
                } else {
                    let item = self.coerce(left, Ty::Object);
                    let container = self.coerce(right, Ty::Object);
                    call("__py_contains_any", vec![container, item], Ty::Bool, span)
                };
                return Ok(if op == py::CmpOp::NotIn { negate(n) } else { n });
            }
            py::CmpOp::Is | py::CmpOp::IsNot => {
                let n = match (left.ty, right.ty) {
                    (Ty::None, Ty::None) => node(
                        TypedExpression::Literal(TypedLiteral::Bool(true)),
                        Ty::Bool,
                        span,
                    ),
                    (Ty::None, t) | (t, Ty::None) if t != Ty::Object => node(
                        TypedExpression::Literal(TypedLiteral::Bool(false)),
                        Ty::Bool,
                        span,
                    ),
                    _ => {
                        let l = self.coerce(left, Ty::Object);
                        let r = self.coerce(right, Ty::Object);
                        call("__py_is_any", vec![l, r], Ty::Bool, span)
                    }
                };
                return Ok(if op == py::CmpOp::IsNot { negate(n) } else { n });
            }
            _ => {}
        }
        if left.ty == Ty::Str && right.ty == Ty::Str {
            let n = match op {
                py::CmpOp::Eq => call("__py_str_eq", vec![left.node, right.node], Ty::Bool, span),
                py::CmpOp::NotEq => negate(call(
                    "__py_str_eq",
                    vec![left.node, right.node],
                    Ty::Bool,
                    span,
                )),
                py::CmpOp::Lt => call("__py_str_lt", vec![left.node, right.node], Ty::Bool, span),
                py::CmpOp::Gt => call("__py_str_lt", vec![right.node, left.node], Ty::Bool, span),
                py::CmpOp::LtE => negate(call(
                    "__py_str_lt",
                    vec![right.node, left.node],
                    Ty::Bool,
                    span,
                )),
                py::CmpOp::GtE => negate(call(
                    "__py_str_lt",
                    vec![left.node, right.node],
                    Ty::Bool,
                    span,
                )),
                _ => unreachable!(),
            };
            return Ok(n);
        }
        if left.ty.is_numeric() && right.ty.is_numeric() {
            let operand_ty = if left.ty == Ty::Float || right.ty == Ty::Float {
                Ty::Float
            } else {
                Ty::Int
            };
            let l = self.coerce(left, operand_ty);
            let r = self.coerce(right, operand_ty);
            let bin = match op {
                py::CmpOp::Eq => BinaryOp::Eq,
                py::CmpOp::NotEq => BinaryOp::Ne,
                py::CmpOp::Lt => BinaryOp::Lt,
                py::CmpOp::LtE => BinaryOp::Le,
                py::CmpOp::Gt => BinaryOp::Gt,
                py::CmpOp::GtE => BinaryOp::Ge,
                _ => unreachable!(),
            };
            return Ok(binary(bin, l, r, Ty::Bool, span));
        }
        // Mixed with None or dynamic: `==` is answered by the runtime
        // and ordering too.
        let l = self.coerce(left, Ty::Object);
        let r = self.coerce(right, Ty::Object);
        Ok(match op {
            py::CmpOp::Eq => call("__py_eq_any", vec![l, r], Ty::Bool, span),
            py::CmpOp::NotEq => negate(call("__py_eq_any", vec![l, r], Ty::Bool, span)),
            py::CmpOp::Lt => call("__py_lt_any", vec![l, r], Ty::Bool, span),
            py::CmpOp::Gt => call("__py_lt_any", vec![r, l], Ty::Bool, span),
            py::CmpOp::LtE => negate(call("__py_lt_any", vec![r, l], Ty::Bool, span)),
            py::CmpOp::GtE => negate(call("__py_lt_any", vec![l, r], Ty::Bool, span)),
            _ => unreachable!(),
        })
    }

    /// `a < b < c` is `a < b and b < c` with `b` evaluated once: a
    /// middle operand that is not a name or a literal is bound to a
    /// hidden local first, and the whole comparison becomes a block
    /// whose value is the chain.
    fn compare(&mut self, c: &py::ExprCompare, span: Span) -> Result<Val> {
        let mut operands = vec![self.expr(&c.left)?];
        for x in c.comparators.iter() {
            operands.push(self.expr(x)?);
        }
        let mut bindings = Vec::new();
        for operand in operands.iter_mut().take(c.ops.len()).skip(1) {
            if matches!(
                operand.node.node,
                TypedExpression::Variable(_) | TypedExpression::Literal(_)
            ) {
                continue;
            }
            let name = self.temp();
            let ty = operand.ty;
            let value = std::mem::replace(
                operand,
                Val {
                    node: var(name, ty, span),
                    ty,
                },
            );
            bindings.push(TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name,
                    ty: ir(ty),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(value.node)),
                    span,
                }),
                Type::Unknown,
                span,
            ));
        }
        // Operands are reused on both sides of a chain, so each is
        // cloned into its comparisons.
        let mut acc: Option<Node> = None;
        for (i, op) in c.ops.iter().enumerate() {
            let left = Val {
                node: operands[i].node.clone(),
                ty: operands[i].ty,
            };
            let right = Val {
                node: operands[i + 1].node.clone(),
                ty: operands[i + 1].ty,
            };
            let cmp = self.compare_one(*op, left, right, span)?;
            acc = Some(match acc {
                None => cmp,
                Some(prev) => binary(BinaryOp::And, prev, cmp, Ty::Bool, span),
            });
        }
        let chain = acc.expect("a comparison has at least one operator");
        let node = if bindings.is_empty() {
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
                Ty::Bool,
                span,
            )
        };
        Ok(Val { node, ty: Ty::Bool })
    }

    /// `a and b` / `a or b`. Between bools it is the IR's short-circuit
    /// operator. Otherwise Python's value semantics: the result is the
    /// operand that decided, so `a` is bound once, tested, and either it
    /// or `b` is the value, both at the join type.
    fn bool_op(&mut self, b: &py::ExprBoolOp, ty: Ty, span: Span) -> Result<Val> {
        let is_and = b.op == py::BoolOp::And;
        let mut vals = Vec::with_capacity(b.values.len());
        for v in &b.values {
            vals.push(self.expr(v)?);
        }
        if ty == Ty::Bool {
            let op = if is_and { BinaryOp::And } else { BinaryOp::Or };
            let mut it = vals.into_iter();
            let mut acc = it.next().expect("a bool op has operands").node;
            for v in it {
                acc = binary(op, acc, v.node, Ty::Bool, span);
            }
            return Ok(Val {
                node: acc,
                ty: Ty::Bool,
            });
        }
        // Right to left, so `a or b or c` nests as `a or (b or c)`.
        let mut it = vals.into_iter().rev();
        let last = it.next().expect("a bool op has operands");
        let mut acc = self.coerce(last, ty);
        for v in it {
            let name = self.temp();
            let first = self.coerce(v, ty);
            let bound = Val {
                node: var(name, ty, span),
                ty,
            };
            let test = self.truthy(Val {
                node: var(name, ty, span),
                ty,
            });
            let (then_branch, else_branch) = if is_and {
                (acc, bound.node)
            } else {
                (bound.node, acc)
            };
            let pick = node(
                TypedExpression::If(TypedIfExpr {
                    condition: Box::new(test),
                    then_branch: Box::new(then_branch),
                    else_branch: Box::new(else_branch),
                }),
                ty,
                span,
            );
            acc = node(
                TypedExpression::Block(TypedBlock {
                    statements: vec![
                        TypedNode::new(
                            TypedStatement::Let(TypedLet {
                                name,
                                ty: ir(ty),
                                mutability: Mutability::Immutable,
                                initializer: Some(Box::new(first)),
                                span,
                            }),
                            Type::Unknown,
                            span,
                        ),
                        TypedNode::new(
                            TypedStatement::Expression(Box::new(pick)),
                            Type::Unknown,
                            span,
                        ),
                    ],
                    span,
                }),
                ty,
                span,
            );
        }
        Ok(Val { node: acc, ty })
    }

    /// A call: `print`, a conversion builtin, or a function the module
    /// defines.
    fn call(&mut self, c: &py::ExprCall, ty: Ty, span: Span) -> Result<Val> {
        if !c.arguments.keywords.is_empty() {
            return unsupported("keyword arguments", c);
        }
        let args = &c.arguments.args;
        if let py::Expr::Name(n) = &*c.func {
            let name = n.id.as_str();
            if let Some(sig) = self.module.funcs.get(name).cloned() {
                if args.len() > sig.params.len() || args.len() + sig.defaults < sig.params.len() {
                    return unsupported(
                        format!(
                            "calling `{name}` with {} argument(s); it takes {}",
                            args.len(),
                            sig.params.len()
                        ),
                        c,
                    );
                }
                let mut lowered = Vec::with_capacity(args.len());
                for (a, (_, pty)) in args.iter().zip(&sig.params) {
                    lowered.push(self.expr_as(a, *pty)?);
                }
                return Ok(Val {
                    node: call(name, lowered, sig.ret, span),
                    ty: sig.ret,
                });
            }
            match name {
                "print" => return self.print(args, span),
                "str" => {
                    let v = self.expr(&args[0])?;
                    let node = self.str_of(v);
                    return Ok(Val { node, ty: Ty::Str });
                }
                "repr" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Str => call("__py_repr_str", vec![v.node], Ty::Str, span),
                        Ty::Object | Ty::Unknown => {
                            call("__py_repr_any", vec![v.node], Ty::Str, span)
                        }
                        _ => self.str_of(v),
                    };
                    return Ok(Val { node, ty: Ty::Str });
                }
                "int" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Int => v.node,
                        Ty::Bool | Ty::Float => cast(v.node, Ty::Int, span),
                        Ty::Str => call("__py_int_of_str", vec![v.node], Ty::Int, span),
                        _ => call("__py_int_any", vec![v.node], Ty::Int, span),
                    };
                    return Ok(Val { node, ty: Ty::Int });
                }
                "float" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Float => v.node,
                        Ty::Bool | Ty::Int => cast(v.node, Ty::Float, span),
                        Ty::Str => call("__py_float_of_str", vec![v.node], Ty::Float, span),
                        _ => call("__py_float_any", vec![v.node], Ty::Float, span),
                    };
                    return Ok(Val {
                        node,
                        ty: Ty::Float,
                    });
                }
                "bool" => {
                    let node = if args.is_empty() {
                        node(
                            TypedExpression::Literal(TypedLiteral::Bool(false)),
                            Ty::Bool,
                            span,
                        )
                    } else {
                        let v = self.expr(&args[0])?;
                        self.truthy(v)
                    };
                    return Ok(Val { node, ty: Ty::Bool });
                }
                "len" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Str => call("__py_str_len", vec![v.node], Ty::Int, span),
                        _ => {
                            let o = self.coerce(v, Ty::Object);
                            call("__py_len_any", vec![o], Ty::Int, span)
                        }
                    };
                    return Ok(Val { node, ty: Ty::Int });
                }
                "abs" if ty != Ty::Object => {
                    let v = self.expr(&args[0])?;
                    let v_ty = if v.ty == Ty::Bool { Ty::Int } else { v.ty };
                    let x = self.coerce(v, v_ty);
                    let name = self.temp();
                    let zero = if v_ty == Ty::Float {
                        node(
                            TypedExpression::Literal(TypedLiteral::Float(0.0)),
                            Ty::Float,
                            span,
                        )
                    } else {
                        int_lit(0, span)
                    };
                    let pick = node(
                        TypedExpression::If(TypedIfExpr {
                            condition: Box::new(binary(
                                BinaryOp::Lt,
                                var(name, v_ty, span),
                                zero,
                                Ty::Bool,
                                span,
                            )),
                            then_branch: Box::new(node(
                                TypedExpression::Unary(TypedUnary {
                                    op: UnaryOp::Minus,
                                    operand: Box::new(var(name, v_ty, span)),
                                }),
                                v_ty,
                                span,
                            )),
                            else_branch: Box::new(var(name, v_ty, span)),
                        }),
                        v_ty,
                        span,
                    );
                    let block = node(
                        TypedExpression::Block(TypedBlock {
                            statements: vec![
                                TypedNode::new(
                                    TypedStatement::Let(TypedLet {
                                        name,
                                        ty: ir(v_ty),
                                        mutability: Mutability::Immutable,
                                        initializer: Some(Box::new(x)),
                                        span,
                                    }),
                                    Type::Unknown,
                                    span,
                                ),
                                TypedNode::new(
                                    TypedStatement::Expression(Box::new(pick)),
                                    Type::Unknown,
                                    span,
                                ),
                            ],
                            span,
                        }),
                        v_ty,
                        span,
                    );
                    return Ok(Val {
                        node: block,
                        ty: v_ty,
                    });
                }
                _ => {}
            }
        }
        unsupported(format!("calling `{}`", types::expr_kind(&c.func)), c)
    }

    /// `print(a, b, ...)`: each argument as `str()`, a space between,
    /// one line written.
    fn print(&mut self, args: &[py::Expr], span: Span) -> Result<Val> {
        let mut line: Option<Node> = None;
        for a in args {
            let v = self.expr(a)?;
            let s = self.str_of(v);
            line = Some(match line {
                None => s,
                Some(prev) => {
                    let with_space = binary(BinaryOp::Add, prev, str_lit(" ", span), Ty::Str, span);
                    binary(BinaryOp::Add, with_space, s, Ty::Str, span)
                }
            });
        }
        let line = line.unwrap_or_else(|| str_lit("", span));
        Ok(Val {
            node: call("__py_print_line", vec![line], Ty::None, span),
            ty: Ty::None,
        })
    }
}
