//! Python statements and expressions to typed nodes.
//!
//! Every node this emits carries the IR type the inference in
//! [`crate::types`] assigned, and every place a value crosses between
//! two types gets an explicit conversion: a cast between numbers, a box
//! into `Any`, a checked unbox out of it. Nothing is left for the
//! compiler to guess. Python's own semantics that the IR does not have
//! (how a bool prints, what `"a" * 3` is) are calls into the shared
//! built-in library.

use crate::types::{self, Elem, Locals, Module, Sig, Ty, Typer};
use crate::{intern, prim, span_of, Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedCast, TypedExpression, TypedFor, TypedFunction,
    TypedIf, TypedIfExpr, TypedLet, TypedLiteral, TypedMethodCall, TypedParameter, TypedPattern,
    TypedRange, TypedStatement, TypedUnary, TypedWhile,
};
use zyntax_typed_ast::{
    BinaryOp, InternedString, Mutability, ParamOwnership, ParameterKind, PrimitiveType, Type,
    TypedNode, UnaryOp, Visibility,
};

pub(crate) type Node = TypedNode<TypedExpression>;
type Stmt = TypedNode<TypedStatement>;

thread_local! {
    /// The built-in library's `List<T>`, for spelling list types. Set
    /// once per program before any lowering.
    static LIST_TYPE: std::cell::Cell<Option<zyntax_typed_ast::TypeId>> =
        const { std::cell::Cell::new(None) };
}

pub(crate) fn set_list_type(id: zyntax_typed_ast::TypeId) {
    LIST_TYPE.with(|c| c.set(Some(id)));
}

/// `List<elem>` as the library declares it.
fn list_type(elem: Type) -> Type {
    let id = LIST_TYPE
        .with(|c| c.get())
        .expect("the library's List<T> is known before lowering");
    zyntax_builtins::list_of(id, elem)
}

/// The IR type a static type is carried as.
pub(crate) fn ir(ty: Ty) -> Type {
    match ty {
        Ty::Int => prim(PrimitiveType::I64),
        Ty::Float => prim(PrimitiveType::F64),
        Ty::Bool => prim(PrimitiveType::Bool),
        Ty::Str => prim(PrimitiveType::String),
        Ty::None => prim(PrimitiveType::Unit),
        Ty::List(e) => list_type(ir(e.ty())),
        Ty::Tuple => list_type(Type::Any),
        Ty::Object | Ty::Unknown => Type::Any,
    }
}

fn method_call(receiver: Node, method: &str, args: Vec<Node>, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::MethodCall(TypedMethodCall {
            receiver: Box::new(receiver),
            method: intern(method),
            type_args: Vec::new(),
            positional_args: args,
            named_args: Vec::new(),
        }),
        ty,
        span,
    )
}

fn bind_names(vars: &mut std::collections::HashMap<String, Ty>, target: &py::Expr, ty: Ty) {
    types::bind_target(vars, target, ty)
}

/// `zb_list_<op>_<kind>`.
fn list_fn(op: &str, elem: Elem) -> String {
    format!("zb_list_{op}_{}", elem.suffix())
}

/// A lowered expression and the static type it has.
pub(crate) struct Val {
    pub(crate) node: Node,
    pub(crate) ty: Ty,
}

pub(crate) fn node(x: TypedExpression, ty: Ty, span: Span) -> Node {
    TypedNode::new(x, ir(ty), span)
}

pub(crate) fn var(name: InternedString, ty: Ty, span: Span) -> Node {
    node(TypedExpression::Variable(name), ty, span)
}

pub(crate) fn int_lit(v: i64, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        Ty::Int,
        span,
    )
}

pub(crate) fn str_lit(s: &str, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::String(intern(s))),
        Ty::Str,
        span,
    )
}

pub(crate) fn binary(op: BinaryOp, left: Node, right: Node, ty: Ty, span: Span) -> Node {
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

pub(crate) fn call(name: &str, args: Vec<Node>, ty: Ty, span: Span) -> Node {
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
    /// Statements an expression needs run before the statement it is
    /// part of: a comprehension's loop, which the IR cannot hold inside
    /// an expression. Drained in front of each statement.
    hoisted: Vec<Stmt>,
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
            hoisted: Vec::new(),
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
            // A list is boxed by reference under a tag of its kind, and
            // read back by checking that tag.
            (Ty::List(e), Ty::Object) => call(&list_fn("box", e), vec![v.node], Ty::Object, span),
            (Ty::Tuple, Ty::Object) => call("zb_box_tuple", vec![v.node], Ty::Object, span),
            (Ty::Object, Ty::List(e)) => call(&list_fn("unbox", e), vec![v.node], target, span),
            (Ty::Object, Ty::Tuple) => call("zb_unbox_tuple", vec![v.node], Ty::Tuple, span),
            // Lists of one kind into lists of dynamic values.
            (Ty::List(e), Ty::List(Elem::Object)) => {
                call(&list_fn("to_any", e), vec![v.node], target, span)
            }
            (Ty::Tuple, Ty::List(Elem::Object)) => Node {
                ty: ir(target),
                ..v.node
            },
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
            Ty::Str => call("zb_str_truthy", vec![v.node], Ty::Bool, span),
            Ty::List(_) | Ty::Tuple => binary(
                BinaryOp::Ne,
                method_call(v.node, "len", vec![], Ty::Int, span),
                int_lit(0, span),
                Ty::Bool,
                span,
            ),
            Ty::None => node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                Ty::Bool,
                span,
            ),
            Ty::Object | Ty::Unknown => call("zb_any_truthy", vec![v.node], Ty::Bool, span),
        }
    }

    /// `str(v)`.
    pub(crate) fn str_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Int => call("zb_str_of_int", vec![v.node], Ty::Str, span),
            Ty::Float => call("zb_float_repr", vec![v.node], Ty::Str, span),
            Ty::Bool => call("zb_bool_repr", vec![v.node], Ty::Str, span),
            Ty::Str => v.node,
            Ty::None => call("zb_none_repr", vec![], Ty::Str, span),
            Ty::List(e) => call(&list_fn("repr", e), vec![v.node], Ty::Str, span),
            Ty::Tuple => call("zb_tuple_repr", vec![v.node], Ty::Str, span),
            Ty::Object | Ty::Unknown => call("zb_any_str", vec![v.node], Ty::Str, span),
        }
    }

    /// `repr(v)`.
    pub(crate) fn repr_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Str => call("zb_str_repr", vec![v.node], Ty::Str, span),
            Ty::Object | Ty::Unknown => call("zb_any_repr", vec![v.node], Ty::Str, span),
            _ => self.str_of(v),
        }
    }

    /// A list literal of `elem` kind from already lowered elements.
    fn list_of(&mut self, items: Vec<Val>, elem: Elem, span: Span) -> Node {
        let items = items
            .into_iter()
            .map(|v| self.coerce(v, elem.ty()))
            .collect();
        node(TypedExpression::Array(items), Ty::List(elem), span)
    }

    /// Bind a value to a hidden local and hand back the name, so it is
    /// evaluated once.
    fn hold(&mut self, v: Val, out: &mut Vec<Stmt>, span: Span) -> Val {
        let name = self.temp();
        let ty = v.ty;
        out.push(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name,
                ty: ir(ty),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(v.node)),
                span,
            }),
            Type::Unknown,
            span,
        ));
        Val {
            node: var(name, ty, span),
            ty,
        }
    }

    /// A block expression: statements, then the value.
    fn block_value(statements: Vec<Stmt>, value: Node, ty: Ty, span: Span) -> Node {
        let mut statements = statements;
        statements.push(TypedNode::new(
            TypedStatement::Expression(Box::new(value)),
            Type::Unknown,
            span,
        ));
        node(
            TypedExpression::Block(TypedBlock { statements, span }),
            ty,
            span,
        )
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
        let mut own = Vec::new();
        self.stmt_into(s, &mut own)?;
        // Whatever the statement's expressions hoisted runs first.
        out.append(&mut self.hoisted);
        out.append(&mut own);
        Ok(())
    }

    fn stmt_into(&mut self, s: &py::Stmt, out: &mut Vec<Stmt>) -> Result<()> {
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
            py::Stmt::Delete(d) => {
                for target in &d.targets {
                    let py::Expr::Subscript(sub) = target else {
                        return unsupported("del of anything but an item", target);
                    };
                    let seq = self.expr(&sub.value)?;
                    let stmt = match seq.ty {
                        Ty::List(e) => {
                            let i = self.expr_as(&sub.slice, Ty::Int)?;
                            call(&list_fn("pop", e), vec![seq.node, i], e.ty(), span)
                        }
                        _ => {
                            let key = self.expr_as(&sub.slice, Ty::Object)?;
                            let seq = self.coerce(seq, Ty::Object);
                            call("zb_any_delitem", vec![seq, key], Ty::None, span)
                        }
                    };
                    push(out, TypedStatement::Expression(Box::new(stmt)));
                }
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
        let n = match target {
            py::Expr::Name(n) => n,
            // `xs[i] = v`
            py::Expr::Subscript(sub) => {
                let seq = self.expr(&sub.value)?;
                let stmt = match seq.ty {
                    Ty::List(e) => {
                        let i = self.expr_as(&sub.slice, Ty::Int)?;
                        let v = self.coerce(value, e.ty());
                        call(&list_fn("set", e), vec![seq.node, i, v], Ty::None, span)
                    }
                    Ty::Object => {
                        let i = self.expr_as(&sub.slice, Ty::Object)?;
                        let v = self.coerce(value, Ty::Object);
                        call("zb_any_setitem", vec![seq.node, i, v], Ty::None, span)
                    }
                    _ => {
                        return unsupported(
                            format!("item assignment on {}", types::expr_kind(&sub.value)),
                            target,
                        )
                    }
                };
                out.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(stmt)),
                    Type::Unknown,
                    span,
                ));
                return Ok(());
            }
            // `a, b = value`: the value once, then each name an element.
            py::Expr::Tuple(t) => {
                let elem_ty = value.ty.element().unwrap_or(Ty::Object);
                let seq = self.hold(value, out, span);
                let n = t.elts.len() as i64;
                let check = match seq.ty {
                    Ty::List(e) => Some(call(
                        &list_fn("expect_len", e),
                        vec![seq.node.clone(), int_lit(n, span)],
                        Ty::None,
                        span,
                    )),
                    Ty::Tuple => Some(call(
                        "zb_list_expect_len_any",
                        vec![seq.node.clone(), int_lit(n, span)],
                        Ty::None,
                        span,
                    )),
                    _ => None,
                };
                if let Some(check) = check {
                    out.push(TypedNode::new(
                        TypedStatement::Expression(Box::new(check)),
                        Type::Unknown,
                        span,
                    ));
                }
                for (i, elt) in t.elts.iter().enumerate() {
                    let item = self.index_value(
                        Val {
                            node: seq.node.clone(),
                            ty: seq.ty,
                        },
                        int_lit(i as i64, span),
                        elem_ty,
                        span,
                    );
                    self.bind(elt, item, span, out)?;
                }
                return Ok(());
            }
            other => {
                return unsupported(format!("assignment to {}", types::expr_kind(other)), other)
            }
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

    /// `seq[i]` for a sequence value and an int index already lowered.
    fn index_value(&mut self, seq: Val, index: Node, elem_ty: Ty, span: Span) -> Val {
        let node = match seq.ty {
            Ty::List(e) => call(&list_fn("get", e), vec![seq.node, index], e.ty(), span),
            Ty::Tuple => call("zb_list_get_any", vec![seq.node, index], Ty::Object, span),
            Ty::Str => call("zb_str_get", vec![seq.node, index], Ty::Str, span),
            _ => {
                let i = self.coerce(
                    Val {
                        node: index,
                        ty: Ty::Int,
                    },
                    Ty::Object,
                );
                call("zb_any_getitem", vec![seq.node, i], Ty::Object, span)
            }
        };
        Val { node, ty: elem_ty }
    }

    /// `for x in <sequence>`: an index loop over the sequence held once,
    /// with the element bound at the top of each iteration.
    fn for_sequence(
        &mut self,
        f: &py::StmtFor,
        extra: Vec<Stmt>,
        span: Span,
    ) -> Result<TypedStatement> {
        let seq = self.expr(&f.iter)?;
        let elem_ty = seq.ty.element().unwrap_or(Ty::Object);
        let mut prologue = Vec::new();
        let seq = match seq.ty {
            // A dynamic iterable is snapshotted into a list of objects.
            Ty::Object => {
                let items = call("zb_any_iter", vec![seq.node], Ty::List(Elem::Object), span);
                self.hold(
                    Val {
                        node: items,
                        ty: Ty::List(Elem::Object),
                    },
                    &mut prologue,
                    span,
                )
            }
            _ => self.hold(seq, &mut prologue, span),
        };
        let counter = self.temp();
        let len = match seq.ty {
            Ty::Str => call("zb_str_chars_len", vec![seq.node.clone()], Ty::Int, span),
            _ => method_call(seq.node.clone(), "len", vec![], Ty::Int, span),
        };
        let item = self.index_value(
            Val {
                node: seq.node.clone(),
                ty: seq.ty,
            },
            var(counter, Ty::Int, span),
            elem_ty,
            span,
        );
        let mut body = Vec::new();
        self.bind(&f.target, item, span, &mut body)?;
        for s in &f.body {
            self.stmt(s, &mut body)?;
        }
        body.extend(extra);
        let loop_stmt = TypedStatement::For(TypedFor {
            pattern: Box::new(TypedNode::new(
                TypedPattern::Identifier {
                    name: counter,
                    mutability: Mutability::Mutable,
                },
                prim(PrimitiveType::I64),
                span,
            )),
            iterator: Box::new(TypedNode::new(
                TypedExpression::Range(TypedRange {
                    start: Some(Box::new(int_lit(0, span))),
                    end: Some(Box::new(len)),
                    inclusive: false,
                }),
                Type::Unknown,
                span,
            )),
            body: TypedBlock {
                statements: body,
                span,
            },
        });
        prologue.push(TypedNode::new(loop_stmt, Type::Unknown, span));
        Ok(TypedStatement::Block(TypedBlock {
            statements: prologue,
            span,
        }))
    }

    fn for_loop(&mut self, f: &py::StmtFor, span: Span) -> Result<TypedStatement> {
        self.for_with_body(f, Vec::new(), span)
    }

    /// `for x in range(...)` as a counted loop; anything else iterates
    /// by index over a sequence. `extra` statements follow the body.
    fn for_with_body(
        &mut self,
        f: &py::StmtFor,
        extra: Vec<Stmt>,
        span: Span,
    ) -> Result<TypedStatement> {
        if f.is_async || !f.orelse.is_empty() {
            return unsupported("async for / for-else", f);
        }
        let range = match &*f.iter {
            py::Expr::Call(c)
                if types::is_name(&c.func, "range") && c.arguments.keywords.is_empty() =>
            {
                Some(c)
            }
            _ => None,
        };
        let (Some(c), py::Expr::Name(target)) = (range, &*f.target) else {
            return self.for_sequence(f, extra, span);
        };
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
        let mut body = self.block(&f.body, span)?;
        body.statements.extend(extra);
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
            py::Expr::Name(n)
                if !self.locals.vars.contains_key(n.id.as_str())
                    && matches!(
                        n.id.as_str(),
                        "int" | "float" | "str" | "bool" | "list" | "tuple" | "dict" | "set"
                    ) =>
            {
                Val {
                    node: str_lit(n.id.as_str(), span),
                    ty: Ty::Str,
                }
            }
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
            py::Expr::List(l) => {
                let Ty::List(elem) = ty else { unreachable!() };
                let mut items = Vec::with_capacity(l.elts.len());
                for e in &l.elts {
                    items.push(self.expr(e)?);
                }
                Val {
                    node: self.list_of(items, elem, span),
                    ty,
                }
            }
            py::Expr::Tuple(t) => {
                let mut items = Vec::with_capacity(t.elts.len());
                for e in &t.elts {
                    items.push(self.expr(e)?);
                }
                let node = self.list_of(items, Elem::Object, span);
                Val {
                    node: Node {
                        ty: ir(Ty::Tuple),
                        ..node
                    },
                    ty: Ty::Tuple,
                }
            }
            py::Expr::Subscript(sub) => self.subscript(sub, ty, span)?,
            py::Expr::ListComp(c) => self.list_comp(c, ty, span)?,
            py::Expr::FString(f) => self.fstring(f, span)?,
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
                            UnaryOp::Minus => "zb_any_neg",
                            UnaryOp::BitNot => "zb_any_invert",
                            _ => "zb_any_pos",
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
                        node: call("zb_str_repeat", vec![left.node, n], Ty::Str, span),
                        ty: Ty::Str,
                    });
                }
                (py::Operator::Mult, Ty::Int | Ty::Bool, Ty::Str) => {
                    let n = self.coerce(left, Ty::Int);
                    return Ok(Val {
                        node: call("zb_str_repeat", vec![right.node, n], Ty::Str, span),
                        ty: Ty::Str,
                    });
                }
                _ => {}
            }
        }
        // Sequences concatenate and repeat.
        match (op, left.ty, right.ty) {
            (py::Operator::Add, Ty::List(e), Ty::List(_)) if ty == left.ty => {
                return Ok(Val {
                    node: call(&list_fn("concat", e), vec![left.node, right.node], ty, span),
                    ty,
                });
            }
            (py::Operator::Add, Ty::Tuple, Ty::Tuple) => {
                return Ok(Val {
                    node: call(
                        "zb_list_concat_any",
                        vec![left.node, right.node],
                        Ty::Tuple,
                        span,
                    ),
                    ty: Ty::Tuple,
                });
            }
            (py::Operator::Mult, Ty::List(_) | Ty::Tuple, Ty::Int | Ty::Bool)
            | (py::Operator::Mult, Ty::Int | Ty::Bool, Ty::List(_) | Ty::Tuple) => {
                let (seq, times) = if matches!(left.ty, Ty::List(_) | Ty::Tuple) {
                    (left, right)
                } else {
                    (right, left)
                };
                let elem = match seq.ty {
                    Ty::List(e) => e,
                    _ => Elem::Object,
                };
                let n = self.coerce(times, Ty::Int);
                let seq_ty = seq.ty;
                return Ok(Val {
                    node: call(&list_fn("repeat", elem), vec![seq.node, n], seq_ty, span),
                    ty: seq_ty,
                });
            }
            _ => {}
        }
        if ty == Ty::Object {
            // At least one side is dynamic: the runtime picks the
            // operation from the tags.
            let l = self.coerce(left, Ty::Object);
            let r = self.coerce(right, Ty::Object);
            let code = int_lit(types::arith_code(op), span);
            return Ok(Val {
                node: call("zb_any_arith", vec![code, l, r], Ty::Object, span),
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
                        "zb_str_contains",
                        vec![right.node, left.node],
                        Ty::Bool,
                        span,
                    )
                } else if let Ty::List(e) = right.ty {
                    let item = self.coerce(left, e.ty());
                    call(
                        &list_fn("contains", e),
                        vec![right.node, item],
                        Ty::Bool,
                        span,
                    )
                } else if right.ty == Ty::Tuple {
                    let item = self.coerce(left, Ty::Object);
                    call(
                        "zb_list_contains_any",
                        vec![right.node, item],
                        Ty::Bool,
                        span,
                    )
                } else {
                    let item = self.coerce(left, Ty::Object);
                    let container = self.coerce(right, Ty::Object);
                    call("zb_any_contains", vec![container, item], Ty::Bool, span)
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
                        call("zb_any_is", vec![l, r], Ty::Bool, span)
                    }
                };
                return Ok(if op == py::CmpOp::IsNot { negate(n) } else { n });
            }
            _ => {}
        }
        // Two sequences compare element by element.
        let seq_kind = |t: Ty| match t {
            Ty::List(e) => Some(e),
            Ty::Tuple => Some(Elem::Object),
            _ => None,
        };
        if let (Some(e), Some(f)) = (seq_kind(left.ty), seq_kind(right.ty)) {
            let (l, r, e) = if e == f {
                (left.node, right.node, e)
            } else {
                (
                    self.coerce(left, Ty::List(Elem::Object)),
                    self.coerce(right, Ty::List(Elem::Object)),
                    Elem::Object,
                )
            };
            let eq = |l: Node, r: Node| call(&list_fn("eq", e), vec![l, r], Ty::Bool, span);
            let lt = |l: Node, r: Node| call(&list_fn("lt", e), vec![l, r], Ty::Bool, span);
            return Ok(match op {
                py::CmpOp::Eq => eq(l, r),
                py::CmpOp::NotEq => negate(eq(l, r)),
                py::CmpOp::Lt => lt(l, r),
                py::CmpOp::Gt => lt(r, l),
                py::CmpOp::LtE => negate(lt(r, l)),
                py::CmpOp::GtE => negate(lt(l, r)),
                _ => unreachable!(),
            });
        }
        if left.ty == Ty::Str && right.ty == Ty::Str {
            let n = match op {
                py::CmpOp::Eq => call("zb_str_eq", vec![left.node, right.node], Ty::Bool, span),
                py::CmpOp::NotEq => negate(call(
                    "zb_str_eq",
                    vec![left.node, right.node],
                    Ty::Bool,
                    span,
                )),
                py::CmpOp::Lt => call("zb_str_lt", vec![left.node, right.node], Ty::Bool, span),
                py::CmpOp::Gt => call("zb_str_lt", vec![right.node, left.node], Ty::Bool, span),
                py::CmpOp::LtE => negate(call(
                    "zb_str_lt",
                    vec![right.node, left.node],
                    Ty::Bool,
                    span,
                )),
                py::CmpOp::GtE => negate(call(
                    "zb_str_lt",
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
            py::CmpOp::Eq => call("zb_any_eq", vec![l, r], Ty::Bool, span),
            py::CmpOp::NotEq => negate(call("zb_any_eq", vec![l, r], Ty::Bool, span)),
            py::CmpOp::Lt => call("zb_any_lt", vec![l, r], Ty::Bool, span),
            py::CmpOp::Gt => call("zb_any_lt", vec![r, l], Ty::Bool, span),
            py::CmpOp::LtE => negate(call("zb_any_lt", vec![r, l], Ty::Bool, span)),
            py::CmpOp::GtE => negate(call("zb_any_lt", vec![l, r], Ty::Bool, span)),
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

    /// `seq[i]` and `seq[a:b:c]`.
    fn subscript(&mut self, sub: &py::ExprSubscript, ty: Ty, span: Span) -> Result<Val> {
        let seq = self.expr(&sub.value)?;
        if let py::Expr::Slice(sl) = &*sub.slice {
            let mut mask = 0;
            let mut bound =
                |this: &mut Self, e: &Option<Box<py::Expr>>, bit: i64| -> Result<Node> {
                    match e {
                        Some(e) => {
                            mask |= bit;
                            this.expr_as(e, Ty::Int)
                        }
                        None => Ok(int_lit(0, span)),
                    }
                };
            let start = bound(self, &sl.lower, 1)?;
            let stop = bound(self, &sl.upper, 2)?;
            let step = bound(self, &sl.step, 4)?;
            let mask = int_lit(mask, span);
            let node = match seq.ty {
                Ty::List(e) => call(
                    &list_fn("slice", e),
                    vec![seq.node, start, stop, step, mask],
                    ty,
                    span,
                ),
                Ty::Tuple => {
                    let sliced = call(
                        "zb_list_slice_any",
                        vec![seq.node, start, stop, step, mask],
                        Ty::Tuple,
                        span,
                    );
                    Node {
                        ty: ir(Ty::Tuple),
                        ..sliced
                    }
                }
                Ty::Str => call(
                    "zb_str_slice",
                    vec![seq.node, start, stop, step, mask],
                    Ty::Str,
                    span,
                ),
                _ => {
                    let o = self.coerce(seq, Ty::Object);
                    call(
                        "zb_any_getslice",
                        vec![o, start, stop, step, mask],
                        Ty::Object,
                        span,
                    )
                }
            };
            return Ok(Val { node, ty });
        }
        match seq.ty {
            Ty::List(_) | Ty::Tuple | Ty::Str => {
                let index = self.expr_as(&sub.slice, Ty::Int)?;
                Ok(self.index_value(seq, index, ty, span))
            }
            _ => {
                let key = self.expr_as(&sub.slice, Ty::Object)?;
                let o = self.coerce(seq, Ty::Object);
                Ok(Val {
                    node: call("zb_any_getitem", vec![o, key], Ty::Object, span),
                    ty: Ty::Object,
                })
            }
        }
    }

    /// `[e for x in it if c]`: a fresh list, a loop appending to it, the
    /// list as the value.
    fn list_comp(&mut self, c: &py::ExprListComp, ty: Ty, span: Span) -> Result<Val> {
        let Ty::List(elem) = ty else { unreachable!() };
        let out = self.temp();
        // Loop variables are the comprehension's own; they shadow the
        // function's for the body and are forgotten after.
        let saved_vars = self.locals.vars.clone();
        let mut statements = vec![TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: out,
                ty: ir(ty),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(self.list_of(Vec::new(), elem, span))),
                span,
            }),
            Type::Unknown,
            span,
        )];
        // Build innermost first: the append, wrapped in each `if`, wrapped
        // in each `for`, from the last generator outwards.
        for g in &c.generators {
            let item_ty = self.ty_of(&g.iter).element().unwrap_or(match &g.iter {
                py::Expr::Call(call) if types::is_name(&call.func, "range") => Ty::Int,
                _ => Ty::Object,
            });
            bind_names(&mut self.locals.vars, &g.target, item_ty);
        }
        let value = self.expr(&c.elt)?;
        let value = self.coerce(value, elem.ty());
        let mut inner: Vec<Stmt> = vec![TypedNode::new(
            TypedStatement::Expression(Box::new(method_call(
                var(out, ty, span),
                "push",
                vec![value],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        )];
        for g in c.generators.iter().rev() {
            for cond in g.ifs.iter().rev() {
                let test = self.expr(cond)?;
                let condition = self.truthy(test);
                inner = vec![TypedNode::new(
                    TypedStatement::If(TypedIf {
                        condition: Box::new(condition),
                        then_block: TypedBlock {
                            statements: inner,
                            span,
                        },
                        else_block: None,
                        span,
                    }),
                    Type::Unknown,
                    span,
                )];
            }
            let for_stmt = py::StmtFor {
                node_index: Default::default(),
                range: g.range,
                is_async: g.is_async,
                target: Box::new(g.target.clone()),
                iter: Box::new(g.iter.clone()),
                body: Default::default(),
                orelse: Default::default(),
            };
            inner = vec![TypedNode::new(
                self.for_with_body(&for_stmt, inner, span)?,
                Type::Unknown,
                span,
            )];
        }
        self.locals.vars = saved_vars;
        statements.extend(inner);
        self.hoisted.extend(statements);
        Ok(Val {
            node: var(out, ty, span),
            ty,
        })
    }

    /// A method call on a value whose type is known.
    fn method(
        &mut self,
        receiver: Val,
        name: &str,
        args: &[py::Expr],
        ty: Ty,
        span: Span,
    ) -> Result<Val> {
        let expect = |n: usize, this: &Self| -> Result<()> {
            let _ = this;
            if args.len() == n {
                Ok(())
            } else {
                Err(Error::Unsupported {
                    what: format!("{name}() with {} argument(s)", args.len()),
                    at: span.start,
                })
            }
        };
        match receiver.ty {
            Ty::List(e) => {
                let list = receiver.node;
                let node = match name {
                    "append" => {
                        expect(1, self)?;
                        let v = self.expr_as(&args[0], e.ty())?;
                        method_call(list, "push", vec![v], Ty::None, span)
                    }
                    "pop" => {
                        let i = if args.is_empty() {
                            int_lit(-1, span)
                        } else {
                            self.expr_as(&args[0], Ty::Int)?
                        };
                        call(&list_fn("pop", e), vec![list, i], e.ty(), span)
                    }
                    "insert" => {
                        expect(2, self)?;
                        let i = self.expr_as(&args[0], Ty::Int)?;
                        let v = self.expr_as(&args[1], e.ty())?;
                        call(&list_fn("insert", e), vec![list, i, v], Ty::None, span)
                    }
                    "remove" | "index" | "count" => {
                        expect(1, self)?;
                        let v = self.expr_as(&args[0], e.ty())?;
                        call(&list_fn(name, e), vec![list, v], ty, span)
                    }
                    "extend" => {
                        expect(1, self)?;
                        let other = self.expr_as(&args[0], Ty::List(e))?;
                        call(&list_fn("extend", e), vec![list, other], Ty::None, span)
                    }
                    "sort" | "reverse" | "copy" => {
                        expect(0, self)?;
                        call(&list_fn(name, e), vec![list], ty, span)
                    }
                    "clear" => {
                        expect(0, self)?;
                        method_call(list, "clear", vec![], Ty::None, span)
                    }
                    _ => {
                        return unsupported(
                            format!("list.{name}"),
                            &args.first().map(|a| a.range()).unwrap_or(
                                ruff_text_size::TextRange::empty(ruff_text_size::TextSize::from(
                                    span.start as u32,
                                )),
                            ),
                        )
                    }
                };
                Ok(Val { node, ty })
            }
            Ty::Str => {
                let s = receiver.node;
                let mut lowered = Vec::with_capacity(args.len());
                for a in args {
                    lowered.push(self.expr(a)?);
                }
                let node = match (name, lowered.len()) {
                    (
                        "upper" | "lower" | "strip" | "lstrip" | "rstrip" | "capitalize" | "title",
                        0,
                    ) => call(&format!("zb_str_{name}"), vec![s], Ty::Str, span),
                    ("startswith" | "endswith", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call(&format!("zb_str_{name}"), vec![s, a], Ty::Bool, span)
                    }
                    ("find", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call("zb_str_index_of", vec![s, a], Ty::Int, span)
                    }
                    ("count", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call("zb_str_count", vec![s, a], Ty::Int, span)
                    }
                    ("replace", 2) => {
                        let b = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call("zb_str_replace", vec![s, a, b], Ty::Str, span)
                    }
                    ("split", 0) => call("zb_str_split_ws", vec![s], Ty::List(Elem::Str), span),
                    ("split", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call("zb_str_split", vec![s, a], Ty::List(Elem::Str), span)
                    }
                    ("join", 1) => {
                        let items = self.coerce(lowered.pop().unwrap(), Ty::List(Elem::Str));
                        call("zb_str_join", vec![s, items], Ty::Str, span)
                    }
                    _ => {
                        return Err(Error::Unsupported {
                            what: format!("str.{name} with {} argument(s)", args.len()),
                            at: span.start,
                        })
                    }
                };
                Ok(Val { node, ty })
            }
            _ => Err(Error::Unsupported {
                what: format!("method `{name}` on a dynamic value"),
                at: span.start,
            }),
        }
    }

    /// A call: `print`, a conversion builtin, or a function the module
    /// defines.
    fn call(&mut self, c: &py::ExprCall, ty: Ty, span: Span) -> Result<Val> {
        if !c.arguments.keywords.is_empty() {
            return unsupported("keyword arguments", c);
        }
        let args = &c.arguments.args;
        if let py::Expr::Attribute(a) = &*c.func {
            let receiver = self.expr(&a.value)?;
            return self.method(receiver, a.attr.as_str(), args, ty, span);
        }
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
                    let node = self.repr_of(v);
                    return Ok(Val { node, ty: Ty::Str });
                }
                "int" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Int => v.node,
                        Ty::Bool | Ty::Float => cast(v.node, Ty::Int, span),
                        Ty::Str => call("zb_str_parse_int", vec![v.node], Ty::Int, span),
                        _ => call("zb_any_int", vec![v.node], Ty::Int, span),
                    };
                    return Ok(Val { node, ty: Ty::Int });
                }
                "float" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Float => v.node,
                        Ty::Bool | Ty::Int => cast(v.node, Ty::Float, span),
                        Ty::Str => call("zb_str_parse_float", vec![v.node], Ty::Float, span),
                        _ => call("zb_any_float", vec![v.node], Ty::Float, span),
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
                        Ty::Str => call("zb_str_chars_len", vec![v.node], Ty::Int, span),
                        Ty::List(_) | Ty::Tuple => {
                            method_call(v.node, "len", vec![], Ty::Int, span)
                        }
                        _ => {
                            let o = self.coerce(v, Ty::Object);
                            call("zb_any_len", vec![o], Ty::Int, span)
                        }
                    };
                    return Ok(Val { node, ty: Ty::Int });
                }
                "sum" if args.len() == 1 => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::List(e @ (Elem::Int | Elem::Float | Elem::Object)) => {
                            call(&list_fn("sum", e), vec![v.node], e.ty(), span)
                        }
                        _ => {
                            let xs = self.coerce(v, Ty::List(Elem::Object));
                            call("zb_list_sum_any", vec![xs], Ty::Object, span)
                        }
                    };
                    return Ok(Val { node, ty });
                }
                "min" | "max" => {
                    // Several arguments are the one-argument form over a
                    // list of them.
                    let list = if args.len() == 1 {
                        self.expr(&args[0])?
                    } else {
                        let mut items = Vec::with_capacity(args.len());
                        for a in args.iter() {
                            items.push(self.expr(a)?);
                        }
                        let elem = Elem::of(ty);
                        Val {
                            node: self.list_of(items, elem, span),
                            ty: Ty::List(elem),
                        }
                    };
                    let list = match list.ty {
                        Ty::List(_) => list,
                        _ => {
                            let e = Elem::Object;
                            Val {
                                node: self.coerce(list, Ty::List(e)),
                                ty: Ty::List(e),
                            }
                        }
                    };
                    let Ty::List(e) = list.ty else { unreachable!() };
                    let node = call(&list_fn(name, e), vec![list.node], e.ty(), span);
                    let v = Val { node, ty: e.ty() };
                    return Ok(Val {
                        node: self.coerce(v, ty),
                        ty,
                    });
                }
                "sorted" | "reversed" | "list" | "tuple" => {
                    let target_elem = match ty {
                        Ty::List(e) => e,
                        _ => Elem::Object,
                    };
                    let source = match args.first() {
                        None => {
                            let empty = self.list_of(Vec::new(), target_elem, span);
                            Val {
                                node: empty,
                                ty: Ty::List(target_elem),
                            }
                        }
                        Some(py::Expr::Call(rc)) if types::is_name(&rc.func, "range") => {
                            let (start, stop, step) = match rc.arguments.args.as_ref() {
                                [end] => (
                                    int_lit(0, span),
                                    self.expr_as(end, Ty::Int)?,
                                    int_lit(1, span),
                                ),
                                [a, b] => (
                                    self.expr_as(a, Ty::Int)?,
                                    self.expr_as(b, Ty::Int)?,
                                    int_lit(1, span),
                                ),
                                [a, b, c2] => (
                                    self.expr_as(a, Ty::Int)?,
                                    self.expr_as(b, Ty::Int)?,
                                    self.expr_as(c2, Ty::Int)?,
                                ),
                                _ => {
                                    return unsupported(
                                        "range() with more than three arguments",
                                        rc,
                                    )
                                }
                            };
                            Val {
                                node: call(
                                    "zb_list_range",
                                    vec![start, stop, step],
                                    Ty::List(Elem::Int),
                                    span,
                                ),
                                ty: Ty::List(Elem::Int),
                            }
                        }
                        Some(a) => {
                            let v = self.expr(a)?;
                            match v.ty {
                                Ty::List(_) => v,
                                Ty::Tuple => Val {
                                    node: Node {
                                        ty: ir(Ty::List(Elem::Object)),
                                        ..v.node
                                    },
                                    ty: Ty::List(Elem::Object),
                                },
                                Ty::Str => Val {
                                    node: call(
                                        "zb_str_chars",
                                        vec![v.node],
                                        Ty::List(Elem::Str),
                                        span,
                                    ),
                                    ty: Ty::List(Elem::Str),
                                },
                                _ => {
                                    let o = self.coerce(v, Ty::Object);
                                    Val {
                                        node: call(
                                            "zb_any_iter",
                                            vec![o],
                                            Ty::List(Elem::Object),
                                            span,
                                        ),
                                        ty: Ty::List(Elem::Object),
                                    }
                                }
                            }
                        }
                    };
                    let Ty::List(e) = source.ty else {
                        unreachable!()
                    };
                    // Every form returns a fresh list.
                    let mut statements = Vec::new();
                    let copy = call(&list_fn("copy", e), vec![source.node], Ty::List(e), span);
                    let held = self.hold(
                        Val {
                            node: copy,
                            ty: Ty::List(e),
                        },
                        &mut statements,
                        span,
                    );
                    let op = match name {
                        "sorted" => Some("sort"),
                        "reversed" => Some("reverse"),
                        _ => None,
                    };
                    if let Some(op) = op {
                        statements.push(TypedNode::new(
                            TypedStatement::Expression(Box::new(call(
                                &list_fn(op, e),
                                vec![held.node.clone()],
                                Ty::None,
                                span,
                            ))),
                            Type::Unknown,
                            span,
                        ));
                    }
                    let result_ty = if name == "tuple" {
                        Ty::Tuple
                    } else {
                        Ty::List(e)
                    };
                    let value = Node {
                        ty: ir(result_ty),
                        ..held.node
                    };
                    return Ok(Val {
                        node: Self::block_value(statements, value, result_ty, span),
                        ty: result_ty,
                    });
                }
                "divmod" if args.len() == 2 => {
                    let a = self.expr(&args[0])?;
                    let b = self.expr(&args[1])?;
                    let mut statements = Vec::new();
                    let a = self.hold(a, &mut statements, span);
                    let b = self.hold(b, &mut statements, span);
                    let q = self.arithmetic(
                        py::Operator::FloorDiv,
                        Val {
                            node: a.node.clone(),
                            ty: a.ty,
                        },
                        Val {
                            node: b.node.clone(),
                            ty: b.ty,
                        },
                        &args[1],
                        span,
                    )?;
                    let r = self.arithmetic(py::Operator::Mod, a, b, &args[1], span)?;
                    let pair = self.list_of(vec![q, r], Elem::Object, span);
                    let value = Node {
                        ty: ir(Ty::Tuple),
                        ..pair
                    };
                    return Ok(Val {
                        node: Self::block_value(statements, value, Ty::Tuple, span),
                        ty: Ty::Tuple,
                    });
                }
                "pow" if args.len() == 2 => {
                    let a = self.expr(&args[0])?;
                    let b = self.expr(&args[1])?;
                    return self.arithmetic(py::Operator::Pow, a, b, &args[1], span);
                }
                "round" => {
                    let v = self.expr(&args[0])?;
                    let node = match (v.ty, args.len()) {
                        (Ty::Int, 1) => v.node,
                        (Ty::Float | Ty::Bool, 1) => {
                            let f = self.coerce(v, Ty::Float);
                            call("zb_round_half_even", vec![f], Ty::Int, span)
                        }
                        (Ty::Int | Ty::Float | Ty::Bool, 2) => {
                            let f = self.coerce(v, Ty::Float);
                            let n = self.expr_as(&args[1], Ty::Int)?;
                            let r = call("zb_round_digits", vec![f, n], Ty::Float, span);
                            return Ok(Val {
                                node: self.coerce(
                                    Val {
                                        node: r,
                                        ty: Ty::Float,
                                    },
                                    ty,
                                ),
                                ty,
                            });
                        }
                        _ => {
                            let o = self.coerce(v, Ty::Object);
                            call("zb_any_round", vec![o], Ty::Object, span)
                        }
                    };
                    return Ok(Val {
                        node: self.coerce(
                            Val {
                                node,
                                ty: if ty == Ty::Object {
                                    Ty::Object
                                } else {
                                    Ty::Int
                                },
                            },
                            ty,
                        ),
                        ty,
                    });
                }
                "type" if args.len() == 1 => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Int => str_lit("int", span),
                        Ty::Float => str_lit("float", span),
                        Ty::Bool => str_lit("bool", span),
                        Ty::Str => str_lit("str", span),
                        Ty::None => str_lit("NoneType", span),
                        Ty::List(_) => str_lit("list", span),
                        Ty::Tuple => str_lit("tuple", span),
                        _ => call("zb_any_type", vec![v.node], Ty::Str, span),
                    };
                    return Ok(Val { node, ty: Ty::Str });
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
            node: call("zb_print_line", vec![line], Ty::None, span),
            ty: Ty::None,
        })
    }
}
