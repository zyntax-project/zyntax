//! Python statements and expressions to typed nodes.
//!
//! Every node this emits carries the IR type the inference in
//! [`crate::types`] assigned, and every place a value crosses between
//! two types gets an explicit conversion: a cast between numbers, a box
//! into `Any`, a checked unbox out of it. Nothing is left for the
//! compiler to guess. Python's own semantics that the IR does not have
//! (how a bool prints, what `"a" * 3` is) are calls into the shared
//! built-in library.

use crate::scope::Scope;
use crate::types::{self, Elem, Locals, Module, Sig, Ty, Typer};
use crate::{intern, prim, span_of, Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use std::collections::{BTreeSet, HashMap};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    TypedBinary, TypedBlock, TypedCall, TypedCast, TypedExpression, TypedFieldAccess, TypedFor,
    TypedFunction, TypedIf, TypedIfExpr, TypedLet, TypedLiteral, TypedMatch, TypedMatchArm,
    TypedMethodCall, TypedParameter, TypedPattern, TypedRange, TypedStatement, TypedUnary,
    TypedWhile,
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

thread_local! {
    /// The struct type of each class, by class index.
    static CLASS_TYPES: std::cell::RefCell<Vec<zyntax_typed_ast::TypeId>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub(crate) fn set_class_types(ids: Vec<zyntax_typed_ast::TypeId>) {
    CLASS_TYPES.with(|c| *c.borrow_mut() = ids);
}

/// The struct type of class `k`.
pub(crate) fn class_type(k: usize) -> Type {
    let id = CLASS_TYPES.with(|c| c.borrow()[k]);
    Type::Named {
        id,
        type_args: Vec::new(),
        const_args: Vec::new(),
        variance: Vec::new(),
        nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
    }
}

/// A field holds its own scalar; anything on the heap is stored boxed,
/// so a list field is one shared header and a class's layout never
/// depends on another's.
pub(crate) fn field_storage(ty: Ty) -> Ty {
    match ty {
        Ty::Int | Ty::Float | Ty::Bool | Ty::Str => ty,
        _ => Ty::Object,
    }
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
        Ty::Tuple | Ty::Dict | Ty::Set => list_type(Type::Any),
        Ty::Class(k) => class_type(k as usize),
        Ty::Gen => Type::Fiber(Box::new(Type::Any)),
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

/// What a comprehension builds.
#[derive(Clone, Copy)]
enum Produce<'a> {
    List(Elem, &'a py::Expr),
    Set(&'a py::Expr),
    Dict(&'a py::Expr, &'a py::Expr),
    /// A generator body: each element yielded.
    Yield(&'a py::Expr),
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

pub(crate) fn cast(value: Node, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: ir(ty),
        }),
        ty,
        span,
    )
}

/// The name of a module function's value adapter.
pub(crate) fn adapter_name(name: &str) -> String {
    format!("{name}$fn")
}

pub(crate) fn new_name(class: &str) -> String {
    format!("{class}$new")
}
pub(crate) fn dispatch_name(method_fn: &str) -> String {
    format!("{method_fn}$dispatch")
}
pub(crate) fn getattr_name(attr: &str) -> String {
    format!("py$getattr${attr}")
}
pub(crate) fn setattr_name(attr: &str) -> String {
    format!("py$setattr${attr}")
}
pub(crate) fn callm_name(method: &str, arity: usize) -> String {
    format!("py$call${method}${arity}")
}

/// A method's signature as its callers see it: without `self`.
pub(crate) fn without_self(sig: &Sig) -> Sig {
    Sig {
        params: sig.params[1..].to_vec(),
        ret: sig.ret,
        defaults: sig
            .defaults
            .get(1..)
            .map(|d| d.to_vec())
            .unwrap_or_default(),
    }
}

/// A loop of one pass: its body leaves it with `break`.
fn one_pass(body: Vec<Stmt>, span: Span) -> Stmt {
    TypedNode::new(
        TypedStatement::While(TypedWhile {
            condition: Box::new(node(
                TypedExpression::Literal(TypedLiteral::Bool(true)),
                Ty::Bool,
                span,
            )),
            body: TypedBlock {
                statements: body,
                span,
            },
            span,
        }),
        Type::Unknown,
        span,
    )
}

fn op_text(op: py::Operator) -> &'static str {
    match op {
        py::Operator::Add => "+",
        py::Operator::Sub => "-",
        py::Operator::Mult => "*",
        py::Operator::MatMult => "@",
        py::Operator::Div => "/",
        py::Operator::Mod => "%",
        py::Operator::Pow => "**",
        py::Operator::LShift => "<<",
        py::Operator::RShift => ">>",
        py::Operator::BitOr => "|",
        py::Operator::BitXor => "^",
        py::Operator::BitAnd => "&",
        py::Operator::FloorDiv => "//",
    }
}

/// An address as a number, sized by the target.
pub(crate) fn addr_type() -> Type {
    Type::Primitive(PrimitiveType::USize)
}

/// The address of function `name`.
pub(crate) fn code_of(name: &str, span: Span) -> Node {
    TypedNode::new(TypedExpression::Variable(intern(name)), addr_type(), span)
}

/// `value` as an address.
pub(crate) fn as_addr(value: Node, span: Span) -> Node {
    TypedNode::new(
        TypedExpression::Cast(TypedCast {
            expr: Box::new(value),
            target_type: addr_type(),
        }),
        addr_type(),
        span,
    )
}

/// A call whose result is an address.
pub(crate) fn addr_call(name: &str, args: Vec<Node>, span: Span) -> Node {
    TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(var(intern(name), Ty::Unknown, span)),
            positional_args: args,
            named_args: Vec::new(),
            type_args: Vec::new(),
        }),
        addr_type(),
        span,
    )
}

pub(crate) fn int32_lit(v: i32, span: Span) -> Node {
    node(
        TypedExpression::Literal(TypedLiteral::Integer(v as i128)),
        Ty::Int,
        span,
    )
}

/// A Python function may keep anything it is handed: store it in a
/// list, capture it, return it. So a caller keeps no claim on a heap
/// value it passes.
fn ownership_of(ty: Ty) -> ParamOwnership {
    match ty {
        Ty::Int | Ty::Float | Ty::Bool | Ty::None => ParamOwnership::Copied,
        _ => ParamOwnership::Owned,
    }
}

fn parameter(name: &str, ty: Ty, span: Span) -> TypedParameter {
    TypedParameter {
        name: intern(name),
        ty: ir(ty),
        mutability: Mutability::Mutable,
        kind: ParameterKind::Regular,
        default_value: None,
        attributes: Vec::new(),
        ownership: ownership_of(ty),
        span,
    }
}

/// A module function in the shape of a function value: the record is
/// ignored, each argument is unboxed to the declared type, the result
/// is boxed.
pub(crate) fn adapter(module: &Module, name: &str, sig: &Sig) -> TypedFunction {
    let span = Span::new(0, 0);
    let scope = Scope::default();
    let mut lowerer = Lowerer::new(
        module,
        name,
        sig.clone(),
        Locals::default(),
        &scope,
        Vec::new(),
        HashMap::new(),
    );
    let mut params = vec![parameter("env", Ty::List(Elem::Object), span)];
    let mut args = Vec::new();
    for (i, (_, ty)) in sig.params.iter().enumerate() {
        let arg = format!("a{i}");
        params.push(parameter(&arg, Ty::Object, span));
        args.push(lowerer.coerce(
            Val {
                node: var(intern(&arg), Ty::Object, span),
                ty: Ty::Object,
            },
            *ty,
        ));
    }
    let result = Val {
        node: call(name, args, sig.ret, span),
        ty: sig.ret,
    };
    let statements = if sig.ret == Ty::None {
        let none = Val {
            node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
            ty: Ty::None,
        };
        let none = lowerer.coerce(none, Ty::Object);
        vec![
            TypedNode::new(
                TypedStatement::Expression(Box::new(result.node)),
                Type::Unknown,
                span,
            ),
            TypedNode::new(
                TypedStatement::Return(Some(Box::new(none))),
                Type::Unknown,
                span,
            ),
        ]
    } else {
        let boxed = lowerer.coerce(result, Ty::Object);
        vec![TypedNode::new(
            TypedStatement::Return(Some(Box::new(boxed))),
            Type::Unknown,
            span,
        )]
    };
    TypedFunction {
        name: intern(&adapter_name(name)),
        annotations: Vec::new(),
        effects: Vec::new(),
        with_handlers: Vec::new(),
        type_params: Vec::new(),
        params,
        return_type: ir(Ty::Object),
        body: Some(TypedBlock { statements, span }),
        visibility: Visibility::Public,
        is_async: false,
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
        link_name: None,
        module: None,
    }
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
    /// The function's name, which nested functions prefix theirs with.
    name: String,
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
    /// Variables shared with nested functions. Each lives in a
    /// one-element list, the cell, that every function using it holds.
    cells: HashMap<String, InternedString>,
    /// The enclosing function's variables this one uses, in the order
    /// their cells sit in the function record after code and arity.
    captured: Vec<String>,
    /// The types of the captured variables, as the owner has them.
    captured_types: HashMap<String, Ty>,
    /// The class whose method this is, for `super()`.
    pub(crate) class: Option<usize>,
    /// How control leaves when an exception is pending: out of the
    /// innermost loop when inside a `try`, out of the function otherwise.
    escapes: Vec<Escape>,
    /// Whether the statement being lowered emitted a pending check, so a
    /// loop containing it re-checks once the loop is left.
    raised: bool,
    /// The variable holding the exception being handled, for a bare
    /// `raise`.
    caught: Option<InternedString>,
    /// The `try` bodies being lowered, innermost last. A `return`,
    /// `break` or `continue` inside one records what it wants in the
    /// body's flag and leaves the body; the flag is acted on after
    /// `finally`.
    try_ctls: Vec<TryCtl>,
    /// Whether the statement being lowered recorded such a flag, so a
    /// loop containing it leaves again once the loop is left.
    redirected: bool,
    /// Whether this function yields: it lowers to a fiber of dynamic
    /// values, and `return` ends it.
    is_generator: bool,
}

/// One `try` body's control flag: 0 fell through, 1 return, 2 break,
/// 3 continue.
struct TryCtl {
    flag: InternedString,
    /// Where a `return` inside the body parks its value.
    ret: Option<InternedString>,
    /// How many loops of the body's own the statement being lowered is
    /// inside; a `break` at depth 0 is the body's to redirect.
    loop_depth: usize,
}

/// Where a pending exception sends control.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Escape {
    /// Return a placeholder from the function; the caller checks.
    Return,
    /// Leave the innermost loop: a `try` body's own loop, or a loop
    /// inside it, whose following check leaves the next.
    Break,
}

/// The module variable holding the exception in flight, `None` when none.
pub(crate) const PENDING: &str = "py$exc";

/// The function record's fixed prefix: code address and arity.
const RECORD_CELLS_AT: usize = 2;

impl<'m> Lowerer<'m> {
    pub(crate) fn new(
        module: &'m Module,
        name: &str,
        sig: Sig,
        locals: Locals,
        scope: &Scope,
        captured: Vec<String>,
        captured_types: HashMap<String, Ty>,
    ) -> Self {
        let bound = sig.params.iter().map(|(n, _)| intern(n)).collect();
        let is_generator = sig.ret == Ty::Gen;
        // A variable a nested function reads or assigns is shared
        // through a cell, as is everything captured from further out.
        let mut cells: BTreeSet<String> = captured.iter().cloned().collect();
        for (_, child) in &scope.children {
            for n in &child.free {
                let own = scope.bound.contains(n)
                    || sig.params.iter().any(|(p, _)| p == n)
                    || captured.contains(n);
                if own && !module.funcs.contains_key(n) {
                    cells.insert(n.clone());
                }
            }
        }
        Self {
            module,
            name: name.to_string(),
            sig,
            locals,
            bound,
            temps: 0,
            hoisted: Vec::new(),
            cells: cells
                .into_iter()
                .map(|n| {
                    let cell = intern(&format!("{n}$cell"));
                    (n, cell)
                })
                .collect(),
            captured,
            captured_types,
            class: None,
            escapes: vec![Escape::Return],
            raised: false,
            caught: None,
            try_ctls: Vec::new(),
            redirected: false,
            is_generator,
        }
    }

    /// `return value`, or the `try` body's way of recording one.
    fn emit_return(&mut self, value: Option<Node>, span: Span, out: &mut Vec<Stmt>) {
        if let Some(ctl) = self.try_ctls.last() {
            let flag = ctl.flag;
            if let (Some(slot), Some(v)) = (ctl.ret, value) {
                out.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(binary(
                        BinaryOp::Assign,
                        var(slot, self.sig.ret, span),
                        v,
                        Ty::None,
                        span,
                    ))),
                    Type::Unknown,
                    span,
                ));
            }
            self.set_flag_and_leave(flag, 1, span, out);
            return;
        }
        out.push(TypedNode::new(
            TypedStatement::Return(value.map(Box::new)),
            Type::Unknown,
            span,
        ));
    }

    /// `break` or `continue` (`code` 2 or 3), or the `try` body's way of
    /// recording one when the loop it means is outside the body.
    fn emit_loop_exit(&mut self, code: i64, span: Span, out: &mut Vec<Stmt>) {
        if let Some(ctl) = self.try_ctls.last() {
            if ctl.loop_depth == 0 {
                let flag = ctl.flag;
                self.set_flag_and_leave(flag, code, span, out);
                return;
            }
        }
        let st = if code == 2 {
            TypedStatement::Break(None)
        } else {
            TypedStatement::Continue
        };
        out.push(TypedNode::new(st, Type::Unknown, span));
    }

    fn set_flag_and_leave(
        &mut self,
        flag: InternedString,
        code: i64,
        span: Span,
        out: &mut Vec<Stmt>,
    ) {
        out.push(TypedNode::new(
            TypedStatement::Expression(Box::new(binary(
                BinaryOp::Assign,
                var(flag, Ty::Int, span),
                int_lit(code, span),
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        ));
        out.push(TypedNode::new(
            TypedStatement::Break(None),
            Type::Unknown,
            span,
        ));
        self.redirected = true;
    }

    /// Lower a loop body that sits inside a `try`, counting the loop so
    /// a `break` inside it is the loop's own.
    fn in_loop<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        if let Some(ctl) = self.try_ctls.last_mut() {
            ctl.loop_depth += 1;
        }
        let r = f(self);
        if let Some(ctl) = self.try_ctls.last_mut() {
            ctl.loop_depth -= 1;
        }
        r
    }

    /// Whether an exception is pending.
    fn pending(&self, span: Span) -> Node {
        binary(
            BinaryOp::Ne,
            call(
                "zb_any_category",
                vec![var(intern(PENDING), Ty::Object, span)],
                Ty::Int,
                span,
            ),
            int_lit(0, span),
            Ty::Bool,
            span,
        )
    }

    /// `py$exc = value`.
    fn set_pending(&mut self, value: Node, span: Span) -> Stmt {
        TypedNode::new(
            TypedStatement::Expression(Box::new(binary(
                BinaryOp::Assign,
                var(intern(PENDING), Ty::Object, span),
                value,
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        )
    }

    /// Leave, the way the current context leaves.
    fn escape(&mut self, span: Span) -> Stmt {
        let st = match self.escapes.last().copied().unwrap_or(Escape::Return) {
            Escape::Break => TypedStatement::Break(None),
            Escape::Return => TypedStatement::Return(self.placeholder(span).map(Box::new)),
        };
        TypedNode::new(st, Type::Unknown, span)
    }

    /// The value a function returns when it leaves with an exception
    /// pending, which its caller never reads.
    fn placeholder(&mut self, span: Span) -> Option<Node> {
        if self.is_generator {
            return None;
        }
        Some(match self.sig.ret {
            Ty::None => return None,
            Ty::Int => int_lit(0, span),
            Ty::Float => node(
                TypedExpression::Literal(TypedLiteral::Float(0.0)),
                Ty::Float,
                span,
            ),
            Ty::Bool => node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                Ty::Bool,
                span,
            ),
            Ty::Str => str_lit("", span),
            Ty::Object | Ty::Unknown => {
                let none = Val {
                    node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                    ty: Ty::None,
                };
                self.coerce(none, Ty::Object)
            }
            other => cast(int_lit(0, span), other, span),
        })
    }

    /// `if <pending> { <escape> }`.
    fn pending_check(&mut self, span: Span) -> Stmt {
        self.raised = true;
        let cond = self.pending(span);
        let leave = self.escape(span);
        TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(cond),
                then_block: TypedBlock {
                    statements: vec![leave],
                    span,
                },
                else_block: None,
                span,
            }),
            Type::Unknown,
            span,
        )
    }

    /// A value from a call that may have raised: held, then checked
    /// before anything uses it.
    fn guard(&mut self, v: Val, span: Span) -> Val {
        let mut pre = Vec::new();
        let held = if v.ty == Ty::None {
            pre.push(TypedNode::new(
                TypedStatement::Expression(Box::new(v.node)),
                Type::Unknown,
                span,
            ));
            Val {
                node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                ty: Ty::None,
            }
        } else {
            self.hold(v, &mut pre, span)
        };
        pre.push(self.pending_check(span));
        self.hoisted.extend(pre);
        held
    }

    /// `raise Class(message)` built here: pending set, control leaving.
    fn raise_named(&mut self, class: &str, message: Node, span: Span, out: &mut Vec<Stmt>) {
        let Some(&k) = self.module.class_index.get(class) else {
            return;
        };
        let instance = Val {
            node: call(&new_name(class), vec![message], Ty::Class(k as u16), span),
            ty: Ty::Class(k as u16),
        };
        let boxed = self.coerce(instance, Ty::Object);
        out.push(self.set_pending(boxed, span));
        self.raised = true;
        out.push(self.escape(span));
    }

    /// Integer division and remainder trap on zero, so the divisor is
    /// checked first and a zero raises.
    fn nonzero(&mut self, divisor: Node, span: Span) -> Node {
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: divisor,
                ty: Ty::Int,
            },
            &mut pre,
            span,
        );
        let mut raise = Vec::new();
        self.raise_named(
            "ZeroDivisionError",
            str_lit("integer division or modulo by zero", span),
            span,
            &mut raise,
        );
        pre.push(TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(binary(
                    BinaryOp::Eq,
                    held.node.clone(),
                    int_lit(0, span),
                    Ty::Bool,
                    span,
                )),
                then_block: TypedBlock {
                    statements: raise,
                    span,
                },
                else_block: None,
                span,
            }),
            Type::Unknown,
            span,
        ));
        self.hoisted.extend(pre);
        held.node
    }

    fn typer(&self) -> Typer<'_> {
        Typer {
            module: self.module,
            vars: &self.locals.vars,
            outer: &self.captured_types,
        }
    }

    fn ty_of(&self, e: &py::Expr) -> Ty {
        self.typer().expr(e)
    }

    fn var_ty(&self, name: &str) -> Ty {
        self.locals
            .vars
            .get(name)
            .or_else(|| self.captured_types.get(name))
            .or_else(|| self.module.globals.get(name))
            .copied()
            .unwrap_or(Ty::Object)
    }

    /// Whether `name` here is the module's variable rather than a local.
    fn is_global(&self, name: &str) -> bool {
        !self.locals.vars.contains_key(name)
            && !self.cells.contains_key(name)
            && self.module.globals.contains_key(name)
    }

    /// Whether `name` is a variable of some kind here, as opposed to a
    /// module function or a builtin.
    fn is_variable(&self, name: &str) -> bool {
        self.cells.contains_key(name)
            || self.locals.vars.contains_key(name)
            || self.module.globals.contains_key(name)
    }

    /// The statements that set up this function's cells: a captured one
    /// comes out of the record, a parameter's starts holding the
    /// parameter, any other starts empty.
    fn cell_prologue(&mut self, span: Span) -> Vec<Stmt> {
        let mut out = Vec::new();
        let names: Vec<String> = self.cells.keys().cloned().collect();
        for name in names {
            let cell = self.cells[&name];
            let init = if let Some(i) = self.captured.iter().position(|c| *c == name) {
                let slot = int_lit((RECORD_CELLS_AT + i) as i64, span);
                let element = call(
                    "zb_list_get_any",
                    vec![var(intern("env"), Ty::List(Elem::Object), span), slot],
                    Ty::Object,
                    span,
                );
                call(
                    "zb_list_unbox_any",
                    vec![element],
                    Ty::List(Elem::Object),
                    span,
                )
            } else if let Some((_, ty)) = self.sig.params.iter().find(|(p, _)| *p == name) {
                let value = Val {
                    node: var(intern(&name), *ty, span),
                    ty: *ty,
                };
                self.list_of(vec![value], Elem::Object, span)
            } else {
                let none = Val {
                    node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                    ty: Ty::None,
                };
                self.list_of(vec![none], Elem::Object, span)
            };
            self.bound.push(cell);
            out.push(TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: cell,
                    ty: ir(Ty::List(Elem::Object)),
                    mutability: Mutability::Mutable,
                    initializer: Some(Box::new(init)),
                    span,
                }),
                Type::Unknown,
                span,
            ));
        }
        out
    }

    /// Read the shared variable `name` out of its cell.
    fn cell_read(&mut self, name: &str, span: Span) -> Val {
        let ty = self.var_ty(name);
        let cell = var(self.cells[name], Ty::List(Elem::Object), span);
        let element = call(
            "zb_list_get_any",
            vec![cell, int_lit(0, span)],
            Ty::Object,
            span,
        );
        let node = self.coerce(
            Val {
                node: element,
                ty: Ty::Object,
            },
            ty,
        );
        Val { node, ty }
    }

    /// A global holds its own scalar type; anything else is stored boxed,
    /// so a list global is one heap header shared by every reader.
    fn storage(ty: Ty) -> Ty {
        match ty {
            Ty::Int | Ty::Float | Ty::Bool | Ty::Str => ty,
            _ => Ty::Object,
        }
    }

    /// Read the module variable `name`.
    fn global_read(&mut self, name: &str, span: Span) -> Val {
        let ty = self.var_ty(name);
        let stored = Self::storage(ty);
        let node = self.coerce(
            Val {
                node: var(intern(name), stored, span),
                ty: stored,
            },
            ty,
        );
        Val { node, ty }
    }

    /// A local no Python program can spell.
    fn temp(&mut self) -> InternedString {
        self.temps += 1;
        intern(&format!("__tmp{}", self.temps))
    }

    // ─── Functions ──────────────────────────────────────────────────

    /// A `def` as a function, under `name`: its own for a module
    /// function, `Class$m` for a method.
    pub(crate) fn function_named(
        &mut self,
        f: &py::StmtFunctionDef,
        name: &str,
    ) -> Result<TypedFunction> {
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
        if self.is_generator {
            prologue.extend(self.generator_prologue(span_of(f)));
        }
        for (p, (_, declared)) in f
            .parameters
            .iter_non_variadic_params()
            .zip(self.sig.params.clone())
        {
            let name = intern(p.parameter.name.as_str());
            if self.is_generator {
                // The argument arrives in the environment, after the
                // record slots and the captured cells.
                let index = RECORD_CELLS_AT + self.captured.len() + params.len();
                params.push(parameter(p.parameter.name.as_str(), declared, span_of(p)));
                let value = self.coerce(
                    Val {
                        node: call(
                            "zb_list_get_any",
                            vec![
                                var(intern("env"), Ty::List(Elem::Object), span_of(p)),
                                int_lit(index as i64, span_of(p)),
                            ],
                            Ty::Object,
                            span_of(p),
                        ),
                        ty: Ty::Object,
                    },
                    declared,
                );
                let local = self.var_ty(p.parameter.name.as_str());
                let value = self.coerce(
                    Val {
                        node: value,
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
                        span: span_of(p),
                    }),
                    Type::Unknown,
                    span_of(p),
                ));
                continue;
            }
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
                ownership: ownership_of(declared),
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
        statements.extend(self.cell_prologue(span));
        for s in &f.body {
            self.stmt(s, &mut statements)?;
        }
        // A generator is a fiber whose declared type is what it yields
        // and whose arguments arrive in its environment.
        let return_type = if self.is_generator {
            Type::Any
        } else {
            ir(self.sig.ret)
        };
        let params = if self.is_generator {
            Vec::new()
        } else {
            params
        };
        Ok(TypedFunction {
            name: intern(name),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params,
            return_type,
            body: Some(TypedBlock { statements, span }),
            visibility: Visibility::Public,
            is_async: false,
            is_fiber: self.is_generator,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
            link_name: None,
            module: None,
        })
    }

    /// The module body as the entry function's statements.
    pub(crate) fn body(&mut self, stmts: &[&py::Stmt]) -> Result<Vec<Stmt>> {
        let mut out = match stmts.first() {
            Some(first) => self.cell_prologue(span_of(*first)),
            None => Vec::new(),
        };
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
            (Ty::Dict, Ty::Object) => call("zb_dict_box", vec![v.node], Ty::Object, span),
            (Ty::Set, Ty::Object) => call("zb_set_box", vec![v.node], Ty::Object, span),
            (Ty::Object, Ty::List(e)) => call(&list_fn("unbox", e), vec![v.node], target, span),
            (Ty::Object, Ty::Tuple) => call("zb_unbox_tuple", vec![v.node], Ty::Tuple, span),
            (Ty::Object, Ty::Dict) => call("zb_dict_unbox", vec![v.node], Ty::Dict, span),
            (Ty::Object, Ty::Set) => call("zb_set_unbox", vec![v.node], Ty::Set, span),
            // An instance is boxed as its address under the class tag, and
            // read back with a check; a subclass instance is its base.
            (Ty::Class(k), Ty::Object) => {
                let address = as_addr(v.node, span);
                call(
                    "zb_box_instance_raw",
                    vec![
                        address,
                        int32_lit(zyntax_builtins::instance_tag(k as usize) as i32, span),
                    ],
                    Ty::Object,
                    span,
                )
            }
            (Ty::Object, Ty::Class(k)) => {
                let address = addr_call(
                    &format!("{}$unbox", self.module.classes[k as usize].name),
                    vec![v.node],
                    span,
                );
                cast(address, target, span)
            }
            // Up or down one chain, the instance is the same address.
            (Ty::Class(a), Ty::Class(b))
                if self.module.is_subclass(a as usize, b as usize)
                    || self.module.is_subclass(b as usize, a as usize) =>
            {
                cast(v.node, target, span)
            }
            // A dict iterates as its keys; a set is its list of elements;
            // a generator is run to exhaustion.
            (Ty::Dict, Ty::List(Elem::Object)) => call("zb_dict_keys", vec![v.node], target, span),
            (Ty::Gen, Ty::List(Elem::Object)) => self.generator_to_list(v.node, span),
            (Ty::Set, Ty::List(Elem::Object)) => Node {
                ty: ir(target),
                ..v.node
            },
            // Lists of one kind into lists of dynamic values.
            (Ty::List(e), Ty::List(Elem::Object)) => {
                call(&list_fn("to_any", e), vec![v.node], target, span)
            }
            (Ty::Tuple, Ty::List(Elem::Object)) => Node {
                ty: ir(target),
                ..v.node
            },
            // None is the null dynamic value.
            (Ty::None, Ty::Object) => node(
                TypedExpression::Literal(TypedLiteral::Null),
                Ty::Object,
                span,
            ),
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
    pub(crate) fn truthy(&mut self, v: Val) -> Node {
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
            Ty::Dict => binary(
                BinaryOp::Ne,
                call("zb_dict_len", vec![v.node], Ty::Int, span),
                int_lit(0, span),
                Ty::Bool,
                span,
            ),
            Ty::Gen => node(
                TypedExpression::Literal(TypedLiteral::Bool(true)),
                Ty::Bool,
                span,
            ),
            Ty::Class(k) => {
                if let Some(r) = self.dunder(k as usize, "__bool__", v.node.clone(), vec![], span) {
                    self.truthy(r)
                } else if let Some(r) = self.dunder(k as usize, "__len__", v.node, vec![], span) {
                    self.truthy(r)
                } else {
                    node(
                        TypedExpression::Literal(TypedLiteral::Bool(true)),
                        Ty::Bool,
                        span,
                    )
                }
            }
            Ty::List(_) | Ty::Tuple | Ty::Set => binary(
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
            Ty::Dict => call("zb_dict_repr", vec![v.node], Ty::Str, span),
            Ty::Set => call("zb_set_repr", vec![v.node], Ty::Str, span),
            Ty::Gen => str_lit("<generator object>", span),
            Ty::Class(k) => {
                let k = k as usize;
                match self
                    .dunder(k, "__str__", v.node.clone(), vec![], span)
                    .or_else(|| self.dunder(k, "__repr__", v.node, vec![], span))
                {
                    Some(r) => self.coerce(r, Ty::Str),
                    None => str_lit(&format!("<{} object>", self.module.classes[k].name), span),
                }
            }
            Ty::Object | Ty::Unknown => call("zb_any_str", vec![v.node], Ty::Str, span),
        }
    }

    /// `repr(v)`.
    pub(crate) fn repr_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Str => call("zb_str_repr", vec![v.node], Ty::Str, span),
            Ty::Class(k) => match self.dunder(k as usize, "__repr__", v.node.clone(), vec![], span)
            {
                Some(r) => self.coerce(r, Ty::Str),
                None => self.str_of(v),
            },
            Ty::Object | Ty::Unknown => call("zb_any_repr", vec![v.node], Ty::Str, span),
            _ => self.str_of(v),
        }
    }

    /// A list literal of `elem` kind from already lowered elements.
    pub(crate) fn list_of(&mut self, items: Vec<Val>, elem: Elem, span: Span) -> Node {
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
        let raised_before = self.raised;
        self.raised = false;
        let mut own = Vec::new();
        self.stmt_into(s, &mut own)?;
        // Whatever the statement's expressions hoisted runs first.
        out.append(&mut self.hoisted);
        out.append(&mut own);
        // A check inside a loop left that loop; inside a `try`, the
        // statement after the loop leaves the next.
        if self.raised
            && matches!(s, py::Stmt::For(_) | py::Stmt::While(_))
            && self.escapes.last() == Some(&Escape::Break)
        {
            let check = self.pending_check(span_of(s));
            out.push(check);
        }
        // Likewise a `return`, `break` or `continue` recorded inside a
        // loop: leave this level too.
        if self.redirected && matches!(s, py::Stmt::For(_) | py::Stmt::While(_)) {
            if let Some(ctl) = self.try_ctls.last() {
                let flag = ctl.flag;
                let sp = span_of(s);
                out.push(TypedNode::new(
                    TypedStatement::If(TypedIf {
                        condition: Box::new(binary(
                            BinaryOp::Ne,
                            var(flag, Ty::Int, sp),
                            int_lit(0, sp),
                            Ty::Bool,
                            sp,
                        )),
                        then_block: TypedBlock {
                            statements: vec![TypedNode::new(
                                TypedStatement::Break(None),
                                Type::Unknown,
                                sp,
                            )],
                            span: sp,
                        },
                        else_block: None,
                        span: sp,
                    }),
                    Type::Unknown,
                    sp,
                ));
            }
        }
        self.raised |= raised_before;
        Ok(())
    }

    fn stmt_into(&mut self, s: &py::Stmt, out: &mut Vec<Stmt>) -> Result<()> {
        let span = span_of(s);
        let push = |out: &mut Vec<Stmt>, st: TypedStatement| {
            out.push(TypedNode::new(st, Type::Unknown, span));
        };
        match s {
            py::Stmt::Pass(_) => {}
            py::Stmt::Return(r) if self.is_generator => {
                if r.value.is_some() {
                    return unsupported("a value returned from a generator", r);
                }
                self.emit_return(None, span, out);
            }
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
                self.emit_return(value.map(|b| *b), span, out);
            }
            py::Stmt::Expr(e) if matches!(&*e.value, py::Expr::Yield(_)) => {
                let py::Expr::Yield(y) = &*e.value else {
                    unreachable!()
                };
                let value = match &y.value {
                    Some(v) => self.expr_as(v, Ty::Object)?,
                    None => {
                        let none = Val {
                            node: node(
                                TypedExpression::Literal(TypedLiteral::Null),
                                Ty::None,
                                span,
                            ),
                            ty: Ty::None,
                        };
                        self.coerce(none, Ty::Object)
                    }
                };
                push(out, TypedStatement::Yield(Box::new(value)));
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
                self.if_chain(&i.test, &i.body, &i.elif_else_clauses, span, out)?;
            }
            py::Stmt::While(w) => {
                if !w.orelse.is_empty() {
                    return unsupported("while/else", w);
                }
                let cond = self.expr(&w.test)?;
                let condition = self.truthy(cond);
                // A test that hoists work re-does it every pass: the loop
                // becomes `while true { work; if not test: break; body }`.
                let pre = std::mem::take(&mut self.hoisted);
                let body = self.in_loop(|this| this.block(&w.body, span))?;
                if pre.is_empty() {
                    push(
                        out,
                        TypedStatement::While(TypedWhile {
                            condition: Box::new(condition),
                            body,
                            span,
                        }),
                    );
                } else {
                    let mut statements = pre;
                    let stop = node(
                        TypedExpression::Unary(TypedUnary {
                            op: UnaryOp::Not,
                            operand: Box::new(condition),
                        }),
                        Ty::Bool,
                        span,
                    );
                    statements.push(TypedNode::new(
                        TypedStatement::If(TypedIf {
                            condition: Box::new(stop),
                            then_block: TypedBlock {
                                statements: vec![TypedNode::new(
                                    TypedStatement::Break(None),
                                    Type::Unknown,
                                    span,
                                )],
                                span,
                            },
                            else_block: None,
                            span,
                        }),
                        Type::Unknown,
                        span,
                    ));
                    statements.extend(body.statements);
                    push(
                        out,
                        TypedStatement::While(TypedWhile {
                            condition: Box::new(node(
                                TypedExpression::Literal(TypedLiteral::Bool(true)),
                                Ty::Bool,
                                span,
                            )),
                            body: TypedBlock { statements, span },
                            span,
                        }),
                    );
                }
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
                        Ty::Dict => {
                            let k = self.expr_as(&sub.slice, Ty::Object)?;
                            call("zb_dict_del", vec![seq.node, k], Ty::None, span)
                        }
                        _ => {
                            let key = self.expr_as(&sub.slice, Ty::Object)?;
                            let seq = self.coerce(seq, Ty::Object);
                            call("zb_any_delitem", vec![seq, key], Ty::None, span)
                        }
                    };
                    let fallible = match &stmt.node {
                        TypedExpression::Call(c) => self.is_fallible_callee(&c.callee),
                        _ => false,
                    };
                    push(out, TypedStatement::Expression(Box::new(stmt)));
                    if fallible {
                        let check = self.pending_check(span);
                        out.push(check);
                    }
                }
            }
            py::Stmt::Break(_) => self.emit_loop_exit(2, span, out),
            py::Stmt::Continue(_) => self.emit_loop_exit(3, span, out),
            py::Stmt::Raise(r) => {
                if r.cause.is_some() {
                    return unsupported("raise ... from ...", r);
                }
                match &r.exc {
                    None => {
                        let Some(caught) = self.caught else {
                            return unsupported("a bare `raise` outside an except clause", r);
                        };
                        let again = var(caught, Ty::Object, span);
                        out.push(self.set_pending(again, span));
                    }
                    Some(e) => {
                        // `raise E` with a bare class is `raise E()`.
                        let value = match &**e {
                            py::Expr::Name(n)
                                if self.module.class_index.contains_key(n.id.as_str()) =>
                            {
                                let k = self.module.class_index[n.id.as_str()];
                                self.construct(k, &[], &[], &**e, span)?
                            }
                            _ => self.expr(e)?,
                        };
                        let boxed = self.coerce(value, Ty::Object);
                        out.push(self.set_pending(boxed, span));
                    }
                }
                self.raised = true;
                out.push(self.escape(span));
            }
            py::Stmt::Assert(a) => {
                let test = self.expr(&a.test)?;
                let cond = self.truthy(test);
                let message = match &a.msg {
                    Some(m) => {
                        let v = self.expr(m)?;
                        self.str_of(v)
                    }
                    None => str_lit("", span),
                };
                let mut raise = Vec::new();
                self.raise_named("AssertionError", message, span, &mut raise);
                let failed = node(
                    TypedExpression::Unary(TypedUnary {
                        op: UnaryOp::Not,
                        operand: Box::new(cond),
                    }),
                    Ty::Bool,
                    span,
                );
                push(
                    out,
                    TypedStatement::If(TypedIf {
                        condition: Box::new(failed),
                        then_block: TypedBlock {
                            statements: raise,
                            span,
                        },
                        else_block: None,
                        span,
                    }),
                );
            }
            py::Stmt::Try(t) => self.try_stmt(t, span, out)?,
            // Scope analysis already made the names module variables.
            py::Stmt::Global(_) => {}
            // The cells this body shares are already set up.
            py::Stmt::Nonlocal(_) => {}
            py::Stmt::FunctionDef(f) => {
                if !f.decorator_list.is_empty() {
                    return unsupported("decorators", f);
                }
                let value = self.nested_def(f, span)?;
                let target = py::Expr::Name(py::ExprName {
                    node_index: Default::default(),
                    range: f.range(),
                    id: f.name.id.clone(),
                    ctx: py::ExprContext::Store,
                });
                self.bind(&target, value, span, out)?;
            }
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
            // `obj.attr = v`
            py::Expr::Attribute(a) => {
                let object = self.expr(&a.value)?;
                let stmt = self.set_attribute(object, a.attr.as_str(), value, span)?;
                out.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(stmt)),
                    Type::Unknown,
                    span,
                ));
                return Ok(());
            }
            // `xs[i] = v`
            py::Expr::Subscript(sub) => {
                let seq = self.expr(&sub.value)?;
                let stmt = match seq.ty {
                    Ty::List(e) => {
                        let i = self.expr_as(&sub.slice, Ty::Int)?;
                        let v = self.coerce(value, e.ty());
                        call(&list_fn("set", e), vec![seq.node, i, v], Ty::None, span)
                    }
                    Ty::Dict => {
                        let k = self.expr_as(&sub.slice, Ty::Object)?;
                        let v = self.coerce(value, Ty::Object);
                        call("zb_dict_set", vec![seq.node, k, v], Ty::None, span)
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
                let fallible = match &stmt.node {
                    TypedExpression::Call(c) => self.is_fallible_callee(&c.callee),
                    _ => false,
                };
                out.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(stmt)),
                    Type::Unknown,
                    span,
                ));
                if fallible {
                    let check = self.pending_check(span);
                    out.push(check);
                }
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
        let name = intern(n.id.as_str());
        if let Some(cell) = self.cells.get(n.id.as_str()).copied() {
            let value = self.coerce(value, ty);
            let boxed = self.coerce(Val { node: value, ty }, Ty::Object);
            let set = call(
                "zb_list_set_any",
                vec![
                    var(cell, Ty::List(Elem::Object), span),
                    int_lit(0, span),
                    boxed,
                ],
                Ty::None,
                span,
            );
            out.push(TypedNode::new(
                TypedStatement::Expression(Box::new(set)),
                Type::Unknown,
                span,
            ));
            return Ok(());
        }
        if self.is_global(n.id.as_str()) {
            let stored = Self::storage(ty);
            let value = self.coerce(value, ty);
            let value = self.coerce(Val { node: value, ty }, stored);
            let assign = binary(
                BinaryOp::Assign,
                var(name, stored, span),
                value,
                Ty::None,
                span,
            );
            out.push(TypedNode::new(
                TypedStatement::Expression(Box::new(assign)),
                Type::Unknown,
                span,
            ));
            return Ok(());
        }
        let value = self.coerce(value, ty);
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

    /// An `if` with its `elif`/`else` tail. What the test hoists runs
    /// before the statement, in `out`, not inside a branch.
    fn if_chain(
        &mut self,
        test: &py::Expr,
        body: &[py::Stmt],
        rest: &[py::ElifElseClause],
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Result<()> {
        let cond = self.expr(test)?;
        let condition = Box::new(self.truthy(cond));
        out.append(&mut self.hoisted);
        let then_block = self.block(body, span)?;
        let else_block = match rest.split_first() {
            None => None,
            Some((clause, tail)) => {
                let clause_span = span_of(clause);
                match &clause.test {
                    Some(t) => {
                        let mut statements = Vec::new();
                        self.if_chain(t, &clause.body, tail, clause_span, &mut statements)?;
                        Some(TypedBlock {
                            statements,
                            span: clause_span,
                        })
                    }
                    None => Some(self.block(&clause.body, clause_span)?),
                }
            }
        };
        out.push(TypedNode::new(
            TypedStatement::If(TypedIf {
                condition,
                then_block,
                else_block,
                span,
            }),
            Type::Unknown,
            span,
        ));
        Ok(())
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
        if seq.ty == Ty::Gen {
            return self.for_generator(f, seq, extra, span);
        }
        let elem_ty = seq.ty.element().unwrap_or(Ty::Object);
        // The iterator is evaluated once, before the loop.
        let mut prologue = std::mem::take(&mut self.hoisted);
        let seq = match seq.ty {
            // A dynamic iterable is snapshotted into a list of objects,
            // and a dict iterates over a snapshot of its keys.
            Ty::Object | Ty::Dict => {
                let items = match seq.ty {
                    Ty::Dict => call("zb_dict_keys", vec![seq.node], Ty::List(Elem::Object), span),
                    _ => call("zb_any_iter", vec![seq.node], Ty::List(Elem::Object), span),
                };
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
        self.in_loop(|this| -> Result<()> {
            for s in &f.body {
                this.stmt(s, &mut body)?;
            }
            Ok(())
        })?;
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
        // The bounds are evaluated once, before the loop.
        let pre = std::mem::take(&mut self.hoisted);
        // The loop counts in the target itself when the target is a
        // plain int local; a shared, global or object-typed target is
        // assigned from a hidden counter each time round.
        let direct = !self.cells.contains_key(target.id.as_str())
            && !self.is_global(target.id.as_str())
            && self.var_ty(target.id.as_str()) == Ty::Int;
        let name = if direct {
            let name = intern(target.id.as_str());
            if !self.bound.contains(&name) {
                self.bound.push(name);
            }
            name
        } else {
            self.temp()
        };
        let mut statements = Vec::new();
        if !direct {
            let counter = Val {
                node: var(name, Ty::Int, span),
                ty: Ty::Int,
            };
            self.bind(&f.target, counter, span, &mut statements)?;
        }
        self.in_loop(|this| -> Result<()> {
            for s in &f.body {
                this.stmt(s, &mut statements)?;
            }
            Ok(())
        })?;
        statements.extend(extra);
        let body = TypedBlock { statements, span };
        let loop_stmt = TypedStatement::For(TypedFor {
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
        });
        if pre.is_empty() {
            return Ok(loop_stmt);
        }
        let mut statements = pre;
        statements.push(TypedNode::new(loop_stmt, Type::Unknown, span));
        Ok(TypedStatement::Block(TypedBlock { statements, span }))
    }

    // ─── Expressions ────────────────────────────────────────────────

    pub(crate) fn expr(&mut self, e: &py::Expr) -> Result<Val> {
        let v = self.expr_unchecked(e)?;
        // A library call that can raise is checked before its value is
        // used, wherever the lowering above produced it.
        let fallible = match &v.node.node {
            TypedExpression::Call(c) => self.is_fallible_callee(&c.callee),
            TypedExpression::Cast(c) => match &c.expr.node {
                TypedExpression::Call(inner) => self.is_fallible_callee(&inner.callee),
                _ => false,
            },
            _ => false,
        };
        if fallible {
            let span = v.node.span;
            return Ok(self.guard(v, span));
        }
        Ok(v)
    }

    fn is_fallible_callee(&self, callee: &Node) -> bool {
        match &callee.node {
            TypedExpression::Variable(n) => n
                .resolve_global()
                .is_some_and(|name| self.module.fallible.contains(&name)),
            _ => false,
        }
    }

    fn expr_unchecked(&mut self, e: &py::Expr) -> Result<Val> {
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
            py::Expr::Name(n) if self.cells.contains_key(n.id.as_str()) => {
                self.cell_read(n.id.as_str(), span)
            }
            py::Expr::Name(n) if self.is_global(n.id.as_str()) => {
                self.global_read(n.id.as_str(), span)
            }
            // A module function named as a value.
            py::Expr::Name(n)
                if !self.is_variable(n.id.as_str())
                    && self.module.funcs.contains_key(n.id.as_str()) =>
            {
                self.function_value(n.id.as_str(), span)
            }
            py::Expr::Name(n) => Val {
                node: var(intern(n.id.as_str()), ty, span),
                ty,
            },
            py::Expr::Lambda(l) => self.lambda(l, span)?,
            py::Expr::Generator(g) => self.generator_expr(g, span)?,
            py::Expr::Attribute(a) => {
                let object = self.expr(&a.value)?;
                self.attribute(object, a.attr.as_str(), span)?
            }
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
                // What a branch hoists runs only when that branch does.
                let outer = std::mem::take(&mut self.hoisted);
                let then_value = self.expr_as(&i.body, ty)?;
                let then_pre = std::mem::take(&mut self.hoisted);
                let else_value = self.expr_as(&i.orelse, ty)?;
                let else_pre = std::mem::replace(&mut self.hoisted, outer);
                let then_branch = if then_pre.is_empty() {
                    then_value
                } else {
                    Self::block_value(then_pre, then_value, ty, span)
                };
                let else_branch = if else_pre.is_empty() {
                    else_value
                } else {
                    Self::block_value(else_pre, else_value, ty, span)
                };
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
            py::Expr::ListComp(c) => {
                let Ty::List(elem) = ty else { unreachable!() };
                self.comprehension(&c.generators, Produce::List(elem, &c.elt), span)?
            }
            py::Expr::SetComp(c) => {
                self.comprehension(&c.generators, Produce::Set(&c.elt), span)?
            }
            py::Expr::DictComp(c) => {
                let Some(key) = &c.key else {
                    return unsupported("`**` in a dict comprehension", c);
                };
                self.comprehension(&c.generators, Produce::Dict(key, &c.value), span)?
            }
            py::Expr::Dict(d) => {
                let mut items = Vec::with_capacity(d.items.len() * 2);
                for item in &d.items {
                    let Some(key) = &item.key else {
                        return unsupported("`**` in a dict literal", d);
                    };
                    items.push(self.expr(key)?);
                    items.push(self.expr(&item.value)?);
                }
                let pairs = self.list_of(items, Elem::Object, span);
                Val {
                    node: call("zb_dict_from_pairs", vec![pairs], Ty::Dict, span),
                    ty: Ty::Dict,
                }
            }
            py::Expr::Set(st) => {
                let mut items = Vec::with_capacity(st.elts.len());
                for e in &st.elts {
                    items.push(self.expr(e)?);
                }
                let elements = self.list_of(items, Elem::Object, span);
                Val {
                    node: call("zb_set_from", vec![elements], Ty::Set, span),
                    ty: Ty::Set,
                }
            }
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
        if let Ty::Class(k) = left.ty {
            let name = match op {
                py::Operator::Add => "__add__",
                py::Operator::Sub => "__sub__",
                py::Operator::Mult => "__mul__",
                py::Operator::Div => "__truediv__",
                py::Operator::FloorDiv => "__floordiv__",
                py::Operator::Mod => "__mod__",
                py::Operator::Pow => "__pow__",
                py::Operator::MatMult => "__matmul__",
                py::Operator::BitAnd => "__and__",
                py::Operator::BitOr => "__or__",
                py::Operator::BitXor => "__xor__",
                py::Operator::LShift => "__lshift__",
                py::Operator::RShift => "__rshift__",
            };
            let other = self.coerce(right, Ty::Object);
            if let Some(r) = self.dunder(k as usize, name, left.node, vec![other], span) {
                return Ok(r);
            }
            return Err(Error::Unsupported {
                what: format!(
                    "`{}` on {}, which defines no {name}",
                    op_text(op),
                    self.module.classes[k as usize].name
                ),
                at: span.start,
            });
        }
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
        // Set algebra.
        if left.ty == Ty::Set && right.ty == Ty::Set {
            let name = match op {
                py::Operator::BitAnd => Some("zb_set_and"),
                py::Operator::BitOr => Some("zb_set_or"),
                py::Operator::Sub => Some("zb_set_sub"),
                py::Operator::BitXor => Some("zb_set_xor"),
                _ => None,
            };
            if let Some(name) = name {
                return Ok(Val {
                    node: call(name, vec![left.node, right.node], Ty::Set, span),
                    ty: Ty::Set,
                });
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
        let r = if operand_ty == Ty::Int && matches!(op, py::Operator::FloorDiv | py::Operator::Mod)
        {
            self.nonzero(r, span)
        } else {
            r
        };
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
        if let Ty::Class(k) = left.ty {
            let dunder = match op {
                py::CmpOp::Eq => Some(("__eq__", false)),
                py::CmpOp::NotEq => Some(("__ne__", false)),
                py::CmpOp::Lt => Some(("__lt__", false)),
                py::CmpOp::LtE => Some(("__le__", false)),
                py::CmpOp::Gt => Some(("__gt__", false)),
                py::CmpOp::GtE => Some(("__ge__", false)),
                _ => None,
            };
            if let Some((name, _)) = dunder {
                let right_ty = right.ty;
                let right_node = right.node.clone();
                let other = self.coerce(right, Ty::Object);
                if let Some(r) = self.dunder(
                    k as usize,
                    name,
                    left.node.clone(),
                    vec![other.clone()],
                    span,
                ) {
                    return Ok(self.truthy(r));
                }
                if name == "__ne__" {
                    if let Some(r) = self.dunder(
                        k as usize,
                        "__eq__",
                        left.node.clone(),
                        vec![other.clone()],
                        span,
                    ) {
                        let t = self.truthy(r);
                        return Ok(negate(t));
                    }
                }
                if matches!(op, py::CmpOp::Eq | py::CmpOp::NotEq) {
                    // No `__eq__`: identity.
                    let l = self.coerce(left, Ty::Object);
                    let same = call("zb_any_same", vec![l, other], Ty::Bool, span);
                    return Ok(if op == py::CmpOp::NotEq {
                        negate(same)
                    } else {
                        same
                    });
                }
                return Err(Error::Unsupported {
                    what: format!(
                        "ordering {} against a {right_ty:?}, which defines no {name}",
                        self.module.classes[k as usize].name
                    ),
                    at: right_node.span.start,
                });
            }
        }
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
                } else if right.ty == Ty::Tuple || right.ty == Ty::Set {
                    let item = self.coerce(left, Ty::Object);
                    call(
                        "zb_list_contains_any",
                        vec![right.node, item],
                        Ty::Bool,
                        span,
                    )
                } else if right.ty == Ty::Dict {
                    let item = self.coerce(left, Ty::Object);
                    call("zb_dict_contains", vec![right.node, item], Ty::Bool, span)
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
                    // Containers are identical when they are the same heap object.
                    (Ty::List(_) | Ty::Tuple | Ty::Dict | Ty::Set, _)
                    | (_, Ty::List(_) | Ty::Tuple | Ty::Dict | Ty::Set) => {
                        let l = self.coerce(left, Ty::Object);
                        let r = self.coerce(right, Ty::Object);
                        call("zb_any_same", vec![l, r], Ty::Bool, span)
                    }
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
        // Sets order by inclusion.
        if left.ty == Ty::Set && right.ty == Ty::Set {
            let subset = |this: &mut Self, a: Node, b: Node| {
                let _ = this;
                call("zb_set_issubset", vec![a, b], Ty::Bool, span)
            };
            let proper = |this: &mut Self, a: Node, b: Node| {
                let sub = subset(this, a.clone(), b.clone());
                let same_size = binary(
                    BinaryOp::Eq,
                    method_call(a, "len", vec![], Ty::Int, span),
                    method_call(b, "len", vec![], Ty::Int, span),
                    Ty::Bool,
                    span,
                );
                binary(BinaryOp::And, sub, negate(same_size), Ty::Bool, span)
            };
            let (l, r) = (left.node, right.node);
            return Ok(match op {
                py::CmpOp::LtE => subset(self, l, r),
                py::CmpOp::GtE => subset(self, r, l),
                py::CmpOp::Lt => proper(self, l, r),
                py::CmpOp::Gt => proper(self, r, l),
                py::CmpOp::Eq => call("zb_set_eq", vec![l, r], Ty::Bool, span),
                py::CmpOp::NotEq => negate(call("zb_set_eq", vec![l, r], Ty::Bool, span)),
                _ => {
                    return Err(Error::Unsupported {
                        what: "this comparison of sets".to_string(),
                        at: span.start,
                    })
                }
            });
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
        for (i, v) in b.values.iter().enumerate() {
            // What a later operand hoists must run only if it is reached,
            // so it stays inside the operand as a block.
            let outer = std::mem::take(&mut self.hoisted);
            let val = self.expr(v)?;
            let mine = std::mem::replace(&mut self.hoisted, outer);
            if i == 0 || mine.is_empty() {
                self.hoisted.extend(mine);
                vals.push(val);
            } else {
                let ty = val.ty;
                vals.push(Val {
                    node: Self::block_value(mine, val.node, ty, span),
                    ty,
                });
            }
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
            Ty::Dict => {
                let key = self.expr_as(&sub.slice, Ty::Object)?;
                Ok(Val {
                    node: call("zb_dict_get", vec![seq.node, key], Ty::Object, span),
                    ty: Ty::Object,
                })
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

    /// `[e for x in it if c]`, `{e for ...}`, `{k: v for ...}`: a fresh
    /// collection, a loop adding to it, the collection as the value.
    fn comprehension(
        &mut self,
        generators: &[py::Comprehension],
        produce: Produce<'_>,
        span: Span,
    ) -> Result<Val> {
        let ty = match produce {
            Produce::List(elem, _) => Ty::List(elem),
            Produce::Set(_) => Ty::Set,
            Produce::Dict(..) => Ty::Dict,
            Produce::Yield(_) => Ty::None,
        };
        let out = self.temp();
        // Loop variables are the comprehension's own; they shadow the
        // function's for the body and are forgotten after.
        let saved_vars = self.locals.vars.clone();
        let mut statements = if matches!(produce, Produce::Yield(_)) {
            Vec::new()
        } else {
            vec![TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: out,
                    ty: ir(ty),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(self.list_of(Vec::new(), Elem::Object, span))),
                    span,
                }),
                Type::Unknown,
                span,
            )]
        };
        if let Produce::List(elem, _) = produce {
            statements[0] = TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: out,
                    ty: ir(ty),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(self.list_of(Vec::new(), elem, span))),
                    span,
                }),
                Type::Unknown,
                span,
            );
        }
        // Build innermost first: the add, wrapped in each `if`, wrapped
        // in each `for`, from the last generator outwards.
        for g in generators {
            let item_ty = self.ty_of(&g.iter).element().unwrap_or(match &g.iter {
                py::Expr::Call(call) if types::is_name(&call.func, "range") => Ty::Int,
                _ => Ty::Object,
            });
            bind_names(&mut self.locals.vars, &g.target, item_ty);
        }
        // What the element and the conditions hoist belongs inside the
        // loop, where they are evaluated.
        let outer_hoisted = std::mem::take(&mut self.hoisted);
        let add = match produce {
            Produce::List(elem, elt) => {
                let value = self.expr(elt)?;
                let value = self.coerce(value, elem.ty());
                method_call(var(out, ty, span), "push", vec![value], Ty::None, span)
            }
            Produce::Set(elt) => {
                let value = self.expr_as(elt, Ty::Object)?;
                call(
                    "zb_set_add",
                    vec![var(out, ty, span), value],
                    Ty::None,
                    span,
                )
            }
            Produce::Dict(key, value) => {
                let k = self.expr_as(key, Ty::Object)?;
                let v = self.expr_as(value, Ty::Object)?;
                call(
                    "zb_dict_set",
                    vec![var(out, ty, span), k, v],
                    Ty::None,
                    span,
                )
            }
            Produce::Yield(elt) => self.expr_as(elt, Ty::Object)?,
        };
        let mut inner: Vec<Stmt> = std::mem::take(&mut self.hoisted);
        inner.push(TypedNode::new(
            if matches!(produce, Produce::Yield(_)) {
                TypedStatement::Yield(Box::new(add))
            } else {
                TypedStatement::Expression(Box::new(add))
            },
            Type::Unknown,
            span,
        ));
        for g in generators.iter().rev() {
            for cond in g.ifs.iter().rev() {
                let test = self.expr(cond)?;
                let condition = self.truthy(test);
                let mut with_test = std::mem::take(&mut self.hoisted);
                with_test.push(TypedNode::new(
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
                ));
                inner = with_test;
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
        self.hoisted = outer_hoisted;
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
            Ty::Dict => {
                let d = receiver.node;
                let node = match (name, args.len()) {
                    ("get", 1) => {
                        let k = self.expr_as(&args[0], Ty::Object)?;
                        let none = self.coerce(
                            Val {
                                node: node(
                                    TypedExpression::Literal(TypedLiteral::Null),
                                    Ty::None,
                                    span,
                                ),
                                ty: Ty::None,
                            },
                            Ty::Object,
                        );
                        call("zb_dict_get_default", vec![d, k, none], Ty::Object, span)
                    }
                    ("get", 2) => {
                        let k = self.expr_as(&args[0], Ty::Object)?;
                        let default = self.expr_as(&args[1], Ty::Object)?;
                        call("zb_dict_get_default", vec![d, k, default], Ty::Object, span)
                    }
                    ("setdefault", 2) => {
                        let k = self.expr_as(&args[0], Ty::Object)?;
                        let v = self.expr_as(&args[1], Ty::Object)?;
                        call("zb_dict_setdefault", vec![d, k, v], Ty::Object, span)
                    }
                    ("pop", 1) => {
                        let k = self.expr_as(&args[0], Ty::Object)?;
                        call("zb_dict_pop", vec![d, k], Ty::Object, span)
                    }
                    ("pop", 2) => {
                        let k = self.expr_as(&args[0], Ty::Object)?;
                        let default = self.expr_as(&args[1], Ty::Object)?;
                        call("zb_dict_pop_default", vec![d, k, default], Ty::Object, span)
                    }
                    ("keys", 0) => call("zb_dict_keys", vec![d], Ty::List(Elem::Object), span),
                    ("values", 0) => call("zb_dict_values", vec![d], Ty::List(Elem::Object), span),
                    ("items", 0) => call("zb_dict_items", vec![d], Ty::List(Elem::Object), span),
                    ("copy", 0) => call("zb_dict_copy", vec![d], Ty::Dict, span),
                    ("clear", 0) => method_call(d, "clear", vec![], Ty::None, span),
                    ("update", 1) => {
                        let other = self.expr_as(&args[0], Ty::Dict)?;
                        call("zb_dict_update", vec![d, other], Ty::None, span)
                    }
                    _ => {
                        return Err(Error::Unsupported {
                            what: format!("dict.{name} with {} argument(s)", args.len()),
                            at: span.start,
                        })
                    }
                };
                Ok(Val { node, ty })
            }
            Ty::Set => {
                let st = receiver.node;
                let node = match (name, args.len()) {
                    ("add", 1) => {
                        let v = self.expr_as(&args[0], Ty::Object)?;
                        call("zb_set_add", vec![st, v], Ty::None, span)
                    }
                    ("remove", 1) => {
                        let v = self.expr_as(&args[0], Ty::Object)?;
                        call("zb_set_remove", vec![st, v], Ty::None, span)
                    }
                    ("discard", 1) => {
                        let v = self.expr_as(&args[0], Ty::Object)?;
                        call("zb_set_discard", vec![st, v], Ty::None, span)
                    }
                    ("clear", 0) => method_call(st, "clear", vec![], Ty::None, span),
                    ("copy", 0) => call("zb_list_copy_any", vec![st], Ty::Set, span),
                    (
                        "union"
                        | "intersection"
                        | "difference"
                        | "symmetric_difference"
                        | "issubset"
                        | "issuperset",
                        1,
                    ) => {
                        let other = self.expr_as(&args[0], Ty::Set)?;
                        let (f, result) = match name {
                            "union" => ("zb_set_or", Ty::Set),
                            "intersection" => ("zb_set_and", Ty::Set),
                            "difference" => ("zb_set_sub", Ty::Set),
                            "symmetric_difference" => ("zb_set_xor", Ty::Set),
                            "issubset" => ("zb_set_issubset", Ty::Bool),
                            _ => ("zb_set_issubset", Ty::Bool),
                        };
                        let args = if name == "issuperset" {
                            vec![other, st]
                        } else {
                            vec![st, other]
                        };
                        call(f, args, result, span)
                    }
                    _ => {
                        return Err(Error::Unsupported {
                            what: format!("set.{name} with {} argument(s)", args.len()),
                            at: span.start,
                        })
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
        let args = &c.arguments.args;
        let keywords = &c.arguments.keywords;
        if let py::Expr::Name(n) = &*c.func {
            let name = n.id.as_str();
            if self.is_variable(name) {
                let callee = self.expr(&c.func)?;
                return self.call_value(callee, args, keywords, c, span);
            }
            if let Some(&k) = self.module.class_index.get(name) {
                return self.construct(k, args, keywords, c, span);
            }
            if let Some(sig) = self.module.funcs.get(name).cloned() {
                let lowered = self.arguments(name, &sig, args, keywords, c)?;
                if sig.ret == Ty::Gen {
                    return Ok(self.start_generator(name, &sig, lowered, Vec::new(), span));
                }
                let v = Val {
                    node: call(name, lowered, sig.ret, span),
                    ty: sig.ret,
                };
                return Ok(self.guard(v, span));
            }
            if name == "print" {
                return self.print(args, keywords, span);
            }
        }
        if let py::Expr::Attribute(a) = &*c.func {
            if types::is_super_call(&a.value) {
                return self.super_call(a.attr.as_str(), args, keywords, c, span);
            }
            let receiver = self.expr(&a.value)?;
            if let Ty::Class(k) = receiver.ty {
                return self.method_on(
                    k as usize,
                    receiver,
                    a.attr.as_str(),
                    args,
                    keywords,
                    c,
                    span,
                );
            }
            if receiver.ty == Ty::Object && !keywords.is_empty() {
                return unsupported("keyword arguments in a call through a value", c);
            }
            if receiver.ty == Ty::Object {
                return self.dynamic_method(receiver, a.attr.as_str(), args, span);
            }
        }
        if !keywords.is_empty() {
            return unsupported("keyword arguments", c);
        }
        if let py::Expr::Attribute(a) = &*c.func {
            let receiver = self.expr(&a.value)?;
            return self.method(receiver, a.attr.as_str(), args, ty, span);
        }
        if !matches!(&*c.func, py::Expr::Name(_)) {
            let callee = self.expr(&c.func)?;
            return self.call_value(callee, args, keywords, c, span);
        }
        if let py::Expr::Name(n) = &*c.func {
            let name = n.id.as_str();
            match name {
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
                "isinstance" if args.len() == 2 => {
                    let v = self.expr(&args[0])?;
                    return self.isinstance(v, &args[1], span);
                }
                "next" if !args.is_empty() && args.len() <= 2 => {
                    let g = self.expr(&args[0])?;
                    if g.ty != Ty::Gen {
                        return unsupported("next() on something other than a generator", c);
                    }
                    let default = match args.get(1) {
                        Some(d) => Some(self.expr_as(d, Ty::Object)?),
                        None => None,
                    };
                    return Ok(self.next_of(g.node, default, span));
                }
                "len" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Str => call("zb_str_chars_len", vec![v.node], Ty::Int, span),
                        Ty::List(_) | Ty::Tuple | Ty::Set => {
                            method_call(v.node, "len", vec![], Ty::Int, span)
                        }
                        Ty::Dict => call("zb_dict_len", vec![v.node], Ty::Int, span),
                        Ty::Class(k) => {
                            match self.dunder(k as usize, "__len__", v.node, vec![], span) {
                                Some(r) => self.coerce(r, Ty::Int),
                                None => {
                                    return Err(Error::Unsupported {
                                        what: format!(
                                            "len() of {}, which defines no __len__",
                                            self.module.classes[k as usize].name
                                        ),
                                        at: span.start,
                                    })
                                }
                            }
                        }
                        _ => {
                            let o = self.coerce(v, Ty::Object);
                            call("zb_any_len", vec![o], Ty::Int, span)
                        }
                    };
                    return Ok(Val { node, ty: Ty::Int });
                }
                // `range` as a value is the list of its numbers.
                "range" if !args.is_empty() && args.len() <= 3 => {
                    let mut bounds = Vec::new();
                    for a in args {
                        bounds.push(self.expr_as(a, Ty::Int)?);
                    }
                    let (start, stop, step) = match bounds.len() {
                        1 => (int_lit(0, span), bounds.remove(0), int_lit(1, span)),
                        2 => {
                            let stop = bounds.remove(1);
                            (bounds.remove(0), stop, int_lit(1, span))
                        }
                        _ => {
                            let step = bounds.remove(2);
                            let stop = bounds.remove(1);
                            (bounds.remove(0), stop, step)
                        }
                    };
                    return Ok(Val {
                        node: call(
                            "zb_list_range",
                            vec![start, stop, step],
                            Ty::List(Elem::Int),
                            span,
                        ),
                        ty: Ty::List(Elem::Int),
                    });
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
                "dict" => {
                    let node = match args.first() {
                        None => call(
                            "zb_dict_from_pairs",
                            vec![self.list_of(Vec::new(), Elem::Object, span)],
                            Ty::Dict,
                            span,
                        ),
                        Some(a) => {
                            let d = self.expr_as(a, Ty::Dict)?;
                            call("zb_dict_copy", vec![d], Ty::Dict, span)
                        }
                    };
                    return Ok(Val { node, ty: Ty::Dict });
                }
                "set" => {
                    let items = match args.first() {
                        None => self.list_of(Vec::new(), Elem::Object, span),
                        Some(a) => {
                            let v = self.expr(a)?;
                            match v.ty {
                                Ty::List(_) | Ty::Tuple | Ty::Set | Ty::Dict => {
                                    self.coerce(v, Ty::List(Elem::Object))
                                }
                                _ => {
                                    let o = self.coerce(v, Ty::Object);
                                    call("zb_any_iter", vec![o], Ty::List(Elem::Object), span)
                                }
                            }
                        }
                    };
                    return Ok(Val {
                        node: call("zb_set_from", vec![items], Ty::Set, span),
                        ty: Ty::Set,
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
                                Ty::Tuple | Ty::Set | Ty::Dict | Ty::Gen => {
                                    let node = self.coerce(v, Ty::List(Elem::Object));
                                    Val {
                                        node,
                                        ty: Ty::List(Elem::Object),
                                    }
                                }
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
                    // Both operands are held before either operation, and
                    // what the operations hoist follows them.
                    let mut statements = Vec::new();
                    let a = self.hold(a, &mut statements, span);
                    let b = self.hold(b, &mut statements, span);
                    self.hoisted.extend(statements);
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
                    return Ok(Val {
                        node: Node {
                            ty: ir(Ty::Tuple),
                            ..pair
                        },
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
                        Ty::Dict => str_lit("dict", span),
                        Ty::Set => str_lit("set", span),
                        Ty::Class(k) => str_lit(&self.module.classes[k as usize].name, span),
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
    /// `try` / `except` / `else` / `finally`.
    ///
    /// The body runs in a loop of one pass that a pending exception
    /// breaks out of. After it, the handlers match the pending exception
    /// in order; the first match clears it and runs. `else` runs when
    /// nothing was raised, `finally` always, and whatever is still
    /// pending afterwards leaves the way the enclosing context does.
    fn try_stmt(&mut self, t: &py::StmtTry, span: Span, out: &mut Vec<Stmt>) -> Result<()> {
        if t.is_star {
            return unsupported("except*", t);
        }
        // The body's control flag and, when the function returns a
        // value, the slot a `return` inside the body parks it in.
        let flag = self.temp();
        out.push(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: flag,
                ty: ir(Ty::Int),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(int_lit(0, span))),
                span,
            }),
            Type::Unknown,
            span,
        ));
        let ret_slot = match self.placeholder(span) {
            Some(zero) => {
                let slot = self.temp();
                out.push(TypedNode::new(
                    TypedStatement::Let(TypedLet {
                        name: slot,
                        ty: ir(self.sig.ret),
                        mutability: Mutability::Mutable,
                        initializer: Some(Box::new(zero)),
                        span,
                    }),
                    Type::Unknown,
                    span,
                ));
                Some(slot)
            }
            None => None,
        };
        let redirected_before = self.redirected;
        self.redirected = false;
        self.try_ctls.push(TryCtl {
            flag,
            ret: ret_slot,
            loop_depth: 0,
        });
        self.escapes.push(Escape::Break);
        let mut body = Vec::new();
        for s in &t.body {
            self.stmt(s, &mut body)?;
        }
        self.escapes.pop();
        self.try_ctls.pop();
        let body_redirected = self.redirected;
        self.redirected = redirected_before;
        body.push(TypedNode::new(
            TypedStatement::Break(None),
            Type::Unknown,
            span,
        ));
        out.push(one_pass(body, span));
        // Whether anything was raised, before a handler clears it.
        let raised = self.temp();
        let pending_now = self.pending(span);
        out.push(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: raised,
                ty: ir(Ty::Bool),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(pending_now)),
                span,
            }),
            Type::Unknown,
            span,
        ));
        // Handlers, first match wins. What matching hoists is computed
        // here, after the body, not before the statement.
        let mut chain: Option<Stmt> = None;
        let mut before_chain = Vec::new();
        for h in t.handlers.iter().rev() {
            let py::ExceptHandler::ExceptHandler(h) = h;
            let hspan = span_of(h);
            let outer_hoisted = std::mem::take(&mut self.hoisted);
            let matches = match h.type_.as_deref() {
                None => node(
                    TypedExpression::Literal(TypedLiteral::Bool(true)),
                    Ty::Bool,
                    hspan,
                ),
                Some(py::Expr::Tuple(classes)) => {
                    let mut any: Option<Node> = None;
                    for c in &classes.elts {
                        let this = self.exception_matches(c, hspan)?;
                        any = Some(match any {
                            None => this,
                            Some(prev) => binary(BinaryOp::Or, prev, this, Ty::Bool, hspan),
                        });
                    }
                    any.unwrap_or_else(|| {
                        node(
                            TypedExpression::Literal(TypedLiteral::Bool(false)),
                            Ty::Bool,
                            hspan,
                        )
                    })
                }
                Some(c) => self.exception_matches(c, hspan)?,
            };
            let mut hoisted_now = std::mem::replace(&mut self.hoisted, outer_hoisted);
            before_chain.append(&mut hoisted_now);
            let mut handler = Vec::new();
            // The exception as the handler names it, then cleared.
            let saved_caught = self.caught;
            let caught_name = self.temp();
            handler.push(TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: caught_name,
                    ty: ir(Ty::Object),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(var(intern(PENDING), Ty::Object, hspan))),
                    span: hspan,
                }),
                Type::Unknown,
                hspan,
            ));
            self.caught = Some(caught_name);
            if let Some(name) = &h.name {
                let ty = self.var_ty(name.as_str());
                let value = Val {
                    node: var(caught_name, Ty::Object, hspan),
                    ty: Ty::Object,
                };
                let target = py::Expr::Name(py::ExprName {
                    node_index: Default::default(),
                    range: name.range(),
                    id: name.id.clone(),
                    ctx: py::ExprContext::Store,
                });
                let value = Val {
                    node: self.coerce(value, ty),
                    ty,
                };
                self.bind(&target, value, hspan, &mut handler)?;
            }
            let none = Val {
                node: node(
                    TypedExpression::Literal(TypedLiteral::Null),
                    Ty::None,
                    hspan,
                ),
                ty: Ty::None,
            };
            let cleared = self.coerce(none, Ty::Object);
            handler.push(self.set_pending(cleared, hspan));
            for s in &h.body {
                self.stmt(s, &mut handler)?;
            }
            self.caught = saved_caught;
            chain = Some(TypedNode::new(
                TypedStatement::If(TypedIf {
                    condition: Box::new(matches),
                    then_block: TypedBlock {
                        statements: handler,
                        span: hspan,
                    },
                    else_block: chain.map(|c| TypedBlock {
                        statements: vec![c],
                        span: hspan,
                    }),
                    span: hspan,
                }),
                Type::Unknown,
                hspan,
            ));
        }
        let mut orelse = Vec::new();
        for s in &t.orelse {
            self.stmt(s, &mut orelse)?;
        }
        if chain.is_some() || !orelse.is_empty() {
            out.extend(before_chain);
            out.push(TypedNode::new(
                TypedStatement::If(TypedIf {
                    condition: Box::new(var(raised, Ty::Bool, span)),
                    then_block: TypedBlock {
                        statements: chain.into_iter().collect(),
                        span,
                    },
                    else_block: if orelse.is_empty() {
                        None
                    } else {
                        Some(TypedBlock {
                            statements: orelse,
                            span,
                        })
                    },
                    span,
                }),
                Type::Unknown,
                span,
            ));
        }
        for s in &t.finalbody {
            self.stmt(s, out)?;
        }
        // Whatever no handler took leaves with the enclosing context.
        let check = self.pending_check(span);
        out.push(check);
        // What the body asked for before it left.
        if body_redirected {
            for code in 1..=3 {
                let mut action = Vec::new();
                match code {
                    1 => {
                        let value = ret_slot.map(|slot| var(slot, self.sig.ret, span));
                        self.emit_return(value, span, &mut action);
                    }
                    _ => self.emit_loop_exit(code, span, &mut action),
                }
                out.push(TypedNode::new(
                    TypedStatement::If(TypedIf {
                        condition: Box::new(binary(
                            BinaryOp::Eq,
                            var(flag, Ty::Int, span),
                            int_lit(code, span),
                            Ty::Bool,
                            span,
                        )),
                        then_block: TypedBlock {
                            statements: action,
                            span,
                        },
                        else_block: None,
                        span,
                    }),
                    Type::Unknown,
                    span,
                ));
            }
        }
        Ok(())
    }

    /// Whether the pending exception is an instance of the class named.
    fn exception_matches(&mut self, class: &py::Expr, span: Span) -> Result<Node> {
        let py::Expr::Name(n) = class else {
            return unsupported(
                "an except clause naming something other than a class",
                class,
            );
        };
        if !self.module.class_index.contains_key(n.id.as_str()) {
            return unsupported(format!("except {}, which is not a class", n.id), class);
        }
        let pending = Val {
            node: var(intern(PENDING), Ty::Object, span),
            ty: Ty::Object,
        };
        let v = self.isinstance(pending, class, span)?;
        Ok(v.node)
    }

    /// The entry function's statements: the module body in a loop of one
    /// pass, then a report of whatever exception nothing caught.
    pub(crate) fn entry_body(&mut self, stmts: &[&py::Stmt]) -> Result<Vec<Stmt>> {
        let span = stmts
            .first()
            .map(|s| span_of(*s))
            .unwrap_or(Span::new(0, 0));
        self.escapes.push(Escape::Break);
        let mut body = self.body(stmts)?;
        self.escapes.pop();
        body.push(TypedNode::new(
            TypedStatement::Break(None),
            Type::Unknown,
            span,
        ));
        let pending = || var(intern(PENDING), Ty::Object, span);
        let eprint = |line: Node| {
            TypedNode::new(
                TypedStatement::Expression(Box::new(call(
                    "zb_eprintln",
                    vec![line],
                    Ty::None,
                    span,
                ))),
                Type::Unknown,
                span,
            )
        };
        // `Name: message`, or just `Name` when the message is empty.
        let text_var = self.temp();
        let kind = call("zb_any_type", vec![pending()], Ty::Str, span);
        let line = binary(
            BinaryOp::Add,
            binary(
                BinaryOp::Add,
                kind.clone(),
                str_lit(": ", span),
                Ty::Str,
                span,
            ),
            var(text_var, Ty::Str, span),
            Ty::Str,
            span,
        );
        let report = vec![
            TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: text_var,
                    ty: ir(Ty::Str),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(call("zb_any_str", vec![pending()], Ty::Str, span))),
                    span,
                }),
                Type::Unknown,
                span,
            ),
            eprint(str_lit("Traceback (most recent call last):", span)),
            TypedNode::new(
                TypedStatement::If(TypedIf {
                    condition: Box::new(call(
                        "zb_str_truthy",
                        vec![var(text_var, Ty::Str, span)],
                        Ty::Bool,
                        span,
                    )),
                    then_block: TypedBlock {
                        statements: vec![eprint(line)],
                        span,
                    },
                    else_block: Some(TypedBlock {
                        statements: vec![eprint(kind)],
                        span,
                    }),
                    span,
                }),
                Type::Unknown,
                span,
            ),
            TypedNode::new(
                TypedStatement::Expression(Box::new(call(
                    "zb_exit",
                    vec![int32_lit(1, span)],
                    Ty::None,
                    span,
                ))),
                Type::Unknown,
                span,
            ),
        ];
        let pending_now = self.pending(span);
        Ok(vec![
            one_pass(body, span),
            TypedNode::new(
                TypedStatement::If(TypedIf {
                    condition: Box::new(pending_now),
                    then_block: TypedBlock {
                        statements: report,
                        span,
                    },
                    else_block: None,
                    span,
                }),
                Type::Unknown,
                span,
            ),
        ])
    }

    /// `match gen.next() { Some(x) => { ... }, _ => { ... } }` as a
    /// statement, with `x` bound as `item` in the first arm.
    fn next_match(
        &mut self,
        gen: Node,
        item: InternedString,
        some: Vec<Stmt>,
        none: Vec<Stmt>,
        span: Span,
    ) -> Stmt {
        let next = method_call(gen, "next", vec![], Ty::Object, span);
        let next = Node {
            ty: Type::Optional(Box::new(Type::Any)),
            ..next
        };
        let some_pattern = TypedNode::new(
            TypedPattern::Constructor {
                constructor: Type::Unresolved(intern("Some")),
                pattern: Box::new(TypedNode::new(
                    TypedPattern::Identifier {
                        name: item,
                        mutability: Mutability::Immutable,
                    },
                    Type::Any,
                    span,
                )),
            },
            Type::Any,
            span,
        );
        let arm = |pattern: TypedNode<TypedPattern>, statements: Vec<Stmt>| TypedMatchArm {
            pattern: Box::new(pattern),
            guard: None,
            body: Box::new(node(
                TypedExpression::Block(TypedBlock { statements, span }),
                Ty::None,
                span,
            )),
        };
        TypedNode::new(
            TypedStatement::Match(TypedMatch {
                scrutinee: Box::new(next),
                arms: vec![
                    arm(some_pattern, some),
                    arm(
                        TypedNode::new(TypedPattern::Wildcard, Type::Any, span),
                        none,
                    ),
                ],
            }),
            Type::Unknown,
            span,
        )
    }

    /// `for target in <generator>`: pull until exhausted.
    fn for_generator(
        &mut self,
        f: &py::StmtFor,
        gen: Val,
        extra: Vec<Stmt>,
        span: Span,
    ) -> Result<TypedStatement> {
        let mut prologue = std::mem::take(&mut self.hoisted);
        let held = self.hold(gen, &mut prologue, span);
        let item = self.temp();
        let mut body = Vec::new();
        self.bind(
            &f.target,
            Val {
                node: var(item, Ty::Object, span),
                ty: Ty::Object,
            },
            span,
            &mut body,
        )?;
        self.in_loop(|this| -> Result<()> {
            for s in &f.body {
                this.stmt(s, &mut body)?;
            }
            Ok(())
        })?;
        body.extend(extra);
        let stop = vec![TypedNode::new(
            TypedStatement::Break(None),
            Type::Unknown,
            span,
        )];
        let pull = self.next_match(held.node, item, body, stop, span);
        prologue.push(one_pass(vec![pull], span));
        Ok(TypedStatement::Block(TypedBlock {
            statements: prologue,
            span,
        }))
    }

    /// Every value a generator yields, as a list.
    fn generator_to_list(&mut self, gen: Node, span: Span) -> Node {
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: gen,
                ty: Ty::Gen,
            },
            &mut pre,
            span,
        );
        let out = self.temp();
        pre.push(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: out,
                ty: ir(Ty::List(Elem::Object)),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(self.list_of(Vec::new(), Elem::Object, span))),
                span,
            }),
            Type::Unknown,
            span,
        ));
        let item = self.temp();
        let push = vec![TypedNode::new(
            TypedStatement::Expression(Box::new(method_call(
                var(out, Ty::List(Elem::Object), span),
                "push",
                vec![var(item, Ty::Object, span)],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        )];
        let stop = vec![TypedNode::new(
            TypedStatement::Break(None),
            Type::Unknown,
            span,
        )];
        let pull = self.next_match(held.node, item, push, stop, span);
        pre.push(one_pass(vec![pull], span));
        self.hoisted.extend(pre);
        var(out, Ty::List(Elem::Object), span)
    }

    /// `next(gen)`: the next value, or the default, or StopIteration.
    fn next_of(&mut self, gen: Node, default: Option<Node>, span: Span) -> Val {
        let mut pre = Vec::new();
        let result = self.temp();
        let initial = match &default {
            Some(d) => d.clone(),
            None => {
                let none = Val {
                    node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                    ty: Ty::None,
                };
                self.coerce(none, Ty::Object)
            }
        };
        pre.push(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: result,
                ty: ir(Ty::Object),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(initial)),
                span,
            }),
            Type::Unknown,
            span,
        ));
        let item = self.temp();
        let take = vec![TypedNode::new(
            TypedStatement::Expression(Box::new(binary(
                BinaryOp::Assign,
                var(result, Ty::Object, span),
                var(item, Ty::Object, span),
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        )];
        let mut exhausted = Vec::new();
        if default.is_none() {
            self.raise_named("StopIteration", str_lit("", span), span, &mut exhausted);
        }
        let pull = self.next_match(gen, item, take, exhausted, span);
        pre.push(pull);
        self.hoisted.extend(pre);
        Val {
            node: var(result, Ty::Object, span),
            ty: Ty::Object,
        }
    }

    /// `(e for x in it if c)`: a generator function of its own, made
    /// and started here.
    fn generator_expr(&mut self, g: &py::ExprGenerator, span: Span) -> Result<Val> {
        let scope = Scope::of_generator(g);
        let (captured, seeds) = self.captures_for(&scope);
        let lifted = self.lifted_name("genexpr");
        let sig = Sig {
            params: Vec::new(),
            ret: Ty::Gen,
            defaults: Vec::new(),
        };
        let mut child = Lowerer::new(
            self.module,
            &lifted,
            sig,
            Locals::default(),
            &scope,
            captured.clone(),
            seeds,
        );
        let mut body = child.generator_prologue(span);
        body.extend(child.cell_prologue(span));
        // The loop yields each element; the comprehension machinery
        // leaves it in the child's hoisted statements.
        child.comprehension(&g.generators, Produce::Yield(&g.elt), span)?;
        body.append(&mut child.hoisted);
        let function = TypedFunction {
            name: intern(&lifted),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params: Vec::new(),
            return_type: Type::Any,
            body: Some(TypedBlock {
                statements: body,
                span,
            }),
            visibility: Visibility::Public,
            is_async: false,
            is_fiber: true,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
            link_name: None,
            module: None,
        };
        self.module.lifted.borrow_mut().push(function);
        let sig = Sig {
            params: Vec::new(),
            ret: Ty::Gen,
            defaults: Vec::new(),
        };
        let cells = self.cells_of(&captured, span);
        Ok(self.start_generator(&lifted, &sig, Vec::new(), cells, span))
    }

    /// `let env = <the fiber's environment>` at the start of a body.
    fn generator_prologue(&mut self, span: Span) -> Vec<Stmt> {
        self.bound.push(intern("env"));
        vec![TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: intern("env"),
                ty: ir(Ty::List(Elem::Object)),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(call(
                    "zb_list_unbox_any",
                    vec![call("zb_fiber_env", vec![], Ty::Object, span)],
                    Ty::List(Elem::Object),
                    span,
                ))),
                span,
            }),
            Type::Unknown,
            span,
        )]
    }

    /// A paused fiber for `code`, its environment holding the cells
    /// and the arguments (already coerced to the parameter types).
    fn start_generator(
        &mut self,
        code: &str,
        sig: &Sig,
        args: Vec<Node>,
        cells: Vec<Val>,
        span: Span,
    ) -> Val {
        let none = || Val {
            node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
            ty: Ty::None,
        };
        let mut items = vec![none(), none()];
        items.extend(cells);
        for (a, (_, ty)) in args.into_iter().zip(&sig.params) {
            items.push(Val { node: a, ty: *ty });
        }
        let env = self.list_of(items, Elem::Object, span);
        let env = self.coerce(
            Val {
                node: env,
                ty: Ty::List(Elem::Object),
            },
            Ty::Object,
        );
        Val {
            node: call(
                "zb_fiber_start",
                vec![code_of(code, span), env],
                Ty::Gen,
                span,
            ),
            ty: Ty::Gen,
        }
    }

    /// `isinstance(v, T)` for a type name or a class.
    fn isinstance(&mut self, v: Val, ty_expr: &py::Expr, span: Span) -> Result<Val> {
        let py::Expr::Name(n) = ty_expr else {
            return unsupported("isinstance() with anything but a type name", ty_expr);
        };
        let lit = |b: bool| Val {
            node: node(
                TypedExpression::Literal(TypedLiteral::Bool(b)),
                Ty::Bool,
                span,
            ),
            ty: Ty::Bool,
        };
        if let Some(&k) = self.module.class_index.get(n.id.as_str()) {
            // The instance's class tag against `k` and every subclass.
            let tag = match v.ty {
                Ty::Class(c) => {
                    if !self.module.is_subclass(c as usize, k)
                        && !self.module.is_subclass(k, c as usize)
                    {
                        return Ok(lit(false));
                    }
                    node(
                        TypedExpression::Field(TypedFieldAccess {
                            object: Box::new(v.node),
                            field: intern("$class"),
                        }),
                        Ty::Int,
                        span,
                    )
                }
                Ty::Object => binary(
                    BinaryOp::Sub,
                    call("zb_any_kind", vec![v.node], Ty::Int, span),
                    int_lit(zyntax_builtins::INSTANCE_KIND_BASE, span),
                    Ty::Int,
                    span,
                ),
                _ => return Ok(lit(false)),
            };
            let mut pre = Vec::new();
            let held = self.hold(
                Val {
                    node: tag,
                    ty: Ty::Int,
                },
                &mut pre,
                span,
            );
            self.hoisted.extend(pre);
            let mut test: Option<Node> = None;
            for c in 0..self.module.classes.len() {
                if self.module.is_subclass(c, k) {
                    let this = binary(
                        BinaryOp::Eq,
                        held.node.clone(),
                        int_lit(c as i64, span),
                        Ty::Bool,
                        span,
                    );
                    test = Some(match test {
                        None => this,
                        Some(prev) => binary(BinaryOp::Or, prev, this, Ty::Bool, span),
                    });
                }
            }
            return Ok(Val {
                node: test.expect("a class is its own subclass"),
                ty: Ty::Bool,
            });
        }
        let name = n.id.as_str();
        let statically = |ty: Ty| -> Option<bool> {
            Some(match (name, ty) {
                ("int", Ty::Int | Ty::Bool) => true,
                ("bool", Ty::Bool) => true,
                ("float", Ty::Float) => true,
                ("str", Ty::Str) => true,
                ("list", Ty::List(_)) => true,
                ("tuple", Ty::Tuple) => true,
                ("dict", Ty::Dict) => true,
                ("set", Ty::Set) => true,
                (_, Ty::Object | Ty::Unknown) => return None,
                _ => false,
            })
        };
        if let Some(answer) = statically(v.ty) {
            return Ok(lit(answer));
        }
        let type_name = call("zb_any_type", vec![v.node], Ty::Str, span);
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: type_name,
                ty: Ty::Str,
            },
            &mut pre,
            span,
        );
        self.hoisted.extend(pre);
        let same = |t: &str| {
            call(
                "zb_str_eq",
                vec![held.node.clone(), str_lit(t, span)],
                Ty::Bool,
                span,
            )
        };
        let node = match name {
            // A bool is an int too.
            "int" => binary(BinaryOp::Or, same("int"), same("bool"), Ty::Bool, span),
            other => same(other),
        };
        Ok(Val { node, ty: Ty::Bool })
    }

    /// `obj.attr` read.
    fn attribute(&mut self, object: Val, attr: &str, span: Span) -> Result<Val> {
        match object.ty {
            Ty::Class(k) => {
                let Some((_, ty)) = self.module.field(k as usize, attr) else {
                    return Err(Error::Unsupported {
                        what: format!(
                            "attribute `{attr}` of {}, which has no such field",
                            self.module.classes[k as usize].name
                        ),
                        at: span.start,
                    });
                };
                let stored = field_storage(ty);
                let field = node(
                    TypedExpression::Field(TypedFieldAccess {
                        object: Box::new(object.node),
                        field: intern(attr),
                    }),
                    stored,
                    span,
                );
                let node = self.coerce(
                    Val {
                        node: field,
                        ty: stored,
                    },
                    ty,
                );
                Ok(Val { node, ty })
            }
            Ty::Object => {
                self.module.attr_reads.borrow_mut().insert(attr.to_string());
                let v = Val {
                    node: call(&getattr_name(attr), vec![object.node], Ty::Object, span),
                    ty: Ty::Object,
                };
                Ok(self.guard(v, span))
            }
            other => Err(Error::Unsupported {
                what: format!("attribute `{attr}` of a {other:?}"),
                at: span.start,
            }),
        }
    }

    /// `obj.attr = value` as a statement expression.
    fn set_attribute(&mut self, object: Val, attr: &str, value: Val, span: Span) -> Result<Node> {
        match object.ty {
            Ty::Class(k) => {
                let Some((_, ty)) = self.module.field(k as usize, attr) else {
                    return Err(Error::Unsupported {
                        what: format!(
                            "attribute `{attr}` of {}, which has no such field",
                            self.module.classes[k as usize].name
                        ),
                        at: span.start,
                    });
                };
                let stored = field_storage(ty);
                let value = self.coerce(value, ty);
                let value = self.coerce(Val { node: value, ty }, stored);
                let field = node(
                    TypedExpression::Field(TypedFieldAccess {
                        object: Box::new(object.node),
                        field: intern(attr),
                    }),
                    stored,
                    span,
                );
                Ok(binary(BinaryOp::Assign, field, value, Ty::None, span))
            }
            Ty::Object => {
                self.module
                    .attr_writes
                    .borrow_mut()
                    .insert(attr.to_string());
                let v = self.coerce(value, Ty::Object);
                Ok(call(
                    &setattr_name(attr),
                    vec![object.node, v],
                    Ty::None,
                    span,
                ))
            }
            other => Err(Error::Unsupported {
                what: format!("assignment to attribute `{attr}` of a {other:?}"),
                at: span.start,
            }),
        }
    }

    /// The call of `method` on an instance of class `k`, through the
    /// dispatcher when a subclass overrides it. `args` are already
    /// coerced to the parameter types.
    pub(crate) fn invoke(
        &mut self,
        k: usize,
        method: &str,
        receiver: Node,
        args: Vec<Node>,
        span: Span,
    ) -> Option<Val> {
        let owner = self.module.method_owner(k, method)?;
        let (sig, fn_name) = self.module.method_sig(k, method)?;
        let ret = sig.ret;
        let target = if self.module.overriders(owner, method).is_empty() {
            fn_name
        } else {
            dispatch_name(&fn_name)
        };
        let receiver = self.coerce(
            Val {
                node: receiver,
                ty: Ty::Class(k as u16),
            },
            Ty::Class(owner as u16),
        );
        let mut all = vec![receiver];
        all.extend(args);
        Some(Val {
            node: call(&target, all, ret, span),
            ty: ret,
        })
    }

    /// A dunder method call with already-boxed operands, when the class
    /// chain defines it.
    fn dunder(
        &mut self,
        k: usize,
        method: &str,
        receiver: Node,
        args: Vec<Node>,
        span: Span,
    ) -> Option<Val> {
        let (sig, _) = self.module.method_sig(k, method)?;
        if sig.params.len() != args.len() + 1 {
            return None;
        }
        let param_tys: Vec<Ty> = sig.params.iter().skip(1).map(|(_, t)| *t).collect();
        let args = args
            .into_iter()
            .zip(param_tys)
            .map(|(a, t)| {
                self.coerce(
                    Val {
                        node: a,
                        ty: Ty::Object,
                    },
                    t,
                )
            })
            .collect();
        self.invoke(k, method, receiver, args, span)
    }

    /// `obj.m(args)` on a receiver of known class.
    #[allow(clippy::too_many_arguments)]
    fn method_on(
        &mut self,
        k: usize,
        receiver: Val,
        method: &str,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        let Some((sig, _)) = self.module.method_sig(k, method) else {
            return Err(Error::Unsupported {
                what: format!(
                    "method `{method}` of {}, which defines none",
                    self.module.classes[k].name
                ),
                at: span.start,
            });
        };
        let sig = without_self(sig);
        let lowered = self.arguments(method, &sig, args, keywords, c)?;
        let v = self
            .invoke(k, method, receiver.node, lowered, span)
            .expect("the method was just found");
        Ok(self.guard(v, span))
    }

    /// `obj.m(args)` on a dynamic receiver: a dispatcher over every class
    /// with such a method, generated once per name and arity.
    fn dynamic_method(
        &mut self,
        receiver: Val,
        method: &str,
        args: &[py::Expr],
        span: Span,
    ) -> Result<Val> {
        let mut lowered = vec![receiver.node];
        for a in args {
            lowered.push(self.expr_as(a, Ty::Object)?);
        }
        self.module
            .dyn_methods
            .borrow_mut()
            .insert((method.to_string(), args.len()));
        let v = Val {
            node: call(&callm_name(method, args.len()), lowered, Ty::Object, span),
            ty: Ty::Object,
        };
        Ok(self.guard(v, span))
    }

    /// `C(args)`: allocate, then `__init__`.
    fn construct(
        &mut self,
        k: usize,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        at: &dyn Ranged,
        span: Span,
    ) -> Result<Val> {
        let name = self.module.classes[k].name.clone();
        let lowered = match self.module.method_sig(k, "__init__") {
            Some((sig, _)) => {
                let sig = without_self(sig);
                self.arguments(&name, &sig, args, keywords, at)?
            }
            None if args.is_empty() && keywords.is_empty() => Vec::new(),
            None => {
                return unsupported(format!("{name}() takes no arguments"), &at.range());
            }
        };
        let v = Val {
            node: call(&new_name(&name), lowered, Ty::Class(k as u16), span),
            ty: Ty::Class(k as u16),
        };
        Ok(self.guard(v, span))
    }

    /// `super().m(args)`: the base class's method, on `self`.
    fn super_call(
        &mut self,
        method: &str,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        let Some(k) = self.class else {
            return unsupported("super() outside a method", c);
        };
        let Some(base) = self.module.classes[k].base else {
            return unsupported(
                format!(
                    "super() in {}, which has no base class",
                    self.module.classes[k].name
                ),
                c,
            );
        };
        let Some(owner) = self.module.method_owner(base, method) else {
            return unsupported(format!("super().{method}, which no base class defines"), c);
        };
        let (sig, fn_name) = self
            .module
            .method_sig(owner, method)
            .expect("the owner defines it");
        let ret = sig.ret;
        let sig = without_self(sig);
        let lowered = self.arguments(method, &sig, args, keywords, c)?;
        let self_name = self.sig.params[0].0.clone();
        let receiver = Val {
            node: var(intern(&self_name), Ty::Class(k as u16), span),
            ty: Ty::Class(k as u16),
        };
        let receiver = self.coerce(receiver, Ty::Class(owner as u16));
        let mut all = vec![receiver];
        all.extend(lowered);
        let v = Val {
            node: call(&fn_name, all, ret, span),
            ty: ret,
        };
        Ok(self.guard(v, span))
    }

    /// A call through a function value: every argument boxed, the result
    /// dynamic.
    fn call_value(
        &mut self,
        callee: Val,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        if !keywords.is_empty() {
            return unsupported("keyword arguments in a call through a value", c);
        }
        if args.len() > zyntax_builtins::functions::MAX_CALL_ARITY {
            return unsupported(
                format!(
                    "a call through a value with more than {} arguments",
                    args.len()
                ),
                c,
            );
        }
        let mut lowered = vec![self.coerce(callee, Ty::Object)];
        for a in args {
            lowered.push(self.expr_as(a, Ty::Object)?);
        }
        let v = Val {
            node: call(
                &format!("zb_call_{}", args.len()),
                lowered,
                Ty::Object,
                span,
            ),
            ty: Ty::Object,
        };
        Ok(self.guard(v, span))
    }

    /// A function record: the code address, the arity and the cells.
    fn record(&mut self, code: &str, arity: usize, cells: Vec<Val>, span: Span) -> Val {
        let cells = self.list_of(cells, Elem::Object, span);
        Val {
            node: call(
                "zb_func_new",
                vec![code_of(code, span), int_lit(arity as i64, span), cells],
                Ty::Object,
                span,
            ),
            ty: Ty::Object,
        }
    }

    /// A module function as a value, through an adapter that unboxes
    /// the arguments and boxes the result.
    fn function_value(&mut self, name: &str, span: Span) -> Val {
        let sig = self.module.funcs[name].clone();
        self.module.adapters.borrow_mut().insert(name.to_string());
        self.record(&adapter_name(name), sig.params.len(), Vec::new(), span)
    }

    /// A `def` inside this function: lifted to a function of its own
    /// that takes the record, and bound here as a value.
    fn nested_def(&mut self, f: &py::StmtFunctionDef, span: Span) -> Result<Val> {
        if f.parameters.vararg.is_some() || f.parameters.kwarg.is_some() {
            return unsupported("*args / **kwargs", &*f.parameters);
        }
        let scope = Scope::of_function(f);
        let sig = types::declared_sig(f);
        let (captured, seeds) = self.captures_for(&scope);
        let locals = types::infer_locals_seeded(self.module, &sig, &f.body, &seeds);
        let lifted = self.lifted_name(f.name.as_str());
        let mut child = Lowerer::new(
            self.module,
            &lifted,
            sig,
            locals,
            &scope,
            captured.clone(),
            seeds,
        );
        let params: Vec<(String, Ty)> = child.sig.params.clone();
        let mut body = Vec::new();
        for s in &f.body {
            child.stmt(s, &mut body)?;
        }
        let function = child.lifted_function(&lifted, &params, body, span);
        self.module.lifted.borrow_mut().push(function);
        let cells = self.cells_of(&captured, span);
        Ok(self.record(&lifted, params.len(), cells, span))
    }

    /// A lambda: a lifted function returning its one expression.
    fn lambda(&mut self, l: &py::ExprLambda, span: Span) -> Result<Val> {
        let scope = Scope::of_lambda(l);
        let mut params: Vec<(String, Ty)> = Vec::new();
        if let Some(ps) = &l.parameters {
            if ps.vararg.is_some() || ps.kwarg.is_some() {
                return unsupported("*args / **kwargs", &**ps);
            }
            for p in ps.iter_non_variadic_params() {
                if p.default.is_some() {
                    return unsupported("a default in a lambda", p);
                }
                params.push((p.parameter.name.to_string(), Ty::Object));
            }
        }
        let sig = Sig {
            params: params.clone(),
            ret: Ty::Object,
            defaults: vec![None; params.len()],
        };
        let (captured, seeds) = self.captures_for(&scope);
        let mut locals = Locals::default();
        for (name, ty) in &params {
            locals.vars.insert(name.clone(), *ty);
        }
        let lifted = self.lifted_name("lambda");
        let mut child = Lowerer::new(
            self.module,
            &lifted,
            sig,
            locals,
            &scope,
            captured.clone(),
            seeds,
        );
        let value = child.expr_as(&l.body, Ty::Object)?;
        let mut body = std::mem::take(&mut child.hoisted);
        body.push(TypedNode::new(
            TypedStatement::Return(Some(Box::new(value))),
            Type::Unknown,
            span,
        ));
        let function = child.lifted_function(&lifted, &params, body, span);
        self.module.lifted.borrow_mut().push(function);
        let cells = self.cells_of(&captured, span);
        Ok(self.record(&lifted, params.len(), cells, span))
    }

    /// Which of this function's cells a nested body uses, in record
    /// order, with their types.
    fn captures_for(&self, scope: &Scope) -> (Vec<String>, HashMap<String, Ty>) {
        let captured: Vec<String> = self
            .cells
            .keys()
            .filter(|n| scope.free.contains(*n))
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let seeds = captured
            .iter()
            .map(|n| (n.clone(), self.var_ty(n)))
            .collect();
        (captured, seeds)
    }

    /// The cells named, boxed, for a record.
    fn cells_of(&mut self, captured: &[String], span: Span) -> Vec<Val> {
        captured
            .iter()
            .map(|n| Val {
                node: var(self.cells[n], Ty::List(Elem::Object), span),
                ty: Ty::List(Elem::Object),
            })
            .collect()
    }

    fn lifted_name(&self, inner: &str) -> String {
        let n = self.module.counter.get();
        self.module.counter.set(n + 1);
        format!("{}${inner}${n}", self.name)
    }

    /// This function in the shape every function value has: the record
    /// first, each argument a dynamic value unboxed to its declared type
    /// in a prologue, the result boxed.
    fn lifted_function(
        &mut self,
        name: &str,
        params: &[(String, Ty)],
        body: Vec<Stmt>,
        span: Span,
    ) -> TypedFunction {
        let mut typed_params = vec![parameter("env", Ty::List(Elem::Object), span)];
        let mut statements = Vec::new();
        for (i, (pname, declared)) in params.iter().enumerate() {
            let arg = format!("a{i}");
            typed_params.push(parameter(&arg, Ty::Object, span));
            let local = self.var_ty(pname);
            let value = self.coerce(
                Val {
                    node: var(intern(&arg), Ty::Object, span),
                    ty: Ty::Object,
                },
                *declared,
            );
            let value = self.coerce(
                Val {
                    node: value,
                    ty: *declared,
                },
                local,
            );
            statements.push(TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: intern(pname),
                    ty: ir(local),
                    mutability: Mutability::Mutable,
                    initializer: Some(Box::new(value)),
                    span,
                }),
                Type::Unknown,
                span,
            ));
        }
        statements.extend(self.cell_prologue(span));
        statements.extend(body);
        // Falling off the end returns None.
        let none = Val {
            node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
            ty: Ty::None,
        };
        let none = self.coerce(none, Ty::Object);
        statements.push(TypedNode::new(
            TypedStatement::Return(Some(Box::new(none))),
            Type::Unknown,
            span,
        ));
        TypedFunction {
            name: intern(name),
            annotations: Vec::new(),
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params: typed_params,
            return_type: ir(Ty::Object),
            body: Some(TypedBlock { statements, span }),
            visibility: Visibility::Public,
            is_async: false,
            is_fiber: false,
            is_pure: false,
            is_external: false,
            calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
            link_name: None,
            module: None,
        }
    }

    /// The arguments of a call to a module function, in parameter order:
    /// positionals first, then keywords by name, then defaults.
    fn arguments(
        &mut self,
        name: &str,
        sig: &Sig,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        at: &dyn Ranged,
    ) -> Result<Vec<Node>> {
        let mut slots: Vec<Option<&py::Expr>> = vec![None; sig.params.len()];
        if args.len() > sig.params.len() {
            return unsupported(
                format!(
                    "calling `{name}` with {} positional argument(s); it takes {}",
                    args.len(),
                    sig.params.len()
                ),
                &at.range(),
            );
        }
        for (slot, a) in slots.iter_mut().zip(args) {
            *slot = Some(a);
        }
        for k in keywords {
            let Some(arg) = &k.arg else {
                return unsupported("`**` argument unpacking", k);
            };
            let Some(i) = sig.params.iter().position(|(p, _)| p == arg.as_str()) else {
                return unsupported(
                    format!("calling `{name}` with an unexpected keyword `{arg}`"),
                    k,
                );
            };
            if slots[i].is_some() {
                return unsupported(format!("calling `{name}` with `{arg}` given twice"), k);
            }
            slots[i] = Some(&k.value);
        }
        let mut lowered = Vec::with_capacity(sig.params.len());
        for (i, ((pname, pty), slot)) in sig.params.iter().zip(&slots).enumerate() {
            let e = match (slot, sig.defaults.get(i).and_then(|d| d.as_ref())) {
                (Some(e), _) => *e,
                (None, Some(default)) => default,
                (None, None) => {
                    return unsupported(
                        format!("calling `{name}` without its argument `{pname}`"),
                        &at.range(),
                    )
                }
            };
            lowered.push(self.expr_as(e, *pty)?);
        }
        Ok(lowered)
    }

    /// `print(*values, sep=" ", end="\n")`.
    fn print(&mut self, args: &[py::Expr], keywords: &[py::Keyword], span: Span) -> Result<Val> {
        let mut sep = str_lit(" ", span);
        let mut end: Option<Node> = None;
        for k in keywords {
            match k.arg.as_ref().map(|a| a.as_str()) {
                Some("sep") => sep = self.expr_as(&k.value, Ty::Str)?,
                Some("end") => end = Some(self.expr_as(&k.value, Ty::Str)?),
                Some(other) => {
                    return unsupported(format!("print(..., {other}=...)"), k);
                }
                None => return unsupported("`**` argument unpacking", k),
            }
        }
        // Every argument is evaluated before any is converted, so a call
        // among them sees the others' side effects the way Python's
        // print does.
        let mut values = Vec::with_capacity(args.len());
        for a in args {
            let v = self.expr(a)?;
            let v = if args.len() > 1 {
                let mut pre = Vec::new();
                let held = self.hold(v, &mut pre, span);
                self.hoisted.extend(pre);
                held
            } else {
                v
            };
            values.push(v);
        }
        let mut line: Option<Node> = None;
        for v in values {
            let s = self.str_of(v);
            line = Some(match line {
                None => s,
                Some(prev) => {
                    let with_sep = binary(BinaryOp::Add, prev, sep.clone(), Ty::Str, span);
                    binary(BinaryOp::Add, with_sep, s, Ty::Str, span)
                }
            });
        }
        let line = line.unwrap_or_else(|| str_lit("", span));
        let node = match end {
            None => call("zb_print_line", vec![line], Ty::None, span),
            Some(end) => call(
                "zb_print_text",
                vec![binary(BinaryOp::Add, line, end, Ty::Str, span)],
                Ty::None,
                span,
            ),
        };
        Ok(Val { node, ty: Ty::None })
    }
}
