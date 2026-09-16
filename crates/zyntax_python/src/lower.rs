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
use crate::stdlib;
use crate::types::{self, Elem, Locals, Module, Sig, Ty, Typer};
use crate::{intern, prim, span_of, Error, Result};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use std::collections::{BTreeSet, HashMap};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    ParameterAttribute, TypedBinary, TypedBlock, TypedCall, TypedCast, TypedExpression,
    TypedFieldAccess, TypedFor, TypedFunction, TypedIf, TypedIfExpr, TypedIndex, TypedLet,
    TypedLiteral, TypedMatch, TypedMatchArm, TypedMethodCall, TypedParameter, TypedPattern,
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

thread_local! {
    /// The program's file the statement being lowered came from, as an
    /// index into its source files; every span made while it is set
    /// names that file.
    static CURRENT_FILE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

pub(crate) fn set_current_file(file: u32) {
    CURRENT_FILE.with(|c| c.set(file));
}

pub(crate) fn current_file() -> u32 {
    CURRENT_FILE.with(|c| c.get())
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

/// A field holds its own scalar, or the pointer to an instance; anything
/// else on the heap is stored boxed, so a list field is one shared
/// header and a class's layout never depends on another's.
pub(crate) fn field_storage(ty: Ty) -> Ty {
    match ty {
        Ty::Int | Ty::Float | Ty::Bool | Ty::Str | Ty::Class(_) => ty,
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
        Ty::List(e) => list_type(elem_ir(e)),
        Ty::Tuple | Ty::Dict | Ty::Set => list_type(Type::Any),
        Ty::Class(k) => class_type(k as usize),
        Ty::Gen => Type::Fiber(Box::new(Type::Any)),
        // A known function value is still the record every function
        // value is.
        Ty::Closure(_) | Ty::Bound(_) | Ty::Builtin(_) | Ty::Object | Ty::Unknown => Type::Any,
    }
}

/// The IR type a list of `e` holds per element: an instance is held by
/// address, everything else as itself.
pub(crate) fn elem_ir(e: Elem) -> Type {
    match e {
        Elem::Class(_) => addr_type(),
        other => ir(other.ty()),
    }
}

/// A call to a list function that returns an element of kind `e`,
/// typed as the element: an address comes back as the instance.
fn elem_call(op: &str, e: Elem, args: Vec<Node>, span: Span) -> Node {
    match e {
        Elem::Class(k) => cast(addr_call(&list_fn(op, e), args, span), Ty::Class(k), span),
        other => call(&list_fn(op, other), args, other.ty(), span),
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

/// Element `i` of a list the lowering built itself and so knows the
/// length of: a cell, a record. Nothing checks the index.
fn slot(list: Node, i: usize, ty: Ty, span: Span) -> Node {
    node(
        TypedExpression::Index(TypedIndex {
            object: Box::new(list),
            index: Box::new(int_lit(i as i64, span)),
        }),
        ty,
        span,
    )
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

/// Something a test can settle the nullness of: a variable, or a field
/// of a variable.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Place {
    Var(InternedString),
    Field(InternedString, String),
}

/// A lowered expression and the static type it has.
#[derive(Clone)]
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
/// A heap value is shared between caller and callee: the callee may keep
/// it, so the caller does not release it, and the same value may arrive
/// twice in one call.
fn ownership_of(ty: Ty) -> ParamOwnership {
    match ty {
        Ty::Int | Ty::Float | Ty::Bool | Ty::None => ParamOwnership::Copied,
        _ => ParamOwnership::Shared,
    }
}

/// Whether a generator expression is the start of a fresh generator,
/// which whoever drains it owns, rather than a name for one held
/// elsewhere.
fn is_generator_start(gen: &Node) -> bool {
    matches!(
        &gen.node,
        TypedExpression::Call(c)
            if matches!(&c.callee.node, TypedExpression::Variable(v)
                if v.resolve_global().as_deref() == Some("zb_fiber_start"))
    )
}

/// `zb_fiber_free(gen)`: a drained generator's fiber released.
fn free_generator(gen: Node, span: Span) -> Stmt {
    TypedNode::new(
        TypedStatement::Expression(Box::new(call("zb_fiber_free", vec![gen], Ty::None, span))),
        Type::Unknown,
        span,
    )
}

/// `zb_release_caught(caught)`: the exception's instance and box freed.
fn release_caught(caught: InternedString, span: Span) -> Stmt {
    TypedNode::new(
        TypedStatement::Expression(Box::new(call(
            "zb_release_caught",
            vec![var(caught, Ty::Object, span)],
            Ty::None,
            span,
        ))),
        Type::Unknown,
        span,
    )
}

/// Whether a body's end can be reached: it neither leaves the function
/// on every path nor ends by leaving a loop.
fn falls_through(body: &[py::Stmt]) -> bool {
    !types::terminates(body)
        && !matches!(
            body.last(),
            Some(py::Stmt::Break(_) | py::Stmt::Continue(_))
        )
}

fn parameter(name: &str, ty: Ty, span: Span) -> TypedParameter {
    TypedParameter {
        name: intern(name),
        ty: ir(ty),
        mutability: Mutability::Mutable,
        kind: ParameterKind::Regular,
        default_value: None,
        attributes: dynamic_attribute(ty, span),
        ownership: ownership_of(ty),
        span,
    }
}

/// The `dynamic` attribute on a parameter Python types dynamically: an
/// unannotated parameter is dynamic by the language's rules, not by
/// omission, and the lowering does not warn about it.
pub(crate) fn dynamic_attribute(ty: Ty, span: Span) -> Vec<ParameterAttribute> {
    if matches!(
        ty,
        Ty::Object | Ty::Closure(_) | Ty::Bound(_) | Ty::Builtin(_)
    ) {
        vec![ParameterAttribute {
            name: intern("dynamic"),
            args: Vec::new(),
            span,
        }]
    } else {
        Vec::new()
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
    lowerer.guards = false;
    let mut params = vec![parameter("env", Ty::List(Elem::Object), span)];
    let mut args = Vec::new();
    let first_default = sig.params.len() - sig.defaults.iter().flatten().count();
    for (i, (_, ty)) in sig.params.iter().enumerate() {
        let arg = format!("a{i}");
        params.push(parameter(&arg, Ty::Object, span));
        let given = var(intern(&arg), Ty::Object, span);
        // An argument left out takes the default kept in the record.
        let given = if i >= first_default {
            lowerer.or_default(given, RECORD_CELLS_AT + (i - first_default), span)
        } else {
            given
        };
        args.push(lowerer.coerce(
            Val {
                node: given,
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
    Err(Error::unsupported(what.into(), at))
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
    pub(crate) hoisted: Vec<Stmt>,
    /// Class-typed variables known not to be None where the lowering
    /// stands: assigned from a constructor, or checked since. Cleared
    /// at every compound statement, so it never crosses a branch or a
    /// loop back edge.
    nonnull: std::collections::HashSet<InternedString>,
    /// Variables that hold an instance for the whole function: assigned
    /// a constructor's result before anything reads them, and assigned
    /// nothing else anywhere; see [`Self::always_instances`].
    always_instance: std::collections::HashSet<InternedString>,
    /// Fields `v.f` of a known instance `v` known not to be None where
    /// the lowering stands, from a test the control flow has settled.
    /// Cleared with `nonnull`, at any call (which may store to the
    /// field), and at a store to a field of that name.
    nonnull_fields: std::collections::HashSet<(InternedString, String)>,
    /// Whether this is the variant of the function that takes its
    /// instance-typed parameters to be instances; see
    /// [`types::trusted_name`].
    pub(crate) trusted: bool,
    /// How many cells the record of this lifted function holds before
    /// the defaults of its parameters.
    defaults_after_cells: usize,
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
    /// Whether this function raises anywhere itself: a `raise`, or a
    /// check after anything but a call to a function of the program.
    /// Those calls are listed in `raise_callees` instead, and the
    /// function raises only if one of them does; see
    /// [`types::RaiseFact`].
    pub(crate) may_raise_own: bool,
    pub(crate) raise_callees: BTreeSet<String>,
    /// The variable holding the exception being handled, for a bare
    /// `raise`.
    caught: Option<InternedString>,
    /// The `try` bodies being lowered, innermost last. A `return`,
    /// `break` or `continue` inside one records what it wants in the
    /// body's flag and leaves the body; the flag is acted on after
    /// `finally`.
    try_ctls: Vec<TryCtl>,
    /// The innermost loop's `else` flag, if it has an `else` suite.
    loop_elses: Vec<Option<InternedString>>,
    /// The `else` flag of the `for` about to be lowered, which its body
    /// takes as it enters the loop.
    for_else: Option<InternedString>,
    /// The `except` clauses being lowered that release their exception
    /// on the way out, innermost last.
    handler_ctls: Vec<HandlerCtl>,
    /// Whether the statement being lowered recorded such a flag, so a
    /// loop containing it leaves again once the loop is left.
    redirected: bool,
    /// Whether this function yields: it lowers to a fiber of dynamic
    /// values, and `return` ends it.
    is_generator: bool,
    /// Whether a checked read this lowerer emits is followed by its
    /// pending check. A lowerer building a function by hand drains no
    /// hoisted statements; its callers check after calling it.
    pub(crate) guards: bool,
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

/// One `except` clause that keeps nothing of its exception, which is
/// released wherever control leaves the clause.
struct HandlerCtl {
    /// The variable holding the caught exception.
    caught: InternedString,
    /// How many loops of the clause's own the statement being lowered
    /// is inside, a `try` body's single pass counted as one: an exit
    /// from a loop at depth 0 leaves the clause.
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
            nonnull: std::collections::HashSet::new(),
            always_instance: std::collections::HashSet::new(),
            nonnull_fields: std::collections::HashSet::new(),
            trusted: false,
            defaults_after_cells: 0,
            guards: true,
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
            may_raise_own: false,
            raise_callees: BTreeSet::new(),
            caught: None,
            try_ctls: Vec::new(),
            loop_elses: Vec::new(),
            for_else: None,
            handler_ctls: Vec::new(),
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
            // The body's pass is left; the return itself happens after
            // the `try`, where the clauses outside the body are left.
            self.release_left_handlers(false, span, out);
            self.set_flag_and_leave(flag, 1, span, out);
            return;
        }
        // The value is computed while the exception is still there.
        let value = match value {
            Some(v) if !self.handler_ctls.is_empty() => Some(
                self.hold(
                    Val {
                        node: v,
                        ty: self.sig.ret,
                    },
                    out,
                    span,
                )
                .node,
            ),
            other => other,
        };
        self.release_left_handlers(true, span, out);
        out.push(TypedNode::new(
            TypedStatement::Return(value.map(Box::new)),
            Type::Unknown,
            span,
        ));
    }

    /// Release the exceptions of the `except` clauses an exit leaves:
    /// every one for a return, else those the loop being left encloses.
    fn release_left_handlers(&mut self, all: bool, span: Span, out: &mut Vec<Stmt>) {
        for ctl in &self.handler_ctls {
            if all || ctl.loop_depth == 0 {
                out.push(release_caught(ctl.caught, span));
            }
        }
    }

    /// `break` or `continue` (`code` 2 or 3), or the `try` body's way of
    /// recording one when the loop it means is outside the body.
    fn emit_loop_exit(&mut self, code: i64, span: Span, out: &mut Vec<Stmt>) {
        self.release_left_handlers(false, span, out);
        if let Some(ctl) = self.try_ctls.last() {
            if ctl.loop_depth == 0 {
                let flag = ctl.flag;
                self.set_flag_and_leave(flag, code, span, out);
                return;
            }
        }
        if code == 2 {
            if let Some(Some(flag)) = self.loop_elses.last() {
                out.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(binary(
                        BinaryOp::Assign,
                        var(*flag, Ty::Bool, span),
                        node(
                            TypedExpression::Literal(TypedLiteral::Bool(false)),
                            Ty::Bool,
                            span,
                        ),
                        Ty::None,
                        span,
                    ))),
                    Type::Unknown,
                    span,
                ));
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
        let flag = self.for_else.take();
        self.in_loop_else(flag, f)
    }

    fn in_loop_else<T>(
        &mut self,
        else_flag: Option<InternedString>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.loop_elses.push(else_flag);
        if let Some(ctl) = self.try_ctls.last_mut() {
            ctl.loop_depth += 1;
        }
        let r = self.in_handler_loop(f);
        if let Some(ctl) = self.try_ctls.last_mut() {
            ctl.loop_depth -= 1;
        }
        self.loop_elses.pop();
        r
    }

    /// Lower a loop body, or a `try` body's single pass, counting it
    /// for every `except` clause it sits in.
    fn in_handler_loop<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        for ctl in &mut self.handler_ctls {
            ctl.loop_depth += 1;
        }
        let r = f(self);
        for ctl in &mut self.handler_ctls {
            ctl.loop_depth -= 1;
        }
        r
    }

    /// Whether an exception is pending.
    fn pending(&self, span: Span) -> Node {
        // No exception is the null dynamic value, so the check is a
        // load and a compare.
        binary(
            BinaryOp::Ne,
            var(intern(PENDING), Ty::Object, span),
            node(
                TypedExpression::Literal(TypedLiteral::Null),
                Ty::Object,
                span,
            ),
            Ty::Bool,
            span,
        )
    }

    /// `py$exc = value`.
    fn set_pending(&mut self, value: Node, span: Span) -> Stmt {
        self.may_raise_own = true;
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

    /// Leave, the way the current context leaves, releasing the
    /// exceptions of the `except` clauses left on the way.
    fn escape_into(&mut self, span: Span, out: &mut Vec<Stmt>) {
        let how = self.escapes.last().copied().unwrap_or(Escape::Return);
        self.release_left_handlers(how == Escape::Return, span, out);
        let st = match how {
            Escape::Break => TypedStatement::Break(None),
            Escape::Return => TypedStatement::Return(self.placeholder(span).map(Box::new)),
        };
        out.push(TypedNode::new(st, Type::Unknown, span));
    }

    /// The value a function returns when it leaves with an exception
    /// pending, which its caller never reads.
    fn placeholder(&mut self, span: Span) -> Option<Node> {
        if self.is_generator {
            return None;
        }
        if self.sig.ret == Ty::None {
            return None;
        }
        Some(self.zero_of(self.sig.ret, span))
    }

    /// A value of `ty` that stands for nothing: what a variable holds
    /// before the branch that assigns it runs.
    fn zero_of(&mut self, ty: Ty, span: Span) -> Node {
        match ty {
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
            Ty::Object | Ty::Unknown | Ty::Closure(_) | Ty::Bound(_) | Ty::Builtin(_) => {
                let none = Val {
                    node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                    ty: Ty::None,
                };
                self.coerce(none, Ty::Object)
            }
            other => cast(int_lit(0, span), other, span),
        }
    }

    /// A value one of two branches computes, when a branch has statements
    /// of its own to run first. Those cannot sit in a block expression (a
    /// loop there is not lowered), so the choice is an `if` statement
    /// assigning a temp, appended to `out`, and the value is the temp.
    fn conditional_value(
        &mut self,
        condition: Node,
        then: (Vec<Stmt>, Node),
        otherwise: (Vec<Stmt>, Node),
        ty: Ty,
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Node {
        let t = self.temp();
        let initial = self.zero_of(ty, span);
        out.push(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: t,
                ty: ir(ty),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(initial)),
                span,
            }),
            Type::Unknown,
            span,
        ));
        let assign = |value: Node| {
            TypedNode::new(
                TypedStatement::Expression(Box::new(binary(
                    BinaryOp::Assign,
                    var(t, ty, span),
                    value,
                    Ty::None,
                    span,
                ))),
                Type::Unknown,
                span,
            )
        };
        let (mut then_statements, then_value) = then;
        then_statements.push(assign(then_value));
        let (mut else_statements, else_value) = otherwise;
        else_statements.push(assign(else_value));
        out.push(TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(condition),
                then_block: TypedBlock {
                    statements: then_statements,
                    span,
                },
                else_block: Some(TypedBlock {
                    statements: else_statements,
                    span,
                }),
                span,
            }),
            Type::Unknown,
            span,
        ));
        var(t, ty, span)
    }

    /// `if <pending> { <escape> }`.
    fn pending_check(&mut self, span: Span) -> Stmt {
        self.raised = true;
        self.may_raise_own = true;
        let cond = self.pending(span);
        let mut leave = Vec::new();
        self.escape_into(span, &mut leave);
        TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(cond),
                then_block: TypedBlock {
                    statements: leave,
                    span,
                },
                else_block: None,
                span,
            }),
            Type::Unknown,
            span,
        )
    }

    /// [`Self::guard`] for a call to function `name` of the program: no
    /// check when the function is known not to raise, and otherwise a
    /// check attributed to the callee rather than to this function.
    fn guard_named(&mut self, v: Val, name: &str, span: Span) -> Val {
        if self.module.non_raising.contains(name) {
            return v;
        }
        self.raise_callees.insert(name.to_string());
        let own = self.may_raise_own;
        let held = self.guard(v, span);
        self.may_raise_own = own;
        held
    }

    /// What this function's lowering found about its raising, for the
    /// module-wide fixed point.
    pub(crate) fn raise_fact(&self) -> types::RaiseFact {
        types::RaiseFact {
            own: self.may_raise_own,
            callees: self.raise_callees.clone(),
        }
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
        if !self.module.class_index.contains_key(class) {
            return;
        }
        // The instance is built and left pending by a cold function, so
        // the path here is an error path to everything downstream: not
        // inlined into, and compiled only if it runs.
        self.module.raisers.borrow_mut().insert(class.to_string());
        out.push(TypedNode::new(
            TypedStatement::Expression(Box::new(call(
                &types::raiser_name(class),
                vec![message],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        ));
        self.may_raise_own = true;
        self.raised = true;
        self.escape_into(span, out);
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

    /// An argument a builtin consumes whole. A generator expression
    /// written there is never seen as a generator: it is built as the
    /// list of its elements, typed by them, since a fiber and a box
    /// per element buy nothing for a value read once and thrown away.
    /// Anything else is lowered as it is.
    fn consumed(&mut self, e: &py::Expr, span: Span) -> Result<Val> {
        if let py::Expr::Generator(g) = e {
            let elem = self.typer().comprehension_elem(&g.generators, &g.elt);
            return self.comprehension(&g.generators, Produce::List(elem, &g.elt), span);
        }
        self.expr(e)
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
                let element = slot(
                    var(intern("env"), Ty::List(Elem::Object), span),
                    RECORD_CELLS_AT + i,
                    Ty::Object,
                    span,
                );
                call(
                    "zb_unbox_list_raw_any",
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
        let element = slot(cell, 0, Ty::Object, span);
        let node = self.trusted(
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
    /// How a module variable of type `ty` is stored: primitives,
    /// strings and instances as themselves, the rest boxed.
    pub(crate) fn storage(ty: Ty) -> Ty {
        match ty {
            Ty::Int | Ty::Float | Ty::Bool | Ty::Str | Ty::Class(_) => ty,
            _ => Ty::Object,
        }
    }

    /// Read the module variable `name`.
    fn global_read(&mut self, name: &str, span: Span) -> Val {
        let ty = self.var_ty(name);
        let stored = Self::storage(ty);
        let node = self.trusted(
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
                let value = self.trusted(
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
                attributes: dynamic_attribute(declared, span_of(p)),
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
        self.always_instance = self.always_instances(&f.body);
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

    // ─── Conversions ────────────────────────────────────────────────

    /// `v` as a value of type `target`, converting where the two differ.
    pub(crate) fn coerce(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        match (v.ty, target) {
            (a, b) if a == b => v.node,
            (_, Ty::Unknown) => v.node,
            // A known function value is a dynamic value already.
            (Ty::Closure(_), Ty::Object | Ty::Closure(_)) | (Ty::Object, Ty::Closure(_)) => v.node,
            (Ty::Bound(_), Ty::Object | Ty::Bound(_)) | (Ty::Object, Ty::Bound(_)) => v.node,
            (Ty::Builtin(_), Ty::Object | Ty::Builtin(_)) | (Ty::Object, Ty::Builtin(_)) => v.node,
            (Ty::Int | Ty::Bool, Ty::Float) => cast(v.node, Ty::Float, span),
            (Ty::Bool, Ty::Int) => cast(v.node, Ty::Int, span),
            (Ty::Int, Ty::Bool) => binary(BinaryOp::Ne, v.node, int_lit(0, span), Ty::Bool, span),
            // A list is boxed by reference under a tag of its kind, and
            // read back by checking that tag.
            (Ty::List(e), Ty::Object) => call(&list_fn("box", e), vec![v.node], Ty::Object, span),
            (Ty::Tuple, Ty::Object) => call("zb_box_tuple", vec![v.node], Ty::Object, span),
            (Ty::Dict, Ty::Object) => call("zb_dict_box", vec![v.node], Ty::Object, span),
            (Ty::Set, Ty::Object) => call("zb_set_box", vec![v.node], Ty::Object, span),
            // A list read out of a box is checked like a primitive: a box
            // of anything else is a TypeError raised where it is used.
            (Ty::Object, Ty::List(e)) => {
                let checked = Val {
                    node: call(&list_fn("unbox", e), vec![v.node], target, span),
                    ty: target,
                };
                if self.guards {
                    self.guard(checked, span).node
                } else {
                    checked.node
                }
            }
            (Ty::Object, Ty::Tuple) => call("zb_unbox_tuple", vec![v.node], Ty::Tuple, span),
            // A primitive read out of a box is checked: a box of another
            // type is a TypeError, raised where the value is used.
            (Ty::Object, Ty::Int | Ty::Float | Ty::Str | Ty::Bool) => {
                let read = match target {
                    Ty::Int => "zb_any_as_i64",
                    Ty::Float => "zb_any_as_f64",
                    Ty::Str => "zb_any_as_str",
                    _ => "zb_any_as_bool",
                };
                let checked = Val {
                    node: call(read, vec![v.node], target, span),
                    ty: target,
                };
                if self.guards {
                    self.guard(checked, span).node
                } else {
                    checked.node
                }
            }
            (Ty::Object, Ty::Dict) => call("zb_dict_unbox", vec![v.node], Ty::Dict, span),
            (Ty::Object, Ty::Set) => call("zb_set_unbox", vec![v.node], Ty::Set, span),
            // Evaluate a None-producing expression before representing its result.
            (Ty::None, Ty::Class(_) | Ty::Object) => {
                let result = if target == Ty::Object {
                    node(TypedExpression::Literal(TypedLiteral::Null), target, span)
                } else {
                    cast(int_lit(0, span), target, span)
                };
                Self::block_value(
                    vec![TypedNode::new(
                        TypedStatement::Expression(Box::new(v.node)),
                        Type::Unknown,
                        span,
                    )],
                    result,
                    target,
                    span,
                )
            }
            // An instance is boxed as its address under the class tag, and
            // read back with a check; a subclass instance is its base. A
            // null instance boxes as None.
            (Ty::Class(k), Ty::Object) => {
                let address = as_addr(v.node, span);
                call(
                    "zb_box_instance",
                    vec![
                        address,
                        int32_lit(zyntax_builtins::instance_tag(k as usize) as i32, span),
                    ],
                    Ty::Object,
                    span,
                )
            }
            // A value of another class is a TypeError, raised where the
            // instance is wanted.
            (Ty::Object, Ty::Class(k)) => {
                let address = addr_call(
                    &format!("{}$unbox", self.module.classes[k as usize].name),
                    vec![v.node],
                    span,
                );
                let checked = Val {
                    node: cast(address, target, span),
                    ty: target,
                };
                if self.guards {
                    self.guard(checked, span).node
                } else {
                    checked.node
                }
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
            // Lists of dynamic values into lists of one kind: each element
            // read back checked. Between two kinds, through the dynamic
            // list.
            (Ty::List(from), Ty::List(to)) => {
                let anys = if from == Elem::Object {
                    v.node
                } else {
                    call(
                        &list_fn("to_any", from),
                        vec![v.node],
                        Ty::List(Elem::Object),
                        span,
                    )
                };
                let mut args = vec![anys];
                if let Elem::Class(k) = to {
                    args.push(int32_lit(
                        zyntax_builtins::instance_tag(k as usize) as i32,
                        span,
                    ));
                }
                let converted = Val {
                    node: call(&list_fn("from_any", to), args, target, span),
                    ty: target,
                };
                if self.guards {
                    self.guard(converted, span).node
                } else {
                    converted.node
                }
            }
            (Ty::Tuple, Ty::List(Elem::Object)) => Node {
                ty: ir(target),
                ..v.node
            },
            // A string's box holds a copy of it, released with the box.
            (Ty::Str, Ty::Object) => call("zb_box_str", vec![v.node], Ty::Object, span),
            // Into the dynamic world: a box. Out of it: a checked read.
            (_, Ty::Object) => cast(v.node, Ty::Object, span),
            (Ty::Object, _) => cast(v.node, target, span),
            // Two types with no conversion between them (a float where an
            // int is needed, a string where a float is). Python does not
            // convert either; the value goes through a box and the read
            // back is the TypeError.
            (_, _) => {
                let boxed = self.coerce(v, Ty::Object);
                self.coerce(
                    Val {
                        node: boxed,
                        ty: Ty::Object,
                    },
                    target,
                )
            }
        }
    }

    /// A value the lowering itself stored, read back as the type it was
    /// stored with: a cell's content, a global's box, an argument in a
    /// generator's environment. Nothing checks it, since nothing else
    /// writes there.
    pub(crate) fn trusted(&mut self, v: Val, target: Ty) -> Node {
        let span = v.node.span;
        let read = match (v.ty, target) {
            (Ty::Object, Ty::Int) => "zb_box_payload_i64",
            (Ty::Object, Ty::Float) => "zb_box_payload_f64",
            (Ty::Object, Ty::Bool) => "zb_box_payload_truth",
            (Ty::Object, Ty::Str) => "zb_box_get_str",
            // The box is known to hold an instance of the class.
            (Ty::Object, Ty::Class(_)) => {
                let address = addr_call("zb_unbox_instance_raw", vec![v.node], span);
                return cast(address, target, span);
            }
            _ => return self.coerce(v, target),
        };
        call(read, vec![v.node], target, span)
    }

    fn expr_as(&mut self, e: &py::Expr, target: Ty) -> Result<Node> {
        let v = self.expr(e)?;
        Ok(self.coerce(v, target))
    }

    /// A value as an element of a list of kind `e`: an instance goes in
    /// by address.
    fn elem_arg(&mut self, v: Val, e: Elem) -> Node {
        let span = v.node.span;
        let node = self.coerce(v, e.ty());
        match e {
            Elem::Class(_) => as_addr(node, span),
            _ => node,
        }
    }

    fn expr_as_elem(&mut self, expr: &py::Expr, e: Elem) -> Result<Node> {
        let v = self.expr(expr)?;
        Ok(self.elem_arg(v, e))
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
            Ty::Gen | Ty::Closure(_) | Ty::Bound(_) | Ty::Builtin(_) => node(
                TypedExpression::Literal(TypedLiteral::Bool(true)),
                Ty::Bool,
                span,
            ),
            // None is false; an instance is true unless its class says.
            Ty::Class(k) => {
                let k = k as usize;
                let has_dunder = self.module.method_sig(k, "__bool__").is_some()
                    || self.module.method_sig(k, "__len__").is_some();
                if !has_dunder {
                    return binary(
                        BinaryOp::Ne,
                        as_addr(v.node, span),
                        int_lit(0, span),
                        Ty::Bool,
                        span,
                    );
                }
                let no = node(
                    TypedExpression::Literal(TypedLiteral::Bool(false)),
                    Ty::Bool,
                    span,
                );
                self.unless_null(v, no, Ty::Bool, |this, held| {
                    if let Some(r) = this.dunder(k, "__bool__", held.node.clone(), vec![], span) {
                        this.truthy(r)
                    } else if let Some(r) = this.dunder(k, "__len__", held.node, vec![], span) {
                        this.truthy(r)
                    } else {
                        node(
                            TypedExpression::Literal(TypedLiteral::Bool(true)),
                            Ty::Bool,
                            span,
                        )
                    }
                })
            }
            Ty::List(_) | Ty::Tuple | Ty::Set => binary(
                BinaryOp::Ne,
                method_call(v.node, "len", vec![], Ty::Int, span),
                int_lit(0, span),
                Ty::Bool,
                span,
            ),
            Ty::None => Self::after_none(
                v.node,
                node(
                    TypedExpression::Literal(TypedLiteral::Bool(false)),
                    Ty::Bool,
                    span,
                ),
                Ty::Bool,
            ),
            Ty::Object | Ty::Unknown => call("zb_any_truthy", vec![v.node], Ty::Bool, span),
        }
    }

    /// `result`, after evaluating a None-typed expression whose value is
    /// known but whose effects are not: a call of a method that returns
    /// nothing still runs.
    fn after_none(none: Node, result: Node, ty: Ty) -> Node {
        if matches!(
            none.node,
            TypedExpression::Literal(_) | TypedExpression::Variable(_)
        ) {
            return result;
        }
        let span = none.span;
        Self::block_value(
            vec![TypedNode::new(
                TypedStatement::Expression(Box::new(none)),
                Type::Unknown,
                span,
            )],
            result,
            ty,
            span,
        )
    }

    /// `str(v)`.
    pub(crate) fn str_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Int => call("zb_str_of_int", vec![v.node], Ty::Str, span),
            Ty::Float => call("zb_float_repr", vec![v.node], Ty::Str, span),
            Ty::Bool => call("zb_bool_repr", vec![v.node], Ty::Str, span),
            Ty::Str => v.node,
            Ty::None => {
                Self::after_none(v.node, call("zb_none_repr", vec![], Ty::Str, span), Ty::Str)
            }
            Ty::List(e) => call(&list_fn("repr", e), vec![v.node], Ty::Str, span),
            Ty::Tuple => call("zb_tuple_repr", vec![v.node], Ty::Str, span),
            Ty::Dict => call("zb_dict_repr", vec![v.node], Ty::Str, span),
            Ty::Set => call("zb_set_repr", vec![v.node], Ty::Str, span),
            Ty::Gen => str_lit("<generator object>", span),
            Ty::Closure(_) => str_lit("<function>", span),
            Ty::Bound(_) => str_lit("<bound method>", span),
            Ty::Builtin(k) => str_lit(
                &format!("<built-in function {}>", types::BUILTIN_VALUES[k as usize]),
                span,
            ),
            Ty::Class(k) => {
                let k = k as usize;
                let none = str_lit(crate::policy::POLICY.none_text, span);
                self.unless_null(v, none, Ty::Str, |this, held| {
                    match this
                        .dunder(k, "__str__", held.node.clone(), vec![], span)
                        .or_else(|| this.dunder(k, "__repr__", held.node, vec![], span))
                    {
                        Some(r) => this.coerce(r, Ty::Str),
                        None => str_lit(&format!("<{} object>", this.module.classes[k].name), span),
                    }
                })
            }
            Ty::Object | Ty::Unknown => call("zb_any_str", vec![v.node], Ty::Str, span),
        }
    }

    /// `repr(v)`.
    pub(crate) fn repr_of(&mut self, v: Val) -> Node {
        let span = v.node.span;
        match v.ty {
            Ty::Str => call("zb_str_repr", vec![v.node], Ty::Str, span),
            Ty::Class(k) => {
                let k = k as usize;
                if self.module.method_sig(k, "__repr__").is_none() {
                    return self.str_of(v);
                }
                let none = str_lit(crate::policy::POLICY.none_text, span);
                self.unless_null(v, none, Ty::Str, |this, held| {
                    match this.dunder(k, "__repr__", held.node.clone(), vec![], span) {
                        Some(r) => this.coerce(r, Ty::Str),
                        None => this.str_of(held),
                    }
                })
            }
            Ty::Object | Ty::Unknown => call("zb_any_repr", vec![v.node], Ty::Str, span),
            _ => self.str_of(v),
        }
    }

    /// A list literal of `elem` kind from already lowered elements.
    pub(crate) fn list_of(&mut self, items: Vec<Val>, elem: Elem, span: Span) -> Node {
        let items = items.into_iter().map(|v| self.elem_arg(v, elem)).collect();
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
        // What is known about instances holds within one straight run of
        // statements: a branch or a loop may reach here from elsewhere.
        let compound = matches!(
            s,
            py::Stmt::If(_)
                | py::Stmt::While(_)
                | py::Stmt::For(_)
                | py::Stmt::Try(_)
                | py::Stmt::With(_)
                | py::Stmt::Match(_)
                | py::Stmt::FunctionDef(_)
                | py::Stmt::ClassDef(_)
        );
        if compound {
            self.nonnull.clear();
            self.nonnull_fields.clear();
        }
        let mut own = Vec::new();
        self.stmt_into(s, &mut own)?;
        if compound {
            self.nonnull.clear();
            self.nonnull_fields.clear();
            // `if x is None: return` settles `x` for what follows: the
            // only way past is the branch the test failed in.
            if let py::Stmt::If(i) = s {
                let leaves = |body: &[py::Stmt]| types::terminates(body);
                if let Some((place, holds_means_instance)) = self.instance_test(&i.test) {
                    let then_leaves = leaves(&i.body);
                    let else_leaves = i.elif_else_clauses.len() == 1
                        && i.elif_else_clauses[0].test.is_none()
                        && leaves(&i.elif_else_clauses[0].body);
                    if (then_leaves && !holds_means_instance && i.elif_else_clauses.is_empty())
                        || (else_leaves && holds_means_instance)
                    {
                        self.assume_place(&place);
                    }
                }
            }
        }
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
                // A bare name of a builtin as a statement does nothing.
                if let py::Expr::Name(n) = &*e.value {
                    if !self.is_variable(n.id.as_str())
                        && !self.module.funcs.contains_key(n.id.as_str())
                        && !self.module.class_index.contains_key(n.id.as_str())
                        && types::builtin_index(n.id.as_str()).is_some()
                    {
                        return Ok(());
                    }
                }
                let v = self.expr(&e.value)?;
                push(out, TypedStatement::Expression(Box::new(v.node)));
            }
            py::Stmt::Assign(a) => {
                if a.targets.len() == 1
                    && self.reversed_slice_assign(&a.targets[0], &a.value, span, out)?
                {
                    return Ok(());
                }
                // A literal that says nothing of its elements is built
                // as the list its name or field holds, which inference
                // typed by what the program puts in it.
                if let [target] = a.targets.as_slice() {
                    if let Some(count) = types::unkinded_list(&a.value) {
                        let kind = match target {
                            py::Expr::Name(n) => match self.var_ty(n.id.as_str()) {
                                Ty::List(e) if e != Elem::Object => Some(e),
                                _ => None,
                            },
                            py::Expr::Attribute(attr) => match self.ty_of(&attr.value) {
                                Ty::Class(k) => {
                                    match self.module.field(k as usize, attr.attr.as_str()) {
                                        Some((_, Ty::List(e))) if e != Elem::Object => Some(e),
                                        _ => None,
                                    }
                                }
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some(e) = kind {
                            let items = if types::is_empty_list(&a.value) {
                                Vec::new()
                            } else {
                                let none = node(
                                    TypedExpression::Literal(TypedLiteral::Null),
                                    Ty::None,
                                    span,
                                );
                                vec![Val {
                                    node: none,
                                    ty: Ty::None,
                                }]
                            };
                            let mut value = Val {
                                node: self.list_of(items, e, span),
                                ty: Ty::List(e),
                            };
                            if let Some(n) = count {
                                let times = self.expr(n)?;
                                value =
                                    self.arithmetic(py::Operator::Mult, value, times, n, span)?;
                            }
                            return self.bind(target, value, span, out);
                        }
                    }
                }
                // `a = b = v` evaluates `v` once and binds each target
                // to it, left to right.
                let value = self.expr(&a.value)?;
                if a.targets.len() == 1 {
                    self.bind(&a.targets[0], value, span, out)?;
                } else {
                    let held = self.hold(value, out, span);
                    for target in &a.targets {
                        let again = Val {
                            node: held.node.clone(),
                            ty: held.ty,
                        };
                        self.bind(target, again, span, out)?;
                    }
                }
            }
            py::Stmt::AnnAssign(a) => {
                let Some(v) = &a.value else {
                    return unsupported("annotation without a value", a);
                };
                // An annotation converts nothing: `x: float = 3` binds the
                // int 3. A dynamic value is read back as the annotation
                // says, with the check a parameter's annotation gets.
                if let Some(e) =
                    types::annotated_empty_list(&self.module.class_index, &a.annotation, v)
                {
                    let value = Val {
                        node: self.list_of(Vec::new(), e, span),
                        ty: Ty::List(e),
                    };
                    return self.bind(&a.target, value, span, out);
                }
                let value = self.expr(v)?;
                let declared =
                    types::annotated_value(&self.module.class_index, &a.annotation, value.ty);
                let value = if declared != value.ty {
                    Val {
                        node: self.coerce(value, declared),
                        ty: declared,
                    }
                } else {
                    value
                };
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
                let cond = self.expr(&w.test)?;
                let condition = self.truthy(cond);
                // A test that hoists work re-does it every pass: the loop
                // becomes `while true { work; if not test: break; body }`.
                let pre = std::mem::take(&mut self.hoisted);
                let else_flag = if w.orelse.is_empty() {
                    None
                } else {
                    let flag = self.temp();
                    out.push(TypedNode::new(
                        TypedStatement::Let(TypedLet {
                            name: flag,
                            ty: ir(Ty::Bool),
                            mutability: Mutability::Mutable,
                            initializer: Some(Box::new(node(
                                TypedExpression::Literal(TypedLiteral::Bool(true)),
                                Ty::Bool,
                                span,
                            ))),
                            span,
                        }),
                        Type::Unknown,
                        span,
                    ));
                    Some(flag)
                };
                let body = self.in_loop_else(else_flag, |this| this.block(&w.body, span))?;
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
                if let Some(flag) = else_flag {
                    let suite = self.loop_else(flag, &w.orelse, span)?;
                    out.push(suite);
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
                    if let py::Expr::Slice(slice) = &*sub.slice {
                        if slice.lower.is_some() || slice.upper.is_some() || slice.step.is_some() {
                            return unsupported("del of a bounded slice", target);
                        }
                        let seq = self.expr(&sub.value)?;
                        if !matches!(seq.ty, Ty::List(_)) {
                            return unsupported("del of a non-list slice", target);
                        }
                        push(
                            out,
                            TypedStatement::Expression(Box::new(method_call(
                                seq.node,
                                "clear",
                                vec![],
                                Ty::None,
                                span,
                            ))),
                        );
                        continue;
                    }
                    let seq = self.expr(&sub.value)?;
                    let stmt = match seq.ty {
                        Ty::List(e) => {
                            let i = self.expr_as(&sub.slice, Ty::Int)?;
                            elem_call("pop", e, vec![seq.node, i], span)
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
                self.escape_into(span, out);
            }
            py::Stmt::Assert(a) => {
                // `assert isinstance(x, C)` right after `x = v`: the
                // binding checked already.
                if let py::Expr::Call(c) = &*a.test {
                    if types::is_name(&c.func, "isinstance") && c.arguments.args.len() == 2 {
                        if let py::Expr::Name(x) = &c.arguments.args[0] {
                            if self.locals.narrowed.contains_key(x.id.as_str()) {
                                return Ok(());
                            }
                        }
                    }
                }
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
            // Imports were resolved when the module was collected; the
            // statement itself does nothing at run time.
            py::Stmt::Import(_) | py::Stmt::ImportFrom(_) => {}
            other => return unsupported(types::stmt_kind(other), other),
        }
        Ok(())
    }

    /// `xs[a:b] = xs[c:d:-1]` over one list name, with bounds that
    /// cannot rebind the name: lowered as one call that reverses the
    /// elements in place when the two ranges name the same ones, so the
    /// flip of a permutation copies nothing. Anything else is left to
    /// the general slice assignment; says whether it was taken.
    fn reversed_slice_assign(
        &mut self,
        target: &py::Expr,
        value: &py::Expr,
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Result<bool> {
        let (py::Expr::Subscript(t), py::Expr::Subscript(v)) = (target, value) else {
            return Ok(false);
        };
        let (py::Expr::Name(tn), py::Expr::Name(vn)) = (&*t.value, &*v.value) else {
            return Ok(false);
        };
        let (py::Expr::Slice(ts), py::Expr::Slice(vs)) = (&*t.slice, &*v.slice) else {
            return Ok(false);
        };
        let minus_one = |e: &Option<Box<py::Expr>>| {
            matches!(e.as_deref(), Some(py::Expr::UnaryOp(u))
                if u.op == py::UnaryOp::USub
                    && matches!(&*u.operand, py::Expr::NumberLiteral(n)
                        if matches!(&n.value, py::Number::Int(i) if i.as_u64() == Some(1))))
        };
        fn pure_bound(e: &Option<Box<py::Expr>>) -> bool {
            fn pure(e: &py::Expr) -> bool {
                match e {
                    py::Expr::Name(_) | py::Expr::NumberLiteral(_) => true,
                    py::Expr::BinOp(b) => pure(&b.left) && pure(&b.right),
                    py::Expr::UnaryOp(u) => pure(&u.operand),
                    _ => false,
                }
            }
            e.as_deref().is_none_or(pure)
        }
        if tn.id != vn.id
            || ts.step.is_some()
            || !minus_one(&vs.step)
            || !pure_bound(&ts.lower)
            || !pure_bound(&ts.upper)
            || !pure_bound(&vs.lower)
            || !pure_bound(&vs.upper)
        {
            return Ok(false);
        }
        let seq = self.expr(&t.value)?;
        let Ty::List(e) = seq.ty else {
            return Ok(false);
        };
        let bounds = |this: &mut Self,
                      lower: &Option<Box<py::Expr>>,
                      upper: &Option<Box<py::Expr>>|
         -> Result<(Node, Node, Node)> {
            let mut mask = 0;
            let mut bound = |this: &mut Self, e: &Option<Box<py::Expr>>, bit: i64| match e {
                Some(e) => {
                    mask |= bit;
                    this.expr_as(e, Ty::Int)
                }
                None => Ok(int_lit(0, span)),
            };
            let lo = bound(this, lower, 1)?;
            let hi = bound(this, upper, 2)?;
            Ok((lo, hi, int_lit(mask, span)))
        };
        let (start, stop, mask) = bounds(self, &ts.lower, &ts.upper)?;
        let (rstart, rstop, rmask) = bounds(self, &vs.lower, &vs.upper)?;
        let call = call(
            &list_fn("assign_reversed_slice", e),
            vec![seq.node, start, stop, mask, rstart, rstop, rmask],
            Ty::None,
            span,
        );
        out.push(TypedNode::new(
            TypedStatement::Expression(Box::new(call)),
            Type::Unknown,
            span,
        ));
        out.push(self.pending_check(span));
        Ok(true)
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
            // A name the next statement asserts to be an instance of a
            // class: the assert's check happens here, on the value, and
            // the name holds the instance.
            py::Expr::Name(n)
                if self
                    .locals
                    .narrowed
                    .get(n.id.as_str())
                    .is_some_and(|&k| value.ty != Ty::Class(k)) =>
            {
                let k = self.locals.narrowed[n.id.as_str()];
                // The check and the value it reads run ahead of the
                // statement, with whatever the checked read hoists after
                // them.
                let mut pre = Vec::new();
                let held = self.hold(value, &mut pre, span);
                self.hoisted.append(&mut pre);
                let class = py::Expr::Name(py::ExprName {
                    node_index: Default::default(),
                    range: target.range(),
                    id: py::name::Name::new(self.module.classes[k as usize].name.clone()),
                    ctx: py::ExprContext::Load,
                });
                let test = self.isinstance(
                    Val {
                        node: held.node.clone(),
                        ty: held.ty,
                    },
                    &class,
                    span,
                )?;
                let mut raise = Vec::new();
                self.raise_named("AssertionError", str_lit("", span), span, &mut raise);
                let failed = node(
                    TypedExpression::Unary(TypedUnary {
                        op: UnaryOp::Not,
                        operand: Box::new(self.truthy(test)),
                    }),
                    Ty::Bool,
                    span,
                );
                self.hoisted.push(TypedNode::new(
                    TypedStatement::If(TypedIf {
                        condition: Box::new(failed),
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
                let instance = Val {
                    node: self.coerce(held, Ty::Class(k)),
                    ty: Ty::Class(k),
                };
                return self.bind(target, instance, span, out);
            }
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
                if let py::Expr::Slice(sl) = &*sub.slice {
                    let rhs_pre = std::mem::take(&mut self.hoisted);
                    let seq = self.expr(&sub.value)?;
                    let target_pre = std::mem::take(&mut self.hoisted);
                    self.hoisted = rhs_pre;
                    let Ty::List(e) = seq.ty else {
                        return unsupported("slice assignment on a non-list", target);
                    };
                    let seq_node = if target_pre.is_empty() {
                        seq.node
                    } else {
                        Self::block_value(target_pre, seq.node, seq.ty, span)
                    };
                    let source = if matches!(value.ty, Ty::List(_)) {
                        value
                    } else {
                        Val {
                            node: self.iterable(value, span),
                            ty: Ty::List(Elem::Object),
                        }
                    };
                    let typed = self.coerce(source, Ty::List(e));
                    let mut mask = 0;
                    let mut bound =
                        |this: &mut Self, expr: &Option<Box<py::Expr>>, bit: i64| -> Result<Node> {
                            match expr {
                                Some(expr) => {
                                    mask |= bit;
                                    let outer = std::mem::take(&mut this.hoisted);
                                    let value = this.expr_as(expr, Ty::Int)?;
                                    let pre = std::mem::replace(&mut this.hoisted, outer);
                                    Ok(if pre.is_empty() {
                                        value
                                    } else {
                                        Self::block_value(pre, value, Ty::Int, span)
                                    })
                                }
                                None => Ok(int_lit(0, span)),
                            }
                        };
                    let start = bound(self, &sl.lower, 1)?;
                    let stop = bound(self, &sl.upper, 2)?;
                    let step = bound(self, &sl.step, 4)?;
                    let call = call(
                        &list_fn("assign_slice", e),
                        vec![typed, seq_node, start, stop, step, int_lit(mask, span)],
                        Ty::None,
                        span,
                    );
                    out.push(TypedNode::new(
                        TypedStatement::Expression(Box::new(call)),
                        Type::Unknown,
                        span,
                    ));
                    out.push(self.pending_check(span));
                    return Ok(());
                }
                let seq = self.expr(&sub.value)?;
                let stmt = match seq.ty {
                    Ty::List(e) => {
                        let i = self.expr_as(&sub.slice, Ty::Int)?;
                        let v = self.elem_arg(value, e);
                        call(&list_fn("set", e), vec![seq.node, i, v], Ty::None, span)
                    }
                    Ty::Dict => {
                        let (k, by) = self.dict_key(&sub.slice)?;
                        let v = self.coerce(value, Ty::Object);
                        call(
                            &format!("zb_dict_set{by}"),
                            vec![seq.node, k, v],
                            Ty::None,
                            span,
                        )
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
            // `[a, b] = v` unpacks as `a, b = v` does.
            py::Expr::List(l) => {
                let elts: Vec<py::Expr> = l.elts.clone();
                let as_tuple = py::Expr::Tuple(py::ExprTuple {
                    elts,
                    ctx: l.ctx,
                    parenthesized: true,
                    range: l.range,
                    node_index: l.node_index.clone(),
                });
                return self.bind(&as_tuple, value, span, out);
            }
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
        // An instance assigned from its constructor is known to be one
        // until the block ends or the variable is assigned again.
        if let Ty::Class(_) = ty {
            if value.ty != Ty::None && self.known_instance(&value.node) {
                self.nonnull.insert(name);
            } else {
                self.nonnull.remove(&name);
            }
        }
        if let Some(cell) = self.cells.get(n.id.as_str()).copied() {
            let value = self.coerce(value, ty);
            let boxed = self.coerce(Val { node: value, ty }, Ty::Object);
            let set = binary(
                BinaryOp::Assign,
                slot(var(cell, Ty::List(Elem::Object), span), 0, Ty::Object, span),
                boxed,
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
        // What the test settles holds in the branch it selects.
        let settled = self.instance_test(test);
        let vars = self.nonnull.clone();
        let fields = self.nonnull_fields.clone();
        if let Some((place, true)) = &settled {
            self.assume_place(place);
        }
        let then_block = self.block(body, span)?;
        self.nonnull = vars.clone();
        self.nonnull_fields = fields.clone();
        if let Some((place, false)) = &settled {
            self.assume_place(place);
        }
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
        self.nonnull = vars;
        self.nonnull_fields = fields;
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
            Ty::List(e) => elem_call("get", e, vec![seq.node, index], span),
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
        if f.orelse.is_empty() {
            return self.for_with_body(f, Vec::new(), span);
        }
        // `for/else` as `while/else`: a flag every `break` clears, and
        // the `else` suite after the loop when it still stands and no
        // exception is pending.
        let flag = self.temp();
        let mut statements = vec![TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: flag,
                ty: ir(Ty::Bool),
                mutability: Mutability::Mutable,
                initializer: Some(Box::new(node(
                    TypedExpression::Literal(TypedLiteral::Bool(true)),
                    Ty::Bool,
                    span,
                ))),
                span,
            }),
            Type::Unknown,
            span,
        )];
        self.for_else = Some(flag);
        let loop_stmt = self.for_with_body(f, Vec::new(), span)?;
        statements.push(TypedNode::new(loop_stmt, Type::Unknown, span));
        statements.push(self.loop_else(flag, &f.orelse, span)?);
        Ok(TypedStatement::Block(TypedBlock { statements, span }))
    }

    /// The `else` suite of a loop: run when the loop's flag still stands
    /// (no `break` cleared it) and no exception is leaving.
    fn loop_else(&mut self, flag: InternedString, orelse: &[py::Stmt], span: Span) -> Result<Stmt> {
        let not_pending = node(
            TypedExpression::Unary(TypedUnary {
                op: UnaryOp::Not,
                operand: Box::new(self.pending(span)),
            }),
            Ty::Bool,
            span,
        );
        let condition = binary(
            BinaryOp::And,
            var(flag, Ty::Bool, span),
            not_pending,
            Ty::Bool,
            span,
        );
        Ok(TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(condition),
                then_block: self.block(orelse, span)?,
                else_block: None,
                span,
            }),
            Type::Unknown,
            span,
        ))
    }

    /// `for x in range(...)` as a counted loop; anything else iterates
    /// by index over a sequence. `extra` statements follow the body.
    fn for_with_body(
        &mut self,
        f: &py::StmtFor,
        extra: Vec<Stmt>,
        span: Span,
    ) -> Result<TypedStatement> {
        if f.is_async {
            return unsupported("async for", f);
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

    // ─── Imported modules ──────────────────────────────────────────

    /// The member `value.attr` names when `value` is an imported
    /// module's name that no variable shadows.
    fn module_member_of(&self, value: &py::Expr, attr: &str) -> Option<stdlib::Member> {
        let py::Expr::Name(m) = value else {
            return None;
        };
        if self.is_variable(m.id.as_str()) {
            return None;
        }
        self.module.module_member(m.id.as_str(), attr)
    }

    /// A module member read as a value: a constant, or what the library
    /// computes for it. A function is not a value here.
    fn member_value(
        &mut self,
        member: stdlib::Member,
        name: &str,
        e: &py::Expr,
        span: Span,
    ) -> Result<Val> {
        Ok(match member {
            stdlib::Member::Float(f) => Val {
                node: node(
                    TypedExpression::Literal(TypedLiteral::Float(f)),
                    Ty::Float,
                    span,
                ),
                ty: Ty::Float,
            },
            stdlib::Member::Int(i) => Val {
                node: int_lit(i, span),
                ty: Ty::Int,
            },
            stdlib::Member::Value { ty, zb } => Val {
                node: call(zb, Vec::new(), ty, span),
                ty,
            },
            stdlib::Member::Func { zb, .. } if zb.starts_with("zb_bisect_") => Val {
                node: call(
                    "zb_func_new",
                    vec![
                        code_of(&format!("{zb}_call"), span),
                        int_lit(zyntax_builtins::functions::VARIADIC_ARITY, span),
                        self.list_of(Vec::new(), Elem::Object, span),
                    ],
                    Ty::Object,
                    span,
                ),
                ty: Ty::Object,
            },
            stdlib::Member::Func { .. } => {
                return unsupported(format!("`{name}` of a module as a value"), e)
            }
        })
    }

    /// A call to a module's function: arguments converted to the
    /// declared parameter types, the library function called.
    fn stdlib_call(
        &mut self,
        member: stdlib::Member,
        name: &str,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        let stdlib::Member::Func { params, ret, zb } = member else {
            return unsupported(format!("calling `{name}`, which is not a function"), c);
        };
        if !keywords.is_empty() {
            return unsupported(format!("keyword arguments to `{name}`"), c);
        }
        if let Some(op) = zb
            .strip_prefix("zb_bisect_")
            .map(|side| format!("bisect_{side}"))
            .or_else(|| {
                zb.strip_prefix("zb_insort_")
                    .map(|side| format!("insort_{side}"))
            })
        {
            if !(2..=4).contains(&args.len()) {
                return unsupported(
                    format!("calling `{name}` with {} argument(s)", args.len()),
                    c,
                );
            }
            let source = self.expr(&args[0])?;
            let Ty::List(elem) = source.ty else {
                return unsupported("bisect on a non-list", &args[0]);
            };
            let mut pre = std::mem::take(&mut self.hoisted);
            let source = self.hold(source, &mut pre, span);
            self.hoisted.extend(pre);
            let value = self.expr_as(&args[1], elem.ty())?;
            let low = if let Some(arg) = args.get(2) {
                self.expr_as(arg, Ty::Int)?
            } else {
                int_lit(0, span)
            };
            let high = if let Some(arg) = args.get(3) {
                self.expr_as(arg, Ty::Int)?
            } else {
                method_call(source.node.clone(), "len", vec![], Ty::Int, span)
            };
            let result = Val {
                node: call(
                    &list_fn(&op, elem),
                    vec![source.node, value, low, high],
                    ret,
                    span,
                ),
                ty: ret,
            };
            return Ok(self.guard(result, span));
        }
        // Forms with a default or a second signature.
        let (params, zb): (Vec<Ty>, &str) = match (zb, args.len()) {
            ("zb_exit", 0) => {
                return Ok(Val {
                    node: call("zb_exit", vec![int_lit(0, span)], Ty::None, span),
                    ty: Ty::None,
                })
            }
            ("zb_exit", 1) if matches!(&args[0], py::Expr::NoneLiteral(_)) => {
                return Ok(Val {
                    node: call("zb_exit", vec![int_lit(0, span)], Ty::None, span),
                    ty: Ty::None,
                })
            }
            ("zb_math_log", 2) => (vec![Ty::Float, Ty::Float], "zb_math_log_base"),
            _ => (params.to_vec(), zb),
        };
        if args.len() != params.len() {
            return unsupported(
                format!(
                    "calling `{name}` with {} argument(s); it takes {}",
                    args.len(),
                    params.len()
                ),
                c,
            );
        }
        let mut lowered = Vec::with_capacity(args.len());
        for (a, &want) in args.iter().zip(params.iter()) {
            let v = self.expr(a)?;
            let node = match (v.ty, want) {
                (Ty::Object, Ty::Float) => call("zb_any_float", vec![v.node], Ty::Float, span),
                (Ty::Object, Ty::Int) => call("zb_any_int", vec![v.node], Ty::Int, span),
                _ => self.coerce(v, want),
            };
            lowered.push(node);
        }
        Ok(Val {
            node: call(zb, lowered, ret, span),
            ty: ret,
        })
    }

    // ─── Iteration builtins ────────────────────────────────────────

    /// A value as a `List<Any>` to iterate: a typed list boxed, a string
    /// its characters, a dynamic value whatever it iterates as.
    /// The class an expression names: a bare name of one of the
    /// program's classes that no variable shadows.
    fn class_named(&self, e: &py::Expr) -> Option<usize> {
        let py::Expr::Name(n) = e else {
            return None;
        };
        if self.is_variable(n.id.as_str()) {
            return None;
        }
        self.module.class_index.get(n.id.as_str()).copied()
    }

    /// Whether a call of the builtin `name` with these arguments is one
    /// the direct lowering handles: the arity the builtin has, no
    /// keywords, and integer bounds for `range`.
    fn builtin_takes(&self, name: &str, c: &py::ExprCall) -> bool {
        let args = &c.arguments.args;
        if !c.arguments.keywords.is_empty()
            || args.iter().any(|a| matches!(a, py::Expr::Starred(_)))
        {
            return false;
        }
        match name {
            "range" => {
                (1..=3).contains(&args.len())
                    && args.iter().all(|a| {
                        matches!(self.ty_of(a), Ty::Int | Ty::Bool | Ty::Object | Ty::Unknown)
                    })
            }
            "len" | "abs" | "repr" | "hash" | "ord" | "chr" | "id" | "iter" | "next" => {
                args.len() == 1
            }
            "str" | "int" | "float" | "bool" | "list" | "tuple" | "set" | "frozenset" | "dict" => {
                args.len() <= 1
            }
            "sorted" | "reversed" | "enumerate" | "sum" | "any" | "all" => {
                (1..=2).contains(&args.len())
            }
            "min" | "max" | "zip" | "map" | "filter" | "print" => !args.is_empty(),
            "round" | "divmod" | "pow" | "isinstance" => (1..=3).contains(&args.len()),
            _ => false,
        }
    }

    /// A dict key as the lookup takes it: a string as itself, for the
    /// lookups that hash and compare a string without boxing it, and
    /// anything else as a dynamic value. The suffix names the lookup.
    fn dict_key(&mut self, e: &py::Expr) -> Result<(Node, &'static str)> {
        if self.ty_of(e) == Ty::Str {
            Ok((self.expr_as(e, Ty::Str)?, "_str"))
        } else {
            Ok((self.expr_as(e, Ty::Object)?, ""))
        }
    }

    fn iterable(&mut self, v: Val, span: Span) -> Node {
        match v.ty {
            Ty::Str => {
                let chars = call("zb_str_chars", vec![v.node], Ty::List(Elem::Str), span);
                call(
                    &list_fn("to_any", Elem::Str),
                    vec![chars],
                    Ty::List(Elem::Object),
                    span,
                )
            }
            Ty::Object => call("zb_any_iter", vec![v.node], Ty::List(Elem::Object), span),
            _ => self.coerce(v, Ty::List(Elem::Object)),
        }
    }

    /// The typed list an iterable argument stands for: a `range(...)` is
    /// its ints, a string its characters, anything else what it holds.
    fn sequence(&mut self, e: &py::Expr, span: Span) -> Result<Val> {
        if let py::Expr::Call(rc) = e {
            if types::is_name(&rc.func, "range") && rc.arguments.keywords.is_empty() {
                let mut bounds = Vec::new();
                for a in rc.arguments.args.iter() {
                    bounds.push(self.expr_as(a, Ty::Int)?);
                }
                let (start, stop, step) = match bounds.len() {
                    1 => (int_lit(0, span), bounds.remove(0), int_lit(1, span)),
                    2 => {
                        let stop = bounds.remove(1);
                        (bounds.remove(0), stop, int_lit(1, span))
                    }
                    3 => {
                        let step = bounds.remove(2);
                        let stop = bounds.remove(1);
                        (bounds.remove(0), stop, step)
                    }
                    _ => return unsupported("range() with more than three arguments", rc),
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
        }
        let v = self.expr(e)?;
        Ok(match v.ty {
            Ty::List(_) => v,
            Ty::Str => Val {
                node: call("zb_str_chars", vec![v.node], Ty::List(Elem::Str), span),
                ty: Ty::List(Elem::Str),
            },
            _ => Val {
                node: self.iterable(v, span),
                ty: Ty::List(Elem::Object),
            },
        })
    }

    /// An argument in a position that takes a function: a lambda, a def,
    /// a variable holding one, or a builtin's name, which becomes a
    /// function over one dynamic argument.
    fn callable_value(&mut self, e: &py::Expr) -> Result<Node> {
        const BUILTINS: [&str; 8] = ["str", "int", "float", "bool", "len", "abs", "repr", "type"];
        // A module's function (`math.sqrt`) or a name brought in from one
        // becomes the lambda that calls it.
        let imported = match e {
            py::Expr::Attribute(a) => {
                match (&*a.value, self.module_member_of(&a.value, a.attr.as_str())) {
                    (py::Expr::Name(m), Some(member)) => {
                        Some((member, format!("{}.{}", m.id.as_str(), a.attr.as_str())))
                    }
                    _ => None,
                }
            }
            py::Expr::Name(n)
                if !self.is_variable(n.id.as_str())
                    && !self.module.funcs.contains_key(n.id.as_str()) =>
            {
                self.module
                    .imported_name(n.id.as_str())
                    .map(|m| (m, n.id.to_string()))
            }
            _ => None,
        };
        if let Some((stdlib::Member::Func { params, .. }, source)) = imported {
            let args: Vec<String> = (0..params.len()).map(|i| format!("a{i}")).collect();
            let text = format!("lambda {}: {source}({})", args.join(", "), args.join(", "));
            let parsed = ruff_python_parser::parse_expression(&text).expect("a module call parses");
            let py::Expr::Lambda(lambda) = &*parsed.into_syntax().body else {
                unreachable!("the source is a lambda")
            };
            let span = span_of(e);
            return Ok(self.lambda(lambda, span)?.node);
        }
        if let py::Expr::Name(n) = e {
            let name = n.id.as_str();
            if !self.is_variable(name)
                && !self.module.funcs.contains_key(name)
                && !self.module.class_index.contains_key(name)
                && BUILTINS.contains(&name)
            {
                // The builtin as a function value is the lambda that
                // applies it.
                let source = format!("lambda a0: {name}(a0)");
                let parsed = ruff_python_parser::parse_expression(&source)
                    .expect("a builtin applied to a name parses");
                let py::Expr::Lambda(lambda) = &*parsed.into_syntax().body else {
                    unreachable!("the source is a lambda")
                };
                let span = span_of(e);
                return Ok(self.lambda(lambda, span)?.node);
            }
        }
        self.expr_as(e, Ty::Object)
    }

    /// `key=` and `reverse=` on an ordering call; anything else is refused.
    fn ordering_keywords(
        &mut self,
        keywords: &[py::Keyword],
        c: &py::ExprCall,
    ) -> Result<(Option<Node>, Option<Node>)> {
        let mut key = None;
        let mut reverse = None;
        for kw in keywords {
            match kw.arg.as_ref().map(|a| a.as_str()) {
                Some("key") => key = Some(self.callable_value(&kw.value)?),
                Some("reverse") => reverse = Some(self.expr_as(&kw.value, Ty::Bool)?),
                Some(other) => {
                    return unsupported(format!("the keyword argument `{other}` here"), c)
                }
                None => return unsupported("** in a call", c),
            }
        }
        Ok((key, reverse))
    }

    /// Sort a list in place by `key` and `reverse`, appending to `out`.
    fn sort_in_place(
        &mut self,
        list: Node,
        e: Elem,
        key: Option<Node>,
        reverse: Option<Node>,
        out: &mut Vec<Stmt>,
        span: Span,
    ) -> Result<()> {
        let statement = |node: Node| {
            TypedNode::new(
                TypedStatement::Expression(Box::new(node)),
                Type::Unknown,
                span,
            )
        };
        if key.is_none() && reverse.is_none() {
            out.push(statement(call(
                &list_fn("sort", e),
                vec![list],
                Ty::None,
                span,
            )));
            return Ok(());
        }
        // Keys are computed once, in a list parallel to the elements.
        let boxed = self.coerce(
            Val {
                node: list.clone(),
                ty: Ty::List(e),
            },
            Ty::List(Elem::Object),
        );
        let keys = match key {
            Some(f) => call("zb_list_map1", vec![f, boxed], Ty::List(Elem::Object), span),
            None => boxed,
        };
        let descending = reverse.unwrap_or_else(|| {
            node(
                TypedExpression::Literal(TypedLiteral::Bool(false)),
                Ty::Bool,
                span,
            )
        });
        out.push(statement(call(
            &list_fn("sort_by", e),
            vec![list, keys, descending],
            Ty::None,
            span,
        )));
        Ok(())
    }

    /// The builtins that produce or consume sequences: `enumerate`, `zip`,
    /// `map`, `filter`, `any`, `all`, and `sorted`, `min`, `max` with a
    /// key. `None` when `name` is not one of them.
    fn iteration_builtin(
        &mut self,
        name: &str,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        ty: Ty,
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Option<Val>> {
        let anys = Ty::List(Elem::Object);
        let node = match (name, args.len()) {
            ("enumerate", 1 | 2) => {
                let v = self.expr(&args[0])?;
                let xs = self.iterable(v, span);
                let start = match (args.get(1), keywords.first()) {
                    (Some(s), _) => self.expr_as(s, Ty::Int)?,
                    (None, Some(kw)) if kw.arg.as_ref().map(|a| a.as_str()) == Some("start") => {
                        self.expr_as(&kw.value, Ty::Int)?
                    }
                    (None, Some(_)) => return unsupported("this keyword on enumerate()", c),
                    (None, None) => int_lit(0, span),
                };
                call("zb_list_enumerate", vec![xs, start], anys, span)
            }
            ("zip", 2 | 3) => {
                let mut lists = Vec::new();
                for a in args {
                    let v = self.expr(a)?;
                    lists.push(self.iterable(v, span));
                }
                let f = if lists.len() == 2 {
                    "zb_list_zip2"
                } else {
                    "zb_list_zip3"
                };
                call(f, lists, anys, span)
            }
            ("map", 2 | 3) => {
                let f = self.callable_value(&args[0])?;
                let mut lowered = vec![f];
                for a in &args[1..] {
                    let v = self.expr(a)?;
                    lowered.push(self.iterable(v, span));
                }
                let f = if lowered.len() == 2 {
                    "zb_list_map1"
                } else {
                    "zb_list_map2"
                };
                call(f, lowered, anys, span)
            }
            ("filter", 2) => {
                let v = self.expr(&args[1])?;
                let xs = self.iterable(v, span);
                if matches!(&args[0], py::Expr::NoneLiteral(_)) {
                    call("zb_list_filter_truthy", vec![xs], anys, span)
                } else {
                    let f = self.callable_value(&args[0])?;
                    call("zb_list_filter", vec![f, xs], anys, span)
                }
            }
            ("any" | "all", 1) => {
                let v = self.consumed(&args[0], span)?;
                let xs = self.iterable(v, span);
                let f = if name == "any" {
                    "zb_list_any"
                } else {
                    "zb_list_all"
                };
                let v = Val {
                    node: call(f, vec![xs], Ty::Bool, span),
                    ty: Ty::Bool,
                };
                return Ok(Some(Val {
                    node: self.coerce(v, ty),
                    ty,
                }));
            }
            ("sorted", 1) if !keywords.is_empty() => {
                let source = self.sequence(&args[0], span)?;
                let Ty::List(e) = source.ty else {
                    unreachable!()
                };
                let (key, reverse) = self.ordering_keywords(keywords, c)?;
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
                self.sort_in_place(held.node.clone(), e, key, reverse, &mut statements, span)?;
                return Ok(Some(Val {
                    node: Self::block_value(statements, held.node, Ty::List(e), span),
                    ty: Ty::List(e),
                }));
            }
            ("min" | "max", _) if !keywords.is_empty() => {
                let (key, reverse) = self.ordering_keywords(keywords, c)?;
                if reverse.is_some() {
                    return unsupported(format!("reverse= on {name}()"), c);
                }
                let Some(f) = key else {
                    return unsupported(format!("{name}() with these keywords"), c);
                };
                let list = if args.len() == 1 {
                    self.sequence(&args[0], span)?
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
                let Ty::List(e) = list.ty else { unreachable!() };
                let mut statements = Vec::new();
                let held = self.hold(list, &mut statements, span);
                self.hoisted.extend(statements);
                let boxed = self.coerce(
                    Val {
                        node: held.node.clone(),
                        ty: Ty::List(e),
                    },
                    Ty::List(Elem::Object),
                );
                let keys = call("zb_list_map1", vec![f, boxed], anys, span);
                let best = Val {
                    node: elem_call(&format!("{name}_by"), e, vec![held.node, keys], span),
                    ty: e.ty(),
                };
                return Ok(Some(Val {
                    node: self.coerce(best, ty),
                    ty,
                }));
            }
            _ => return Ok(None),
        };
        Ok(Some(Val { node, ty: anys }))
    }

    // ─── Expressions ────────────────────────────────────────────────

    pub(crate) fn expr(&mut self, e: &py::Expr) -> Result<Val> {
        let mut v = self.expr_unchecked(e)?;
        // Whatever a call did to an object's fields, nothing settled
        // about them before it holds after.
        if matches!(e, py::Expr::Call(_)) {
            self.nonnull_fields.clear();
        }
        // A function value whose function is known is the record every
        // function value is; only a call reads the type, off the callee
        // expression itself.
        if let Ty::Closure(_) | Ty::Bound(_) | Ty::Builtin(_) = v.ty {
            v.ty = Ty::Object;
        }
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
            py::Expr::Name(n) if n.id.as_str() == "__name__" && !self.is_variable("__name__") => {
                Val {
                    node: str_lit(&self.module.name, span),
                    ty: Ty::Str,
                }
            }
            py::Expr::Name(n)
                if !self.is_variable(n.id.as_str())
                    && !self.module.funcs.contains_key(n.id.as_str())
                    && self.module.imported_name(n.id.as_str()).is_some() =>
            {
                let member = self.module.imported_name(n.id.as_str()).expect("checked");
                self.member_value(member, n.id.as_str(), e, span)?
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
                self.function_value(n.id.as_str(), span)?
            }
            py::Expr::Name(n)
                if n.id.as_str() == "range"
                    && !self.is_variable("range")
                    && !self.module.class_index.contains_key("range") =>
            {
                Val {
                    node: call(
                        "zb_func_new",
                        vec![
                            code_of("zb_range_call", span),
                            int_lit(zyntax_builtins::functions::VARIADIC_ARITY, span),
                            self.list_of(Vec::new(), Elem::Object, span),
                        ],
                        Ty::Object,
                        span,
                    ),
                    ty: Ty::Object,
                }
            }
            py::Expr::Name(n)
                if n.id.as_str() == "frozenset"
                    && !self.is_variable("frozenset")
                    && !self.module.class_index.contains_key("frozenset") =>
            {
                Val {
                    node: call(
                        "zb_func_new",
                        vec![
                            code_of("zb_frozenset_call", span),
                            int_lit(zyntax_builtins::functions::VARIADIC_ARITY, span),
                            self.list_of(Vec::new(), Elem::Object, span),
                        ],
                        Ty::Object,
                        span,
                    ),
                    ty: Ty::Object,
                }
            }
            py::Expr::Name(n)
                if !self.is_variable(n.id.as_str())
                    && !self.module.class_index.contains_key(n.id.as_str())
                    && types::builtin_index(n.id.as_str()).is_some() =>
            {
                return unsupported(format!("`{}` as a value", n.id.as_str()), e);
            }
            py::Expr::Name(n) => Val {
                node: var(intern(n.id.as_str()), ty, span),
                ty,
            },
            py::Expr::Lambda(l) => self.lambda(l, span)?,
            py::Expr::Generator(g) => self.generator_expr(g, span)?,
            py::Expr::Attribute(a) => {
                if let Some(member) = self.module_member_of(&a.value, a.attr.as_str()) {
                    return self.member_value(member, a.attr.as_str(), e, span);
                }
                let object = self.expr(&a.value)?;
                self.attribute(object, a.attr.as_str(), span)?
            }
            // `"..." % values` with a literal format.
            py::Expr::BinOp(b)
                if b.op == py::Operator::Mod && matches!(&*b.left, py::Expr::StringLiteral(_)) =>
            {
                let py::Expr::StringLiteral(l) = &*b.left else {
                    unreachable!()
                };
                let template = l.value.to_str().to_string();
                self.percent_format(&template, &b.right, span)?
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
                if then_pre.is_empty() && else_pre.is_empty() {
                    Val {
                        node: node(
                            TypedExpression::If(TypedIfExpr {
                                condition: Box::new(condition),
                                then_branch: Box::new(then_value),
                                else_branch: Box::new(else_value),
                            }),
                            ty,
                            span,
                        ),
                        ty,
                    }
                } else {
                    let mut out = Vec::new();
                    let picked = self.conditional_value(
                        condition,
                        (then_pre, then_value),
                        (else_pre, else_value),
                        ty,
                        span,
                        &mut out,
                    );
                    self.hoisted.extend(out);
                    Val { node: picked, ty }
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
                // Keys that are distinct literals need no search for
                // an earlier equal key: the literal lays the dict's
                // storage out itself, the index slot first.
                let distinct = distinct_literal_keys(d);
                let mut items = Vec::with_capacity(d.items.len() * 2 + 1);
                if distinct {
                    items.push(Val {
                        node: node(
                            TypedExpression::Literal(TypedLiteral::Null),
                            Ty::Object,
                            span,
                        ),
                        ty: Ty::Object,
                    });
                }
                for item in &d.items {
                    let Some(key) = &item.key else {
                        return unsupported("`**` in a dict literal", d);
                    };
                    items.push(self.expr(key)?);
                    items.push(self.expr(&item.value)?);
                }
                let pairs = self.list_of(items, Elem::Object, span);
                let maker = if distinct {
                    "zb_dict_from_distinct"
                } else {
                    "zb_dict_from_pairs"
                };
                Val {
                    node: call(maker, vec![pairs], Ty::Dict, span),
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
            let name = types::dunder_name(op);
            if let Some(r) = self.dunder(k as usize, name, left.node, vec![right], span) {
                return Ok(r);
            }
            return Err(Error::unsupported_span(
                format!(
                    "`{}` on {}, which defines no {name}",
                    op_text(op),
                    self.module.classes[k as usize].name
                ),
                span,
            ));
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
            // operation from the tags. An integer on the other side
            // travels as itself.
            let code = int_lit(types::arith_code(op), span);
            let node = match (left.ty, right.ty) {
                (Ty::Object, Ty::Int | Ty::Bool) => {
                    let r = self.coerce(right, Ty::Int);
                    call(
                        "zb_any_arith_i64",
                        vec![code, left.node, r],
                        Ty::Object,
                        span,
                    )
                }
                (Ty::Int | Ty::Bool, Ty::Object) => {
                    let l = self.coerce(left, Ty::Int);
                    call(
                        "zb_i64_arith_any",
                        vec![code, l, right.node],
                        Ty::Object,
                        span,
                    )
                }
                _ => {
                    let l = self.coerce(left, Ty::Object);
                    let r = self.coerce(right, Ty::Object);
                    call("zb_any_arith", vec![code, l, r], Ty::Object, span)
                }
            };
            return Ok(Val {
                node,
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
                if let Some(r) = self.dunder(
                    k as usize,
                    name,
                    left.node.clone(),
                    vec![right.clone()],
                    span,
                ) {
                    return Ok(self.truthy(r));
                }
                if name == "__ne__" {
                    if let Some(r) = self.dunder(
                        k as usize,
                        "__eq__",
                        left.node.clone(),
                        vec![right.clone()],
                        span,
                    ) {
                        let t = self.truthy(r);
                        return Ok(negate(t));
                    }
                }
                if matches!(op, py::CmpOp::Eq | py::CmpOp::NotEq) {
                    // No `__eq__`: identity.
                    let l = self.coerce(left, Ty::Object);
                    let other = self.coerce(right, Ty::Object);
                    let same = call("zb_any_same", vec![l, other], Ty::Bool, span);
                    return Ok(if op == py::CmpOp::NotEq {
                        negate(same)
                    } else {
                        same
                    });
                }
                return Err(Error::unsupported_span(
                    format!(
                        "ordering {} against a {right_ty:?}, which defines no {name}",
                        self.module.classes[k as usize].name
                    ),
                    right_node.span,
                ));
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
                    let item = self.elem_arg(left, e);
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
                    let by = if left.ty == Ty::Str { "_str" } else { "" };
                    let item = if left.ty == Ty::Str {
                        left.node
                    } else {
                        self.coerce(left, Ty::Object)
                    };
                    call(
                        &format!("zb_dict_contains{by}"),
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
                    // An instance is None when its pointer is null.
                    (Ty::None, Ty::Class(_)) => binary(
                        BinaryOp::Eq,
                        as_addr(right.node, span),
                        int_lit(0, span),
                        Ty::Bool,
                        span,
                    ),
                    (Ty::Class(_), Ty::None) => binary(
                        BinaryOp::Eq,
                        as_addr(left.node, span),
                        int_lit(0, span),
                        Ty::Bool,
                        span,
                    ),
                    (Ty::Class(_), Ty::Class(_)) => binary(
                        BinaryOp::Eq,
                        as_addr(left.node, span),
                        as_addr(right.node, span),
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
                    return Err(Error::unsupported_span(
                        "this comparison of sets".to_string(),
                        span,
                    ))
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
        // Each operand with what it hoists; a later operand's statements
        // run only if it is reached. The first operand's run regardless.
        let mut vals: Vec<(Vec<Stmt>, Val)> = Vec::with_capacity(b.values.len());
        for (i, v) in b.values.iter().enumerate() {
            let outer = std::mem::take(&mut self.hoisted);
            let val = self.expr(v)?;
            let mine = std::mem::replace(&mut self.hoisted, outer);
            if i == 0 {
                self.hoisted.extend(mine);
                vals.push((Vec::new(), val));
            } else {
                vals.push((mine, val));
            }
        }
        let needs_statements = vals.iter().any(|(pre, _)| !pre.is_empty());
        if needs_statements {
            // Right to left: each step chooses between its operand and
            // the rest, and the rest's statements go under the choice.
            let mut it = vals.into_iter().rev();
            let (last_pre, last) = it.next().expect("a bool op has operands");
            let mut rest = (last_pre, self.coerce(last, ty));
            for (pre, v) in it {
                let mut statements = pre;
                let coerced = self.coerce(v, ty);
                let bound = self.hold(Val { node: coerced, ty }, &mut statements, span);
                let test = self.truthy(Val {
                    node: bound.node.clone(),
                    ty,
                });
                let own = (Vec::new(), bound.node);
                let (then, otherwise) = if is_and { (rest, own) } else { (own, rest) };
                let picked =
                    self.conditional_value(test, then, otherwise, ty, span, &mut statements);
                rest = (statements, picked);
            }
            let (statements, picked) = rest;
            self.hoisted.extend(statements);
            return Ok(Val { node: picked, ty });
        }
        let vals: Vec<Val> = vals.into_iter().map(|(_, v)| v).collect();
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
                let (key, by) = self.dict_key(&sub.slice)?;
                Ok(Val {
                    node: call(
                        &format!("zb_dict_get{by}"),
                        vec![seq.node, key],
                        Ty::Object,
                        span,
                    ),
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
        let initializer = match produce {
            Produce::List(elem, _) => Some(self.list_of(Vec::new(), elem, span)),
            Produce::Dict(..) => Some(call("zb_dict_new", vec![], Ty::Dict, span)),
            _ => None,
        };
        if let Some(initializer) = initializer {
            statements[0] = TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: out,
                    ty: ir(ty),
                    mutability: Mutability::Immutable,
                    initializer: Some(Box::new(initializer)),
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
                let value = self.expr_as_elem(elt, elem)?;
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
                Err(Error::unsupported_span(
                    format!("{name}() with {} argument(s)", args.len()),
                    span,
                ))
            }
        };
        match receiver.ty {
            Ty::List(e) => {
                let list = receiver.node;
                let node = match name {
                    "append" => {
                        expect(1, self)?;
                        let v = self.expr_as_elem(&args[0], e)?;
                        method_call(list, "push", vec![v], Ty::None, span)
                    }
                    "pop" => {
                        let i = if args.is_empty() {
                            int_lit(-1, span)
                        } else {
                            self.expr_as(&args[0], Ty::Int)?
                        };
                        elem_call("pop", e, vec![list, i], span)
                    }
                    "insert" => {
                        expect(2, self)?;
                        let i = self.expr_as(&args[0], Ty::Int)?;
                        let v = self.expr_as_elem(&args[1], e)?;
                        call(&list_fn("insert", e), vec![list, i, v], Ty::None, span)
                    }
                    "remove" | "index" | "count" => {
                        expect(1, self)?;
                        let v = self.expr_as_elem(&args[0], e)?;
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
                        let (k, by) = self.dict_key(&args[0])?;
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
                        call(
                            &format!("zb_dict_get_default{by}"),
                            vec![d, k, none],
                            Ty::Object,
                            span,
                        )
                    }
                    ("get", 2) => {
                        let (k, by) = self.dict_key(&args[0])?;
                        let default = self.expr_as(&args[1], Ty::Object)?;
                        call(
                            &format!("zb_dict_get_default{by}"),
                            vec![d, k, default],
                            Ty::Object,
                            span,
                        )
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
                        return Err(Error::unsupported_span(
                            format!("dict.{name} with {} argument(s)", args.len()),
                            span,
                        ))
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
                        return Err(Error::unsupported_span(
                            format!("set.{name} with {} argument(s)", args.len()),
                            span,
                        ))
                    }
                };
                Ok(Val { node, ty })
            }
            Ty::Str => {
                let s = receiver.node;
                let mut lowered = Vec::with_capacity(args.len());
                for a in args {
                    lowered.push(self.consumed(a, span)?);
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
                        return Err(Error::unsupported_span(
                            format!("str.{name} with {} argument(s)", args.len()),
                            span,
                        ))
                    }
                };
                Ok(Val { node, ty })
            }
            _ => Err(Error::unsupported_span(
                format!("method `{name}` on a dynamic value"),
                span,
            )),
        }
    }

    /// A call: `print`, a conversion builtin, or a function the module
    /// defines.
    fn call(&mut self, c: &py::ExprCall, ty: Ty, span: Span) -> Result<Val> {
        let args = &c.arguments.args;
        let keywords = &c.arguments.keywords;
        if let py::Expr::Attribute(a) = &*c.func {
            if types::is_name(&a.value, "frozenset")
                && a.attr.as_str() == "union"
                && !self.is_variable("frozenset")
                && keywords.is_empty()
                && !args.is_empty()
            {
                let mut result = self.expr_as(&args[0], Ty::Set)?;
                for arg in &args[1..] {
                    let other = self.expr_as(arg, Ty::Set)?;
                    result = call("zb_set_or", vec![result, other], Ty::Set, span);
                }
                if args.len() == 1 {
                    result = call("zb_list_copy_any", vec![result], Ty::Set, span);
                }
                return Ok(Val {
                    node: result,
                    ty: Ty::Set,
                });
            }
        }
        // A function of an imported module, named through the module or
        // brought in by name.
        if let py::Expr::Attribute(a) = &*c.func {
            if let Some(member) = self.module_member_of(&a.value, a.attr.as_str()) {
                return self.stdlib_call(member, a.attr.as_str(), args, keywords, c, span);
            }
        }
        if let py::Expr::Name(n) = &*c.func {
            let name = n.id.as_str();
            if self.is_variable(name) {
                match self.typer().callee_ty(&c.func) {
                    Ty::Closure(k) => {
                        let callee = self.expr(&c.func)?;
                        return self.call_closure(k, callee, args, keywords, c, span);
                    }
                    // The method on the receiver as it is: the record
                    // the name holds is not read.
                    // The builtin's own call, under its own name, where
                    // the arguments are ones it takes; anything else goes
                    // through the record and fails as it would.
                    Ty::Builtin(k) if self.builtin_takes(types::BUILTIN_VALUES[k as usize], c) => {
                        let builtin = py::ExprCall {
                            func: Box::new(py::Expr::Name(py::ExprName {
                                node_index: Default::default(),
                                range: n.range,
                                id: py::name::Name::new(types::BUILTIN_VALUES[k as usize]),
                                ctx: py::ExprContext::Load,
                            })),
                            ..c.clone()
                        };
                        return self.call(&builtin, ty, span);
                    }
                    Ty::Bound(k) => {
                        let info = self.module.bounds[k as usize].clone();
                        let receiver = self.expr(&py::Expr::Name(info.receiver))?;
                        match receiver.ty {
                            Ty::Class(class) => {
                                return self.method_on(
                                    class as usize,
                                    receiver,
                                    &info.method,
                                    args,
                                    keywords,
                                    c,
                                    span,
                                );
                            }
                            Ty::List(_) if keywords.is_empty() => {
                                return self.method(receiver, &info.method, args, ty, span);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
                let callee = self.expr(&c.func)?;
                return self.call_value(callee, args, keywords, c, span);
            }
            if !self.module.funcs.contains_key(name) && !self.module.class_index.contains_key(name)
            {
                if let Some(member) = self.module.imported_name(name) {
                    return self.stdlib_call(member, name, args, keywords, c, span);
                }
            }
            if let Some(&k) = self.module.class_index.get(name) {
                return self.construct(k, args, keywords, c, span);
            }
            if let Some(sig) = self.module.funcs.get(name).cloned() {
                let lowered = self.arguments(name, &sig, args, keywords, c)?;
                if sig.ret == Ty::Gen {
                    return Ok(self.start_generator(name, &sig, lowered, Vec::new(), span));
                }
                let target = self.call_target(name, &sig.params, &lowered);
                let v = Val {
                    node: call(&target, lowered, sig.ret, span),
                    ty: sig.ret,
                };
                return Ok(self.guard_named(v, &target, span));
            }
            if name == "print" {
                return self.print(args, keywords, span);
            }
        }
        if let py::Expr::Attribute(a) = &*c.func {
            if types::is_super_call(&a.value) {
                return self.super_call(a.attr.as_str(), args, keywords, c, span);
            }
            // `Class.method(obj, ...)`: the method called with an explicit
            // receiver, which must be an instance of the class.
            if let Some(k) = self.class_named(&a.value) {
                if args.is_empty() {
                    return unsupported("a method called through its class without a receiver", c);
                }
                let receiver = self.expr(&args[0])?;
                let receiver = match receiver.ty {
                    Ty::Class(_) => receiver,
                    _ => Val {
                        node: self.coerce(receiver, Ty::Class(k as u16)),
                        ty: Ty::Class(k as u16),
                    },
                };
                return self.class_method_on(
                    k,
                    receiver,
                    a.attr.as_str(),
                    &args[1..],
                    keywords,
                    c,
                    span,
                );
            }
            let receiver = self.expr(&a.value)?;
            let receiver = if receiver.ty == Ty::None {
                Val {
                    node: self.coerce(receiver, Ty::Object),
                    ty: Ty::Object,
                }
            } else {
                receiver
            };
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
        // `xs.sort(key=..., reverse=...)` on a list.
        if let py::Expr::Attribute(a) = &*c.func {
            if a.attr.as_str() == "sort" && !keywords.is_empty() {
                let receiver = self.expr(&a.value)?;
                if let Ty::List(e) = receiver.ty {
                    let (key, reverse) = self.ordering_keywords(keywords, c)?;
                    let mut statements = Vec::new();
                    let held = self.hold(receiver, &mut statements, span);
                    self.sort_in_place(held.node, e, key, reverse, &mut statements, span)?;
                    let none = node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span);
                    return Ok(Val {
                        node: Self::block_value(statements, none, Ty::None, span),
                        ty: Ty::None,
                    });
                }
                return unsupported("sort() with keyword arguments on a non-list", c);
            }
        }
        if let py::Expr::Name(n) = &*c.func {
            if let Some(v) = self.iteration_builtin(n.id.as_str(), args, keywords, ty, c, span)? {
                return Ok(v);
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
            if let Ty::Closure(k) = self.ty_of(&c.func) {
                return self.call_closure(k, callee, args, keywords, c, span);
            }
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
                "input" if args.len() <= 1 => {
                    let node = match args.first() {
                        None => call("zb_read_line", Vec::new(), Ty::Str, span),
                        Some(p) => {
                            let prompt = self.expr(p)?;
                            let prompt = self.str_of(prompt);
                            call("zb_input", vec![prompt], Ty::Str, span)
                        }
                    };
                    return Ok(Val { node, ty: Ty::Str });
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
                                    return Err(Error::unsupported_span(
                                        format!(
                                            "len() of {}, which defines no __len__",
                                            self.module.classes[k as usize].name
                                        ),
                                        span,
                                    ))
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
                // `ord` of a one-character string, `chr` of a code point.
                "ord" if args.len() == 1 => {
                    let v = self.expr(&args[0])?;
                    let s = self.coerce(v, Ty::Str);
                    let ok = Val {
                        node: call("zb_str_ord", vec![s], Ty::Int, span),
                        ty: Ty::Int,
                    };
                    return Ok(self.guard(ok, span));
                }
                "chr" if args.len() == 1 => {
                    let v = self.expr(&args[0])?;
                    let code = self.coerce(v, Ty::Int);
                    let ok = Val {
                        node: call("zb_str_chr", vec![code], Ty::Str, span),
                        ty: Ty::Str,
                    };
                    return Ok(self.guard(ok, span));
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
                "sum" if args.len() == 1 || args.len() == 2 => {
                    let v = self.consumed(&args[0], span)?;
                    let (node, sum_ty) = match v.ty {
                        Ty::List(e @ (Elem::Int | Elem::Float | Elem::Object)) => {
                            (call(&list_fn("sum", e), vec![v.node], e.ty(), span), e.ty())
                        }
                        _ => {
                            let xs = self.iterable(v, span);
                            (
                                call("zb_list_sum_any", vec![xs], Ty::Object, span),
                                Ty::Object,
                            )
                        }
                    };
                    // `sum(xs, start)` is `start + sum(xs)`.
                    let total = match args.get(1) {
                        None => Val { node, ty: sum_ty },
                        Some(start) => {
                            let mut statements = Vec::new();
                            let total = self.hold(Val { node, ty: sum_ty }, &mut statements, span);
                            self.hoisted.extend(statements);
                            let start = self.expr(start)?;
                            self.arithmetic(py::Operator::Add, start, total, &args[0], span)?
                        }
                    };
                    return Ok(Val {
                        node: self.coerce(total, ty),
                        ty,
                    });
                }
                "min" | "max" => {
                    // Several arguments are the one-argument form over a
                    // list of them.
                    let list = if args.len() == 1 {
                        self.consumed(&args[0], span)?
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
                    let node = elem_call(name, e, vec![list.node], span);
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
                            let v = self.consumed(a, span)?;
                            match v.ty {
                                Ty::Dict => call("zb_dict_copy", vec![v.node], Ty::Dict, span),
                                // Anything else is a sequence of pairs.
                                _ => {
                                    let items = self.iterable(v, span);
                                    call("zb_dict_from_tuples", vec![items], Ty::Dict, span)
                                }
                            }
                        }
                    };
                    return Ok(Val { node, ty: Ty::Dict });
                }
                "set" | "frozenset" => {
                    let items = match args.first() {
                        None => self.list_of(Vec::new(), Elem::Object, span),
                        Some(a) => {
                            let v = self.consumed(a, span)?;
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
                            let v = self.consumed(a, span)?;
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
                    // A tuple holds dynamic values: a typed source is
                    // boxed into a fresh list first.
                    let source = if name == "tuple" && source.ty != Ty::List(Elem::Object) {
                        let node = self.coerce(source, Ty::List(Elem::Object));
                        Val {
                            node,
                            ty: Ty::List(Elem::Object),
                        }
                    } else {
                        source
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
                "pow" if args.len() == 3 => {
                    let a = self.expr_as(&args[0], Ty::Int)?;
                    let b = self.expr_as(&args[1], Ty::Int)?;
                    let m = self.expr_as(&args[2], Ty::Int)?;
                    let v = Val {
                        node: call("zb_int_pow_mod", vec![a, b, m], Ty::Int, span),
                        ty: Ty::Int,
                    };
                    return Ok(Val {
                        node: self.coerce(v, ty),
                        ty,
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
                        // With digits, the result keeps the argument's type:
                        // an int rounded to tens is an int.
                        (Ty::Int | Ty::Float | Ty::Bool, 2) => {
                            let f = self.coerce(v, Ty::Float);
                            let n = self.expr_as(&args[1], Ty::Int)?;
                            let r = call("zb_round_digits", vec![f, n], Ty::Float, span);
                            let node = if ty == Ty::Int {
                                cast(r, Ty::Int, span)
                            } else {
                                self.coerce(
                                    Val {
                                        node: r,
                                        ty: Ty::Float,
                                    },
                                    ty,
                                )
                            };
                            return Ok(Val { node, ty });
                        }
                        (_, 2) => {
                            let o = self.coerce(v, Ty::Object);
                            let n = self.expr_as(&args[1], Ty::Int)?;
                            call("zb_any_round_digits", vec![o, n], Ty::Object, span)
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
                "abs" if args.len() == 1 && ty == Ty::Object => {
                    let o = self.expr_as(&args[0], Ty::Object)?;
                    return Ok(Val {
                        node: call("zb_any_abs", vec![o], Ty::Object, span),
                        ty: Ty::Object,
                    });
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
        self.in_handler_loop(|this| -> Result<()> {
            for s in &t.body {
                this.stmt(s, &mut body)?;
            }
            Ok(())
        })?;
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
                // The match settled the class, so the read is trusted.
                let value = Val {
                    node: self.trusted(value, ty),
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
            // A clause that keeps nothing of the exception releases it:
            // at once when it never reads it, else wherever it leaves.
            let keeps = crate::scope::handler_keeps_exception(
                &h.body,
                h.name.as_ref().map(|n| n.id.as_str()),
            );
            let released_on_leaving = !keeps && h.name.is_some();
            if !keeps && h.name.is_none() {
                handler.push(release_caught(caught_name, hspan));
            }
            if released_on_leaving {
                self.handler_ctls.push(HandlerCtl {
                    caught: caught_name,
                    loop_depth: 0,
                });
            }
            for s in &h.body {
                self.stmt(s, &mut handler)?;
            }
            if released_on_leaving {
                self.handler_ctls.pop();
                if falls_through(&h.body) {
                    handler.push(release_caught(caught_name, hspan));
                }
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
    pub(crate) fn entry_body(&mut self, stmts: &[(&py::Stmt, Option<&str>)]) -> Result<Vec<Stmt>> {
        let span = stmts
            .first()
            .map(|(s, _)| span_of(*s))
            .unwrap_or(Span::new(0, 0));
        self.escapes.push(Escape::Break);
        // Each statement is reported against the module it came from,
        // and its spans name that module's file.
        let mut body = match stmts.first() {
            Some((first, _)) => self.cell_prologue(span_of(*first)),
            None => Vec::new(),
        };
        for (s, origin) in stmts {
            set_current_file(self.module.file_of(*origin));
            self.stmt(s, &mut body).map_err(|e| match origin {
                Some(m) => e.in_module(m),
                None => e,
            })?;
        }
        set_current_file(0);
        self.escapes.pop();
        body.push(TypedNode::new(
            TypedStatement::Break(None),
            Type::Unknown,
            span,
        ));
        // An exception left pending ends the program; the report and
        // the exit are the library's, kept out of this function's code.
        let report = vec![TypedNode::new(
            TypedStatement::Expression(Box::new(call(
                "zb_uncaught",
                vec![var(intern(PENDING), Ty::Object, span)],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        )];
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
        let held_from = gen.node.clone();
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
        let fresh = is_generator_start(&held_from);
        let pull = self.next_match(held.node.clone(), item, body, stop, span);
        prologue.push(one_pass(vec![pull], span));
        if fresh {
            prologue.push(free_generator(held.node, span));
        }
        Ok(TypedStatement::Block(TypedBlock {
            statements: prologue,
            span,
        }))
    }

    /// Every value a generator yields, as a list.
    fn generator_to_list(&mut self, gen: Node, span: Span) -> Node {
        let mut pre = Vec::new();
        let fresh = is_generator_start(&gen);
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
        let pull = self.next_match(held.node.clone(), item, push, stop, span);
        pre.push(one_pass(vec![pull], span));
        if fresh {
            pre.push(free_generator(held.node, span));
        }
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

    /// A paused fiber for `code` of a closure: the environment is the
    /// record's list (its cells sit where the environment keeps them)
    /// followed by the arguments (already coerced).
    fn start_generator_from(
        &mut self,
        code: &str,
        sig: &Sig,
        record: Node,
        args: Vec<Node>,
        span: Span,
    ) -> Val {
        let items: Vec<Val> = args
            .into_iter()
            .zip(&sig.params)
            .map(|(a, (_, ty))| Val { node: a, ty: *ty })
            .collect();
        let rest = self.list_of(items, Elem::Object, span);
        let env = call(
            "zb_list_concat_any",
            vec![record, rest],
            Ty::List(Elem::Object),
            span,
        );
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

    /// Take `name` to hold an instance from here on, as generated code
    /// that has just unboxed one does.
    pub(crate) fn assume_instance(&mut self, name: InternedString) {
        self.nonnull.insert(name);
    }

    /// Whether the lowering knows `node` is an instance and not None:
    /// a method's own receiver, a constructor's result, or a variable
    /// checked or assigned an instance since the block began.
    fn known_instance(&self, node: &Node) -> bool {
        match &node.node {
            TypedExpression::Variable(name) => {
                let is_self = self.class.is_some()
                    && self
                        .sig
                        .params
                        .first()
                        .is_some_and(|(p, _)| intern(p) == *name);
                // The trusted variant takes every instance-typed parameter
                // to be one.
                let trusted_param = self.trusted
                    && self
                        .sig
                        .params
                        .iter()
                        .any(|(p, t)| matches!(t, Ty::Class(_)) && intern(p) == *name);
                is_self
                    || trusted_param
                    || self.nonnull.contains(name)
                    || self.always_instance.contains(name)
            }
            TypedExpression::Call(c) => match &c.callee.node {
                TypedExpression::Variable(callee) => callee.resolve_global().is_some_and(|n| {
                    n.ends_with("$new")
                        || self.module.returns_instance.contains(&n)
                        || n.strip_suffix("$trusted")
                            .is_some_and(|base| self.module.returns_instance.contains(base))
                }),
                _ => false,
            },
            // An upcast of an instance is the same instance.
            TypedExpression::Cast(c) => {
                matches!(c.expr.ty, Type::Named { .. }) && self.known_instance(&c.expr)
            }
            TypedExpression::Field(f) => match &f.object.node {
                TypedExpression::Variable(v) => self
                    .nonnull_fields
                    .iter()
                    .any(|(var, field)| var == v && intern(field) == f.field),
                _ => false,
            },
            _ => false,
        }
    }

    /// What a test settles about an instance when it holds (`true`) or
    /// fails (`false`): `x is None`, `x is not None`, `x`, `not x`, for
    /// `x` a variable or a field of one. Returns the place and whether
    /// the test holding means the place is an instance.
    fn instance_test(&self, test: &py::Expr) -> Option<(Place, bool)> {
        match test {
            py::Expr::Compare(c) if c.ops.len() == 1 && c.comparators.len() == 1 => {
                let holds_means_instance = match c.ops[0] {
                    py::CmpOp::Is => false,
                    py::CmpOp::IsNot => true,
                    _ => return None,
                };
                let (place, other) = if matches!(c.comparators[0], py::Expr::NoneLiteral(_)) {
                    (&*c.left, &c.comparators[0])
                } else if matches!(&*c.left, py::Expr::NoneLiteral(_)) {
                    (&c.comparators[0], &*c.left)
                } else {
                    return None;
                };
                let _ = other;
                Some((self.place_of(place)?, holds_means_instance))
            }
            py::Expr::UnaryOp(u) if matches!(u.op, py::UnaryOp::Not) => {
                let (place, holds) = self.instance_test(&u.operand)?;
                Some((place, !holds))
            }
            other => {
                // The truth of an instance is its being one, unless the
                // class says otherwise.
                let place = self.place_of(other)?;
                let k = match self.place_ty(&place) {
                    Ty::Class(k) => k as usize,
                    _ => return None,
                };
                if self.module.method_sig(k, "__bool__").is_some()
                    || self.module.method_sig(k, "__len__").is_some()
                {
                    return None;
                }
                Some((place, true))
            }
        }
    }

    /// A variable, or a field of a variable holding an instance.
    fn place_of(&self, e: &py::Expr) -> Option<Place> {
        match e {
            py::Expr::Name(n) if self.is_variable(n.id.as_str()) => {
                Some(Place::Var(intern(n.id.as_str())))
            }
            py::Expr::Attribute(a) => match &*a.value {
                py::Expr::Name(n) if self.is_variable(n.id.as_str()) => {
                    Some(Place::Field(intern(n.id.as_str()), a.attr.to_string()))
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn place_ty(&self, place: &Place) -> Ty {
        match place {
            Place::Var(v) => v
                .resolve_global()
                .map(|n| self.var_ty(&n))
                .unwrap_or(Ty::Object),
            Place::Field(v, f) => {
                let Some(n) = v.resolve_global() else {
                    return Ty::Object;
                };
                match self.var_ty(&n) {
                    Ty::Class(k) => self
                        .module
                        .field(k as usize, f)
                        .map(|(_, t)| t)
                        .unwrap_or(Ty::Object),
                    _ => Ty::Object,
                }
            }
        }
    }

    /// Take `place` to hold an instance from here on. A field counts
    /// only while its object is known to be one.
    fn assume_place(&mut self, place: &Place) {
        match place {
            Place::Var(v) => {
                self.nonnull.insert(*v);
            }
            Place::Field(v, f) => {
                let object = var(*v, Ty::Object, Span::new(0, 0));
                if self.known_instance(&object) {
                    self.nonnull_fields.insert((*v, f.clone()));
                }
            }
        }
    }

    /// The variables of `body` that hold an instance throughout: each is
    /// first mentioned by a top-level `x = C(...)` and every other
    /// assignment to it, anywhere in the body, is another constructor
    /// call at the top level.
    fn always_instances(&self, body: &[py::Stmt]) -> std::collections::HashSet<InternedString> {
        use ruff_python_ast::visitor::{walk_expr, walk_stmt, Visitor};
        struct Names {
            mentioned: std::collections::HashSet<String>,
            stored: std::collections::HashSet<String>,
        }
        impl<'a> Visitor<'a> for Names {
            fn visit_stmt(&mut self, stmt: &'a py::Stmt) {
                match stmt {
                    py::Stmt::FunctionDef(f) => {
                        self.stored.insert(f.name.to_string());
                        // A nested function may write through nonlocal.
                        walk_stmt(self, stmt);
                    }
                    py::Stmt::ClassDef(c) => {
                        self.stored.insert(c.name.to_string());
                        walk_stmt(self, stmt);
                    }
                    py::Stmt::Global(g) => {
                        self.stored.extend(g.names.iter().map(|n| n.to_string()));
                    }
                    py::Stmt::Nonlocal(g) => {
                        self.stored.extend(g.names.iter().map(|n| n.to_string()));
                    }
                    py::Stmt::Import(i) => {
                        self.stored
                            .extend(i.names.iter().map(|a| a.name.to_string()));
                    }
                    py::Stmt::ImportFrom(i) => {
                        self.stored
                            .extend(i.names.iter().map(|a| a.name.to_string()));
                    }
                    py::Stmt::Try(t) => {
                        for h in &t.handlers {
                            let py::ExceptHandler::ExceptHandler(h) = h;
                            if let Some(n) = &h.name {
                                self.stored.insert(n.to_string());
                            }
                        }
                        walk_stmt(self, stmt);
                    }
                    _ => walk_stmt(self, stmt),
                }
            }
            fn visit_expr(&mut self, expr: &'a py::Expr) {
                if let py::Expr::Name(n) = expr {
                    self.mentioned.insert(n.id.to_string());
                    if !matches!(n.ctx, py::ExprContext::Load) {
                        self.stored.insert(n.id.to_string());
                    }
                }
                walk_expr(self, expr);
            }
        }
        let constructor_of = |value: &py::Expr| -> bool {
            match value {
                py::Expr::Call(c) => match &*c.func {
                    py::Expr::Name(n) => {
                        self.module.class_index.contains_key(n.id.as_str())
                            && !self.is_variable(n.id.as_str())
                    }
                    _ => false,
                },
                _ => false,
            }
        };
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut candidates: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut disqualified: std::collections::HashSet<String> = std::collections::HashSet::new();
        for s in body {
            let mut names = Names {
                mentioned: Default::default(),
                stored: Default::default(),
            };
            names.visit_stmt(s);
            let constructed = match s {
                py::Stmt::Assign(a) if a.targets.len() == 1 && constructor_of(&a.value) => {
                    match &a.targets[0] {
                        py::Expr::Name(n) => Some(n.id.to_string()),
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(name) = &constructed {
                // The value's own mentions come first; the target is
                // bound after.
                names.mentioned.remove(name);
                names.stored.remove(name);
                if !seen.contains(name) {
                    candidates.insert(name.clone());
                }
                seen.insert(name.clone());
            }
            // Anything else assigned in this statement is not a
            // constructor result; anything mentioned before its
            // constructor assignment was read unbound or from elsewhere.
            disqualified.extend(names.stored.iter().cloned());
            for m in names.mentioned {
                if !seen.contains(&m) {
                    seen.insert(m.clone());
                    disqualified.insert(m);
                }
            }
        }
        candidates
            .difference(&disqualified)
            .filter(|n| matches!(self.var_ty(n), Ty::Class(_)))
            .filter(|n| !self.cells.contains_key(n.as_str()))
            .map(|n| intern(n))
            .collect()
    }

    /// The function a call to `name` with `args` goes to: the trusted
    /// variant when there is one and every instance-typed argument is
    /// known to be an instance, `name` itself otherwise.
    fn call_target(&self, name: &str, params: &[(String, Ty)], args: &[Node]) -> String {
        if !self.module.trusted.contains(name) {
            return name.to_string();
        }
        let all_known = params
            .iter()
            .zip(args)
            .all(|((_, t), a)| !matches!(t, Ty::Class(_)) || self.known_instance(a));
        if all_known {
            types::trusted_name(name)
        } else {
            name.to_string()
        }
    }

    /// An instance about to be read through: None raises AttributeError
    /// here, so the read can trust the pointer. The check is hoisted
    /// ahead of the expression, and a variable checked once is known
    /// for the rest of the block.
    fn checked_instance(&mut self, object: Val, attr: &str, span: Span) -> Val {
        if self.known_instance(&object.node) {
            return object;
        }
        let held = match &object.node.node {
            TypedExpression::Variable(name) => {
                self.nonnull.insert(*name);
                object
            }
            _ => {
                let mut pre = Vec::new();
                let held = self.hold(object, &mut pre, span);
                self.hoisted.extend(pre);
                held
            }
        };
        let is_null = binary(
            BinaryOp::Eq,
            as_addr(held.node.clone(), span),
            int_lit(0, span),
            Ty::Bool,
            span,
        );
        let mut raise = Vec::new();
        self.raise_named(
            "AttributeError",
            str_lit(
                &format!("'NoneType' object has no attribute '{attr}'"),
                span,
            ),
            span,
            &mut raise,
        );
        self.hoisted.push(TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(is_null),
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
        held
    }

    /// `then(v)` when `v` is an instance, `when_null` when it is None.
    fn unless_null(
        &mut self,
        v: Val,
        when_null: Node,
        ty: Ty,
        then: impl FnOnce(&mut Self, Val) -> Node,
    ) -> Node {
        let span = v.node.span;
        if self.known_instance(&v.node) {
            return then(self, v);
        }
        let mut pre = Vec::new();
        let held = self.hold(v, &mut pre, span);
        self.hoisted.extend(pre);
        let is_instance = binary(
            BinaryOp::Ne,
            as_addr(held.node.clone(), span),
            int_lit(0, span),
            Ty::Bool,
            span,
        );
        // What `then` hoists belongs inside the branch.
        let outer = std::mem::take(&mut self.hoisted);
        let value = then(self, held);
        let inner = std::mem::take(&mut self.hoisted);
        self.hoisted = outer;
        let mut out = Vec::new();
        let result = self.conditional_value(
            is_instance,
            (inner, value),
            (Vec::new(), when_null),
            ty,
            span,
            &mut out,
        );
        self.hoisted.extend(out);
        result
    }

    /// `obj.attr` read.
    fn attribute(&mut self, object: Val, attr: &str, span: Span) -> Result<Val> {
        // A value known to be None reads as any dynamic value would:
        // the AttributeError is raised at run time.
        let object = if object.ty == Ty::None {
            Val {
                node: self.coerce(object, Ty::Object),
                ty: Ty::Object,
            }
        } else {
            object
        };
        if let Some(arity) = types::bound_method_arity(self.module, object.ty, attr) {
            return self.bound_method(object, attr, arity, span);
        }
        match object.ty {
            Ty::Class(k) => {
                let object = self.checked_instance(object, attr, span);
                let Some((_, ty)) = self.module.field(k as usize, attr) else {
                    return Err(Error::unsupported_span(
                        format!(
                            "attribute `{attr}` of {}, which has no such field",
                            self.module.classes[k as usize].name
                        ),
                        span,
                    ));
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
                let node = self.trusted(
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
            other => Err(Error::unsupported_span(
                format!("attribute `{attr}` of a {other:?}"),
                span,
            )),
        }
    }

    /// A function record with the receiver held in its first cell.
    fn bound_method(&mut self, receiver: Val, method: &str, arity: i64, span: Span) -> Result<Val> {
        let receiver = if matches!(receiver.ty, Ty::Class(_)) {
            self.checked_instance(receiver, method, span)
        } else {
            receiver
        };
        let code = self.lifted_name(&format!("bound${method}"));
        let receiver_ty = receiver.ty;
        let variadic = method == "pop" && matches!(receiver_ty, Ty::List(_));
        let params: Vec<(String, Ty)> = (0..if variadic { 0 } else { arity as usize })
            .map(|i| (format!("a{i}"), Ty::Object))
            .collect();
        let sig = Sig {
            params: params.clone(),
            ret: Ty::Object,
            defaults: vec![None; params.len()],
        };
        let mut locals = Locals::default();
        locals.vars.extend(params.iter().cloned());
        let mut child = Lowerer::new(
            self.module,
            &code,
            sig,
            locals,
            &Scope::default(),
            Vec::new(),
            HashMap::new(),
        );
        let env = var(intern("env"), Ty::List(Elem::Object), span);
        let held = Val {
            node: call(
                "zb_list_get_any",
                vec![env, int_lit(RECORD_CELLS_AT as i64, span)],
                Ty::Object,
                span,
            ),
            ty: Ty::Object,
        };
        let receiver_node = child.coerce(held, receiver_ty);
        let held = Val {
            node: receiver_node,
            ty: receiver_ty,
        };
        let mut body = Vec::new();
        let result = if variadic {
            let packed = var(intern("a0"), Ty::Object, span);
            let args = call(
                "zb_list_unbox_any",
                vec![packed],
                Ty::List(Elem::Object),
                span,
            );
            let count = method_call(args.clone(), "len", Vec::new(), Ty::Int, span);
            let too_many = binary(
                BinaryOp::Gt,
                count.clone(),
                int_lit(1, span),
                Ty::Bool,
                span,
            );
            let mut fail = Vec::new();
            child.raise_named(
                "TypeError",
                str_lit("pop() takes at most one argument", span),
                span,
                &mut fail,
            );
            body.push(TypedNode::new(
                TypedStatement::If(TypedIf {
                    condition: Box::new(too_many),
                    then_block: TypedBlock {
                        statements: fail,
                        span,
                    },
                    else_block: None,
                    span,
                }),
                Type::Unknown,
                span,
            ));
            let index = intern("index");
            body.push(TypedNode::new(
                TypedStatement::Let(TypedLet {
                    name: index,
                    ty: ir(Ty::Int),
                    mutability: Mutability::Mutable,
                    initializer: Some(Box::new(int_lit(-1, span))),
                    span,
                }),
                Type::Unknown,
                span,
            ));
            let read = Val {
                node: call(
                    "zb_list_get_any",
                    vec![args, int_lit(0, span)],
                    Ty::Object,
                    span,
                ),
                ty: Ty::Object,
            };
            child.guards = false;
            let read = child.coerce(read, Ty::Int);
            child.guards = true;
            let assign = TypedNode::new(
                TypedStatement::Expression(Box::new(binary(
                    BinaryOp::Assign,
                    var(index, Ty::Int, span),
                    read,
                    Ty::None,
                    span,
                ))),
                Type::Unknown,
                span,
            );
            body.push(TypedNode::new(
                TypedStatement::If(TypedIf {
                    condition: Box::new(binary(
                        BinaryOp::Eq,
                        count,
                        int_lit(1, span),
                        Ty::Bool,
                        span,
                    )),
                    then_block: TypedBlock {
                        statements: vec![assign, child.pending_check(span)],
                        span,
                    },
                    else_block: None,
                    span,
                }),
                Type::Unknown,
                span,
            ));
            let Ty::List(e) = receiver_ty else {
                unreachable!()
            };
            Val {
                node: elem_call("pop", e, vec![held.node, var(index, Ty::Int, span)], span),
                ty: e.ty(),
            }
        } else {
            let source = format!(
                "f({})",
                (0..arity as usize)
                    .map(|i| format!("a{i}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let parsed = ruff_python_parser::parse_expression(&source)
                .expect("bound method arguments parse");
            let py::Expr::Call(c) = &*parsed.into_syntax().body else {
                unreachable!()
            };
            match receiver_ty {
                Ty::List(e) => {
                    let ty = match method {
                        "index" | "count" => Ty::Int,
                        "copy" => Ty::List(e),
                        _ => Ty::None,
                    };
                    child.method(held, method, &c.arguments.args, ty, span)?
                }
                Ty::Class(k) => {
                    let (sig, _) = child
                        .module
                        .method_sig(k as usize, method)
                        .expect("found at binding");
                    let mut args = Vec::new();
                    for (expr, (_, ty)) in c.arguments.args.iter().zip(sig.params.iter().skip(1)) {
                        args.push(child.expr_as(expr, *ty)?);
                    }
                    child
                        .invoke(k as usize, method, held.node, args, span)
                        .expect("found at binding")
                }
                _ => unreachable!(),
            }
        };
        let boxed = child.coerce(result, Ty::Object);
        body.append(&mut child.hoisted);
        body.push(TypedNode::new(
            TypedStatement::Return(Some(Box::new(boxed))),
            Type::Unknown,
            span,
        ));
        let mut function = child.lifted_function(&code, &params, body, span);
        if variadic {
            function.params.push(parameter("a0", Ty::Object, span));
        }
        self.module.lifted.borrow_mut().push(function);
        let record_arity = arity;
        let cell = self.coerce(receiver, Ty::Object);
        let cells = self.list_of(
            vec![Val {
                node: cell,
                ty: Ty::Object,
            }],
            Elem::Object,
            span,
        );
        Ok(Val {
            node: call(
                "zb_func_new",
                vec![code_of(&code, span), int_lit(record_arity, span), cells],
                Ty::Object,
                span,
            ),
            ty: Ty::Object,
        })
    }

    /// `obj.attr = value` as a statement expression.
    fn set_attribute(&mut self, object: Val, attr: &str, value: Val, span: Span) -> Result<Node> {
        // The field may now hold anything, on this object or another
        // that aliases it.
        self.nonnull_fields.retain(|(_, field)| field != attr);
        let object = if object.ty == Ty::None {
            Val {
                node: self.coerce(object, Ty::Object),
                ty: Ty::Object,
            }
        } else {
            object
        };
        match object.ty {
            Ty::Class(k) => {
                let object = self.checked_instance(object, attr, span);
                let Some((_, ty)) = self.module.field(k as usize, attr) else {
                    return Err(Error::unsupported_span(
                        format!(
                            "attribute `{attr}` of {}, which has no such field",
                            self.module.classes[k as usize].name
                        ),
                        span,
                    ));
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
            other => Err(Error::unsupported_span(
                format!("assignment to attribute `{attr}` of a {other:?}"),
                span,
            )),
        }
    }

    /// The call of `method` on an instance of class `k`, through the
    /// dispatcher when a subclass overrides it. `args` are already
    /// coerced to the parameter types.
    /// The function a call of `method` on class `k` reaches: the method
    /// itself, or its dispatcher when a subclass overrides it.
    fn invoke_target(&self, k: usize, method: &str) -> Option<String> {
        let owner = self.module.method_owner(k, method)?;
        let (_, fn_name) = self.module.method_sig(k, method)?;
        // A constructor is never dispatched: it is called by its class.
        Some(
            if method == "__init__" || self.module.overriders(owner, method).is_empty() {
                fn_name
            } else {
                dispatch_name(&fn_name)
            },
        )
    }

    pub(crate) fn invoke(
        &mut self,
        k: usize,
        method: &str,
        receiver: Node,
        args: Vec<Node>,
        span: Span,
    ) -> Option<Val> {
        Some(self.invoke_targeted(k, method, receiver, args, span)?.0)
    }

    /// [`Self::invoke`], also naming the function called, which a
    /// check after the call is attributed to.
    fn invoke_targeted(
        &mut self,
        k: usize,
        method: &str,
        receiver: Node,
        args: Vec<Node>,
        span: Span,
    ) -> Option<(Val, String)> {
        let (sig, fn_name) = self.module.method_sig(k, method)?;
        let params = without_self(sig).params;
        let owner = self.module.method_owner(k, method)?;
        let target = self.invoke_target(k, method)?;
        // A dispatched call returns what any method it may reach does.
        let ret = if target == fn_name {
            sig.ret
        } else {
            self.module.dispatched_ret(k, method).unwrap_or(sig.ret)
        };
        // A method nothing overrides may go to its trusted variant.
        let target = if target == fn_name {
            self.call_target(&fn_name, &params, &args)
        } else {
            target
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
        Some((
            Val {
                node: call(&target, all, ret, span),
                ty: ret,
            },
            target,
        ))
    }

    /// A dunder method call on typed operands, when the class chain
    /// defines it; checked for a raise like any other call.
    pub(crate) fn dunder(
        &mut self,
        k: usize,
        method: &str,
        receiver: Node,
        args: Vec<Val>,
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
            .map(|(a, t)| self.coerce(a, t))
            .collect();
        let (v, target) = self.invoke_targeted(k, method, receiver, args, span)?;
        if !self.guards {
            return Some(v);
        }
        Some(self.guard_named(v, &target, span))
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
        self.method_call(k, receiver, method, args, keywords, c, span, true)
    }

    /// `Class.m(obj, args)`: the class's own method, whatever `obj`'s
    /// class overrides.
    #[allow(clippy::too_many_arguments)]
    fn class_method_on(
        &mut self,
        k: usize,
        receiver: Val,
        method: &str,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        self.method_call(k, receiver, method, args, keywords, c, span, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn method_call(
        &mut self,
        k: usize,
        receiver: Val,
        method: &str,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
        dispatched: bool,
    ) -> Result<Val> {
        let Some((sig, fn_name)) = self.module.method_sig(k, method) else {
            return Err(Error::unsupported_span(
                format!(
                    "method `{method}` of {}, which defines none",
                    self.module.classes[k].name
                ),
                span,
            ));
        };
        let sig = without_self(sig);
        let receiver = self.checked_instance(receiver, method, span);
        let lowered = self.arguments(method, &sig, args, keywords, c)?;
        if dispatched {
            let (v, target) = self
                .invoke_targeted(k, method, receiver.node, lowered, span)
                .expect("the method was just found");
            return Ok(self.guard_named(v, &target, span));
        }
        let owner = self
            .module
            .method_owner(k, method)
            .expect("the method was just found");
        let target = self.call_target(&fn_name, &sig.params, &lowered);
        let receiver = self.coerce(receiver, Ty::Class(owner as u16));
        let mut all = vec![receiver];
        all.extend(lowered);
        let v = Val {
            node: call(&target, all, sig.ret, span),
            ty: sig.ret,
        };
        Ok(self.guard_named(v, &target, span))
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
        let constructor = new_name(&name);
        let v = Val {
            node: call(&constructor, lowered, Ty::Class(k as u16), span),
            ty: Ty::Class(k as u16),
        };
        Ok(self.guard_named(v, &constructor, span))
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
        Ok(self.guard_named(v, &fn_name, span))
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

    /// A call through a value whose function inference knows: the
    /// record is handed to the function's typed entry along with the
    /// arguments as it declares them, and nothing is boxed. A call that
    /// does not fit the parameters goes through the record, which
    /// reports it as Python does.
    fn call_closure(
        &mut self,
        k: u16,
        callee: Val,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        let info = self.module.closures.borrow()[k as usize].clone();
        if !keywords.is_empty() || args.len() != info.sig.params.len() {
            return self.call_value(callee, args, keywords, c, span);
        }
        // A generator closure: the call starts its fiber, the record
        // being the environment's head.
        if info.sig.ret == Ty::Gen {
            let record = self.coerce(callee, Ty::Object);
            let record = call(
                "zb_unbox_list_raw_any",
                vec![record],
                Ty::List(Elem::Object),
                span,
            );
            let lowered = self.arguments(&info.name, &info.sig, args, keywords, c)?;
            return Ok(self.start_generator_from(
                &info.typed_name(),
                &info.sig,
                record,
                lowered,
                span,
            ));
        }
        let mut lowered = Vec::new();
        if info.captures {
            let record = self.coerce(callee, Ty::Object);
            lowered.push(call(
                "zb_unbox_list_raw_any",
                vec![record],
                Ty::List(Elem::Object),
                span,
            ));
        }
        lowered.extend(self.arguments(&info.name, &info.sig, args, keywords, c)?);
        let typed = info.typed_name();
        let v = Val {
            node: call(&typed, lowered, info.sig.ret, span),
            ty: info.sig.ret,
        };
        Ok(self.guard_named(v, &typed, span))
    }

    /// A function record: the code address, the arity and the cells.
    /// A function record: the code, the arity word, the cells, then the
    /// values of the parameters with defaults, evaluated here once, as
    /// Python evaluates them where the function is defined.
    fn record(
        &mut self,
        code: &str,
        arity: usize,
        mut cells: Vec<Val>,
        defaults: Vec<Val>,
        span: Span,
    ) -> Val {
        let least = arity - defaults.len();
        cells.extend(defaults);
        let cells = self.list_of(cells, Elem::Object, span);
        let word = zyntax_builtins::functions::arity_word(least, arity);
        Val {
            node: call(
                "zb_func_new",
                vec![code_of(code, span), int_lit(word, span), cells],
                Ty::Object,
                span,
            ),
            ty: Ty::Object,
        }
    }

    /// The defaults of a signature, evaluated and boxed, in parameter
    /// order; Python allows them only at the end of the parameters.
    fn default_values(&mut self, sig: &Sig) -> Result<Vec<Val>> {
        let mut out = Vec::new();
        for d in sig.defaults.iter().flatten() {
            out.push(self.expr_as(d, Ty::Object).map(|node| Val {
                node,
                ty: Ty::Object,
            })?);
        }
        Ok(out)
    }

    /// A module function as a value, through an adapter that unboxes
    /// the arguments and boxes the result.
    fn function_value(&mut self, name: &str, span: Span) -> Result<Val> {
        let sig = self.module.funcs[name].clone();
        self.module.adapters.borrow_mut().insert(name.to_string());
        let defaults = self.default_values(&sig)?;
        Ok(self.record(
            &adapter_name(name),
            sig.params.len(),
            Vec::new(),
            defaults,
            span,
        ))
    }

    /// A `def` inside this function: lifted to a function of its own
    /// that takes the record, and bound here as a value.
    fn nested_def(&mut self, f: &py::StmtFunctionDef, span: Span) -> Result<Val> {
        if f.parameters.vararg.is_some() || f.parameters.kwarg.is_some() {
            return unsupported("*args / **kwargs", &*f.parameters);
        }
        let scope = Scope::of_function(f);
        let known = self.closure_info(f.range.start().to_u32());
        let sig = match &known {
            Some(info) => info.sig.clone(),
            None => types::declared_sig(f),
        };
        let (captured, seeds) = self.captures_for(&scope);
        let locals = types::infer_locals_seeded(self.module, &sig, &f.body, &seeds);
        let lifted = match &known {
            Some(info) => info.name.clone(),
            None => self.lifted_name(f.name.as_str()),
        };
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
        // A nested def that yields is a generator function as a module
        // one is, its arguments and cells arriving in the fiber's
        // environment; the record names a starter that builds that
        // environment from a call's arguments and starts the fiber.
        if child.is_generator {
            // Known: the fiber under the typed name, which a direct call
            // starts; the record names the starter. Unknown: the record
            // is all there is.
            let (fiber, starter) = match &known {
                Some(info) => (info.typed_name(), info.name.clone()),
                None => (lifted.clone(), format!("{lifted}$start")),
            };
            let function = child.function_named(f, &fiber)?;
            self.module.lifted.borrow_mut().push(function);
            if child.sig.defaults.iter().any(Option::is_some) {
                return unsupported("a default on a nested generator", &*f.parameters);
            }
            let adapter = self.generator_starter(&starter, &fiber, &child.sig.clone(), span);
            self.module.lifted.borrow_mut().push(adapter);
            let cells = self.cells_of(&captured, span);
            return Ok(self.record(&starter, params.len(), cells, Vec::new(), span));
        }
        let mut body = Vec::new();
        for s in &f.body {
            child.stmt(s, &mut body)?;
        }
        child.defaults_after_cells = captured.len();
        self.lift(&mut child, known.as_ref(), &lifted, &params, body, f)?;
        let cells = self.cells_of(&captured, span);
        let sig = child.sig.clone();
        let defaults = self.default_values(&sig)?;
        Ok(self.record(&lifted, params.len(), cells, defaults, span))
    }

    /// The function a generator's record names: the shape every
    /// function value has, starting the fiber for `code` with the
    /// record's cells and the call's arguments in its environment.
    fn generator_starter(
        &mut self,
        name: &str,
        code: &str,
        sig: &Sig,
        span: Span,
    ) -> TypedFunction {
        // A check that fails leaves with a placeholder of this shape.
        let ret = std::mem::replace(&mut self.sig.ret, Ty::Object);
        let mut typed_params = vec![parameter("env", Ty::List(Elem::Object), span)];
        let mut statements = Vec::new();
        let mut args = Vec::new();
        for (i, (_, declared)) in sig.params.iter().enumerate() {
            let arg = format!("a{i}");
            typed_params.push(parameter(&arg, Ty::Object, span));
            let value = self.coerce(
                Val {
                    node: var(intern(&arg), Ty::Object, span),
                    ty: Ty::Object,
                },
                *declared,
            );
            statements.append(&mut self.hoisted);
            args.push(value);
        }
        let record = var(intern("env"), Ty::List(Elem::Object), span);
        let started = self.start_generator_from(code, sig, record, args, span);
        let boxed = self.coerce(started, Ty::Object);
        statements.append(&mut self.hoisted);
        statements.push(TypedNode::new(
            TypedStatement::Return(Some(Box::new(boxed))),
            Type::Unknown,
            span,
        ));
        self.sig.ret = ret;
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

    /// A lambda: a lifted function returning its one expression.
    fn lambda(&mut self, l: &py::ExprLambda, span: Span) -> Result<Val> {
        let scope = Scope::of_lambda(l);
        let known = self.closure_info(l.range.start().to_u32());
        let mut params: Vec<(String, Ty)> = Vec::new();
        if let Some(ps) = &l.parameters {
            if ps.vararg.is_some() || ps.kwarg.is_some() {
                return unsupported("*args / **kwargs", &**ps);
            }
            for p in ps.iter_non_variadic_params() {
                params.push((p.parameter.name.to_string(), Ty::Object));
            }
        }
        let sig = match &known {
            Some(info) => info.sig.clone(),
            None => Sig {
                params: params.clone(),
                ret: Ty::Object,
                defaults: l
                    .parameters
                    .as_ref()
                    .map(|ps| {
                        ps.iter_non_variadic_params()
                            .map(|p| p.default.as_deref().cloned())
                            .collect()
                    })
                    .unwrap_or_default(),
            },
        };
        let params = sig.params.clone();
        let (captured, seeds) = self.captures_for(&scope);
        let mut locals = Locals::default();
        for (name, ty) in &params {
            locals.vars.insert(name.clone(), *ty);
        }
        let lifted = match &known {
            Some(info) => info.name.clone(),
            None => self.lifted_name("lambda"),
        };
        let ret = sig.ret;
        let mut child = Lowerer::new(
            self.module,
            &lifted,
            sig,
            locals,
            &scope,
            captured.clone(),
            seeds,
        );
        let value = child.expr_as(&l.body, ret)?;
        let mut body = std::mem::take(&mut child.hoisted);
        body.push(TypedNode::new(
            TypedStatement::Return(Some(Box::new(value))),
            Type::Unknown,
            span,
        ));
        child.defaults_after_cells = captured.len();
        self.lift(&mut child, known.as_ref(), &lifted, &params, body, l)?;
        let cells = self.cells_of(&captured, span);
        let sig = child.sig.clone();
        let defaults = self.default_values(&sig)?;
        Ok(self.record(&lifted, params.len(), cells, defaults, span))
    }

    /// What inference knows of the closure defined at `start` of the
    /// current file, if it is one.
    fn closure_info(&self, start: u32) -> Option<types::ClosureInfo> {
        let k = self.module.closure_at(current_file(), start)?;
        self.module.closures.borrow().get(k as usize).cloned()
    }

    /// Declare a nested body's functions: for a closure inference knows,
    /// its typed entry and the adapter in front of it that the record
    /// names; otherwise the one function every function value has.
    fn lift(
        &mut self,
        child: &mut Lowerer<'_>,
        known: Option<&types::ClosureInfo>,
        lifted: &str,
        params: &[(String, Ty)],
        body: Vec<Stmt>,
        at: &dyn Ranged,
    ) -> Result<()> {
        let span = span_of(&at.range());
        match known {
            Some(info) => {
                if !info.captures && !child.captured.is_empty() {
                    return unsupported(
                        format!(
                            "internal: `{lifted}` captures {} where inference saw no capture",
                            child.captured.join(", ")
                        ),
                        &at.range(),
                    );
                }
                // The body's raising is the typed entry's; the adapter
                // built next adds its own checked reads.
                self.module
                    .raise_facts
                    .borrow_mut()
                    .insert(info.typed_name(), child.raise_fact());
                let typed =
                    child.typed_function(&info.typed_name(), params, body, info.captures, span);
                let adapter =
                    child.adapter_function(lifted, &info.typed_name(), params, info.captures, span);
                let mut lifted = self.module.lifted.borrow_mut();
                lifted.push(typed);
                lifted.push(adapter);
            }
            None => {
                let function = child.lifted_function(lifted, params, body, span);
                self.module.lifted.borrow_mut().push(function);
            }
        }
        Ok(())
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

    /// A closure's typed entry: the record when the body reads the
    /// cells in it, then each parameter as inference typed it, to the
    /// result as inference typed it.
    fn typed_function(
        &mut self,
        name: &str,
        params: &[(String, Ty)],
        body: Vec<Stmt>,
        with_env: bool,
        span: Span,
    ) -> TypedFunction {
        let mut typed_params = Vec::new();
        if with_env {
            typed_params.push(parameter("env", Ty::List(Elem::Object), span));
        }
        let mut statements = Vec::new();
        for (pname, declared) in params {
            typed_params.push(parameter(pname, *declared, span));
            // A parameter the body also assigns another type to lives as
            // an object from the start.
            let local = self.var_ty(pname);
            if local != *declared {
                let value = self.coerce(
                    Val {
                        node: var(intern(pname), *declared, span),
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
        }
        statements.extend(self.cell_prologue(span));
        statements.extend(body);
        // Falling off the end returns None.
        let none = Val {
            node: node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
            ty: Ty::None,
        };
        let none = self.coerce(none, self.sig.ret);
        statements.append(&mut self.hoisted);
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
        }
    }

    /// The adapter a record names for a closure with a typed entry: the
    /// shape every function value has, each argument read out of its
    /// box as the entry's parameter type, the result boxed.
    fn adapter_function(
        &mut self,
        name: &str,
        typed: &str,
        params: &[(String, Ty)],
        with_env: bool,
        span: Span,
    ) -> TypedFunction {
        // A check that fails leaves with a placeholder of this shape.
        let ret = std::mem::replace(&mut self.sig.ret, Ty::Object);
        let mut typed_params = vec![parameter("env", Ty::List(Elem::Object), span)];
        let mut statements = Vec::new();
        let mut args = Vec::new();
        if with_env {
            args.push(var(intern("env"), Ty::List(Elem::Object), span));
        }
        for (i, (_, declared)) in params.iter().enumerate() {
            let arg = format!("a{i}");
            typed_params.push(parameter(&arg, Ty::Object, span));
            let value = self.coerce(
                Val {
                    node: var(intern(&arg), Ty::Object, span),
                    ty: Ty::Object,
                },
                *declared,
            );
            statements.append(&mut self.hoisted);
            args.push(value);
        }
        let result = Val {
            node: call(typed, args, ret, span),
            ty: ret,
        };
        let boxed = self.coerce(result, Ty::Object);
        statements.append(&mut self.hoisted);
        statements.push(TypedNode::new(
            TypedStatement::Return(Some(Box::new(boxed))),
            Type::Unknown,
            span,
        ));
        self.sig.ret = ret;
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

    /// This function in the shape every function value has: the record
    /// first, each argument a dynamic value unboxed to its declared type
    /// in a prologue, the result boxed.
    /// `given` unless it is the missing-argument marker, then the record's
    /// element at `slot`.
    fn or_default(&mut self, given: Node, slot: usize, span: Span) -> Node {
        let missing = call(
            "zb_any_same",
            vec![
                given.clone(),
                call("zb_missing_arg", vec![], Ty::Object, span),
            ],
            Ty::Bool,
            span,
        );
        let kept = call(
            "zb_list_get_any",
            vec![
                var(intern("env"), Ty::List(Elem::Object), span),
                int_lit(slot as i64, span),
            ],
            Ty::Object,
            span,
        );
        node(
            TypedExpression::If(TypedIfExpr {
                condition: Box::new(missing),
                then_branch: Box::new(kept),
                else_branch: Box::new(given),
            }),
            Ty::Object,
            span,
        )
    }

    fn lifted_function(
        &mut self,
        name: &str,
        params: &[(String, Ty)],
        body: Vec<Stmt>,
        span: Span,
    ) -> TypedFunction {
        let mut typed_params = vec![parameter("env", Ty::List(Elem::Object), span)];
        let mut statements = Vec::new();
        let first_default = params.len() - self.sig.defaults.iter().flatten().count();
        for (i, (pname, declared)) in params.iter().enumerate() {
            let arg = format!("a{i}");
            typed_params.push(parameter(&arg, Ty::Object, span));
            let local = self.var_ty(pname);
            let given = var(intern(&arg), Ty::Object, span);
            // An argument left out arrives as the marker and takes the
            // default the record keeps after the cells.
            let given = if i >= first_default {
                self.or_default(
                    given,
                    RECORD_CELLS_AT + self.defaults_after_cells + (i - first_default),
                    span,
                )
            } else {
                given
            };
            let value = self.coerce(
                Val {
                    node: given,
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
            // The checked read's pending check, ahead of the binding.
            statements.append(&mut self.hoisted);
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
            // A nested def that yields is a generator like any other.
            is_fiber: self.is_generator,
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

/// Whether every key of a dict literal is a string literal and no two
/// are equal, so the pairs are the dict's pairs as written.
fn distinct_literal_keys(d: &py::ExprDict) -> bool {
    let mut seen: Vec<String> = Vec::with_capacity(d.items.len());
    for item in &d.items {
        let Some(py::Expr::StringLiteral(s)) = &item.key else {
            return false;
        };
        let text = s.value.to_str().to_string();
        if seen.contains(&text) {
            return false;
        }
        seen.push(text);
    }
    true
}
