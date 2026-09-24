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
use crate::{Error, Result, intern, prim, span_of};
use ruff_python_ast as py;
use ruff_text_size::Ranged;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::collections::BTreeSet;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::typed_ast::{
    ParameterAttribute, TypedAnnotation, TypedBinary, TypedBlock, TypedCall, TypedCast,
    TypedExpression, TypedFieldAccess, TypedFor, TypedFunction, TypedIf, TypedIfExpr, TypedIndex,
    TypedLet, TypedLiteral, TypedMatch, TypedMatchArm, TypedMethodCall, TypedParameter,
    TypedPattern, TypedRange, TypedStatement, TypedUnary, TypedWhile,
};
use zyntax_typed_ast::{
    BinaryOp, InternedString, Mutability, ParamOwnership, ParameterKind, PrimitiveType, Type,
    TypedNode, UnaryOp, Visibility,
};

pub(crate) type Node = TypedNode<TypedExpression>;
pub(crate) type Stmt = TypedNode<TypedStatement>;

pub(crate) fn strict_fp_annotation(span: Span) -> TypedAnnotation {
    TypedAnnotation {
        name: intern("strict_fp"),
        args: Vec::new(),
        span,
    }
}

/// Keep Python module variables separate from the built-in library's
/// constants and globals, which share the compiler's symbol table.
pub(crate) fn global_symbol(name: &str) -> InternedString {
    intern(&format!("py$global${name}"))
}

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

/// The library's `List<T>` type id.
pub(crate) fn list_type_id() -> zyntax_typed_ast::TypeId {
    LIST_TYPE
        .with(|c| c.get())
        .expect("the library's List<T> is known before lowering")
}

/// `List<elem>` as the library declares it.
fn list_type(elem: Type) -> Type {
    zyntax_builtins::list_of(list_type_id(), elem)
}

/// How a tuple stores a field of type `ty`: a scalar, an instance or a
/// nested tuple as itself; a list, dict or set boxed, since their
/// headers have an identity a copy inside the tuple would lose; anything
/// dynamic as the box it is.
pub(crate) fn tuple_field_storage(ty: Ty) -> Ty {
    match ty {
        Ty::Int | Ty::Float | Ty::Bool | Ty::Str | Ty::Class(_) | Ty::Tuple(_) => ty,
        _ => Ty::Object,
    }
}

/// How the library compares, prints, boxes and reads back a tuple's
/// field of type `ty`, stored as [`tuple_field_storage`] says.
fn field_of(ty: Ty) -> zyntax_builtins::lists::Field {
    use zyntax_builtins::lists::Field;
    match ty {
        Ty::Int => Field::Int,
        Ty::Float => Field::Float,
        Ty::Bool => Field::Bool,
        Ty::Str => Field::Str,
        Ty::Dict(_) => Field::Dict { ty: ir(ty) },
        Ty::Set => Field::Set { ty: ir(ty) },
        Ty::Class(k) => Field::Instance {
            ty: class_type(k as usize),
            tag: zyntax_builtins::instance_tag(k as usize) as i32,
        },
        Ty::List(e @ Elem::Array(c)) => {
            types::note_array_kind(c.storage());
            Field::Array {
                suffix: e.suffix(),
                ty: ir(ty),
                tag: c.tag(),
                letter: c.letter().to_string(),
            }
        }
        Ty::List(e) => Field::List {
            suffix: e.suffix(),
            ty: ir(ty),
        },
        Ty::Tuple(k) => Field::Tuple {
            suffix: types::tuple_suffix(k),
            ty: ir(ty),
        },
        Ty::None
        | Ty::Bytes
        | Ty::File(_)
        | Ty::Gen
        | Ty::Closure(_)
        | Ty::Bound(_)
        | Ty::Builtin(_)
        | Ty::Object
        | Ty::Unknown => Field::Any,
    }
}

/// The library functions of the program's tuple shapes: every shape gets
/// its own (equality, order, repr, boxing, reading back), and a shape
/// the lowering used as a list's element gets the list functions too.
/// Shapes are declared in interning order, which puts a shape after the
/// shapes its fields name.
pub(crate) fn shape_declarations(
    module: &Module,
    list_type: zyntax_typed_ast::TypeId,
) -> Vec<TypedNode<zyntax_typed_ast::typed_ast::TypedDeclaration>> {
    let _ = module;
    let lists = types::tuple_lists();
    let mut out = Vec::new();
    // The storage kinds the library does not carry, before anything
    // that calls their functions.
    for kind in types::array_kinds() {
        if !zyntax_builtins::Kind::LIBRARY.contains(&kind) {
            out.extend(zyntax_builtins::lists::array_kind_declarations(
                kind, list_type,
            ));
        }
    }
    for k in 0..types::tuple_shape_count() as u16 {
        let suffix = types::tuple_suffix(k);
        let tuple_ty = ir(Ty::Tuple(k));
        let fields: Vec<zyntax_builtins::lists::Field> = types::tuple_shape(k)
            .into_iter()
            .map(|t| field_of(t.settled()))
            .collect();
        out.extend(zyntax_builtins::lists::tuple_declarations(
            list_type,
            &suffix,
            tuple_ty.clone(),
            &fields,
        ));
        if lists.contains(&k) {
            out.extend(zyntax_builtins::lists::tuple_list_declarations(
                list_type, k, &suffix, tuple_ty,
            ));
        }
    }
    out
}

/// The IR type a static type is carried as.
pub(crate) fn ir(ty: Ty) -> Type {
    match ty {
        Ty::Int => prim(PrimitiveType::I64),
        Ty::Float => prim(PrimitiveType::F64),
        Ty::Bool => prim(PrimitiveType::Bool),
        Ty::Str | Ty::Bytes => prim(PrimitiveType::String),
        Ty::None => prim(PrimitiveType::Unit),
        Ty::List(e) => list_type(elem_ir(e)),
        // A shape is a value struct with one field per element, each
        // stored as `tuple_field_storage` says.
        Ty::Tuple(k) => Type::Tuple(
            types::tuple_shape(k)
                .into_iter()
                .map(|t| ir(tuple_field_storage(t)))
                .collect(),
        ),
        Ty::Dict(_) | Ty::Set => list_type(Type::Any),
        // A file is the record the library keeps: a list of its parts.
        Ty::File(_) => list_type(Type::Any),
        Ty::Class(k) => class_type(k as usize),
        Ty::Gen => Type::Fiber(Box::new(Type::Any)),
        // A known function value is still the record every function
        // value is.
        Ty::Closure(_) | Ty::Bound(_) | Ty::Builtin(_) | Ty::Object | Ty::Unknown => Type::Any,
    }
}

/// The IR type a list of `e` holds per element: an instance is held by
/// address, an array's element at its typecode's width, everything
/// else as itself.
pub(crate) fn elem_ir(e: Elem) -> Type {
    match e {
        Elem::Class(_) => addr_type(),
        Elem::Tuple(k) => {
            types::note_tuple_list(k);
            ir(Ty::Tuple(k))
        }
        Elem::Array(c) => c.storage().ty(),
        other => ir(other.ty()),
    }
}

/// A call whose result has IR type `ty`.
fn typed_call(name: &str, args: Vec<Node>, ty: Type, span: Span) -> Node {
    TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(var(intern(name), Ty::Unknown, span)),
            positional_args: args,
            named_args: Vec::new(),
            type_args: Vec::new(),
        }),
        ty,
        span,
    )
}

/// A call to a list function that returns an element of kind `e`,
/// typed as the element: an address comes back as the instance, an
/// array's element widened to the number it reads as.
fn elem_call(op: &str, e: Elem, args: Vec<Node>, span: Span) -> Node {
    match e {
        Elem::Class(k) => cast(addr_call(&list_fn(op, e), args, span), Ty::Class(k), span),
        Elem::Array(c) => {
            let stored = typed_call(&list_fn(op, e), args, c.storage().ty(), span);
            if c.narrows() {
                cast(stored, c.item(), span)
            } else {
                stored
            }
        }
        other => call(&list_fn(op, other), args, other.ty(), span),
    }
}

pub(crate) fn method_call(
    receiver: Node,
    method: &str,
    args: Vec<Node>,
    ty: Ty,
    span: Span,
) -> Node {
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

fn bind_names(vars: &mut HashMap<String, Ty>, target: &py::Expr, ty: Ty) {
    types::bind_target(vars, target, ty)
}

/// `zb_list_<op>_<kind>`. A list of tuples has its functions generated
/// for the shape, and an array's storage kind its functions and its
/// hook arms, so both are noted.
pub(crate) fn list_fn(op: &str, elem: Elem) -> String {
    match elem {
        Elem::Tuple(k) => types::note_tuple_list(k),
        Elem::Array(c) => types::note_array_kind(c.storage()),
        _ => {}
    }
    format!("zb_list_{op}_{}", elem.suffix())
}

/// Element `i` of a list the lowering built itself and so knows the
/// length of: a cell, a record. Nothing checks the index. On a tuple
/// value the same expression is the field at `i`.
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

/// A tuple value of shape `ty` from its elements, each already of the
/// shape's element type and stored as `tuple_field_storage` says.
fn tuple_value(items: Vec<Node>, ty: Ty, span: Span) -> Node {
    node(TypedExpression::Tuple(items), ty, span)
}

/// Where the hold of a value read more than once goes.
enum Hold<'a> {
    /// Into this statement list, which the reads follow.
    Into(&'a mut Vec<Stmt>),
    /// Ahead of the statement, with everything else the expression
    /// hoists, in the order it is emitted: a hold of a part comes after
    /// the hold of its whole. Where nothing hoists, back to the caller,
    /// to put in front of the reads.
    Ahead,
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
fn is_generator_start(r#gen: &Node) -> bool {
    matches!(
        &r#gen.node,
        TypedExpression::Call(c)
            if matches!(&c.callee.node, TypedExpression::Variable(v)
                if v.resolve_global().as_deref() == Some("zb_fiber_start"))
    )
}

/// `zb_fiber_free(r#gen)`: a drained generator's fiber released.
fn free_generator(r#gen: Node, span: Span) -> Stmt {
    TypedNode::new(
        TypedStatement::Expression(Box::new(call("zb_fiber_free", vec![r#gen], Ty::None, span))),
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
/// The name of the dispatcher for a method of `class` that only its
/// subclasses define.
pub(crate) fn abstract_name(class: &str, method: &str) -> String {
    format!("{class}${method}$abstract")
}

/// The name of the function a class is called through as a value.
pub(crate) fn class_adapter_name(class: &str) -> String {
    format!("{class}$value")
}

/// `C$value(env, a0, ...)`: the class constructed from boxed arguments,
/// the instance boxed. What a class is as a value.
pub(crate) fn class_adapter(module: &Module, k: usize) -> TypedFunction {
    let span = Span::new(0, 0);
    let scope = Scope::default();
    let class = &module.classes[k];
    let sig = match module.method_sig(k, "__init__") {
        Some((sig, _)) => without_self(sig),
        None => Sig {
            params: Vec::new(),
            ret: Ty::None,
            defaults: Vec::new(),
        },
    };
    let mut lowerer = Lowerer::new(
        module,
        &class_adapter_name(&class.name),
        sig.clone(),
        Locals::default(),
        &scope,
        Vec::new(),
        HashMap::default(),
    );
    lowerer.guards = false;
    let mut params = vec![parameter("env", Ty::List(Elem::Object), span)];
    let mut args = Vec::new();
    let first_default = sig.params.len() - sig.defaults.iter().flatten().count();
    for (i, (_, ty)) in sig.params.iter().enumerate() {
        let arg = format!("a{i}");
        params.push(parameter(&arg, Ty::Object, span));
        let given = var(intern(&arg), Ty::Object, span);
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
    let instance = Val {
        node: call(&new_name(&class.name), args, Ty::Class(k as u16), span),
        ty: Ty::Class(k as u16),
    };
    let boxed = lowerer.coerce(instance, Ty::Object);
    let statements = vec![TypedNode::new(
        TypedStatement::Return(Some(Box::new(boxed))),
        Type::Unknown,
        span,
    )];
    TypedFunction {
        name: intern(&class_adapter_name(&class.name)),
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
        HashMap::default(),
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

pub(crate) fn unsupported<T>(what: impl Into<String>, at: &impl Ranged) -> Result<T> {
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
    /// Compiler names for comprehension targets, scoped to their expression.
    comp_symbols: HashMap<String, InternedString>,
    /// Statements an expression needs run before the statement it is
    /// part of: a comprehension's loop, which the IR cannot hold inside
    /// an expression. Drained in front of each statement.
    pub(crate) hoisted: Vec<Stmt>,
    /// Class-typed variables known not to be None where the lowering
    /// stands: assigned from a constructor, or checked since. A
    /// compound statement keeps only those it does not assign.
    nonnull: HashSet<InternedString>,
    /// Parameters the body assigns somewhere: what the caller passed is
    /// not what they hold from then on, so neither `self` nor a trusted
    /// instance parameter is one once the body assigns it.
    reassigned_params: HashSet<InternedString>,
    /// Variables that hold an instance for the whole function: assigned
    /// a constructor's result before anything reads them, and assigned
    /// nothing else anywhere; see [`Self::always_instances`].
    always_instance: HashSet<InternedString>,
    /// Fields `v.f` of a known instance `v` known not to be None where
    /// the lowering stands, from a test the control flow has settled.
    /// Cleared with `nonnull`, at any call (which may store to the
    /// field), and at a store to a field of that name.
    nonnull_fields: HashSet<(InternedString, String)>,
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
        // Assigned by this body, or shared with a nested one that may.
        let reassigned_params = sig
            .params
            .iter()
            .filter(|(n, _)| scope.bound.contains(n) || cells.contains(n))
            .map(|(n, _)| intern(n))
            .collect();
        Self {
            module,
            name: name.to_string(),
            sig,
            locals,
            bound,
            temps: 0,
            comp_symbols: HashMap::default(),
            hoisted: Vec::new(),
            nonnull: HashSet::default(),
            reassigned_params,
            always_instance: HashSet::default(),
            nonnull_fields: HashSet::default(),
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
        if let Some(ctl) = self.try_ctls.last()
            && ctl.loop_depth == 0
        {
            let flag = ctl.flag;
            self.set_flag_and_leave(flag, code, span, out);
            return;
        }
        if code == 2
            && let Some(Some(flag)) = self.loop_elses.last()
        {
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

    /// An expression Python always raises at: the operands are
    /// evaluated for their effects, the exception is raised, and the
    /// value stands for nothing.
    fn raised_value(
        &mut self,
        operands: Vec<Node>,
        class: &str,
        message: &str,
        ty: Ty,
        span: Span,
    ) -> Node {
        let mut pre: Vec<Stmt> = operands
            .into_iter()
            .map(|n| TypedNode::new(TypedStatement::Expression(Box::new(n)), Type::Unknown, span))
            .collect();
        self.raise_named(class, str_lit(message, span), span, &mut pre);
        self.hoisted.extend(pre);
        self.zero_of(ty, span)
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
            let elem = Elem::of(
                self.typer()
                    .comprehension_elem(&g.generators, &g.elt)
                    .settled(),
            );
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
        !self.comp_symbols.contains_key(name)
            && !self.locals.vars.contains_key(name)
            && !self.cells.contains_key(name)
            && self.module.globals.contains_key(name)
    }

    /// Whether `name` is a variable of some kind here, as opposed to a
    /// module function or a builtin.
    fn is_variable(&self, name: &str) -> bool {
        self.comp_symbols.contains_key(name)
            || self.cells.contains_key(name)
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
            // What boxing the value holds runs ahead of the cell.
            out.append(&mut self.hoisted);
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
                node: var(global_symbol(name), stored, span),
                ty: stored,
            },
            ty,
        );
        Val { node, ty }
    }

    /// Read the class attribute `attr`: its constant, or its module
    /// variable.
    fn class_attr_read(&mut self, attr: &crate::class_attrs::ClassAttr, span: Span) -> Val {
        use crate::class_attrs::Constant as C;
        match &attr.constant {
            Some(c) => {
                let (lit, ty) = match c {
                    C::Int(i) => (TypedLiteral::Integer(*i as i128), Ty::Int),
                    C::Float(f) => (TypedLiteral::Float(*f), Ty::Float),
                    C::Bool(b) => (TypedLiteral::Bool(*b), Ty::Bool),
                    C::Str(text) => {
                        return Val {
                            node: str_lit(text, span),
                            ty: Ty::Str,
                        };
                    }
                    C::None => (TypedLiteral::Null, Ty::None),
                };
                Val {
                    node: node(TypedExpression::Literal(lit), ty, span),
                    ty,
                }
            }
            None => {
                let global = attr.global.clone();
                self.global_read(&global, span)
            }
        }
    }

    /// The class attribute `attr` of class `k`, or an error naming what
    /// the class lacks.
    fn class_attr_of(
        &self,
        k: usize,
        attr: &str,
        span: Span,
    ) -> Result<crate::class_attrs::ClassAttr> {
        self.module.class_attr(k, attr).cloned().ok_or_else(|| {
            Error::unsupported_span(
                format!(
                    "attribute `{attr}` of {}, which has no such field or class attribute",
                    self.module.classes[k].name
                ),
                span,
            )
        })
    }

    /// Store `value` in the module variable `name`, typed `ty`.
    fn store_global(&mut self, name: &str, value: Val, ty: Ty, span: Span, out: &mut Vec<Stmt>) {
        let stored = Self::storage(ty);
        let value = self.coerce(value, ty);
        let value = self.coerce(Val { node: value, ty }, stored);
        let assign = binary(
            BinaryOp::Assign,
            var(global_symbol(name), stored, span),
            value,
            Ty::None,
            span,
        );
        out.push(TypedNode::new(
            TypedStatement::Expression(Box::new(assign)),
            Type::Unknown,
            span,
        ));
    }

    /// A local no Python program can spell.
    pub(crate) fn temp(&mut self) -> InternedString {
        self.temps += 1;
        intern(&format!("__tmp{}", self.temps))
    }

    pub(crate) fn local_symbol(&self, name: &str) -> InternedString {
        self.comp_symbols
            .get(name)
            .copied()
            .unwrap_or_else(|| intern(name))
    }

    // ─── Functions ──────────────────────────────────────────────────

    /// A `def` as a function, under `name`: its own for a module
    /// function, `Class$m` for a method.
    /// A function of this signature whose body raises `message` as a
    /// TypeError: what stands in for a method the frontend cannot
    /// compile and the program never names, so that calling it reports
    /// the form rather than the program failing to build.
    pub(crate) fn stub_function(&mut self, name: &str, message: &str, span: Span) -> TypedFunction {
        let params = self
            .sig
            .params
            .iter()
            .map(|(p, ty)| parameter(p, *ty, span))
            .collect();
        let mut statements = Vec::new();
        self.raise_named("TypeError", str_lit(message, span), span, &mut statements);
        let ret = self.sig.ret;
        let value = if ret == Ty::None {
            None
        } else {
            Some(Box::new(self.zero_of(ret, span)))
        };
        statements.push(TypedNode::new(
            TypedStatement::Return(value),
            Type::Unknown,
            span,
        ));
        TypedFunction {
            name: intern(name),
            annotations: vec![strict_fp_annotation(span)],
            effects: Vec::new(),
            with_handlers: Vec::new(),
            type_params: Vec::new(),
            params,
            return_type: ir(ret),
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
                // What the conversion holds runs ahead of the binding.
                prologue.append(&mut self.hoisted);
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
            annotations: vec![strict_fp_annotation(span)],
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
            // An array is boxed by reference under a tag of its typecode,
            // and read back by checking that tag: a list stored the same
            // way is not an array, nor is an array of another typecode.
            (Ty::List(e @ Elem::Array(c)), Ty::Object) => call(
                &list_fn("box_tagged", e),
                vec![v.node, int_lit(c.tag(), span)],
                Ty::Object,
                span,
            ),
            (Ty::Object, Ty::List(e @ Elem::Array(c))) => {
                let checked = Val {
                    node: call(
                        &list_fn("unbox_tagged", e),
                        vec![v.node, int_lit(c.tag(), span), str_lit(c.letter(), span)],
                        target,
                        span,
                    ),
                    ty: target,
                };
                if self.guards {
                    self.guard(checked, span).node
                } else {
                    checked.node
                }
            }
            // A list is boxed by reference under a tag of its kind, and
            // read back by checking that tag.
            (Ty::List(e), Ty::Object) => call(&list_fn("box", e), vec![v.node], Ty::Object, span),
            // A tuple is boxed as the tagged list of its boxed elements,
            // and read back field by field once the tag and the length
            // are checked.
            (Ty::Tuple(_), Ty::Object) => self.box_tuple(v, span),
            (Ty::Object, Ty::Tuple(k)) => self.unbox_tuple(v, k, false, span),
            // Between two shapes of one arity, element by element; of
            // different arities, through the box, whose length check
            // is the ValueError.
            (Ty::Tuple(a), Ty::Tuple(k)) => {
                let to = types::tuple_shape(k);
                if types::tuple_shape(a).len() != to.len() {
                    let boxed = self.box_tuple(v, span);
                    return self.unbox_tuple(
                        Val {
                            node: boxed,
                            ty: Ty::Object,
                        },
                        k,
                        false,
                        span,
                    );
                }
                let (fields, pre) = self.tuple_fields(v, span, Hold::Ahead);
                let value = self.tuple_of_items(fields, target, span);
                if pre.is_empty() {
                    value
                } else {
                    Self::block_value(pre, value, target, span)
                }
            }
            (Ty::Dict(_), Ty::Object) => call("zb_dict_box", vec![v.node], Ty::Object, span),
            // Every dict shape is stored the same way.
            (Ty::Dict(_), Ty::Dict(_)) => v.node,
            (Ty::Set, Ty::Object) => call("zb_set_box", vec![v.node], Ty::Object, span),
            // Bytes box under their own category, so a box of text is
            // never read as them; a file is boxed as the list it is.
            (Ty::Bytes, Ty::Object) => call("zb_box_bytes", vec![v.node], Ty::Object, span),
            (Ty::Object, Ty::Bytes) => {
                let checked = Val {
                    node: call("zb_any_as_bytes", vec![v.node], target, span),
                    ty: target,
                };
                if self.guards {
                    self.guard(checked, span).node
                } else {
                    checked.node
                }
            }
            (Ty::File(_), Ty::File(_)) => v.node,
            (Ty::File(_), Ty::Object) => call("zb_file_box", vec![v.node], Ty::Object, span),
            (Ty::Object, Ty::File(_)) => {
                let checked = Val {
                    node: call("zb_file_unbox", vec![v.node], target, span),
                    ty: target,
                };
                if self.guards {
                    self.guard(checked, span).node
                } else {
                    checked.node
                }
            }
            // Bytes iterate as their values.
            (Ty::Bytes, Ty::List(Elem::Int)) => {
                call("zb_bytes_to_list", vec![v.node], target, span)
            }
            (Ty::Bytes, Ty::List(Elem::Object)) => {
                let ints = call("zb_bytes_to_list", vec![v.node], Ty::List(Elem::Int), span);
                call(&list_fn("to_any", Elem::Int), vec![ints], target, span)
            }
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
            (Ty::Object, Ty::Dict(_)) => call("zb_dict_unbox", vec![v.node], target, span),
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
            // An instance is boxed as its address under its class's tag,
            // and read back with a check; a subclass instance is its base.
            // A null instance boxes as None. A value typed as a class with
            // subclasses may be any of them, so its tag is read from it.
            (Ty::Class(k), Ty::Object) => {
                let address = as_addr(v.node, span);
                if self.module.classes[k as usize].descendants > 1 {
                    call("zb_hook_box_instance", vec![address], Ty::Object, span)
                } else {
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
            (Ty::Dict(_), Ty::List(Elem::Object)) => {
                call("zb_dict_keys", vec![v.node], target, span)
            }
            // A dict's keys as a list of the kind they are.
            (Ty::Dict(_), Ty::List(e)) => {
                let keys = call("zb_dict_keys", vec![v.node], Ty::List(Elem::Object), span);
                self.coerce(
                    Val {
                        node: keys,
                        ty: Ty::List(Elem::Object),
                    },
                    Ty::List(e),
                )
            }
            (Ty::Gen, Ty::List(Elem::Object)) => self.generator_to_list(v.node, span),
            (Ty::Set, Ty::List(Elem::Object)) => call("zb_set_items", vec![v.node], target, span),
            // Lists of one kind into lists of dynamic values.
            (Ty::List(e), Ty::List(Elem::Object)) => {
                call(&list_fn("to_any", e), vec![v.node], target, span)
            }
            // An array from the numbers it reads as, or into them, or
            // from an array reading as the same: converted directly,
            // each value checked on its way into a narrower typecode.
            (Ty::List(from), Ty::List(to))
                if from.code().is_some_and(|c| Elem::of(c.item()) == to)
                    || to.code().is_some_and(|c| Elem::of(c.item()) == from)
                    || from
                        .code()
                        .zip(to.code())
                        .is_some_and(|(a, b)| a.item() == b.item()) =>
            {
                let mut node = v.node;
                // Out of the array's storage first, when it is narrow.
                if let Some(c) = from.code().filter(|c| c.narrows()) {
                    let wide = Ty::List(Elem::of(c.item()));
                    node = call(&list_fn("to_wide", from), vec![node], wide, span);
                }
                match to.code().filter(|c| c.narrows()) {
                    Some(_) => {
                        let converted = Val {
                            node: call(&list_fn("from_wide", to), vec![node], target, span),
                            ty: target,
                        };
                        if self.guards {
                            self.guard(converted, span).node
                        } else {
                            converted.node
                        }
                    }
                    // The same storage: the copy every conversion is.
                    None => call(&list_fn("copy", to), vec![node], target, span),
                }
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
            // A tuple's elements as a list of the kind wanted.
            (Ty::Tuple(_), Ty::List(e)) => {
                let (fields, pre) = self.tuple_fields(v, span, Hold::Ahead);
                let list = self.list_of(fields, e, span);
                if pre.is_empty() {
                    list
                } else {
                    Self::block_value(pre, list, target, span)
                }
            }
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

    /// [`Self::bind`] of a value that statements already in `out`
    /// produced: whatever the binding hoists is placed in `out` just
    /// ahead of the binding's own statements, since the statement-wide
    /// hoist would run before the value exists.
    pub(crate) fn bind_after(
        &mut self,
        target: &py::Expr,
        value: Val,
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Result<()> {
        let saved = std::mem::take(&mut self.hoisted);
        let at = out.len();
        let result = self.bind(target, value, span, out);
        let hoisted = std::mem::replace(&mut self.hoisted, saved);
        out.splice(at..at, hoisted);
        result
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
            (Ty::Object, Ty::Str | Ty::Bytes) => "zb_box_get_str",
            // The box holds an instance of the class, or is the null box
            // None stores as.
            (Ty::Object, Ty::Class(_)) => {
                let address = addr_call("zb_unbox_instance", vec![v.node], span);
                return cast(address, target, span);
            }
            (Ty::Object, Ty::Tuple(k)) => return self.unbox_tuple(v, k, true, span),
            // The box is known to hold a list of the kind, or the
            // header a dict or set is.
            (Ty::Object, Ty::List(e)) => {
                return call(
                    &format!("zb_unbox_list_raw_{}", e.suffix()),
                    vec![v.node],
                    target,
                    span,
                );
            }
            (Ty::Object, Ty::Dict(_) | Ty::Set | Ty::File(_)) => {
                return call("zb_unbox_list_raw_any", vec![v.node], target, span);
            }
            _ => return self.coerce(v, target),
        };
        call(read, vec![v.node], target, span)
    }

    pub(crate) fn expr_as(&mut self, e: &py::Expr, target: Ty) -> Result<Node> {
        let v = self.expr(e)?;
        Ok(self.coerce(v, target))
    }

    /// A tuple as the list of its elements, of the one kind they all
    /// are when they are; anything else as it is.
    fn tuple_as_list(&mut self, v: Val) -> Val {
        match v.ty {
            Ty::Tuple(_) => {
                let list = Ty::List(Elem::of(v.ty.element().unwrap_or(Ty::Object)));
                Val {
                    node: self.coerce(v, list),
                    ty: list,
                }
            }
            _ => v,
        }
    }

    /// The fields of a tuple value, each as its own type. A tuple read
    /// more than once is held first, where `place` says; a hold that
    /// comes back is for the caller to put in front of the fields' use.
    fn tuple_fields(&mut self, v: Val, span: Span, place: Hold<'_>) -> (Vec<Val>, Vec<Stmt>) {
        let shape = v.ty.tuple_elems().expect("a tuple value");
        let mut pre = Vec::new();
        let held = if shape.len() > 1 {
            self.hold(v, &mut pre, span)
        } else {
            v
        };
        match place {
            Hold::Into(out) => out.append(&mut pre),
            Hold::Ahead if self.guards => self.hoisted.append(&mut pre),
            Hold::Ahead => {}
        }
        let fields = shape
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                let ty = ty.settled();
                let stored = tuple_field_storage(ty);
                let read = Val {
                    node: slot(held.node.clone(), i, stored, span),
                    ty: stored,
                };
                Val {
                    node: self.trusted(read, ty),
                    ty,
                }
            })
            .collect();
        (fields, pre)
    }

    /// The elements of a tuple of shape `ty`, each as the shape's element
    /// type, stored as the shape stores them.
    fn tuple_of_items(&mut self, items: Vec<Val>, ty: Ty, span: Span) -> Node {
        let shape = ty.tuple_elems().expect("a shape");
        let stored: Vec<Node> = items
            .into_iter()
            .zip(shape)
            .map(|(item, elem)| {
                let elem = elem.settled();
                // A box going into a slot that stores boxes keeps its
                // identity: checked, not unboxed and boxed again.
                if item.ty == Ty::Object && tuple_field_storage(elem) == Ty::Object {
                    let check = match elem {
                        Ty::List(e) => Some(list_fn("as_box", e)),
                        Ty::Dict(_) => Some("zb_dict_as_box".to_string()),
                        Ty::Set => Some("zb_set_as_box".to_string()),
                        _ => None,
                    };
                    let Some(check) = check else {
                        return item.node;
                    };
                    let checked = Val {
                        node: call(&check, vec![item.node], Ty::Object, span),
                        ty: Ty::Object,
                    };
                    return if self.guards {
                        self.guard(checked, span).node
                    } else {
                        checked.node
                    };
                }
                let as_elem = Val {
                    node: self.coerce(item, elem),
                    ty: elem,
                };
                self.coerce(as_elem, tuple_field_storage(elem))
            })
            .collect();
        tuple_value(stored, ty, span)
    }

    /// A tuple value as a dynamic value: the tagged list of its boxed
    /// elements.
    fn box_tuple(&mut self, v: Val, span: Span) -> Node {
        let (fields, pre) = self.tuple_fields(v, span, Hold::Ahead);
        let list = self.list_of(fields, Elem::Object, span);
        let boxed = call("zb_box_tuple", vec![list], Ty::Object, span);
        if pre.is_empty() {
            boxed
        } else {
            Self::block_value(pre, boxed, Ty::Object, span)
        }
    }

    /// A dynamic value as a tuple of shape `k`: the box is checked to
    /// hold a tuple of that length, and each element is read back as
    /// the shape says. A `trusted` value is one the lowering boxed
    /// itself, so nothing is checked.
    fn unbox_tuple(&mut self, v: Val, k: u16, trusted: bool, span: Span) -> Node {
        let shape = types::tuple_shape(k);
        let target = Ty::Tuple(k);
        let anys = Ty::List(Elem::Object);
        let read = if trusted {
            "zb_unbox_tuple_raw"
        } else {
            "zb_unbox_tuple"
        };
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: call(read, vec![v.node], anys, span),
                ty: anys,
            },
            &mut pre,
            span,
        );
        if !trusted && self.guards {
            pre.push(self.pending_check(span));
            pre.push(TypedNode::new(
                TypedStatement::Expression(Box::new(call(
                    "zb_list_expect_len_any",
                    vec![held.node.clone(), int_lit(shape.len() as i64, span)],
                    Ty::None,
                    span,
                ))),
                Type::Unknown,
                span,
            ));
            pre.push(self.pending_check(span));
        }
        if self.guards {
            self.hoisted.append(&mut pre);
        }
        let items = shape
            .into_iter()
            .enumerate()
            .map(|(i, ty)| {
                let item = Val {
                    node: call(
                        "zb_list_get_unchecked_any",
                        vec![held.node.clone(), int_lit(i as i64, span)],
                        Ty::Object,
                        span,
                    ),
                    ty: Ty::Object,
                };
                let ty = ty.settled();
                let node = if trusted {
                    self.trusted(item, ty)
                } else {
                    self.coerce(item, ty)
                };
                Val { node, ty }
            })
            .collect();
        let value = self.tuple_of_items(items, target, span);
        if pre.is_empty() {
            value
        } else {
            Self::block_value(pre, value, target, span)
        }
    }

    /// A value as an element of a list of kind `e`: an instance goes in
    /// by address.
    /// A value as an element of kind `e`: converted to what the kind
    /// reads as, an instance to its address, an array's element to its
    /// stored width, which for an integer typecode checks the range and
    /// raises OverflowError where it is used.
    fn elem_arg(&mut self, v: Val, e: Elem) -> Node {
        let span = v.node.span;
        let node = self.coerce(v, e.ty());
        match e {
            Elem::Class(_) => as_addr(node, span),
            Elem::Array(c) if c.narrows() => self.narrowed(node, c, span),
            _ => node,
        }
    }

    /// `value`, a number of what typecode `c` reads as, at the width
    /// `c` stores.
    fn narrowed(&mut self, value: Node, c: types::Code, span: Span) -> Node {
        let stored = c.storage().ty();
        if c.storage() == zyntax_builtins::Kind::F32 {
            return TypedNode::new(
                TypedExpression::Cast(TypedCast {
                    expr: Box::new(value),
                    target_type: stored.clone(),
                }),
                stored,
                span,
            );
        }
        let narrowed = typed_call(
            &list_fn("narrow", Elem::Array(c)),
            vec![value],
            stored.clone(),
            span,
        );
        // Held at the stored width, then checked before anything uses
        // it, as `guard` holds a value of one of the frontend's types.
        let name = self.temp();
        self.hoisted.push(TypedNode::new(
            TypedStatement::Let(TypedLet {
                name,
                ty: stored.clone(),
                mutability: Mutability::Immutable,
                initializer: Some(Box::new(narrowed)),
                span,
            }),
            Type::Unknown,
            span,
        ));
        let check = self.pending_check(span);
        self.hoisted.push(check);
        TypedNode::new(TypedExpression::Variable(name), stored, span)
    }

    fn expr_as_elem(&mut self, expr: &py::Expr, e: Elem) -> Result<Node> {
        let v = self.expr(expr)?;
        Ok(self.elem_arg(v, e))
    }

    /// A search of array `xs` for `probe`: `contains`, `count`,
    /// `index_or_neg`. A number the storage cannot hold is in no array,
    /// so it is not narrowed but answered as absent; a value of another
    /// type is absent too.
    fn array_search(
        &mut self,
        probe: Val,
        xs: Node,
        e: Elem,
        c: types::Code,
        op: &str,
        span: Span,
    ) -> Result<Node> {
        let (ty, absent) = match op {
            "contains" => (Ty::Bool, TypedLiteral::Bool(false)),
            "count" => (Ty::Int, TypedLiteral::Integer(0)),
            _ => (Ty::Int, TypedLiteral::Integer(-1)),
        };
        let absent = node(TypedExpression::Literal(absent), ty, span);
        let numeric = match (c.item(), probe.ty) {
            (Ty::Int, Ty::Int | Ty::Bool) | (Ty::Float, Ty::Int | Ty::Float | Ty::Bool) => true,
            (Ty::Int, Ty::Float) => {
                return Err(Error::unsupported_span(
                    "a float searched for in an integer array",
                    span,
                ));
            }
            (_, Ty::Object | Ty::Unknown) => {
                return Err(Error::unsupported_span(
                    "a dynamic value searched for in an array",
                    span,
                ));
            }
            _ => false,
        };
        if !numeric {
            return Ok(Self::block_value(
                vec![
                    TypedNode::new(
                        TypedStatement::Expression(Box::new(probe.node)),
                        Type::Unknown,
                        span,
                    ),
                    TypedNode::new(
                        TypedStatement::Expression(Box::new(xs)),
                        Type::Unknown,
                        span,
                    ),
                ],
                absent,
                ty,
                span,
            ));
        }
        let wide = self.coerce(probe, c.item());
        if !c.narrows() {
            return Ok(call(&list_fn(op, e), vec![xs, wide], ty, span));
        }
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: wide,
                ty: c.item(),
            },
            &mut pre,
            span,
        );
        let fits = call(
            &list_fn("in_range", e),
            vec![held.node.clone()],
            Ty::Bool,
            span,
        );
        let stored = c.storage().ty();
        let narrowed = TypedNode::new(
            TypedExpression::Cast(TypedCast {
                expr: Box::new(held.node),
                target_type: stored.clone(),
            }),
            stored,
            span,
        );
        let found = call(&list_fn(op, e), vec![xs, narrowed], ty, span);
        let value = node(
            TypedExpression::If(TypedIfExpr {
                condition: Box::new(fits),
                then_branch: Box::new(found),
                else_branch: Box::new(absent),
            }),
            ty,
            span,
        );
        Ok(Self::block_value(pre, value, ty, span))
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
            Ty::Bytes => binary(
                BinaryOp::Ne,
                call("zb_str_len", vec![v.node], Ty::Int, span),
                int_lit(0, span),
                Ty::Bool,
                span,
            ),
            Ty::File(_) => Self::after_none(
                v.node,
                node(
                    TypedExpression::Literal(TypedLiteral::Bool(true)),
                    Ty::Bool,
                    span,
                ),
                Ty::Bool,
            ),
            Ty::Dict(_) => binary(
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
            Ty::List(_) => binary(
                BinaryOp::Ne,
                method_call(v.node, "len", vec![], Ty::Int, span),
                int_lit(0, span),
                Ty::Bool,
                span,
            ),
            // A shape has at least one element.
            Ty::Tuple(_) => Self::after_none(
                v.node,
                node(
                    TypedExpression::Literal(TypedLiteral::Bool(true)),
                    Ty::Bool,
                    span,
                ),
                Ty::Bool,
            ),
            Ty::Set => binary(
                BinaryOp::Ne,
                call("zb_set_len", vec![v.node], Ty::Int, span),
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
            Ty::Bytes => call("zb_bytes_repr", vec![v.node], Ty::Str, span),
            Ty::File(_) => str_lit("<file>", span),
            Ty::None => {
                Self::after_none(v.node, call("zb_none_repr", vec![], Ty::Str, span), Ty::Str)
            }
            // An array prints under its typecode: `array('i', [1, 2])`,
            // and `array('i')` when empty.
            Ty::List(e @ Elem::Array(c)) => {
                let mut pre = Vec::new();
                let held = self.hold(v, &mut pre, span);
                let empty = binary(
                    BinaryOp::Eq,
                    method_call(held.node.clone(), "len", vec![], Ty::Int, span),
                    int_lit(0, span),
                    Ty::Bool,
                    span,
                );
                let text = node(
                    TypedExpression::If(TypedIfExpr {
                        condition: Box::new(empty),
                        then_branch: Box::new(str_lit(&format!("array('{}')", c.letter()), span)),
                        else_branch: Box::new(call(
                            &list_fn("items", e),
                            vec![
                                held.node,
                                str_lit(&format!("array('{}', [", c.letter()), span),
                                str_lit("])", span),
                            ],
                            Ty::Str,
                            span,
                        )),
                    }),
                    Ty::Str,
                    span,
                );
                Self::block_value(pre, text, Ty::Str, span)
            }
            Ty::List(e) => call(&list_fn("repr", e), vec![v.node], Ty::Str, span),
            Ty::Tuple(_) => {
                let items = self.coerce(v, Ty::List(Elem::Object));
                call("zb_tuple_repr", vec![items], Ty::Str, span)
            }
            Ty::Dict(_) => call("zb_dict_repr", vec![v.node], Ty::Str, span),
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
            Ty::Bytes => call("zb_bytes_repr", vec![v.node], Ty::Str, span),
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
    /// The primitive every target of an unpacking is a plain local of,
    /// when they all are: `x, y, z = p` with `x`, `y`, `z` floats.
    fn unpacks_primitive_names(&self, elts: &[py::Expr]) -> Option<Ty> {
        let mut kind = None;
        for elt in elts {
            let py::Expr::Name(n) = elt else { return None };
            let name = n.id.as_str();
            if self.is_global(name)
                || self.cells.contains_key(name)
                || self.comp_symbols.contains_key(name)
            {
                return None;
            }
            let ty = self.var_ty(name);
            if !matches!(ty, Ty::Int | Ty::Float) {
                return None;
            }
            if kind.is_some_and(|k| k != ty) {
                return None;
            }
            kind = Some(ty);
        }
        kind
    }

    /// `x, y, z = p` where `p` is a dynamic value and the targets are
    /// locals of one primitive: when the box holds a list of that kind
    /// the elements are read from it directly, with no boxes made for
    /// them; any other value goes the general way, unpacked through a
    /// list of dynamic values and converted one by one.
    fn bind_unpacked_primitives(
        &mut self,
        t: &py::ExprTuple,
        value: Val,
        kind: Ty,
        span: Span,
        out: &mut Vec<Stmt>,
    ) -> Result<()> {
        let elem = Elem::of(kind);
        let list_kind = match kind {
            Ty::Int => zyntax_builtins::Kind::Int,
            _ => zyntax_builtins::Kind::Float,
        };
        // Every target is declared before the branch, so both arms
        // assign it.
        for elt in &t.elts {
            let py::Expr::Name(n) = elt else { continue };
            let name = self.local_symbol(n.id.as_str());
            if !self.bound.contains(&name) {
                self.bound.push(name);
                let initial = self.zero_of(kind, span);
                out.push(TypedNode::new(
                    TypedStatement::Let(TypedLet {
                        name,
                        ty: ir(kind),
                        mutability: Mutability::Mutable,
                        initializer: Some(Box::new(initial)),
                        span,
                    }),
                    Type::Unknown,
                    span,
                ));
            }
        }
        let source = self.hold(value, out, span);
        // The kind above the category byte of the box's tag.
        let is_kind = binary(
            BinaryOp::Eq,
            call("zb_any_kind", vec![source.node.clone()], Ty::Int, span),
            int_lit(list_kind.list_tag() >> 8, span),
            Ty::Bool,
            span,
        );
        let n = t.elts.len() as i64;

        // The direct arm: the list read as itself.
        let mut direct = Vec::new();
        let xs = self.hold(
            Val {
                node: call(
                    &format!("zb_unbox_list_raw_{}", list_kind.suffix()),
                    vec![source.node.clone()],
                    Ty::List(elem),
                    span,
                ),
                ty: Ty::List(elem),
            },
            &mut direct,
            span,
        );
        direct.push(TypedNode::new(
            TypedStatement::Expression(Box::new(call(
                &list_fn("expect_len", elem),
                vec![xs.node.clone(), int_lit(n, span)],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        ));
        direct.push(self.pending_check(span));
        for (i, elt) in t.elts.iter().enumerate() {
            let item = Val {
                node: elem_call(
                    "get_unchecked",
                    elem,
                    vec![xs.node.clone(), int_lit(i as i64, span)],
                    span,
                ),
                ty: kind,
            };
            self.bind(elt, item, span, &mut direct)?;
        }

        // The general arm: what any other dynamic value goes through.
        let mut general = Vec::new();
        let items = Val {
            node: call(
                "zb_any_iter",
                vec![source.node.clone()],
                Ty::List(Elem::Object),
                span,
            ),
            ty: Ty::List(Elem::Object),
        };
        let seq = self.hold(items, &mut general, span);
        general.push(self.pending_check(span));
        general.push(TypedNode::new(
            TypedStatement::Expression(Box::new(call(
                "zb_list_expect_len_any",
                vec![seq.node.clone(), int_lit(n, span)],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        ));
        general.push(self.pending_check(span));
        for (i, elt) in t.elts.iter().enumerate() {
            let item = Val {
                node: call(
                    "zb_list_get_unchecked_any",
                    vec![seq.node.clone(), int_lit(i as i64, span)],
                    Ty::Object,
                    span,
                ),
                ty: Ty::Object,
            };
            let item = self.hold(item, &mut general, span);
            self.bind(elt, item, span, &mut general)?;
        }

        out.push(TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(is_kind),
                then_block: TypedBlock {
                    statements: direct,
                    span,
                },
                else_block: Some(TypedBlock {
                    statements: general,
                    span,
                }),
                span,
            }),
            Type::Unknown,
            span,
        ));
        Ok(())
    }

    pub(crate) fn hold(&mut self, v: Val, out: &mut Vec<Stmt>, span: Span) -> Val {
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
    pub(crate) fn block_value(statements: Vec<Stmt>, value: Node, ty: Ty, span: Span) -> Node {
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

    pub(crate) fn stmt(&mut self, s: &py::Stmt, out: &mut Vec<Stmt>) -> Result<()> {
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
        // A variable the statement assigns may be None on some path
        // through it, or at the top of its loop's next pass; one it
        // never assigns holds what it held, into the statement and out
        // of it. A field is written by whatever the statement calls.
        let kept = if compound {
            let scope = Scope::of_body(Vec::new(), std::slice::from_ref(s));
            // What an `except` or a pattern binds is not a stored name,
            // and a nested body may write through `nonlocal`.
            let binds_unseen =
                matches!(s, py::Stmt::Try(_) | py::Stmt::Match(_)) || !scope.children.is_empty();
            if binds_unseen {
                self.nonnull.clear();
            } else {
                self.nonnull
                    .retain(|v| !v.resolve_global().is_some_and(|n| scope.bound.contains(&n)));
            }
            self.nonnull_fields.clear();
            Some(self.nonnull.clone())
        } else {
            None
        };
        let mut own = Vec::new();
        self.stmt_into(s, &mut own)?;
        if let Some(kept) = kept {
            self.nonnull = kept;
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
        if self.redirected
            && matches!(s, py::Stmt::For(_) | py::Stmt::While(_))
            && let Some(ctl) = self.try_ctls.last()
        {
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
                if let py::Expr::Name(n) = &*e.value
                    && !self.is_variable(n.id.as_str())
                    && !self.module.funcs.contains_key(n.id.as_str())
                    && !self.module.class_index.contains_key(n.id.as_str())
                    && types::builtin_index(n.id.as_str()).is_some()
                {
                    return Ok(());
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
                // A tuple literal consumed immediately by an unpacking
                // assignment has no observable tuple identity. Evaluate
                // every RHS expression first, then bind the targets in
                // order, retaining each element's inferred type.
                if let ([target], py::Expr::Tuple(source)) = (a.targets.as_slice(), &*a.value) {
                    let targets = match target {
                        py::Expr::Tuple(t) => Some(&t.elts),
                        py::Expr::List(l) => Some(&l.elts),
                        _ => None,
                    };
                    if let Some(targets) = targets.filter(|t| {
                        t.len() == source.elts.len()
                            && t.iter().all(|target| {
                                !matches!(target, py::Expr::Tuple(_) | py::Expr::List(_))
                            })
                    }) {
                        let mut values = Vec::with_capacity(source.elts.len());
                        for expr in &source.elts {
                            let value = self.expr(expr)?;
                            out.append(&mut self.hoisted);
                            values.push(self.hold(value, out, span));
                        }
                        // What a binding hoists (a `__setitem__` call
                        // checked for a raise) follows the holds it reads.
                        for (target, value) in targets.iter().zip(values) {
                            self.bind_after(target, value, span, out)?;
                        }
                        return Ok(());
                    }
                }
                // A literal that says nothing of its elements is built
                // as the list its name or field holds, which inference
                // typed by what the program puts in it.
                if let [target] = a.targets.as_slice()
                    && let Some(count) = types::unkinded_list(&a.value)
                {
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
                            let none =
                                node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span);
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
                            value = self.arithmetic(py::Operator::Mult, value, times, n, span)?;
                        }
                        return self.bind(target, value, span, out);
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
                // The body runs only when the test held.
                if let Some((place, true)) = self.instance_test(&w.test) {
                    self.assume_place(&place);
                }
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
                        Ty::Dict(_) => {
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
                if let py::Expr::Call(c) = &*a.test
                    && types::is_name(&c.func, "isinstance")
                    && c.arguments.args.len() == 2
                    && let py::Expr::Name(x) = &c.arguments.args[0]
                    && self.locals.narrowed.contains_key(x.id.as_str())
                {
                    return Ok(());
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
            py::Stmt::With(w) => self.with_stmt(w, span, out)?,
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
            // A class statement of the module body: its methods are the
            // module's functions, its attributes not fixed as constants
            // are stored now, in order.
            py::Stmt::ClassDef(c) if self.module.class_index.contains_key(c.name.as_str()) => {
                let k = self.module.class_index[c.name.as_str()];
                let declared = crate::class_attrs::declared_in(c)?;
                let names: Vec<String> = declared.iter().map(|d| d.name.clone()).collect();
                for d in &declared {
                    let attr = self.class_attr_of(k, &d.name, span)?;
                    if attr.constant.is_some() {
                        continue;
                    }
                    // A sibling read here would be the class's, not the
                    // module's; only constants are read that way.
                    if crate::class_attrs::reads_any(d.value, &names) {
                        return unsupported(
                            format!(
                                "class attribute `{}` computed from another attribute that is not a constant",
                                d.name
                            ),
                            d.value,
                        );
                    }
                    let value = self.expr(d.value)?;
                    let ty = self.var_ty(&attr.global);
                    let value_span = crate::span_of(d.value);
                    self.store_global(&attr.global, value, ty, value_span, out);
                }
            }
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
    pub(crate) fn bind(
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
            // `C.X = v`: the class attribute's module variable.
            // `sys.stdout = f`: print writes to `f` from here on; `None`
            // or the value read from `sys.stdout` before restores it.
            py::Expr::Attribute(a)
                if self.module_member_of(&a.value, a.attr.as_str()).is_some() =>
            {
                if !matches!(&*a.value, py::Expr::Name(m) if m.id.as_str() == "sys")
                    || a.attr.as_str() != "stdout"
                {
                    return unsupported("assignment to a module attribute", target);
                }
                let v = self.coerce(value, Ty::Object);
                out.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(call(
                        "zb_set_stdout",
                        vec![v],
                        Ty::None,
                        span,
                    ))),
                    Type::Unknown,
                    span,
                ));
                return Ok(());
            }
            py::Expr::Attribute(a) if let Some(k) = self.module.class_of_expr(&a.value) => {
                let attr = self.class_attr_of(k, a.attr.as_str(), span)?;
                let ty = self.var_ty(&attr.global);
                self.store_global(&attr.global, value, ty, span, out);
                return Ok(());
            }
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
                    let seq_node = if target_pre.is_empty() {
                        seq.node
                    } else {
                        Self::block_value(target_pre, seq.node, seq.ty, span)
                    };
                    let elem = match seq.ty {
                        Ty::List(e) => Some(e),
                        Ty::Object => None,
                        _ => return unsupported("slice assignment on a non-list", target),
                    };
                    let source = if matches!(value.ty, Ty::List(_)) {
                        value
                    } else {
                        Val {
                            node: self.iterable(value, span),
                            ty: Ty::List(Elem::Object),
                        }
                    };
                    let typed = match elem {
                        Some(e) => self.coerce(source, Ty::List(e)),
                        None => self.coerce(source, Ty::List(Elem::Object)),
                    };
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
                    let call = match elem {
                        Some(e) => call(
                            &list_fn("assign_slice", e),
                            vec![typed, seq_node, start, stop, step, int_lit(mask, span)],
                            Ty::None,
                            span,
                        ),
                        // A dynamic sequence: the runtime picks the kind.
                        None => call(
                            "zb_any_assign_slice",
                            vec![seq_node, typed, start, stop, step, int_lit(mask, span)],
                            Ty::None,
                            span,
                        ),
                    };
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
                    Ty::Dict(_) => {
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
                        let i = self.expr(&sub.slice)?;
                        let v = self.coerce(value, Ty::Object);
                        if i.ty == Ty::Int {
                            call(
                                "zb_any_setitem_i64",
                                vec![seq.node, i.node, v],
                                Ty::None,
                                span,
                            )
                        } else {
                            let i = self.coerce(i, Ty::Object);
                            call("zb_any_setitem", vec![seq.node, i, v], Ty::None, span)
                        }
                    }
                    // An instance stores through its class's `__setitem__`.
                    Ty::Class(k) => {
                        let i = self.expr(&sub.slice)?;
                        match self.dunder(k as usize, "__setitem__", seq.node, vec![i, value], span)
                        {
                            Some(r) => r.node,
                            None => {
                                return Err(Error::unsupported_span(
                                    format!(
                                        "item assignment on {}, which defines no __setitem__",
                                        self.module.classes[k as usize].name
                                    ),
                                    span,
                                ));
                            }
                        }
                    }
                    _ => {
                        return unsupported(
                            format!("item assignment on {}", types::expr_kind(&sub.value)),
                            target,
                        );
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
                let starred = t.elts.iter().any(|e| matches!(e, py::Expr::Starred(_)));
                // A tuple of the targets' own arity is taken apart field
                // by field, each held before any target is bound; one of
                // another arity is unpacked as the list of its elements,
                // whose length check raises.
                if let Some(shape) = value.ty.tuple_elems()
                    && shape.len() == t.elts.len()
                    && !starred
                {
                    let (fields, _) = self.tuple_fields(value, span, Hold::Into(out));
                    let mut items = Vec::with_capacity(fields.len());
                    for field in fields {
                        items.push(self.hold(field, out, span));
                    }
                    for (elt, item) in t.elts.iter().zip(items) {
                        self.bind_after(elt, item, span, out)?;
                    }
                    return Ok(());
                }
                // A set unpacks as the list of its values.
                let value = if value.ty == Ty::Set || matches!(value.ty, Ty::Tuple(_)) {
                    Val {
                        node: self.coerce(value, Ty::List(Elem::Object)),
                        ty: Ty::List(Elem::Object),
                    }
                } else {
                    value
                };
                // Resolve a dynamic iterable once before reading its fields.
                let dynamic = value.ty == Ty::Object;
                if dynamic && let Some(kind) = self.unpacks_primitive_names(&t.elts) {
                    return self.bind_unpacked_primitives(t, value, kind, span, out);
                }
                let value = if dynamic {
                    Val {
                        node: call(
                            "zb_any_iter",
                            vec![value.node],
                            Ty::List(Elem::Object),
                            span,
                        ),
                        ty: Ty::List(Elem::Object),
                    }
                } else {
                    value
                };
                let elem_ty = value.ty.element().unwrap_or(Ty::Object);
                let seq = self.hold(value, out, span);
                if dynamic {
                    out.push(self.pending_check(span));
                }
                let n = t.elts.len() as i64;
                let check = match seq.ty {
                    Ty::List(e) => Some(call(
                        &list_fn("expect_len", e),
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
                    out.push(self.pending_check(span));
                }
                let mut items = Vec::with_capacity(t.elts.len());
                for i in 0..t.elts.len() {
                    let index = int_lit(i as i64, span);
                    let item = match seq.ty {
                        Ty::List(e) => Val {
                            node: elem_call(
                                "get_unchecked",
                                e,
                                vec![seq.node.clone(), index],
                                span,
                            ),
                            ty: elem_ty,
                        },
                        _ => self.index_value(
                            Val {
                                node: seq.node.clone(),
                                ty: seq.ty,
                            },
                            index,
                            elem_ty,
                            span,
                        ),
                    };
                    items.push(self.hold(item, out, span));
                }
                for (elt, item) in t.elts.iter().zip(items) {
                    self.bind_after(elt, item, span, out)?;
                }
                return Ok(());
            }
            other => {
                return unsupported(format!("assignment to {}", types::expr_kind(other)), other);
            }
        };
        let ty = self.var_ty(n.id.as_str());
        let name = self.local_symbol(n.id.as_str());
        // A checked read here depends on the preceding unpack statements.
        // Emit its pending-exception check after the store, not in the
        // statement-wide hoist that runs before those statements.
        let check_after = self.guards
            && value.ty == Ty::Object
            && matches!(
                ty,
                Ty::Int | Ty::Float | Ty::Bool | Ty::Str | Ty::List(_) | Ty::Class(_)
            );
        // An instance assigned from its constructor is known to be one
        // until the block ends or the variable is assigned again.
        if let Ty::Class(_) = ty {
            if value.ty != Ty::None && self.known_instance(&value.node) {
                self.nonnull.insert(name);
            } else {
                self.nonnull.remove(&name);
            }
        }
        if !self.comp_symbols.contains_key(n.id.as_str())
            && let Some(cell) = self.cells.get(n.id.as_str()).copied()
        {
            if check_after {
                self.guards = false;
            }
            let value = self.coerce(value, ty);
            if check_after {
                self.guards = true;
            }
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
            if check_after {
                out.push(self.pending_check(span));
            }
            return Ok(());
        }
        if self.is_global(n.id.as_str()) {
            if check_after {
                self.guards = false;
            }
            self.store_global(n.id.as_str(), value, ty, span, out);
            if check_after {
                self.guards = true;
                out.push(self.pending_check(span));
            }
            return Ok(());
        }
        if check_after {
            self.guards = false;
        }
        let value = self.coerce(value, ty);
        if check_after {
            self.guards = true;
        }
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
            if check_after {
                out.push(self.pending_check(span));
            }
            return Ok(());
        }
        let assign = binary(BinaryOp::Assign, var(name, ty, span), value, Ty::None, span);
        out.push(TypedNode::new(
            TypedStatement::Expression(Box::new(assign)),
            Type::Unknown,
            span,
        ));
        if check_after {
            out.push(self.pending_check(span));
        }
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
        // A list's element comes out as the list's kind, then as the
        // type wanted when that differs (a dict's keys, read as they are
        // typed).
        if let Ty::List(e) = seq.ty
            && e.ty() != elem_ty
        {
            let read = Val {
                node: elem_call("get", e, vec![seq.node, index], span),
                ty: e.ty(),
            };
            return Val {
                node: self.coerce(read, elem_ty),
                ty: elem_ty,
            };
        }
        let node = match seq.ty {
            Ty::List(e) => elem_call("get", e, vec![seq.node, index], span),
            // An index only the runtime knows reads the tuple as the
            // list of its elements.
            Ty::Tuple(_) => {
                let e = Elem::of(elem_ty);
                let items = Val {
                    node: self.coerce(seq, Ty::List(e)),
                    ty: Ty::List(e),
                };
                return self.index_value(items, index, elem_ty, span);
            }
            Ty::Str => call("zb_str_get", vec![seq.node, index], Ty::Str, span),
            Ty::Bytes => call("zb_bytes_index", vec![seq.node, index], Ty::Int, span),
            _ => call(
                "zb_any_getitem_i64",
                vec![seq.node, index],
                Ty::Object,
                span,
            ),
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
        self.for_items(f, seq, None, extra, span)
    }

    /// A loop over the items of `seq` by position. With `index_from`,
    /// the target is a pair bound to the position counted from there
    /// and the item, as `enumerate` yields them.
    fn for_items(
        &mut self,
        f: &py::StmtFor,
        seq: Val,
        index_from: Option<Node>,
        extra: Vec<Stmt>,
        span: Span,
    ) -> Result<TypedStatement> {
        // A generator's items are pulled into a list first.
        let seq = if seq.ty == Ty::Gen {
            Val {
                node: self.iterable(seq, span),
                ty: Ty::List(Elem::Object),
            }
        } else {
            seq
        };
        // A tuple iterates as the list of its elements, whose kind is
        // the element the loop variable takes.
        let seq = self.tuple_as_list(seq);
        let elem_ty = seq.ty.element().unwrap_or(Ty::Object);
        // The iterator is evaluated once, before the loop.
        let mut prologue = std::mem::take(&mut self.hoisted);
        let seq = match seq.ty {
            // A dynamic iterable is snapshotted into a list of objects,
            // and a dict iterates over a snapshot of its keys.
            Ty::Object | Ty::Dict(_) => {
                let items = match seq.ty {
                    Ty::Dict(_) => {
                        call("zb_dict_keys", vec![seq.node], Ty::List(Elem::Object), span)
                    }
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
        // The index origin is evaluated once as well.
        let index_from = index_from.map(|start| {
            self.hold(
                Val {
                    node: start,
                    ty: Ty::Int,
                },
                &mut prologue,
                span,
            )
        });
        let counter = self.temp();
        let len = match seq.ty {
            Ty::Str => call("zb_str_chars_len", vec![seq.node.clone()], Ty::Int, span),
            Ty::Bytes => call("zb_str_len", vec![seq.node.clone()], Ty::Int, span),
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
        // What reading the item hoists (a checked read of a dynamic
        // element) runs inside the body, ahead of the binding.
        let mut body = std::mem::take(&mut self.hoisted);
        match index_from {
            Some(start) => {
                let py::Expr::Tuple(t) = &*f.target else {
                    unreachable!("an enumerate loop binds a pair");
                };
                let index = Val {
                    node: binary(
                        BinaryOp::Add,
                        start.node,
                        var(counter, Ty::Int, span),
                        Ty::Int,
                        span,
                    ),
                    ty: Ty::Int,
                };
                self.bind_after(&t.elts[0], index, span, &mut body)?;
                self.bind_after(&t.elts[1], item, span, &mut body)?;
            }
            None => self.bind_after(&f.target, item, span, &mut body)?,
        }
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
        // `for i, x in enumerate(xs)` counts alongside the sequence: no
        // list of pairs is built.
        if let py::Expr::Call(c) = &*f.iter
            && types::is_name(&c.func, "enumerate")
            && !self.is_variable("enumerate")
            && matches!(c.arguments.args.len(), 1 | 2)
            && c.arguments
                .keywords
                .iter()
                .all(|k| k.arg.as_ref().is_some_and(|a| a == "start"))
            && c.arguments.args.len() + c.arguments.keywords.len() <= 2
            && matches!(&*f.target, py::Expr::Tuple(t) if t.elts.len() == 2 && !t.elts.iter().any(|e| matches!(e, py::Expr::Starred(_))))
        {
            let seq = self.expr(&c.arguments.args[0])?;
            let start = match (c.arguments.args.get(1), c.arguments.keywords.first()) {
                (Some(s), _) => self.expr_as(s, Ty::Int)?,
                (None, Some(kw)) => self.expr_as(&kw.value, Ty::Int)?,
                (None, None) => int_lit(0, span),
            };
            return self.for_items(f, seq, Some(start), extra, span);
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
        let mut pre = std::mem::take(&mut self.hoisted);
        // The loop counts in the target itself when the target is a
        // plain int local the body never assigns and the step's sign is
        // known; a shared, global or object-typed target, or one the
        // body writes (a nested loop over the same name), is assigned
        // from a hidden counter each time round, since the range runs on
        // regardless.
        let body_writes_target = Scope::of_body(Vec::new(), &f.body)
            .bound
            .contains(target.id.as_str());
        let step_sign = match c.arguments.args.get(2) {
            None => Some(1),
            Some(e) => types::int_literal(e).map(|v| v.signum()),
        };
        let direct = !self.cells.contains_key(target.id.as_str())
            && !self.is_global(target.id.as_str())
            && self.var_ty(target.id.as_str()) == Ty::Int
            && !body_writes_target
            && step_sign.is_some_and(|sign| sign != 0);
        // A direct loop assigns the target the start and leaves it one
        // step past the last value it ran with; Python leaves the target
        // as it was when the range is empty, else the last value. The
        // bounds are held so the loop runs only when the range is not
        // empty and its end can be put right.
        let (start, end, step, ran, fixup) = if direct {
            let held_start = self.hold(
                Val {
                    node: start,
                    ty: Ty::Int,
                },
                &mut pre,
                span,
            );
            let held_end = self.hold(
                Val {
                    node: end,
                    ty: Ty::Int,
                },
                &mut pre,
                span,
            );
            let step_node = step.map(|step| {
                self.hold(
                    Val {
                        node: step,
                        ty: Ty::Int,
                    },
                    &mut pre,
                    span,
                )
                .node
            });
            let name = self.local_symbol(target.id.as_str());
            let counter = || var(name, Ty::Int, span);
            let (past_end, ran) = if step_sign == Some(1) {
                (
                    binary(
                        BinaryOp::Ge,
                        counter(),
                        held_end.node.clone(),
                        Ty::Bool,
                        span,
                    ),
                    binary(
                        BinaryOp::Gt,
                        held_end.node.clone(),
                        held_start.node.clone(),
                        Ty::Bool,
                        span,
                    ),
                )
            } else {
                (
                    binary(
                        BinaryOp::Le,
                        counter(),
                        held_end.node.clone(),
                        Ty::Bool,
                        span,
                    ),
                    binary(
                        BinaryOp::Lt,
                        held_end.node.clone(),
                        held_start.node.clone(),
                        Ty::Bool,
                        span,
                    ),
                )
            };
            let back = binary(
                BinaryOp::Sub,
                counter(),
                step_node.clone().unwrap_or_else(|| int_lit(1, span)),
                Ty::Int,
                span,
            );
            // A `break` leaves the counter at the value it ran with,
            // short of the end; only a run to the end steps back.
            let fixup = TypedNode::new(
                TypedStatement::If(TypedIf {
                    condition: Box::new(past_end),
                    then_block: TypedBlock {
                        statements: vec![TypedNode::new(
                            TypedStatement::Expression(Box::new(binary(
                                BinaryOp::Assign,
                                counter(),
                                back,
                                Ty::None,
                                span,
                            ))),
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
            );
            (
                held_start.node,
                held_end.node,
                step_node,
                Some(ran),
                Some(fixup),
            )
        } else {
            (start, end, step, None, None)
        };
        let name = if direct {
            let name = self.local_symbol(target.id.as_str());
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
        let Some(ran) = ran else {
            if pre.is_empty() {
                return Ok(loop_stmt);
            }
            let mut statements = pre;
            statements.push(TypedNode::new(loop_stmt, Type::Unknown, span));
            return Ok(TypedStatement::Block(TypedBlock { statements, span }));
        };
        let mut guarded = vec![TypedNode::new(loop_stmt, Type::Unknown, span)];
        guarded.extend(fixup);
        let mut statements = pre;
        statements.push(TypedNode::new(
            TypedStatement::If(TypedIf {
                condition: Box::new(ran),
                then_block: TypedBlock {
                    statements: guarded,
                    span,
                },
                else_block: None,
                span,
            }),
            Type::Unknown,
            span,
        ));
        Ok(TypedStatement::Block(TypedBlock { statements, span }))
    }

    // ─── Imported modules ──────────────────────────────────────────

    /// The member `value.attr` names when `value` is an imported
    /// module's name that no variable shadows.
    fn module_member_of(&self, value: &py::Expr, attr: &str) -> Option<stdlib::Member> {
        // `os.path.join`: the submodule's member, named through it.
        if let py::Expr::Attribute(sub) = value {
            return self.module_member_of(&sub.value, &format!("{}.{attr}", sub.attr.as_str()));
        }
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
            stdlib::Member::Str(s) => Val {
                node: str_lit(s, span),
                ty: Ty::Str,
            },
            stdlib::Member::Value { ty, zb } => Val {
                node: call(zb, Vec::new(), ty, span),
                ty,
            },
            stdlib::Member::ArrayType => {
                return unsupported(format!("`{name}` as a value"), e);
            }
            stdlib::Member::Binary(_) => Val {
                node: self.callable_value(e)?,
                ty: Ty::Object,
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
                return unsupported(format!("`{name}` of a module as a value"), e);
            }
        })
    }

    /// `array(typecode[, initializer])`: the array of the typecode,
    /// filled from the initializer's items, which are converted as the
    /// typecode requires and copied, so the array shares nothing with
    /// what it was made from. The typecode is the array's static type,
    /// so it has to be spelled out.
    fn array_new(
        &mut self,
        args: &[py::Expr],
        keywords: &[py::Keyword],
        c: &py::ExprCall,
        span: Span,
    ) -> Result<Val> {
        if !keywords.is_empty() {
            return unsupported("keyword arguments to array()", c);
        }
        let (Some(first), true) = (args.first(), args.len() <= 2) else {
            return unsupported(format!("array() with {} argument(s)", args.len()), c);
        };
        let py::Expr::StringLiteral(code) = first else {
            return unsupported(
                "array() with a typecode that is not a string literal; the typecode is the array's type",
                first,
            );
        };
        let typecode = code.value.to_str();
        let code = match stdlib::array_code(typecode) {
            Ok(code) => code,
            Err(why) => {
                return unsupported(format!("array typecode {typecode:?}: {why}"), first);
            }
        };
        let elem = Elem::Array(code);
        let target = Ty::List(elem);
        let Some(init) = args.get(1) else {
            return Ok(Val {
                node: self.list_of(Vec::new(), elem, span),
                ty: target,
            });
        };
        let source = self.expr(init)?;
        let node = match source.ty {
            // The same typecode: a copy of its own.
            Ty::List(e) if e == elem => {
                call(&list_fn("copy", elem), vec![source.node], target, span)
            }
            Ty::List(_) | Ty::Tuple(_) => self.coerce(source, target),
            // A dynamic iterable, a set or a generator: its items first.
            Ty::Set | Ty::Object | Ty::Gen => {
                let items = Val {
                    node: self.iterable(source, span),
                    ty: Ty::List(Elem::Object),
                };
                self.coerce(items, target)
            }
            // Bytes are the array's storage, as `frombytes` reads them:
            // a list of zeros of the length they fill, then the copy.
            Ty::Bytes => {
                let mut pre = Vec::new();
                let data = self.hold(source, &mut pre, span);
                let size = call("zb_str_len", vec![data.node.clone()], Ty::Int, span);
                let count = binary(
                    BinaryOp::Div,
                    size.clone(),
                    int_lit(code.itemsize(), span),
                    Ty::Int,
                    span,
                );
                let ragged = binary(
                    BinaryOp::Ne,
                    binary(
                        BinaryOp::Rem,
                        size,
                        int_lit(code.itemsize(), span),
                        Ty::Int,
                        span,
                    ),
                    int_lit(0, span),
                    Ty::Bool,
                    span,
                );
                let mut raise = Vec::new();
                self.raise_named(
                    "ValueError",
                    str_lit("bytes length not a multiple of item size", span),
                    span,
                    &mut raise,
                );
                pre.push(TypedNode::new(
                    TypedStatement::If(TypedIf {
                        condition: Box::new(ragged),
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
                let zero = Val {
                    node: int_lit(0, span),
                    ty: Ty::Int,
                };
                let one = self.list_of(vec![zero], elem, span);
                let zeros = call(&list_fn("repeat", elem), vec![one, count], target, span);
                let xs = self.hold(
                    Val {
                        node: zeros,
                        ty: target,
                    },
                    &mut pre,
                    span,
                );
                pre.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(call(
                        "zb_bytes_copy_out",
                        vec![
                            data.node,
                            crate::bytes::field(xs.node.clone(), "data", Ty::Int, span),
                        ],
                        Ty::Int,
                        span,
                    ))),
                    Type::Unknown,
                    span,
                ));
                Self::block_value(pre, xs.node, target, span)
            }
            Ty::Str => {
                return unsupported("array() from a string", init);
            }
            _ => {
                return unsupported("array() from a value that is not iterable", init);
            }
        };
        Ok(Val { node, ty: target })
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
        if let stdlib::Member::ArrayType = member {
            return self.array_new(args, keywords, c, span);
        }
        // `operator.add(a, b)`: the operator itself.
        if let stdlib::Member::Binary(op) = member {
            if args.len() != 2 || !keywords.is_empty() {
                return unsupported(format!("calling `{name}` with these arguments"), c);
            }
            let l = self.expr(&args[0])?;
            let r = self.expr(&args[1])?;
            return self.arithmetic(op, l, r, &args[1], span);
        }
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
                });
            }
            ("zb_exit", 1) if matches!(&args[0], py::Expr::NoneLiteral(_)) => {
                return Ok(Val {
                    node: call("zb_exit", vec![int_lit(0, span)], Ty::None, span),
                    ty: Ty::None,
                });
            }
            ("zb_math_log", 2) => (vec![Ty::Float, Ty::Float], "zb_math_log_base"),
            ("zb_random_seed", 0) => (Vec::new(), "zb_random_seed_clock"),
            ("zb_stringio_new", 0) => (Vec::new(), "zb_stringio_empty"),
            ("zb_struct_unpack", 2) | ("zb_struct_calcsize", 1) => {
                let Some(py::Expr::StringLiteral(fmt)) = args.first() else {
                    return unsupported("struct with a format that is not a literal", c);
                };
                let text = fmt.value.to_str();
                let Some((big, fields, size)) = stdlib::struct_format(text) else {
                    return unsupported(format!("the struct format {text:?}"), c);
                };
                if zb == "zb_struct_calcsize" {
                    return Ok(Val {
                        node: int_lit(size, span),
                        ty: Ty::Int,
                    });
                }
                return self.struct_unpack(&args[1], big, &fields, size, span);
            }
            // The one codec here is hex, named by a literal.
            ("zb_codecs_decode", 2) => {
                let Some(py::Expr::StringLiteral(codec)) = args.get(1) else {
                    return unsupported("codecs.decode with a codec that is not a literal", c);
                };
                let codec = codec.value.to_str().to_lowercase();
                if codec != "hex" && codec != "hex_codec" {
                    return unsupported(format!("the `{codec}` codec"), c);
                }
                let data = self.expr_as(&args[0], Ty::Bytes)?;
                return Ok(Val {
                    node: call("zb_bytes_from_hex", vec![data], Ty::Bytes, span),
                    ty: Ty::Bytes,
                });
            }
            ("zb_random_seed", 1) if matches!(&args[0], py::Expr::NoneLiteral(_)) => {
                return Ok(Val {
                    node: call("zb_random_seed_clock", vec![], Ty::None, span),
                    ty: Ty::None,
                });
            }
            ("zb_random_randrange", 2) => (vec![Ty::Int, Ty::Int], "zb_random_randrange2"),
            ("zb_random_randrange", 3) => (vec![Ty::Int, Ty::Int, Ty::Int], "zb_random_randrange3"),
            ("zb_list_reduce", 2) => (
                vec![Ty::Object, Ty::List(Elem::Object)],
                "zb_list_reduce_first",
            ),
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
            // The function `reduce` applies: a value it can be called
            // through, whatever spells it.
            if zb.starts_with("zb_list_reduce") && lowered.is_empty() {
                lowered.push(self.callable_value(a)?);
                continue;
            }
            let v = self.expr(a)?;
            let node = match (v.ty, want) {
                (Ty::Object, Ty::Float) => call("zb_any_float", vec![v.node], Ty::Float, span),
                (Ty::Object, Ty::Int) => call("zb_any_int", vec![v.node], Ty::Int, span),
                // A sequence of dynamic values: a string's characters,
                // a typed list boxed.
                (t, Ty::List(Elem::Object)) if t != Ty::List(Elem::Object) => {
                    self.iterable(v, span)
                }
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
    /// A dict key and the suffix of the dict functions that take it as
    /// it is: a string or a tuple of known shape goes unboxed.
    fn dict_key(&mut self, e: &py::Expr) -> Result<(Node, String)> {
        match self.ty_of(e) {
            Ty::Str => Ok((self.expr_as(e, Ty::Str)?, "_str".to_string())),
            Ty::Tuple(k) => {
                let v = self.expr(e)?;
                Ok((v.node, format!("_{}", types::tuple_suffix(k))))
            }
            _ => Ok((self.expr_as(e, Ty::Object)?, String::new())),
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
        if let py::Expr::Call(rc) = e
            && types::is_name(&rc.func, "range")
            && rc.arguments.keywords.is_empty()
        {
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
        const BUILTINS: [&str; 9] = [
            "str", "int", "float", "bool", "len", "abs", "repr", "type", "eval",
        ];
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
        let arity = match imported {
            Some((stdlib::Member::Func { params, .. }, _)) => Some(params.len()),
            Some((stdlib::Member::Binary(_), _)) => Some(2),
            _ => None,
        };
        if let (Some(arity), Some((_, source))) = (arity, imported) {
            let args: Vec<String> = (0..arity).map(|i| format!("a{i}")).collect();
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
                    return unsupported(format!("the keyword argument `{other}` here"), c);
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

    /// A library result read as `ty`: a call that can raise is checked
    /// before its value is read, so a raised lookup is not also the
    /// TypeError of reading nothing as the value.
    fn read_as(&mut self, v: Val, ty: Ty, span: Span) -> Node {
        let fallible = match &v.node.node {
            TypedExpression::Call(c) => self.is_fallible_callee(&c.callee),
            _ => false,
        };
        let v = if fallible && self.guards && v.ty != ty {
            self.guard(v, span)
        } else {
            v
        };
        self.coerce(v, ty)
    }

    fn is_fallible_callee(&self, callee: &Node) -> bool {
        match &callee.node {
            TypedExpression::Variable(n) => n
                .resolve_global()
                .is_some_and(|name| self.is_fallible_name(&name)),
            _ => false,
        }
    }

    /// Whether the library function `name` can raise. The functions
    /// generated for a tuple shape raise where their dynamic-list
    /// counterparts do; reading a tuple back and ordering tuples can.
    fn is_fallible_name(&self, name: &str) -> bool {
        if self.module.fallible.contains(name) {
            return true;
        }
        if let Some(rest) = name.strip_prefix("zb_tuple_") {
            return rest.starts_with("read_") || rest.starts_with("lt_");
        }
        if let Some(rest) = name.strip_prefix("zb_list_")
            && let Some((op, suffix)) = rest.rsplit_once('_')
            && suffix.starts_with('t')
            && suffix[1..].chars().all(|c| c.is_ascii_digit())
        {
            return self.module.fallible.contains(&format!("zb_list_{op}_any"));
        }
        // An array storage kind's functions are generated with the
        // program and raise where the library's kinds do; narrowing and
        // the checked unbox raise on their own.
        if let Some(rest) = name.strip_prefix("zb_list_")
            && let Some((op, suffix)) = rest.rsplit_once('_')
            && zyntax_builtins::Kind::ALL
                .iter()
                .any(|k| !zyntax_builtins::Kind::LIBRARY.contains(k) && k.suffix() == suffix)
        {
            return matches!(op, "narrow" | "from_wide" | "unbox_tagged")
                || self.module.fallible.contains(&format!("zb_list_{op}_i64"))
                || self.module.fallible.contains(&format!("zb_list_{op}_f64"));
        }
        // A set or dict probed by a shape hashes the fields as their
        // boxes would be hashed, and raises where the boxed lookup does.
        if let Some(suffix) = name.strip_prefix("zb_set_contains_")
            && suffix.starts_with('t')
            && suffix[1..].chars().all(|c| c.is_ascii_digit())
        {
            return self.module.fallible.contains("zb_set_contains");
        }
        if let Some(rest) = name.strip_prefix("zb_dict_")
            && let Some((op, suffix)) = rest.rsplit_once('_')
            && suffix.starts_with('t')
            && suffix[1..].chars().all(|c| c.is_ascii_digit())
        {
            return self.module.fallible.contains(&format!("zb_dict_{op}"));
        }
        false
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
            py::Expr::BytesLiteral(b) => {
                let bytes: Vec<u8> = b.value.bytes().collect();
                self.bytes_lit(&bytes, span)
            }
            py::Expr::Name(n)
                if !self.locals.vars.contains_key(n.id.as_str())
                    && matches!(
                        n.id.as_str(),
                        "int"
                            | "float"
                            | "str"
                            | "bytes"
                            | "bool"
                            | "list"
                            | "tuple"
                            | "dict"
                            | "set"
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
            // The path of the source file this code is written in.
            py::Expr::Name(n) if n.id.as_str() == "__file__" && !self.is_variable("__file__") => {
                let path = self
                    .module
                    .file_names
                    .get(current_file() as usize)
                    .cloned()
                    .unwrap_or_default();
                Val {
                    node: str_lit(&path, span),
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
            py::Expr::Name(n) if self.comp_symbols.contains_key(n.id.as_str()) => Val {
                node: var(self.local_symbol(n.id.as_str()), ty, span),
                ty,
            },
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
            // Any other builtin as a value: the name it is bound to is
            // typed as the builtin, so a call through the name is the
            // builtin's own; the record is for a call that is not.
            py::Expr::Name(n)
                if !self.is_variable(n.id.as_str())
                    && !self.module.class_index.contains_key(n.id.as_str())
                    && let Some(k) = types::builtin_index(n.id.as_str()) =>
            {
                let held = Val {
                    node: str_lit(n.id.as_str(), span),
                    ty: Ty::Str,
                };
                let env = self.list_of(vec![held], Elem::Object, span);
                Val {
                    node: call(
                        "zb_func_new",
                        vec![
                            code_of("zb_builtin_value_call", span),
                            int_lit(zyntax_builtins::functions::VARIADIC_ARITY, span),
                            env,
                        ],
                        Ty::Object,
                        span,
                    ),
                    ty: Ty::Builtin(k),
                }
            }
            // A module class named as a value constructs when called.
            py::Expr::Name(n)
                if !self.is_variable(n.id.as_str())
                    && let Some(&k) = self.module.class_index.get(n.id.as_str()) =>
            {
                self.class_value(k, span)?
            }
            py::Expr::Name(n) => Val {
                node: var(self.local_symbol(n.id.as_str()), ty, span),
                ty,
            },
            py::Expr::Lambda(l) => self.lambda(l, span)?,
            py::Expr::Generator(g) => self.generator_expr(g, span)?,
            py::Expr::Attribute(a) => {
                if let Some(member) = self.module_member_of(&a.value, a.attr.as_str()) {
                    return self.member_value(member, a.attr.as_str(), e, span);
                }
                // `C.X`: the class's attribute.
                if let Some(k) = self.module.class_of_expr(&a.value) {
                    let attr = self.class_attr_of(k, a.attr.as_str(), span)?;
                    return Ok(self.class_attr_read(&attr, span));
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
            // `b"..." % values` with a literal format.
            py::Expr::BinOp(b)
                if b.op == py::Operator::Mod && matches!(&*b.left, py::Expr::BytesLiteral(_)) =>
            {
                let py::Expr::BytesLiteral(l) = &*b.left else {
                    unreachable!()
                };
                let template: Vec<u8> = l.value.bytes().collect();
                self.percent_format_bytes(&template, &b.right, span)?
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
            // A literal of a shape is the value struct; one of no shape
            // (empty, or spreading a sequence) is a boxed tuple.
            py::Expr::Tuple(t) => {
                let mut items = Vec::with_capacity(t.elts.len());
                for e in &t.elts {
                    items.push(self.expr(e)?);
                }
                match ty.tuple_elems() {
                    Some(shape) if shape.len() == items.len() => Val {
                        node: self.tuple_of_items(items, ty, span),
                        ty,
                    },
                    _ => {
                        let list = self.list_of(items, Elem::Object, span);
                        Val {
                            node: call("zb_box_tuple", vec![list], Ty::Object, span),
                            ty: Ty::Object,
                        }
                    }
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
                let dict_ty = match ty {
                    Ty::Dict(_) => ty,
                    _ => types::dynamic_dict(),
                };
                Val {
                    node: call(maker, vec![pairs], dict_ty, span),
                    ty: dict_ty,
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
                    // An instance answers through its class's dunder.
                    Ty::Class(k) => {
                        let name = match op {
                            UnaryOp::Minus => "__neg__",
                            UnaryOp::BitNot => "__invert__",
                            _ => "__pos__",
                        };
                        match self.dunder(k as usize, name, operand.node, vec![], span) {
                            Some(r) => r,
                            None => {
                                return Err(Error::unsupported_span(
                                    format!(
                                        "unary `{}` on {}, which defines no {name}",
                                        match op {
                                            UnaryOp::Minus => "-",
                                            UnaryOp::BitNot => "~",
                                            _ => "+",
                                        },
                                        self.module.classes[k as usize].name
                                    ),
                                    span,
                                ));
                            }
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
        // `3 * d` with `d` an instance: the reflected method on the right.
        if let Ty::Class(k) = right.ty
            && left.ty != Ty::Object
        {
            let name = types::reflected_dunder_name(op);
            if let Some(r) = self.dunder(k as usize, name, right.node, vec![left], span) {
                return Ok(r);
            }
            return Err(Error::unsupported_span(
                format!(
                    "`{}` with {} on the right, which defines no {name}",
                    op_text(op),
                    self.module.classes[k as usize].name
                ),
                span,
            ));
        }
        // A sequence repeated by a dynamic count: the count is an int
        // or the TypeError Python raises, and the repeat stays typed.
        let repeatable = |t: Ty| matches!(t, Ty::List(_) | Ty::Str | Ty::Bytes | Ty::Tuple(_));
        if op == py::Operator::Mult
            && ((repeatable(left.ty) && right.ty == Ty::Object)
                || (left.ty == Ty::Object && repeatable(right.ty)))
        {
            let (left, right) = if right.ty == Ty::Object {
                let n = self.coerce(right, Ty::Int);
                (
                    left,
                    Val {
                        node: n,
                        ty: Ty::Int,
                    },
                )
            } else {
                let n = self.coerce(left, Ty::Int);
                (
                    Val {
                        node: n,
                        ty: Ty::Int,
                    },
                    right,
                )
            };
            return self.arithmetic(op, left, right, right_expr, span);
        }
        let ty = types::binop(op, left.ty, right.ty, right_expr);
        // Strings have their own operators.
        if left.ty == Ty::Str || right.ty == Ty::Str {
            match (op, left.ty, right.ty) {
                (py::Operator::Add, Ty::Str, Ty::Str) => {
                    return Ok(Val {
                        node: binary(BinaryOp::Add, left.node, right.node, Ty::Str, span),
                        ty: Ty::Str,
                    });
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
        // Bytes concatenate, repeat and format.
        if left.ty == Ty::Bytes || right.ty == Ty::Bytes {
            match (op, left.ty, right.ty) {
                // Concatenation copies bytes, as it does for strings.
                (py::Operator::Add, Ty::Bytes, Ty::Bytes) => {
                    return Ok(Val {
                        node: binary(BinaryOp::Add, left.node, right.node, Ty::Bytes, span),
                        ty: Ty::Bytes,
                    });
                }
                (py::Operator::Mult, Ty::Bytes, Ty::Int | Ty::Bool) => {
                    let n = self.coerce(right, Ty::Int);
                    return Ok(Val {
                        node: call("zb_bytes_repeat", vec![left.node, n], Ty::Bytes, span),
                        ty: Ty::Bytes,
                    });
                }
                (py::Operator::Mult, Ty::Int | Ty::Bool, Ty::Bytes) => {
                    let n = self.coerce(left, Ty::Int);
                    return Ok(Val {
                        node: call("zb_bytes_repeat", vec![right.node, n], Ty::Bytes, span),
                        ty: Ty::Bytes,
                    });
                }
                // A computed format: the values boxed, converted at run
                // time.
                (py::Operator::Mod, Ty::Bytes, _) => {
                    let args = match right.ty {
                        Ty::Tuple(_) => self.coerce(right, Ty::List(Elem::Object)),
                        _ => {
                            let one = self.coerce(right, Ty::Object);
                            node(
                                TypedExpression::Array(vec![one]),
                                Ty::List(Elem::Object),
                                span,
                            )
                        }
                    };
                    return Ok(Val {
                        node: call("zb_bytes_format", vec![left.node, args], Ty::Bytes, span),
                        ty: Ty::Bytes,
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
        // Sequences concatenate and repeat. An array concatenates with
        // an array of its own typecode and nothing else.
        match (op, left.ty, right.ty) {
            (py::Operator::Add, Ty::List(a), Ty::List(b))
                if a != b && (a.code().is_some() || b.code().is_some()) =>
            {
                let message = match (a.code(), b.code()) {
                    (Some(_), Some(_)) => "bad argument type for built-in operation".to_string(),
                    (Some(_), None) => "can only append array (not \"list\") to array".to_string(),
                    _ => "can only concatenate list (not \"array.array\") to list".to_string(),
                };
                let node =
                    self.raised_value(vec![left.node, right.node], "TypeError", &message, ty, span);
                return Ok(Val { node, ty });
            }
            (py::Operator::Add, Ty::List(e), Ty::List(_)) if ty == left.ty => {
                return Ok(Val {
                    node: call(&list_fn("concat", e), vec![left.node, right.node], ty, span),
                    ty,
                });
            }
            // Two shapes concatenate into the wider one.
            (py::Operator::Add, Ty::Tuple(_), Ty::Tuple(_)) if matches!(ty, Ty::Tuple(_)) => {
                let (mut fields, mut pre) = self.tuple_fields(left, span, Hold::Ahead);
                let (more, mut pre_right) = self.tuple_fields(right, span, Hold::Ahead);
                fields.extend(more);
                pre.append(&mut pre_right);
                let value = self.tuple_of_items(fields, ty, span);
                return Ok(Val {
                    node: if pre.is_empty() {
                        value
                    } else {
                        Self::block_value(pre, value, ty, span)
                    },
                    ty,
                });
            }
            // A repeated tuple is a boxed tuple of the repeated elements.
            (py::Operator::Mult, Ty::Tuple(_), Ty::Int | Ty::Bool)
            | (py::Operator::Mult, Ty::Int | Ty::Bool, Ty::Tuple(_)) => {
                let (seq, times) = if matches!(left.ty, Ty::Tuple(_)) {
                    (left, right)
                } else {
                    (right, left)
                };
                let items = self.coerce(seq, Ty::List(Elem::Object));
                let n = self.coerce(times, Ty::Int);
                let repeated = call(
                    &list_fn("repeat", Elem::Object),
                    vec![items, n],
                    Ty::List(Elem::Object),
                    span,
                );
                return Ok(Val {
                    node: call("zb_box_tuple", vec![repeated], Ty::Object, span),
                    ty: Ty::Object,
                });
            }
            (py::Operator::Mult, Ty::List(_), Ty::Int | Ty::Bool)
            | (py::Operator::Mult, Ty::Int | Ty::Bool, Ty::List(_)) => {
                let (seq, times) = if matches!(left.ty, Ty::List(_)) {
                    (left, right)
                } else {
                    (right, left)
                };
                let Ty::List(elem) = seq.ty else {
                    unreachable!()
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
                (Ty::Object, Ty::Object)
                    if matches!(
                        op,
                        py::Operator::Add
                            | py::Operator::Sub
                            | py::Operator::Mult
                            | py::Operator::Div
                    ) =>
                {
                    // Numeric containers lose their element type when boxed.
                    // Keep the common float operation in the caller and use
                    // the full dispatcher for every other pair of values.
                    const FLOAT_CATEGORY: i64 = 4;
                    let bin = match op {
                        py::Operator::Add => BinaryOp::Add,
                        py::Operator::Sub => BinaryOp::Sub,
                        py::Operator::Mult => BinaryOp::Mul,
                        py::Operator::Div => BinaryOp::Div,
                        _ => unreachable!(),
                    };
                    let mut pre = Vec::new();
                    let l = self.hold(left, &mut pre, span);
                    let r = self.hold(right, &mut pre, span);
                    let is_float = |v: &Val| {
                        binary(
                            BinaryOp::Eq,
                            call("zb_any_category", vec![v.node.clone()], Ty::Int, span),
                            int_lit(FLOAT_CATEGORY, span),
                            Ty::Bool,
                            span,
                        )
                    };
                    let both_float =
                        binary(BinaryOp::And, is_float(&l), is_float(&r), Ty::Bool, span);
                    let lf = call("zb_box_get_f64", vec![l.node.clone()], Ty::Float, span);
                    let rf = call("zb_box_get_f64", vec![r.node.clone()], Ty::Float, span);
                    let fast = call(
                        "zb_box_f64",
                        vec![binary(bin, lf, rf, Ty::Float, span)],
                        Ty::Object,
                        span,
                    );
                    let slow = call("zb_any_arith", vec![code, l.node, r.node], Ty::Object, span);
                    let mut slow_pre = Vec::new();
                    let slow = self.hold(
                        Val {
                            node: slow,
                            ty: Ty::Object,
                        },
                        &mut slow_pre,
                        span,
                    );
                    slow_pre.push(self.pending_check(span));
                    let chosen = self.conditional_value(
                        both_float,
                        (Vec::new(), fast),
                        (slow_pre, slow.node),
                        Ty::Object,
                        span,
                        &mut pre,
                    );
                    self.hoisted.extend(pre);
                    chosen
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
                if name == "__ne__"
                    && let Some(r) = self.dunder(
                        k as usize,
                        "__eq__",
                        left.node.clone(),
                        vec![right.clone()],
                        span,
                    )
                {
                    let t = self.truthy(r);
                    return Ok(negate(t));
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
                } else if let Ty::List(e @ Elem::Array(c)) = right.ty {
                    self.array_search(left, right.node, e, c, "contains", span)?
                } else if let Ty::List(e) = right.ty {
                    let item = self.elem_arg(left, e);
                    call(
                        &list_fn("contains", e),
                        vec![right.node, item],
                        Ty::Bool,
                        span,
                    )
                } else if right.ty == Ty::Set {
                    // A tuple of known shape probes by its fields, no box.
                    if let Ty::Tuple(k) = left.ty {
                        call(
                            &format!("zb_set_contains_{}", types::tuple_suffix(k)),
                            vec![right.node, left.node],
                            Ty::Bool,
                            span,
                        )
                    } else {
                        let item = self.coerce(left, Ty::Object);
                        call("zb_set_contains", vec![right.node, item], Ty::Bool, span)
                    }
                } else if matches!(right.ty, Ty::Tuple(_)) {
                    // Through the list of the elements, of the one kind
                    // they all are when they are.
                    let e = Elem::of(right.ty.element().unwrap_or(Ty::Object));
                    let e = if left.ty == e.ty() { e } else { Elem::Object };
                    let items = self.coerce(right, Ty::List(e));
                    let item = self.elem_arg(left, e);
                    call(&list_fn("contains", e), vec![items, item], Ty::Bool, span)
                } else if matches!(right.ty, Ty::Dict(_)) {
                    let (item, by) = match left.ty {
                        Ty::Str => (left.node, "_str".to_string()),
                        Ty::Tuple(k) => (left.node, format!("_{}", types::tuple_suffix(k))),
                        _ => (self.coerce(left, Ty::Object), String::new()),
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
                    (Ty::List(_) | Ty::Tuple(_) | Ty::Dict(_) | Ty::Set, _)
                    | (_, Ty::List(_) | Ty::Tuple(_) | Ty::Dict(_) | Ty::Set) => {
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
                    call("zb_set_len", vec![a], Ty::Int, span),
                    call("zb_set_len", vec![b], Ty::Int, span),
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
                    ));
                }
            });
        }
        // A tuple and a list are never equal and have no order.
        let tuple_sides = (
            matches!(left.ty, Ty::Tuple(_)),
            matches!(right.ty, Ty::Tuple(_)),
        );
        if matches!(
            (tuple_sides, left.ty, right.ty),
            ((true, false), _, Ty::List(_)) | ((false, true), Ty::List(_), _)
        ) {
            let lit = |b: bool| {
                node(
                    TypedExpression::Literal(TypedLiteral::Bool(b)),
                    Ty::Bool,
                    span,
                )
            };
            return Ok(match op {
                py::CmpOp::Eq => Self::after_none(
                    left.node,
                    Self::after_none(right.node, lit(false), Ty::Bool),
                    Ty::Bool,
                ),
                py::CmpOp::NotEq => Self::after_none(
                    left.node,
                    Self::after_none(right.node, lit(true), Ty::Bool),
                    Ty::Bool,
                ),
                _ => {
                    return Err(Error::unsupported_span(
                        "ordering a tuple against a list".to_string(),
                        span,
                    ));
                }
            });
        }
        // Two sequences compare element by element; tuples as the lists
        // of their elements.
        let left = self.tuple_as_list(left);
        let right = self.tuple_as_list(right);
        let seq_kind = |t: Ty| match t {
            Ty::List(e) => Some(e),
            _ => None,
        };
        // An array equals no list, and is ordered against none; two
        // arrays reading as the same number compare as those numbers.
        if let (Some(e), Some(f)) = (seq_kind(left.ty), seq_kind(right.ty))
            && e != f
            && (e.code().is_some() || f.code().is_some())
        {
            let (mut left, mut right) = (left, right);
            for side in [&mut left, &mut right] {
                if let Some(c) = seq_kind(side.ty).and_then(Elem::code) {
                    // As the list of the numbers it reads as: the same
                    // header where the storage is already that wide.
                    let wide = Ty::List(Elem::of(c.item()));
                    if c.narrows() {
                        let e = Elem::Array(c);
                        side.node =
                            call(&list_fn("to_wide", e), vec![side.node.clone()], wide, span);
                    }
                    side.ty = wide;
                }
            }
            let both_arrays = e.code().is_some() && f.code().is_some();
            if both_arrays && left.ty == right.ty {
                let Ty::List(w) = left.ty else { unreachable!() };
                let eq = |l: Node, r: Node| call(&list_fn("eq", w), vec![l, r], Ty::Bool, span);
                let lt = |l: Node, r: Node| call(&list_fn("lt", w), vec![l, r], Ty::Bool, span);
                let (l, r) = (left.node, right.node);
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
            if !both_arrays {
                let lit = |b: bool| {
                    node(
                        TypedExpression::Literal(TypedLiteral::Bool(b)),
                        Ty::Bool,
                        span,
                    )
                };
                let effects = vec![
                    TypedNode::new(
                        TypedStatement::Expression(Box::new(left.node.clone())),
                        Type::Unknown,
                        span,
                    ),
                    TypedNode::new(
                        TypedStatement::Expression(Box::new(right.node.clone())),
                        Type::Unknown,
                        span,
                    ),
                ];
                return Ok(match op {
                    py::CmpOp::Eq => Self::block_value(effects, lit(false), Ty::Bool, span),
                    py::CmpOp::NotEq => Self::block_value(effects, lit(true), Ty::Bool, span),
                    _ => {
                        let (a, b) = if e.code().is_some() {
                            ("array.array", "list")
                        } else {
                            ("list", "array.array")
                        };
                        self.raised_value(
                            vec![left.node, right.node],
                            "TypeError",
                            &format!(
                                "'{}' not supported between instances of '{a}' and '{b}'",
                                match op {
                                    py::CmpOp::Lt => "<",
                                    py::CmpOp::Gt => ">",
                                    py::CmpOp::LtE => "<=",
                                    _ => ">=",
                                }
                            ),
                            Ty::Bool,
                            span,
                        )
                    }
                });
            }
            // Two arrays of different numbers compare as dynamic values.
            let (l, r) = (
                self.coerce(left, Ty::List(Elem::Object)),
                self.coerce(right, Ty::List(Elem::Object)),
            );
            let eq = |l: Node, r: Node| call("zb_list_eq_any", vec![l, r], Ty::Bool, span);
            let lt = |l: Node, r: Node| call("zb_list_lt_any", vec![l, r], Ty::Bool, span);
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
        if left.ty == Ty::Bytes && right.ty == Ty::Bytes {
            let eq = || {
                binary(
                    BinaryOp::Ne,
                    call("zb_bytes_eq", vec![left.node, right.node], Ty::Int, span),
                    int_lit(0, span),
                    Ty::Bool,
                    span,
                )
            };
            return match op {
                py::CmpOp::Eq => Ok(eq()),
                py::CmpOp::NotEq => Ok(negate(eq())),
                _ => Err(Error::unsupported_span("ordering bytes".to_string(), span)),
            };
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
            py::CmpOp::LtE => call("zb_any_le", vec![l, r], Ty::Bool, span),
            py::CmpOp::GtE => call("zb_any_le", vec![r, l], Ty::Bool, span),
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
        // `globals()[name]`: the module variable the string names.
        if let py::Expr::Call(c) = &*sub.value
            && matches!(&*c.func, py::Expr::Name(n) if n.id.as_str() == "globals" && !self.is_variable("globals"))
            && c.arguments.args.is_empty()
        {
            return self.globals_lookup(&sub.slice, span);
        }
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
                // Literal bounds cut a shape out of a shape; any other
                // slice is a tuple of a length only the runtime knows.
                Ty::Tuple(k) => {
                    let shape = types::tuple_shape(k);
                    if let Some(picked) = types::tuple_slice(shape.len(), sl)
                        && ty == types::tuple_of(picked.iter().map(|&i| shape[i]).collect())
                    {
                        let (fields, pre) = self.tuple_fields(seq, span, Hold::Ahead);
                        let items = picked
                            .iter()
                            .map(|&i| Val {
                                node: fields[i].node.clone(),
                                ty: fields[i].ty,
                            })
                            .collect();
                        let value = self.tuple_of_items(items, ty, span);
                        return Ok(Val {
                            node: if pre.is_empty() {
                                value
                            } else {
                                Self::block_value(pre, value, ty, span)
                            },
                            ty,
                        });
                    }
                    let items = self.coerce(seq, Ty::List(Elem::Object));
                    let sliced = call(
                        "zb_list_slice_any",
                        vec![items, start, stop, step, mask],
                        Ty::List(Elem::Object),
                        span,
                    );
                    call("zb_box_tuple", vec![sliced], Ty::Object, span)
                }
                Ty::Str => call(
                    "zb_str_slice",
                    vec![seq.node, start, stop, step, mask],
                    Ty::Str,
                    span,
                ),
                Ty::Bytes => call(
                    "zb_bytes_slice",
                    vec![seq.node, start, stop, step, mask],
                    Ty::Bytes,
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
            // A literal index into a shape is that field, read as the
            // shape stores it.
            Ty::Tuple(k)
                if let Some(i) = types::constant_index(&sub.slice, types::tuple_shape(k).len()) =>
            {
                let field = types::tuple_shape(k)[i].settled();
                let stored = tuple_field_storage(field);
                let read = Val {
                    node: slot(seq.node, i, stored, span),
                    ty: stored,
                };
                Ok(Val {
                    node: self.trusted(read, field),
                    ty: field,
                })
            }
            Ty::List(_) | Ty::Tuple(_) | Ty::Str | Ty::Bytes => {
                let index = self.expr_as(&sub.slice, Ty::Int)?;
                Ok(self.index_value(seq, index, ty, span))
            }
            // The stored value is dynamic; it is read as the shape says.
            Ty::Dict(_) => {
                let (key, by) = self.dict_key(&sub.slice)?;
                let value = Val {
                    node: call(
                        &format!("zb_dict_get{by}"),
                        vec![seq.node, key],
                        Ty::Object,
                        span,
                    ),
                    ty: Ty::Object,
                };
                Ok(Val {
                    node: self.read_as(value, ty, span),
                    ty,
                })
            }
            // An instance answers through its class's `__getitem__`.
            Ty::Class(k) => {
                let key = self.expr(&sub.slice)?;
                match self.dunder(k as usize, "__getitem__", seq.node, vec![key], span) {
                    Some(r) => Ok(r),
                    None => Err(Error::unsupported_span(
                        format!(
                            "subscript of {}, which defines no __getitem__",
                            self.module.classes[k as usize].name
                        ),
                        span,
                    )),
                }
            }
            _ => {
                let key = self.expr(&sub.slice)?;
                let o = self.coerce(seq, Ty::Object);
                let node = if key.ty == Ty::Int {
                    call("zb_any_getitem_i64", vec![o, key.node], Ty::Object, span)
                } else {
                    let key = self.coerce(key, Ty::Object);
                    call("zb_any_getitem", vec![o, key], Ty::Object, span)
                };
                Ok(Val {
                    node,
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
            Produce::Dict(..) => types::dynamic_dict(),
            Produce::Yield(_) => Ty::None,
        };
        let out = self.temp();
        // Loop variables are the comprehension's own; they shadow the
        // function's for the body and are forgotten after.
        let saved_vars = self.locals.vars.clone();
        let saved_symbols = self.comp_symbols.clone();
        let mut target_names = BTreeSet::new();
        for g in generators {
            let mut names = HashMap::default();
            bind_names(&mut names, &g.target, Ty::Object);
            target_names.extend(names.into_keys());
        }
        for name in target_names {
            let symbol = self.temp();
            self.comp_symbols.insert(name, symbol);
        }
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
            Produce::Set(_) => Some(call("zb_set_new", vec![], Ty::Set, span)),
            Produce::Dict(..) => Some(call("zb_dict_new", vec![], ty, span)),
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
            let item_ty = self.typer().item_ty(&g.iter).settled();
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
        self.comp_symbols = saved_symbols;
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
            Ty::List(e @ Elem::Array(c)) if !matches!(name, "append" | "pop" | "insert") => {
                let list = receiver.node;
                let node = match name {
                    // The storage as it lies in memory; the host reads
                    // it through the header, at the element width.
                    "tobytes" => {
                        expect(0, self)?;
                        let bytes = Ty::List(Elem::Array(types::Code::UB));
                        let storage = cast(list, bytes, span);
                        return Ok(Val {
                            node: call(
                                "zb_bytes_of_storage",
                                vec![storage, int_lit(c.itemsize(), span)],
                                Ty::Bytes,
                                span,
                            ),
                            ty: Ty::Bytes,
                        });
                    }
                    // A number the storage cannot hold is in no array.
                    "count" => {
                        expect(1, self)?;
                        let v = self.expr(&args[0])?;
                        self.array_search(v, list, e, c, "count", span)?
                    }
                    "index" | "remove" => {
                        expect(1, self)?;
                        let v = self.expr(&args[0])?;
                        let at = self.array_search(v, list.clone(), e, c, "index_or_neg", span)?;
                        let mut pre = Vec::new();
                        let held = self.hold(
                            Val {
                                node: at,
                                ty: Ty::Int,
                            },
                            &mut pre,
                            span,
                        );
                        let mut raise = Vec::new();
                        self.raise_named(
                            "ValueError",
                            str_lit(&format!("array.{name}(x): x not in array"), span),
                            span,
                            &mut raise,
                        );
                        pre.push(TypedNode::new(
                            TypedStatement::If(TypedIf {
                                condition: Box::new(binary(
                                    BinaryOp::Lt,
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
                        if name == "index" {
                            held.node
                        } else {
                            method_call(list, "remove_at", vec![held.node], e.ty(), span)
                        }
                    }
                    // From an array of the typecode, or from anything
                    // else whose items become elements; an array of
                    // another typecode is refused as CPython refuses it.
                    "extend" | "fromlist" => {
                        expect(1, self)?;
                        let other = self.expr(&args[0])?;
                        match other.ty {
                            Ty::List(f) if f.code().is_some() && f != e => self.raised_value(
                                vec![list, other.node],
                                "TypeError",
                                "can only extend with array of same kind",
                                Ty::None,
                                span,
                            ),
                            Ty::List(_) | Ty::Tuple(_) | Ty::Object if name == "extend" => {
                                let items = self.coerce(other, receiver.ty);
                                call(&list_fn("extend", e), vec![list, items], Ty::None, span)
                            }
                            Ty::List(_) => {
                                let items = self.coerce(other, receiver.ty);
                                call(&list_fn("extend", e), vec![list, items], Ty::None, span)
                            }
                            _ if name == "fromlist" => self.raised_value(
                                vec![list, other.node],
                                "TypeError",
                                "arg must be list",
                                Ty::None,
                                span,
                            ),
                            _ => {
                                let items = Val {
                                    node: self.iterable(other, span),
                                    ty: Ty::List(Elem::Object),
                                };
                                let items = self.coerce(items, receiver.ty);
                                call(&list_fn("extend", e), vec![list, items], Ty::None, span)
                            }
                        }
                    }
                    "reverse" => {
                        expect(0, self)?;
                        call(&list_fn(name, e), vec![list], ty, span)
                    }
                    "tolist" => {
                        expect(0, self)?;
                        self.coerce(
                            Val {
                                node: list,
                                ty: receiver.ty,
                            },
                            ty,
                        )
                    }
                    "frombytes" | "tofile" | "fromfile" | "tounicode" | "fromunicode"
                    | "buffer_info" | "byteswap" => {
                        return Err(Error::unsupported_span(format!("array.{name}"), span));
                    }
                    _ => {
                        return Err(Error::unsupported_span(
                            format!("array.{name}, which an array does not have"),
                            span,
                        ));
                    }
                };
                Ok(Val { node, ty })
            }
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
                    // An array extends from a list as a list does.
                    "extend" | "fromlist" => {
                        expect(1, self)?;
                        let other = self.expr_as(&args[0], Ty::List(e))?;
                        call(&list_fn("extend", e), vec![list, other], Ty::None, span)
                    }
                    "sort" | "reverse" | "copy" => {
                        expect(0, self)?;
                        call(&list_fn(name, e), vec![list], ty, span)
                    }
                    "tolist" => {
                        expect(0, self)?;
                        call(&list_fn("copy", e), vec![list], ty, span)
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
                        );
                    }
                };
                Ok(Val { node, ty })
            }
            // The stored keys and values are dynamic; what comes out is
            // read as the shape says.
            Ty::Dict(_) => {
                let d = receiver.node;
                let dict_ty = receiver.ty;
                let mut produced = Ty::Object;
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
                    ("keys", 0) => {
                        produced = Ty::List(Elem::Object);
                        call("zb_dict_keys", vec![d], produced, span)
                    }
                    ("values", 0) => {
                        produced = Ty::List(Elem::Object);
                        call("zb_dict_values", vec![d], produced, span)
                    }
                    ("items", 0) => {
                        produced = Ty::List(Elem::Object);
                        call("zb_dict_items", vec![d], produced, span)
                    }
                    ("copy", 0) => {
                        produced = dict_ty;
                        call("zb_dict_copy", vec![d], dict_ty, span)
                    }
                    ("clear", 0) => {
                        produced = Ty::None;
                        method_call(d, "clear", vec![], Ty::None, span)
                    }
                    ("update", 1) => {
                        produced = Ty::None;
                        let other = self.expr_as(&args[0], types::dynamic_dict())?;
                        call("zb_dict_update", vec![d, other], Ty::None, span)
                    }
                    _ => {
                        return Err(Error::unsupported_span(
                            format!("dict.{name} with {} argument(s)", args.len()),
                            span,
                        ));
                    }
                };
                let value = Val { node, ty: produced };
                Ok(Val {
                    node: self.read_as(value, ty, span),
                    ty,
                })
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
                    ("clear", 0) => call("zb_set_clear", vec![st], Ty::None, span),
                    ("copy", 0) => call("zb_set_copy", vec![st], Ty::Set, span),
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
                        ));
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
                    ("strip" | "lstrip" | "rstrip", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call(&format!("zb_str_{name}_chars"), vec![s, a], Ty::Str, span)
                    }
                    ("find", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call("zb_str_find", vec![s, a], Ty::Int, span)
                    }
                    ("index", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call("zb_str_index", vec![s, a], Ty::Int, span)
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
                    ("splitlines", 0) => {
                        call("zb_str_splitlines", vec![s], Ty::List(Elem::Str), span)
                    }
                    ("split", 1) => {
                        let a = self.coerce(lowered.pop().unwrap(), Ty::Str);
                        call("zb_str_split", vec![s, a], Ty::List(Elem::Str), span)
                    }
                    ("join", 1) => {
                        let items = self.coerce(lowered.pop().unwrap(), Ty::List(Elem::Str));
                        call("zb_str_join", vec![s, items], Ty::Str, span)
                    }
                    // A string is its UTF-8 already; any other encoding
                    // is refused below.
                    ("encode", 0) => s,
                    ("encode", 1) if self.is_utf8_name(&args[0]) => s,
                    _ => {
                        return Err(Error::unsupported_span(
                            format!("str.{name} with {} argument(s)", args.len()),
                            span,
                        ));
                    }
                };
                Ok(Val { node, ty })
            }
            Ty::Bytes => {
                let b = receiver.node;
                let node = match (name, args.len()) {
                    ("decode", 0) => call("zb_bytes_decode", vec![b], Ty::Str, span),
                    ("decode", 1) if self.is_utf8_name(&args[0]) => {
                        call("zb_bytes_decode", vec![b], Ty::Str, span)
                    }
                    // A digest is bytes; its spelling is theirs.
                    ("hex" | "hexdigest", 0) => call("zb_bytes_hex", vec![b], Ty::Str, span),
                    ("digest", 0) => b,
                    _ => {
                        return Err(Error::unsupported_span(
                            format!("bytes.{name} with {} argument(s)", args.len()),
                            span,
                        ));
                    }
                };
                Ok(Val { node, ty })
            }
            Ty::File(mode) => {
                let f = receiver.node;
                let node = match (name, args.len()) {
                    ("write", 1) => {
                        let data = self.expr_as(&args[0], mode.content())?;
                        call("zb_file_write", vec![f, data], Ty::None, span)
                    }
                    ("read", 0) => call("zb_file_read", vec![f], mode.content(), span),
                    ("read", 1) => {
                        let count = self.expr_as(&args[0], Ty::Int)?;
                        call("zb_file_read_n", vec![f, count], mode.content(), span)
                    }
                    ("getvalue", 0) => call("zb_file_getvalue", vec![f], mode.content(), span),
                    ("close", 0) => call("zb_file_close", vec![f], Ty::None, span),
                    ("flush", 0) => Self::after_none(
                        f,
                        node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                        Ty::None,
                    ),
                    _ => {
                        return Err(Error::unsupported_span(
                            format!("file.{name} with {} argument(s)", args.len()),
                            span,
                        ));
                    }
                };
                Ok(Val { node, ty })
            }
            // A tuple answers its two methods as the list of its elements.
            Ty::Tuple(_) if matches!(name, "count" | "index") => {
                expect(1, self)?;
                let v = self.expr(&args[0])?;
                let e = Elem::of(receiver.ty.element().unwrap_or(Ty::Object));
                let e = if v.ty == e.ty() { e } else { Elem::Object };
                let items = self.coerce(receiver, Ty::List(e));
                let v = self.elem_arg(v, e);
                Ok(Val {
                    node: call(&list_fn(name, e), vec![items, v], ty, span),
                    ty,
                })
            }
            _ => Err(Error::unsupported_span(
                format!("method `{name}` on a dynamic value"),
                span,
            )),
        }
    }

    /// `f(a, *rest)` with `rest` a dynamic value: its items, as many as
    /// the callee has parameters left, read into temporaries the call is
    /// then made with. A callee whose parameter count is not known, or
    /// a starred argument that is not last, is refused.
    fn spread_dynamic(&mut self, c: &py::ExprCall, span: Span) -> Result<Vec<py::Expr>> {
        let args = &c.arguments.args;
        let starred = args
            .iter()
            .filter(|a| matches!(a, py::Expr::Starred(_)))
            .count();
        let Some(py::Expr::Starred(s)) = args.last().filter(|_| starred == 1) else {
            return unsupported("a starred argument that is not the last", c);
        };
        let given = args.len() - 1;
        let Some(want) = self.spread_count(&c.func, given) else {
            return unsupported(
                "a starred argument to a call whose parameter count is not known",
                c,
            );
        };
        let value = self.expr(&s.value)?;
        let items = Val {
            node: self.iterable(value, span),
            ty: Ty::List(Elem::Object),
        };
        let mut pre = Vec::new();
        let held = self.hold(items, &mut pre, span);
        pre.push(TypedNode::new(
            TypedStatement::Expression(Box::new(call(
                "zb_list_expect_len_any",
                vec![held.node.clone(), int_lit(want as i64, span)],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        ));
        pre.push(self.pending_check(span));
        let mut out: Vec<py::Expr> = args[..given].to_vec();
        for i in 0..want {
            let item = Val {
                node: call(
                    "zb_list_get_unchecked_any",
                    vec![held.node.clone(), int_lit(i as i64, span)],
                    Ty::Object,
                    span,
                ),
                ty: Ty::Object,
            };
            let temp = self.hold(item, &mut pre, span);
            let TypedExpression::Variable(name) = temp.node.node else {
                unreachable!("a held value is a variable");
            };
            let id = name.resolve_global().expect("a temporary's name");
            self.locals.vars.insert(id.clone(), Ty::Object);
            out.push(py::Expr::Name(py::ExprName {
                node_index: Default::default(),
                range: s.range,
                id: py::name::Name::new(id),
                ctx: py::ExprContext::Load,
            }));
        }
        self.hoisted.extend(pre);
        Ok(out)
    }

    /// How many parameters a call of `func` with `given` positional
    /// arguments has left: from the function's or method's signature,
    /// or from the classes defining the method when they agree.
    fn spread_count(&self, func: &py::Expr, given: usize) -> Option<usize> {
        let left = |params: usize, skip: usize| params.checked_sub(skip + given);
        match func {
            py::Expr::Name(n) if !self.is_variable(n.id.as_str()) => {
                let name = n.id.as_str();
                if let Some(&k) = self.module.class_index.get(name) {
                    let (sig, _) = self.module.method_sig(k, "__init__")?;
                    return left(sig.params.len(), 1);
                }
                left(self.module.funcs.get(name)?.params.len(), 0)
            }
            py::Expr::Attribute(a) => match self.ty_of(&a.value) {
                Ty::Class(k) => {
                    let (sig, _) = self.module.method_sig(k as usize, a.attr.as_str())?;
                    left(sig.params.len(), 1)
                }
                Ty::Object => {
                    let mut counts = (0..self.module.classes.len())
                        .filter_map(|k| self.module.method_sig(k, a.attr.as_str()))
                        .map(|(sig, _)| sig.params.len());
                    let first = counts.next()?;
                    counts.all(|n| n == first).then_some(())?;
                    left(first, 1)
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// A call: `print`, a conversion builtin, or a function the module
    /// defines.
    fn call(&mut self, c: &py::ExprCall, ty: Ty, span: Span) -> Result<Val> {
        // `f(*t)` with `t` a tuple of known shape is `f(t[0], t[1], ...)`.
        if c.arguments
            .args
            .iter()
            .any(|a| matches!(a, py::Expr::Starred(_)))
        {
            let spread = match types::spread_starred(&c.arguments.args, |e| self.ty_of(e)) {
                Some(spread) => spread,
                None => self.spread_dynamic(c, span)?,
            };
            let expanded = py::ExprCall {
                arguments: py::Arguments {
                    args: spread.into_boxed_slice(),
                    ..c.arguments.clone()
                },
                ..c.clone()
            };
            return self.call(&expanded, ty, span);
        }
        let args = &c.arguments.args;
        let keywords = &c.arguments.keywords;
        if let py::Expr::Attribute(a) = &*c.func
            && types::is_name(&a.value, "frozenset")
            && a.attr.as_str() == "union"
            && !self.is_variable("frozenset")
            && keywords.is_empty()
            && !args.is_empty()
        {
            // One copy of the first set, the rest added into it.
            let first = self.expr_as(&args[0], Ty::Set)?;
            let result = call("zb_set_copy", vec![first], Ty::Set, span);
            let mut pre = Vec::new();
            let held = self.hold(
                Val {
                    node: result,
                    ty: Ty::Set,
                },
                &mut pre,
                span,
            );
            for arg in &args[1..] {
                let other = self.expr_as(arg, Ty::Set)?;
                pre.push(TypedNode::new(
                    TypedStatement::Expression(Box::new(call(
                        "zb_set_update",
                        vec![held.node.clone(), other],
                        Ty::None,
                        span,
                    ))),
                    Type::Unknown,
                    span,
                ));
            }
            let result = Self::block_value(pre, held.node, Ty::Set, span);
            return Ok(Val {
                node: result,
                ty: Ty::Set,
            });
        }
        // `"...{}".format(args)` with a literal template is built here.
        if let py::Expr::Attribute(a) = &*c.func
            && a.attr.as_str() == "format"
            && let py::Expr::StringLiteral(template) = &*a.value
        {
            let template = template.value.to_str().to_string();
            return self.str_format(&template, args, keywords, span);
        }
        // `sys.stdout.write(s)` and `.flush()`: print's own path, which
        // follows a redirection; flushing is the host's business.
        if let py::Expr::Attribute(a) = &*c.func
            && let py::Expr::Attribute(inner) = &*a.value
            && matches!(&*inner.value, py::Expr::Name(m) if m.id.as_str() == "sys" && !self.is_variable("sys"))
            && inner.attr.as_str() == "stdout"
            && keywords.is_empty()
        {
            let node = match (a.attr.as_str(), args.len()) {
                ("write", 1) => {
                    let text = self.expr_as(&args[0], Ty::Str)?;
                    call("zb_print_text", vec![text], Ty::None, span)
                }
                ("flush", 0) => node(TypedExpression::Literal(TypedLiteral::Null), Ty::None, span),
                _ => return unsupported("this method of sys.stdout", c),
            };
            return Ok(Val { node, ty: Ty::None });
        }
        // A function of an imported module, named through the module or
        // brought in by name.
        if let py::Expr::Attribute(a) = &*c.func
            && let Some(member) = self.module_member_of(&a.value, a.attr.as_str())
        {
            return self.stdlib_call(member, a.attr.as_str(), args, keywords, c, span);
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
                    // An instance is called through its class's `__call__`.
                    Ty::Class(k) => {
                        let receiver = self.expr(&c.func)?;
                        return self
                            .method_on(k as usize, receiver, "__call__", args, keywords, c, span);
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
            // `range` given arguments it does not take is called through
            // its value, whose call raises the TypeError when it runs.
            if name == "range"
                && !self.module.funcs.contains_key(name)
                && !self.module.class_index.contains_key(name)
                && !self.builtin_takes(name, c)
            {
                let callee = self.expr(&c.func)?;
                return self.call_value(callee, args, keywords, c, span);
            }
            if !self.module.funcs.contains_key(name)
                && !self.module.class_index.contains_key(name)
                && let Some(member) = self.module.imported_name(name)
            {
                return self.stdlib_call(member, name, args, keywords, c, span);
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
        if let py::Expr::Attribute(a) = &*c.func
            && a.attr.as_str() == "sort"
            && !keywords.is_empty()
        {
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
        if let py::Expr::Name(n) = &*c.func
            && let Some(v) = self.iteration_builtin(n.id.as_str(), args, keywords, ty, c, span)?
        {
            return Ok(v);
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
            if let Ty::Class(k) = callee.ty {
                return self.method_on(k as usize, callee, "__call__", args, keywords, c, span);
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
                // `hex`, `oct`, `bin`: the alternate form of the base.
                "hex" | "oct" | "bin" if args.len() == 1 => {
                    let v = self.expr(&args[0])?;
                    let v = Val {
                        node: self.coerce(v, Ty::Int),
                        ty: Ty::Int,
                    };
                    let spec = crate::format::parse_spec(match name {
                        "hex" => "#x",
                        "oct" => "#o",
                        _ => "#b",
                    })
                    .expect("a fixed spec parses");
                    return Ok(Val {
                        node: self.format(v, &spec, span),
                        ty: Ty::Str,
                    });
                }
                "bytes" => return self.bytes_call(args, c, span),
                "open" => return self.open_call(c, span),
                "eval" => return self.eval_call(args, c, span),
                "int" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Int => v.node,
                        Ty::Bool | Ty::Float => cast(v.node, Ty::Int, span),
                        Ty::Str => call("zb_int_of_str", vec![v.node], Ty::Int, span),
                        // An instance answers through `__int__`.
                        Ty::Class(k) => {
                            match self.dunder(k as usize, "__int__", v.node, vec![], span) {
                                Some(r) => self.coerce(r, Ty::Int),
                                None => {
                                    return Err(Error::unsupported_span(
                                        format!(
                                            "int() of {}, which defines no __int__",
                                            self.module.classes[k as usize].name
                                        ),
                                        span,
                                    ));
                                }
                            }
                        }
                        _ => call("zb_any_int", vec![v.node], Ty::Int, span),
                    };
                    return Ok(Val { node, ty: Ty::Int });
                }
                "float" => {
                    let v = self.expr(&args[0])?;
                    let node = match v.ty {
                        Ty::Float => v.node,
                        Ty::Bool | Ty::Int => cast(v.node, Ty::Float, span),
                        Ty::Str => call("zb_float_of_str", vec![v.node], Ty::Float, span),
                        Ty::Class(k) => {
                            match self.dunder(k as usize, "__float__", v.node, vec![], span) {
                                Some(r) => self.coerce(r, Ty::Float),
                                None => {
                                    return Err(Error::unsupported_span(
                                        format!(
                                            "float() of {}, which defines no __float__",
                                            self.module.classes[k as usize].name
                                        ),
                                        span,
                                    ));
                                }
                            }
                        }
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
                        Ty::Bytes => call("zb_str_len", vec![v.node], Ty::Int, span),
                        Ty::List(_) => method_call(v.node, "len", vec![], Ty::Int, span),
                        // A shape's length is its arity.
                        Ty::Tuple(k) => Self::after_none(
                            v.node,
                            int_lit(types::tuple_shape(k).len() as i64, span),
                            Ty::Int,
                        ),
                        Ty::Set => call("zb_set_len", vec![v.node], Ty::Int, span),
                        Ty::Dict(_) => call("zb_dict_len", vec![v.node], Ty::Int, span),
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
                                    ));
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
                    let ok = match v.ty {
                        // One byte's value.
                        Ty::Bytes => Val {
                            node: call("zb_bytes_ord", vec![v.node], Ty::Int, span),
                            ty: Ty::Int,
                        },
                        Ty::Object => Val {
                            node: call("zb_any_ord", vec![v.node], Ty::Int, span),
                            ty: Ty::Int,
                        },
                        _ => {
                            let s = self.coerce(v, Ty::Str);
                            Val {
                                node: call("zb_str_ord", vec![s], Ty::Int, span),
                                ty: Ty::Int,
                            }
                        }
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
                    let v = self.tuple_as_list(v);
                    // Bytes sum as their values.
                    let v = if v.ty == Ty::Bytes {
                        Val {
                            node: self.coerce(v, Ty::List(Elem::Int)),
                            ty: Ty::List(Elem::Int),
                        }
                    } else {
                        v
                    };
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
                        let v = self.consumed(&args[0], span)?;
                        self.tuple_as_list(v)
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
                                node: self.iterable(list, span),
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
                    // The shape is what inference gave the call.
                    let dict_ty = match ty {
                        Ty::Dict(_) => ty,
                        _ => types::dynamic_dict(),
                    };
                    let node = match args.first() {
                        None => call(
                            "zb_dict_from_pairs",
                            vec![self.list_of(Vec::new(), Elem::Object, span)],
                            dict_ty,
                            span,
                        ),
                        Some(a) => {
                            let v = self.consumed(a, span)?;
                            match v.ty {
                                Ty::Dict(_) => call("zb_dict_copy", vec![v.node], dict_ty, span),
                                // Anything else is a sequence of pairs.
                                _ => {
                                    let items = self.iterable(v, span);
                                    call("zb_dict_from_tuples", vec![items], dict_ty, span)
                                }
                            }
                        }
                    };
                    return Ok(Val { node, ty: dict_ty });
                }
                "set" | "frozenset" => {
                    let items = match args.first() {
                        None => self.list_of(Vec::new(), Elem::Object, span),
                        Some(a) => {
                            let v = self.consumed(a, span)?;
                            match v.ty {
                                Ty::List(_) | Ty::Tuple(_) | Ty::Set | Ty::Dict(_) => {
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
                "tuple" if matches!(ty, Ty::Tuple(_)) => {
                    let v = self.expr(&args[0])?;
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
                                    );
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
                                // An array's elements as the numbers they
                                // read as; the copy below is then a list.
                                Ty::List(Elem::Array(c)) => {
                                    let wide = Ty::List(Elem::of(c.item()));
                                    let node = self.coerce(v, wide);
                                    Val { node, ty: wide }
                                }
                                Ty::List(_) => v,
                                // A tuple's elements as the list the
                                // result is typed as.
                                Ty::Tuple(_) => {
                                    let node = self.coerce(v, Ty::List(target_elem));
                                    Val {
                                        node,
                                        ty: Ty::List(target_elem),
                                    }
                                }
                                Ty::Set | Ty::Dict(_) | Ty::Gen => {
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
                                Ty::Bytes => Val {
                                    node: call(
                                        "zb_bytes_to_list",
                                        vec![v.node],
                                        Ty::List(Elem::Int),
                                        span,
                                    ),
                                    ty: Ty::List(Elem::Int),
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
                    // A tuple of a length only the runtime knows is a
                    // boxed tuple.
                    let (value, result_ty) = if name == "tuple" {
                        (
                            call("zb_box_tuple", vec![held.node], Ty::Object, span),
                            Ty::Object,
                        )
                    } else {
                        (held.node, Ty::List(e))
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
                    // The pair, of the shape inference gave it.
                    let pair = match ty.tuple_elems() {
                        Some(shape) if shape.len() == 2 => Val {
                            node: self.tuple_of_items(vec![q, r], ty, span),
                            ty,
                        },
                        _ => {
                            let list = self.list_of(vec![q, r], Elem::Object, span);
                            Val {
                                node: call("zb_box_tuple", vec![list], Ty::Object, span),
                                ty: Ty::Object,
                            }
                        }
                    };
                    return Ok(pair);
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
                        Ty::Bytes => str_lit("bytes", span),
                        Ty::None => str_lit("NoneType", span),
                        Ty::List(_) => str_lit("list", span),
                        Ty::Tuple(_) => str_lit("tuple", span),
                        Ty::Dict(_) => str_lit("dict", span),
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

    /// `match r#gen.next() { Some(x) => { ... }, _ => { ... } }` as a
    /// statement, with `x` bound as `item` in the first arm.
    fn next_match(
        &mut self,
        r#gen: Node,
        item: InternedString,
        some: Vec<Stmt>,
        none: Vec<Stmt>,
        span: Span,
    ) -> Stmt {
        let next = method_call(r#gen, "next", vec![], Ty::Object, span);
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
        r#gen: Val,
        extra: Vec<Stmt>,
        span: Span,
    ) -> Result<TypedStatement> {
        let mut prologue = std::mem::take(&mut self.hoisted);
        let held_from = r#gen.node.clone();
        let held = self.hold(r#gen, &mut prologue, span);
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
    fn generator_to_list(&mut self, r#gen: Node, span: Span) -> Node {
        let mut pre = Vec::new();
        let fresh = is_generator_start(&r#gen);
        let held = self.hold(
            Val {
                node: r#gen,
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

    /// `next(r#gen)`: the next value, or the default, or StopIteration.
    fn next_of(&mut self, r#gen: Node, default: Option<Node>, span: Span) -> Val {
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
        let pull = self.next_match(r#gen, item, take, exhausted, span);
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
                ("bytes", Ty::Bytes) => true,
                ("list", Ty::List(_)) => true,
                ("tuple", Ty::Tuple(_)) => true,
                ("dict", Ty::Dict(_)) => true,
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
                let as_passed = !self.reassigned_params.contains(name);
                let is_self = as_passed
                    && self.class.is_some()
                    && self
                        .sig
                        .params
                        .first()
                        .is_some_and(|(p, _)| intern(p) == *name);
                // The trusted variant takes every instance-typed parameter
                // to be one.
                let trusted_param = as_passed
                    && self.trusted
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
    fn always_instances(&self, body: &[py::Stmt]) -> HashSet<InternedString> {
        use ruff_python_ast::visitor::{Visitor, walk_expr, walk_stmt};
        struct Names {
            mentioned: HashSet<String>,
            stored: HashSet<String>,
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
        let mut seen: HashSet<String> = HashSet::default();
        let mut candidates: HashSet<String> = HashSet::default();
        let mut disqualified: HashSet<String> = HashSet::default();
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
                let Some((_, ty)) = self.module.field(k as usize, attr) else {
                    // The class's attribute, once the instance is known
                    // not to be None.
                    let class_attr = self.class_attr_of(k as usize, attr, span)?;
                    let object = self.checked_instance(object, attr, span);
                    let value = self.class_attr_read(&class_attr, span);
                    let evaluated = TypedNode::new(
                        TypedStatement::Expression(Box::new(object.node)),
                        Type::Unknown,
                        span,
                    );
                    return Ok(Val {
                        node: Self::block_value(vec![evaluated], value.node, value.ty, span),
                        ty: value.ty,
                    });
                };
                let object = self.checked_instance(object, attr, span);
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
            // An array's typecode and element size are its type's.
            Ty::List(Elem::Array(c)) if matches!(attr, "typecode" | "itemsize") => {
                let (value, ty) = if attr == "typecode" {
                    (str_lit(c.letter(), span), Ty::Str)
                } else {
                    (int_lit(c.itemsize(), span), Ty::Int)
                };
                let evaluated = TypedNode::new(
                    TypedStatement::Expression(Box::new(object.node)),
                    Type::Unknown,
                    span,
                );
                Ok(Val {
                    node: Self::block_value(vec![evaluated], value, ty, span),
                    ty,
                })
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
            HashMap::default(),
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
                        "copy" | "tolist" => Ty::List(e),
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
        // A generator method starts its fiber, `self` in the environment
        // with the arguments; a dispatcher over overrides does that for
        // whichever it reaches.
        if sig.ret == Ty::Gen && target == fn_name {
            let sig = sig.clone();
            let started = self.start_generator(&fn_name, &sig, all, Vec::new(), span);
            return Some((started, fn_name));
        }
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
            // A method only subclasses define goes to whichever the
            // instance's class holds.
            if let Some(sig) = self.module.abstract_sig(k, method) {
                let sig = without_self(&sig);
                let receiver = self.checked_instance(receiver, method, span);
                let lowered = self.arguments(method, &sig, args, keywords, c)?;
                self.module
                    .abstract_calls
                    .borrow_mut()
                    .insert((k, method.to_string()));
                let target = abstract_name(&self.module.classes[k].name, method);
                let mut all = vec![receiver.node];
                all.extend(lowered);
                let v = Val {
                    node: call(&target, all, sig.ret, span),
                    ty: sig.ret,
                };
                return Ok(self.guard_named(v, &target, span));
            }
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
        // Where a class defines `__call__` at this arity, an instance
        // goes to it and anything else to the function record.
        let callable_class = (0..self.module.classes.len()).any(|c| {
            self.module
                .method_sig(c, "__call__")
                .is_some_and(|(sig, _)| sig.params.len() == args.len() + 1)
        });
        let node = if callable_class {
            let mut pre = Vec::new();
            let held: Vec<Node> = lowered
                .into_iter()
                .map(|n| {
                    self.hold(
                        Val {
                            node: n,
                            ty: Ty::Object,
                        },
                        &mut pre,
                        span,
                    )
                    .node
                })
                .collect();
            self.module
                .dyn_methods
                .borrow_mut()
                .insert(("__call__".to_string(), args.len()));
            let is_instance = call("zb_any_is_instance", vec![held[0].clone()], Ty::Bool, span);
            let picked = node(
                TypedExpression::If(TypedIfExpr {
                    condition: Box::new(is_instance),
                    then_branch: Box::new(call(
                        &callm_name("__call__", args.len()),
                        held.clone(),
                        Ty::Object,
                        span,
                    )),
                    else_branch: Box::new(call(
                        &format!("zb_call_{}", args.len()),
                        held,
                        Ty::Object,
                        span,
                    )),
                }),
                Ty::Object,
                span,
            );
            Self::block_value(pre, picked, Ty::Object, span)
        } else {
            call(
                &format!("zb_call_{}", args.len()),
                lowered,
                Ty::Object,
                span,
            )
        };
        let v = Val {
            node,
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

    /// A module class as a value: a record whose call constructs.
    fn class_value(&mut self, k: usize, span: Span) -> Result<Val> {
        let sig = match self.module.method_sig(k, "__init__") {
            Some((sig, _)) => without_self(sig),
            None => Sig {
                params: Vec::new(),
                ret: Ty::None,
                defaults: Vec::new(),
            },
        };
        self.module.class_adapters.borrow_mut().insert(k);
        let defaults = self.default_values(&sig)?;
        let name = class_adapter_name(&self.module.classes[k].name);
        Ok(self.record(&name, sig.params.len(), Vec::new(), defaults, span))
    }

    /// `struct.unpack(fmt, data)`: the buffer checked for the format's
    /// size, then each field read at its offset into the tuple.
    fn struct_unpack(
        &mut self,
        data: &py::Expr,
        big: bool,
        fields: &[(i64, stdlib::Field)],
        size: i64,
        span: Span,
    ) -> Result<Val> {
        let data = self.expr_as(data, Ty::Bytes)?;
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: data,
                ty: Ty::Bytes,
            },
            &mut pre,
            span,
        );
        pre.push(TypedNode::new(
            TypedStatement::Expression(Box::new(call(
                "zb_struct_expect",
                vec![held.node.clone(), int_lit(size, span)],
                Ty::None,
                span,
            ))),
            Type::Unknown,
            span,
        ));
        pre.push(self.pending_check(span));
        let big = int_lit(big as i64, span);
        let items: Vec<Val> = fields
            .iter()
            .map(|(offset, field)| {
                let node = match *field {
                    stdlib::Field::Int { size, signed } => call(
                        "zb_struct_int",
                        vec![
                            held.node.clone(),
                            int_lit(*offset, span),
                            int_lit(size, span),
                            int_lit(signed as i64, span),
                            big.clone(),
                        ],
                        Ty::Int,
                        span,
                    ),
                    stdlib::Field::Bool => binary(
                        BinaryOp::Ne,
                        call(
                            "zb_struct_int",
                            vec![
                                held.node.clone(),
                                int_lit(*offset, span),
                                int_lit(1, span),
                                int_lit(0, span),
                                big.clone(),
                            ],
                            Ty::Int,
                            span,
                        ),
                        int_lit(0, span),
                        Ty::Bool,
                        span,
                    ),
                    stdlib::Field::Float { size } => call(
                        "zb_struct_float",
                        vec![
                            held.node.clone(),
                            int_lit(*offset, span),
                            int_lit(size, span),
                            big.clone(),
                        ],
                        Ty::Float,
                        span,
                    ),
                    stdlib::Field::Bytes { size } => call(
                        "zb_bytes_slice_raw",
                        vec![
                            held.node.clone(),
                            int_lit(*offset, span),
                            int_lit(*offset + size, span),
                        ],
                        Ty::Bytes,
                        span,
                    ),
                };
                Val {
                    node,
                    ty: field.ty(),
                }
            })
            .collect();
        let ty = types::tuple_of(items.iter().map(|v| v.ty).collect());
        let tuple = self.tuple_of_items(items, ty, span);
        // The buffer's hold and check run ahead of the statement.
        self.hoisted.extend(pre);
        Ok(Val { node: tuple, ty })
    }

    /// `globals()[s]`: the module variable of this file the string
    /// names, boxed; a KeyError for any other string.
    fn globals_lookup(&mut self, key: &py::Expr, span: Span) -> Result<Val> {
        let text = self.expr_as(key, Ty::Str)?;
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: text,
                ty: Ty::Str,
            },
            &mut pre,
            span,
        );
        let file = current_file();
        let prefix = self
            .module
            .files
            .iter()
            .find(|(_, index)| **index == file)
            .map(|(m, _)| format!("{m}$"))
            .unwrap_or_default();
        let mut names: Vec<(String, String)> = self
            .module
            .globals
            .keys()
            .filter_map(|full| {
                full.strip_prefix(prefix.as_str())
                    .filter(|rest| !rest.contains('$'))
                    .map(|short| (short.to_string(), full.clone()))
            })
            .collect();
        names.sort();
        let mut raise = Vec::new();
        self.raise_named("KeyError", held.node.clone(), span, &mut raise);
        raise.push(TypedNode::new(
            TypedStatement::Expression(Box::new(node(
                TypedExpression::Literal(TypedLiteral::Null),
                Ty::Object,
                span,
            ))),
            Type::Unknown,
            span,
        ));
        let mut chosen = Self::block_value(
            raise,
            node(
                TypedExpression::Literal(TypedLiteral::Null),
                Ty::Object,
                span,
            ),
            Ty::Object,
            span,
        );
        for (short, full) in names.into_iter().rev() {
            let value = self.global_read(&full, span);
            let boxed = self.coerce(value, Ty::Object);
            let test = call(
                "zb_str_eq",
                vec![held.node.clone(), str_lit(&short, span)],
                Ty::Bool,
                span,
            );
            chosen = node(
                TypedExpression::If(TypedIfExpr {
                    condition: Box::new(test),
                    then_branch: Box::new(boxed),
                    else_branch: Box::new(chosen),
                }),
                Ty::Object,
                span,
            );
        }
        Ok(Val {
            node: Self::block_value(pre, chosen, Ty::Object, span),
            ty: Ty::Object,
        })
    }

    /// `eval(s)`: the module's function or class the string names, as
    /// a value, or the number it spells. Nothing else is evaluated.
    fn eval_call(&mut self, args: &[py::Expr], c: &py::ExprCall, span: Span) -> Result<Val> {
        let [arg] = args else {
            return unsupported("eval() with these arguments", c);
        };
        let text = self.expr_as(arg, Ty::Str)?;
        let mut pre = Vec::new();
        let held = self.hold(
            Val {
                node: text,
                ty: Ty::Str,
            },
            &mut pre,
            span,
        );
        // The names of the module this function is in.
        let file = current_file();
        let prefix = self
            .module
            .files
            .iter()
            .find(|(_, index)| **index == file)
            .map(|(m, _)| format!("{m}$"))
            .unwrap_or_default();
        let module_level = |name: &str| {
            name.strip_prefix(prefix.as_str())
                .filter(|rest| !rest.contains('$'))
                .map(str::to_string)
        };
        let mut functions: Vec<(String, String)> = self
            .module
            .funcs
            .keys()
            .filter_map(|full| module_level(full).map(|short| (short, full.clone())))
            .collect();
        functions.sort();
        let mut classes: Vec<(String, usize)> = self
            .module
            .class_index
            .iter()
            .filter_map(|(full, &k)| module_level(full).map(|short| (short, k)))
            .collect();
        classes.sort();
        let mut chosen = call("zb_eval_literal", vec![held.node.clone()], Ty::Object, span);
        let mut arms: Vec<(String, Node)> = Vec::new();
        for (short, full) in functions {
            let value = self.function_value(&full, span)?.node;
            arms.push((short, value));
        }
        for (short, k) in classes {
            let value = self.class_value(k, span)?.node;
            arms.push((short, value));
        }
        for (short, value) in arms.into_iter().rev() {
            let test = call(
                "zb_str_eq",
                vec![held.node.clone(), str_lit(&short, span)],
                Ty::Bool,
                span,
            );
            chosen = node(
                TypedExpression::If(TypedIfExpr {
                    condition: Box::new(test),
                    then_branch: Box::new(value),
                    else_branch: Box::new(chosen),
                }),
                Ty::Object,
                span,
            );
        }
        Ok(Val {
            node: Self::block_value(pre, chosen, Ty::Object, span),
            ty: Ty::Object,
        })
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
                    );
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
        let mut file: Option<Node> = None;
        for k in keywords {
            match k.arg.as_ref().map(|a| a.as_str()) {
                Some("sep") => sep = self.expr_as(&k.value, Ty::Str)?,
                Some("end") => end = Some(self.expr_as(&k.value, Ty::Str)?),
                // `file=f` writes the line to an open text file.
                Some("file") => {
                    let f = self.expr(&k.value)?;
                    match f.ty {
                        Ty::File(_) => file = Some(f.node),
                        _ => {
                            return unsupported(
                                "print(..., file=) to something other than a file",
                                k,
                            );
                        }
                    }
                }
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
        if let Some(file) = file {
            let end = end.unwrap_or_else(|| str_lit("\n", span));
            let text = binary(BinaryOp::Add, line, end, Ty::Str, span);
            return Ok(Val {
                node: call("zb_file_write", vec![file, text], Ty::None, span),
                ty: Ty::None,
            });
        }
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
