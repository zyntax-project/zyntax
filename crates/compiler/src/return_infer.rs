//! Return types for declarations that don't state one.
//!
//! `def f(x) { ... }` has no return annotation, and the parser records
//! that absence as `Type::Unknown`. Two neighbouring types mean
//! something narrower: `Unit` says the function returns no value, and
//! `Any` says it returns a dynamically typed one. Keeping all three
//! apart is what lets the return type come from the body instead of
//! from a guess, and what lets `def f(): Any` declare a box.
//!
//! Both the signature the function is compiled with and the entry the
//! call-site table records come from [`effective_return_type`], so a
//! caller and its callee cannot disagree about what comes back.

use zyntax_typed_ast::typed_ast::{TypedBlock, TypedExpression, TypedLoop, TypedStatement};
use zyntax_typed_ast::{PrimitiveType, Type, TypedFunction, TypedLiteral, TypedNode};

fn unit() -> Type {
    Type::Primitive(PrimitiveType::Unit)
}

/// The return type a function actually has.
///
/// A declaration that states its return type keeps it verbatim. One that
/// doesn't gets its type from the body:
///
/// * no path returns a value → `Unit`
/// * every returned value agrees on a type → that type
/// * otherwise → `Dynamic`, resolved at runtime
pub(crate) fn effective_return_type(func: &TypedFunction) -> Type {
    if !is_unstated(&func.return_type) {
        return func.return_type.clone();
    }

    let Some(body) = func.body.as_ref() else {
        // A declaration with no body (extern, trait method) has nothing
        // to infer from.
        return unit();
    };

    let func_name = func.name.resolve_global().unwrap_or_default();
    infer_from_body(&func_name, body)
}

/// The type a body returns, for a context that has no annotation to fall
/// back on. Same rules as [`effective_return_type`].
pub(crate) fn infer_from_body(func_name: &str, body: &TypedBlock) -> Type {
    let mut returned = Vec::new();
    collect_returned_types(body, &mut returned);
    if let Some(tail) = implicit_tail_return(func_name, body) {
        returned.push(tail);
    }
    join_returned_types(returned)
}

/// Whether a declaration left its return type unstated.
pub(crate) fn is_unstated(ty: &Type) -> bool {
    // `Any` is a stated type — it declares a dynamically typed value —
    // so only the absence of an annotation counts as unstated.
    matches!(ty, Type::Unknown)
}

/// The type of a body's implicit trailing-expression return, if it has
/// one. Mirrors the CFG builder, which treats a function body of exactly
/// one bare expression as an implicit return. `main` is always void, so it
/// never has one.
fn implicit_tail_return(func_name: &str, body: &TypedBlock) -> Option<Type> {
    if func_name == "main" || body.statements.len() != 1 {
        return None;
    }
    match &body.statements[0].node {
        TypedStatement::Expression(expr) => Some(returned_type_of(expr)),
        _ => None,
    }
}

/// The type a returned expression contributes.
///
/// This is `expr.ty` except for integer literals: the parser types those
/// as `i32` provisionally, expecting a later context — an annotation, an
/// assignment — to settle the width. An unstated return type is no such
/// context, so a literal takes the default width instead of the
/// provisional one. Unary and binary nodes carry their operand's type by
/// the same convention, so the walk follows them.
pub(crate) fn returned_type_of(expr: &TypedNode<TypedExpression>) -> Type {
    match &expr.node {
        TypedExpression::Literal(TypedLiteral::Integer(_)) => Type::Primitive(PrimitiveType::I64),
        TypedExpression::Unary(unary) => returned_type_of(&unary.operand),
        TypedExpression::Binary(binary) => returned_type_of(&binary.left),
        _ => expr.ty.clone(),
    }
}

/// Collect the type of every `return <expr>` reachable in `body`,
/// including those nested inside control flow.
///
/// Only the enclosing function's returns count: a `return` inside a nested
/// lambda or coroutine body belongs to that body, not this one, and those
/// live in expression position so this statement walk does not reach them.
fn collect_returned_types(body: &TypedBlock, out: &mut Vec<Type>) {
    for stmt in &body.statements {
        match &stmt.node {
            TypedStatement::Return(Some(expr)) => out.push(returned_type_of(expr)),
            TypedStatement::Block(inner) => collect_returned_types(inner, out),
            TypedStatement::If(if_stmt) => {
                collect_returned_types(&if_stmt.then_block, out);
                if let Some(else_block) = &if_stmt.else_block {
                    collect_returned_types(else_block, out);
                }
            }
            TypedStatement::While(while_stmt) => collect_returned_types(&while_stmt.body, out),
            TypedStatement::For(for_stmt) => collect_returned_types(&for_stmt.body, out),
            TypedStatement::ForCStyle(for_stmt) => collect_returned_types(&for_stmt.body, out),
            TypedStatement::Loop(loop_stmt) => match loop_stmt {
                TypedLoop::ForEach { body, .. }
                | TypedLoop::ForCStyle { body, .. }
                | TypedLoop::While { body, .. }
                | TypedLoop::DoWhile { body, .. }
                | TypedLoop::Infinite { body } => collect_returned_types(body, out),
            },
            TypedStatement::Match(match_stmt) => {
                // Arm bodies are expressions; a `return` inside one is an
                // expression-position return this walk can't see, so only
                // block-bodied arms contribute.
                for arm in &match_stmt.arms {
                    if let TypedExpression::Block(block) = &arm.body.node {
                        collect_returned_types(block, out);
                    }
                }
            }
            TypedStatement::Try(try_stmt) => {
                collect_returned_types(&try_stmt.body, out);
                for catch in &try_stmt.catch_clauses {
                    collect_returned_types(&catch.body, out);
                }
                if let Some(finally) = &try_stmt.finally_block {
                    collect_returned_types(finally, out);
                }
            }
            TypedStatement::With(with_stmt) => collect_returned_types(&with_stmt.body, out),
            TypedStatement::Select(select) => {
                for arm in &select.arms {
                    collect_returned_types(&arm.body, out);
                }
                if let Some(default) = &select.default {
                    collect_returned_types(default, out);
                }
            }
            _ => {}
        }
    }
}

/// Reduce the types a body returns to the one the function has.
///
/// Any type the parser left unsettled, or any disagreement between paths,
/// falls back to `Dynamic` — which lowers to a machine word and is
/// resolved at runtime, exactly as before this inferred anything.
fn join_returned_types(returned: Vec<Type>) -> Type {
    if returned.is_empty() {
        return unit();
    }

    let mut settled: Option<Type> = None;
    for ty in returned {
        // `Never` is a diverging path (`return panic()`), which
        // constrains nothing.
        if matches!(ty, Type::Never) {
            continue;
        }
        if !is_statically_settled(&ty) {
            return Type::Dynamic;
        }
        match &settled {
            None => settled = Some(ty),
            Some(prev) if *prev == ty => {}
            Some(_) => return Type::Dynamic,
        }
    }

    settled.unwrap_or_else(unit)
}

/// Whether a type is concrete enough to be a return type.
pub(crate) fn is_statically_settled(ty: &Type) -> bool {
    !matches!(
        ty,
        Type::Any
            | Type::Unknown
            | Type::Dynamic
            | Type::Error
            | Type::Primitive(PrimitiveType::Unit)
    )
}
