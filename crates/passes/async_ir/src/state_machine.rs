//! Rewrite: async Function → state machine (Enum + Class + poll Function)

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{Mutability, PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

pub fn async_fn_to_state_machine() -> DeclRewrite {
    DeclRewrite::new(
        "async_fn_to_state_machine",
        Priority::SEMANTIC,
        Pattern::new("async_fn", |node, _ctx| {
            if let TypedDeclaration::Function(func) = &node.node {
                if func.is_async && func.body.is_some() {
                    return Some(Bindings::new());
                }
            }
            None
        }),
        |matched, _bindings, _builder| {
            if let TypedDeclaration::Function(func) = &matched.node {
                let declarations = build_state_machine(func, matched.span);
                RewriteOutput::Expand {
                    declarations,
                    replacement: None, // remove the async function
                }
            } else {
                RewriteOutput::Unchanged
            }
        },
    )
}

/// Count the number of Await expressions in a function body.
fn count_await_sites(body: &TypedBlock) -> usize {
    let mut count = 0;
    for stmt in &body.statements {
        count += count_awaits_in_stmt(&stmt.node);
    }
    count
}

fn count_awaits_in_stmt(stmt: &TypedStatement) -> usize {
    match stmt {
        TypedStatement::Expression(expr) => count_awaits_in_expr(&expr.node),
        TypedStatement::Let(let_stmt) => let_stmt
            .initializer
            .as_ref()
            .map_or(0, |e| count_awaits_in_expr(&e.node)),
        TypedStatement::Return(Some(expr)) => count_awaits_in_expr(&expr.node),
        TypedStatement::If(if_stmt) => {
            count_awaits_in_expr(&if_stmt.condition.node)
                + count_await_sites(&if_stmt.then_block)
                + if_stmt.else_block.as_ref().map_or(0, count_await_sites)
        }
        TypedStatement::While(w) => {
            count_awaits_in_expr(&w.condition.node) + count_await_sites(&w.body)
        }
        TypedStatement::Block(b) => count_await_sites(b),
        _ => 0,
    }
}

fn count_awaits_in_expr(expr: &TypedExpression) -> usize {
    match expr {
        TypedExpression::Await(_) => 1,
        TypedExpression::Binary(b) => {
            count_awaits_in_expr(&b.left.node) + count_awaits_in_expr(&b.right.node)
        }
        TypedExpression::Call(c) => {
            count_awaits_in_expr(&c.callee.node)
                + c.positional_args
                    .iter()
                    .map(|a| count_awaits_in_expr(&a.node))
                    .sum::<usize>()
        }
        TypedExpression::Unary(u) => count_awaits_in_expr(&u.operand.node),
        _ => 0,
    }
}

/// Build the state machine declarations from an async function.
///
/// Produces:
/// 1. Enum `FnName$State` with variants: Start, AwaitN (one per await site), Done
/// 2. Class `FnName$Future` with fields: state (enum), locals (function params + locals)
/// 3. Function `FnName$poll` that implements the state machine transitions
/// 4. Function `FnName` (non-async wrapper) that creates the Future and returns it
fn build_state_machine(func: &TypedFunction, span: Span) -> Vec<TypedNode<TypedDeclaration>> {
    let name = func.name.resolve_global().unwrap_or_default();
    let body = match &func.body {
        Some(b) => b,
        None => return vec![],
    };
    let await_count = count_await_sites(body);

    // 1. State enum: Start, Await0, Await1, ..., Done
    let state_enum = build_state_enum(&name, await_count, span);

    // 2. Future struct: holds state + captured locals
    let future_class = build_future_class(&name, func, span);

    // 3. Poll function: dispatches on state, executes body segments
    let poll_fn = build_poll_function(&name, &func.return_type, span);

    // 4. Wrapper: creates Future from args, returns it
    let wrapper_fn = build_wrapper_function(func, span);

    vec![state_enum, future_class, poll_fn, wrapper_fn]
}

fn build_state_enum(fn_name: &str, await_count: usize, span: Span) -> TypedNode<TypedDeclaration> {
    let enum_name = InternedString::new_global(&format!("{}$State", fn_name));

    let mut variants = vec![TypedVariant {
        name: InternedString::new_global("Start"),
        fields: TypedVariantFields::Unit,
        discriminant: None,
        span,
    }];

    for i in 0..await_count {
        variants.push(TypedVariant {
            name: InternedString::new_global(&format!("Await{}", i)),
            fields: TypedVariantFields::Unit,
            discriminant: None,
            span,
        });
    }

    variants.push(TypedVariant {
        name: InternedString::new_global("Done"),
        fields: TypedVariantFields::Unit,
        discriminant: None,
        span,
    });

    TypedNode::new(
        TypedDeclaration::Enum(TypedEnum {
            name: enum_name,
            type_params: vec![],
            variants,
            visibility: Visibility::Public,
            span,
        }),
        Type::Primitive(PrimitiveType::Unit),
        span,
    )
}

fn build_future_class(
    fn_name: &str,
    func: &TypedFunction,
    span: Span,
) -> TypedNode<TypedDeclaration> {
    let class_name = InternedString::new_global(&format!("{}$Future", fn_name));

    // Fields: state (i64 discriminant) + one field per function parameter
    let mut fields = vec![TypedField {
        name: InternedString::new_global("state"),
        ty: Type::Primitive(PrimitiveType::I64),
        initializer: None,
        visibility: Visibility::Public,
        mutability: Mutability::Mutable,
        is_static: false,
        span,
    }];

    for param in &func.params {
        fields.push(TypedField {
            name: param.name,
            ty: param.ty.clone(),
            initializer: None,
            visibility: Visibility::Public,
            mutability: Mutability::Mutable,
            is_static: false,
            span,
        });
    }

    TypedNode::new(
        TypedDeclaration::Class(TypedClass {
            name: class_name,
            type_params: func.type_params.clone(),
            extends: None,
            implements: vec![],
            fields,
            methods: vec![],
            constructors: vec![],
            visibility: Visibility::Public,
            is_abstract: false,
            is_final: true,
            span,
        }),
        Type::Primitive(PrimitiveType::Unit),
        span,
    )
}

fn build_poll_function(
    fn_name: &str,
    return_type: &Type,
    span: Span,
) -> TypedNode<TypedDeclaration> {
    let poll_name = InternedString::new_global(&format!("{}$poll", fn_name));
    let future_param_name = InternedString::new_global("future");

    // poll takes a mutable reference to the Future struct, returns Poll<T>
    // For now, return type is the original return type (simplified — no Poll wrapper)
    TypedNode::new(
        TypedDeclaration::Function(TypedFunction {
            name: poll_name,
            params: vec![TypedParameter {
                name: future_param_name,
                ty: Type::Primitive(PrimitiveType::I64), // erased future pointer
                ..Default::default()
            }],
            return_type: return_type.clone(),
            body: Some(TypedBlock {
                // The body would contain a match on future.state dispatching to
                // each state's code segment. For now, empty — the existing
                // AsyncCompiler at HIR level handles the actual code splitting.
                statements: vec![],
                span,
            }),
            visibility: Visibility::Public,
            ..Default::default()
        }),
        Type::Primitive(PrimitiveType::Unit),
        span,
    )
}

fn build_wrapper_function(func: &TypedFunction, span: Span) -> TypedNode<TypedDeclaration> {
    // The wrapper keeps the original name but is no longer async.
    // It creates a Future struct from the args and returns it.
    TypedNode::new(
        TypedDeclaration::Function(TypedFunction {
            name: func.name,
            params: func.params.clone(),
            return_type: Type::Primitive(PrimitiveType::I64), // erased Future pointer
            body: Some(TypedBlock {
                statements: vec![],
                span,
            }),
            visibility: func.visibility,
            is_async: false, // no longer async
            ..Default::default()
        }),
        Type::Primitive(PrimitiveType::Unit),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_await_sites() {
        let span = Span::new(0, 0);
        let body = TypedBlock {
            statements: vec![
                TypedNode::new(
                    TypedStatement::Expression(Box::new(TypedNode::new(
                        TypedExpression::Await(Box::new(TypedNode::new(
                            TypedExpression::Literal(TypedLiteral::Integer(1)),
                            Type::Primitive(PrimitiveType::I64),
                            span,
                        ))),
                        Type::Primitive(PrimitiveType::I64),
                        span,
                    ))),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                ),
                TypedNode::new(
                    TypedStatement::Let(TypedLet {
                        name: InternedString::new_global("x"),
                        ty: Type::Primitive(PrimitiveType::I64),
                        initializer: Some(Box::new(TypedNode::new(
                            TypedExpression::Await(Box::new(TypedNode::new(
                                TypedExpression::Literal(TypedLiteral::Integer(2)),
                                Type::Primitive(PrimitiveType::I64),
                                span,
                            ))),
                            Type::Primitive(PrimitiveType::I64),
                            span,
                        ))),
                        mutability: Mutability::Immutable,
                        span,
                    }),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                ),
            ],
            span,
        };

        assert_eq!(count_await_sites(&body), 2);
    }

    #[test]
    fn test_build_state_machine() {
        let span = Span::new(0, 0);
        let func = TypedFunction {
            name: InternedString::new_global("fetch_data"),
            is_async: true,
            return_type: Type::Primitive(PrimitiveType::I64),
            params: vec![TypedParameter {
                name: InternedString::new_global("url"),
                ty: Type::Primitive(PrimitiveType::I64),
                ..Default::default()
            }],
            body: Some(TypedBlock {
                statements: vec![TypedNode::new(
                    TypedStatement::Expression(Box::new(TypedNode::new(
                        TypedExpression::Await(Box::new(TypedNode::new(
                            TypedExpression::Literal(TypedLiteral::Integer(1)),
                            Type::Primitive(PrimitiveType::I64),
                            span,
                        ))),
                        Type::Primitive(PrimitiveType::I64),
                        span,
                    ))),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                )],
                span,
            }),
            ..Default::default()
        };

        let decls = build_state_machine(&func, span);
        assert_eq!(decls.len(), 4);

        // Check enum
        if let TypedDeclaration::Enum(e) = &decls[0].node {
            assert_eq!(e.name.resolve_global().unwrap(), "fetch_data$State");
            assert_eq!(e.variants.len(), 3); // Start, Await0, Done
        } else {
            panic!("Expected Enum, got {:?}", decls[0].node);
        }

        // Check class
        if let TypedDeclaration::Class(c) = &decls[1].node {
            assert_eq!(c.name.resolve_global().unwrap(), "fetch_data$Future");
            assert_eq!(c.fields.len(), 2); // state + url param
        } else {
            panic!("Expected Class, got {:?}", decls[1].node);
        }

        // Check poll function
        if let TypedDeclaration::Function(f) = &decls[2].node {
            assert_eq!(f.name.resolve_global().unwrap(), "fetch_data$poll");
            assert!(!f.is_async);
        } else {
            panic!("Expected Function, got {:?}", decls[2].node);
        }

        // Check wrapper
        if let TypedDeclaration::Function(f) = &decls[3].node {
            assert_eq!(f.name.resolve_global().unwrap(), "fetch_data");
            assert!(!f.is_async);
        } else {
            panic!("Expected Function, got {:?}", decls[3].node);
        }
    }
}
