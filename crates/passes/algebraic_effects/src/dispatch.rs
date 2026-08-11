//! Rewrite: TypedDeclaration::EffectHandler → functions + vtable variable

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

pub fn handler_decl_to_impl() -> DeclRewrite {
    // The rewrite keeps the handler declaration it matched, so the
    // pattern would match it again on the next fixpoint iteration and
    // emit another copy of every op function. Record which handlers
    // have been expanded so the second visit stops matching, the way
    // `extract_effect_annotations` reads its own output.
    let expanded: Arc<Mutex<HashSet<InternedString>>> = Arc::default();
    let seen = Arc::clone(&expanded);
    DeclRewrite::new(
        "handler_decl_to_impl",
        Priority::SEMANTIC,
        Pattern::new("handler_decl", move |node, _ctx| match &node.node {
            TypedDeclaration::EffectHandler(h) => {
                let already = seen.lock().map(|s| s.contains(&h.name)).unwrap_or(false);
                (!already).then(Bindings::new)
            }
            _ => None,
        }),
        move |matched, _bindings, _builder| {
            if let TypedDeclaration::EffectHandler(handler) = &matched.node {
                if let Ok(mut s) = expanded.lock() {
                    if !s.insert(handler.name) {
                        return RewriteOutput::Unchanged;
                    }
                }
                let declarations = build_handler_declarations(handler);
                // Phase H, M1: KEEP the EffectHandler declaration so
                // `LoweringContext::lower_effect_handler` can build
                // the `HirEffectHandler` entry in `module.handlers`.
                // Without it, `module.handlers` is empty after
                // lowering and `PerformEffect`'s handler-lookup at
                // `cranelift_backend.rs:3796` fails for all effects.
                RewriteOutput::Expand {
                    declarations,
                    replacement: Some(matched.clone()),
                }
            } else {
                RewriteOutput::Unchanged
            }
        },
    )
}

/// Build standalone functions from handler operation implementations.
/// Each handler op becomes: `def HandlerName$op_name(params...) { body }`
fn build_handler_declarations(handler: &TypedEffectHandler) -> Vec<TypedNode<TypedDeclaration>> {
    let handler_name = handler.name.resolve_global().unwrap_or_default();
    let span = handler.span;

    handler
        .handlers
        .iter()
        .map(|impl_| {
            let op_name = impl_.op_name.resolve_global().unwrap_or_default();
            let func_name = InternedString::new_global(&format!("{}${}", handler_name, op_name));

            TypedNode::new(
                TypedDeclaration::Function(TypedFunction {
                    name: func_name,
                    type_params: impl_.type_params.clone(),
                    params: impl_.params.clone(),
                    return_type: impl_.return_type.clone(),
                    body: impl_.body.clone(),
                    visibility: Visibility::Public,
                    is_pure: false,
                    // An `async def` handler op becomes an async fn: it awaits
                    // in its body and parks the performer until it resumes.
                    is_async: impl_.is_async,
                    ..Default::default()
                }),
                Type::Primitive(PrimitiveType::Unit),
                span,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::Mutability;

    #[test]
    fn test_build_handler_declarations() {
        let span = Span::new(0, 0);
        let handler = TypedEffectHandler {
            name: InternedString::new_global("StateHandler"),
            effect_name: InternedString::new_global("State"),
            type_params: vec![],
            fields: vec![TypedField {
                name: InternedString::new_global("state"),
                ty: Type::Primitive(PrimitiveType::I64),
                initializer: None,
                visibility: Visibility::Public,
                mutability: Mutability::Mutable,
                is_static: false,
                span,
            }],
            handlers: vec![
                TypedEffectHandlerImpl {
                    op_name: InternedString::new_global("get"),
                    return_type: Type::Primitive(PrimitiveType::I64),
                    body: Some(TypedBlock {
                        statements: vec![],
                        span,
                    }),
                    ..Default::default()
                },
                TypedEffectHandlerImpl {
                    op_name: InternedString::new_global("set"),
                    params: vec![TypedParameter {
                        name: InternedString::new_global("value"),
                        ty: Type::Primitive(PrimitiveType::I64),
                        ..Default::default()
                    }],
                    return_type: Type::Primitive(PrimitiveType::Unit),
                    body: Some(TypedBlock {
                        statements: vec![],
                        span,
                    }),
                    ..Default::default()
                },
            ],
            span,
        };

        let decls = build_handler_declarations(&handler);
        assert_eq!(decls.len(), 2);

        if let TypedDeclaration::Function(f) = &decls[0].node {
            assert_eq!(f.name.resolve_global().unwrap(), "StateHandler$get");
        } else {
            panic!("Expected Function");
        }
        if let TypedDeclaration::Function(f) = &decls[1].node {
            assert_eq!(f.name.resolve_global().unwrap(), "StateHandler$set");
        } else {
            panic!("Expected Function");
        }
    }
}
