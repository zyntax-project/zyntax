//! Rewrite: TypedDeclaration::EffectHandler → functions + vtable variable

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

pub fn handler_decl_to_impl() -> DeclRewrite {
    DeclRewrite::new(
        "handler_decl_to_impl",
        Priority::SEMANTIC,
        Pattern::new("handler_decl", |node, _ctx| {
            matches!(&node.node, TypedDeclaration::EffectHandler(_)).then(Bindings::new)
        }),
        |matched, _bindings, _builder| {
            if let TypedDeclaration::EffectHandler(handler) = &matched.node {
                let declarations = build_handler_declarations(handler);
                RewriteOutput::Expand {
                    declarations,
                    replacement: None, // remove the EffectHandler declaration
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
