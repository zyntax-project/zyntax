//! Rewrite: TypedDeclaration::EffectHandler → vtable instance + handler run function
//!
//! A handler like:
//!   handler StateHandler for State {
//!       state: i64
//!       def get(self) -> i64 { return self.state }
//!       def set(self, value: i64) { self.state = value }
//!   }
//!
//! Becomes:
//!   1. A variable holding the vtable instance:
//!      let StateHandler$vtable = State$OpTable { get_fn: StateHandler$get, set_fn: StateHandler$set }
//!   2. Functions for each operation handler:
//!      def StateHandler$get(handler_state: i64) -> i64 { ... }
//!      def StateHandler$set(handler_state: i64, value: i64) { ... }
//!   3. A run function that installs the handler:
//!      def StateHandler$run(body_fn: fn() -> i64, initial_state: i64) -> i64 { ... }

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

/// Match TypedDeclaration::EffectHandler and expand into functions + vtable instance.
pub fn handler_decl_to_impl() -> DeclRewrite {
    DeclRewrite::new(
        "handler_decl_to_impl",
        Priority::SEMANTIC,
        Pattern::new("handler_decl", |node, _ctx| {
            if let TypedDeclaration::EffectHandler(handler) = &node.node {
                let mut bindings = Bindings::new();
                bindings.bind_span("span", handler.span);
                Some(bindings)
            } else {
                None
            }
        }),
        |_bindings, _builder| {
            // Same limitation as vtable.rs — apply fn doesn't receive the matched node.
            // TODO: Wire the matched EffectHandler through to build_handler_functions().
            RewriteOutput::Unchanged
        },
    )
}

/// Build the handler's operation functions from an EffectHandler definition.
/// Each handler operation impl becomes a standalone function with a mangled name.
#[allow(dead_code)]
fn build_handler_functions(handler: &TypedEffectHandler) -> Vec<TypedNode<TypedDeclaration>> {
    let handler_name = handler.name.resolve_global().unwrap_or_default();
    let span = handler.span;

    handler
        .handlers
        .iter()
        .map(|impl_| {
            let op_name = impl_.op_name.resolve_global().unwrap_or_default();
            let func_name = InternedString::new_global(&format!("{}${}", handler_name, op_name));

            let func = TypedFunction {
                name: func_name,
                type_params: impl_.type_params.clone(),
                params: impl_.params.clone(),
                return_type: impl_.return_type.clone(),
                body: impl_.body.clone(),
                visibility: Visibility::Public,
                is_pure: false,
                ..Default::default()
            };

            TypedNode::new(
                TypedDeclaration::Function(func),
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
    fn test_build_handler_functions() {
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
                    params: vec![],
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

        let funcs = build_handler_functions(&handler);
        assert_eq!(funcs.len(), 2);

        if let TypedDeclaration::Function(f) = &funcs[0].node {
            assert_eq!(f.name.resolve_global().unwrap(), "StateHandler$get");
        } else {
            panic!("Expected Function declaration");
        }

        if let TypedDeclaration::Function(f) = &funcs[1].node {
            assert_eq!(f.name.resolve_global().unwrap(), "StateHandler$set");
        } else {
            panic!("Expected Function declaration");
        }
    }
}
