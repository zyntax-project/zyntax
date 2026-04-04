//! Rewrite: TypedDeclaration::Effect → Class(OpTable)
//!
//! An effect like:
//!   effect Probabilistic {
//!       def sample<T>(distribution: Distribution<T>): T
//!       def observe<T>(value: T, distribution: Distribution<T>): ()
//!   }
//!
//! Becomes a class with one function-pointer field per operation:
//!   class Probabilistic$OpTable {
//!       sample_fn: fn(i64) -> i64,    // erased function pointers
//!       observe_fn: fn(i64, i64) -> (),
//!   }

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{Mutability, PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

/// Match TypedDeclaration::Effect and expand into a Class(OpTable).
pub fn effect_decl_to_vtable() -> DeclRewrite {
    DeclRewrite::new(
        "effect_decl_to_vtable",
        Priority::SEMANTIC,
        Pattern::new("effect_decl", |node, _ctx| {
            if let TypedDeclaration::Effect(effect) = &node.node {
                let mut bindings = Bindings::new();
                bindings.bind_span("span", effect.span);
                Some(bindings)
            } else {
                None
            }
        }),
        |_bindings, _builder| {
            // The actual Effect node is matched, but we need the original data.
            // Since the current Bindings contract doesn't carry the original node,
            // we bind the effect in the pattern and reconstruct here.
            //
            // In the current architecture, the walker passes the node to the pattern
            // but the apply fn only receives Bindings. To access the original Effect,
            // we'd need to bind it as a decl. For now, return Unchanged and let the
            // existing HIR lowering handle it.
            //
            // TODO: Extend Bindings or Rewrite<T> to pass the matched node to apply().
            // Once that's done, this rewrite synthesizes:
            //   Class { name: "{effect_name}$OpTable", fields: [fn_ptr per op] }
            RewriteOutput::Unchanged
        },
    )
}

/// Build an OpTable class declaration from an Effect definition.
/// This is the builder function that will be called once the rewrite
/// has access to the matched Effect node.
#[allow(dead_code)]
fn build_op_table(effect: &TypedEffect) -> TypedNode<TypedDeclaration> {
    let span = effect.span;
    let effect_name = effect.name.resolve_global().unwrap_or_default();
    let table_name = InternedString::new_global(&format!("{}$OpTable", effect_name));

    // One field per operation — each is a function pointer (erased as i64)
    let fields: Vec<TypedField> = effect
        .operations
        .iter()
        .map(|op| {
            let op_name = op.name.resolve_global().unwrap_or_default();
            let field_name = InternedString::new_global(&format!("{}_fn", op_name));
            TypedField {
                name: field_name,
                ty: Type::Primitive(PrimitiveType::I64), // erased function pointer
                initializer: None,
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                span,
            }
        })
        .collect();

    let class = TypedClass {
        name: table_name,
        type_params: effect.type_params.clone(),
        extends: None,
        implements: vec![],
        fields,
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_final: true,
        span,
    };

    TypedNode::new(
        TypedDeclaration::Class(class),
        Type::Primitive(PrimitiveType::Unit),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::source::Span;

    #[test]
    fn test_build_op_table() {
        let span = Span::new(0, 0);
        let effect = TypedEffect {
            name: InternedString::new_global("State"),
            type_params: vec![],
            operations: vec![
                TypedEffectOp {
                    name: InternedString::new_global("get"),
                    type_params: vec![],
                    params: vec![],
                    return_type: Type::Primitive(PrimitiveType::I64),
                    span,
                },
                TypedEffectOp {
                    name: InternedString::new_global("set"),
                    type_params: vec![],
                    params: vec![TypedParameter {
                        name: InternedString::new_global("value"),
                        ty: Type::Primitive(PrimitiveType::I64),
                        ..Default::default()
                    }],
                    return_type: Type::Primitive(PrimitiveType::Unit),
                    span,
                },
            ],
            span,
        };

        let result = build_op_table(&effect);
        if let TypedDeclaration::Class(class) = &result.node {
            assert_eq!(class.name.resolve_global().unwrap(), "State$OpTable");
            assert_eq!(class.fields.len(), 2);
            assert_eq!(class.fields[0].name.resolve_global().unwrap(), "get_fn");
            assert_eq!(class.fields[1].name.resolve_global().unwrap(), "set_fn");
            assert!(class.is_final);
        } else {
            panic!("Expected Class declaration");
        }
    }
}
