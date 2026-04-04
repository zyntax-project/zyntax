//! Rewrite: TypedDeclaration::Effect → Class(OpTable)

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{Mutability, PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

pub fn effect_decl_to_vtable() -> DeclRewrite {
    DeclRewrite::new(
        "effect_decl_to_vtable",
        Priority::SEMANTIC,
        Pattern::new("effect_decl", |node, _ctx| {
            matches!(&node.node, TypedDeclaration::Effect(_)).then(Bindings::new)
        }),
        |matched, _bindings, _builder| {
            if let TypedDeclaration::Effect(effect) = &matched.node {
                let op_table = build_op_table(effect);
                RewriteOutput::Expand {
                    declarations: vec![op_table],
                    replacement: None, // remove the Effect declaration
                }
            } else {
                RewriteOutput::Unchanged
            }
        },
    )
}

/// Build a Class declaration representing the operation table for an effect.
/// One field per operation — each is an erased function pointer (i64).
fn build_op_table(effect: &TypedEffect) -> TypedNode<TypedDeclaration> {
    let span = effect.span;
    let effect_name = effect.name.resolve_global().unwrap_or_default();
    let table_name = InternedString::new_global(&format!("{}$OpTable", effect_name));

    let fields: Vec<TypedField> = effect
        .operations
        .iter()
        .map(|op| {
            let op_name = op.name.resolve_global().unwrap_or_default();
            TypedField {
                name: InternedString::new_global(&format!("{}_fn", op_name)),
                ty: Type::Primitive(PrimitiveType::I64), // erased function pointer
                initializer: None,
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                span,
            }
        })
        .collect();

    TypedNode::new(
        TypedDeclaration::Class(TypedClass {
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
        }),
        Type::Primitive(PrimitiveType::Unit),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        } else {
            panic!("Expected Class");
        }
    }
}
