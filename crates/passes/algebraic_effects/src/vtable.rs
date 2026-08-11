//! Rewrite: TypedDeclaration::Effect → Class(OpTable)

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use zyntax_typed_ast::type_registry::{Mutability, PrimitiveType, Type, Visibility};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

pub fn effect_decl_to_vtable() -> DeclRewrite {
    // Keeping the matched Effect declaration means the pattern matches
    // it again next iteration, so the op-table class would be emitted
    // once per fixpoint iteration. Record what has been built.
    let built: Arc<Mutex<HashSet<InternedString>>> = Arc::default();
    let seen = Arc::clone(&built);
    DeclRewrite::new(
        "effect_decl_to_vtable",
        Priority::SEMANTIC,
        Pattern::new("effect_decl", move |node, _ctx| match &node.node {
            TypedDeclaration::Effect(e) => {
                let already = seen.lock().map(|s| s.contains(&e.name)).unwrap_or(false);
                (!already).then(Bindings::new)
            }
            _ => None,
        }),
        move |matched, _bindings, _builder| {
            if let TypedDeclaration::Effect(effect) = &matched.node {
                if let Ok(mut s) = built.lock() {
                    if !s.insert(effect.name) {
                        return RewriteOutput::Unchanged;
                    }
                }
                let op_table = build_op_table(effect);
                // Phase H, M1: KEEP the Effect declaration alongside
                // the generated Class$OpTable. Downstream lowering
                // (`LoweringContext::lower_effect_decl`) needs the
                // Effect to build the `HirEffect` entry in
                // `module.effects` — which the SSA builder's
                // `effect_op_map` reads to detect effect-op calls.
                // Previously this rewrite removed Effect, leaving
                // `module.effects` empty and silently disabling
                // PerformEffect emission for ALL source-level effects.
                RewriteOutput::Expand {
                    declarations: vec![op_table],
                    replacement: Some(matched.clone()),
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
            annotations: vec![],
            span,
        }),
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
        } else {
            panic!("Expected Class");
        }
    }
}
