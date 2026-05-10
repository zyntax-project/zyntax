//! Rewrite: extract `@effect(EffectName, ...)` annotations into the
//! structured `TypedFunction.effects` field.
//!
//! Phase H, M1, step 1. The parser preserves all annotations as raw
//! `TypedAnnotation` entries on `TypedFunction.annotations`. The
//! `effects: Vec<InternedString>` field exists but is never populated
//! from the parser — downstream code (the SSA builder, krio's
//! `HirSuspendingFns`) needs this normalized form to detect that a
//! function performs effects.
//!
//! This rewrite runs at `NORMALIZATION` priority so it fires before
//! any semantic rewrites in the algebraic_effects pass. It is
//! idempotent — once `effects` is populated, re-running is a no-op.

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::typed_ast::*;

/// Extract `@effect(E1, E2, ...)` annotation arguments into
/// `TypedFunction.effects`. Idempotent.
pub fn extract_effect_annotations() -> DeclRewrite {
    DeclRewrite::new(
        "extract_effect_annotations",
        Priority::NORMALIZATION,
        Pattern::new("fn_with_unprocessed_effect_annotations", |node, _ctx| {
            if let TypedDeclaration::Function(func) = &node.node {
                // Match only if the function has an @effect annotation
                // AND `effects` hasn't already been populated. The
                // second condition makes the rewrite idempotent —
                // important because the pattern engine may re-iterate.
                if !func.effects.is_empty() {
                    return None;
                }
                let has_effect_ann = func.annotations.iter().any(|a| {
                    a.name.resolve_global().as_deref() == Some("effect")
                });
                if has_effect_ann {
                    return Some(Bindings::new());
                }
            }
            None
        }),
        |matched, _bindings, _builder| {
            if let TypedDeclaration::Function(func) = &matched.node {
                let mut new_func = func.clone();
                for ann in &func.annotations {
                    if ann.name.resolve_global().as_deref() != Some("effect") {
                        continue;
                    }
                    for arg in &ann.args {
                        match arg {
                            TypedAnnotationArg::Positional(TypedAnnotationValue::Identifier(
                                name,
                            )) => {
                                new_func.effects.push(*name);
                            }
                            // @effect(Probabilistic, Differentiable) — args
                            // are identifiers. Other arg shapes are
                            // ignored (lenient — matches what
                            // `@deprecated("msg")` etc. would do).
                            _ => {}
                        }
                    }
                }
                RewriteOutput::ReplaceDecl(TypedNode::new(
                    TypedDeclaration::Function(new_func),
                    matched.ty.clone(),
                    matched.span,
                ))
            } else {
                RewriteOutput::Unchanged
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pattern_engine::{EngineConfig, PatternEngine};
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::{PrimitiveType, Type};
    use zyntax_typed_ast::InternedString;

    #[test]
    fn at_effect_annotation_populates_effects_field() {
        let span = Span::new(0, 0);

        // @effect(State, Log) def use_effects() {}
        let func = TypedFunction {
            name: InternedString::new_global("use_effects"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![],
                span,
            }),
            annotations: vec![TypedAnnotation {
                name: InternedString::new_global("effect"),
                args: vec![
                    TypedAnnotationArg::Positional(TypedAnnotationValue::Identifier(
                        InternedString::new_global("State"),
                    )),
                    TypedAnnotationArg::Positional(TypedAnnotationValue::Identifier(
                        InternedString::new_global("Log"),
                    )),
                ],
                span,
            }],
            ..Default::default()
        };

        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Function(func),
                Type::Primitive(PrimitiveType::Unit),
                span,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(super::super::Pass);
        engine.finalize().unwrap();
        let result = engine.run(&mut program, &registry);
        assert!(result.changed);

        let func = program.declarations.iter().find_map(|d| {
            if let TypedDeclaration::Function(f) = &d.node {
                Some(f)
            } else {
                None
            }
        });
        let func = func.expect("function should still be in program");
        let effect_names: Vec<String> = func
            .effects
            .iter()
            .map(|s| s.resolve_global().unwrap_or_default())
            .collect();
        assert_eq!(effect_names, vec!["State", "Log"]);
    }

    #[test]
    fn no_effect_annotation_leaves_effects_empty() {
        let span = Span::new(0, 0);
        let func = TypedFunction {
            name: InternedString::new_global("plain"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![],
                span,
            }),
            annotations: vec![TypedAnnotation {
                name: InternedString::new_global("inline"),
                args: vec![],
                span,
            }],
            ..Default::default()
        };

        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Function(func),
                Type::Primitive(PrimitiveType::Unit),
                span,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(super::super::Pass);
        engine.finalize().unwrap();
        engine.run(&mut program, &registry);

        let func = program
            .declarations
            .iter()
            .find_map(|d| {
                if let TypedDeclaration::Function(f) = &d.node {
                    Some(f)
                } else {
                    None
                }
            })
            .expect("function should still be in program");
        assert!(func.effects.is_empty(), "@inline-only fn should have no effects");
    }
}
