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

/// Extract `@effect(E1, E2, ...)` and `@with(H1, H2, ...)` annotation
/// arguments into the structured `TypedFunction.effects` and
/// `TypedFunction.with_handlers` fields. Idempotent — once both
/// fields are populated (or the function has neither annotation),
/// the rewrite stops matching.
pub fn extract_effect_annotations() -> DeclRewrite {
    DeclRewrite::new(
        "extract_effect_annotations",
        Priority::NORMALIZATION,
        Pattern::new("fn_with_unprocessed_effect_annotations", |node, _ctx| {
            if let TypedDeclaration::Function(func) = &node.node {
                if !func.effects.is_empty() || !func.with_handlers.is_empty() {
                    // Already processed by a prior iteration.
                    return None;
                }
                let has_effect_or_with = func.annotations.iter().any(|a| {
                    matches!(
                        a.name.resolve_global().as_deref(),
                        Some("effect") | Some("with")
                    )
                });
                if has_effect_or_with {
                    return Some(Bindings::new());
                }
            }
            None
        }),
        |matched, _bindings, _builder| {
            if let TypedDeclaration::Function(func) = &matched.node {
                let mut new_func = func.clone();
                for ann in &func.annotations {
                    let ann_name = ann.name.resolve_global();
                    let target: Option<&mut Vec<_>> = match ann_name.as_deref() {
                        Some("effect") => Some(&mut new_func.effects),
                        Some("with") => Some(&mut new_func.with_handlers),
                        _ => None,
                    };
                    let target = match target {
                        Some(t) => t,
                        None => continue,
                    };
                    for arg in &ann.args {
                        if let TypedAnnotationArg::Positional(TypedAnnotationValue::Identifier(
                            name,
                        )) = arg
                        {
                            target.push(*name);
                        }
                        // Other arg shapes (Call exprs like `MCMC(warmup=500)`)
                        // are out of scope for M1 — the simple identifier form
                        // is what the grammar's annotation_ident_value produces
                        // for the bare-handler-name case.
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

    fn ident_arg(name: &str) -> TypedAnnotationArg {
        TypedAnnotationArg::Positional(TypedAnnotationValue::Identifier(
            InternedString::new_global(name),
        ))
    }

    fn run_pass_on(func: TypedFunction) -> TypedFunction {
        let span = Span::new(0, 0);
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
        program
            .declarations
            .into_iter()
            .find_map(|d| {
                if let TypedDeclaration::Function(f) = d.node {
                    Some(f)
                } else {
                    None
                }
            })
            .expect("function should still be in program")
    }

    fn names(v: &[InternedString]) -> Vec<String> {
        v.iter()
            .map(|s| s.resolve_global().unwrap_or_default())
            .collect()
    }

    #[test]
    fn at_effect_annotation_populates_effects_field() {
        let span = Span::new(0, 0);
        let func = TypedFunction {
            name: InternedString::new_global("use_effects"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![],
                span,
            }),
            annotations: vec![TypedAnnotation {
                name: InternedString::new_global("effect"),
                args: vec![ident_arg("State"), ident_arg("Log")],
                span,
            }],
            ..Default::default()
        };
        let func = run_pass_on(func);
        assert_eq!(names(&func.effects), vec!["State", "Log"]);
        assert!(func.with_handlers.is_empty());
    }

    #[test]
    fn at_with_annotation_populates_with_handlers_field() {
        let span = Span::new(0, 0);
        let func = TypedFunction {
            name: InternedString::new_global("run"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![],
                span,
            }),
            annotations: vec![TypedAnnotation {
                name: InternedString::new_global("with"),
                args: vec![ident_arg("ConsoleLogger")],
                span,
            }],
            ..Default::default()
        };
        let func = run_pass_on(func);
        assert!(func.effects.is_empty());
        assert_eq!(names(&func.with_handlers), vec!["ConsoleLogger"]);
    }

    #[test]
    fn at_effect_and_at_with_combined() {
        let span = Span::new(0, 0);
        let func = TypedFunction {
            name: InternedString::new_global("compute"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![],
                span,
            }),
            annotations: vec![
                TypedAnnotation {
                    name: InternedString::new_global("effect"),
                    args: vec![ident_arg("Probabilistic")],
                    span,
                },
                TypedAnnotation {
                    name: InternedString::new_global("with"),
                    args: vec![ident_arg("MCMC"), ident_arg("ConsoleLogger")],
                    span,
                },
            ],
            ..Default::default()
        };
        let func = run_pass_on(func);
        assert_eq!(names(&func.effects), vec!["Probabilistic"]);
        assert_eq!(names(&func.with_handlers), vec!["MCMC", "ConsoleLogger"]);
    }

    #[test]
    fn no_effect_or_with_annotation_leaves_fields_empty() {
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
        let func = run_pass_on(func);
        assert!(func.effects.is_empty(), "@inline-only fn should have no effects");
        assert!(
            func.with_handlers.is_empty(),
            "@inline-only fn should have no handlers"
        );
    }
}
