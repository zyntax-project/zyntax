//! Algebraic effects pass — transforms effect declarations into plain TypedAST constructs.
//!
//! Two complete rewrites:
//! - `effect_decl_to_vtable`: Effect → Class(OpTable) with fn-ptr fields per operation
//! - `handler_decl_to_impl`: EffectHandler → standalone functions per handler operation
//!
//! One pending rewrite:
//! - `effect_op_to_dispatch`: Call to effect op → handler dispatch (needs MatchContext integration)
//!
//! After the complete rewrites fire, the program contains no Effect or EffectHandler
//! declarations — they've been expanded into classes and functions.

mod annotations;
mod continuation;
mod dispatch;
mod vtable;

use pattern_engine::{PatternEngine, PatternPass};

pub struct Pass;

impl PatternPass for Pass {
    fn name(&self) -> &'static str {
        "algebraic-effects"
    }

    fn description(&self) -> &'static str {
        "CPS-transform effect operations, generate handler dispatch tables"
    }

    fn dependencies(&self) -> &[&'static str] {
        &["normalization"]
    }

    fn register(&self, engine: &mut PatternEngine) {
        // Phase H, M1 step 1: normalize @effect(...) annotations into
        // the structured `TypedFunction.effects` field BEFORE any
        // semantic rewrites fire — downstream passes (and eventually
        // the SSA builder) consume this normalized form.
        engine.register_decl_rewrite(annotations::extract_effect_annotations());
        engine.register_decl_rewrite(vtable::effect_decl_to_vtable());
        engine.register_decl_rewrite(dispatch::handler_decl_to_impl());
        engine.register_expr_rewrite(continuation::effect_op_to_dispatch());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pattern_engine::{EngineConfig, PatternEngine};
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::{PrimitiveType, Type, Visibility};
    use zyntax_typed_ast::typed_ast::*;
    use zyntax_typed_ast::InternedString;

    #[test]
    fn test_pass_registers_with_dependency() {
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(Pass);
        engine.finalize().unwrap();
    }

    #[test]
    fn test_missing_dependency_errors() {
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(Pass);
        assert!(engine.finalize().is_err());
    }

    #[test]
    fn test_effect_decl_rewritten_to_class() {
        let span = Span::new(0, 0);
        let effect = TypedEffect {
            name: InternedString::new_global("State"),
            type_params: vec![],
            operations: vec![TypedEffectOp {
                name: InternedString::new_global("get"),
                type_params: vec![],
                params: vec![],
                return_type: Type::Primitive(PrimitiveType::I64),
                span,
            }],
            span,
        };

        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Effect(effect),
                Type::Primitive(PrimitiveType::Unit),
                span,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(Pass);
        engine.finalize().unwrap();

        let result = engine.run(&mut program, &registry);
        assert!(result.changed);

        // The Effect declaration should be gone, replaced by a Class
        assert!(
            !program
                .declarations
                .iter()
                .any(|d| matches!(&d.node, TypedDeclaration::Effect(_))),
            "Effect declaration should be removed"
        );
        assert!(
            program
                .declarations
                .iter()
                .any(|d| matches!(&d.node, TypedDeclaration::Class(_))),
            "Class(OpTable) should be added"
        );

        // Check the class name
        if let Some(class_decl) = program
            .declarations
            .iter()
            .find(|d| matches!(&d.node, TypedDeclaration::Class(_)))
        {
            if let TypedDeclaration::Class(class) = &class_decl.node {
                assert_eq!(class.name.resolve_global().unwrap(), "State$OpTable");
                assert_eq!(class.fields.len(), 1);
                assert_eq!(class.fields[0].name.resolve_global().unwrap(), "get_fn");
            }
        }
    }

    #[test]
    fn test_handler_decl_rewritten_to_functions() {
        let span = Span::new(0, 0);
        let handler = TypedEffectHandler {
            name: InternedString::new_global("MyHandler"),
            effect_name: InternedString::new_global("State"),
            type_params: vec![],
            fields: vec![],
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

        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::EffectHandler(handler),
                Type::Primitive(PrimitiveType::Unit),
                span,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(Pass);
        engine.finalize().unwrap();

        let result = engine.run(&mut program, &registry);
        assert!(result.changed);

        // EffectHandler should be gone
        assert!(
            !program
                .declarations
                .iter()
                .any(|d| matches!(&d.node, TypedDeclaration::EffectHandler(_))),
            "EffectHandler should be removed"
        );

        // Two functions should be added
        let func_names: Vec<String> = program
            .declarations
            .iter()
            .filter_map(|d| {
                if let TypedDeclaration::Function(f) = &d.node {
                    f.name.resolve_global()
                } else {
                    None
                }
            })
            .collect();
        assert!(func_names.contains(&"MyHandler$get".to_string()));
        assert!(func_names.contains(&"MyHandler$set".to_string()));
    }
}
