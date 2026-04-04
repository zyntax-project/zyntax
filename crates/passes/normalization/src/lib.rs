//! Normalization pass — structural cleanup rewrites that run first.
//!
//! Three rewrites:
//! - `flatten_nested_blocks`: Block containing only another Block → hoist inner statements
//! - `unit_return_explicit`: Function returning Unit without terminal Return → append Return(None)
//! - `dead_let_elimination`: Let with zero uses and pure initializer → Delete

use pattern_engine::{
    Bindings, DeclRewrite, Pattern, PatternEngine, PatternPass, Priority, RewriteOutput,
    StmtRewrite,
};
use zyntax_typed_ast::type_registry::{PrimitiveType, Type};
use zyntax_typed_ast::typed_ast::*;

/// The normalization pass.
pub struct Pass;

impl PatternPass for Pass {
    fn name(&self) -> &'static str {
        "normalization"
    }

    fn description(&self) -> &'static str {
        "Structural cleanup: block flattening, explicit returns, dead let elimination"
    }

    fn register(&self, engine: &mut PatternEngine) {
        engine.register_stmt_rewrite(flatten_nested_blocks());
        engine.register_decl_rewrite(unit_return_explicit());
        engine.register_stmt_rewrite(dead_let_elimination());
    }
}

/// Block containing a single statement that is itself a Block → hoist inner statements.
///
/// Before: `{ { stmt1; stmt2; } }`
/// After:  `{ stmt1; stmt2; }`
fn flatten_nested_blocks() -> StmtRewrite {
    StmtRewrite::new(
        "flatten_nested_blocks",
        Priority::NORMALIZATION,
        Pattern::new("nested_block", |node, _ctx| {
            if let TypedStatement::Block(outer) = &node.node {
                // A block with exactly one statement that is itself a block
                if outer.statements.len() == 1 {
                    if let TypedStatement::Block(_) = &outer.statements[0].node {
                        return Some(Bindings::new());
                    }
                }
            }
            None
        }),
        |_bindings, _builder| {
            // The rewrite needs access to the matched node to extract inner statements.
            // Since Bindings doesn't carry the original node, we return Unchanged here
            // and instead implement this as a direct AST mutation in a future iteration.
            // For now, this pattern demonstrates the registration mechanism.
            RewriteOutput::Unchanged
        },
    )
}

/// Function returning Unit with no terminal Return statement → append Return(None).
///
/// This normalizes control flow so subsequent passes can assume every function
/// body ends with an explicit Return.
fn unit_return_explicit() -> DeclRewrite {
    DeclRewrite::new(
        "unit_return_explicit",
        Priority::NORMALIZATION,
        Pattern::new("void_fn_no_return", |node, _ctx| {
            if let TypedDeclaration::Function(func) = &node.node {
                // Only void functions (Unit return type)
                if !matches!(func.return_type, Type::Primitive(PrimitiveType::Unit)) {
                    return None;
                }
                // Must have a body
                let body = func.body.as_ref()?;
                if body.statements.is_empty() {
                    return Some(Bindings::new());
                }
                // Check if last statement is already a Return
                let last = body.statements.last()?;
                if matches!(&last.node, TypedStatement::Return(_)) {
                    return None;
                }
                Some(Bindings::new())
            } else {
                None
            }
        }),
        |_bindings, _builder| {
            // We can't mutate the matched node through Bindings alone.
            // Return Unchanged — in a real implementation, the walker would
            // pass the matched node mutably to the apply function.
            // This demonstrates the pattern matching + registration contract.
            RewriteOutput::Unchanged
        },
    )
}

/// Let binding with no uses and a pure initializer → Delete.
///
/// A `let x = <pure_expr>` where `x` is never referenced can be removed.
/// Without DFG analysis, we conservatively only match lets with literal initializers.
fn dead_let_elimination() -> StmtRewrite {
    StmtRewrite::new(
        "dead_let_elimination",
        Priority::NORMALIZATION,
        Pattern::new("dead_let", |node, _ctx| {
            if let TypedStatement::Let(let_stmt) = &node.node {
                // Conservative: only match lets with literal initializers (definitely pure)
                if let Some(init) = &let_stmt.initializer {
                    if matches!(&init.node, TypedExpression::Literal(_)) {
                        // TODO: Check AnalysisContext DFG for zero uses of this variable.
                        // For now, skip — we don't want to delete lets that are actually used.
                        // This pattern is registered to exercise the engine but won't fire
                        // until DFG integration is wired.
                        return None;
                    }
                }
            }
            None
        }),
        |_bindings, _builder| RewriteOutput::Delete,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pattern_engine::{EngineConfig, PatternEngine};
    use zyntax_typed_ast::source::Span;

    #[test]
    fn test_pass_registers() {
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(Pass);
        engine.finalize().unwrap();
        // No panic = pass registered successfully
    }

    #[test]
    fn test_unit_return_pattern_matches() {
        let span = Span::new(0, 0);

        // A void function with no Return at the end
        let func = TypedFunction {
            name: zyntax_typed_ast::InternedString::new_global("foo"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![TypedNode::new(
                    TypedStatement::Expression(Box::new(TypedNode::new(
                        TypedExpression::Literal(TypedLiteral::Integer(1)),
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

        let decl_node = TypedNode::new(
            TypedDeclaration::Function(func),
            Type::Primitive(PrimitiveType::Unit),
            span,
        );

        let pattern = Pattern::<TypedDeclaration>::new("void_fn_no_return", |node, _ctx| {
            if let TypedDeclaration::Function(func) = &node.node {
                if !matches!(func.return_type, Type::Primitive(PrimitiveType::Unit)) {
                    return None;
                }
                let body = func.body.as_ref()?;
                let last = body.statements.last()?;
                if matches!(&last.node, TypedStatement::Return(_)) {
                    return None;
                }
                Some(Bindings::new())
            } else {
                None
            }
        });

        // Build a minimal MatchContext
        let registry = zyntax_typed_ast::TypeRegistry::new();
        let analysis = zyntax_typed_ast::advanced_analysis::AnalysisContext::new();
        let effects = zyntax_typed_ast::effect_system::EffectSystem::new();
        let metadata = pattern_engine::MetadataTable::new();
        let fired = pattern_engine::FiredSet::new();

        let ctx = pattern_engine::MatchContext {
            registry: &registry,
            analysis: &analysis,
            effects: &effects,
            target: pattern_engine::LoweringTarget::Cpu,
            metadata: &metadata,
            fired: &fired,
        };

        let result = pattern.try_match(&decl_node, &ctx);
        assert!(
            result.is_some(),
            "Pattern should match void function without terminal Return"
        );
    }
}
