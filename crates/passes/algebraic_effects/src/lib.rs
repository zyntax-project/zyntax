//! Algebraic effects pass — transforms effect declarations into plain TypedAST constructs.
//!
//! Three rewrites:
//! - `effect_decl_to_vtable`: TypedDeclaration::Effect → Class(OpTable) of function pointers
//! - `handler_decl_to_impl`: TypedDeclaration::EffectHandler → vtable instance + handler run fn
//! - `effect_op_to_dispatch`: Call to effect op → handler dispatch through vtable
//!
//! After this pass, the program contains no Effect or EffectHandler declarations.
//! The lowering pipeline sees only classes, functions, and variables.

mod continuation;
mod dispatch;
mod vtable;

use pattern_engine::{
    Bindings, DeclRewrite, ExprRewrite, Pattern, PatternEngine, PatternPass, Priority,
    RewriteOutput,
};
use zyntax_typed_ast::typed_ast::*;

/// The algebraic effects pass.
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
        engine.register_decl_rewrite(vtable::effect_decl_to_vtable());
        engine.register_decl_rewrite(dispatch::handler_decl_to_impl());
        engine.register_expr_rewrite(continuation::effect_op_to_dispatch());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pattern_engine::{EngineConfig, PatternEngine};

    #[test]
    fn test_pass_registers_with_dependency() {
        let mut engine = PatternEngine::new(EngineConfig::default());
        // Need normalization registered first (dependency)
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(Pass);
        engine.finalize().unwrap();
    }

    #[test]
    fn test_missing_dependency_errors() {
        let mut engine = PatternEngine::new(EngineConfig::default());
        // Register effects pass WITHOUT normalization → should error
        engine.register_pass(Pass);
        let result = engine.finalize();
        assert!(result.is_err());
    }
}
