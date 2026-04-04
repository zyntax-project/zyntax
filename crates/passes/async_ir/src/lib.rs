//! Async-IR pass — transforms async functions into stackless state machine structs.
//!
//! Two rewrites:
//! - `async_fn_to_state_machine`: async Function with Await → Enum(states) + Class(future) + Function(poll)
//! - `await_expr_to_poll`: Await(inner) → Match(poll_dispatch)
//!
//! After this pass, no async/await remains in the TypedAST.
//! The lowering pipeline sees only plain enums, classes, and functions.

mod state_machine;
mod suspension;

use pattern_engine::{
    Bindings, DeclRewrite, ExprRewrite, Pattern, PatternEngine, PatternPass, Priority,
    RewriteOutput,
};
use zyntax_typed_ast::typed_ast::*;

/// The async-IR pass.
pub struct Pass;

impl PatternPass for Pass {
    fn name(&self) -> &'static str {
        "async-ir"
    }

    fn description(&self) -> &'static str {
        "Transform async/await into stackless state machines"
    }

    fn dependencies(&self) -> &[&'static str] {
        &["normalization"]
    }

    fn register(&self, engine: &mut PatternEngine) {
        engine.register_decl_rewrite(state_machine::async_fn_to_state_machine());
        engine.register_expr_rewrite(suspension::await_expr_to_poll());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pattern_engine::{EngineConfig, PatternEngine};

    #[test]
    fn test_pass_registers() {
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(Pass);
        engine.finalize().unwrap();
    }
}
