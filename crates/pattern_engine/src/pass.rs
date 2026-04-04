use crate::context::LoweringTarget;
use crate::engine::PatternEngine;

/// A named group of rewrites representing a coherent transformation.
pub trait PatternPass: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;

    /// Register this pass's rewrites with the engine.
    fn register(&self, engine: &mut PatternEngine);

    /// Passes that must run before this one.
    fn dependencies(&self) -> &[&'static str] {
        &[]
    }

    /// Whether this pass is required for the given target.
    fn required_for(&self, _target: LoweringTarget) -> bool {
        true
    }
}
