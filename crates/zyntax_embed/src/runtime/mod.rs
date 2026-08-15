//! ZyntaxRuntime - Compiler execution API for embedding Zyntax
//!
//! This module provides a high-level API for compiling and executing Zyntax code
//! from Rust, with automatic value conversion and async/await support.

use crate::convert::FromZyntax;
use crate::error::{ConversionError, ZyntaxError};
use crate::grammar::{GrammarError, LanguageGrammar};
use crate::value::ZyntaxValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use zyntax_compiler::{
    cranelift_backend::CraneliftBackend,
    hir::{HirId, HirModule},
    lowering::AstLowering, // For lower_program trait method
    runtime::{Executor, Waker as RuntimeWaker},
    tiered_backend::{OptimizationTier, TieredBackend, TieredConfig, TieredStatistics},
    zrtl::DynamicValue,
    CompilationConfig,
    CompilerError,
};

/// Handler state (Phase 3): synthesize the state struct, constructor, and
/// implicit `self` param for every stateful handler. Runs on the parsed
/// program before the type registry is snapshotted. See the call site in
/// `lower_typed_program`.
mod classic;
mod events;
mod handler_state;
mod native_call;
mod promise;
mod tiered;
mod types;

pub use classic::{ExternalFunction, ZyntaxRuntime};
use events::capture_runtime_events_from_program;
use handler_state::synthesize_handler_state;
use native_call::{
    call_dynamic_function, call_native_with_signature, call_with_signature, dynamic_to_i64,
};
pub use promise::{
    drive_tasks, AsyncPollResult, PromiseAll, PromiseAllSettled, PromiseAllState, PromiseRace,
    PromiseRaceState, PromiseState, SettledResult, ZyntaxPromise,
};
pub use tiered::{
    EffectHandlerToken, FiberToken, HandlerContext, HandlerContextScope, HandlerFrame,
    HandlerInstance, HostFiberInfo, HostFiberStep, TieredRuntime,
};
pub use types::{
    BuiltinResolver, ChainedResolver, CompiledImportResolverCallback, ExportedSymbol,
    ImportContext, ImportError, ImportManager, ImportResolverCallback, ImportResolverTrait,
    ModuleArchitecture, NativeSignature, NativeType, ResolvedImport, RuntimeError, RuntimeEvent,
    RuntimeResult, SymbolKind,
};

/// A compiled Zyntax runtime ready for execution
///
/// `ZyntaxRuntime` provides a safe interface for:
/// - Compiling Zyntax source code or TypedAST
/// - Calling functions with automatic value conversion
/// - Managing async operations via promises
///
/// # Example
///
/// ```ignore
/// use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};
///
/// let mut runtime = ZyntaxRuntime::new()?;
/// runtime.compile_source("fn add(a: i32, b: i32) -> i32 { a + b }")?;
///
/// let result: i32 = runtime.call("add", &[42.into(), 8.into()])?;
/// assert_eq!(result, 50);
/// ```

/// Hook point for the krio-async state-machine lowering. With the
/// `krio-async-backend` feature on, this runs `krio_adapter`'s
/// orchestrator over every async fn in the module before the legacy
/// `async_support::AsyncCompiler` path executes inside
/// `backend::compile_module`. With the feature off, this is a no-op
/// — leaves the existing async pipeline as-is so behavior is
/// identical to pre-krio builds.
///
/// Run BEFORE compile_module so the backend sees a module whose
/// async functions have already been converted to state machines via
/// the krio path (and the legacy AsyncCompiler will be a no-op since
/// `is_async` is still set; once Phase F lands, the legacy path is
/// gated off when this fires).
/// Phase I.3b: route resumable-effect fns through krio's captures-lift
/// + poll-fn transform. Unlike `apply_krio_async_lowering` (which is
/// cfg-gated and wraps async fns in a Promise-returning entry), this
/// path:
///
///   * Runs unconditionally (no feature gate) — Tier 3 resumable
///     effects need this regardless of how async fns are routed.
///   * Generates a *synchronous* entry wrapper via `generate_sync_entry`
///     so the user-visible signature `(args...) -> T` is preserved.
///     The wrapper drives the poll loop inline until Ready.
///
/// No-op when the module has no fns with resumable handlers.
pub(super) fn apply_krio_effect_lowering(
    module: &mut zyntax_compiler::HirModule,
) -> RuntimeResult<()> {
    crate::krio_lowering::apply_krio_effect_lowering(module)
        .map_err(|e| RuntimeError::Execution(e.0))
}

pub(super) fn apply_krio_async_lowering(
    module: &mut zyntax_compiler::HirModule,
) -> RuntimeResult<()> {
    crate::krio_lowering::apply_krio_async_lowering(module)
        .map_err(|e| RuntimeError::Execution(e.0))
}

/// Rewrite first-class fiber HIR ops (`FiberNew` / `FiberResume` /
/// `FiberYield` / ...) to `Call::Symbol("krio_fiber_*")` so the
/// backend sees only symbol calls. No-op on modules with no fiber
/// ops — runs unconditionally because the rewrite is cheap and the
/// alternative is per-frontend opt-in plumbing.
pub(super) fn apply_krio_fiber_lowering(module: &mut zyntax_compiler::HirModule) {
    zyntax_compiler::fiber_lowering::apply_krio_fiber_lowering(module)
}

// ============================================================================
// Tiered JIT Runtime
// ============================================================================

/// Whether a language may export a name another language holds.
///
/// A symbol name is shared across every language in a runtime. The same
/// language taking its own name again is a reload; a different one
/// taking it would leave the first calling the second's function.
pub(super) fn export_conflicts(holder: Option<&str>, taker: Option<&str>) -> bool {
    match (holder, taker) {
        (Some(holder), Some(taker)) => holder != taker,
        // Nobody said which language, so there is nothing to compare.
        // A host exporting by hand is trusted, as it always was.
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_language_cannot_take_a_symbol_the_first_exported() {
        assert!(
            export_conflicts(Some("python"), Some("typescript")),
            "two languages cannot share one symbol name"
        );
        assert!(
            !export_conflicts(Some("python"), Some("python")),
            "a language reloading its own module is not a collision"
        );
        assert!(
            !export_conflicts(None, Some("python")),
            "a name a host exported by hand is not claimed by any language"
        );
        assert!(
            !export_conflicts(Some("python"), None),
            "a host exporting by hand is trusted, as it always was"
        );
        assert!(!export_conflicts(None, None));
    }
}
