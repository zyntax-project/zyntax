//! Phase E4: orchestrator — runs the full krio adoption pipeline
//! against a single async `HirFunction`.
//!
//! Composes Phases A–E2 into one entry point. Hosts call this once
//! per async fn during compile_module (typically in place of the
//! legacy `async_support::compile_async_function`):
//!
//! ```ignore
//! krio_adapter::orchestrator::lower_async_function(
//!     function,
//!     module,
//!     frame_ptr,                  // future-struct `self` SSA value
//!     STATE_SLOT,                 // fixed slot reserved for state-id
//!     &live_out_per_block,        // from existing analysis::LivenessAnalysis
//! )?;
//! ```
//!
//! After the call, `function` has:
//!   * Save instructions before each yielding Return
//!   * Load instructions at each resume entry
//!   * Downstream uses of saved values rewritten to loaded values
//!   * A new entry block dispatching on state_id via Switch
//!
//! From here the existing Cranelift backend lowers the HIR normally —
//! `AsyncSaveSlot`/`AsyncLoadSlot` already have lowerings (Phase E3).
//!
//! This crate intentionally does **not** modify `compile_module`
//! directly. The integration step (gating the AsyncCompiler call
//! behind a feature, adding `krio_adapter` as a dep of
//! zyntax_compiler) requires breaking the krio_adapter→
//! zyntax_compiler dependency cycle and is best handled at the
//! caller layer (e.g. `zyntax_embed::ZyntaxRuntime::compile_module`
//! invoking this orchestrator before handing the module to the
//! backend).

use std::collections::{HashMap, HashSet};

use krio_async::StateMachineLayout;
use zyntax_compiler::hir::{HirFunction, HirId, HirModule};

use crate::{
    emit, HirAsyncHooks, HirBlockId, HirCoroCfg, HirFnId, HirLiveness, HirLocalId, HirSuspendingFns,
};

/// Result of orchestrating the krio pipeline for a single async fn.
/// Tests + downstream code use the layout / liveness for assertions
/// or further lowering.
pub struct LowerResult {
    pub layout: StateMachineLayout<HirBlockId, HirLocalId, HirFnId>,
    pub liveness: HirLiveness,
    /// Saved-SSA-HirId → freshly-loaded SSA HirId, returned by
    /// `emit_save_load`.
    pub rewrites: HashMap<HirId, HirId>,
}

/// Errors the orchestrator may surface.
#[derive(Debug)]
pub enum LowerError {
    /// `krio_async::transform_to_state_machine` rejected the layout
    /// (currently krio's `TransformError` enum has no concrete
    /// variants — Phase 3 v2 covers everything we care about — but
    /// it's #[non_exhaustive] so we keep the error path open).
    Transform,
}

impl core::fmt::Display for LowerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LowerError::Transform => f.write_str("krio_async::transform_to_state_machine refused"),
        }
    }
}

impl std::error::Error for LowerError {}

/// Run Phases A–E2 for a single async function.
///
/// `frame_ptr` is the SSA `HirId` of the future-struct pointer the
/// poll fn receives (typically `function.signature.params[0].id`).
/// `state_slot` is the slot index reserved for the state-id (the
/// caller must keep this distinct from any captures-lift slot — krio
/// allocates from 0 upward, so reserving slot 0 for state and passing
/// `state_slot = N+1` past krio's max is the safest convention; tests
/// just use `0` because the canonical fixture has no overlap).
///
/// `live_out_per_block` is the host's existing per-block liveness
/// data (produced by `zyntax_compiler::analysis::LivenessAnalysis`).
pub fn lower_async_function(
    function: &mut HirFunction,
    module: &HirModule,
    frame_ptr: HirId,
    state_slot: u32,
    live_out_per_block: &HashMap<HirId, HashSet<HirId>>,
) -> Result<LowerResult, LowerError> {
    let fn_id = function.id;
    let suspending = HirSuspendingFns::from_module(module);

    let mut cfg = HirCoroCfg::new(function);
    let liveness = HirLiveness::build(&mut cfg, live_out_per_block);
    let hooks = HirAsyncHooks {
        suspending: &suspending,
    };

    let layout = krio_async::transform_to_state_machine(
        &mut cfg,
        fn_id,
        &suspending,
        &hooks,
        &liveness.map,
    )
    .map_err(|_| LowerError::Transform)?;

    let rewrites = emit::emit_save_load(&mut cfg, &layout, &liveness, frame_ptr);
    emit::emit_dispatcher(&mut cfg, &layout, frame_ptr, state_slot);

    Ok(LowerResult {
        layout,
        liveness,
        rewrites,
    })
}

/// Lower every `is_async` function in `module`. Convenience wrapper
/// for callers that want the bulk path.
///
/// `frame_ptr_for` resolves the per-function frame pointer (typically
/// `module.functions[&id].signature.params[0].id` for async fns
/// lowered with `(self: *mut Future, ...)` signatures). Callers that
/// don't yet have a future-struct convention can pass a closure that
/// allocates a fresh SSA value of pointer type for each function.
pub fn lower_async_module<F>(
    module: &mut HirModule,
    state_slot: u32,
    mut frame_ptr_for: F,
    live_out_per_block: &HashMap<HirId, HashMap<HirId, HashSet<HirId>>>,
) -> Result<HashMap<HirId, LowerResult>, LowerError>
where
    F: FnMut(&HirFunction) -> HirId,
{
    let async_fn_ids: Vec<HirId> = module
        .functions
        .values()
        .filter(|f| f.signature.is_async)
        .map(|f| f.id)
        .collect();

    let mut results = HashMap::new();
    for fn_id in async_fn_ids {
        // Pull the function out, lower, put back. Lets us pass
        // &module immutably while mutating one function.
        let mut function = module
            .functions
            .swap_remove(&fn_id)
            .expect("async fn id from module");
        let frame = frame_ptr_for(&function);
        let live_out = live_out_per_block
            .get(&fn_id)
            .cloned()
            .unwrap_or_default();
        let result = lower_async_function(&mut function, module, frame, state_slot, &live_out)?;
        module.functions.insert(fn_id, function);
        results.insert(fn_id, result);
    }
    Ok(results)
}
