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
    /// (param SSA HirId, slot index) for each function parameter,
    /// allocated AFTER krio's captures-lift slots so they don't
    /// collide. The runtime ABI entry function uses these to know
    /// where to store args in the state-machine struct; the poll
    /// function's param-load prologue uses them to know where to
    /// load from.
    pub param_slots: Vec<(HirId, u32)>,
    /// Slot reserved for the state-id u32 (used by the dispatcher).
    /// Allocated above all captures-lift slots and param slots.
    pub state_slot: u32,
    /// Total number of 8-byte slots the state-machine struct needs.
    /// Used by the entry function to size the malloc. Computed as
    /// `max(captures_slot, state_slot, param_slots) + 1`.
    pub num_slots: u32,
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
///
/// `suspending` MUST contain `function.id` — caller computes it once
/// per module so that `lower_async_module`-style iteration that
/// temporarily takes the function out of the module doesn't break
/// the call-graph view. See [`HirSuspendingFns::from_module`].
pub fn lower_async_function(
    function: &mut HirFunction,
    suspending: &HirSuspendingFns,
    frame_ptr: HirId,
    _state_slot_hint: u32,
    live_out_per_block: &HashMap<HirId, HashSet<HirId>>,
) -> Result<LowerResult, LowerError> {
    let fn_id = function.id;

    // Snapshot param SSA HirIds before any mutation. The SSA builder
    // creates separate HirIds for body-facing param values (with
    // `HirValueKind::Parameter(idx)`) — these are what the body
    // references. They differ from `function.signature.params[i].id`
    // (matching the convention in async_support.rs:285-293). Look up
    // the SSA HirId per param-index; fall back to signature.params[i].id
    // if no Parameter(idx) value exists (rare — happens for fns
    // built entirely synthetically).
    let num_params = function.signature.params.len();
    let param_hir_ids: Vec<HirId> = (0..num_params)
        .map(|idx| {
            function
                .values
                .iter()
                .find(|(_, v)| matches!(&v.kind, zyntax_compiler::hir::HirValueKind::Parameter(i) if *i as usize == idx))
                .map(|(id, _)| *id)
                .unwrap_or_else(|| function.signature.params[idx].id)
        })
        .collect();

    let mut cfg = HirCoroCfg::new(function);
    let liveness = HirLiveness::build(&mut cfg, live_out_per_block);
    let hooks = HirAsyncHooks { suspending };

    let layout = krio_async::transform_to_state_machine(
        &mut cfg, fn_id, suspending, &hooks, &liveness.map,
    )
    .map_err(|_| LowerError::Transform)?;

    // Compute the highest slot index krio's captures-lift used. krio's
    // allocator (krio-async/src/lib.rs:538) starts at 0 and increments
    // per unique LocalId. We allocate state_slot + param_slots STRICTLY
    // ABOVE this max so they can't collide.
    let max_capture_slot: i64 = layout
        .yield_saves
        .iter()
        .flat_map(|(_, entries)| entries.iter().map(|(slot, _)| *slot as i64))
        .chain(
            layout
                .resume_loads
                .iter()
                .flat_map(|(_, entries)| entries.iter().map(|(slot, _)| *slot as i64)),
        )
        .max()
        .unwrap_or(-1);
    let state_slot: u32 = (max_capture_slot + 1) as u32;
    let param_slots: Vec<(HirId, u32)> = param_hir_ids
        .iter()
        .enumerate()
        .map(|(i, hir)| (*hir, state_slot + 1 + i as u32))
        .collect();
    let after_params_slot: u32 = state_slot + 1 + param_slots.len() as u32;

    let rewrites = emit::emit_save_load(&mut cfg, &layout, &liveness, frame_ptr);
    emit::emit_dispatcher(&mut cfg, &layout, frame_ptr, state_slot);

    // Phase F.2: replace `Intrinsic::Await` calls with the
    // poll-the-inner-promise state machine. Allocates two slots per
    // await (promise + result) starting after param_slots.
    let cfg_function = cfg.function_mut();
    let after_awaits_slot = crate::abi_emit::lower_await_calls(
        cfg_function,
        &layout,
        frame_ptr,
        state_slot,
        after_params_slot,
    );
    let num_slots: u32 = after_awaits_slot;

    // F.2 follow-up: the await lowering creates new resume blocks
    // that may now be predecessors of phi blocks (e.g. loop headers
    // when the loop body contains an await). The original phi.incoming
    // entries reference the OLD loop body block which no longer
    // branches to the phi. Repair the predecessors so Cranelift sees
    // valid SSA.
    crate::abi_emit::repair_phi_predecessors(cfg_function);

    Ok(LowerResult {
        layout,
        liveness,
        rewrites,
        param_slots,
        state_slot,
        num_slots,
    })
}

/// Convenience wrapper that computes `suspending` from the module
/// internally. Use when the function being lowered is still in the
/// module — i.e. NOT in `lower_async_module`'s swap-remove loop.
pub fn lower_async_function_in_module(
    function: &mut HirFunction,
    module: &HirModule,
    frame_ptr: HirId,
    state_slot: u32,
    live_out_per_block: &HashMap<HirId, HashSet<HirId>>,
) -> Result<LowerResult, LowerError> {
    let suspending = HirSuspendingFns::from_module(module);
    lower_async_function(function, &suspending, frame_ptr, state_slot, live_out_per_block)
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
    // Compute the suspending-fn taint set ONCE up front, before any
    // swap_remove churn. Krio's `transform_to_state_machine` early-
    // exits when `is_suspending(fn_id)` returns false, so the call-
    // graph view must be intact while each function is lowered —
    // which it isn't if we recompute it after pulling the function
    // out of the module.
    let suspending = HirSuspendingFns::from_module(module);

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
        let result =
            lower_async_function(&mut function, &suspending, frame, state_slot, &live_out)?;
        module.functions.insert(fn_id, function);
        results.insert(fn_id, result);
    }
    Ok(results)
}
