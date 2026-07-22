//! Post-`compile_to_hir` krio passes: async-fn → poll-fn SM and
//! resumable-effect fn → poll-fn SM. Both are also wired through
//! the native `ZyntaxRuntime::compile_typed_program` path (see
//! `runtime.rs`'s thin delegates), but they live here so the
//! wasm-target `zyntax_wasm::run_impl` can call them too without
//! pulling in the `feature = "native"` runtime module.
//!
//! Moved out of `runtime.rs` so the wasm path can reuse them.
//! Behavior is identical on both targets.

use std::collections::{HashMap, HashSet};

use zyntax_compiler::analysis::AnalysisRunner;
use zyntax_compiler::hir::HirId;

/// Local error type. The native runtime wraps these into
/// `RuntimeError::Execution`; the wasm shim renders them as
/// compile-error strings.
#[derive(Debug)]
pub struct KrioLoweringError(pub String);

impl std::fmt::Display for KrioLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for KrioLoweringError {}

/// Phase I.3b: route resumable-effect fns through krio's
/// captures-lift + poll-fn transform. Wraps each one in a SYNC
/// entry (returns T, not `*Promise<T>`) so the user-visible
/// signature is preserved.
///
/// No-op when the module has no fns with resumable handlers.
pub fn apply_krio_effect_lowering(
    module: &mut zyntax_compiler::HirModule,
) -> Result<(), KrioLoweringError> {
    // Predicate: does `f` perform an effect that has a resumable
    // handler (so it needs the captures-lift state-machine transform)?
    let has_resumable_effect = |f: &zyntax_compiler::hir::HirFunction| -> bool {
        f.signature.effects.iter().any(|effect_name| {
            if let Some(effect) = module.effects.values().find(|e| e.name == *effect_name) {
                module
                    .handlers
                    .values()
                    .filter(|h| h.effect_id == effect.id)
                    .any(|h| h.implementations.iter().any(|i| i.is_resumable))
            } else {
                false
            }
        })
    };

    // Composition guardrail: a `fiber def` that performs a resumable
    // effect would get BOTH the state-machine transform (here) and
    // fiber lowering, whose save/restore models disagree on any value
    // live across both a perform and a yield. Reject rather than
    // silently miscompile. (Lifted once the handler stack is made
    // fiber-aware — see the fiber×effect×async plan, Phase 4/5.)
    if let Some(bad) = module
        .functions
        .values()
        .find(|f| f.signature.is_fiber && has_resumable_effect(f))
    {
        return Err(KrioLoweringError(format!(
            "fiber function `{}` performs a resumable algebraic effect — this composition is \
             not yet supported (the fiber stack switch and the effect state machine have \
             incompatible suspension models). Perform the effect from a non-fiber function, \
             or use a non-resumable handler.",
            bad.name.resolve_global().unwrap_or_default()
        )));
    }

    // Gate on a raw `PerformEffect` still present in the body — NOT just
    // the signature's effect list. `apply_krio_async_lowering` runs first
    // and rewrites an `async` performing fn into a promise entry (whose
    // `is_async` flag is cleared but whose signature keeps the `@effect`
    // list) plus a poll fn (whose performs are already lowered to the
    // dispatch machinery). Both would match a signature-only filter and be
    // re-lowered here — a double transform that clobbers the async SM with
    // an empty-yield layout. A raw `PerformEffect` in the body is present
    // only on genuinely-sync performing fns the async pass left untouched.
    let body_has_perform = |f: &zyntax_compiler::hir::HirFunction| -> bool {
        f.blocks.values().any(|b| {
            b.instructions.iter().any(|i| {
                matches!(
                    i,
                    zyntax_compiler::hir::HirInstruction::PerformEffect { .. }
                )
            })
        })
    };
    let resumable_fn_ids: Vec<HirId> = module
        .functions
        .values()
        .filter(|f| {
            !f.signature.is_async
                && !f.signature.effects.is_empty()
                && body_has_perform(f)
                && has_resumable_effect(f)
        })
        .map(|f| f.id)
        .collect();

    if resumable_fn_ids.is_empty() {
        return Ok(());
    }

    let mut live_out: HashMap<HirId, HashMap<HirId, HashSet<HirId>>> = HashMap::new();
    let mut analyzer = AnalysisRunner::new(module.clone());
    if let Ok(analysis) = analyzer.run_all() {
        for (fn_id, fn_analysis) in &analysis.functions {
            live_out.insert(*fn_id, fn_analysis.liveness.live_out.clone());
        }
    }

    let suspending = krio_adapter::HirSuspendingFns::from_module(module);

    // `(effect_id, op_name) → (handler-fn id, is_async)`. The async bit lets
    // the perform-site emitter drive the handler's promise instead of using
    // its return value directly.
    let mut handler_resolution: HashMap<(HirId, zyntax_typed_ast::InternedString), (HirId, bool)> =
        HashMap::new();
    for handler in module.handlers.values() {
        for impl_ in &handler.implementations {
            if !impl_.is_resumable {
                continue;
            }
            let mangled = zyntax_compiler::mangle_handler_op_name(handler.name, impl_.op_name);
            if let Some(handler_fn_id) = module
                .functions
                .iter()
                .find(|(_, f)| f.name.resolve_global().as_deref() == Some(mangled.as_str()))
                .map(|(id, _)| *id)
            {
                handler_resolution.insert(
                    (handler.effect_id, impl_.op_name),
                    (handler_fn_id, impl_.is_async),
                );
            }
        }
    }

    // Op-index resolution: `(effect_id, op_name) → position of the op
    // in the effect's operation list`. Must match the op-table
    // ordering built in `LoweringContext::build_op_table` so
    // `with H { }` regional dispatch resolves to the right slot. Keyed
    // by effect_id so it's independent of which handler is active.
    let mut op_index_resolution: HashMap<(HirId, zyntax_typed_ast::InternedString), u64> =
        HashMap::new();
    for effect in module.effects.values() {
        for (idx, op) in effect.operations.iter().enumerate() {
            op_index_resolution.insert((effect.id, op.name), idx as u64);
        }
    }

    for fn_id in resumable_fn_ids {
        let mut function = match module.functions.swap_remove(&fn_id) {
            Some(f) => f,
            None => continue,
        };
        let original_signature = function.signature.clone();
        let original_name = function.name;

        let frame_ptr = function
            .signature
            .params
            .first()
            .map(|p| p.id)
            .unwrap_or_else(HirId::new);
        let live_out_for_fn = live_out.get(&fn_id).cloned().unwrap_or_default();

        let lower_result = krio_adapter::orchestrator::lower_async_function(
            &mut function,
            &suspending,
            frame_ptr,
            0,
            &live_out_for_fn,
        )
        .map_err(|e| KrioLoweringError(format!("krio-effect lowering failed: {e}")))?;

        let _ = krio_adapter::abi_emit::reshape_to_poll_abi(
            &mut function,
            &lower_result.layout,
            &lower_result.param_slots,
        );

        let new_poll_id = HirId::new();
        function.id = new_poll_id;

        let post_reshape_frame_ptr = function
            .signature
            .params
            .first()
            .map(|p| p.id)
            .unwrap_or(frame_ptr);
        krio_adapter::abi_emit::upgrade_resume_struct_at_perform_sites(
            &mut function,
            &handler_resolution,
            &op_index_resolution,
            new_poll_id,
            post_reshape_frame_ptr,
            lower_result.resume_scratch_slot,
            lower_result.refcount_slot,
        );

        let mut entry_fn = krio_adapter::abi_emit::generate_sync_entry(
            original_name,
            &original_signature,
            new_poll_id,
            lower_result.num_slots,
            &lower_result.param_slots,
            lower_result.state_slot,
            lower_result.refcount_slot,
        );
        entry_fn.id = fn_id;

        log::debug!(
            "[krio-effect] {}: state_slot={} num_slots={}",
            original_name.resolve_global().unwrap_or_default(),
            lower_result.state_slot,
            lower_result.num_slots,
        );

        module.functions.insert(function.id, function);
        module.functions.insert(entry_fn.id, entry_fn);
    }

    Ok(())
}

/// Phase F.x: route async fns through krio's captures-lift +
/// poll-fn transform. Each async fn becomes a Promise-returning
/// entry wrapper + a separate poll fn.
///
/// Gated on `feature = "krio-async-backend"` — without it, this is
/// a no-op and the legacy `transform_async_function` in the SSA
/// lowering owns the async transform.
pub fn apply_krio_async_lowering(
    _module: &mut zyntax_compiler::HirModule,
) -> Result<(), KrioLoweringError> {
    #[cfg(feature = "krio-async-backend")]
    {
        let mut live_out: HashMap<HirId, HashMap<HirId, HashSet<HirId>>> = HashMap::new();
        let mut analyzer = AnalysisRunner::new(_module.clone());
        match analyzer.run_all() {
            Ok(analysis) => {
                for (fn_id, fn_analysis) in &analysis.functions {
                    live_out.insert(*fn_id, fn_analysis.liveness.live_out.clone());
                }
            }
            Err(e) => {
                log::warn!(
                    "[krio-async] AnalysisRunner failed; krio path will run with empty liveness: {:?}",
                    e
                );
            }
        }

        let suspending = krio_adapter::HirSuspendingFns::from_module(_module);

        // An `async def` may ALSO perform a resumable algebraic effect.
        // `lower_async_function` (below) already turns the `PerformEffect`
        // into a suspension/resume state, but the handler-dispatch +
        // `Resume<T>` construction lives in a separate pass
        // (`upgrade_resume_struct_at_perform_sites`) that the non-async
        // resumable path runs. The effect path skips async fns, so we run
        // that upgrade here too — otherwise the perform site parks with no
        // way to reach its handler. Build the same resolution maps it needs.
        let mut handler_resolution: HashMap<
            (HirId, zyntax_typed_ast::InternedString),
            (HirId, bool),
        > = HashMap::new();
        for handler in _module.handlers.values() {
            for impl_ in &handler.implementations {
                if !impl_.is_resumable {
                    continue;
                }
                let mangled = zyntax_compiler::mangle_handler_op_name(handler.name, impl_.op_name);
                if let Some(handler_fn_id) = _module
                    .functions
                    .iter()
                    .find(|(_, f)| f.name.resolve_global().as_deref() == Some(mangled.as_str()))
                    .map(|(id, _)| *id)
                {
                    handler_resolution.insert(
                        (handler.effect_id, impl_.op_name),
                        (handler_fn_id, impl_.is_async),
                    );
                }
            }
        }
        let mut op_index_resolution: HashMap<(HirId, zyntax_typed_ast::InternedString), u64> =
            HashMap::new();
        for effect in _module.effects.values() {
            for (idx, op) in effect.operations.iter().enumerate() {
                op_index_resolution.insert((effect.id, op.name), idx as u64);
            }
        }
        // Effect names whose handler set includes a resumable impl — used
        // to decide per-fn whether the resume-struct upgrade must run.
        let resumable_effect_names: HashSet<zyntax_typed_ast::InternedString> = _module
            .effects
            .values()
            .filter(|effect| {
                _module
                    .handlers
                    .values()
                    .filter(|h| h.effect_id == effect.id)
                    .any(|h| h.implementations.iter().any(|i| i.is_resumable))
            })
            .map(|effect| effect.name)
            .collect();

        let async_fn_ids: Vec<HirId> = _module
            .functions
            .values()
            .filter(|f| f.signature.is_async)
            .map(|f| f.id)
            .collect();

        let mut lowered_count = 0_usize;
        for fn_id in async_fn_ids {
            let mut function = _module
                .functions
                .swap_remove(&fn_id)
                .expect("async fn id was just enumerated from this module");

            let original_signature = function.signature.clone();
            let original_name = function.name;

            let frame_ptr = function
                .signature
                .params
                .first()
                .map(|p| p.id)
                .unwrap_or_else(HirId::new);

            let live_out_for_fn = live_out.get(&fn_id).cloned().unwrap_or_default();

            let lower_result = krio_adapter::orchestrator::lower_async_function(
                &mut function,
                &suspending,
                frame_ptr,
                0,
                &live_out_for_fn,
            )
            .map_err(|e| KrioLoweringError(format!("krio-async lowering failed: {e}")))?;

            let poll_fn_name = krio_adapter::abi_emit::reshape_to_poll_abi(
                &mut function,
                &lower_result.layout,
                &lower_result.param_slots,
            );

            let new_poll_id = HirId::new();
            let original_id = function.id;
            function.id = new_poll_id;

            // Phase I.4 — fix CreateClosure references emitted by
            // krio Phase I.2 (which captured `function.id` AT EMIT
            // TIME, when it still equalled the original fn_id). After
            // the swap, those refs point at the entry function — not
            // the poll fn. Rewrite them so the closure pointer feeds
            // `register_future`'s `poll_fn_ptr` slot correctly.
            //
            // Same pass also fixes `__zyntax_register_future` Calls'
            // sm_ptr arg (args[1]): Phase I.2 captured the
            // pre-reshape `frame_ptr` (a dummy `HirId::new()` for
            // fns with no params), never in Cranelift's value_map.
            // The post-reshape sm_ptr is the freshly-minted
            // Parameter(0) value.
            let post_reshape_sm_ptr_id = function
                .values
                .iter()
                .find(|(_, v)| matches!(v.kind, zyntax_compiler::hir::HirValueKind::Parameter(0)))
                .map(|(id, _)| *id);
            for block in function.blocks.values_mut() {
                for inst in &mut block.instructions {
                    if let zyntax_compiler::hir::HirInstruction::CreateClosure {
                        function: target,
                        ..
                    } = inst
                    {
                        if *target == original_id {
                            *target = new_poll_id;
                        }
                    }
                    if let zyntax_compiler::hir::HirInstruction::Call {
                        callee: zyntax_compiler::hir::HirCallable::Symbol(name),
                        args,
                        ..
                    } = inst
                    {
                        if name == "__zyntax_register_future" && args.len() >= 2 {
                            if let Some(sm_ptr) = post_reshape_sm_ptr_id {
                                args[1] = sm_ptr;
                            }
                        }
                    }
                }
            }

            // If this async fn performs a resumable effect, build the
            // handler-dispatch + `Resume<T>` machinery at each perform site
            // (the async analogue of the non-async resumable path). Without
            // this the perform site parks with no route to its handler.
            if original_signature
                .effects
                .iter()
                .any(|e| resumable_effect_names.contains(e))
            {
                let post_reshape_frame_ptr = function
                    .signature
                    .params
                    .first()
                    .map(|p| p.id)
                    .unwrap_or(frame_ptr);
                krio_adapter::abi_emit::upgrade_resume_struct_at_perform_sites(
                    &mut function,
                    &handler_resolution,
                    &op_index_resolution,
                    new_poll_id,
                    post_reshape_frame_ptr,
                    lower_result.resume_scratch_slot,
                    lower_result.refcount_slot,
                );
            }

            let mut entry_fn = krio_adapter::abi_emit::generate_promise_entry(
                original_name,
                &original_signature,
                new_poll_id,
                lower_result.num_slots,
                &lower_result.param_slots,
                lower_result.state_slot,
            );
            entry_fn.id = fn_id;

            log::debug!(
                "[krio-async] {}: state_slot={} num_slots={} param_slots={} captures={}",
                original_name.resolve_global().unwrap_or_default(),
                lower_result.state_slot,
                lower_result.num_slots,
                lower_result.param_slots.len(),
                lower_result
                    .layout
                    .yield_saves
                    .iter()
                    .map(|(_, e)| e.len())
                    .sum::<usize>(),
            );

            let _ = poll_fn_name;
            _module.functions.insert(function.id, function);
            _module.functions.insert(entry_fn.id, entry_fn);
            lowered_count += 1;
        }

        log::debug!(
            "[krio-async] lowered {} async fns via krio adapter",
            lowered_count
        );
    }
    Ok(())
}
