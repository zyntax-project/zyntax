//! Stage tests for Phase H, M4 — `lower_perform_effect_calls`.
//!
//! Verifies the end-to-end orchestrator pipeline against a function
//! shaped like `@effect(State) fn run() { let x = ...; let r = perform get(); x + r }`:
//!
//!   * the function is correctly seeded as suspending (M3)
//!   * krio's transform produces yield/resume pairs at the perform site
//!   * `lower_perform_effect_calls` replaces the PerformEffect site
//!     with the handler-dispatch state machine (renumber result,
//!     append ready_block, prepend AsyncLoadSlot in resume entry)
//!   * the original `Intrinsic::Await` lowering is unaffected (it
//!     short-circuits when the yield-block contains a PerformEffect
//!     instead of an Await)
//!
//! Run with: `cargo test -p krio_adapter --test stages_h4_perform_lowering`

mod common;

use std::collections::HashSet;

use krio_adapter::orchestrator;
use zyntax_compiler::hir::{
    HirCallable, HirId, HirInstruction, HirTerminator, HirType, HirValue, HirValueKind, Intrinsic,
};

use common::{
    live_out_for_entry_only, make_effectful_function_with_one_perform, module_of,
    EffectfulFnFixture,
};

#[test]
fn h4_orchestrator_lowers_perform_effect_to_state_machine() {
    let EffectfulFnFixture {
        mut function,
        live_across,
        perform_result,
    } = make_effectful_function_with_one_perform();

    // Plant a frame pointer SSA value (real pipeline pulls this from
    // the poll fn's first param).
    let frame = HirId::new();
    function.values.insert(
        frame,
        HirValue {
            id: frame,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            kind: HirValueKind::Instruction,
            uses: HashSet::new(),
            span: None,
        },
    );

    let module = module_of(function.clone());
    let live_out = live_out_for_entry_only(&function, live_across);

    let result =
        orchestrator::lower_async_function_in_module(&mut function, &module, frame, 16, &live_out)
            .expect("orchestrator must succeed for an @effect-annotated fn");

    // ── Dispatcher present at function entry ──
    let entry = &function.blocks[&function.entry_block];
    assert!(
        matches!(entry.terminator, HirTerminator::Switch { .. }),
        "entry must be the Switch dispatcher"
    );

    // ── PerformEffect with renumbered result is preserved ──
    // M4 keeps the PerformEffect inst (the backend's existing handler
    // dispatches it) but renumbers its `result` to a fresh HirId, so
    // the original `perform_result` HirId is reserved for the
    // AsyncLoadSlot in resume_entry.
    let mut found_perform_with_renumbered_result = false;
    let mut found_perform_with_original_result = false;
    for block in function.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::PerformEffect { result, .. } = inst {
                if *result == Some(perform_result) {
                    found_perform_with_original_result = true;
                } else {
                    found_perform_with_renumbered_result = true;
                }
            }
        }
    }
    assert!(
        found_perform_with_renumbered_result,
        "PerformEffect should have a renumbered result HirId"
    );
    assert!(
        !found_perform_with_original_result,
        "the original `perform_result` HirId must NOT be defined by PerformEffect — \
         that HirId is reserved for the AsyncLoadSlot in resume_entry"
    );

    // ── AsyncLoadSlot defining the original perform_result in resume ──
    let mut found_load_for_perform = false;
    for block in function.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::AsyncLoadSlot { result, .. } = inst {
                if *result == perform_result {
                    found_load_for_perform = true;
                }
            }
        }
    }
    assert!(
        found_load_for_perform,
        "resume_entry must prepend AsyncLoadSlot defining perform_result"
    );

    // ── ready_block (post-perform): saves result + state, returns 0 ──
    // Identifying it: a block with label "perform_ready".
    let ready_block = function
        .blocks
        .values()
        .find(|b| {
            b.label
                .map(|l| l.resolve_global().as_deref() == Some("perform_ready"))
                .unwrap_or(false)
        })
        .expect("perform_ready block should exist");
    let save_count = ready_block
        .instructions
        .iter()
        .filter(|i| matches!(i, HirInstruction::AsyncSaveSlot { .. }))
        .count();
    assert_eq!(
        save_count, 2,
        "ready_block has 2 saves: result + state-bump"
    );
    assert!(
        matches!(ready_block.terminator, HirTerminator::Return { .. }),
        "ready_block returns to the runtime"
    );

    // ── No leftover Intrinsic::Await (sanity — fixture has none) ──
    let mut found_await = false;
    for block in function.blocks.values() {
        for inst in &block.instructions {
            if matches!(
                inst,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Await),
                    ..
                }
            ) {
                found_await = true;
            }
        }
    }
    assert!(!found_await, "fixture has no Await; none should appear");

    // num_slots accounts for: state(slot 0) + 0 params + 1 capture (slot 1)
    // + 1 perform-result (slot 2) = max=2, so num_slots = 3.
    assert!(
        result.num_slots >= 3,
        "num_slots should account for perform's result slot; got {}",
        result.num_slots
    );
}

#[test]
fn h4_lower_async_module_includes_effectful_functions() {
    // Verifies the orchestrator's bulk path picks up effect-annotated
    // functions whose effects have a resumable handler. Per the
    // Phase-H followup, Tier 1 (non-resumable) effects skip krio in
    // favor of the Cranelift backend's direct PerformEffect dispatch;
    // Tier 3 (resumable) effects need krio's captures-lift. The
    // bulk filter looks up handler resumability in `module.handlers`.
    use indexmap::IndexMap;
    use zyntax_compiler::hir::{HirEffect, HirEffectHandler, HirEffectHandlerImpl, HirTerminator};
    use zyntax_typed_ast::InternedString;

    let EffectfulFnFixture { function, .. } = make_effectful_function_with_one_perform();
    let mut module = module_of(function);

    // Add a State effect declaration to the module so the filter can
    // resolve `effects = [State]` on the function to a HirId.
    let state_effect = HirEffect {
        id: HirId::new(),
        name: InternedString::new_global("State"),
        type_params: vec![],
        operations: vec![],
    };
    let state_effect_id = state_effect.id;
    module.effects.insert(state_effect.id, state_effect);

    // Add a RESUMABLE handler for State. The single-impl handler's
    // `is_resumable = true` marker is what `function_has_resumable_effect`
    // uses to admit the fn to krio.
    let handler_entry_id = HirId::new();
    let mut handler_blocks: IndexMap<HirId, zyntax_compiler::hir::HirBlock> = IndexMap::new();
    handler_blocks.insert(
        handler_entry_id,
        zyntax_compiler::hir::HirBlock {
            id: handler_entry_id,
            label: None,
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Unreachable,
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        },
    );
    let handler = HirEffectHandler {
        id: HirId::new(),
        name: InternedString::new_global("StateHandler"),
        effect_id: state_effect_id,
        type_params: vec![],
        state_fields: vec![],
        implementations: vec![HirEffectHandlerImpl {
            op_name: InternedString::new_global("get"),
            type_params: vec![],
            params: vec![],
            return_type: zyntax_compiler::hir::HirType::I32,
            entry_block: handler_entry_id,
            blocks: handler_blocks,
            is_resumable: true,
        }],
    };
    module.handlers.insert(handler.id, handler);

    // Plant a frame pointer per function (the closure must mint one).
    let mut frame_minter = |_func: &zyntax_compiler::hir::HirFunction| -> HirId { HirId::new() };

    // No real liveness — empty map. krio will produce an empty layout
    // for the function, but the filter must still admit it.
    let live_out_per_block = std::collections::HashMap::new();
    let results =
        orchestrator::lower_async_module(&mut module, 16, &mut frame_minter, &live_out_per_block)
            .expect("lower_async_module must succeed for effectful fn");

    assert_eq!(
        results.len(),
        1,
        "the @effect(State) fn with a resumable State handler should be lowered (1 entry); got {}",
        results.len()
    );
}

#[test]
fn h4_lower_async_module_skips_non_resumable_effects() {
    // The complement: an @effect-annotated fn whose effect's handler
    // is NOT resumable (Tier 1) skips krio entirely. The Cranelift
    // backend's direct PerformEffect dispatch handles it.
    use indexmap::IndexMap;
    use zyntax_compiler::hir::{HirEffect, HirEffectHandler, HirEffectHandlerImpl, HirTerminator};
    use zyntax_typed_ast::InternedString;

    let EffectfulFnFixture { function, .. } = make_effectful_function_with_one_perform();
    let mut module = module_of(function);

    let state_effect = HirEffect {
        id: HirId::new(),
        name: InternedString::new_global("State"),
        type_params: vec![],
        operations: vec![],
    };
    let state_effect_id = state_effect.id;
    module.effects.insert(state_effect.id, state_effect);

    // Handler with `is_resumable = false` — the Tier 1 case.
    let handler_entry_id = HirId::new();
    let mut handler_blocks: IndexMap<HirId, zyntax_compiler::hir::HirBlock> = IndexMap::new();
    handler_blocks.insert(
        handler_entry_id,
        zyntax_compiler::hir::HirBlock {
            id: handler_entry_id,
            label: None,
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Unreachable,
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        },
    );
    let handler = HirEffectHandler {
        id: HirId::new(),
        name: InternedString::new_global("StateHandler"),
        effect_id: state_effect_id,
        type_params: vec![],
        state_fields: vec![],
        implementations: vec![HirEffectHandlerImpl {
            op_name: InternedString::new_global("get"),
            type_params: vec![],
            params: vec![],
            return_type: zyntax_compiler::hir::HirType::I32,
            entry_block: handler_entry_id,
            blocks: handler_blocks,
            is_resumable: false,
        }],
    };
    module.handlers.insert(handler.id, handler);

    let mut frame_minter = |_func: &zyntax_compiler::hir::HirFunction| -> HirId { HirId::new() };

    let live_out_per_block = std::collections::HashMap::new();
    let results =
        orchestrator::lower_async_module(&mut module, 16, &mut frame_minter, &live_out_per_block)
            .expect("lower_async_module must succeed (no-op)");

    assert_eq!(
        results.len(),
        0,
        "the @effect(State) fn with a non-resumable handler should NOT \
         be lowered through krio (Tier 1 direct-dispatch path); got {}",
        results.len()
    );
}
