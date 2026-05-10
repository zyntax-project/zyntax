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
    // Verifies the orchestrator's bulk path now picks up effect-
    // annotated functions, not just `is_async = true` ones.
    let EffectfulFnFixture { function, .. } = make_effectful_function_with_one_perform();
    let mut module = module_of(function);

    // Plant a frame pointer per function (the closure must mint one).
    let mut frame_minter = |_func: &zyntax_compiler::hir::HirFunction| -> HirId {
        // Simple constant-id minting — real pipeline uses param[0].
        HirId::new()
    };

    // No real liveness — empty map. krio will produce an empty layout
    // for the function, but the filter must still admit it.
    let live_out_per_block = std::collections::HashMap::new();
    let results = orchestrator::lower_async_module(
        &mut module,
        16,
        &mut frame_minter,
        &live_out_per_block,
    )
    .expect("lower_async_module must succeed for effectful fn");

    assert_eq!(
        results.len(),
        1,
        "the effect-annotated fn should have been lowered (1 entry in results)"
    );
}
