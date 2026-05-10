//! Stage tests for Phases E1 (save/load emission) and E2 (dispatcher
//! emission). Each test composes outputs from earlier stages — same
//! pattern as `stages_a_through_d.rs` — and asserts on the resulting
//! HIR shape.
//!
//! Run with: `cargo test -p krio_adapter --test stages_e1_e2_emit`

mod common;

use std::collections::HashSet;

use krio_adapter::{emit, HirAsyncHooks, HirCoroCfg, HirLiveness, HirSuspendingFns};
use zyntax_compiler::hir::{
    HirCallable, HirId, HirInstruction, HirTerminator, HirType, HirValue, HirValueKind, Intrinsic,
};

use common::{
    live_out_for_entry_only, make_async_function_with_one_await, module_of, AsyncFnFixture,
};

/// Test driver that runs Phases A–E2 against the canonical fixture
/// and returns intermediate artifacts for assertions. Mirrors the
/// production orchestrator (Phase E4) but exposes each stage's
/// output for tests.
struct StagedRun {
    function: zyntax_compiler::hir::HirFunction,
    layout: krio_async::StateMachineLayout<
        krio_adapter::HirBlockId,
        krio_adapter::HirLocalId,
        krio_adapter::HirFnId,
    >,
    liveness: HirLiveness,
    /// Original SSA HirIds → freshly-loaded SSA HirIds, returned by
    /// `emit_save_load`.
    rewrites: std::collections::HashMap<HirId, HirId>,
    live_across: HirId,
    await_result: HirId,
}

fn run_through_e2(state_slot: u32, captures_slot_base: u32, emit_dispatcher: bool) -> StagedRun {
    let _ = captures_slot_base; // krio's slot allocator picks slots; reserved for future use
    let AsyncFnFixture {
        mut function,
        live_across,
        await_result,
    } = make_async_function_with_one_await();
    let fn_id = function.id;
    let live_out = live_out_for_entry_only(&function, live_across);

    // Plant a "frame pointer" SSA value so save/load have something
    // to reference. In the real pipeline this is the poll fn's first
    // param; for the test harness we just register an instruction
    // value in function.values.
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
    let suspending = HirSuspendingFns::from_module(&module);

    let mut cfg = HirCoroCfg::new(&mut function);
    let liveness = HirLiveness::build(&mut cfg, &live_out);
    let hooks = HirAsyncHooks {
        suspending: &suspending,
    };
    let layout =
        krio_async::transform_to_state_machine(&mut cfg, fn_id, &suspending, &hooks, &liveness.map)
            .expect("transform must succeed");

    let rewrites = emit::emit_save_load(&mut cfg, &layout, &liveness, frame);
    if emit_dispatcher {
        emit::emit_dispatcher(&mut cfg, &layout, frame, state_slot);
    }
    drop(cfg);

    StagedRun {
        function,
        layout,
        liveness,
        rewrites,
        live_across,
        await_result,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase E1 — save/load emission
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e1_save_inserted_at_yield_block() {
    let StagedRun {
        function,
        layout,
        liveness: _,
        ..
    } = run_through_e2(0, 1, false);

    // The yield block (the original entry split at the await) should
    // have at least one AsyncSaveSlot before its terminator.
    let yield_block_id = layout.yield_blocks[0].0;
    let yield_hir = function
        .blocks
        .keys()
        .nth(yield_block_id.0 as usize)
        .copied()
        .expect("yield block in function");
    let yield_block = &function.blocks[&yield_hir];
    let saves: Vec<&HirInstruction> = yield_block
        .instructions
        .iter()
        .filter(|inst| matches!(inst, HirInstruction::AsyncSaveSlot { .. }))
        .collect();
    assert!(
        !saves.is_empty(),
        "expected at least one AsyncSaveSlot in yield block, got {} instructions",
        yield_block.instructions.len()
    );
}

#[test]
fn e1_load_inserted_at_resume_entry() {
    let StagedRun {
        function, layout, ..
    } = run_through_e2(0, 1, false);

    // Resume entry (state 1) should have its first instruction be an
    // AsyncLoadSlot.
    let resume_id = layout.resume_entries[1];
    let resume_hir = function
        .blocks
        .keys()
        .nth(resume_id.0 as usize)
        .copied()
        .expect("resume entry in function");
    let resume_block = &function.blocks[&resume_hir];
    assert!(
        !resume_block.instructions.is_empty(),
        "resume block should have at least the AsyncLoadSlot"
    );
    assert!(
        matches!(
            resume_block.instructions[0],
            HirInstruction::AsyncLoadSlot { .. }
        ),
        "first instruction of resume block must be AsyncLoadSlot, got {:?}",
        resume_block.instructions[0]
    );
}

#[test]
fn e1_post_load_uses_are_rewritten() {
    let StagedRun {
        function,
        rewrites,
        live_across,
        ..
    } = run_through_e2(0, 1, false);

    // The live_across SSA id should have been rewritten to a fresh
    // post-load id in any block reachable from the resume entry.
    let fresh_id = rewrites
        .get(&live_across)
        .copied()
        .expect("live_across should have a rewrite mapping");
    assert_ne!(fresh_id, live_across);

    // Find the post-await `Add(live_across, await_result)` instruction.
    // After E1, its `left` operand should be the FRESH id, not
    // `live_across`. (await_result is local to the resume side; no
    // rewrite needed.)
    let mut found_rewritten = false;
    for block in function.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary { left, .. } = inst {
                if *left == fresh_id {
                    found_rewritten = true;
                }
                assert_ne!(
                    *left, live_across,
                    "post-load uses of live_across must be rewritten"
                );
            }
        }
    }
    assert!(
        found_rewritten,
        "expected to find a Binary instruction whose left operand was rewritten"
    );
}

#[test]
fn e1_save_value_round_trips_via_liveness() {
    let StagedRun {
        layout,
        liveness,
        live_across,
        ..
    } = run_through_e2(0, 1, false);

    // The save's LocalId in the layout maps back to live_across via
    // liveness.local_to_hir — that's what made E1 know which SSA id
    // to spill.
    let saved_local = layout.yield_saves[0].1[0].1;
    assert_eq!(liveness.local_to_hir[&saved_local], live_across);
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase E2 — dispatcher prologue
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2_dispatcher_replaces_entry_with_state_switch() {
    let StagedRun { function, .. } = run_through_e2(0, 1, true);

    // After dispatcher emission, function.entry_block points at a new
    // block whose terminator is a Switch on a freshly-loaded state
    // value, with one case per resume_entry.
    let entry_hir = function.entry_block;
    let entry = &function.blocks[&entry_hir];

    assert!(
        matches!(entry.terminator, HirTerminator::Switch { .. }),
        "dispatcher entry must be a Switch terminator, got {:?}",
        entry.terminator
    );
    // First instruction should be the state-id load.
    assert!(
        !entry.instructions.is_empty(),
        "dispatcher entry must have at least the state-load instruction"
    );
    assert!(
        matches!(entry.instructions[0], HirInstruction::AsyncLoadSlot { .. }),
        "dispatcher entry's first inst must load state id"
    );
}

#[test]
fn e2_dispatcher_has_one_case_per_resume_state() {
    let StagedRun {
        function, layout, ..
    } = run_through_e2(0, 1, true);

    let entry_hir = function.entry_block;
    let entry = &function.blocks[&entry_hir];
    if let HirTerminator::Switch { cases, .. } = &entry.terminator {
        assert_eq!(
            cases.len(),
            layout.resume_entries.len(),
            "one switch case per resume_entry"
        );
    } else {
        panic!("expected Switch terminator");
    }
}

#[test]
fn e2_dispatcher_default_targets_resume_entry_zero() {
    let StagedRun {
        function, layout, ..
    } = run_through_e2(0, 1, true);

    let entry_hir = function.entry_block;
    let entry = &function.blocks[&entry_hir];
    if let HirTerminator::Switch { default, .. } = &entry.terminator {
        let expected = function
            .blocks
            .keys()
            .nth(layout.resume_entries[0].0 as usize)
            .copied()
            .expect("resume_entries[0] in function");
        assert_eq!(*default, expected);
    } else {
        panic!("expected Switch terminator");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-cutting: AwaitResult is NOT rewritten (it's defined inside the
// resume side, not a captured value).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e1_await_result_not_in_rewrite_map() {
    let StagedRun {
        rewrites,
        await_result,
        ..
    } = run_through_e2(0, 1, false);
    assert!(
        !rewrites.contains_key(&await_result),
        "await_result is defined post-suspension; must not be in the captures-rewrite map"
    );
}
