//! Stage tests for Phases A–D. Each `#[test]` exercises ONE stage in
//! isolation against a realistic fixture (see `common::make_async_function_with_one_await`).
//! As Phase E lands, new files (`stages_e_*.rs`) plug into the same
//! pattern: take the previous stage's output, run the next stage,
//! assert on the result.
//!
//! Run with: `cargo test -p krio_adapter --test stages_a_through_d`

mod common;

use krio_adapter::{HirAsyncHooks, HirBlockId, HirCoroCfg, HirLiveness, HirSuspendingFns};
use krio_async::{AsyncHooks, SuspendingFns, SuspensionSite};
use krio_stackless::CoroCfg;
use zyntax_compiler::hir::{HirCallable, HirInstruction, Intrinsic};

use common::{
    live_out_for_entry_only, make_async_function_with_one_await, module_of, AsyncFnFixture,
};

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 — Adapter trait setup (Phase A scaffolding)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_a_cfg_initialises_with_function_blocks() {
    let AsyncFnFixture { mut function, .. } = make_async_function_with_one_await();
    let cfg = HirCoroCfg::new(&mut function);
    assert_eq!(cfg.block_count(), 1, "fixture has one block");
    let ids = cfg.block_ids();
    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], HirBlockId(0));
}

#[test]
fn stage_a_cfg_block_id_round_trip() {
    let AsyncFnFixture { mut function, .. } = make_async_function_with_one_await();
    let entry_hir = function.entry_block;
    let cfg = HirCoroCfg::new(&mut function);
    let bb = cfg.hir_to_block_id(entry_hir).expect("entry must map");
    assert_eq!(cfg.block_id_to_hir(bb), entry_hir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2 — `CoroCfg` mutating ops (Phase B)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_b_split_after_partitions_the_entry_block() {
    let AsyncFnFixture { mut function, .. } = make_async_function_with_one_await();
    let entry_hir = function.entry_block;

    let pre_split_count = function.blocks[&entry_hir].instructions.len();
    assert_eq!(pre_split_count, 4, "fixture entry has 4 instructions");

    let mut cfg = HirCoroCfg::new(&mut function);
    // Split AT the await call (idx 2 — the Intrinsic::Await is the
    // third instruction). After split:
    //   src has [Binary, Call(foo), Call(Await)] (3) + Unreachable
    //   new_bb has [Binary x+r] + original Return terminator
    let new_bb = cfg.split_after(HirBlockId(0), 2);
    let new_hir = cfg.block_id_to_hir(new_bb);

    let src = &cfg.function().blocks[&entry_hir];
    assert_eq!(src.instructions.len(), 3);
    let tail = &cfg.function().blocks[&new_hir];
    assert_eq!(tail.instructions.len(), 1);
    assert!(matches!(
        tail.terminator,
        zyntax_compiler::hir::HirTerminator::Return { .. }
    ));
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 3 — `SuspendingFns` taint analysis (Phase C)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_c_suspending_set_includes_async_fn() {
    let AsyncFnFixture { function, .. } = make_async_function_with_one_await();
    let fn_id = function.id;
    let module = module_of(function);
    let s = HirSuspendingFns::from_module(&module);
    assert!(s.is_suspending(fn_id), "async fn must be in suspending set");
    // The fixture has no host yield primitives; nothing should test
    // positive on is_yield_primitive.
    assert!(!s.is_yield_primitive(fn_id));
}

#[test]
fn stage_c_classify_finds_intrinsic_await_only() {
    let AsyncFnFixture { mut function, .. } = make_async_function_with_one_await();
    let entry_hir = function.entry_block;
    let module = module_of(function);
    let suspending = HirSuspendingFns::from_module(&module);
    let mut function = module
        .functions
        .into_iter()
        .next()
        .expect("module has the function")
        .1;
    let cfg = HirCoroCfg::new(&mut function);
    let hooks = HirAsyncHooks {
        suspending: &suspending,
    };

    // Walk every instruction in the entry block, classify it, and
    // tally how many are DirectYield. The fixture has exactly one
    // Intrinsic::Await call (idx 2).
    let mut direct_yields = 0;
    let block = cfg.function().blocks[&entry_hir].instructions.len();
    for idx in 0..block {
        match hooks.classify(&cfg, HirBlockId(0), idx) {
            Some(SuspensionSite::DirectYield { .. }) => direct_yields += 1,
            Some(_) => panic!("unexpected non-DirectYield site at idx {}", idx),
            None => {}
        }
    }
    assert_eq!(direct_yields, 1, "exactly one Intrinsic::Await in fixture");
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 4 — Liveness mapping (Phase D)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_d_liveness_records_one_site_with_one_local() {
    let AsyncFnFixture {
        mut function,
        live_across,
        ..
    } = make_async_function_with_one_await();
    let live_out = live_out_for_entry_only(&function, live_across);

    let mut cfg = HirCoroCfg::new(&mut function);
    let liveness = HirLiveness::build(&mut cfg, &live_out);

    assert_eq!(liveness.map.at_site.len(), 1, "one await site");
    let (site, locals) = &liveness.map.at_site[0];
    assert_eq!(*site, (HirBlockId(0), 2), "await is at idx 2");
    assert_eq!(locals.len(), 1, "one live SSA value");

    // Round-trip: the LocalId resolves back to live_across
    let local = locals[0];
    assert_eq!(liveness.local_to_hir[&local], live_across);
    assert_eq!(liveness.hir_to_local[&live_across], local);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 5 — krio's transform run end-to-end through the adapter
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_transform_produces_state_machine_layout() {
    let AsyncFnFixture {
        mut function,
        live_across,
        ..
    } = make_async_function_with_one_await();
    let fn_id = function.id;
    let live_out = live_out_for_entry_only(&function, live_across);

    let module = module_of(function.clone());
    let suspending = HirSuspendingFns::from_module(&module);

    let mut cfg = HirCoroCfg::new(&mut function);
    let liveness = HirLiveness::build(&mut cfg, &live_out);
    let hooks = HirAsyncHooks {
        suspending: &suspending,
    };

    let layout =
        krio_async::transform_to_state_machine(&mut cfg, fn_id, &suspending, &hooks, &liveness.map)
            .expect("transform must succeed for canonical fixture");

    // Original entry + one resume entry = 2 states.
    assert_eq!(layout.resume_entries.len(), 2);
    assert_eq!(layout.resume_entries[0], HirBlockId(0));

    // One yield block, one save site, one load site.
    assert_eq!(layout.yield_blocks.len(), 1);
    assert_eq!(layout.yield_saves.len(), 1);
    assert_eq!(layout.resume_loads.len(), 1);

    // Save's LocalId round-trips to live_across.
    let saved_local = layout.yield_saves[0].1[0].1;
    assert_eq!(liveness.local_to_hir[&saved_local], live_across);
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-cutting: every Phase A–D run should leave the function still
// contains the original Intrinsic::Await call before any save/load
// emission step (Phase E1) replaces it.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn stage_pre_e1_function_still_has_intrinsic_await() {
    let AsyncFnFixture {
        mut function,
        live_across,
        ..
    } = make_async_function_with_one_await();
    let fn_id = function.id;
    let live_out = live_out_for_entry_only(&function, live_across);

    let module = module_of(function.clone());
    let suspending = HirSuspendingFns::from_module(&module);
    let mut cfg = HirCoroCfg::new(&mut function);
    let liveness = HirLiveness::build(&mut cfg, &live_out);
    let hooks = HirAsyncHooks {
        suspending: &suspending,
    };
    let _ =
        krio_async::transform_to_state_machine(&mut cfg, fn_id, &suspending, &hooks, &liveness.map)
            .unwrap();

    // After the transform, the Intrinsic::Await call is still in some
    // block (krio splits around it but doesn't replace it). Phase E1's
    // save/load emission is what replaces / adorns it. This test
    // pins down that pre-condition.
    let mut found_await = false;
    for block in cfg.function().blocks.values() {
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
    assert!(
        found_await,
        "transform should not erase the Intrinsic::Await; that's E1's job"
    );
}
