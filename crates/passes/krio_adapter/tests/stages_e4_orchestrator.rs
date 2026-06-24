//! Stage tests for Phase E4 — the orchestrator that runs Phases A–E2
//! end-to-end against an async function.
//!
//! Verifies that calling `lower_async_function` once on the canonical
//! fixture produces a fully-rewritten HIR that:
//!   * has a Switch dispatcher at entry,
//!   * has saves before the yielding return,
//!   * has loads at the resume entry, and
//!   * has post-load uses correctly rewritten.
//!
//! Run with: `cargo test -p krio_adapter --test stages_e4_orchestrator`

mod common;

use std::collections::HashSet;

use krio_adapter::orchestrator;
use zyntax_compiler::hir::{
    HirCallable, HirId, HirInstruction, HirTerminator, HirType, HirValue, HirValueKind, Intrinsic,
};

use common::{
    live_out_for_entry_only, make_async_function_with_one_await, module_of, AsyncFnFixture,
};

#[test]
fn e4_full_pipeline_produces_dispatcher_save_load() {
    let AsyncFnFixture {
        mut function,
        live_across,
        await_result,
    } = make_async_function_with_one_await();

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
            .expect("orchestrator must succeed");

    // ── Dispatcher present at function entry ──
    let entry = &function.blocks[&function.entry_block];
    assert!(
        matches!(entry.terminator, HirTerminator::Switch { .. }),
        "function entry must be the Switch dispatcher; got {:?}",
        entry.terminator
    );
    assert!(
        matches!(entry.instructions[0], HirInstruction::AsyncLoadSlot { .. }),
        "dispatcher's first inst must be the state-id load"
    );

    // ── Save in the yield block ──
    let yield_id = result.layout.yield_blocks[0].0;
    let yield_hir = function
        .blocks
        .keys()
        .nth(yield_id.0 as usize)
        .copied()
        .unwrap();
    let yield_block = &function.blocks[&yield_hir];
    let save_count = yield_block
        .instructions
        .iter()
        .filter(|i| matches!(i, HirInstruction::AsyncSaveSlot { .. }))
        .count();
    // After F.2 with re-entry handling, the yield block is a thin
    // re-entry check (LoadSlot + Eq + CondBranch). All saves moved
    // to first_call_block (captures-lift + promise persist) and
    // ready_block (result + state). Yield block itself has 0 saves.
    assert_eq!(
        save_count, 0,
        "yield block now contains only the re-entry check"
    );
    // Total saves across the function should be 4: captures-lift +
    // promise (in first_call_block), result + state (in ready_block).
    let total_saves = function
        .blocks
        .values()
        .flat_map(|b| b.instructions.iter())
        .filter(|i| matches!(i, HirInstruction::AsyncSaveSlot { .. }))
        .count();
    // Total saves = 1 captures-lift + 1 promise persist (first_call_block)
    //              + 1 result + 1 promise-clear + 1 state-bump (ready_block) = 5
    assert_eq!(
        total_saves, 5,
        "total saves: 1 captures + 1 promise + 1 result + 1 promise-clear + 1 state"
    );

    // ── Load in the resume entry, downstream uses rewritten ──
    let resume_id = result.layout.resume_entries[1];
    let resume_hir = function
        .blocks
        .keys()
        .nth(resume_id.0 as usize)
        .copied()
        .unwrap();
    let resume_block = &function.blocks[&resume_hir];
    assert!(matches!(
        resume_block.instructions[0],
        HirInstruction::AsyncLoadSlot { .. }
    ));

    let fresh = result.rewrites[&live_across];
    let mut found_rewritten_use = false;
    for block in function.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary { left, right, .. } = inst {
                if *left == fresh || *right == fresh {
                    found_rewritten_use = true;
                }
                assert_ne!(*left, live_across);
                assert_ne!(*right, live_across);
            }
        }
    }
    assert!(found_rewritten_use, "rewritten use should appear somewhere");

    // Phase F.2: `Intrinsic::Await` is now REPLACED by the
    // poll-the-inner-promise state machine emitted by `abi_emit::
    // lower_await_calls`. Verify it's been erased — finding one would
    // be a regression in the await lowering.
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
    assert!(
        !found_await,
        "Intrinsic::Await should have been replaced by lower_await_calls"
    );

    let _ = await_result;
}

#[test]
fn e4_orchestrator_no_op_for_non_async_function() {
    // Synthetic sync fn: build a one-block module with a non-async fn.
    use indexmap::IndexMap;
    use zyntax_compiler::hir::{
        HirBlock, HirFunction, HirFunctionSignature, HirModule, HirTerminator,
    };
    use zyntax_typed_ast::InternedString;

    let sig = HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::Void],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: true,
    };
    let mut function = HirFunction::new(InternedString::new_global("sync"), sig);
    function.is_external = false;
    let bb_id = HirId::new();
    let bb = HirBlock {
        id: bb_id,
        label: None,
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return { values: vec![] },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    let mut blocks = IndexMap::new();
    blocks.insert(bb_id, bb);
    function.blocks = blocks;
    function.entry_block = bb_id;

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

    let mut module = HirModule::new(InternedString::new_global("m"));
    module.functions.insert(function.id, function.clone());

    let original_entry = function.entry_block;
    let mut function_owned = function;
    let result = orchestrator::lower_async_function_in_module(
        &mut function_owned,
        &module,
        frame,
        16,
        &std::collections::HashMap::new(),
    )
    .expect("orchestrator must succeed");

    // Non-async fn → empty layout from krio (transform short-circuits
    // when is_suspending is false). No dispatcher, no save/load.
    assert!(result.layout.resume_entries.is_empty());
    assert!(result.layout.yield_saves.is_empty());
    assert!(result.layout.resume_loads.is_empty());
    assert_eq!(
        function_owned.entry_block, original_entry,
        "non-async fn entry block must be untouched"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Phase I.3: generate_sync_entry — sync entry wrapper that drives the
// poll loop inline and returns the value with the original signature
// (instead of a *Promise). Companion to generate_promise_entry.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn generate_sync_entry_produces_poll_loop_with_original_signature() {
    use indexmap::IndexMap;
    use krio_adapter::abi_emit::generate_sync_entry;
    use zyntax_compiler::hir::{HirFunctionSignature, HirParam, ParamAttributes};
    use zyntax_typed_ast::InternedString;

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: InternedString::new_global("p0"),
            ty: HirType::I32,
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::I64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    let poll_fn_id = HirId::new();
    let num_slots = 3;
    let param_slots = vec![(sig.params[0].id, 1u32)];
    let state_slot = 0;
    // SM layout: slot 0 = state, slot 1 = param, slot 2 = refcount.
    let refcount_slot = 2;

    let entry = generate_sync_entry(
        InternedString::new_global("my_effect_fn"),
        &sig,
        poll_fn_id,
        num_slots,
        &param_slots,
        state_slot,
        refcount_slot,
    );

    // Signature preserved: (i32) -> i64
    assert_eq!(entry.signature.params.len(), 1);
    assert!(matches!(entry.signature.params[0].ty, HirType::I32));
    assert_eq!(entry.signature.returns.len(), 1);
    assert!(matches!(entry.signature.returns[0], HirType::I64));

    // is_async cleared — the entry is the synchronous public face.
    assert!(!entry.signature.is_async);

    // Three blocks: entry, poll_block, return_block.
    assert_eq!(entry.blocks.len(), 3);

    // Find the poll block by label.
    let poll_block = entry
        .blocks
        .values()
        .find(|b| {
            b.label
                .map(|l| l.resolve_global().as_deref() == Some("sync_entry_poll"))
                .unwrap_or(false)
        })
        .expect("sync_entry_poll block exists");

    // poll_block contains exactly: IndirectCall + Binary(Eq).
    assert!(matches!(
        poll_block.instructions[0],
        zyntax_compiler::hir::HirInstruction::IndirectCall { .. }
    ));
    assert!(matches!(
        poll_block.instructions[1],
        zyntax_compiler::hir::HirInstruction::Binary { .. }
    ));

    // Terminator: CondBranch back to self on Pending, forward on Ready.
    assert!(matches!(
        poll_block.terminator,
        zyntax_compiler::hir::HirTerminator::CondBranch {
            true_target,
            ..
        } if true_target == poll_block.id
    ));

    let _ = IndexMap::<HirId, u32>::new(); // suppress unused-import
}
