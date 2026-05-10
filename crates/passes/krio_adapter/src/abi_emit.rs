//! ABI scaffolding emitter — wraps a krio-transformed HIR function in
//! the runtime-expected calling convention.
//!
//! After `orchestrator::lower_async_function` runs, the function has
//! its captures-lift transform but still has its original signature
//! `(p1, p2, ...) -> T`. The ZynML runtime calls async functions
//! through a Promise-returning entry function whose poll fn has
//! signature `fn(*mut u8) -> i64` (0 = pending, positive = ready
//! value encoded as i64). This module bridges between the two:
//!
//! * [`reshape_to_poll_abi`] mutates the krio-transformed function
//!   in place into the poll ABI: drops original params, adds a
//!   single `state_machine: *mut u8` param, prepends a load-prologue
//!   that reads original-param values from their assigned slots,
//!   rewrites yield-block returns to `0_i64` (pending), wraps
//!   non-yield returns with a cast to i64 (ready value).
//!
//! * [`generate_promise_entry`] returns a NEW function with the
//!   original async name and signature `(p1, p2, ...) -> *Promise<T>`.
//!   It mallocs the state machine, stores params at their slots,
//!   mallocs the 16-byte Promise struct, and returns it.

use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use krio_async::StateMachineLayout;
use zyntax_compiler::hir::{
    BinaryOp, CastOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature,
    HirFunctionType, HirId, HirInstruction, HirParam, HirStructType, HirTerminator, HirType,
    HirValue, HirValueKind, Intrinsic, ParamAttributes,
};
use zyntax_typed_ast::InternedString;

use crate::{HirBlockId, HirFnId, HirLocalId};

/// Per-await slot pair allocated by [`lower_await_calls`].
#[derive(Debug, Clone, Copy)]
pub struct AwaitSlots {
    /// Slot holding the inner Promise pointer (for cross-poll persistence).
    pub promise_slot: u32,
    /// Slot holding the i64-encoded ready value.
    pub result_slot: u32,
}

/// Replace each `Intrinsic::Await` call site with the runtime's
/// poll-the-inner-promise state machine. After this pass:
///
///   * The yield block's body has the original Promise-producing
///     instructions, then the poll prologue (load promise.poll_fn,
///     IndirectCall it with promise.state_machine, branch on result).
///   * Two new blocks per await: `pending` (returns 0) and `ready`
///     (saves result + bumps state to next, returns 0).
///   * The corresponding resume block gains an `AsyncLoadSlot` at the
///     top so the original `r = await foo()` binding still resolves.
///
/// `state_slot` is the slot the dispatcher reads/writes for the
/// current state-id. `frame` is the current SSA value used in
/// AsyncSaveSlot/AsyncLoadSlot ops; `reshape_to_poll_abi` will
/// rewrite all such ops later to point at the new state_machine
/// param, so any HirId here is fine as long as it's the same one
/// krio used.
///
/// `start_slot` is the first free slot after captures+state+params;
/// each await consumes 2 slots. Returns the next free slot
/// (= start_slot + 2 * num_awaits).
/// After F.2's CFG restructuring (await-call lowering + reshape +
/// dispatcher), some phi nodes have stale predecessor references.
/// Specifically: when a loop body contains an await, krio splits at
/// the suspension and the post-suspension code lives in a new resume
/// block that branches back to the loop header. The loop header's
/// phi nodes still reference the OLD predecessor (the original loop
/// body block), but that block no longer branches to the loop header
/// (it now CondBranches to await_pending/ready).
///
/// This repair pass:
/// 1. Computes actual predecessors per block from the current CFG.
/// 2. For each phi, identifies entries whose predecessor no longer
///    targets this block, and swaps them positionally with new
///    predecessors that aren't in the phi yet.
///
/// Without this, Cranelift IR verification fails:
///   "jump block4: got 0, expected 3"
/// (because the new predecessor's Branch terminator doesn't supply
/// phi args matching the block's parameter count).
pub fn repair_phi_predecessors(function: &mut HirFunction) {
    use HirTerminator as T;
    fn successors(t: &T) -> Vec<HirId> {
        match t {
            T::Return { .. } | T::Unreachable => vec![],
            T::Branch { target } => vec![*target],
            T::CondBranch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            T::Switch { default, cases, .. } => {
                let mut s = vec![*default];
                s.extend(cases.iter().map(|(_, t)| *t));
                s
            }
            T::Invoke { normal, unwind, .. } => vec![*normal, *unwind],
            T::PatternMatch {
                patterns, default, ..
            } => {
                let mut s: Vec<HirId> = patterns.iter().map(|p| p.target).collect();
                if let Some(d) = default {
                    s.push(*d);
                }
                s
            }
        }
    }

    // Build actual predecessor map.
    let mut actual_preds: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for (id, block) in &function.blocks {
        for succ in successors(&block.terminator) {
            actual_preds.entry(succ).or_default().push(*id);
        }
    }

    let block_ids: Vec<HirId> = function.blocks.keys().copied().collect();
    for block_id in block_ids {
        let actual = actual_preds.get(&block_id).cloned().unwrap_or_default();
        let actual_set: HashSet<HirId> = actual.iter().copied().collect();
        let block = match function.blocks.get_mut(&block_id) {
            Some(b) => b,
            None => continue,
        };
        if block.phis.is_empty() {
            continue;
        }
        // Determine stale vs missing preds for THIS block (uniform
        // across all its phis — they all share the same predecessor
        // list).
        let phi_preds: HashSet<HirId> =
            block.phis[0].incoming.iter().map(|(_, p)| *p).collect();
        let stale: Vec<HirId> = phi_preds
            .iter()
            .copied()
            .filter(|p| !actual_set.contains(p))
            .collect();
        let new: Vec<HirId> = actual.iter().copied().filter(|p| !phi_preds.contains(p)).collect();
        if stale.len() != new.len() {
            // Heuristic gives up — log and continue. This means the
            // CFG transform did something more complex than 1:1
            // pred replacement.
            log::debug!(
                "[phi-repair] block {:?}: {} stale preds vs {} new preds — skipping",
                block_id,
                stale.len(),
                new.len()
            );
            continue;
        }
        // Positional swap: stale[i] → new[i] in every phi's incoming.
        let swap_map: HashMap<HirId, HirId> = stale.iter().zip(new.iter()).map(|(s, n)| (*s, *n)).collect();
        for phi in &mut block.phis {
            for (_, pred) in &mut phi.incoming {
                if let Some(&new_pred) = swap_map.get(pred) {
                    *pred = new_pred;
                }
            }
        }
    }
}

pub fn lower_await_calls(
    function: &mut HirFunction,
    layout: &StateMachineLayout<HirBlockId, HirLocalId, HirFnId>,
    frame: HirId,
    state_slot: u32,
    start_slot: u32,
) -> u32 {
    let mut next_slot = start_slot;

    // Snapshot the block-iteration order from the time krio ran.
    // krio's HirBlockId values index this list. We need the snapshot
    // because we'll be mutating `function.blocks` (adding pending +
    // ready blocks per await) and don't want our seq → HirId mapping
    // to shift mid-loop.
    let block_seq_to_hir: Vec<HirId> = function.blocks.keys().copied().collect();

    let resolve_seq = |bb: HirBlockId| -> Option<HirId> {
        block_seq_to_hir.get(bb.0 as usize).copied()
    };

    // Iterate yield blocks. `layout.yield_blocks[i] = (block_id, next_state)`.
    // The corresponding resume entry is `layout.resume_entries[next_state]`.
    let yield_blocks_snapshot: Vec<(HirBlockId, u32)> = layout.yield_blocks.clone();
    for (yield_seq, next_state) in yield_blocks_snapshot {
        let yield_hir = match resolve_seq(yield_seq) {
            Some(h) => h,
            None => continue,
        };
        let resume_seq = match layout.resume_entries.get(next_state as usize) {
            Some(r) => *r,
            None => continue,
        };
        let resume_hir = match resolve_seq(resume_seq) {
            Some(h) => h,
            None => continue,
        };

        // Find the Call(Intrinsic::Await) instruction in the yield block.
        let yield_block = function.blocks.get(&yield_hir).expect("yield block exists");
        let mut await_idx: Option<usize> = None;
        let mut await_promise: Option<HirId> = None;
        let mut await_result: Option<HirId> = None;
        for (i, inst) in yield_block.instructions.iter().enumerate() {
            if let HirInstruction::Call {
                callee: HirCallable::Intrinsic(Intrinsic::Await),
                args,
                result,
                ..
            } = inst
            {
                await_idx = Some(i);
                await_promise = args.first().copied();
                await_result = *result;
                break;
            }
        }
        let (await_idx, promise_ptr, result_id) = match (await_idx, await_promise, await_result) {
            (Some(i), Some(p), Some(r)) => (i, p, r),
            // No actual await call: krio still considered this a
            // suspension site (e.g. a yield expression). Skip — the
            // simple captures-lift transform from F.1 handles it.
            _ => continue,
        };
        // The SSA builder sometimes leaves Intrinsic::Await's result
        // typed as Void (the future's inner type isn't propagated
        // through the call). Fall back to I64 in that case — the slot
        // holds a 64-bit value anyway, and downstream Binary ops in
        // the resume block typically operate at I64.
        let result_ty = function
            .values
            .get(&result_id)
            .map(|v| v.ty.clone())
            .unwrap_or(HirType::I64);
        let result_ty = if matches!(result_ty, HirType::Void) {
            HirType::I64
        } else {
            result_ty
        };
        // Also update function.values so downstream code reading the
        // type sees the same fallback (e.g., the Cranelift backend's
        // type_cache, Binary op type checks).
        if let Some(v) = function.values.get_mut(&result_id) {
            v.ty = result_ty.clone();
        }

        // Allocate two slots for this await.
        let promise_slot = next_slot;
        next_slot += 1;
        let result_slot = next_slot;
        next_slot += 1;

        // === Re-entrant await design ===
        // The naive approach (always run Call(inner) on every entry to
        // the yield block) breaks chains: the outer fn polls the inner,
        // gets 0 (Pending sentinel), returns 0 to runtime, runtime polls
        // outer again — outer state is still 0 so it re-enters yield and
        // creates a FRESH inner promise, throwing away progress.
        //
        // Fix: stash the inner promise in `promise_slot` on first entry;
        // on subsequent entries to this yield block, skip the Call and
        // re-poll the existing promise. The entry function zero-inits
        // every slot, so `promise_slot == 0` is a reliable "not called
        // yet" sentinel.
        //
        // CFG after this transform:
        //
        //   yield_block:
        //     existing = AsyncLoadSlot(promise_slot)
        //     is_first = (existing == 0)
        //     branch is_first, first_call_block, poll_block
        //
        //   first_call_block:
        //     [original pre-await code: Call(inner_fn, args) → promise]
        //     [captures-lift saves from emit_save_load]
        //     AsyncSaveSlot promise_slot ← promise
        //     branch poll_block
        //
        //   poll_block:
        //     current = AsyncLoadSlot(promise_slot)
        //     nested_sm = Load(current[0])
        //     poll_fn = Load(current[8])
        //     poll_result = IndirectCall poll_fn(nested_sm)
        //     is_pending = (poll_result == 0)
        //     branch is_pending, pending_block, ready_block
        //
        //   pending_block: return 0
        //   ready_block: save result, save next_state, return 0

        // Pending and ready blocks (constants for them are simple).
        let pending_zero = mint_const_i64(&mut function.values, 0);
        let pending_block_id = HirId::new();
        let pending_block = HirBlock {
            id: pending_block_id,
            label: Some(InternedString::new_global("await_pending")),
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Return {
                values: vec![pending_zero],
            },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };

        let next_state_const = mint_const_i64(&mut function.values, next_state as i64);
        let ready_zero = mint_const_i64(&mut function.values, 0);
        let ready_block_id = HirId::new();
        let ready_block = HirBlock {
            id: ready_block_id,
            label: Some(InternedString::new_global("await_ready")),
            phis: vec![],
            instructions: vec![],
            terminator: HirTerminator::Return {
                values: vec![ready_zero],
            },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };

        // Build poll_block (loads current promise from slot every time
        // → handles both first-call and re-entry uniformly).
        let poll_block_id = HirId::new();
        let current_promise_id = mint_value(
            &mut function.values,
            HirType::Ptr(Box::new(HirType::U8)),
            HirValueKind::Instruction,
        );
        let nested_sm_id = mint_value(
            &mut function.values,
            HirType::Ptr(Box::new(HirType::U8)),
            HirValueKind::Instruction,
        );
        let const_8_pb = mint_const_i64(&mut function.values, 8);
        let poll_fn_slot_id = mint_value(
            &mut function.values,
            HirType::Ptr(Box::new(HirType::U8)),
            HirValueKind::Instruction,
        );
        let poll_fn_ty = HirType::Function(Box::new(HirFunctionType {
            params: vec![HirType::Ptr(Box::new(HirType::U8))],
            returns: vec![HirType::I64],
            lifetime_params: vec![],
            is_variadic: false,
        }));
        let poll_fn_ptr_id = mint_value(
            &mut function.values,
            poll_fn_ty.clone(),
            HirValueKind::Instruction,
        );
        let poll_result_id =
            mint_value(&mut function.values, HirType::I64, HirValueKind::Instruction);
        let zero_const_pb = mint_const_i64(&mut function.values, 0);
        let is_pending_id =
            mint_value(&mut function.values, HirType::Bool, HirValueKind::Instruction);
        let poll_block = HirBlock {
            id: poll_block_id,
            label: Some(InternedString::new_global("await_poll")),
            phis: vec![],
            instructions: vec![
                HirInstruction::AsyncLoadSlot {
                    result: current_promise_id,
                    ty: HirType::Ptr(Box::new(HirType::U8)),
                    frame,
                    slot: promise_slot,
                },
                HirInstruction::Load {
                    result: nested_sm_id,
                    ty: HirType::Ptr(Box::new(HirType::U8)),
                    ptr: current_promise_id,
                    align: 8,
                    volatile: false,
                },
                HirInstruction::Binary {
                    result: poll_fn_slot_id,
                    op: BinaryOp::Add,
                    ty: HirType::I64,
                    left: current_promise_id,
                    right: const_8_pb,
                },
                HirInstruction::Load {
                    result: poll_fn_ptr_id,
                    ty: poll_fn_ty,
                    ptr: poll_fn_slot_id,
                    align: 8,
                    volatile: false,
                },
                HirInstruction::IndirectCall {
                    result: Some(poll_result_id),
                    func_ptr: poll_fn_ptr_id,
                    args: vec![nested_sm_id],
                    return_ty: HirType::I64,
                },
                HirInstruction::Binary {
                    result: is_pending_id,
                    op: BinaryOp::Eq,
                    ty: HirType::I64,
                    left: poll_result_id,
                    right: zero_const_pb,
                },
            ],
            terminator: HirTerminator::CondBranch {
                condition: is_pending_id,
                true_target: pending_block_id,
                false_target: ready_block_id,
            },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };

        // Now wire ready_block's body. Saving poll_result_id (defined
        // in poll_block which dominates ready_block) is fine SSA.
        // We finalize ready_block's instructions after poll_block exists
        // so we can reference poll_result_id.
        // (Block was created above with empty instructions — set them now.)
        // We'll insert into function.blocks BEFORE filling ready_block,
        // since ready_block is created standalone here.
        // Actually we still own `ready_block` as a local var (haven't
        // inserted yet), so just push instructions.
        let mut ready_block = ready_block;
        ready_block.instructions = vec![
            HirInstruction::AsyncSaveSlot {
                frame,
                slot: result_slot,
                value: poll_result_id,
            },
            HirInstruction::AsyncSaveSlot {
                frame,
                slot: state_slot,
                value: next_state_const,
            },
        ];

        // Build first_call_block: contains the original pre-await
        // instructions (the Call to the inner fn) + captures-lift saves
        // + the AsyncSaveSlot to stash the new promise. Then branches
        // to poll_block.
        let first_call_block_id = HirId::new();
        let yield_block = function
            .blocks
            .get_mut(&yield_hir)
            .expect("yield block exists");
        let kept_pre: Vec<HirInstruction> =
            yield_block.instructions[..await_idx].to_vec();
        let kept_post: Vec<HirInstruction> = yield_block
            .instructions
            .iter()
            .skip(await_idx + 1)
            .cloned()
            .collect();

        let mut first_call_insts = kept_pre;
        first_call_insts.extend(kept_post);
        first_call_insts.push(HirInstruction::AsyncSaveSlot {
            frame,
            slot: promise_slot,
            value: promise_ptr,
        });
        let first_call_block = HirBlock {
            id: first_call_block_id,
            label: Some(InternedString::new_global("await_first_call")),
            phis: vec![],
            instructions: first_call_insts,
            terminator: HirTerminator::Branch {
                target: poll_block_id,
            },
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        };

        // Now the yield_block becomes a thin re-entry check.
        let existing_promise_id = mint_value(
            &mut function.values,
            HirType::Ptr(Box::new(HirType::U8)),
            HirValueKind::Instruction,
        );
        let zero_check_const = mint_const_i64(&mut function.values, 0);
        let is_first_call_id =
            mint_value(&mut function.values, HirType::Bool, HirValueKind::Instruction);
        yield_block.instructions = vec![
            HirInstruction::AsyncLoadSlot {
                result: existing_promise_id,
                ty: HirType::Ptr(Box::new(HirType::U8)),
                frame,
                slot: promise_slot,
            },
            HirInstruction::Binary {
                result: is_first_call_id,
                op: BinaryOp::Eq,
                ty: HirType::I64,
                left: existing_promise_id,
                right: zero_check_const,
            },
        ];
        yield_block.terminator = HirTerminator::CondBranch {
            condition: is_first_call_id,
            true_target: first_call_block_id,
            false_target: poll_block_id,
        };

        // Insert all new blocks into the function.
        function.blocks.insert(first_call_block_id, first_call_block);
        function.blocks.insert(poll_block_id, poll_block);
        function.blocks.insert(pending_block_id, pending_block);
        function.blocks.insert(ready_block_id, ready_block);

        // At the resume block, prepend an AsyncLoadSlot that defines
        // `result_id` (the original `r` HirId from the await call).
        // The original Call(Intrinsic::Await) defining r is gone now,
        // so r is no longer multiply-defined.
        let load_result_inst = HirInstruction::AsyncLoadSlot {
            result: result_id,
            ty: result_ty,
            frame,
            slot: result_slot,
        };
        let resume_block = function
            .blocks
            .get_mut(&resume_hir)
            .expect("resume block exists");
        let mut new_insts = vec![load_result_inst];
        new_insts.extend(resume_block.instructions.drain(..));
        resume_block.instructions = new_insts;
    }

    next_slot
}

/// Reshape a krio-transformed function into the runtime poll ABI.
///
/// Pre: `function` was mutated by `lower_async_function` — it has the
/// captures-lift transform (AsyncSaveSlot/AsyncLoadSlot ops + Switch
/// dispatcher at entry), and its signature is the original
/// `(p1, p2, ...) -> T` shape.
///
/// Post: signature is `(state_machine: *mut u8) -> i64`. A new entry
/// block is prepended that loads each original param from its slot
/// and branches to the krio dispatcher. All `frame: X` references in
/// AsyncSaveSlot/AsyncLoadSlot ops point at the new state_machine
/// param. Yield-block returns produce `0_i64` (pending sentinel).
/// Non-yield returns wrap their value in a cast to i64.
///
/// The function is renamed `{name}$poll` and `is_async` is cleared.
/// Returns the new poll fn name.
pub fn reshape_to_poll_abi(
    function: &mut HirFunction,
    layout: &StateMachineLayout<HirBlockId, HirLocalId, HirFnId>,
    param_slots: &[(HirId, u32)],
) -> InternedString {
    let original_return_ty = function
        .signature
        .returns
        .first()
        .cloned()
        .unwrap_or(HirType::Void);

    // Step 0: neutralize the original Parameter kinds. The Cranelift
    // backend (cranelift_backend.rs:1432) iterates `function.values`
    // for `HirValueKind::Parameter(idx)` to map params to the entry
    // block's incoming Cranelift values. Leaving the original params
    // as Parameter(N) would collide with the new state_machine param
    // we're about to mint as Parameter(0). The originals are now
    // orphans — body uses got rewritten to load results in Step 3 —
    // so flip them to `Instruction` so they're harmless side data.
    for v in function.values.values_mut() {
        if matches!(v.kind, HirValueKind::Parameter(_)) {
            v.kind = HirValueKind::Instruction;
        }
    }

    // Step 1: mint a fresh `state_machine` SSA value — this becomes
    // the function's sole param, AND the new `frame` for every
    // AsyncSaveSlot/AsyncLoadSlot op currently in the function.
    let sm_ptr_id = HirId::new();
    function.values.insert(
        sm_ptr_id,
        HirValue {
            id: sm_ptr_id,
            ty: HirType::Ptr(Box::new(HirType::U8)),
            kind: HirValueKind::Parameter(0),
            uses: HashSet::new(),
            span: None,
        },
    );

    // Step 2: rewrite all `frame: X` references in async ops to the
    // new state_machine param. Krio originally pointed `frame` at the
    // old first param (or a synthetic HirId from the caller closure);
    // either way, we now own the frame.
    for block in function.blocks.values_mut() {
        for inst in &mut block.instructions {
            match inst {
                HirInstruction::AsyncSaveSlot { frame, .. }
                | HirInstruction::AsyncLoadSlot { frame, .. } => {
                    *frame = sm_ptr_id;
                }
                _ => {}
            }
        }
    }

    // Step 3: build the param-load prologue. For each original param,
    // emit AsyncLoadSlot { result: <fresh>, ty, frame: sm_ptr, slot }
    // and rewrite all uses of the original param HirId to the load
    // result.
    let mut prologue_insts: Vec<HirInstruction> = Vec::with_capacity(param_slots.len());
    let mut rewrites: IndexMap<HirId, HirId> = IndexMap::new();
    for (orig_param_id, slot) in param_slots {
        // Pull the original param's HIR type from values[].
        let orig_ty = function
            .values
            .get(orig_param_id)
            .map(|v| v.ty.clone())
            .unwrap_or(HirType::I64);
        let load_id = HirId::new();
        function.values.insert(
            load_id,
            HirValue {
                id: load_id,
                ty: orig_ty.clone(),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        prologue_insts.push(HirInstruction::AsyncLoadSlot {
            result: load_id,
            ty: orig_ty,
            frame: sm_ptr_id,
            slot: *slot,
        });
        rewrites.insert(*orig_param_id, load_id);
    }

    // Apply rewrites across all instructions, terminators, AND phi
    // node incoming values. Phis reference SSA values directly via
    // `incoming: Vec<(value_hir_id, predecessor_block_id)>`, so they
    // can hold stale references to original param HirIds (e.g. an
    // initial loop-counter value flowing into a phi from the entry
    // block). Without rewriting these, Cranelift sees an undefined
    // operand and rejects the jump argument count.
    for block in function.blocks.values_mut() {
        for phi in &mut block.phis {
            for (value, _pred) in &mut phi.incoming {
                if let Some(&new_id) = rewrites.get(value) {
                    *value = new_id;
                }
            }
        }
        for inst in &mut block.instructions {
            inst.replace_uses(&rewrites);
        }
        block.terminator.replace_uses(&rewrites);
    }

    // Step 4: insert the prologue block BEFORE the existing entry
    // (which is krio's dispatcher). The prologue branches to the
    // dispatcher unconditionally.
    let old_entry_hir = function.entry_block;
    let prologue_id = HirId::new();
    let prologue_block = HirBlock {
        id: prologue_id,
        label: Some(InternedString::new_global("krio_param_prologue")),
        phis: vec![],
        instructions: prologue_insts,
        terminator: HirTerminator::Branch {
            target: old_entry_hir,
        },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![old_entry_hir],
    };

    // IndexMap insertion order matters for backend iteration. Build a
    // new IndexMap with the prologue first, then all existing blocks.
    let mut new_blocks: IndexMap<HirId, HirBlock> = IndexMap::new();
    new_blocks.insert(prologue_id, prologue_block);
    for (id, block) in function.blocks.drain(..) {
        new_blocks.insert(id, block);
    }
    function.blocks = new_blocks;
    function.entry_block = prologue_id;

    // Step 5: rewrite yield-block returns to const 0_i64 (pending).
    let yield_block_seq_ids: HashSet<HirBlockId> =
        layout.yield_saves.iter().map(|(bb, _)| *bb).collect();
    // Map seq IDs back to HirIds. Krio's HirCoroCfg uses 0-indexed
    // seq numbers over function.blocks insertion order — but we just
    // inserted the prologue at position 0, shifting everything by 1.
    // Resolve using the BLOCK INDEX at the time krio ran (i.e., the
    // current block list MINUS the prologue we just added).
    let yield_hir_ids: HashSet<HirId> = function
        .blocks
        .keys()
        .skip(1) // skip the prologue we just prepended
        .enumerate()
        .filter_map(|(i, bb_id)| {
            if yield_block_seq_ids.contains(&HirBlockId(i as u32)) {
                Some(*bb_id)
            } else {
                None
            }
        })
        .collect();

    // Pending sentinel constant.
    let zero_i64_id = HirId::new();
    function.values.insert(
        zero_i64_id,
        HirValue {
            id: zero_i64_id,
            ty: HirType::I64,
            kind: HirValueKind::Constant(HirConstant::I64(0)),
            uses: HashSet::new(),
            span: None,
        },
    );

    // Step 6: walk every Return-terminated block.
    //   - if it's a yield block: replace return values with [zero_i64]
    //   - otherwise: insert a cast to i64 of the original return value
    //     and replace `values` with the cast result
    let block_ids: Vec<HirId> = function.blocks.keys().copied().collect();
    for bb_id in block_ids {
        // Need to inspect terminator BEFORE potentially mutating
        // instructions.
        let is_yield = yield_hir_ids.contains(&bb_id);
        let block = function
            .blocks
            .get_mut(&bb_id)
            .expect("block id from snapshot");
        let new_terminator = match &block.terminator {
            HirTerminator::Return { values } => {
                if is_yield {
                    HirTerminator::Return {
                        values: vec![zero_i64_id],
                    }
                } else if values.is_empty() {
                    // Void return → encode as constant 1 (Ready, no value).
                    let one_id = HirId::new();
                    function.values.insert(
                        one_id,
                        HirValue {
                            id: one_id,
                            ty: HirType::I64,
                            kind: HirValueKind::Constant(HirConstant::I64(1)),
                            uses: HashSet::new(),
                            span: None,
                        },
                    );
                    HirTerminator::Return {
                        values: vec![one_id],
                    }
                } else {
                    let original_val = values[0];
                    let original_ty = function
                        .values
                        .get(&original_val)
                        .map(|v| v.ty.clone())
                        .unwrap_or(original_return_ty.clone());
                    let cast_id = HirId::new();
                    function.values.insert(
                        cast_id,
                        HirValue {
                            id: cast_id,
                            ty: HirType::I64,
                            kind: HirValueKind::Instruction,
                            uses: HashSet::new(),
                            span: None,
                        },
                    );
                    let cast_op = pick_to_i64_cast(&original_ty);
                    // re-fetch the block to insert cast (we lost &mut to it)
                    let block_mut = function
                        .blocks
                        .get_mut(&bb_id)
                        .expect("block id from snapshot");
                    block_mut.instructions.push(HirInstruction::Cast {
                        op: cast_op,
                        result: cast_id,
                        ty: HirType::I64,
                        operand: original_val,
                    });
                    HirTerminator::Return {
                        values: vec![cast_id],
                    }
                }
            }
            other => other.clone(),
        };
        function
            .blocks
            .get_mut(&bb_id)
            .expect("block id from snapshot")
            .terminator = new_terminator;
    }

    // Step 7: replace the function's signature with the poll ABI.
    let sm_param = HirParam {
        id: sm_ptr_id,
        name: InternedString::new_global("state_machine"),
        ty: HirType::Ptr(Box::new(HirType::U8)),
        attributes: ParamAttributes::default(),
    };
    function.signature.params = vec![sm_param];
    function.signature.returns = vec![HirType::I64];
    function.signature.is_async = false;

    // Step 8: rename to `{name}$poll`.
    let old_name = function
        .name
        .resolve_global()
        .unwrap_or_else(|| String::from("anon"));
    let new_name = InternedString::new_global(&format!("{}$poll", old_name));
    function.name = new_name;
    new_name
}

/// Pick the right cast op to convert `from_ty` into i64 for the
/// runtime's i64-encoded ready-value channel.
fn pick_to_i64_cast(from_ty: &HirType) -> CastOp {
    match from_ty {
        HirType::Bool | HirType::U8 | HirType::U16 | HirType::U32 | HirType::U64 => CastOp::ZExt,
        HirType::I8 | HirType::I16 | HirType::I32 => CastOp::SExt,
        HirType::I64 => CastOp::Bitcast, // same width — bitcast is a no-op cast
        HirType::F32 | HirType::F64 => CastOp::Bitcast,
        HirType::Ptr(_) | HirType::Opaque(_) | HirType::Function(_) => CastOp::PtrToInt,
        _ => CastOp::Bitcast,
    }
}

/// Pick the right cast op to convert `to_ty` (a parameter type) FROM
/// its source into i64 for storing into the state-machine's
/// 8-byte slot. The runtime stores params in 8-byte slots so we
/// always extend up to i64.
fn pick_param_to_i64_cast(from_ty: &HirType) -> CastOp {
    pick_to_i64_cast(from_ty)
}

/// Generate a Promise-returning entry function that wraps a krio'd
/// poll function. This is the function the runtime sees as `{name}`.
///
/// Body sketch:
/// ```text
///   sm = malloc(num_slots * 8)
///   *(sm + state_slot * 8) = 0_u32         // initial state
///   for (param, slot) in param_slots:
///       *(sm + slot * 8) = cast<i64>(param) // store args
///   poll_ptr = &poll_fn                    // CreateClosure no captures
///   promise = malloc(16)
///   *(promise + 0) = sm
///   *(promise + 8) = poll_ptr
///   return promise
/// ```
pub fn generate_promise_entry(
    original_name: InternedString,
    original_signature: &HirFunctionSignature,
    poll_fn_id: HirId,
    num_slots: u32,
    param_slots: &[(HirId, u32)],
    state_slot: u32,
) -> HirFunction {
    let promise_struct_ty = HirType::Struct(HirStructType {
        name: Some(InternedString::new_global("Promise")),
        fields: vec![
            HirType::Ptr(Box::new(HirType::U8)), // state_machine
            HirType::Function(Box::new(HirFunctionType {
                params: vec![HirType::Ptr(Box::new(HirType::U8))],
                returns: vec![HirType::I64],
                lifetime_params: vec![],
                is_variadic: false,
            })), // poll_fn
        ],
        packed: false,
    });
    let promise_ptr_ty = HirType::Ptr(Box::new(promise_struct_ty.clone()));

    // Entry function signature: keep original params, return *Promise.
    let entry_sig = HirFunctionSignature {
        params: original_signature.params.clone(),
        returns: vec![promise_ptr_ty.clone()],
        type_params: original_signature.type_params.clone(),
        const_params: original_signature.const_params.clone(),
        lifetime_params: original_signature.lifetime_params.clone(),
        is_variadic: false,
        is_async: false, // the entry IS the public face; no further async transform
        effects: original_signature.effects.clone(),
        is_pure: false,
    };

    let mut entry = HirFunction::new(original_name, entry_sig);

    // Re-mint param SSA values so they're owned by THIS function.
    // The new param HirIds match the entry's signature.params.id.
    let mut entry_param_ids: Vec<HirId> = Vec::with_capacity(original_signature.params.len());
    for (i, p) in original_signature.params.iter().enumerate() {
        entry.values.insert(
            p.id,
            HirValue {
                id: p.id,
                ty: p.ty.clone(),
                kind: HirValueKind::Parameter(i as u32),
                uses: HashSet::new(),
                span: None,
            },
        );
        entry_param_ids.push(p.id);
    }

    // Build a single-block body that does the malloc/store dance.
    let entry_block_id = entry.entry_block;
    let mut instructions: Vec<HirInstruction> = Vec::new();

    // sm_size = num_slots * 8
    let sm_size_id = mint_const_i64(&mut entry.values, num_slots as i64 * 8);

    // sm = malloc(sm_size)
    let sm_ptr_id = mint_value(
        &mut entry.values,
        HirType::Ptr(Box::new(HirType::U8)),
        HirValueKind::Instruction,
    );
    instructions.push(HirInstruction::Call {
        result: Some(sm_ptr_id),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![sm_size_id],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });

    // Zero-initialize EVERY slot. Some await re-entry checks rely on
    // promise_slot == 0 as the "not yet called" sentinel. malloc
    // doesn't zero memory, so we must explicitly write 0 to every
    // slot here. (State and param slots get overwritten below; doing
    // a uniform pass keeps the logic simple.)
    let zero_init_const = mint_const_i64(&mut entry.values, 0);
    for slot_idx in 0..num_slots {
        let offset_const = mint_const_i64(&mut entry.values, slot_idx as i64 * 8);
        let slot_ptr = mint_value(
            &mut entry.values,
            HirType::Ptr(Box::new(HirType::I64)),
            HirValueKind::Instruction,
        );
        instructions.push(HirInstruction::Binary {
            result: slot_ptr,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: sm_ptr_id,
            right: offset_const,
        });
        instructions.push(HirInstruction::Store {
            ptr: slot_ptr,
            value: zero_init_const,
            align: 8,
            volatile: false,
        });
    }

    // Initialize state_id slot to 0 (as i64, matching uniform 8-byte
    // slot layout — the dispatcher reads i64 from the slot).
    let state_init_id = mint_const_i64(&mut entry.values, 0);
    let state_slot_offset_id = mint_const_i64(&mut entry.values, state_slot as i64 * 8);
    let state_field_ptr_id = mint_value(
        &mut entry.values,
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    instructions.push(HirInstruction::Binary {
        result: state_field_ptr_id,
        op: BinaryOp::Add,
        ty: HirType::I64,
        left: sm_ptr_id,
        right: state_slot_offset_id,
    });
    instructions.push(HirInstruction::Store {
        ptr: state_field_ptr_id,
        value: state_init_id,
        align: 8,
        volatile: false,
    });

    // Store each param into its slot, casting up to i64. The
    // param HirIds used here are the ENTRY function's signature
    // params (re-minted as Parameter(i) values above), NOT the
    // body-facing SSA HirIds from `param_slots[i].0` which only
    // existed in the krio'd poll function. We use param_slots only
    // for the slot indices, taking them positionally.
    debug_assert_eq!(
        param_slots.len(),
        original_signature.params.len(),
        "param_slots count must match signature.params count"
    );
    for (i, p) in original_signature.params.iter().enumerate() {
        let slot = param_slots[i].1;
        let param_ty = p.ty.clone();
        let cast_id = mint_value(&mut entry.values, HirType::I64, HirValueKind::Instruction);
        let cast_op = pick_param_to_i64_cast(&param_ty);
        instructions.push(HirInstruction::Cast {
            op: cast_op,
            result: cast_id,
            ty: HirType::I64,
            operand: p.id,
        });
        let slot_offset_id = mint_const_i64(&mut entry.values, slot as i64 * 8);
        let field_ptr_id = mint_value(
            &mut entry.values,
            HirType::Ptr(Box::new(HirType::I64)),
            HirValueKind::Instruction,
        );
        instructions.push(HirInstruction::Binary {
            result: field_ptr_id,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: sm_ptr_id,
            right: slot_offset_id,
        });
        instructions.push(HirInstruction::Store {
            ptr: field_ptr_id,
            value: cast_id,
            align: 8,
            volatile: false,
        });
    }

    // Get the poll fn pointer via CreateClosure (matches the legacy
    // pattern at async_support.rs:1898 — the backend lowers this to a
    // function-address load).
    let poll_fn_ty = HirType::Function(Box::new(HirFunctionType {
        params: vec![HirType::Ptr(Box::new(HirType::U8))],
        returns: vec![HirType::I64],
        lifetime_params: vec![],
        is_variadic: false,
    }));
    let poll_fn_ptr_id = mint_value(&mut entry.values, poll_fn_ty.clone(), HirValueKind::Instruction);
    instructions.push(HirInstruction::CreateClosure {
        result: poll_fn_ptr_id,
        closure_ty: poll_fn_ty,
        function: poll_fn_id,
        captures: vec![],
    });

    // Allocate the 16-byte Promise struct.
    let promise_size_id = mint_const_i64(&mut entry.values, 16);
    let promise_ptr_id = mint_value(&mut entry.values, promise_ptr_ty.clone(), HirValueKind::Instruction);
    instructions.push(HirInstruction::Call {
        result: Some(promise_ptr_id),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![promise_size_id],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });

    // Store sm_ptr at Promise+0.
    instructions.push(HirInstruction::Store {
        ptr: promise_ptr_id,
        value: sm_ptr_id,
        align: 8,
        volatile: false,
    });

    // Store poll_fn_ptr at Promise+8.
    let promise_offset_8_id = mint_const_i64(&mut entry.values, 8);
    let poll_field_ptr_id = mint_value(
        &mut entry.values,
        HirType::Ptr(Box::new(HirType::U8)),
        HirValueKind::Instruction,
    );
    instructions.push(HirInstruction::Binary {
        result: poll_field_ptr_id,
        op: BinaryOp::Add,
        ty: HirType::I64,
        left: promise_ptr_id,
        right: promise_offset_8_id,
    });
    instructions.push(HirInstruction::Store {
        ptr: poll_field_ptr_id,
        value: poll_fn_ptr_id,
        align: 8,
        volatile: false,
    });

    // Populate the entry block.
    let entry_block = entry
        .blocks
        .get_mut(&entry_block_id)
        .expect("HirFunction::new always creates an entry block");
    entry_block.instructions = instructions;
    entry_block.terminator = HirTerminator::Return {
        values: vec![promise_ptr_id],
    };

    let _ = entry_param_ids; // prevent unused warning if we extend later

    entry
}

fn mint_value(
    values: &mut IndexMap<HirId, HirValue>,
    ty: HirType,
    kind: HirValueKind,
) -> HirId {
    let id = HirId::new();
    values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn mint_const_i64(values: &mut IndexMap<HirId, HirValue>, val: i64) -> HirId {
    mint_value(
        values,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(val)),
    )
}
