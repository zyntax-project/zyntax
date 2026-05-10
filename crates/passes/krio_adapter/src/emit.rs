//! Phase E1 + E2: emit save/load HIR instructions and the dispatcher
//! prologue from a krio `StateMachineLayout`.
//!
//! These are the host-side compilers of krio's per-block recipe:
//!
//! * `emit_save_load`: walks `layout.yield_saves` and `layout.resume_loads`,
//!   inserts `HirInstruction::AsyncSaveSlot` before yield-block returns
//!   and `HirInstruction::AsyncLoadSlot` at the start of each resume
//!   entry. Subsequent uses of the saved SSA value (in the resume
//!   region) are rewritten to point at the freshly-loaded value via
//!   the existing `HirInstruction::replace_uses` machinery.
//!
//! * `emit_dispatcher`: turns the function entry block into a switch
//!   on the state-id local that targets `layout.resume_entries[i]`
//!   for each state. State 0 dispatches to the original entry; states
//!   1..N to each split tail.
//!
//! Both functions take the same `&mut HirCoroCfg` they were given to
//! krio's `transform_to_state_machine`, so all the block-id round-
//! tripping stays internally consistent.
//!
//! ## Frame pointer convention
//!
//! `AsyncSaveSlot` / `AsyncLoadSlot` take a `frame: HirId` operand —
//! the future struct's `self` pointer. Today's lowering pulls it from
//! the function's first parameter (`is_async = true` functions are
//! lowered with their poll-body taking `(state: *mut Future, ...)`
//! by the existing async pipeline). The Cranelift backend (Phase E3)
//! materialises the actual offsets per slot.

use std::collections::{HashMap, HashSet};

use krio_async::{BlockKind, StateMachineLayout};
use zyntax_compiler::hir::{
    HirBlock, HirConstant, HirId, HirInstruction, HirTerminator, HirType, HirValue, HirValueKind,
};

use crate::{HirBlockId, HirCoroCfg, HirFnId, HirLiveness};

/// Emit `AsyncSaveSlot` / `AsyncLoadSlot` instructions per the
/// captures-lift recipe in `layout`. Returns a map from saved SSA
/// HirIds to their freshly-defined load result HirIds — Phase E2's
/// dispatcher emission consumes this for downstream use rewriting.
///
/// `frame_ptr` is the SSA value that holds the future struct's
/// `self` pointer (typically `function.signature.params[0].id`).
///
/// ## What it does, per yield/resume pair
///
/// For each `(yield_block, vec[(slot, hir_local)])` in
/// `layout.yield_saves`:
///   * Look up each `hir_local`'s SSA HirId via `liveness.local_to_hir`
///   * Insert `AsyncSaveSlot { frame, slot, value: ssa_hir_id }`
///     immediately before the yield block's terminator. Saves are
///     pushed in slot-index order.
///
/// For each `(resume_block, vec[(slot, fresh_local)])` in
/// `layout.resume_loads`:
///   * For each entry, mint a fresh SSA HirId of the saved value's
///     type and insert `AsyncLoadSlot { result, ty, frame, slot }`
///     at the *front* of the resume block.
///   * Record the original→fresh HirId mapping; the caller (or
///     subsequent emission steps) rewrites uses across the resume
///     region.
pub fn emit_save_load(
    cfg: &mut HirCoroCfg<'_>,
    layout: &StateMachineLayout<HirBlockId, crate::HirLocalId, HirFnId>,
    liveness: &HirLiveness,
    frame_ptr: HirId,
) -> HashMap<HirId, HirId> {
    // ── Saves ──
    for (yield_block, saves) in &layout.yield_saves {
        // Collect saves into a Vec we can walk after the (cfg, function)
        // borrow is released — keeps lifetimes clean. Do nothing if the
        // block isn't actually in the cfg (defensive).
        if cfg.block_id_to_hir(*yield_block) == HirId::new() {
            // dummy comparison; HirId::new() generates a fresh UUID
            // each time — so this is always false. Keeping the
            // structure for symmetry with future bounds checks.
        }
        let yield_hir = cfg.block_id_to_hir(*yield_block);
        let func = cfg.function_mut();
        if let Some(block) = func.blocks.get_mut(&yield_hir) {
            for (slot, hir_local) in saves {
                let value = match liveness.local_to_hir.get(hir_local) {
                    Some(&v) => v,
                    None => continue, // not a captures-lift slot we mapped
                };
                let inst = HirInstruction::AsyncSaveSlot {
                    frame: frame_ptr,
                    slot: *slot,
                    value,
                };
                block.instructions.push(inst);
            }
        }
    }

    // ── Loads (and freshly-defined SSA values for them) ──
    let mut original_to_fresh: HashMap<HirId, HirId> = HashMap::new();

    for (resume_block, loads) in &layout.resume_loads {
        let resume_hir = cfg.block_id_to_hir(*resume_block);
        // First mint the fresh SSA values + register them in function.values.
        // We do this in a separate pass because each minting needs &mut on
        // function.values, and the block.instructions.insert below also
        // needs &mut on function.blocks — both are fields of HirFunction.
        let mut fresh_loads: Vec<(u32, HirId, HirType, HirId)> = Vec::new(); // (slot, fresh_id, ty, original_id)
        for (slot, hir_local) in loads {
            let original_id = match liveness.local_to_hir.get(hir_local) {
                Some(&v) => v,
                None => continue,
            };
            let original_ty = cfg
                .function()
                .values
                .get(&original_id)
                .map(|v| v.ty.clone())
                .unwrap_or(HirType::I64);
            let fresh_id = HirId::new();
            cfg.function_mut().values.insert(
                fresh_id,
                HirValue {
                    id: fresh_id,
                    ty: original_ty.clone(),
                    kind: HirValueKind::Instruction,
                    uses: HashSet::new(),
                    span: None,
                },
            );
            original_to_fresh.insert(original_id, fresh_id);
            fresh_loads.push((*slot, fresh_id, original_ty, original_id));
        }

        // Now insert the AsyncLoadSlot instructions at the front of the
        // block, in slot-index order so the IR reads cleanly.
        if let Some(block) = cfg.function_mut().blocks.get_mut(&resume_hir) {
            // Insert in reverse so each insert(0, _) results in the
            // intended order at the front.
            for (slot, fresh_id, ty, _orig) in fresh_loads.iter().rev() {
                block.instructions.insert(
                    0,
                    HirInstruction::AsyncLoadSlot {
                        result: *fresh_id,
                        ty: ty.clone(),
                        frame: frame_ptr,
                        slot: *slot,
                    },
                );
            }
        }

        // Rewrite uses of the original HirId → fresh HirId in EVERY
        // statement and terminator of the resume block onward. krio's
        // captures-lift contract: "the host just needs to call
        // load_value(frame, slot) and store the result"; and
        // "downstream uses to point at the fresh id" — but krio
        // doesn't know about HIR's use chains, so we do it here.
        if !fresh_loads.is_empty() {
            rewrite_uses_in_reachable(cfg, *resume_block, &original_to_fresh);
        }
    }

    original_to_fresh
}

/// Walk all blocks reachable from `start_bb` (via `cfg.function`'s
/// terminators) and rewrite every operand HirId via `mapping`.
/// Skips the loaded-result-defining instructions themselves: their
/// `value` field is already the fresh id; only post-load uses get
/// rebound.
fn rewrite_uses_in_reachable(
    cfg: &mut HirCoroCfg<'_>,
    start_bb: HirBlockId,
    mapping: &HashMap<HirId, HirId>,
) {
    use indexmap::IndexMap as _;
    let start_hir = cfg.block_id_to_hir(start_bb);
    let func = cfg.function_mut();

    // Reachability over the post-transform CFG.
    let mut visited: HashSet<HirId> = HashSet::new();
    let mut stack: Vec<HirId> = vec![start_hir];

    let full_replacement: indexmap::IndexMap<HirId, HirId> =
        mapping.iter().map(|(k, v)| (*k, *v)).collect();

    // The walk carries the ACTIVE replacement map per-path. A phi
    // node is a "rewrite barrier": once execution passes through a
    // phi block whose result HirId is in our replacement keys, that
    // entry must drop OUT of the active map for downstream blocks.
    // Reason: the phi re-introduces the canonical name at the join,
    // so all uses past the phi should resolve to the phi.result, not
    // to the fresh id from the back-edge. Without this propagation,
    // emit_save_load's rewrites bleed into post-loop code (the
    // function exit, etc.), causing Cranelift dominance failures.
    //
    // Reachability is approximated by visited (we revisit a block
    // only if a more-restrictive map arrives — but to keep
    // complexity bounded we just visit each block once with the
    // first map that reaches it; the loop pattern is the only place
    // multiple maps could differ, and that's exactly what the phi
    // barrier handles).
    type ReplMap = indexmap::IndexMap<HirId, HirId>;
    let mut path_stack: Vec<(HirId, ReplMap)> = vec![(start_hir, full_replacement)];

    while let Some((bb, active_map_into)) = path_stack.pop() {
        if !visited.insert(bb) {
            continue;
        }
        let block = match func.blocks.get_mut(&bb) {
            Some(b) => b,
            None => continue,
        };

        // Shrink the active map: drop entries whose original HirId is
        // re-defined by a phi in THIS block. This block uses the phi
        // result (canonical), and downstream blocks reachable only
        // through this phi should also see the canonical value.
        let block_phi_results: HashSet<HirId> =
            block.phis.iter().map(|p| p.result).collect();
        let mut active_map = active_map_into;
        active_map.retain(|orig, _| !block_phi_results.contains(orig));

        if active_map.is_empty() {
            // Nothing to rewrite past here; still walk successors so
            // they're marked visited (and stop), but no instr edits.
            for succ in successors_of(&block.terminator) {
                path_stack.push((succ, ReplMap::new()));
            }
            continue;
        }

        for inst in &mut block.instructions {
            inst.replace_uses(&active_map);
        }
        rewrite_terminator_uses(&mut block.terminator, &active_map);
        for succ in successors_of(&block.terminator) {
            if !visited.contains(&succ) {
                path_stack.push((succ, active_map.clone()));
            }
        }
    }
}

fn rewrite_terminator_uses(term: &mut HirTerminator, mapping: &indexmap::IndexMap<HirId, HirId>) {
    let remap = |id: &mut HirId| {
        if let Some(&new) = mapping.get(id) {
            *id = new;
        }
    };
    match term {
        HirTerminator::Return { values } => {
            for v in values {
                remap(v);
            }
        }
        HirTerminator::CondBranch { condition, .. } => remap(condition),
        HirTerminator::Switch { value, .. } => remap(value),
        HirTerminator::Invoke { args, .. } => {
            for v in args {
                remap(v);
            }
        }
        HirTerminator::PatternMatch { value, .. } => remap(value),
        HirTerminator::Branch { .. } | HirTerminator::Unreachable => {}
    }
}

fn successors_of(term: &HirTerminator) -> smallvec::SmallVec<[HirId; 4]> {
    let mut out = smallvec::SmallVec::new();
    match term {
        HirTerminator::Branch { target } => out.push(*target),
        HirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            out.push(*true_target);
            out.push(*false_target);
        }
        HirTerminator::Switch { default, cases, .. } => {
            for (_, t) in cases {
                out.push(*t);
            }
            out.push(*default);
        }
        HirTerminator::Invoke { normal, unwind, .. } => {
            out.push(*normal);
            out.push(*unwind);
        }
        HirTerminator::PatternMatch {
            patterns, default, ..
        } => {
            for p in patterns {
                out.push(p.target);
            }
            if let Some(d) = default {
                out.push(*d);
            }
        }
        HirTerminator::Return { .. } | HirTerminator::Unreachable => {}
    }
    out
}

/// Phase E2: emit the dispatcher prologue.
///
/// Replaces the function's entry block with a new prologue block:
///
/// ```text
/// new_entry:
///     state = AsyncLoadSlot(frame, slot=STATE_SLOT, ty=i64)
///     switch state {
///         0 → resume_entries[0]   ;; the original entry
///         1 → resume_entries[1]
///         …
///         default → resume_entries[0]
///     }
/// ```
///
/// `state_slot` is a fixed slot reserved for the state id (typically 0;
/// the host should reserve it before calling [`emit_save_load`] so it
/// doesn't collide with captures-lift slots).
///
/// Returns the new entry block id. Caller updates `function.entry_block`.
pub fn emit_dispatcher(
    cfg: &mut HirCoroCfg<'_>,
    layout: &StateMachineLayout<HirBlockId, crate::HirLocalId, HirFnId>,
    frame_ptr: HirId,
    state_slot: u32,
) -> HirBlockId {
    // Nothing to dispatch if there are no resume entries (krio would
    // already have returned an empty layout for non-suspending fns).
    if layout.resume_entries.is_empty() {
        return HirBlockId(0); // no-op; caller can ignore
    }

    // Mint the new entry block. The block id is appended to cfg's seq
    // tables, so we can reference it via HirBlockId.
    use krio_stackless::CoroCfg;
    let new_entry = cfg.new_block();
    let new_entry_hir = cfg.block_id_to_hir(new_entry);

    // Mint the SSA value the AsyncLoadSlot will write into.
    let state_val = HirId::new();
    cfg.function_mut().values.insert(
        state_val,
        HirValue {
            id: state_val,
            ty: HirType::I64,
            kind: HirValueKind::Instruction,
            uses: HashSet::new(),
            span: None,
        },
    );

    // Build the switch's case table: state index → resume entry hir id.
    let cases: Vec<(HirConstant, HirId)> = layout
        .resume_entries
        .iter()
        .enumerate()
        .map(|(i, bb)| (HirConstant::I64(i as i64), cfg.block_id_to_hir(*bb)))
        .collect();
    let default_target = cfg.block_id_to_hir(layout.resume_entries[0]);

    // Populate the new entry block.
    let new_block = cfg
        .function_mut()
        .blocks
        .get_mut(&new_entry_hir)
        .expect("new_block was just created");
    new_block.instructions.push(HirInstruction::AsyncLoadSlot {
        result: state_val,
        ty: HirType::I64,
        frame: frame_ptr,
        slot: state_slot,
    });
    new_block.terminator = HirTerminator::Switch {
        value: state_val,
        default: default_target,
        cases,
    };

    // Re-point the function's entry to the dispatcher.
    cfg.function_mut().entry_block = new_entry_hir;

    new_entry
}
