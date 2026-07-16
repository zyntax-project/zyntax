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

use krio_async::StateMachineLayout;
use zyntax_compiler::hir::{
    HirConstant, HirId, HirInstruction, HirTerminator, HirType, HirValue, HirValueKind,
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
#[allow(clippy::type_complexity)]
pub fn emit_save_load(
    cfg: &mut HirCoroCfg<'_>,
    layout: &StateMachineLayout<HirBlockId, crate::HirLocalId, HirFnId>,
    liveness: &HirLiveness,
    frame_ptr: HirId,
) -> (HashMap<HirId, HirId>, HashMap<HirId, Vec<(HirId, HirId)>>) {
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
    // For each saved value, mint a fresh SSA id per resume block that
    // reloads it and insert the `AsyncLoadSlot`. We DON'T rewrite uses
    // per-resume-block here (the old approach) — a value defined once
    // before a loop and reloaded inside it needs a loop-header phi, not a
    // blind original→reload substitution that breaks the entry path.
    // Collect the reloads and do proper SSA reconstruction afterwards.
    let mut original_to_fresh: HashMap<HirId, HirId> = HashMap::new();
    // original SSA id → [(reload id, resume-block hir id)]
    let mut reloads: HashMap<HirId, Vec<(HirId, HirId)>> = HashMap::new();

    for (resume_block, loads) in &layout.resume_loads {
        let resume_hir = cfg.block_id_to_hir(*resume_block);
        let mut fresh_loads: Vec<(u32, HirId, HirType)> = Vec::new(); // (slot, fresh_id, ty)
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
            reloads
                .entry(original_id)
                .or_default()
                .push((fresh_id, resume_hir));
            fresh_loads.push((*slot, fresh_id, original_ty));
        }

        // Insert the AsyncLoadSlot instructions at the front of the block,
        // in slot-index order so the IR reads cleanly.
        if let Some(block) = cfg.function_mut().blocks.get_mut(&resume_hir) {
            for (slot, fresh_id, ty) in fresh_loads.iter().rev() {
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
    }

    // SSA reconstruction is deferred to AFTER `emit_dispatcher` wires the
    // dispatcher→resume-entry edges — until then the resume blocks are
    // orphaned and dominance can't be computed. The orchestrator calls
    // `repair_ssa_for_reloads` with the returned `reloads` map.
    (original_to_fresh, reloads)
}

/// Number of blocks reachable from `start` over terminator edges.
fn reachable_block_count(func: &zyntax_compiler::hir::HirFunction, start: HirId) -> usize {
    let mut seen: HashSet<HirId> = HashSet::new();
    let mut stack = vec![start];
    while let Some(b) = stack.pop() {
        if !seen.insert(b) {
            continue;
        }
        if let Some(block) = func.blocks.get(&b) {
            for s in successors_of(&block.terminator) {
                stack.push(s);
            }
        }
    }
    seen.len()
}

/// Rebuild `successors`/`predecessors` on every block from its terminator.
/// The async lowering leaves these cached fields empty/stale; the dominator
/// tree needs them accurate.
fn rebuild_cfg_edges(func: &mut zyntax_compiler::hir::HirFunction) {
    for b in func.blocks.values_mut() {
        b.successors.clear();
        b.predecessors.clear();
    }
    let edges: Vec<(HirId, HirId)> = func
        .blocks
        .iter()
        .flat_map(|(bid, b)| {
            let from = *bid;
            successors_of(&b.terminator)
                .into_iter()
                .map(move |s| (from, s))
        })
        .collect();
    for (from, to) in edges {
        if let Some(b) = func.blocks.get_mut(&from) {
            b.successors.push(to);
        }
        if let Some(b) = func.blocks.get_mut(&to) {
            b.predecessors.push(from);
        }
    }
}

/// Iterated dominance frontier of a set of definition blocks — the blocks
/// that need a phi for a value defined in those blocks.
fn iterated_dominance_frontier(
    dt: &zyntax_compiler::analysis::DominatorTree,
    defs: &HashSet<HirId>,
) -> HashSet<HirId> {
    let mut idf = HashSet::new();
    let mut worklist: Vec<HirId> = defs.iter().copied().collect();
    while let Some(b) = worklist.pop() {
        for &f in dt.frontier(b) {
            if idf.insert(f) {
                worklist.push(f);
            }
        }
    }
    idf
}

/// For a single logical value with defs at `def_value` (block → SSA id),
/// compute the reaching definition entering each block: its own def if it
/// has one, else the reaching def of its immediate dominator.
fn compute_reaching_defs(
    dt: &zyntax_compiler::analysis::DominatorTree,
    def_value: &HashMap<HirId, HirId>,
) -> HashMap<HirId, HirId> {
    let mut reaching: HashMap<HirId, HirId> = HashMap::new();
    // RPO visits each block after its immediate dominator.
    for &b in dt.rpo() {
        if let Some(&v) = def_value.get(&b) {
            reaching.insert(b, v);
        } else if let Some(idom) = dt.idom(b) {
            if let Some(&v) = reaching.get(&idom) {
                reaching.insert(b, v);
            }
        }
    }
    reaching
}

/// Proper SSA reconstruction for values that are saved before a suspend and
/// reloaded on resume. The reload creates a second definition of the value;
/// where the pre-suspend definition and the reload paths merge (loop
/// headers), a phi is required. Blindly rewriting uses to the reload
/// instead breaks the first (pre-suspend) iteration, where the reload is
/// undefined. For each saved value: place phis at the iterated dominance
/// frontier of {original-def, reload blocks}, then rename every use to the
/// reaching definition and wire up the phi operands per predecessor.
pub fn repair_ssa_for_reloads(
    func: &mut zyntax_compiler::hir::HirFunction,
    reloads: &HashMap<HirId, Vec<(HirId, HirId)>>,
) {
    use zyntax_compiler::analysis::DominatorTree;
    use zyntax_compiler::hir::{HirPhi, HirType, HirValue, HirValueKind};

    if reloads.is_empty() {
        return;
    }

    rebuild_cfg_edges(func);
    // The async transform leaves `entry_block` pointing at the ORIGINAL
    // (pre-dispatcher) entry, which no longer dominates the resume blocks.
    // The real entry after the transform is the dispatcher prologue — a
    // no-predecessor block. There can be several no-pred blocks (orphaned
    // closure bodies etc.), so pick the one that reaches the most blocks.
    // Root the dominator tree there so every block (incl. resume entries)
    // is in its RPO.
    let real_entry = func
        .blocks
        .iter()
        .filter(|(_, b)| b.predecessors.is_empty())
        .map(|(id, _)| *id)
        .max_by_key(|&c| reachable_block_count(func, c));
    if let Some(e) = real_entry {
        func.entry_block = e;
    }
    let dt = DominatorTree::new(func);

    // Defining block of every SSA value (params → entry block).
    let mut def_block: HashMap<HirId, HirId> = HashMap::new();
    for p in &func.signature.params {
        def_block.insert(p.id, func.entry_block);
    }
    for (bid, block) in &func.blocks {
        for phi in &block.phis {
            def_block.insert(phi.result, *bid);
        }
        for inst in &block.instructions {
            if let Some(r) = crate::instruction_result(inst) {
                def_block.insert(r, *bid);
            }
        }
    }

    for (original, reload_list) in reloads {
        // Definition sites: the original def block + every reload block.
        let odb = def_block.get(original).copied().unwrap_or(func.entry_block);
        let mut def_blocks: HashSet<HirId> = HashSet::new();
        def_blocks.insert(odb);
        for (_, rb) in reload_list {
            def_blocks.insert(*rb);
        }

        let ty = func
            .values
            .get(original)
            .map(|v| v.ty.clone())
            .unwrap_or(HirType::I64);

        // Reaching definitions from the base defs only (no phis yet). Used
        // to prune spurious phis: a phi is only needed where the value has
        // ≥2 distinct reaching defs among the block's predecessors. A value
        // re-defined each iteration inside the loop (e.g. the `let Some(x)`
        // binding) is NOT live-in at the loop header — one predecessor has
        // no reaching def — so it must not get a (would-be-undefined) phi.
        let mut base_defs: HashMap<HirId, HirId> = HashMap::new();
        base_defs.insert(odb, *original);
        for (rid, rb) in reload_list {
            base_defs.insert(*rb, *rid);
        }
        let reaching0 = compute_reaching_defs(&dt, &base_defs);

        // Allocate a phi value per IDF block that is a genuine merge.
        let phi_blocks = iterated_dominance_frontier(&dt, &def_blocks);
        let mut def_value = base_defs.clone();
        let mut phi_result: HashMap<HirId, HirId> = HashMap::new();
        for &pb in &phi_blocks {
            // Skip blocks that are already definition sites: the original
            // def block (when it's a loop-header phi, e.g. `sum`/`i` — the
            // existing phi already merges the back-edge) or a reload block.
            // Inserting another phi there would double-define the value.
            if def_value.contains_key(&pb) {
                continue;
            }
            // Prune: only a real merge (≥2 distinct reaching defs among
            // predecessors) needs a phi. Fewer means the value isn't
            // live-in here on all paths — a phi would introduce an
            // undefined operand that gets read at runtime.
            let distinct: HashSet<HirId> = func
                .blocks
                .get(&pb)
                .map(|b| {
                    b.predecessors
                        .iter()
                        .filter_map(|q| reaching0.get(q).copied())
                        .collect()
                })
                .unwrap_or_default();
            if distinct.len() < 2 {
                continue;
            }
            let pid = HirId::new();
            func.values.insert(
                pid,
                HirValue {
                    id: pid,
                    ty: ty.clone(),
                    kind: HirValueKind::Instruction,
                    uses: HashSet::new(),
                    span: None,
                },
            );
            def_value.insert(pb, pid);
            phi_result.insert(pb, pid);
        }

        let reaching = compute_reaching_defs(&dt, &def_value);

        // Rename uses (instructions + terminator, NOT phis) to the reaching
        // def. Phis are wired separately below so their per-predecessor
        // operands stay correct.
        for (bid, block) in func.blocks.iter_mut() {
            let rd = match reaching.get(bid) {
                Some(&v) => v,
                None => continue,
            };
            if rd == *original {
                continue;
            }
            let map: indexmap::IndexMap<HirId, HirId> = std::iter::once((*original, rd)).collect();
            for inst in &mut block.instructions {
                inst.replace_uses(&map);
            }
            rewrite_terminator_uses(&mut block.terminator, &map);
        }

        // Wire the inserted phis: each predecessor contributes the reaching
        // def at its exit. Only the phis we actually inserted (not skipped
        // pre-existing def sites). A predecessor with no reaching def is one
        // where the value isn't defined on that path (an over-placed phi for
        // a value not live-in there) — fill it with `Undef` so the phi is
        // well-formed for every predecessor; such phis are dead and get
        // cleaned up downstream.
        for (&pb, &pid) in &phi_result {
            let preds = func
                .blocks
                .get(&pb)
                .map(|b| b.predecessors.clone())
                .unwrap_or_default();
            let mut incoming: Vec<(HirId, HirId)> = Vec::with_capacity(preds.len());
            for q in &preds {
                let v = if let Some(&v) = reaching.get(q) {
                    v
                } else {
                    let uid = HirId::new();
                    func.values.insert(
                        uid,
                        HirValue {
                            id: uid,
                            ty: ty.clone(),
                            kind: HirValueKind::Undef,
                            uses: HashSet::new(),
                            span: None,
                        },
                    );
                    uid
                };
                incoming.push((v, *q));
            }
            if let Some(block) = func.blocks.get_mut(&pb) {
                block.phis.push(HirPhi {
                    result: pid,
                    ty: ty.clone(),
                    incoming,
                });
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
