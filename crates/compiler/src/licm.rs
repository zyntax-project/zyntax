//! HIR-level loop-invariant code motion.
//!
//! For each natural loop produced by `analysis::loops`, hoists pure
//! instructions whose operands are all defined outside the loop into
//! the loop's preheader block. The preheader is the unique
//! outside-the-loop predecessor of the header; we use it as-is
//! rather than synthesising one. Loops that don't have a unique
//! outside-the-loop predecessor are skipped — a separate "ensure
//! preheader" pass would be cheap to add later and is the only thing
//! gating us from broader coverage.
//!
//! ## What we hoist
//!
//! Pure, non-trapping instructions whose operands are *loop-
//! invariant* — defined outside the loop or in a block that
//! dominates the loop header:
//!
//!   * `Binary`           (add / sub / mul / bitwise / shifts /
//!                         comparisons, excluding `Div` and `Rem`
//!                         to avoid changing trap semantics)
//!   * `Unary`            (Neg / Not / FNeg)
//!   * `Cast`
//!   * `GetElementPtr`    — pure address arithmetic
//!   * `ExtractValue`     — pure aggregate field read
//!   * `Select`           — pure ternary
//!   * `Load`             — hoisted only when (a) its pointer operand
//!                          is loop-invariant AND (b) a per-Load
//!                          alias check proves no Store in the loop
//!                          body can write to overlapping memory. The
//!                          alias check walks back through GEP+Cast
//!                          chains on both the Load's pointer and
//!                          every Store's pointer to extract
//!                          `(root_ssa, byte_offset, byte_size)`
//!                          tuples, then disambiguates by root SSA
//!                          (different SSA values ⇒ different memory)
//!                          or by non-overlapping constant offsets on
//!                          the same root. Without this gate the
//!                          common ref-class hot loop where
//!                          `body.vx` is both read and written each
//!                          iteration silently deoptimises.
//!
//! ## What we don't touch
//!
//!   * `Store` / atomics / fences
//!   * `Call` / `IndirectCall` — arbitrary side effects, even
//!                                marked-pure callees can observe
//!                                globals we don't model
//!   * `Alloca` — each iteration's alloca is distinct
//!   * Integer `Div` / `Rem` — moving them changes whether a
//!                              div-by-zero trap fires at all
//!   * `CreateClosure` / `AsyncSaveSlot` / `AsyncLoadSlot` —
//!                              identity / frame state
//!   * `Phi` — control-flow-dependent by definition
//!
//! ## Limits we currently accept
//!
//! No dominance check before hoisting from a conditionally-executed
//! sub-block — hoisting `r = a + b` from inside an `if` to preheader
//! means `r` is computed unconditionally instead of conditionally.
//! For pure non-trapping ops this is a perf trade-off (extra work
//! when the conditional path wouldn't have been taken), not a
//! correctness issue. For Loads of invariant pointers, the loaded
//! value is the same in either case and the pointer is constant for
//! the loop's lifetime, so the worst case is an extra memory read.
//! A future tightening could re-add the rayzor-style "block dominates
//! every exiting block" guard when we have callers that need it.
//!
//! ## Algorithm sketch
//!
//! 1. Build dominator tree + loop forest.
//! 2. For each loop (innermost first — propagates inner hoists to
//!    outer-loop visibility), determine the preheader: the unique
//!    predecessor of the header outside the loop body. Skip if
//!    absent or non-unique.
//! 3. Seed an *invariant set* with everything defined outside the
//!    loop body (any value whose defining block isn't in the loop).
//! 4. Iterate over instructions in the loop body. For each one
//!    whose result is not yet in the set and whose operands are all
//!    in the set and which is in our "safe to hoist" list, mark as
//!    invariant and queue for hoisting.
//! 5. Loop until no new invariants found. Each pass may unlock new
//!    candidates (transitive invariance).
//! 6. Move hoisted instructions to the END of the preheader's
//!    instruction list, immediately before its terminator. Order
//!    among hoisted instructions preserves their original block-
//!    visit order so operand definitions still come before uses.
//!
//! The hoist is a literal block-to-block move; it preserves SSA
//! because the result HirId is unchanged — every use that was
//! inside the loop now reads from a value defined in the preheader
//! (which dominates the loop body).

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, HirConstant, HirFunction, HirId, HirInstruction, HirTerminator, HirType,
};
use std::collections::{HashMap, HashSet};

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LicmStats {
    /// Number of instructions hoisted to a preheader.
    pub hoisted: usize,
    /// Number of loops considered (whether or not anything hoisted).
    pub loops_visited: usize,
    /// Number of loops skipped because no unique preheader exists.
    pub loops_skipped_no_preheader: usize,
}

/// Run LICM over `func`.
pub fn run(func: &mut HirFunction) -> LicmStats {
    // SSA construction doesn't reliably populate `block.successors` /
    // `block.predecessors` — different lowering paths write the
    // fields at different times, the optimisation passes that splice
    // the CFG don't always re-derive them, and our downstream
    // `DominatorTree` / `LoopForest` analyses both read directly off
    // those fields. Without this rebuild, LICM sees zero edges and
    // detects zero loops on every function — the pass becomes a
    // silent no-op. Rebuild the edges from terminators here so the
    // analyses are working from ground truth. Same shape as
    // `inline::rebuild_cfg_edges`; keeping a local copy avoids a
    // cross-module dep (LICM doesn't currently depend on `inline`).
    rebuild_cfg_edges(func);

    let dt = DominatorTree::new(func);
    let lf = LoopForest::detect(func, &dt);
    if lf.loops().is_empty() {
        return LicmStats::default();
    }

    let mut stats = LicmStats::default();

    // Innermost-first ordering — `LoopForest::loops()` already
    // returns smaller-body-first. Processing inner loops before
    // outer ones means an inner-loop hoist makes its result visible
    // to the outer-loop invariance check on the same run.
    for lp in lf.loops() {
        stats.loops_visited += 1;
        let preheader = match unique_outside_predecessor(func, lp) {
            Some(p) => p,
            None => {
                stats.loops_skipped_no_preheader += 1;
                continue;
            }
        };

        stats.hoisted += hoist_loop(func, lp, preheader);
    }

    stats
}

/// Module-level entry — runs LICM on every function in `module`.
pub fn run_module(module: &mut crate::hir::HirModule) -> LicmStats {
    let mut total = LicmStats::default();
    for func in module.functions.values_mut() {
        let s = run(func);
        total.hoisted += s.hoisted;
        total.loops_visited += s.loops_visited;
        total.loops_skipped_no_preheader += s.loops_skipped_no_preheader;
    }
    total
}

/// Find the header's unique outside-the-loop predecessor. Returns
/// `None` when there is zero or more than one — we don't synthesise
/// a preheader in this pass.
/// Re-derive `block.successors` / `block.predecessors` from each
/// block's terminator. Idempotent; safe to call before any analysis
/// that reads those fields.
fn rebuild_cfg_edges(func: &mut HirFunction) {
    let mut succ_map: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for (&id, block) in &func.blocks {
        let succs = match &block.terminator {
            HirTerminator::Branch { target } => vec![*target],
            HirTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            HirTerminator::Switch { default, cases, .. } => {
                let mut v = vec![*default];
                for (_, t) in cases {
                    v.push(*t);
                }
                v
            }
            HirTerminator::Invoke { normal, unwind, .. } => vec![*normal, *unwind],
            HirTerminator::PatternMatch { .. }
            | HirTerminator::Return { .. }
            | HirTerminator::Unreachable => vec![],
        };
        succ_map.insert(id, succs);
    }
    let mut pred_map: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for (&src, succs) in &succ_map {
        for &t in succs {
            pred_map.entry(t).or_default().push(src);
        }
    }
    for (id, block) in func.blocks.iter_mut() {
        block.successors = succ_map.remove(id).unwrap_or_default();
        block.predecessors = pred_map.remove(id).unwrap_or_default();
    }
}

fn unique_outside_predecessor(func: &HirFunction, lp: &NaturalLoop) -> Option<HirId> {
    let header = func.blocks.get(&lp.header)?;
    let outside: Vec<HirId> = header
        .predecessors
        .iter()
        .copied()
        .filter(|p| !lp.body.contains(p))
        .collect();
    if outside.len() == 1 {
        Some(outside[0])
    } else {
        None
    }
}

/// Hoist invariant instructions from `lp.body` into `preheader`.
/// Returns the number of instructions hoisted.
fn hoist_loop(func: &mut HirFunction, lp: &NaturalLoop, preheader: HirId) -> usize {
    // Seed invariant set with every value defined outside the
    // loop body. By "defined outside" we mean: produced by an
    // instruction whose containing block isn't in `lp.body`, or
    // produced as a Parameter / Global / Constant (none of those
    // live in any block).
    let mut invariant: HashSet<HirId> = HashSet::new();

    // Walk every value in the function; if its kind isn't an
    // Instruction it lives outside any block by construction
    // (Param / Constant / Global / Undef). Mark those invariant
    // upfront.
    for (id, val) in &func.values {
        match val.kind {
            crate::hir::HirValueKind::Constant(_)
            | crate::hir::HirValueKind::Parameter(_)
            | crate::hir::HirValueKind::Global(_)
            | crate::hir::HirValueKind::Undef => {
                invariant.insert(*id);
            }
            crate::hir::HirValueKind::Instruction => {
                // Defer — see if its defining block is outside the
                // loop. This is O(blocks * insts); the body sizes we
                // see don't make it interesting.
            }
        }
    }

    // For Instruction-kind values, also mark them invariant if the
    // block that produces them isn't part of the loop body.
    let mut value_block: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();
    for (block_id, block) in &func.blocks {
        for inst in &block.instructions {
            if let Some(res) = instruction_result(inst) {
                value_block.insert(res, *block_id);
            }
        }
        // Phi results are defined at the head of their block;
        // record so the operand-check can see "header phi" as
        // loop-internal even though it's a Phi not an Instruction.
        for phi in &block.phis {
            value_block.insert(phi.result, *block_id);
        }
    }
    for (val_id, block_id) in &value_block {
        if !lp.body.contains(block_id) {
            invariant.insert(*val_id);
        }
    }

    // Identity-phi propagation. A loop-header phi shaped as
    // `%p = phi T [%self, back-edge], [%init, preheader]` (every
    // non-self incoming equals %init AND %init is loop-invariant) is
    // a tautology — it always carries %init. Mark %p as invariant and
    // record the substitution `%p → %init` so any hoisted instruction
    // referencing %p gets rewritten to use %init (otherwise the
    // hoisted inst would land in the preheader and reference a SSA
    // value defined in the loop header — undefined at that point).
    // We don't delete the phi; it stays in the loop header so any in-
    // loop use of `%p` still works (the runtime phi value equals %init
    // either way).
    let mut identity_subst: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();
    for _ in 0..64 {
        let mut changed = false;
        for &block_id in &lp.body {
            let Some(block) = func.blocks.get(&block_id) else {
                continue;
            };
            for phi in &block.phis {
                if invariant.contains(&phi.result) {
                    continue;
                }
                let mut consensus: Option<HirId> = None;
                let mut all_match = true;
                for (val, _pred) in &phi.incoming {
                    if *val == phi.result {
                        continue;
                    }
                    if !invariant.contains(val) {
                        all_match = false;
                        break;
                    }
                    match consensus {
                        None => consensus = Some(*val),
                        Some(prev) if prev == *val => {}
                        _ => {
                            all_match = false;
                            break;
                        }
                    }
                }
                if all_match {
                    if let Some(repl) = consensus {
                        invariant.insert(phi.result);
                        // Chase prior substitutions so the recorded
                        // replacement always resolves to a value
                        // defined outside the loop.
                        let mut final_repl = repl;
                        while let Some(&next) = identity_subst.get(&final_repl) {
                            if next == final_repl {
                                break;
                            }
                            final_repl = next;
                        }
                        identity_subst.insert(phi.result, final_repl);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Pre-compute every Store's memory location in the loop body.
    // Used by the per-Load alias check below to decide whether a
    // candidate Load is safe to hoist. A Store's "location" is the
    // (root SSA pointer, byte offset) pair extracted by walking back
    // through GEP+Cast chains; if we can't trace the chain to a clean
    // (root, const_offset), the location is conservatively `None`
    // (treat as may-alias).
    let addr_index = build_addr_index(func);
    let store_locs: Vec<MemLoc> = lp
        .body
        .iter()
        .filter_map(|b| func.blocks.get(b))
        .flat_map(|b| b.instructions.iter())
        .filter_map(|inst| match inst {
            HirInstruction::Store { ptr, value, .. } => Some(extract_mem_loc(
                *ptr,
                value_byte_size(func, *value),
                &identity_subst,
                &addr_index,
            )),
            _ => None,
        })
        .collect();

    // Iterate-to-fixed-point: each pass may unlock new candidates
    // because a hoisted instruction's result becomes invariant for
    // the next pass.
    let mut total_hoisted = 0;
    for _ in 0..32 {
        let mut to_hoist: Vec<(HirId, HirInstruction)> = Vec::new();

        // Collect candidates from every block in the loop body.
        // Preserve original (block-iteration, position) order so
        // when we re-insert into the preheader, operand
        // definitions still precede their uses.
        for &block_id in &lp.body {
            let Some(block) = func.blocks.get(&block_id) else {
                continue;
            };
            for inst in &block.instructions {
                let Some(res) = instruction_result(inst) else {
                    continue;
                };
                if invariant.contains(&res) {
                    continue; // already invariant — was hoisted on
                              // a prior iteration or wasn't a body
                              // instruction.
                }
                if !is_safe_to_hoist(inst) {
                    continue;
                }
                if !operands_all_invariant(inst, &invariant) {
                    continue;
                }
                // Per-Load alias check: a Load with an invariant
                // pointer can only be safely hoisted if no Store in
                // the loop body might write to the same memory. We
                // walk back through GEP+Cast chains to extract the
                // (root, offset, size) tuple for the Load and for
                // every Store, then ask whether ranges may overlap.
                if let HirInstruction::Load { ptr, ty, .. } = inst {
                    let load_loc =
                        extract_mem_loc(*ptr, hir_ty_byte_size(ty), &identity_subst, &addr_index);
                    // A Load with an entirely opaque root (no GEP+Cast
                    // chain we can trace) can't be disambiguated from
                    // any in-loop Store. Skip it.
                    if load_loc.root.is_none() {
                        continue;
                    }
                    if store_locs.iter().any(|s| load_loc.may_alias(s)) {
                        continue;
                    }
                }
                invariant.insert(res);
                let mut cloned = inst.clone();
                apply_subst(&mut cloned, &identity_subst);
                to_hoist.push((block_id, cloned));
            }
        }

        if to_hoist.is_empty() {
            break;
        }

        // Remove hoisted instructions from their original blocks.
        let hoisted_ids: HashSet<HirId> = to_hoist
            .iter()
            .filter_map(|(_, inst)| instruction_result(inst))
            .collect();
        for (block_id, _) in &to_hoist {
            if let Some(block) = func.blocks.get_mut(block_id) {
                block
                    .instructions
                    .retain(|inst| match instruction_result(inst) {
                        Some(r) => !hoisted_ids.contains(&r),
                        None => true,
                    });
            }
        }

        // Insert into the preheader, before its terminator (which
        // for HIR is stored separately from `instructions`, so we
        // just append).
        if let Some(ph) = func.blocks.get_mut(&preheader) {
            for (_, inst) in &to_hoist {
                ph.instructions.push(inst.clone());
            }
        }

        total_hoisted += to_hoist.len();
    }

    total_hoisted
}

/// Is `inst` shape we consider safe to relocate? Excludes ops with
/// side effects or non-deterministic trap semantics.
fn is_safe_to_hoist(inst: &HirInstruction) -> bool {
    match inst {
        HirInstruction::Binary { op, .. } => match op {
            // Integer Div / Rem can trap on zero — hoisting them
            // out of a loop changes whether the trap fires per
            // iteration, so we leave them in place.
            BinaryOp::Div | BinaryOp::Rem => false,
            _ => true,
        },
        HirInstruction::Unary { .. } | HirInstruction::Cast { .. } => true,
        HirInstruction::GetElementPtr { .. } | HirInstruction::ExtractValue { .. } => true,
        HirInstruction::Select { .. } => true,
        // `Load` is shape-eligible — the actual hoist decision is
        // gated on a per-Load alias check in `hoist_loop` that
        // proves no Store in the loop body writes to overlapping
        // memory. Returning `true` here just says "the instruction
        // shape is hoistable in principle"; the caller does the
        // soundness check.
        HirInstruction::Load { .. } => true,
        _ => false,
    }
}

/// Are every operand referenced by `inst` already in `invariant`?
fn operands_all_invariant(inst: &HirInstruction, invariant: &HashSet<HirId>) -> bool {
    let mut all_in = true;
    let check = |id: HirId| invariant.contains(&id);
    match inst {
        HirInstruction::Binary { left, right, .. } => {
            all_in = check(*left) && check(*right);
        }
        HirInstruction::Unary { operand, .. } => all_in = check(*operand),
        HirInstruction::Cast { operand, .. } => all_in = check(*operand),
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            all_in = check(*ptr) && indices.iter().all(|i| check(*i));
        }
        HirInstruction::ExtractValue { aggregate, .. } => all_in = check(*aggregate),
        HirInstruction::Load { ptr, .. } => all_in = check(*ptr),
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            all_in = check(*condition) && check(*true_val) && check(*false_val);
        }
        _ => all_in = false,
    }
    all_in
}

/// Memory location of a Load/Store, used by the per-Load alias check.
///
/// `root` is the SSA value at the base of the GEP+Cast chain — for a
/// typical ref-class field access this is the body-pointer load/phi.
/// `offset` is the constant byte offset from that root (None if the
/// chain has a non-constant index along the way). `size` is the
/// byte width of the value being loaded/stored.
#[derive(Debug, Clone, Copy)]
struct MemLoc {
    root: Option<HirId>,
    offset: Option<u64>,
    size: u32,
}

impl MemLoc {
    /// May the regions described by `self` and `other` overlap?
    ///
    /// Returns `true` conservatively when we can't prove they don't.
    /// The fast win: when both sides have a *known root* and the
    /// roots differ, we rely on the SSA no-aliasing assumption
    /// (distinct pointer SSA values refer to distinct memory) — that
    /// case returns `false` even if one side has an unknown offset
    /// or unknown size, because the roots already establish disjoint
    /// allocations. Only when roots match (or one is unknown) do we
    /// fall back to byte-range comparison.
    fn may_alias(&self, other: &MemLoc) -> bool {
        match (self.root, other.root) {
            (Some(r1), Some(r2)) if r1 != r2 => {
                // Different SSA roots: rely on the SSA-level
                // no-aliasing assumption. Holds for the patterns ZynML
                // emits — each malloc / alloca produces a fresh root,
                // table reads through different indices produce
                // different SSA values, etc.
                false
            }
            (None, _) | (_, None) => {
                // At least one side's root is opaque; can't reason.
                true
            }
            _ => {
                // Same root. Need byte-range info.
                let (Some(o1), s1, Some(o2), s2) =
                    (self.offset, self.size, other.offset, other.size)
                else {
                    // Same root but at least one offset/size is
                    // unknown: must conservatively assume overlap.
                    return true;
                };
                if s1 == 0 || s2 == 0 {
                    return true;
                }
                let end1 = o1.saturating_add(s1 as u64);
                let end2 = o2.saturating_add(s2 as u64);
                !(end1 <= o2 || end2 <= o1)
            }
        }
    }
}

/// Compact representation of the GEP/Cast chain we need to chase.
/// Owning copies of operands/indices keep us from holding a borrow on
/// `func.blocks` while the alias check runs.
enum AddrLink {
    /// Bitcast — chase to the operand verbatim.
    Cast(HirId),
    /// GEP — chase to `base`, optionally with constant byte-offset
    /// contribution from the indices (`None` if any index was
    /// non-constant).
    Gep {
        base: HirId,
        const_offset: Option<u64>,
    },
}

/// Build a `HirId → AddrLink` lookup over every `Cast` / `GEP`
/// instruction in `func`. Built once per `hoist_loop` call and
/// shared across the per-Load and per-Store alias-check chases.
/// Without this index every chase step walked `func.blocks ×
/// instructions` looking for the defining instruction — for a
/// function with N instructions and K Loads we paid O(N²·K).
fn build_addr_index(func: &HirFunction) -> HashMap<HirId, AddrLink> {
    let mut idx: HashMap<HirId, AddrLink> = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            match inst {
                HirInstruction::Cast {
                    result, operand, ..
                } => {
                    idx.insert(*result, AddrLink::Cast(*operand));
                }
                HirInstruction::GetElementPtr {
                    result,
                    ptr: base,
                    indices,
                    ..
                } => {
                    let mut const_offset: Option<u64> = Some(0);
                    for &id in indices {
                        if let Some(off) = const_offset {
                            if let Some(v) = constant_i64_value(func, id) {
                                const_offset = Some(off.saturating_add(v as u64));
                            } else {
                                const_offset = None;
                            }
                        }
                    }
                    idx.insert(
                        *result,
                        AddrLink::Gep {
                            base: *base,
                            const_offset,
                        },
                    );
                }
                _ => {}
            }
        }
    }
    idx
}

/// Walk back through GEP+Cast chains starting at `ptr` to extract a
/// `MemLoc { root, offset, size }`. `size` comes from the caller
/// (byte width of the Load/Store's value type). The chain step
/// lookup is O(1) via the pre-built `addr_index`.
fn extract_mem_loc(
    ptr: HirId,
    size: u32,
    identity_subst: &indexmap::IndexMap<HirId, HirId>,
    addr_index: &HashMap<HirId, AddrLink>,
) -> MemLoc {
    let mut current = identity_subst.get(&ptr).copied().unwrap_or(ptr);
    let mut total_offset: u64 = 0;
    let mut offset_known = true;
    // Cap the chase depth so a malformed SSA graph can't lock up.
    for _ in 0..32 {
        match addr_index.get(&current) {
            Some(AddrLink::Cast(operand)) => {
                current = identity_subst.get(operand).copied().unwrap_or(*operand);
            }
            Some(AddrLink::Gep { base, const_offset }) => {
                match const_offset {
                    Some(v) => total_offset = total_offset.saturating_add(*v),
                    None => offset_known = false,
                }
                current = identity_subst.get(base).copied().unwrap_or(*base);
            }
            None => break,
        }
    }
    MemLoc {
        root: Some(current),
        offset: if offset_known {
            Some(total_offset)
        } else {
            None
        },
        size,
    }
}

/// Constant i64 value of `id` if its SSA value is a `Constant::I64`
/// (or a small-int constant promotable to i64). Returns `None` for
/// non-constants.
fn constant_i64_value(func: &HirFunction, id: HirId) -> Option<i64> {
    let v = func.values.get(&id)?;
    match &v.kind {
        crate::hir::HirValueKind::Constant(c) => match c {
            HirConstant::I8(x) => Some(*x as i64),
            HirConstant::I16(x) => Some(*x as i64),
            HirConstant::I32(x) => Some(*x as i64),
            HirConstant::I64(x) => Some(*x),
            HirConstant::U8(x) => Some(*x as i64),
            HirConstant::U16(x) => Some(*x as i64),
            HirConstant::U32(x) => Some(*x as i64),
            HirConstant::U64(x) => Some(*x as i64),
            _ => None,
        },
        _ => None,
    }
}

/// Byte size of an `HirType` for the purpose of alias-range checks.
/// Conservative — returns 0 for shapes we don't model, which makes
/// the alias check default to "may overlap".
fn hir_ty_byte_size(ty: &HirType) -> u32 {
    match ty {
        HirType::I8 | HirType::U8 | HirType::Bool => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Ptr(_) | HirType::Ref { .. } => 8,
        _ => 0,
    }
}

/// Byte size of the value at `id`, derived from its HIR type.
fn value_byte_size(func: &HirFunction, id: HirId) -> u32 {
    func.values
        .get(&id)
        .map(|v| hir_ty_byte_size(&v.ty))
        .unwrap_or(0)
}

/// Rewrite every operand of `inst` whose id matches a key in `subst`
/// to the mapped value. Used at hoist time to replace identity-phi
/// references with the phi's preheader incoming value, so the hoisted
/// instruction in the preheader doesn't reference a SSA value defined
/// in the loop header.
fn apply_subst(inst: &mut HirInstruction, subst: &indexmap::IndexMap<HirId, HirId>) {
    if subst.is_empty() {
        return;
    }
    let r = |id: &mut HirId| {
        if let Some(&v) = subst.get(id) {
            *id = v;
        }
    };
    match inst {
        HirInstruction::Binary { left, right, .. } => {
            r(left);
            r(right);
        }
        HirInstruction::Unary { operand, .. } => r(operand),
        HirInstruction::Cast { operand, .. } => r(operand),
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            r(ptr);
            for i in indices.iter_mut() {
                r(i);
            }
        }
        HirInstruction::ExtractValue { aggregate, .. } => r(aggregate),
        HirInstruction::InsertValue {
            aggregate, value, ..
        } => {
            r(aggregate);
            r(value);
        }
        HirInstruction::Load { ptr, .. } => r(ptr),
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            r(condition);
            r(true_val);
            r(false_val);
        }
        _ => {}
    }
}

fn instruction_result(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. } => Some(*result),
        HirInstruction::Unary { result, .. } => Some(*result),
        HirInstruction::Cast { result, .. } => Some(*result),
        HirInstruction::GetElementPtr { result, .. } => Some(*result),
        HirInstruction::ExtractValue { result, .. } => Some(*result),
        HirInstruction::InsertValue { result, .. } => Some(*result),
        HirInstruction::Load { result, .. } => Some(*result),
        HirInstruction::Alloca { result, .. } => Some(*result),
        HirInstruction::Call { result, .. } => *result,
        HirInstruction::IndirectCall { result, .. } => *result,
        HirInstruction::Select { result, .. } => Some(*result),
        HirInstruction::Atomic { result, .. } => Some(*result),
        _ => None,
    }
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirFunctionSignature, HirTerminator, HirType, HirValue, HirValueKind,
    };
    use zyntax_typed_ast::InternedString;

    fn empty_sig(ret: HirType) -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![ret],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        }
    }

    fn mk_func() -> (HirFunction, HirId, HirId, HirId, HirId) {
        // Build the canonical while-loop shape:
        //   entry → header → body → header (back edge)
        //              ↓
        //             exit
        let mut f = HirFunction::new(InternedString::new_global("t"), empty_sig(HirType::I32));
        let entry = HirId::new();
        let header = HirId::new();
        let body = HirId::new();
        let exit = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        for id in [entry, header, body, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        f.blocks.get_mut(&entry).unwrap().successors = vec![header];
        f.blocks.get_mut(&header).unwrap().predecessors = vec![entry, body];
        f.blocks.get_mut(&header).unwrap().successors = vec![body, exit];
        f.blocks.get_mut(&body).unwrap().predecessors = vec![header];
        f.blocks.get_mut(&body).unwrap().successors = vec![header];
        f.blocks.get_mut(&exit).unwrap().predecessors = vec![header];

        let cond_id = HirId::new();
        f.values.insert(
            cond_id,
            HirValue {
                id: cond_id,
                ty: HirType::Bool,
                kind: HirValueKind::Constant(crate::hir::HirConstant::Bool(true)),
                uses: Default::default(),
                span: None,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
        f.blocks.get_mut(&header).unwrap().terminator = HirTerminator::CondBranch {
            condition: cond_id,
            true_target: body,
            false_target: exit,
        };
        f.blocks.get_mut(&body).unwrap().terminator = HirTerminator::Branch { target: header };
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![] };

        (f, entry, header, body, exit)
    }

    fn add_param(f: &mut HirFunction, ty: HirType, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Parameter(idx),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn add_inst(f: &mut HirFunction, ty: HirType) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    #[test]
    fn hoists_invariant_add_into_preheader() {
        // body: r = a + b   ← a and b are params, both invariant
        let (mut f, entry, _header, body, _exit) = mk_func();
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r,
                ty: HirType::I32,
                left: a,
                right: b,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 1);
        assert!(f.blocks[&body].instructions.is_empty());
        assert_eq!(f.blocks[&entry].instructions.len(), 1);
    }

    #[test]
    fn does_not_hoist_when_operand_is_loop_internal() {
        // body: phi i = ...
        //       r   = i + 1    ← i is body-local (phi), not invariant
        let (mut f, _entry, header, body, _exit) = mk_func();
        let one = HirId::new();
        f.values.insert(
            one,
            HirValue {
                id: one,
                ty: HirType::I32,
                kind: HirValueKind::Constant(crate::hir::HirConstant::I32(1)),
                uses: Default::default(),
                span: None,
            },
        );
        // Put a Phi result in the header so it's defined in the
        // loop body — emulating the loop induction variable.
        let i = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&header)
            .unwrap()
            .phis
            .push(crate::hir::HirPhi {
                result: i,
                ty: HirType::I32,
                incoming: vec![],
            });
        let r = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r,
                ty: HirType::I32,
                left: i,
                right: one,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 0, "i depends on phi — must stay in loop");
    }

    #[test]
    fn does_not_hoist_div_even_if_invariant() {
        // body: r = a / b   ← a and b invariant, but Div may trap
        // and hoisting it changes whether trap fires per iteration.
        let (mut f, _entry, _header, body, _exit) = mk_func();
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let r = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Div,
                result: r,
                ty: HirType::I32,
                left: a,
                right: b,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 0);
        assert_eq!(f.blocks[&body].instructions.len(), 1);
    }

    #[test]
    fn chained_invariants_hoist_in_order() {
        // body: r1 = a + b
        //       r2 = r1 * c   ← becomes invariant after r1 hoist
        // Both should end up in the preheader, with r1 before r2.
        let (mut f, entry, _header, body, _exit) = mk_func();
        let a = add_param(&mut f, HirType::I32, 0);
        let b = add_param(&mut f, HirType::I32, 1);
        let c = add_param(&mut f, HirType::I32, 2);
        let r1 = add_inst(&mut f, HirType::I32);
        let r2 = add_inst(&mut f, HirType::I32);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r1,
                ty: HirType::I32,
                left: a,
                right: b,
            });
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Mul,
                result: r2,
                ty: HirType::I32,
                left: r1,
                right: c,
            });

        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 2);
        assert!(f.blocks[&body].instructions.is_empty());
        let entry_insts = &f.blocks[&entry].instructions;
        assert_eq!(entry_insts.len(), 2);
        // r1's add comes before r2's mul.
        let p_r1 = entry_insts
            .iter()
            .position(|i| matches!(i, HirInstruction::Binary { result, .. } if *result == r1))
            .unwrap();
        let p_r2 = entry_insts
            .iter()
            .position(|i| matches!(i, HirInstruction::Binary { result, .. } if *result == r2))
            .unwrap();
        assert!(p_r1 < p_r2, "operand def must precede use after hoisting");
    }

    #[test]
    fn function_with_no_loops_is_a_noop() {
        let mut f = HirFunction::new(InternedString::new_global("nop"), empty_sig(HirType::I32));
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![] };
        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 0);
        assert_eq!(stats.loops_visited, 0);
    }

    #[test]
    fn hoists_load_when_body_has_no_memory_effect() {
        // body: r = *p   ← p is an invariant param, body never stores
        // → Load should be hoisted into the preheader.
        let (mut f, entry, _header, body, _exit) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let r = add_inst(&mut f, HirType::I64);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Load {
                result: r,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            });
        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 1);
        assert!(f.blocks[&body].instructions.is_empty());
        assert_eq!(f.blocks[&entry].instructions.len(), 1);
    }

    #[test]
    fn does_not_hoist_load_when_body_stores_through_same_ptr() {
        // body: r = *p
        //       store v -> p
        //
        // The Load reads what the Store wrote on the previous
        // iteration — the per-Load alias check sees same root
        // (param `p`), same offset (0), same size (8) and refuses
        // to hoist. This is exactly the ref-class hot loop pattern
        // (`body.vx = body.vx - …`) the alias check exists to
        // protect.
        let (mut f, _entry, _header, body, _exit) = mk_func();
        let p = add_param(&mut f, HirType::Ptr(Box::new(HirType::I64)), 0);
        let v = add_param(&mut f, HirType::I64, 1);
        let r = add_inst(&mut f, HirType::I64);
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Load {
                result: r,
                ty: HirType::I64,
                ptr: p,
                align: 8,
                volatile: false,
            });
        f.blocks
            .get_mut(&body)
            .unwrap()
            .instructions
            .push(HirInstruction::Store {
                value: v,
                ptr: p,
                align: 8,
                volatile: false,
            });
        let stats = run(&mut f);
        assert_eq!(stats.hoisted, 0, "same-ptr Store blocks Load hoist");
    }
}
