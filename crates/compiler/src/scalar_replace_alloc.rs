//! Scalar replacement of non-escaping heap allocations.
//!
//! Catches the shape:
//!
//! ```text
//! ;; A heap object that never escapes the function — its pointer is
//! ;; only ever passed to GEP / Load / Store / (optionally) Free.
//! v_ptr = call Intrinsic::Malloc(size)         ;; *u8
//! v_gep = gep ptr u8, v_ptr, [const_off]
//! store v_x, v_gep                              ;; field write
//! v_v   = load i64, v_gep                       ;; field read (same field)
//! ...
//! call Intrinsic::Free(v_ptr)                  ;; optional free
//! ```
//!
//! For each non-escaping malloc, the pass forwards every `Load` from a
//! tracked GEP to the most-recent stored value at that field offset and
//! deletes the `Call(Intrinsic::Malloc)`, every tracked `GetElementPtr`,
//! every `Store` whose ptr was tracked, every `Load` whose ptr was
//! tracked, and any matching `Call(Intrinsic::Free)`. The fields become
//! SSA "scalar registers" — pre-Store, the field reads as `Undef`;
//! post-Store, as the stored value.
//!
//! This complements `aggregate_split`. `aggregate_split` works on
//! struct-typed `Load` round-trips that Cranelift's mem2reg already
//! handles for stack values but cannot reach when the slot is the
//! result of a `Call(Intrinsic::Malloc)`. This pass fills that gap.
//!
//! Conditions for safety:
//! * The `Call(Intrinsic::Malloc)` and all its tracked-pointer
//!   derivations (GEPs / pointer-typed Casts) live in a single block.
//! * Every tracked pointer is only used as: the `ptr` of a `Load`,
//!   the `ptr` of a `Store` (where the stored value is itself NOT a
//!   tracked pointer — otherwise the allocation escapes through
//!   another slot), the `arg` of `Call(Intrinsic::Free)`, the `ptr`
//!   of a `GetElementPtr`, or the operand of a pointer-typed `Cast`.
//! * GEP indices are constant integers (resolved from `HirConstant`),
//!   scaled to bytes by the GEP's `ty` the way the backends scale them.
//! * Every Load's pointer resolves to a known field offset for which
//!   we know the stored type.
//! * No instruction outside the home block references any tracked id
//!   (the malloc result, GEPs, casts). Phi-incomings and terminator
//!   operands in any other block also count.
//!
//! When all hold, the pass:
//! 1. For each field offset, mints a fresh `HirValue` of kind `Undef`
//!    typed to that field's payload type. This is the SSA "register"
//!    the field's storage becomes.
//! 2. Walks the home block in linear order tracking the current SSA
//!    value of each field (initialised to the Undef ids). On a tracked
//!    `Store`, advances the current value of that field to the stored
//!    value's id. On a tracked `Load`, records a substitution from
//!    the Load result to the current field id.
//! 3. Runs `replace_uses` across the whole function with the recorded
//!    map, then deletes the malloc call, every tracked GEP/Cast/Load/
//!    Store, and every matched Free.
//!
//! Risks (see project_sroa_target_was_wrong.md):
//! * Type-erasing casts on the malloc result (e.g. `Ptr<U8>` → `I64`)
//!   are the canonical escape — `Cast` to a non-pointer type aborts
//!   the candidate.
//! * Non-constant GEP indices abort the candidate (counted as
//!   `escapes_skipped`).
//! * Cross-block reachability via phi or unreachable blocks
//!   short-circuits to abort; v1 is intentionally single-block.

use crate::hir::{
    HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirType, HirValueKind,
    Intrinsic,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ScalarReplaceAllocStats {
    /// Number of `Call(Intrinsic::Malloc)` allocations the pass looked
    /// at as candidates (including ones it ultimately rejected).
    pub candidates_examined: usize,
    /// Number of `Call(Intrinsic::Malloc)` calls actually removed.
    pub mallocs_eliminated: usize,
    /// Number of matched `Call(Intrinsic::Free)` calls removed.
    pub frees_eliminated: usize,
    /// Number of candidates rejected because the malloc pointer was
    /// observed to escape (passed to a non-Free call, returned,
    /// stored into another slot, cast to non-pointer, used cross-block,
    /// etc.).
    pub escapes_skipped: usize,
}

impl ScalarReplaceAllocStats {
    fn combine(&mut self, other: ScalarReplaceAllocStats) {
        self.candidates_examined += other.candidates_examined;
        self.mallocs_eliminated += other.mallocs_eliminated;
        self.frees_eliminated += other.frees_eliminated;
        self.escapes_skipped += other.escapes_skipped;
    }
}

pub fn run_module(module: &mut HirModule) -> ScalarReplaceAllocStats {
    let mut total = ScalarReplaceAllocStats::default();
    if std::env::var("ZYNTAX_DISABLE_SCALAR_REPLACE_ALLOC").is_ok() {
        return total;
    }
    for func in module.functions.values_mut() {
        if func.is_external {
            continue;
        }
        total.combine(run_function(func));
    }
    // Optional debug surface: `ZYNTAX_SRA_DUMP=1` prints a one-line
    // summary when the pass touches any module. Off by default so it
    // doesn't litter compile output.
    if std::env::var("ZYNTAX_SRA_DUMP").is_ok() && total.candidates_examined > 0 {
        eprintln!(
            "scalar_replace_alloc: examined={} eliminated={} frees={} escapes={}",
            total.candidates_examined,
            total.mallocs_eliminated,
            total.frees_eliminated,
            total.escapes_skipped
        );
    }
    total
}

fn run_function(func: &mut HirFunction) -> ScalarReplaceAllocStats {
    let mut stats = ScalarReplaceAllocStats::default();
    let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();
    for bid in block_ids {
        stats.combine(run_block(func, bid));
    }
    stats
}

/// Run scalar-replace-alloc on one block. Each non-escaping malloc is
/// rewritten and removed; the sweep repeats until none remain.
fn run_block(func: &mut HirFunction, bid: HirId) -> ScalarReplaceAllocStats {
    let mut stats = ScalarReplaceAllocStats::default();
    // collect_candidates returns malloc-result HirIds in document order.
    // We rebuild on each iteration so positional indices stay valid
    // across mutations.
    loop {
        let candidates = collect_candidates(func, bid);
        if candidates.is_empty() {
            return stats;
        }
        let mut applied_one = false;
        for malloc_result in candidates {
            stats.candidates_examined += 1;
            match build_candidate(func, bid, malloc_result) {
                Some(c) => {
                    let (mallocs, frees) = apply_candidate(func, bid, &c);
                    stats.mallocs_eliminated += mallocs;
                    stats.frees_eliminated += frees;
                    applied_one = true;
                    // Restart sweep — block indices have shifted.
                    break;
                }
                None => {
                    stats.escapes_skipped += 1;
                }
            }
        }
        if !applied_one {
            return stats;
        }
    }
}

/// Find every `Call(Intrinsic::Malloc)` in the block and return its
/// result HirId.
fn collect_candidates(func: &HirFunction, bid: HirId) -> Vec<HirId> {
    let block = match func.blocks.get(&bid) {
        Some(b) => b,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for inst in &block.instructions {
        if let HirInstruction::Call {
            result: Some(r),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            ..
        } = inst
        {
            out.push(*r);
        }
    }
    out
}

/// All the per-candidate state we need to rewrite the block.
#[derive(Debug)]
struct Candidate {
    /// Instruction index of the `Call(Intrinsic::Malloc)`.
    malloc_iidx: usize,
    /// The HirId returned by the malloc — the seed of the tracked set.
    malloc_result: HirId,
    /// All pointer-typed HirIds derived (transitively) from malloc_result.
    tracked: HashSet<HirId>,
    /// `gep_id → constant byte-offset` for every tracked GEP.
    /// We canonicalise the offset to a single u64 key. For multi-index
    /// GEPs we sum the index values (treating them as raw byte offsets,
    /// matching how aggregate_split synthesises u8-typed GEPs).
    gep_field: HashMap<HirId, u64>,
    /// `byte_offset → field's value type`. Inferred from `Load.ty`
    /// (preferred) or fallen back to the type of stored values.
    field_ty: HashMap<u64, HirType>,
    /// Instruction indices of every `Call(Intrinsic::Free)` whose
    /// argument is in `tracked`.
    free_iidxs: Vec<usize>,
    /// Instruction indices of every `GetElementPtr` whose `ptr` is in
    /// `tracked` (including derived GEPs).
    gep_iidxs: Vec<usize>,
    /// Instruction indices of every pointer-typed `Cast` whose
    /// `operand` is in `tracked`.
    cast_iidxs: Vec<usize>,
    /// Instruction indices of every `Load` whose `ptr` is in `tracked`.
    load_iidxs: Vec<usize>,
    /// Instruction indices of every `Store` whose `ptr` is in `tracked`.
    store_iidxs: Vec<usize>,
    /// Linear, document-order list of `(byte_offset, stored_value_id)`
    /// pairs from tracked Stores. Used by the rewrite to drive the
    /// running "current field value" map.
    stores_linear: Vec<(usize, u64, HirId)>,
    /// Linear, document-order list of `(load_result, byte_offset)`
    /// pairs from tracked Loads.
    loads_linear: Vec<(usize, HirId, u64)>,
}

/// Build a Candidate for the given malloc_result, or return None if
/// it escapes / has unsupported shape.
fn build_candidate(func: &HirFunction, bid: HirId, malloc_result: HirId) -> Option<Candidate> {
    let block = func.blocks.get(&bid)?;

    // Locate the malloc instruction index.
    let malloc_iidx = block.instructions.iter().position(|i| {
        matches!(
            i,
            HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                ..
            } if *r == malloc_result
        )
    })?;

    // Block-scoped constant map for resolving GEP indices.
    // Keyed by HirId, value is the i64 const value. Only integer
    // constants are populated.
    let mut const_map: HashMap<HirId, i64> = HashMap::new();
    for (id, val) in &func.values {
        if let HirValueKind::Constant(c) = &val.kind {
            if let Some(n) = const_as_i64(c) {
                const_map.insert(*id, n);
            }
        }
    }

    let mut tracked: HashSet<HirId> = HashSet::new();
    tracked.insert(malloc_result);
    let mut gep_field: HashMap<HirId, u64> = HashMap::new();
    let mut gep_iidxs: Vec<usize> = Vec::new();
    let mut cast_iidxs: Vec<usize> = Vec::new();
    let mut load_iidxs: Vec<usize> = Vec::new();
    let mut store_iidxs: Vec<usize> = Vec::new();
    let mut free_iidxs: Vec<usize> = Vec::new();
    let mut field_ty: HashMap<u64, HirType> = HashMap::new();
    let mut stores_linear: Vec<(usize, u64, HirId)> = Vec::new();
    let mut loads_linear: Vec<(usize, HirId, u64)> = Vec::new();

    // Iterative fixed-point over the block. Single linear sweep
    // suffices for the well-formed cases since the tracked set is
    // populated in def-order; but we loop until quiescence to handle
    // out-of-order derivations defensively.
    loop {
        let before_len = tracked.len();
        for (idx, inst) in block.instructions.iter().enumerate() {
            match inst {
                HirInstruction::GetElementPtr {
                    result,
                    ty,
                    ptr,
                    indices,
                } if tracked.contains(ptr) => {
                    if tracked.contains(result) {
                        continue;
                    }
                    // Resolve indices to a constant byte offset using the
                    // same stride rule the backends apply: a GEP's `ty`
                    // names what one index step covers. `U8`/`I8` is the
                    // byte-offset form aggregate_split emits; a pointer or
                    // an array steps by its element's size.
                    let mut offset: i64 = 0;
                    let mut cur_ty = ty.clone();
                    let mut all_const = true;
                    for idx_id in indices {
                        let v = match const_map.get(idx_id) {
                            Some(v) => *v,
                            None => {
                                all_const = false;
                                break;
                            }
                        };
                        match &cur_ty {
                            HirType::U8 | HirType::I8 => offset += v,
                            HirType::Ptr(inner) => {
                                offset += v.saturating_mul(size_of_hir_ty(inner) as i64);
                                cur_ty = (**inner).clone();
                            }
                            HirType::Array(elem, _) => {
                                offset += v.saturating_mul(size_of_hir_ty(elem) as i64);
                                cur_ty = (**elem).clone();
                            }
                            // A struct-typed GEP steps to a field offset the
                            // backend reads from its own layout cache. This
                            // pass has no layout of its own, so it declines
                            // the candidate rather than guess one.
                            _ => return None,
                        }
                    }
                    if !all_const || offset < 0 {
                        return None;
                    }
                    // Inherit the source's field offset (a GEP off a
                    // tracked GEP adds offsets).
                    let base_offset = gep_field.get(ptr).copied().unwrap_or(0);
                    let total_offset = base_offset.saturating_add(offset as u64);
                    tracked.insert(*result);
                    gep_field.insert(*result, total_offset);
                    if !gep_iidxs.contains(&idx) {
                        gep_iidxs.push(idx);
                    }
                }
                HirInstruction::Cast {
                    result,
                    ty,
                    operand,
                    ..
                } if tracked.contains(operand) => {
                    if tracked.contains(result) {
                        continue;
                    }
                    // Cast to non-pointer type → the value escapes
                    // through an integer encoding (or similar). Bail.
                    if !matches!(ty, HirType::Ptr(_)) {
                        return None;
                    }
                    // Pointer-to-pointer cast inherits the source's
                    // field offset, if any.
                    let inherit_offset = gep_field.get(operand).copied();
                    tracked.insert(*result);
                    if let Some(off) = inherit_offset {
                        gep_field.insert(*result, off);
                    }
                    if !cast_iidxs.contains(&idx) {
                        cast_iidxs.push(idx);
                    }
                }
                _ => {}
            }
        }
        if tracked.len() == before_len {
            break;
        }
    }

    // Pass 2: walk the home block in order, classifying every
    // instruction's interaction with `tracked`. Any unsupported use
    // → abort.
    for (idx, inst) in block.instructions.iter().enumerate() {
        // Skip the malloc itself — its `result` is in tracked but
        // we want it removed at the rewrite step.
        if idx == malloc_iidx {
            continue;
        }
        match inst {
            HirInstruction::GetElementPtr { result, ptr, .. } if tracked.contains(ptr) => {
                // Already classified by fixpoint above. Sanity: the
                // result must be in tracked (or we'd have aborted on
                // non-const indices).
                if !tracked.contains(result) {
                    return None;
                }
            }
            HirInstruction::Cast {
                result, operand, ..
            } if tracked.contains(operand) => {
                if !tracked.contains(result) {
                    return None;
                }
            }
            HirInstruction::Load {
                result, ty, ptr, ..
            } if tracked.contains(ptr) => {
                let off = match gep_field.get(ptr).copied() {
                    Some(o) => o,
                    None => {
                        // Loading directly from the malloc base (no
                        // intermediate GEP) is offset 0.
                        if *ptr == malloc_result {
                            0
                        } else {
                            return None;
                        }
                    }
                };
                // Record the field type (Load.ty wins over any
                // previously-recorded Store type).
                field_ty.insert(off, ty.clone());
                loads_linear.push((idx, *result, off));
                if !load_iidxs.contains(&idx) {
                    load_iidxs.push(idx);
                }
            }
            HirInstruction::Store { value, ptr, .. } if tracked.contains(ptr) => {
                // Escape via the *value* slot would mean storing one
                // tracked ptr through another — abort.
                if tracked.contains(value) {
                    return None;
                }
                let off = match gep_field.get(ptr).copied() {
                    Some(o) => o,
                    None => {
                        if *ptr == malloc_result {
                            0
                        } else {
                            return None;
                        }
                    }
                };
                // Type-of-stored value, only used if no Load fixes the
                // field's type more precisely.
                if let Some(ty) = type_of(func, *value) {
                    field_ty.entry(off).or_insert(ty);
                }
                stores_linear.push((idx, off, *value));
                if !store_iidxs.contains(&idx) {
                    store_iidxs.push(idx);
                }
            }
            HirInstruction::Store { value, .. } if tracked.contains(value) => {
                // Storing a tracked ptr into some other slot escapes.
                return None;
            }
            HirInstruction::Call { callee, args, .. } => {
                let any_tracked = args.iter().any(|a| tracked.contains(a));
                if !any_tracked {
                    if let HirCallable::Indirect(v) = callee {
                        if tracked.contains(v) {
                            return None;
                        }
                    }
                    continue;
                }
                // The only Call that may consume a tracked pointer is
                // Intrinsic::Free with exactly one arg in tracked.
                match callee {
                    HirCallable::Intrinsic(Intrinsic::Free) => {
                        if args.len() != 1 || !tracked.contains(&args[0]) {
                            return None;
                        }
                        if !free_iidxs.contains(&idx) {
                            free_iidxs.push(idx);
                        }
                    }
                    _ => return None,
                }
            }
            other => {
                // Any other instruction touching a tracked id is an
                // escape (Return / IndirectCall / Atomic / CreateClosure
                // / Async* / Throw / ExtractValue / InsertValue …).
                let uses_tracked = other.operands().iter().any(|o| tracked.contains(o));
                if uses_tracked {
                    return None;
                }
            }
        }
    }

    // Terminator can't reference any tracked id.
    if term_uses_any(&block.terminator, &tracked) {
        return None;
    }

    // Phis IN this block — incoming for tracked is impossible (tracked
    // is post-def) but check defensively.
    for phi in &block.phis {
        for (val, _) in &phi.incoming {
            if tracked.contains(val) {
                return None;
            }
        }
    }

    // Cross-block escape sweep: any other block referencing any
    // tracked id (instructions / terminator / phi-incomings) aborts.
    for (other_bid, blk) in &func.blocks {
        if *other_bid == bid {
            continue;
        }
        for inst in &blk.instructions {
            let uses_tracked = inst.operands().iter().any(|o| tracked.contains(o));
            if uses_tracked {
                return None;
            }
            // Indirect call's callee isn't in operands() but still uses ids.
            if let HirInstruction::Call {
                callee: HirCallable::Indirect(v),
                ..
            } = inst
            {
                if tracked.contains(v) {
                    return None;
                }
            }
        }
        if term_uses_any(&blk.terminator, &tracked) {
            return None;
        }
        for phi in &blk.phis {
            for (val, _) in &phi.incoming {
                if tracked.contains(val) {
                    return None;
                }
            }
        }
    }

    // Every load's offset must have a field type (we look it up at
    // rewrite time).
    for (_, _, off) in &loads_linear {
        if !field_ty.contains_key(off) {
            return None;
        }
    }

    Some(Candidate {
        malloc_iidx,
        malloc_result,
        tracked,
        gep_field,
        field_ty,
        free_iidxs,
        gep_iidxs,
        cast_iidxs,
        load_iidxs,
        store_iidxs,
        stores_linear,
        loads_linear,
    })
}

/// Rewrite the block per the Candidate. Returns `(mallocs_removed,
/// frees_removed)`.
fn apply_candidate(func: &mut HirFunction, bid: HirId, c: &Candidate) -> (usize, usize) {
    // 1. Mint Undef registers per field offset.
    let mut undef_for_field: HashMap<u64, HirId> = HashMap::new();
    for (off, ty) in &c.field_ty {
        let undef_id = func.create_value(ty.clone(), HirValueKind::Undef);
        undef_for_field.insert(*off, undef_id);
    }

    // 2. Walk the block in document order, maintaining current_field
    //    (initialised to the Undef ids). On a tracked Store, advance
    //    the field's current value to the stored value. On a tracked
    //    Load, record load_result → current_field as a substitution.
    let mut current_field: HashMap<u64, HirId> = undef_for_field.clone();
    let mut replacements: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();

    // Walk by instruction index so we see Stores and Loads in order.
    let block = func.blocks.get(&bid).expect("home block missing");
    let store_by_idx: HashMap<usize, (u64, HirId)> = c
        .stores_linear
        .iter()
        .map(|(idx, off, val)| (*idx, (*off, *val)))
        .collect();
    let load_by_idx: HashMap<usize, (HirId, u64)> = c
        .loads_linear
        .iter()
        .map(|(idx, res, off)| (*idx, (*res, *off)))
        .collect();
    for idx in 0..block.instructions.len() {
        if let Some((off, val)) = store_by_idx.get(&idx) {
            current_field.insert(*off, *val);
        }
        if let Some((res, off)) = load_by_idx.get(&idx) {
            let cur = current_field
                .get(off)
                .copied()
                .expect("undef should have been seeded");
            replacements.insert(*res, cur);
        }
    }

    // 3. Apply substitutions across the whole function (instructions,
    //    terminators, phi-incomings).
    if !replacements.is_empty() {
        for blk in func.blocks.values_mut() {
            for inst in &mut blk.instructions {
                inst.replace_uses(&replacements);
            }
            blk.terminator.replace_uses(&replacements);
            for phi in &mut blk.phis {
                for (val, _) in &mut phi.incoming {
                    if let Some(&new) = replacements.get(val) {
                        *val = new;
                    }
                }
            }
        }
    }

    // 4. Build the set of instruction indices to delete:
    //    malloc + tracked GEPs + tracked Casts + tracked Loads +
    //    tracked Stores + matched Frees.
    let mut to_remove: HashSet<usize> = HashSet::new();
    to_remove.insert(c.malloc_iidx);
    for i in &c.gep_iidxs {
        to_remove.insert(*i);
    }
    for i in &c.cast_iidxs {
        to_remove.insert(*i);
    }
    for i in &c.load_iidxs {
        to_remove.insert(*i);
    }
    for i in &c.store_iidxs {
        to_remove.insert(*i);
    }
    for i in &c.free_iidxs {
        to_remove.insert(*i);
    }

    let block = func.blocks.get_mut(&bid).unwrap();
    let original = std::mem::take(&mut block.instructions);
    let mut new_insts: Vec<HirInstruction> = Vec::with_capacity(original.len());
    for (idx, inst) in original.into_iter().enumerate() {
        if to_remove.contains(&idx) {
            continue;
        }
        new_insts.push(inst);
    }
    block.instructions = new_insts;

    // 5. Drop orphaned values from the function table.
    //    - The malloc's own result.
    //    - Every tracked GEP/Cast/Load result (their defs are gone).
    //    - Every Load result we replaced (also gone via deletion).
    func.values.shift_remove(&c.malloc_result);
    for id in &c.tracked {
        if *id == c.malloc_result {
            continue;
        }
        func.values.shift_remove(id);
    }
    // Load results aren't in tracked (they're scalar values, not
    // pointers) — clean them up via the replacements map.
    for (orphan, _) in &replacements {
        func.values.shift_remove(orphan);
    }

    (1, c.free_iidxs.len())
}

// ─── helpers ───────────────────────────────────────────────────────

fn const_as_i64(c: &HirConstant) -> Option<i64> {
    match c {
        HirConstant::I8(v) => Some(*v as i64),
        HirConstant::I16(v) => Some(*v as i64),
        HirConstant::I32(v) => Some(*v as i64),
        HirConstant::I64(v) => Some(*v),
        HirConstant::U8(v) => Some(*v as i64),
        HirConstant::U16(v) => Some(*v as i64),
        HirConstant::U32(v) => Some(*v as i64),
        HirConstant::U64(v) => i64::try_from(*v).ok(),
        _ => None,
    }
}

fn type_of(func: &HirFunction, id: HirId) -> Option<HirType> {
    func.values.get(&id).map(|v| v.ty.clone())
}

fn term_uses_any(term: &crate::hir::HirTerminator, set: &HashSet<HirId>) -> bool {
    use crate::hir::HirTerminator;
    match term {
        HirTerminator::Return { values } => values.iter().any(|v| set.contains(v)),
        HirTerminator::CondBranch { condition, .. } => set.contains(condition),
        HirTerminator::Switch { value, .. } => set.contains(value),
        HirTerminator::Invoke { args, .. } => args.iter().any(|v| set.contains(v)),
        HirTerminator::PatternMatch { value, .. } => set.contains(value),
        HirTerminator::Branch { .. } | HirTerminator::Unreachable => false,
    }
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        HirBlock, HirCallable, HirFunctionSignature, HirModule, HirTerminator, HirValue,
    };
    use indexmap::IndexMap;
    use zyntax_typed_ast::{AstArena, InternedString};

    fn sig() -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        }
    }

    fn mk_func() -> (HirFunction, HirId) {
        let mut f = HirFunction::new(InternedString::new_global("t"), sig());
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        (f, entry)
    }

    fn add_const_i64(f: &mut HirFunction, v: i64) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Constant(HirConstant::I64(v)),
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    fn add_const_u64(f: &mut HirFunction, v: u64) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::U64,
                kind: HirValueKind::Constant(HirConstant::U64(v)),
                uses: HashSet::new(),
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
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    fn push(f: &mut HirFunction, entry: HirId, inst: HirInstruction) {
        f.blocks.get_mut(&entry).unwrap().instructions.push(inst);
    }

    fn empty_module(f: HirFunction) -> (HirModule, HirId) {
        let func_id = f.id;
        let mut module = HirModule {
            id: HirId::new(),
            name: InternedString::new_global("test_mod"),
            functions: IndexMap::new(),
            globals: IndexMap::new(),
            types: IndexMap::new(),
            imports: vec![],
            exports: vec![],
            version: 0,
            dependencies: HashSet::new(),
            effects: IndexMap::new(),
            handlers: IndexMap::new(),
        };
        module.functions.insert(func_id, f);
        (module, func_id)
    }

    fn count_malloc(module: &HirModule, func_id: HirId, entry: HirId) -> usize {
        let f = &module.functions[&func_id];
        let blk = &f.blocks[&entry];
        blk.instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                        ..
                    }
                )
            })
            .count()
    }

    fn count_free(module: &HirModule, func_id: HirId, entry: HirId) -> usize {
        let f = &module.functions[&func_id];
        let blk = &f.blocks[&entry];
        blk.instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    HirInstruction::Call {
                        callee: HirCallable::Intrinsic(Intrinsic::Free),
                        ..
                    }
                )
            })
            .count()
    }

    /// The pinned validation test — moved out of `aggregate_split` and
    /// no longer `#[ignore]` because this pass closes the gap.
    ///
    /// Shape built:
    /// ```text
    ///   ptr   = call Intrinsic::Malloc(size_u64)        ; *u8
    ///   off   = const i64 0
    ///   gep   = gep ptr u8, ptr, [off]
    ///   v42   = const i64 42
    ///   store v42, gep
    ///   load  = load i64, gep
    ///   return load
    /// ```
    /// scalar_replace_alloc should detect the non-escape and delete
    /// the Call(Intrinsic::Malloc), the GEP, the Store, and the Load.
    /// The Return's value should be forwarded to v42.
    #[test]
    fn eliminates_non_escaping_malloc() {
        let _arena = AstArena::new();

        let (mut f, entry) = mk_func();
        let size = add_const_u64(&mut f, 8);
        let malloc_ptr = add_inst(&mut f, HirType::Ptr(Box::new(HirType::U8)));
        let off = add_const_i64(&mut f, 0);
        let gep = add_inst(&mut f, HirType::Ptr(Box::new(HirType::U8)));
        let v42 = add_const_i64(&mut f, 42);
        let loaded = add_inst(&mut f, HirType::I64);

        push(
            &mut f,
            entry,
            HirInstruction::Call {
                result: Some(malloc_ptr),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                args: vec![size],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::GetElementPtr {
                result: gep,
                ty: HirType::U8,
                ptr: malloc_ptr,
                indices: vec![off],
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Store {
                value: v42,
                ptr: gep,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: loaded,
                ty: HirType::I64,
                ptr: gep,
                align: 8,
                volatile: false,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
            values: vec![loaded],
        };

        let (mut module, func_id) = empty_module(f);
        let stats = run_module(&mut module);

        assert_eq!(stats.mallocs_eliminated, 1, "malloc not eliminated");
        assert_eq!(stats.frees_eliminated, 0, "no Free in the input");
        assert_eq!(count_malloc(&module, func_id, entry), 0);

        // The Return must now reference v42 directly (the load result
        // was substituted out).
        let term = &module.functions[&func_id].blocks[&entry].terminator;
        match term {
            HirTerminator::Return { values } => {
                assert_eq!(values, &vec![v42]);
            }
            _ => panic!("expected Return"),
        }
    }

    /// Malloc with a paired `Intrinsic::Free` — both should be removed.
    #[test]
    fn eliminates_malloc_with_paired_free() {
        let _arena = AstArena::new();
        let (mut f, entry) = mk_func();
        let size = add_const_u64(&mut f, 8);
        let malloc_ptr = add_inst(&mut f, HirType::Ptr(Box::new(HirType::U8)));
        let off = add_const_i64(&mut f, 0);
        let gep = add_inst(&mut f, HirType::Ptr(Box::new(HirType::U8)));
        let v99 = add_const_i64(&mut f, 99);
        let loaded = add_inst(&mut f, HirType::I64);

        push(
            &mut f,
            entry,
            HirInstruction::Call {
                result: Some(malloc_ptr),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                args: vec![size],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::GetElementPtr {
                result: gep,
                ty: HirType::U8,
                ptr: malloc_ptr,
                indices: vec![off],
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Store {
                value: v99,
                ptr: gep,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Load {
                result: loaded,
                ty: HirType::I64,
                ptr: gep,
                align: 8,
                volatile: false,
            },
        );
        push(
            &mut f,
            entry,
            HirInstruction::Call {
                result: None,
                callee: HirCallable::Intrinsic(Intrinsic::Free),
                args: vec![malloc_ptr],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
            values: vec![loaded],
        };

        let (mut module, func_id) = empty_module(f);
        let stats = run_module(&mut module);

        assert_eq!(stats.mallocs_eliminated, 1);
        assert_eq!(stats.frees_eliminated, 1);
        assert_eq!(count_malloc(&module, func_id, entry), 0);
        assert_eq!(count_free(&module, func_id, entry), 0);
    }

    /// A malloc whose pointer is returned (escapes) — must be left
    /// alone, counted as `escapes_skipped`.
    #[test]
    fn leaves_escaping_malloc_alone() {
        let _arena = AstArena::new();
        let (mut f, entry) = mk_func();
        let size = add_const_u64(&mut f, 8);
        let malloc_ptr = add_inst(&mut f, HirType::Ptr(Box::new(HirType::U8)));

        push(
            &mut f,
            entry,
            HirInstruction::Call {
                result: Some(malloc_ptr),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                args: vec![size],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        // Sig is `returns: vec![HirType::I64]`; we abuse it here to
        // express "the malloc result feeds the terminator". The pass
        // shouldn't care about Return type-correctness — it only cares
        // that a tracked id reaches a terminator.
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
            values: vec![malloc_ptr],
        };

        let (mut module, func_id) = empty_module(f);
        let stats = run_module(&mut module);

        assert_eq!(stats.mallocs_eliminated, 0);
        assert_eq!(stats.escapes_skipped, 1);
        assert_eq!(count_malloc(&module, func_id, entry), 1);
    }
}

/// Byte size of an HIR type, matching `aggregate_split`'s layout.
fn size_of_hir_ty(ty: &HirType) -> usize {
    match ty {
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 | HirType::Ptr(_) => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => s.fields.iter().map(size_of_hir_ty).sum::<usize>().max(1),
        HirType::Array(elem, n) => size_of_hir_ty(elem).saturating_mul(*n as usize),
        _ => 8,
    }
}
