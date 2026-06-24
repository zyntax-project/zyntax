//! Aggregate-split: replace struct round-trips with direct field access.
//!
//! Catches the two shapes that come up in any kernel touching an
//! array-of-structs:
//!
//! ```text
//! ;; read-modify-write of one or more fields
//! v0 = load <Body>, ptr
//! v1 = extractvalue v0, [0]           ;; b.x
//! v2 = extractvalue v0, [3]           ;; b.vx
//! v3 = fadd v1, v2                    ;; new x
//! v4 = insertvalue v0, v3, [0]        ;; b with new x
//! store v4, ptr
//! ```
//!
//! ```text
//! ;; pure read of one or more fields
//! v0 = load <Body>, ptr
//! v1 = extractvalue v0, [0]           ;; b.x
//! v2 = extractvalue v0, [3]           ;; b.vx
//! ;; (v0 has no other use)
//! ```
//!
//! Both become direct GEP + scalar Load / Store at the field's byte
//! offset:
//!
//! ```text
//! ;; rewritten read-modify-write
//! v_xp = gep ptr, [byte_offset(0)]
//! v1   = load f64, v_xp               ;; field load
//! v_vp = gep ptr, [byte_offset(3)]
//! v2   = load f64, v_vp
//! v3   = fadd v1, v2
//! store v3, v_xp                      ;; field store of new x
//! ```
//!
//! That's the difference between `nbody`'s 50 000-iter inner loop
//! shuffling 56-byte structs through registers vs. touching 8 bytes
//! per field-update. Cranelift's JIT can't recover this on its own —
//! struct-typed `load` + `store` round-trips look unconditionally
//! necessary to the late-stage instcombine.
//!
//! Conditions for safety (per Load):
//! * Load and Store live in the same basic block.
//! * Every use of the loaded value is either an `ExtractValue` or
//!   an `InsertValue`, and every intermediate `InsertValue` has
//!   exactly one use feeding into the next link of the chain (or
//!   the terminating Store).
//! * No memory-side-effecting instruction (Store to a different
//!   ptr, Call, IndirectCall, Atomic, Fence, AsyncSaveSlot,
//!   CreateClosure) sits between the Load and the terminating
//!   Store.
//! * The terminating Store's `ptr` is exactly the same `HirId` as
//!   the Load's `ptr`. (A future extension could chase a `cse`-style
//!   canonical-ptr substitution map; the current intra-block
//!   guarantee is strong enough that pointer-derivation almost
//!   always matches verbatim.)
//!
//! When all four hold, the pass:
//! 1. Computes the byte offset of each field referenced by an
//!    ExtractValue or modified by an InsertValue, using
//!    `size_of_hir_ty` for cumulative struct layout (every ZynML
//!    struct is unpadded so the running sum is exact).
//! 2. For each ExtractValue, inserts `gep ptr, [offset]` + `load
//!    field_ty` *at the ExtractValue's position* and substitutes the
//!    ExtractValue result with the new Load result.
//! 3. For each (field_idx, new_value) pair pulled from the
//!    InsertValue chain, emits `gep ptr, [offset]` + `store
//!    new_value` *at the original Store's position* (preserving
//!    program order across multiple field updates).
//! 4. Marks the original Load, InsertValues, ExtractValues, and
//!    Store for removal.
//!
//! The intra-block / linear-chain restriction keeps the analysis
//! simple. Cross-block field promotion and forward-store-elimination
//! would need a proper aliasing / SSA dataflow framework; both are
//! the natural follow-up.

use crate::hir::{
    CastOp, HirBlock, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirStructType,
    HirType, HirValue, HirValueKind,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AggregateSplitStats {
    /// Number of struct-typed Loads whose round-trip was eliminated.
    pub round_trips_removed: usize,
    /// Number of field-level Load / Store pairs emitted in their place.
    pub field_accesses_emitted: usize,
    /// ExtractValues rewritten to direct field Loads (read-only
    /// shape, no store back).
    pub field_reads_only: usize,
}

impl AggregateSplitStats {
    fn combine(&mut self, other: AggregateSplitStats) {
        self.round_trips_removed += other.round_trips_removed;
        self.field_accesses_emitted += other.field_accesses_emitted;
        self.field_reads_only += other.field_reads_only;
    }
}

pub fn run_module(module: &mut HirModule) -> AggregateSplitStats {
    let mut total = AggregateSplitStats::default();
    if std::env::var("ZYNTAX_DISABLE_AGGREGATE_SPLIT").is_ok() {
        return total;
    }
    for func in module.functions.values_mut() {
        if func.is_external {
            continue;
        }
        total.combine(run_function(func));
    }
    total
}

fn run_function(func: &mut HirFunction) -> AggregateSplitStats {
    let mut stats = AggregateSplitStats::default();
    let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();
    for bid in block_ids {
        stats.combine(run_block(func, bid));
    }
    stats
}

/// Run aggregate-split on one block. Multiple Loads can each become
/// independent rewrite targets, so we sweep the block in document
/// order and apply the first successful rewrite, then restart the
/// sweep (HirIds remain stable so indexes shift but the rewrite
/// loop converges in O(loads · insts)).
fn run_block(func: &mut HirFunction, bid: HirId) -> AggregateSplitStats {
    let mut stats = AggregateSplitStats::default();
    loop {
        let plan = match build_plan(func, bid) {
            Some(p) => p,
            None => return stats,
        };
        apply_plan(func, bid, &plan);
        stats.round_trips_removed += if plan.has_store { 1 } else { 0 };
        stats.field_accesses_emitted += plan.field_reads.len() + plan.field_writes.len();
        if !plan.has_store {
            stats.field_reads_only += 1;
        }
    }
}

/// Description of a rewrite the block needs.
#[derive(Debug)]
struct SplitPlan {
    /// Index of the Load to remove.
    load_idx: usize,
    load_result: HirId,
    ptr: HirId,
    struct_ty: HirStructType,
    /// (extractvalue_idx_in_block, field_index) pairs the read path
    /// needs to be rewritten at.
    field_reads: Vec<(usize, u32)>,
    /// `field_index → new_value_to_store` plus the source insertvalue
    /// instructions to delete.
    field_writes: Vec<(u32, HirId)>,
    insertvalue_idxs: Vec<usize>,
    /// Whether there's a terminating Store to delete.
    has_store: bool,
    /// Position of the Store in the block (or 0 if none).
    store_idx: usize,
}

fn build_plan(func: &HirFunction, bid: HirId) -> Option<SplitPlan> {
    let block = func.blocks.get(&bid)?;

    // For every Load of struct type, check if the pattern holds.
    for (load_idx, inst) in block.instructions.iter().enumerate() {
        let (load_result, load_ptr, struct_ty) = match inst {
            HirInstruction::Load {
                result, ty, ptr, ..
            } => match ty {
                HirType::Struct(s) if !s.fields.is_empty() => (*result, *ptr, s.clone()),
                _ => continue,
            },
            _ => continue,
        };

        // Collect every instruction that uses `load_result` or any
        // derived InsertValue chain link. We also need to know which
        // (if any) Store consumes the chain's tail.
        if let Some(plan) =
            try_build_plan_for_load(block, load_idx, load_result, load_ptr, &struct_ty)
        {
            // Cross-block escape check: every chain value (load
            // result + every InsertValue result) must not be
            // referenced outside this block, and within this block
            // the only references are the ones already validated by
            // the linear-chain walk. The terminator may not consume
            // chain values; nor may any phi in a successor block.
            //
            // chain set: load_result + all insertvalue results
            let mut chain: HashSet<HirId> = HashSet::new();
            chain.insert(plan.load_result);
            for iv_idx in &plan.insertvalue_idxs {
                if let Some(HirInstruction::InsertValue { result, .. }) =
                    block.instructions.get(*iv_idx)
                {
                    chain.insert(*result);
                }
            }
            if !chain_safe_to_remove(func, bid, &chain, plan.store_idx, plan.has_store) {
                continue;
            }
            return Some(plan);
        }
    }
    None
}

/// Verify no chain value is referenced anywhere we won't be rewriting:
/// - terminator of this block
/// - phi incomings or instructions of any other block
/// - the same block after the terminating Store (if any)
fn chain_safe_to_remove(
    func: &HirFunction,
    home_bid: HirId,
    chain: &HashSet<HirId>,
    store_idx: usize,
    has_store: bool,
) -> bool {
    let home = match func.blocks.get(&home_bid) {
        Some(b) => b,
        None => return false,
    };
    // Terminator of the home block.
    if terminator_uses_any(&home.terminator, chain) {
        return false;
    }
    // Anything in home block after the terminating store.
    if has_store {
        for inst in home.instructions.iter().skip(store_idx + 1) {
            if uses_any(inst, chain) {
                return false;
            }
        }
    }
    // Every other block: instructions, phis, terminator.
    for (bid, blk) in &func.blocks {
        if *bid == home_bid {
            continue;
        }
        for inst in &blk.instructions {
            if uses_any(inst, chain) {
                return false;
            }
        }
        for phi in &blk.phis {
            for (val, _) in &phi.incoming {
                if chain.contains(val) {
                    return false;
                }
            }
        }
        if terminator_uses_any(&blk.terminator, chain) {
            return false;
        }
    }
    true
}

fn terminator_uses_any(term: &crate::hir::HirTerminator, chain: &HashSet<HirId>) -> bool {
    use crate::hir::HirTerminator;
    match term {
        HirTerminator::Return { values } => values.iter().any(|v| chain.contains(v)),
        HirTerminator::CondBranch { condition, .. } => chain.contains(condition),
        HirTerminator::Switch { value, .. } => chain.contains(value),
        _ => false,
    }
}

fn try_build_plan_for_load(
    block: &HirBlock,
    load_idx: usize,
    load_result: HirId,
    load_ptr: HirId,
    struct_ty: &HirStructType,
) -> Option<SplitPlan> {
    // Walk the block from `load_idx + 1` collecting:
    //   * ExtractValue(chain, [i]) uses — read the field directly
    //   * InsertValue(chain, val, [i]) — produces next chain link
    //   * Store(chain_tail, ptr) — terminator
    //
    // `chain` IDs known so far: starts as {load_result}; each
    // InsertValue extends it with its result HirId. ExtractValue
    // result IDs are *consumers* not chain links.
    let mut chain: HashSet<HirId> = HashSet::new();
    chain.insert(load_result);

    // (block_index, field_idx). Order matters for re-insertion.
    let mut field_reads: Vec<(usize, u32)> = Vec::new();
    let mut insert_links: Vec<(usize, u32, HirId, HirId)> = Vec::new();
    // (field_idx, new_value, insertvalue_inst_idx, insertvalue_result)
    let mut current_tail: HirId = load_result;
    let mut store_position: Option<usize> = None;

    for (idx, inst) in block.instructions.iter().enumerate().skip(load_idx + 1) {
        match inst {
            HirInstruction::ExtractValue {
                result,
                aggregate,
                indices,
                ..
            } => {
                if chain.contains(aggregate) {
                    if indices.len() != 1 {
                        return None;
                    }
                    let field_idx = indices[0];
                    if (field_idx as usize) >= struct_ty.fields.len() {
                        return None;
                    }
                    field_reads.push((idx, field_idx));
                    // ExtractValue result MUST NOT itself be in the
                    // chain — record but don't mark as chain.
                    let _ = result;
                } else if uses_any(inst, &chain) {
                    // Some other use we don't handle.
                    return None;
                }
            }
            HirInstruction::InsertValue {
                result,
                aggregate,
                value,
                indices,
                ..
            } => {
                if *aggregate != current_tail {
                    // We support only the linear chain shape: each
                    // InsertValue's aggregate is the previous tail.
                    // Branches off the chain require a multi-tail
                    // analysis we haven't written.
                    if chain.contains(aggregate) {
                        return None;
                    }
                    if uses_any(inst, &chain) {
                        return None;
                    }
                    continue;
                }
                if indices.len() != 1 {
                    return None;
                }
                let field_idx = indices[0];
                if (field_idx as usize) >= struct_ty.fields.len() {
                    return None;
                }
                insert_links.push((idx, field_idx, *value, *result));
                chain.insert(*result);
                current_tail = *result;
            }
            HirInstruction::Store { value, ptr, .. } => {
                if *value != current_tail {
                    // Storing some unrelated value while the chain
                    // is live is fine as long as it doesn't write to
                    // our `load_ptr`. The same-ptr case is the only
                    // memory-killing one for our load.
                    if *ptr == load_ptr {
                        // Mid-chain write to the same ptr — bail.
                        return None;
                    }
                    continue;
                }
                if *ptr != load_ptr {
                    // Storing the chain tail somewhere else means we
                    // can't fuse with this load.
                    return None;
                }
                store_position = Some(idx);
                break;
            }
            // Anything that could read the load result indirectly
            // bail. We're conservative about the "use" set: if it
            // touches anything in `chain`, it's not part of the
            // pattern we can rewrite.
            other => {
                if uses_any(other, &chain) {
                    return None;
                }
                // Memory-side-effecting non-chain instructions kill
                // the load's memory dependency. The Store, Call,
                // etc. that follows our load could overwrite the
                // memory location before our terminating store —
                // bail to stay correct.
                if has_memory_side_effect(other) {
                    return None;
                }
            }
        }
    }

    // Validate: every InsertValue result has its only use either be
    // the next InsertValue or the terminator Store. We tracked this
    // by checking `aggregate == current_tail` at construction, so the
    // chain is already linear. The remaining concern is that *some*
    // instruction outside this block reads chain values, but the
    // intra-block guarantee handles that too — cross-block uses
    // would only matter if the value escaped this block, which we
    // forbid by requiring the Store to consume the tail (or none).
    //
    // If no Store is found and there are field reads, this is the
    // pure-read pattern: rewrite the Loads, drop the original Load
    // if it has no other use.

    let has_store = store_position.is_some();
    if !has_store && field_reads.is_empty() {
        // Nothing to rewrite for this load.
        return None;
    }
    if has_store && insert_links.is_empty() {
        // Store of the original Load result with no modifications —
        // identity copy. Pointless but not wrong; let some other
        // pass clean it up to keep this pass focused on the
        // field-update shape.
        return None;
    }

    let store_idx = store_position.unwrap_or(0);
    let field_writes: Vec<(u32, HirId)> = insert_links
        .iter()
        .map(|(_, idx, val, _)| (*idx, *val))
        .collect();
    let insertvalue_idxs: Vec<usize> = insert_links.iter().map(|(idx, _, _, _)| *idx).collect();

    Some(SplitPlan {
        load_idx,
        load_result,
        ptr: load_ptr,
        struct_ty: struct_ty.clone(),
        field_reads,
        field_writes,
        insertvalue_idxs,
        has_store,
        store_idx,
    })
}

fn apply_plan(func: &mut HirFunction, bid: HirId, plan: &SplitPlan) {
    // Pre-compute per-field byte offsets.
    let offsets = field_byte_offsets(&plan.struct_ty);

    // 1. Synthesise new HirValues / instructions, then splice them
    // into the block at the appropriate indices.
    let mut replacements: HashMap<HirId, HirId> = HashMap::new();
    let mut new_value_decls: Vec<(HirId, HirValue)> = Vec::new();
    let mut new_extract_insts: Vec<(usize, Vec<HirInstruction>)> = Vec::new();
    let mut new_store_insts: Vec<HirInstruction> = Vec::new();

    let ptr_ty_for_field = |field_ty: &HirType| HirType::Ptr(Box::new(field_ty.clone()));

    // Re-write each ExtractValue site.
    for (idx, field_idx) in &plan.field_reads {
        let field_ty = plan.struct_ty.fields[*field_idx as usize].clone();
        let offset = offsets[*field_idx as usize] as i64;

        // Offset constant.
        let offset_id = HirId::new();
        new_value_decls.push((
            offset_id,
            HirValue {
                id: offset_id,
                ty: HirType::I64,
                kind: HirValueKind::Constant(HirConstant::I64(offset)),
                uses: HashSet::new(),
                span: None,
            },
        ));

        // GEP to byte offset.
        let gep_id = HirId::new();
        let gep_ty = ptr_ty_for_field(&HirType::U8);
        new_value_decls.push((
            gep_id,
            HirValue {
                id: gep_id,
                ty: gep_ty,
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        ));

        // Cast u8 ptr → field-typed ptr.
        let field_ptr_id = HirId::new();
        new_value_decls.push((
            field_ptr_id,
            HirValue {
                id: field_ptr_id,
                ty: ptr_ty_for_field(&field_ty),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        ));

        // The replacement load.
        let new_load_id = HirId::new();
        new_value_decls.push((
            new_load_id,
            HirValue {
                id: new_load_id,
                ty: field_ty.clone(),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        ));

        let gep_inst = HirInstruction::GetElementPtr {
            result: gep_id,
            ty: HirType::U8,
            ptr: plan.ptr,
            indices: vec![offset_id],
        };
        let cast_inst = HirInstruction::Cast {
            result: field_ptr_id,
            op: CastOp::Bitcast,
            ty: ptr_ty_for_field(&field_ty),
            operand: gep_id,
        };
        let load_inst = HirInstruction::Load {
            result: new_load_id,
            ty: field_ty,
            ptr: field_ptr_id,
            align: 8,
            volatile: false,
        };

        new_extract_insts.push((*idx, vec![gep_inst, cast_inst, load_inst]));

        // Record substitution from the OLD ExtractValue result HirId
        // to the new field-Load result.
        // We need to find the ExtractValue's result HirId. The plan
        // recorded `(idx, field_idx)`; pull the result from the
        // block.
        let block = func.blocks.get(&bid).unwrap();
        if let Some(HirInstruction::ExtractValue { result, .. }) = block.instructions.get(*idx) {
            replacements.insert(*result, new_load_id);
        }
    }

    // Build the field-store sequence for the rewrite, only if there's
    // a store.
    if plan.has_store {
        for (field_idx, new_value) in &plan.field_writes {
            let field_ty = plan.struct_ty.fields[*field_idx as usize].clone();
            let offset = offsets[*field_idx as usize] as i64;

            let offset_id = HirId::new();
            new_value_decls.push((
                offset_id,
                HirValue {
                    id: offset_id,
                    ty: HirType::I64,
                    kind: HirValueKind::Constant(HirConstant::I64(offset)),
                    uses: HashSet::new(),
                    span: None,
                },
            ));

            let gep_id = HirId::new();
            new_value_decls.push((
                gep_id,
                HirValue {
                    id: gep_id,
                    ty: ptr_ty_for_field(&HirType::U8),
                    kind: HirValueKind::Instruction,
                    uses: HashSet::new(),
                    span: None,
                },
            ));

            let field_ptr_id = HirId::new();
            new_value_decls.push((
                field_ptr_id,
                HirValue {
                    id: field_ptr_id,
                    ty: ptr_ty_for_field(&field_ty),
                    kind: HirValueKind::Instruction,
                    uses: HashSet::new(),
                    span: None,
                },
            ));

            new_store_insts.push(HirInstruction::GetElementPtr {
                result: gep_id,
                ty: HirType::U8,
                ptr: plan.ptr,
                indices: vec![offset_id],
            });
            new_store_insts.push(HirInstruction::Cast {
                result: field_ptr_id,
                op: CastOp::Bitcast,
                ty: ptr_ty_for_field(&field_ty),
                operand: gep_id,
            });
            new_store_insts.push(HirInstruction::Store {
                value: *new_value,
                ptr: field_ptr_id,
                align: 8,
                volatile: false,
            });
        }
    }

    // Apply all value declarations.
    for (id, val) in new_value_decls {
        func.values.insert(id, val);
    }

    // Now splice into the block. Process from highest index downward
    // so earlier indexes don't shift.
    let mut to_remove: Vec<usize> = Vec::new();
    to_remove.push(plan.load_idx);
    for idx in &plan.insertvalue_idxs {
        to_remove.push(*idx);
    }
    for (idx, _) in &plan.field_reads {
        to_remove.push(*idx);
    }
    if plan.has_store {
        to_remove.push(plan.store_idx);
    }
    to_remove.sort_unstable();
    to_remove.dedup();

    // Build a fresh instruction list: walk the block, drop removed
    // indices, insert new instructions at the right positions.
    let block = func.blocks.get_mut(&bid).unwrap();
    let original = std::mem::take(&mut block.instructions);
    let mut new_insts: Vec<HirInstruction> = Vec::with_capacity(original.len() + 16);

    let mut extract_replacement_map: HashMap<usize, Vec<HirInstruction>> = HashMap::new();
    for (idx, insts) in new_extract_insts {
        extract_replacement_map.insert(idx, insts);
    }

    let store_idx_for_writes = plan.store_idx;

    for (idx, inst) in original.into_iter().enumerate() {
        if to_remove.contains(&idx) {
            // If this slot was an ExtractValue we substituted, splice
            // its replacement instructions here.
            if let Some(replacement) = extract_replacement_map.remove(&idx) {
                new_insts.extend(replacement);
            }
            // If this slot was the terminating Store, emit the
            // field-store sequence here.
            if plan.has_store && idx == store_idx_for_writes {
                new_insts.extend(new_store_insts.drain(..));
            }
            continue;
        }
        new_insts.push(inst);
    }
    block.instructions = new_insts;

    // Apply the result-substitution (old ExtractValue results →
    // new field-Load results) across the whole function.
    if !replacements.is_empty() {
        let map: indexmap::IndexMap<HirId, HirId> = replacements.into_iter().collect();
        for blk in func.blocks.values_mut() {
            for inst in &mut blk.instructions {
                inst.replace_uses(&map);
            }
            blk.terminator.replace_uses(&map);
            for phi in &mut blk.phis {
                for (val, _) in &mut phi.incoming {
                    if let Some(&new) = map.get(val) {
                        *val = new;
                    }
                }
            }
        }
        // Drop the orphan values from the function table.
        for (orphan, _) in &map {
            func.values.shift_remove(orphan);
        }
    }

    // Drop the (now-removed) Load result, InsertValue results from
    // the function's value table — they're orphaned.
    func.values.shift_remove(&plan.load_result);
    let block = func.blocks.get(&bid).unwrap();
    let _ = block; // shadowed earlier, just keeping the var
}

/// Byte offsets of each field within an unpadded struct.
fn field_byte_offsets(s: &HirStructType) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(s.fields.len());
    let mut running = 0usize;
    for f in &s.fields {
        offsets.push(running);
        running += size_of_hir_ty(f);
    }
    offsets
}

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

fn uses_any(inst: &HirInstruction, chain: &HashSet<HirId>) -> bool {
    let mut any = false;
    let chk = |id: &HirId| -> bool { chain.contains(id) };
    match inst {
        HirInstruction::Binary { left, right, .. } => any = chk(left) || chk(right),
        HirInstruction::Unary { operand, .. } => any = chk(operand),
        HirInstruction::Cast { operand, .. } => any = chk(operand),
        HirInstruction::Load { ptr, .. } => any = chk(ptr),
        HirInstruction::Store { value, ptr, .. } => any = chk(value) || chk(ptr),
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            any = chk(ptr) || indices.iter().any(chk);
        }
        HirInstruction::ExtractValue { aggregate, .. } => any = chk(aggregate),
        HirInstruction::InsertValue {
            aggregate, value, ..
        } => any = chk(aggregate) || chk(value),
        HirInstruction::Call { callee, args, .. } => {
            if let crate::hir::HirCallable::Indirect(v) = callee {
                any = chk(v);
            }
            any |= args.iter().any(chk);
        }
        HirInstruction::IndirectCall { func_ptr, args, .. } => {
            any = chk(func_ptr) || args.iter().any(chk);
        }
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => any = chk(condition) || chk(true_val) || chk(false_val),
        _ => {}
    }
    any
}

fn has_memory_side_effect(inst: &HirInstruction) -> bool {
    matches!(
        inst,
        HirInstruction::Store { .. }
            | HirInstruction::Call { .. }
            | HirInstruction::IndirectCall { .. }
            | HirInstruction::Atomic { .. }
            | HirInstruction::Fence { .. }
            | HirInstruction::AsyncSaveSlot { .. }
            | HirInstruction::CreateClosure { .. }
    )
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        HirBlock, HirCallable, HirFunctionSignature, HirModule, HirTerminator, Intrinsic,
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

    // The pinned `eliminates_non_escaping_malloc` test that previously
    // lived here (with `#[ignore]`) has moved to
    // `scalar_replace_alloc::tests::eliminates_non_escaping_malloc`
    // and no longer needs to be ignored: the new pass closes the gap.
}
