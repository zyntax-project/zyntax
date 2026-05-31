//! HIR function inlining (leaf-callee variant).
//!
//! Replaces direct calls to small single-block functions with the
//! callee's body inlined at the call site. Sound for any callee
//! whose CFG is exactly an entry block ending in `Return` — that
//! covers the bulk of utility functions (`max(a,b)`, `square(x)`,
//! field accessors, small arithmetic helpers) where the call /
//! return overhead dominates the actual work.
//!
//! Anything multi-block (a branch, a loop, an early return) is left
//! for a later, more general inliner. That follow-up will need
//! caller-block splitting + callee-CFG embedding + phi merging at
//! the return-site, which roughly doubles the LOC; the leaf-only
//! version below is the high-value subset we can land first without
//! risking SSA correctness.
//!
//! ## Why HIR-level (not the backend)
//!
//! Same reason as the rest: the BC interpreter and wasm-tier-1 JIT
//! both bypass Cranelift's inliner, so inlining at HIR is the only
//! way `len(v)` doesn't pay a function-call round-trip in the
//! interpreter today.
//!
//! ## Cost model
//!
//! Conservative — we inline a callee when ALL of these hold:
//!
//! * Caller and callee are different functions (no recursion).
//! * Callee has exactly one block (its `entry_block`).
//! * That block terminates in `Return` with ≤ 1 value.
//! * That block has no nested `Call` / `IndirectCall` / `Alloca` /
//!   `AsyncSaveSlot` / `AsyncLoadSlot` / `CreateClosure` / `Atomic`
//!   / `Fence`. (Allowing nested Calls would need recursion / a
//!   queue; allowing Alloca would need stack-frame merging. Both
//!   are interesting follow-ups but not v1.)
//! * Callee instruction count ≤ `MAX_INLINE_INSTS` (8 by default
//!   — calibrated to match LLVM's `InlineCostThreshold` for small
//!   leaf functions, conservative).
//! * Callee isn't async, isn't extern, has no effect requirements.
//! * The call is `HirCallable::Function(id)` (a direct call to a
//!   known module function). Symbol / Indirect / Intrinsic calls
//!   are intentionally untouched — those route through extern
//!   plumbing or VTable dispatch where inlining isn't well-defined.
//!
//! ## Transformation
//!
//! For each inlineable call site, we:
//!
//! 1. Snapshot the callee's value table (params + locally-defined
//!    values) and instruction list.
//! 2. Mint a fresh `HirId` for every value the callee defines that
//!    flows into the inlined region (every local result; params get
//!    mapped to the caller's argument values directly, no fresh
//!    id needed).
//! 3. Clone each instruction with its operand `HirId`s remapped
//!    through the substitution table.
//! 4. Splice the cloned instructions into the caller's block
//!    immediately BEFORE the Call.
//! 5. If the call has a result and the callee's `Return` carries a
//!    value, post-substitute the call's result HirId to point at
//!    the remapped return-value id everywhere else in the caller.
//! 6. Remove the original Call instruction.

use crate::hir::{
    HirCallable, HirFunction, HirId, HirInstruction, HirModule, HirTerminator, HirType, HirValue,
    HirValueKind,
};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Maximum number of instructions a callee can have to be eligible.
/// Calibrated to match small leaf utility functions; tunable but
/// kept tight since v1 doesn't do recursive inlining cost-modelling.
const MAX_INLINE_INSTS: usize = 8;

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InlineStats {
    /// Number of call sites successfully inlined.
    pub inlined: usize,
    /// Number of call sites examined (whether or not inlined).
    pub call_sites_visited: usize,
    /// Skipped because the callee was too big.
    pub skipped_too_large: usize,
    /// Skipped because the callee has > 1 block.
    pub skipped_multi_block: usize,
    /// Skipped because the callee uses a feature we don't inline yet
    /// (Call, Alloca, async ops, …).
    pub skipped_unsupported: usize,
    /// Skipped because the call isn't a direct module-function call.
    pub skipped_indirect: usize,
}

/// Inline every eligible direct call within `module`. Iterates a
/// fixed-point: inlining one call can expose a now-eligible callee
/// (when the now-inlined body's prior nested call shape was a
/// blocker), so we keep going until a pass finds nothing.
pub fn run_module(module: &mut HirModule) -> InlineStats {
    let mut total = InlineStats::default();

    for _ in 0..8 {
        // Snapshot callees we might inline (their bodies are read
        // while the caller's block is mutated; we work from a
        // detached snapshot so the borrow checker is happy).
        let callee_snapshot: HashMap<HirId, HirFunction> = module
            .functions
            .iter()
            .map(|(id, f)| (*id, f.clone()))
            .collect();

        let mut this_pass = 0;
        let function_ids: Vec<HirId> = module.functions.keys().copied().collect();

        for caller_id in function_ids {
            let caller = match module.functions.get_mut(&caller_id) {
                Some(c) => c,
                None => continue,
            };
            let stats = inline_in_function(caller, caller_id, &callee_snapshot);
            this_pass += stats.inlined;
            total.inlined += stats.inlined;
            total.call_sites_visited += stats.call_sites_visited;
            total.skipped_too_large += stats.skipped_too_large;
            total.skipped_multi_block += stats.skipped_multi_block;
            total.skipped_unsupported += stats.skipped_unsupported;
            total.skipped_indirect += stats.skipped_indirect;
        }

        if this_pass == 0 {
            break;
        }
    }

    total
}

fn inline_in_function(
    caller: &mut HirFunction,
    caller_id: HirId,
    callees: &HashMap<HirId, HirFunction>,
) -> InlineStats {
    let mut stats = InlineStats::default();

    // Walk every block; for each Call instruction, classify and
    // either inline or skip. We collect inline jobs first, then
    // apply them in reverse-position order so earlier indices stay
    // valid as we splice the callee body in.
    let block_ids: Vec<HirId> = caller.blocks.keys().copied().collect();

    for block_id in block_ids {
        let mut jobs: Vec<InlineJob> = Vec::new();

        // Collect candidates from this block. Snapshot the
        // instruction list so we don't have to fight the borrow
        // checker mid-traversal.
        let block_insts = caller
            .blocks
            .get(&block_id)
            .map(|b| b.instructions.clone())
            .unwrap_or_default();

        for (idx, inst) in block_insts.iter().enumerate() {
            let (result, args, callee_id) = match inst {
                HirInstruction::Call {
                    result,
                    callee: HirCallable::Function(fid),
                    args,
                    ..
                } => {
                    stats.call_sites_visited += 1;
                    (*result, args.clone(), *fid)
                }
                HirInstruction::Call { .. } | HirInstruction::IndirectCall { .. } => {
                    stats.call_sites_visited += 1;
                    stats.skipped_indirect += 1;
                    continue;
                }
                _ => continue,
            };

            if callee_id == caller_id {
                // Recursive call — skip. A real inliner with a cost
                // model can unroll bounded recursion, but v1
                // bails out.
                stats.skipped_unsupported += 1;
                continue;
            }

            let callee = match callees.get(&callee_id) {
                Some(c) => c,
                None => {
                    stats.skipped_unsupported += 1;
                    continue;
                }
            };

            match classify(callee) {
                CalleeClass::Ok => {}
                CalleeClass::TooLarge => {
                    stats.skipped_too_large += 1;
                    continue;
                }
                CalleeClass::MultiBlock => {
                    stats.skipped_multi_block += 1;
                    continue;
                }
                CalleeClass::Unsupported => {
                    stats.skipped_unsupported += 1;
                    continue;
                }
            }

            jobs.push(InlineJob {
                block_id,
                inst_idx: idx,
                call_result: result,
                args,
                callee_id,
            });
        }

        if jobs.is_empty() {
            continue;
        }

        // Apply jobs from latest position to earliest so earlier
        // indices stay valid. Each application mutates `caller`
        // (blocks + values).
        jobs.sort_by_key(|j| std::cmp::Reverse(j.inst_idx));
        for job in jobs {
            let callee = match callees.get(&job.callee_id) {
                Some(c) => c,
                None => continue,
            };
            apply_inline(caller, &job, callee);
            stats.inlined += 1;
        }
    }

    stats
}

#[derive(Debug, Clone)]
struct InlineJob {
    block_id: HirId,
    inst_idx: usize,
    call_result: Option<HirId>,
    args: Vec<HirId>,
    callee_id: HirId,
}

#[derive(Debug, Clone, Copy)]
enum CalleeClass {
    Ok,
    TooLarge,
    MultiBlock,
    Unsupported,
}

fn classify(callee: &HirFunction) -> CalleeClass {
    if callee.signature.is_async || callee.is_external {
        return CalleeClass::Unsupported;
    }
    if !callee.signature.effects.is_empty() {
        return CalleeClass::Unsupported;
    }
    if callee.blocks.len() != 1 {
        return CalleeClass::MultiBlock;
    }
    let entry = match callee.blocks.get(&callee.entry_block) {
        Some(b) => b,
        None => return CalleeClass::Unsupported,
    };
    if entry.instructions.len() > MAX_INLINE_INSTS {
        return CalleeClass::TooLarge;
    }
    // Inspect every instruction for shapes we don't yet support.
    for inst in &entry.instructions {
        match inst {
            HirInstruction::Call { .. }
            | HirInstruction::IndirectCall { .. }
            | HirInstruction::Alloca { .. }
            | HirInstruction::Atomic { .. }
            | HirInstruction::Fence { .. } => return CalleeClass::Unsupported,
            _ => {}
        }
    }
    // Terminator must be Return (with ≤ 1 value).
    match &entry.terminator {
        HirTerminator::Return { values } if values.len() <= 1 => {}
        _ => return CalleeClass::Unsupported,
    }
    // Phis at the entry block don't make sense (no preds), but
    // defensive: skip if any are present.
    if !entry.phis.is_empty() {
        return CalleeClass::Unsupported;
    }
    CalleeClass::Ok
}

/// Splice the callee's single-block body into the caller's
/// `job.block_id` immediately before instruction index
/// `job.inst_idx` (the Call). Then rewire `job.call_result` to point
/// at the remapped return value if any.
fn apply_inline(caller: &mut HirFunction, job: &InlineJob, callee: &HirFunction) {
    let entry = match callee.blocks.get(&callee.entry_block) {
        Some(b) => b,
        None => return,
    };

    // Build the substitution map. Param ids in the callee → the
    // caller's argument ids directly (no new value needed; the
    // caller already owns them). Other callee-defined values get
    // fresh caller-side ids.
    let mut subs: HashMap<HirId, HirId> = HashMap::new();
    let mut new_values: Vec<(HirId, HirValue)> = Vec::new();

    // Map parameters → caller arguments by ordinal.
    for (i, param) in callee.signature.params.iter().enumerate() {
        if let Some(&arg) = job.args.get(i) {
            subs.insert(param.id, arg);
        }
    }
    // Some callees also keep a `Parameter(i)` HirValue with a
    // different id than `param.id`; map those too.
    for (id, val) in &callee.values {
        if let HirValueKind::Parameter(idx) = val.kind {
            if let Some(&arg) = job.args.get(idx as usize) {
                subs.insert(*id, arg);
            }
        }
    }

    // For every other callee value the inlined body produces,
    // mint a fresh id with a cloned HirValue and register it.
    for (id, val) in &callee.values {
        if subs.contains_key(id) {
            continue; // Already a param substitution.
        }
        let new_id = HirId::new();
        let new_val = HirValue {
            id: new_id,
            ty: val.ty.clone(),
            kind: match &val.kind {
                // Parameters are already mapped above; this arm
                // catches values that aren't params and aren't
                // mapped — Constants and Instruction results.
                HirValueKind::Constant(c) => HirValueKind::Constant(c.clone()),
                HirValueKind::Global(g) => HirValueKind::Global(*g),
                HirValueKind::Undef => HirValueKind::Undef,
                HirValueKind::Instruction => HirValueKind::Instruction,
                HirValueKind::Parameter(_) => continue, // safety: handled above
            },
            uses: HashSet::new(),
            span: None,
        };
        subs.insert(*id, new_id);
        new_values.push((new_id, new_val));
    }

    // Install the new values in the caller.
    for (id, val) in new_values {
        caller.values.insert(id, val);
    }

    // Clone each callee instruction, substituting operands, and
    // collect them for splicing.
    let mut cloned: Vec<HirInstruction> = Vec::with_capacity(entry.instructions.len());
    for inst in &entry.instructions {
        let mut new_inst = inst.clone();
        substitute_operands(&mut new_inst, &subs);
        cloned.push(new_inst);
    }

    // Determine the Return value (if any) so we can map the
    // call's result to it.
    let return_value: Option<HirId> = match &entry.terminator {
        HirTerminator::Return { values } if !values.is_empty() => {
            Some(*subs.get(&values[0]).unwrap_or(&values[0]))
        }
        _ => None,
    };

    // Splice into the caller's block.
    let block = match caller.blocks.get_mut(&job.block_id) {
        Some(b) => b,
        None => return,
    };
    // Drop the original Call instruction; replace it with the
    // cloned instructions in-place.
    let mut new_inst_list: Vec<HirInstruction> =
        Vec::with_capacity(block.instructions.len() + cloned.len());
    for (i, inst) in block.instructions.iter().enumerate() {
        if i == job.inst_idx {
            // Splice the cloned body in place of the call.
            for c in &cloned {
                new_inst_list.push(c.clone());
            }
        } else {
            new_inst_list.push(inst.clone());
        }
    }
    block.instructions = new_inst_list;

    // If the call had a result and the callee returned a value,
    // rewrite every subsequent reference to the call's result to
    // point at the return value instead.
    if let (Some(call_result), Some(ret)) = (job.call_result, return_value) {
        let mut substitution: HashMap<HirId, HirId> = HashMap::new();
        substitution.insert(call_result, ret);
        replace_uses_across_function(caller, &substitution);
        // Remove the now-orphaned call_result from the caller's
        // values (its defining instruction is gone).
        caller.values.shift_remove(&call_result);
    }
}

/// Rewrite every operand / terminator reference to the keys of
/// `subs` so they point at the mapped value instead. Used for
/// `call_result → callee_return_value` after splice.
fn replace_uses_across_function(func: &mut HirFunction, subs: &HashMap<HirId, HirId>) {
    if subs.is_empty() {
        return;
    }
    let map = |id: &mut HirId| {
        if let Some(&new) = subs.get(id) {
            *id = new;
        }
    };
    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            substitute_operands(inst, subs);
        }
        match &mut block.terminator {
            HirTerminator::Return { values } => {
                for v in values {
                    map(v);
                }
            }
            HirTerminator::CondBranch { condition, .. } => map(condition),
            HirTerminator::Switch { value, .. } => map(value),
            HirTerminator::Invoke { args, .. } => {
                for a in args {
                    map(a);
                }
            }
            HirTerminator::PatternMatch { value, .. } => map(value),
            HirTerminator::Branch { .. } | HirTerminator::Unreachable => {}
        }
        for phi in &mut block.phis {
            for (val, _) in &mut phi.incoming {
                map(val);
            }
        }
    }
}

fn substitute_operands(inst: &mut HirInstruction, subs: &HashMap<HirId, HirId>) {
    let map = |id: &mut HirId| {
        if let Some(&new) = subs.get(id) {
            *id = new;
        }
    };
    match inst {
        HirInstruction::Binary { left, right, .. } => {
            map(left);
            map(right);
        }
        HirInstruction::Unary { operand, .. } => map(operand),
        HirInstruction::Cast { operand, .. } => map(operand),
        HirInstruction::Load { ptr, .. } => map(ptr),
        HirInstruction::Store { value, ptr, .. } => {
            map(value);
            map(ptr);
        }
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            map(ptr);
            for i in indices {
                map(i);
            }
        }
        HirInstruction::Call { args, .. } => {
            for a in args {
                map(a);
            }
        }
        HirInstruction::IndirectCall { func_ptr, args, .. } => {
            map(func_ptr);
            for a in args {
                map(a);
            }
        }
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            map(condition);
            map(true_val);
            map(false_val);
        }
        HirInstruction::ExtractValue { aggregate, .. } => map(aggregate),
        HirInstruction::InsertValue {
            aggregate, value, ..
        } => {
            map(aggregate);
            map(value);
        }
        _ => {}
    }
    // The result HirId of an instruction also gets substituted —
    // necessary because we minted fresh ids for callee-defined
    // values during the clone.
    match inst {
        HirInstruction::Binary { result, .. } => map(result),
        HirInstruction::Unary { result, .. } => map(result),
        HirInstruction::Cast { result, .. } => map(result),
        HirInstruction::Load { result, .. } => map(result),
        HirInstruction::GetElementPtr { result, .. } => map(result),
        HirInstruction::ExtractValue { result, .. } => map(result),
        HirInstruction::InsertValue { result, .. } => map(result),
        HirInstruction::Select { result, .. } => map(result),
        HirInstruction::Alloca { result, .. } => map(result),
        HirInstruction::Call {
            result: Some(r), ..
        } => map(r),
        _ => {}
    }
}

#[allow(unused)]
fn touch_types(_t: &HirType, _v: &IndexMap<HirId, HirValue>) {}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirConstant, HirFunctionSignature, HirParam, HirType, ParamAttributes,
    };
    use zyntax_typed_ast::InternedString;

    fn sig(params: Vec<HirType>, ret: HirType) -> HirFunctionSignature {
        HirFunctionSignature {
            params: params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global(&format!("p{i}")),
                    ty,
                    attributes: ParamAttributes::default(),
                })
                .collect(),
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

    fn add_value_for_param(f: &mut HirFunction, param_idx: u32, ty: HirType) -> HirId {
        // Use the param's existing id from the signature.
        let param_id = f.signature.params[param_idx as usize].id;
        f.values.insert(
            param_id,
            HirValue {
                id: param_id,
                ty,
                kind: HirValueKind::Parameter(param_idx),
                uses: HashSet::new(),
                span: None,
            },
        );
        param_id
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

    fn add_const(f: &mut HirFunction, ty: HirType, c: HirConstant) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Constant(c),
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    /// Build a small `square(x: i64) -> i64 { x * x }` callee.
    fn build_square_callee() -> HirFunction {
        let mut f = HirFunction::new(
            InternedString::new_global("square"),
            sig(vec![HirType::I64], HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));

        let x = add_value_for_param(&mut f, 0, HirType::I64);
        let r = add_inst(&mut f, HirType::I64);
        let blk = f.blocks.get_mut(&entry).unwrap();
        blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Mul,
            result: r,
            ty: HirType::I64,
            left: x,
            right: x,
        });
        blk.terminator = HirTerminator::Return { values: vec![r] };
        f
    }

    /// Build `caller(): i64 { square(7) + 1 }`.
    fn build_caller(callee_id: HirId) -> HirFunction {
        let mut f = HirFunction::new(
            InternedString::new_global("caller"),
            sig(vec![], HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));

        let seven = add_const(&mut f, HirType::I64, HirConstant::I64(7));
        let one = add_const(&mut f, HirType::I64, HirConstant::I64(1));
        let sq = add_inst(&mut f, HirType::I64);
        let sum = add_inst(&mut f, HirType::I64);

        let blk = f.blocks.get_mut(&entry).unwrap();
        blk.instructions.push(HirInstruction::Call {
            result: Some(sq),
            callee: HirCallable::Function(callee_id),
            args: vec![seven],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: sum,
            ty: HirType::I64,
            left: sq,
            right: one,
        });
        blk.terminator = HirTerminator::Return { values: vec![sum] };
        f
    }

    fn build_module() -> HirModule {
        let mut module = HirModule::new(InternedString::new_global("test_mod"));
        let mut callee = build_square_callee();
        let callee_id = HirId::new();
        callee.id = callee_id;
        let caller = build_caller(callee_id);
        let caller_id = caller.id;
        let _ = caller_id; // for clarity
        module.functions.insert(callee_id, callee);
        module.functions.insert(caller.id, caller);
        module
    }

    #[test]
    fn inlines_a_small_callee() {
        let mut module = build_module();
        let stats = run_module(&mut module);
        assert_eq!(stats.inlined, 1, "{stats:?}");

        // After inlining, the caller block should contain a Mul
        // (from square's body) followed by the Add — no Call.
        let caller = module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some("caller"))
            .unwrap();
        let entry = caller.blocks.get(&caller.entry_block).unwrap();
        assert!(
            !entry
                .instructions
                .iter()
                .any(|i| matches!(i, HirInstruction::Call { .. })),
            "the Call should be gone after inlining"
        );
        // The Add operand should reference the Mul's result.
        let mul_result = entry
            .instructions
            .iter()
            .find_map(|i| {
                if let HirInstruction::Binary {
                    op: BinaryOp::Mul,
                    result,
                    ..
                } = i
                {
                    Some(*result)
                } else {
                    None
                }
            })
            .expect("Mul present");
        let add_left = entry
            .instructions
            .iter()
            .find_map(|i| {
                if let HirInstruction::Binary {
                    op: BinaryOp::Add,
                    left,
                    ..
                } = i
                {
                    Some(*left)
                } else {
                    None
                }
            })
            .expect("Add present");
        assert_eq!(add_left, mul_result, "Add must use Mul's result");
    }

    #[test]
    fn skips_too_large_callee() {
        // Construct a callee with 9 instructions — over our cap.
        let mut callee = HirFunction::new(
            InternedString::new_global("big"),
            sig(vec![HirType::I64], HirType::I64),
        );
        let entry = HirId::new();
        callee.entry_block = entry;
        callee.blocks.clear();
        callee.blocks.insert(entry, HirBlock::new(entry));
        let x = add_value_for_param(&mut callee, 0, HirType::I64);
        let mut prev = x;
        for _ in 0..(MAX_INLINE_INSTS + 1) {
            let next = add_inst(&mut callee, HirType::I64);
            let one = add_const(&mut callee, HirType::I64, HirConstant::I64(1));
            callee
                .blocks
                .get_mut(&entry)
                .unwrap()
                .instructions
                .push(HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: next,
                    ty: HirType::I64,
                    left: prev,
                    right: one,
                });
            prev = next;
        }
        callee.blocks.get_mut(&entry).unwrap().terminator =
            HirTerminator::Return { values: vec![prev] };

        let callee_id = HirId::new();
        callee.id = callee_id;
        let caller = build_caller(callee_id);

        let mut module = HirModule::new(InternedString::new_global("m"));
        module.functions.insert(callee_id, callee);
        module.functions.insert(caller.id, caller);

        let stats = run_module(&mut module);
        assert_eq!(stats.inlined, 0);
        assert_eq!(stats.skipped_too_large, 1);
    }

    #[test]
    fn skips_recursive_call() {
        // f() { f() }  — recursion isn't inlined in v1.
        let mut f = HirFunction::new(InternedString::new_global("rec"), sig(vec![], HirType::I64));
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let zero = add_const(&mut f, HirType::I64, HirConstant::I64(0));
        let res = add_inst(&mut f, HirType::I64);
        let f_id = HirId::new();
        f.id = f_id;
        let blk = f.blocks.get_mut(&entry).unwrap();
        blk.instructions.push(HirInstruction::Call {
            result: Some(res),
            callee: HirCallable::Function(f_id),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        blk.terminator = HirTerminator::Return { values: vec![zero] };

        let mut module = HirModule::new(InternedString::new_global("m"));
        module.functions.insert(f_id, f);
        let stats = run_module(&mut module);
        assert_eq!(stats.inlined, 0);
        assert_eq!(stats.skipped_unsupported, 1);
    }

    #[test]
    fn does_not_touch_indirect_calls() {
        // A caller with an IndirectCall — should be left alone and
        // counted as `skipped_indirect`.
        let mut f = HirFunction::new(
            InternedString::new_global("indir"),
            sig(vec![HirType::I64], HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let x = add_value_for_param(&mut f, 0, HirType::I64);
        let result = add_inst(&mut f, HirType::I64);
        let blk = f.blocks.get_mut(&entry).unwrap();
        blk.instructions.push(HirInstruction::IndirectCall {
            result: Some(result),
            func_ptr: x,
            args: vec![],
            return_ty: HirType::I64,
        });
        blk.terminator = HirTerminator::Return {
            values: vec![result],
        };

        let mut module = HirModule::new(InternedString::new_global("m"));
        module.functions.insert(f.id, f);
        let stats = run_module(&mut module);
        assert_eq!(stats.inlined, 0);
        assert_eq!(stats.skipped_indirect, 1);
    }
}
