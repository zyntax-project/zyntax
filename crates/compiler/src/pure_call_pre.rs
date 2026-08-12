//! Cross-branch pure-call elimination via speculative hoisting.
//!
//! Dominator-tree CSE ([`crate::cse`]) collapses a redundant pure call
//! only when one call *dominates* the other. Recursive self-inlining
//! routinely produces a redundancy that dominator-CSE can't see: the
//! depth-1 inline of `fib` makes the `fib(n-1)` expansion and the
//! `fib(n-2)` expansion each call `fib(n-3)`, but those two calls live on
//! separate CFG branches (each guarded by its own base-case test), so
//! neither dominates the other.
//!
//! This pass eliminates that partial redundancy by hoisting the shared
//! call to the nearest block that dominates *both* sites and rewriting
//! both original results to the single hoisted value. Because the hoist
//! target runs on more paths than the original call sites (it may run
//! when neither original would have), the move is only legal when the
//! callee is safe to evaluate speculatively — pure, total, and
//! fault-free. That certificate comes from
//! [`crate::purity::speculation_safe_module`]; without it a function is
//! never hoisted.
//!
//! ## Why this is sound
//!
//! * **Same value.** Two calls share a value number only when they name
//!   the same speculation-safe function with arguments equal in affine
//!   `(base, offset)` form. Each argument's base SSA value dominates the
//!   hoist point (checked explicitly), so on any execution reaching a
//!   former call site the base held the same value at the hoist point
//!   that it held at the site — the recomputed argument is identical.
//! * **No new effect.** Speculation-safety guarantees the extra
//!   evaluations on off-paths terminate, never trap, and have no
//!   observable effect, so running the call more often is invisible.
//! * **Availability.** The hoist target dominates both sites, so the
//!   hoisted result is defined on every path that reaches either site.
//!
//! Turned off with `ZYNTAX_DISABLE_PURE_CALL_PRE=1`, or per-process
//! through [`set_enabled`], for A/B measurement.

use crate::analysis::DominatorTree;
use crate::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirType,
    HirValue, HirValueKind,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PureCallPreStats {
    /// Redundant call sites removed (each collapsed into a hoisted call).
    pub hoisted: usize,
    /// Call groups examined.
    pub groups_visited: usize,
}

/// Whether the pass is on. `0` defers to the environment, `1` forces it
/// on and `2` forces it off, so a caller compiling the same source both
/// ways in one process does not have to mutate the environment.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Force the pass on or off for subsequent compiles on any thread, or
/// pass `None` to go back to reading the environment.
///
/// Measuring what the pass is worth means compiling one source with it
/// and without it, which an environment variable alone cannot express
/// inside a single process.
pub fn set_enabled(enabled: Option<bool>) {
    let value = match enabled {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    OVERRIDE.store(value, Ordering::Relaxed);
}

/// Whether the next `run_module` will do anything.
pub fn is_enabled() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => !std::env::var("ZYNTAX_DISABLE_PURE_CALL_PRE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false),
    }
}

/// Run the hoist over every function. Purity + speculation-safety are
/// recomputed from `signature.is_pure`, so run [`crate::purity`] first.
pub fn run_module(module: &mut HirModule) -> PureCallPreStats {
    if !is_enabled() {
        return PureCallPreStats::default();
    }
    let safe = crate::purity::speculation_safe_module(module);
    let mut stats = PureCallPreStats::default();
    for func in module.functions.values_mut() {
        let s = run_function(func, &safe);
        stats.hoisted += s.hoisted;
        stats.groups_visited += s.groups_visited;
    }
    stats
}

/// One call site: where it lives and what it computes.
struct CallSite {
    /// Block containing the call.
    block: HirId,
    /// The call's result value id.
    result: HirId,
    /// Callee (a speculation-safe function).
    callee: HirId,
    /// Affine `(base, offset)` form of each argument.
    arg_forms: Vec<(HirId, i128)>,
    /// Generic instantiation, part of the identity.
    type_args: Vec<HirType>,
    const_args: Vec<HirConstant>,
}

fn run_function(
    func: &mut HirFunction,
    safe: &std::collections::HashSet<HirId>,
) -> PureCallPreStats {
    let mut stats = PureCallPreStats::default();

    let bin_defs = collect_bin_defs(func);
    let int_consts = collect_int_consts(func);

    // Gather every speculation-safe pure call with a result.
    let mut sites: Vec<CallSite> = Vec::new();
    for (bid, block) in &func.blocks {
        for inst in &block.instructions {
            if let HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Function(fid),
                args,
                type_args,
                const_args,
                ..
            } = inst
            {
                if safe.contains(fid) {
                    sites.push(CallSite {
                        block: *bid,
                        result: *r,
                        callee: *fid,
                        arg_forms: args
                            .iter()
                            .map(|a| affine_of(*a, &bin_defs, &int_consts))
                            .collect(),
                        type_args: type_args.clone(),
                        const_args: const_args.clone(),
                    });
                }
            }
        }
    }
    if sites.len() < 2 {
        return stats;
    }

    // Group by value number: (callee, arg forms, type/const args).
    type Key = (HirId, Vec<(HirId, i128)>, Vec<HirType>, Vec<HirConstant>);
    let mut groups: HashMap<Key, Vec<usize>> = HashMap::new();
    for (i, s) in sites.iter().enumerate() {
        let key = (
            s.callee,
            s.arg_forms.clone(),
            s.type_args.clone(),
            s.const_args.clone(),
        );
        groups.entry(key).or_default().push(i);
    }

    let dt = DominatorTree::new(func);
    // `def_block[v]` = block that defines value `v` (instruction result).
    let def_block = instruction_def_blocks(func);

    // Collect the rewrites, then apply once (can't mutate `func` while
    // holding the immutable `sites`/`dt` borrows).
    struct Plan {
        hoist_block: HirId,
        callee: HirId,
        arg_forms: Vec<(HirId, i128)>,
        type_args: Vec<HirType>,
        const_args: Vec<HirConstant>,
        result_ty: HirType,
        redundant: Vec<HirId>, // original call results to replace
    }
    let mut plans: Vec<Plan> = Vec::new();

    for (key, members) in &groups {
        if members.len() < 2 {
            continue;
        }
        stats.groups_visited += 1;

        // Nearest common dominator of all member blocks.
        let blocks: Vec<HirId> = members.iter().map(|&i| sites[i].block).collect();
        let Some(ncd) = nearest_common_dominator(&dt, &blocks) else {
            continue;
        };

        // Every argument base must dominate the hoist block, so the
        // recomputed argument is available and identical there. Params /
        // globals / constants dominate everything (no def block).
        let bases_ok = key.1.iter().all(|(base, _)| match def_block.get(base) {
            Some(db) => dt.dominates(*db, ncd),
            None => true,
        });
        if !bases_ok {
            continue;
        }

        // If the common dominator is itself one of the member blocks and
        // that member's call dominates all others, dominator-CSE already
        // handles it without speculation — skip (belt-and-suspenders;
        // cse runs alongside).
        let result_ty = func
            .values
            .get(&sites[members[0]].result)
            .map(|v| v.ty.clone())
            .unwrap_or(HirType::I64);

        plans.push(Plan {
            hoist_block: ncd,
            callee: key.0,
            arg_forms: key.1.clone(),
            type_args: key.2.clone(),
            const_args: key.3.clone(),
            result_ty,
            redundant: members.iter().map(|&i| sites[i].result).collect(),
        });
    }

    if plans.is_empty() {
        return stats;
    }

    // Apply: for each plan synthesise arg computations + the call at the
    // end of the hoist block (before its terminator), then substitute
    // every redundant result with the hoisted result and delete the
    // original call instructions.
    let mut substitutions: HashMap<HirId, HirId> = HashMap::new();
    for plan in plans {
        // Build argument values in the hoist block.
        let mut new_insts: Vec<HirInstruction> = Vec::new();
        let mut arg_ids: Vec<HirId> = Vec::new();
        for (base, offset) in &plan.arg_forms {
            if *offset == 0 {
                arg_ids.push(*base);
                continue;
            }
            // arg = base + offset (as base - (-offset) or base + offset).
            let ty = func
                .values
                .get(base)
                .map(|v| v.ty.clone())
                .unwrap_or(HirType::I64);
            let c_id = fresh_int_const(func, &ty, *offset);
            let r_id = fresh_inst_value(func, &ty);
            new_insts.push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r_id,
                ty,
                left: *base,
                right: c_id,
            });
            arg_ids.push(r_id);
        }
        let call_res = fresh_inst_value(func, &plan.result_ty);
        new_insts.push(HirInstruction::Call {
            result: Some(call_res),
            callee: HirCallable::Function(plan.callee),
            args: arg_ids,
            type_args: plan.type_args,
            const_args: plan.const_args,
            is_tail: false,
        });

        // Insert at the end of the hoist block's instruction list
        // (terminator is a separate field, so "end of instructions" is
        // before the branch).
        if let Some(b) = func.blocks.get_mut(&plan.hoist_block) {
            b.instructions.extend(new_insts);
        }

        for r in plan.redundant {
            substitutions.insert(r, call_res);
            stats.hoisted += 1;
        }
        // The hoisted call itself replaces one real evaluation, so it
        // isn't a net add of the group's first member.
        stats.hoisted -= 1;
    }

    // Rewrite uses and drop the now-redundant original calls.
    crate::cse::apply_substitutions_public(func, &substitutions);
    crate::cse::remove_redundant_instructions_public(func, &substitutions);

    stats
}

// ─── helpers ──────────────────────────────────────────────────────────

fn nearest_common_dominator(dt: &DominatorTree, blocks: &[HirId]) -> Option<HirId> {
    let mut iter = blocks.iter();
    let mut acc = *iter.next()?;
    for &b in iter {
        acc = nca2(dt, acc, b)?;
    }
    Some(acc)
}

/// Nearest common ancestor of two blocks in the dominator tree.
fn nca2(dt: &DominatorTree, a: HirId, b: HirId) -> Option<HirId> {
    // Collect a's dominator chain (a, idom(a), …, entry).
    let mut chain_a = vec![a];
    let mut cur = a;
    while let Some(p) = dt.idom(cur) {
        chain_a.push(p);
        cur = p;
    }
    // Walk b's chain until we hit something in a's chain.
    let mut cur = b;
    loop {
        if chain_a.contains(&cur) {
            return Some(cur);
        }
        cur = dt.idom(cur)?;
    }
}

/// `result → block` for every instruction result.
fn instruction_def_blocks(func: &HirFunction) -> HashMap<HirId, HirId> {
    let mut m = HashMap::new();
    for (bid, block) in &func.blocks {
        for inst in &block.instructions {
            if let Some(r) = inst_result(inst) {
                m.insert(r, *bid);
            }
        }
        for phi in &block.phis {
            m.insert(phi.result, *bid);
        }
    }
    m
}

fn inst_result(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. }
        | HirInstruction::Unary { result, .. }
        | HirInstruction::Cast { result, .. }
        | HirInstruction::GetElementPtr { result, .. }
        | HirInstruction::ExtractValue { result, .. }
        | HirInstruction::InsertValue { result, .. }
        | HirInstruction::Load { result, .. }
        | HirInstruction::Alloca { result, .. }
        | HirInstruction::Select { result, .. } => Some(*result),
        HirInstruction::Call { result, .. } | HirInstruction::IndirectCall { result, .. } => {
            *result
        }
        _ => None,
    }
}

fn fresh_inst_value(func: &mut HirFunction, ty: &HirType) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty: ty.clone(),
            kind: HirValueKind::Instruction,
            uses: Default::default(),
            span: None,
        },
    );
    id
}

fn fresh_int_const(func: &mut HirFunction, ty: &HirType, v: i128) -> HirId {
    let c = int_constant(ty, v);
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty: ty.clone(),
            kind: HirValueKind::Constant(c),
            uses: Default::default(),
            span: None,
        },
    );
    id
}

fn int_constant(ty: &HirType, v: i128) -> HirConstant {
    match ty {
        HirType::I8 => HirConstant::I8(v as i8),
        HirType::I16 => HirConstant::I16(v as i16),
        HirType::I32 => HirConstant::I32(v as i32),
        HirType::I64 => HirConstant::I64(v as i64),
        HirType::I128 => HirConstant::I128(v),
        HirType::U8 => HirConstant::U8(v as u8),
        HirType::U16 => HirConstant::U16(v as u16),
        HirType::U32 => HirConstant::U32(v as u32),
        HirType::U64 => HirConstant::U64(v as u64),
        _ => HirConstant::I64(v as i64),
    }
}

fn collect_bin_defs(func: &HirFunction) -> HashMap<HirId, (BinaryOp, HirId, HirId)> {
    let mut m = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary {
                op: op @ (BinaryOp::Add | BinaryOp::Sub),
                result,
                left,
                right,
                ..
            } = inst
            {
                m.insert(*result, (*op, *left, *right));
            }
        }
    }
    m
}

fn collect_int_consts(func: &HirFunction) -> HashMap<HirId, i128> {
    let mut m = HashMap::new();
    for (id, v) in &func.values {
        if let HirValueKind::Constant(c) = &v.kind {
            let iv = match c {
                HirConstant::I8(x) => *x as i128,
                HirConstant::I16(x) => *x as i128,
                HirConstant::I32(x) => *x as i128,
                HirConstant::I64(x) => *x as i128,
                HirConstant::I128(x) => *x,
                HirConstant::U8(x) => *x as i128,
                HirConstant::U16(x) => *x as i128,
                HirConstant::U32(x) => *x as i128,
                HirConstant::U64(x) => *x as i128,
                _ => continue,
            };
            m.insert(*id, iv);
        }
    }
    m
}

fn affine_of(
    id: HirId,
    bin_defs: &HashMap<HirId, (BinaryOp, HirId, HirId)>,
    int_consts: &HashMap<HirId, i128>,
) -> (HirId, i128) {
    fn go(
        id: HirId,
        bin_defs: &HashMap<HirId, (BinaryOp, HirId, HirId)>,
        int_consts: &HashMap<HirId, i128>,
        depth: u32,
    ) -> (HirId, i128) {
        if depth > 64 {
            return (id, 0);
        }
        if let Some((op, l, r)) = bin_defs.get(&id).copied() {
            match op {
                BinaryOp::Add => {
                    if let Some(c) = int_consts.get(&r) {
                        let (b, o) = go(l, bin_defs, int_consts, depth + 1);
                        return (b, o.wrapping_add(*c));
                    }
                    if let Some(c) = int_consts.get(&l) {
                        let (b, o) = go(r, bin_defs, int_consts, depth + 1);
                        return (b, o.wrapping_add(*c));
                    }
                }
                BinaryOp::Sub => {
                    if let Some(c) = int_consts.get(&r) {
                        let (b, o) = go(l, bin_defs, int_consts, depth + 1);
                        return (b, o.wrapping_sub(*c));
                    }
                }
                _ => {}
            }
        }
        (id, 0)
    }
    go(id, bin_defs, int_consts, 0)
}
