//! Conservative function-purity inference.
//!
//! A function is *pure* when its return value depends only on its
//! arguments and evaluating it produces no observable effect — no
//! stores, no memory reads that could alias mutable state, no algebraic
//! effects, no I/O, no dynamic dispatch. Purity is what lets later
//! passes treat two identical calls as the same value (call CSE) and
//! reason about redundancy across the control-flow graph.
//!
//! ## Why we compute it here (not lean on the backend)
//!
//! LLVM infers `memory(none)` and CSEs duplicate pure calls in its
//! optimiser; the BC interpreter, the wasm tier, and Cranelift's
//! single-pass backend don't. Computing purity at the HIR level lets
//! *every* backend share the result — see the call-CSE consumer in
//! [`crate::cse`].
//!
//! ## The lattice + fixpoint
//!
//! Purity is a greatest-fixpoint property over the call graph: a
//! function is pure iff every instruction it contains is value-pure
//! *and* every function it calls is pure. Mutual/self recursion means
//! we can't decide one function in isolation, so we iterate:
//!
//!   1. Optimistically assume every locally-clean function is pure.
//!   2. Repeatedly demote any function that calls a known-impure one.
//!   3. Stop when a full sweep demotes nothing.
//!
//! Self-recursion (like `fib` calling `fib`) is handled naturally: the
//! optimistic seed keeps `fib` pure through the fixpoint because its
//! only callee — itself — never gets demoted.
//!
//! ## Conservatism
//!
//! Every classification errs toward *impure*. Anything we don't model
//! precisely (loads, allocations, closures, trait dispatch, effects,
//! fibers, async, vector memory ops, non-`Function` callees) makes the
//! enclosing function impure. Marking a truly-impure function pure would
//! be a miscompile once call-CSE trusts the flag, so the whitelist below
//! is deliberately narrow: only instructions whose result is a pure
//! function of their SSA operands are allowed.

use crate::hir::{
    BinaryOp, HirCallable, HirFunction, HirId, HirInstruction, HirModule, HirValueKind,
};
use std::collections::{HashMap, HashSet};

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PurityStats {
    /// Functions marked pure this run.
    pub pure: usize,
    /// Functions left impure.
    pub impure: usize,
    /// Fixpoint iterations taken.
    pub iterations: usize,
}

/// Infer purity for every function in `module` and write the result
/// back into each function's `signature.is_pure`. Returns stats.
pub fn infer_module(module: &mut HirModule) -> PurityStats {
    // Step 1: local screen. A function that contains an instruction we
    // can't prove value-pure is impure regardless of its callees, and
    // never needs revisiting. Functions that pass the local screen are
    // *candidates* whose purity still hinges on their callees.
    let mut candidate: HashSet<HirId> = HashSet::new();
    // callee_fns[f] = the set of directly-called HirIds for a candidate.
    let mut callee_fns: HashMap<HirId, Vec<HirId>> = HashMap::new();

    for (id, func) in &module.functions {
        // External / declaration-only functions have no body we can
        // inspect — treat as impure (we can't see what they do).
        if func.is_external || func.blocks.is_empty() {
            continue;
        }
        if let Some(callees) = locally_pure_callees(func) {
            candidate.insert(*id);
            callee_fns.insert(*id, callees);
        }
    }

    // Step 2: greatest-fixpoint demotion. Start optimistic (all
    // candidates pure) and demote any whose callee is not a pure
    // candidate, until stable.
    let mut pure: HashSet<HirId> = candidate.clone();
    let mut iterations = 0;
    loop {
        iterations += 1;
        let mut changed = false;
        // Collect demotions first so we don't mutate `pure` mid-scan.
        let mut demote: Vec<HirId> = Vec::new();
        for id in &pure {
            let callees = &callee_fns[id];
            // Pure only if every callee is itself currently pure. A
            // callee not in `pure` (impure, external, or a non-candidate)
            // taints this function.
            if !callees.iter().all(|c| pure.contains(c)) {
                demote.push(*id);
            }
        }
        for id in demote {
            pure.remove(&id);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // Step 3: write back.
    let mut stats = PurityStats {
        iterations,
        ..Default::default()
    };
    for (id, func) in &mut module.functions {
        let is_pure = pure.contains(id);
        func.signature.is_pure = is_pure;
        if func.is_external || func.blocks.is_empty() {
            // Leave the flag as written (false); don't count declarations.
            continue;
        }
        if is_pure {
            stats.pure += 1;
        } else {
            stats.impure += 1;
        }
    }
    stats
}

/// Screen one function for *local* purity: every instruction must be
/// value-pure. Returns `Some(callees)` — the HirIds of every directly
/// called `Function` — when the body is locally clean, or `None` if any
/// instruction disqualifies it outright.
///
/// The returned callee list is what the fixpoint uses to propagate
/// impurity across the call graph. A call to anything other than a
/// direct `Function` (symbol, intrinsic, indirect) is treated as
/// disqualifying here rather than as a callee, because we can't name a
/// HirId to check its purity.
fn locally_pure_callees(func: &HirFunction) -> Option<Vec<HirId>> {
    let mut callees = Vec::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            match inst {
                // Value-pure: result is a deterministic function of the
                // SSA operands, no memory or control effect.
                HirInstruction::Binary { .. }
                | HirInstruction::Unary { .. }
                | HirInstruction::Cast { .. }
                | HirInstruction::Select { .. }
                | HirInstruction::ExtractValue { .. }
                | HirInstruction::InsertValue { .. }
                | HirInstruction::GetElementPtr { .. }
                | HirInstruction::CreateUnion { .. }
                | HirInstruction::GetUnionDiscriminant { .. }
                | HirInstruction::ExtractUnionValue { .. }
                | HirInstruction::VectorSplat { .. }
                | HirInstruction::VectorExtractLane { .. }
                | HirInstruction::VectorInsertLane { .. }
                | HirInstruction::VectorHorizontalReduce { .. }
                | HirInstruction::VectorUnaryOp { .. }
                | HirInstruction::VectorMinMax { .. }
                | HirInstruction::VectorDot { .. } => {}

                // A direct call is pure-*iff* its callee is; record the
                // callee for the fixpoint. Any other callable shape
                // (indirect, intrinsic, symbol) is opaque → impure.
                HirInstruction::Call { callee, .. } => match callee {
                    HirCallable::Function(fid) => callees.push(*fid),
                    _ => return None,
                },

                // Everything else is potentially effectful or reads
                // mutable state — conservatively disqualifying.
                _ => return None,
            }
        }
    }
    Some(callees)
}

// ─── Speculation safety (totality + no-fault) ─────────────────────────
//
// A pure function can be *speculated* — evaluated on a path where the
// source program wouldn't have called it — only if doing so can never
// change observable behaviour. Purity already rules out effects; what's
// left is: the call must always terminate and never fault. This is the
// gate the cross-branch pure-call hoist ([`crate::pure_call_pre`]) needs
// before it may move a call to a dominating block that runs more often
// than the original call sites.
//
// The analysis is deliberately narrow. It certifies a function only when
// termination is structurally obvious:
//   * pure (no effects) and non-faulting (no integer div/rem, which trap
//     on zero),
//   * an acyclic CFG (no intra-function loop that could spin forever),
//   * every callee is itself speculation-safe, and
//   * if self-recursive, the recursion is well-founded: some integer
//     parameter strictly decreases on every self-call and a range guard
//     (`param < c` / `param <= c`) at the entry sends small values to a
//     recursion-free base case. Decreasing an integer that bottoms out
//     at a `<` guard terminates for every starting value — including the
//     smaller/negative ones a speculative call might introduce.
//
// Mutual recursion, non-range base guards (`param == 0`), and anything
// that doesn't fit are conservatively rejected: the worst case of a
// rejection is a missed optimisation, whereas a wrong acceptance is a
// hang. `Add`/`Sub` are the only offset-bearing ops modelled, matching
// [`crate::cse`]'s affine normalisation.

/// The set of functions safe to speculatively evaluate. Requires purity
/// to have been inferred already (reads `signature.is_pure`).
pub fn speculation_safe_module(module: &HirModule) -> HashSet<HirId> {
    let mut candidate: HashSet<HirId> = HashSet::new();
    let mut callees_of: HashMap<HirId, HashSet<HirId>> = HashMap::new();
    for (id, f) in &module.functions {
        if f.is_external || f.blocks.is_empty() || !f.signature.is_pure {
            continue;
        }
        if has_faulting_op(f) || !cfg_is_acyclic(f) {
            continue;
        }
        candidate.insert(*id);
        callees_of.insert(*id, direct_function_callees(f));
    }

    let mut safe: HashSet<HirId> = HashSet::new();
    loop {
        let mut changed = false;
        for id in &candidate {
            if safe.contains(id) {
                continue;
            }
            let callees = &callees_of[id];
            // Every non-self callee must already be certified safe. A
            // callee outside `candidate` (impure / faulting / looping /
            // external) can never enter `safe`, so this also rejects any
            // function reaching such a callee.
            let non_self_all_safe = callees
                .iter()
                .filter(|c| **c != *id)
                .all(|c| safe.contains(c));
            if !non_self_all_safe {
                continue;
            }
            let self_recursive = callees.contains(id);
            let ok = if self_recursive {
                structurally_decreasing(&module.functions[id])
            } else {
                true
            };
            if ok {
                safe.insert(*id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    safe
}

/// Any instruction that can trap at runtime disqualifies speculation.
/// Integer `Div`/`Rem` trap on a zero divisor; a speculative call could
/// hit a divisor the original control flow avoided.
fn has_faulting_op(func: &HirFunction) -> bool {
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary {
                op: BinaryOp::Div | BinaryOp::Rem,
                ..
            } = inst
            {
                return true;
            }
        }
    }
    false
}

/// Direct `Function` callees (as a set).
fn direct_function_callees(func: &HirFunction) -> HashSet<HirId> {
    let mut s = HashSet::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Call {
                callee: HirCallable::Function(fid),
                ..
            } = inst
            {
                s.insert(*fid);
            }
        }
    }
    s
}

/// True when the block CFG has no cycle (a loop could run forever, which
/// would make a speculated call non-terminating). DFS with a colour
/// marking; a grey→grey edge is a back-edge.
fn cfg_is_acyclic(func: &HirFunction) -> bool {
    #[derive(Clone, Copy, PartialEq)]
    enum Colour {
        White,
        Grey,
        Black,
    }
    let mut colour: HashMap<HirId, Colour> =
        func.blocks.keys().map(|k| (*k, Colour::White)).collect();
    // Iterative DFS with an explicit stack of (block, child-index).
    let mut stack: Vec<(HirId, usize)> = Vec::new();
    let entry = func.entry_block;
    if !func.blocks.contains_key(&entry) {
        return true;
    }
    stack.push((entry, 0));
    colour.insert(entry, Colour::Grey);
    while let Some((blk, idx)) = stack.pop() {
        let succ = successors(func, blk);
        if idx < succ.len() {
            stack.push((blk, idx + 1));
            let next = succ[idx];
            match colour.get(&next).copied().unwrap_or(Colour::Black) {
                Colour::Grey => return false, // back-edge → cycle
                Colour::White => {
                    colour.insert(next, Colour::Grey);
                    stack.push((next, 0));
                }
                Colour::Black => {}
            }
        } else {
            colour.insert(blk, Colour::Black);
        }
    }
    true
}

/// Successor blocks of `blk` per its terminator.
fn successors(func: &HirFunction, blk: HirId) -> Vec<HirId> {
    use crate::hir::HirTerminator::*;
    match &func.blocks[&blk].terminator {
        Return { .. } | Unreachable => vec![],
        Branch { target } => vec![*target],
        CondBranch {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        Switch { default, cases, .. } => {
            let mut v = vec![*default];
            v.extend(cases.iter().map(|(_, t)| *t));
            v
        }
        Invoke { normal, unwind, .. } => vec![*normal, *unwind],
        PatternMatch {
            default, patterns, ..
        } => {
            let mut v: Vec<HirId> = patterns.iter().map(|p| p.target).collect();
            if let Some(d) = default {
                v.push(*d);
            }
            v
        }
    }
}

/// Well-founded self-recursion check: some integer parameter strictly
/// decreases on every self-call, and a range guard at the entry routes
/// small values to a recursion-free base. See the module note above for
/// why this guarantees termination on every input.
fn structurally_decreasing(func: &HirFunction) -> bool {
    let self_id = func.id;
    // param index → value id.
    let mut param_id: HashMap<u32, HirId> = HashMap::new();
    for (id, v) in &func.values {
        if let HirValueKind::Parameter(idx) = v.kind {
            param_id.insert(idx, *id);
        }
    }
    if param_id.is_empty() {
        return false;
    }
    let bin_defs = affine_bin_defs(func);
    let int_consts = affine_int_consts(func);

    // All self-calls, with their argument lists.
    let self_calls: Vec<&Vec<HirId>> = func
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .filter_map(|inst| match inst {
            HirInstruction::Call {
                callee: HirCallable::Function(fid),
                args,
                ..
            } if *fid == self_id => Some(args),
            _ => None,
        })
        .collect();
    if self_calls.is_empty() {
        return false;
    }

    // Try each parameter position as the decreasing measure.
    for (&idx, &pid) in &param_id {
        // Every self-call must strictly decrease param `idx`.
        let decreases = self_calls.iter().all(|args| {
            args.get(idx as usize)
                .map(|a| {
                    let (base, off) = affine_of(*a, &bin_defs, &int_consts);
                    base == pid && off < 0
                })
                .unwrap_or(false)
        });
        if !decreases {
            continue;
        }
        if entry_range_guard_to_base(func, pid, &bin_defs, &int_consts, self_id) {
            return true;
        }
    }
    false
}

/// The entry block must branch on `x < c` / `x <= c` where `x` is an
/// affine form of `pid`, with the true (small-value) side reaching no
/// self-call — a genuine recursion-free base case for small inputs.
fn entry_range_guard_to_base(
    func: &HirFunction,
    pid: HirId,
    bin_defs: &HashMap<HirId, (BinaryOp, HirId, HirId)>,
    int_consts: &HashMap<HirId, i128>,
    self_id: HirId,
) -> bool {
    let entry = match func.blocks.get(&func.entry_block) {
        Some(b) => b,
        None => return false,
    };
    let crate::hir::HirTerminator::CondBranch {
        condition,
        true_target,
        false_target,
    } = &entry.terminator
    else {
        return false;
    };
    // The condition must be `Lt`/`Le` with the affine-of-pid on the left
    // and a constant on the right (`x < c`), so the true side is the
    // small-value side.
    let Some((op, left, right)) = bin_defs
        .get(condition)
        .copied()
        .and_then(|(op, l, r)| matches!(op, BinaryOp::Lt | BinaryOp::Le).then_some((op, l, r)))
    else {
        return false;
    };
    let _ = op;
    let (lbase, _loff) = affine_of(left, bin_defs, int_consts);
    if lbase != pid || !int_consts.contains_key(&right) {
        return false;
    }
    // No self-call may be reachable from the true (base) side, and the
    // false side is where recursion lives.
    !self_call_reachable(func, *true_target, self_id) && {
        let _ = false_target;
        true
    }
}

/// Is a call to `self_id` reachable from `start` (inclusive)?
fn self_call_reachable(func: &HirFunction, start: HirId, self_id: HirId) -> bool {
    let mut seen: HashSet<HirId> = HashSet::new();
    let mut stack = vec![start];
    while let Some(b) = stack.pop() {
        if !seen.insert(b) {
            continue;
        }
        let Some(block) = func.blocks.get(&b) else {
            continue;
        };
        for inst in &block.instructions {
            if let HirInstruction::Call {
                callee: HirCallable::Function(fid),
                ..
            } = inst
            {
                if *fid == self_id {
                    return true;
                }
            }
        }
        stack.extend(successors(func, b));
    }
    false
}

// ─── Small affine resolver (shared shape with `cse`) ──────────────────

fn affine_bin_defs(func: &HirFunction) -> HashMap<HirId, (BinaryOp, HirId, HirId)> {
    let mut m = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary {
                op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::Lt | BinaryOp::Le),
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

fn affine_int_consts(func: &HirFunction) -> HashMap<HirId, i128> {
    let mut m = HashMap::new();
    for (id, v) in &func.values {
        if let HirValueKind::Constant(c) = &v.kind {
            let iv = match c {
                crate::hir::HirConstant::I8(x) => *x as i128,
                crate::hir::HirConstant::I16(x) => *x as i128,
                crate::hir::HirConstant::I32(x) => *x as i128,
                crate::hir::HirConstant::I64(x) => *x as i128,
                crate::hir::HirConstant::I128(x) => *x,
                crate::hir::HirConstant::U8(x) => *x as i128,
                crate::hir::HirConstant::U16(x) => *x as i128,
                crate::hir::HirConstant::U32(x) => *x as i128,
                crate::hir::HirConstant::U64(x) => *x as i128,
                _ => continue,
            };
            m.insert(*id, iv);
        }
    }
    m
}

/// Canonical `(base, offset)` for an integer value via `add`/`sub` by a
/// constant. Only `+c` / `-c` shift the offset; anything else is a base.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirConstant, HirFunctionSignature, HirInstruction, HirTerminator,
        HirType, HirValue, HirValueKind,
    };
    use zyntax_typed_ast::InternedString;

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

    fn mk(name: &str) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global(name), sig());
        let entry = HirId::new();
        f.id = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f
    }

    fn param(f: &mut HirFunction, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(idx),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn konst(f: &mut HirFunction, c: HirConstant) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Constant(c),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn result(f: &mut HirFunction) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn push(f: &mut HirFunction, inst: HirInstruction) {
        let e = f.entry_block;
        f.blocks.get_mut(&e).unwrap().instructions.push(inst);
    }

    fn ret(f: &mut HirFunction, v: HirId) {
        let e = f.entry_block;
        f.blocks.get_mut(&e).unwrap().terminator = HirTerminator::Return { values: vec![v] };
    }

    /// A leaf arithmetic function is pure.
    #[test]
    fn arithmetic_leaf_is_pure() {
        let mut m = HirModule::new(InternedString::new_global("m"));
        let mut f = mk("add1");
        let n = param(&mut f, 0);
        let one = konst(&mut f, HirConstant::I64(1));
        let r = result(&mut f);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r,
                ty: HirType::I64,
                left: n,
                right: one,
            },
        );
        ret(&mut f, r);
        let fid = f.id;
        m.functions.insert(fid, f);

        let stats = infer_module(&mut m);
        assert_eq!(stats.pure, 1);
        assert!(m.functions[&fid].signature.is_pure);
    }

    /// Self-recursion stays pure through the fixpoint (the `fib` case).
    #[test]
    fn self_recursive_is_pure() {
        let mut m = HirModule::new(InternedString::new_global("m"));
        let mut f = mk("fib");
        let fid = f.id;
        let n = param(&mut f, 0);
        let c = konst(&mut f, HirConstant::I64(1));
        let sub = result(&mut f);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: sub,
                ty: HirType::I64,
                left: n,
                right: c,
            },
        );
        let call = result(&mut f);
        push(
            &mut f,
            HirInstruction::Call {
                result: Some(call),
                callee: HirCallable::Function(fid),
                args: vec![sub],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        ret(&mut f, call);
        m.functions.insert(fid, f);

        let stats = infer_module(&mut m);
        assert!(
            m.functions[&fid].signature.is_pure,
            "self-recursive pure fn should stay pure, stats={stats:?}"
        );
    }

    /// A store makes the function impure, and any pure caller of it is
    /// demoted transitively.
    #[test]
    fn store_taints_transitively() {
        let mut m = HirModule::new(InternedString::new_global("m"));

        // impure: writes memory.
        let mut writer = mk("writer");
        let wid = writer.id;
        let p = param(&mut writer, 0);
        let v = konst(&mut writer, HirConstant::I64(7));
        push(
            &mut writer,
            HirInstruction::Store {
                value: v,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        ret(&mut writer, v);
        m.functions.insert(wid, writer);

        // caller: locally clean but calls the impure writer.
        let mut caller = mk("caller");
        let cid = caller.id;
        let a = param(&mut caller, 0);
        let r = result(&mut caller);
        push(
            &mut caller,
            HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Function(wid),
                args: vec![a],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        ret(&mut caller, r);
        m.functions.insert(cid, caller);

        infer_module(&mut m);
        assert!(!m.functions[&wid].signature.is_pure, "writer is impure");
        assert!(
            !m.functions[&cid].signature.is_pure,
            "caller of impure fn must be demoted"
        );
    }

    // ─── speculation-safety (totality gate) ──────────────────────────

    /// Build a self-recursive `fib`-shaped function:
    ///   entry: c = <guard_op>(n, 2); brcond c -> base, rec
    ///   base:  return n
    ///   rec:   a = n <arg_op> 1; r = call self(a); return r
    /// The knobs let tests flip individual termination preconditions.
    fn mk_recursive(
        guard_op: BinaryOp,
        arg_op: BinaryOp,
        add_back_edge: bool,
    ) -> (HirModule, HirId) {
        let mut m = HirModule::new(InternedString::new_global("m"));
        let mut f = mk("rec");
        let fid = f.id;
        let entry = f.entry_block;
        let base = HirId::new();
        let rec = HirId::new();
        f.blocks.insert(base, HirBlock::new(base));
        f.blocks.insert(rec, HirBlock::new(rec));

        let n = param(&mut f, 0);
        let two = konst(&mut f, HirConstant::I64(2));
        let one = konst(&mut f, HirConstant::I64(1));

        // entry: c = guard(n, 2); brcond c -> base, rec
        let c = result(&mut f);
        f.blocks
            .get_mut(&entry)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: guard_op,
                result: c,
                ty: HirType::I64,
                left: n,
                right: two,
            });
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::CondBranch {
            condition: c,
            true_target: base,
            false_target: rec,
        };

        // base: return n
        f.blocks.get_mut(&base).unwrap().terminator = HirTerminator::Return { values: vec![n] };

        // rec: a = n arg_op 1 ; r = call self(a) ; return r  (+opt back-edge)
        let a = result(&mut f);
        f.blocks
            .get_mut(&rec)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: arg_op,
                result: a,
                ty: HirType::I64,
                left: n,
                right: one,
            });
        let r = result(&mut f);
        f.blocks
            .get_mut(&rec)
            .unwrap()
            .instructions
            .push(HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Function(fid),
                args: vec![a],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
        f.blocks.get_mut(&rec).unwrap().terminator = if add_back_edge {
            // A back-edge rec -> entry makes the CFG cyclic.
            HirTerminator::Branch { target: entry }
        } else {
            HirTerminator::Return { values: vec![r] }
        };

        m.functions.insert(fid, f);
        (m, fid)
    }

    #[test]
    fn fib_shape_is_speculation_safe() {
        // decreasing arg (n-1), range guard (n < 2), acyclic → total.
        let (mut m, fid) = mk_recursive(BinaryOp::Lt, BinaryOp::Sub, false);
        infer_module(&mut m);
        let safe = speculation_safe_module(&m);
        assert!(
            safe.contains(&fid),
            "fib-shaped fn should be speculation-safe"
        );
    }

    #[test]
    fn equality_base_guard_is_not_safe() {
        // `n == 2` base guard doesn't cover small/negative values, so a
        // speculated call could recurse forever → must be rejected.
        let (mut m, fid) = mk_recursive(BinaryOp::Eq, BinaryOp::Sub, false);
        infer_module(&mut m);
        assert!(!speculation_safe_module(&m).contains(&fid));
    }

    #[test]
    fn increasing_arg_is_not_safe() {
        // self-call passes n+1 (grows away from the base) → not total.
        let (mut m, fid) = mk_recursive(BinaryOp::Lt, BinaryOp::Add, false);
        infer_module(&mut m);
        assert!(!speculation_safe_module(&m).contains(&fid));
    }

    #[test]
    fn looping_cfg_is_not_safe() {
        // A back-edge makes the CFG cyclic; a loop could spin forever.
        let (mut m, fid) = mk_recursive(BinaryOp::Lt, BinaryOp::Sub, true);
        infer_module(&mut m);
        assert!(!speculation_safe_module(&m).contains(&fid));
    }
}
