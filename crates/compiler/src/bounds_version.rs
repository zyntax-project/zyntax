//! Counted loops versioned on the bounds of their indices.
//!
//! A loop `for i in start..N` (a header phi stepped by a positive
//! constant, the header leaving when `i < N` fails, `N` defined outside
//! the loop) whose body compares `i + b` against zero or against a value
//! defined outside the loop, typically a list length a list access tests
//! its index against, gets a second copy in which those compares are
//! constants. The preheader tests once that every index the loop can
//! form lies in `[0, len)`:
//!
//! ```text
//! start + b >= 0                      for each offset b
//! N <= len        (b <= 0)            for each (b, len)
//! len >= b && N <= len - b  (b > 0)
//! N <= i64::MAX - step + 1  (step > 1, so the counter cannot wrap)
//! ```
//!
//! and enters the copy when all hold and the original loop otherwise. In
//! the copy, every iteration runs with `start <= i <= N - 1`, so each
//! folded compare has the value it would have had; the original runs
//! whenever some iteration might fail a check, and raises where it did.
//!
//! The bound is an SSA value defined outside the loop, so the copy
//! compares against exactly what the original does: whether a length
//! may be read once before the loop is the question LICM answered when
//! it hoisted the load, and a loop that may resize its list keeps the
//! load inside and is not versioned here.
//!
//! Only innermost loops with one way in are versioned, and only when the
//! copy decides at least one index against a bound: a sign check alone
//! does not pay for a second copy of the loop. The version test ends
//! the preheader in a branch between two loop headers, which a second
//! run takes as a loop it does not version. Values the loop defines and
//! code after it reads meet in a phi at the exit block that dominates
//! the read; a loop whose exits cannot carry them is left.

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, CastOp, HirBlock, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirPhi,
    HirTerminator, HirType, HirValue, HirValueKind,
};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Largest loop, in instructions, that is copied.
const MAX_VERSIONED_BODY: usize = 256;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BoundsVersionStats {
    /// Loops given a check-free copy.
    pub versioned: usize,
    /// Compares the copies decide.
    pub folded: usize,
}

/// `ZYNTAX_DISABLE_BOUNDS_VERSION=1` leaves every loop as it is; safe
/// to run with.
fn disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_BOUNDS_VERSION").is_some())
}

/// `ZYNTAX_TRACE_BOUNDS_VERSION=1` prints each loop versioned and why a
/// loop with folds was refused; safe to run with.
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZYNTAX_TRACE_BOUNDS_VERSION").is_some())
}

pub fn run_module(module: &mut HirModule) -> BoundsVersionStats {
    let mut total = BoundsVersionStats::default();
    if disabled() {
        return total;
    }
    for func in module.functions_to_optimize() {
        if func.is_external {
            continue;
        }
        let s = run_function(func);
        total.versioned += s.versioned;
        total.folded += s.folded;
    }
    total
}

pub fn run_function(func: &mut HirFunction) -> BoundsVersionStats {
    let mut stats = BoundsVersionStats::default();
    let mut tried: HashSet<HirId> = HashSet::new();
    // A function with no loop to version leaves exactly as it came,
    // edge lists included.
    let edges: Vec<(HirId, Vec<HirId>, Vec<HirId>)> = func
        .blocks
        .iter()
        .map(|(id, b)| (*id, b.predecessors.clone(), b.successors.clone()))
        .collect();
    loop {
        rebuild_cfg_edges(func);
        let dt = DominatorTree::new(func);
        let lf = LoopForest::detect(func, &dt);
        let next = lf
            .loops()
            .iter()
            .find(|lp| !tried.contains(&lp.header) && innermost(lp, &lf));
        let Some(lp) = next else { break };
        tried.insert(lp.header);
        let plan = match recognise(func, &dt, lp) {
            Ok(plan) => plan,
            Err(why) => {
                if trace() {
                    eprintln!(
                        "[bounds_version] {}: loop {:?} kept: {why}",
                        func.name.resolve_global().unwrap_or_default(),
                        lp.header
                    );
                }
                continue;
            }
        };
        if trace() {
            eprintln!(
                "[bounds_version] {}: loop {:?} versioned, {} compares decided",
                func.name.resolve_global().unwrap_or_default(),
                lp.header,
                plan.folds.len()
            );
        }
        stats.versioned += 1;
        stats.folded += plan.folds.len();
        let lp = lp.clone();
        apply(func, &lp, plan, &mut tried);
    }
    if stats.versioned == 0 {
        for (id, preds, succs) in edges {
            if let Some(b) = func.blocks.get_mut(&id) {
                b.predecessors = preds;
                b.successors = succs;
            }
        }
    } else {
        rebuild_cfg_edges(func);
    }
    stats
}

fn innermost(lp: &NaturalLoop, lf: &LoopForest) -> bool {
    lf.loops()
        .iter()
        .all(|other| other.header == lp.header || !lp.body.contains(&other.header))
}

/// What the version test needs of one bound: `idx = i + offset` never
/// negative, and below `len` when there is one.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Fact {
    offset: i64,
    len: Option<HirId>,
}

struct Plan {
    preheader: HirId,
    /// The preheader ends in a conditional branch; the loop's edge gets
    /// a block of its own for the version test.
    split: bool,
    start: HirId,
    limit: HirId,
    step: i64,
    facts: Vec<Fact>,
    /// Compare results in the loop and the constant each is in the copy.
    folds: HashMap<HirId, bool>,
    /// Loop values read after the loop, by the exit block that dominates
    /// the read.
    merges: Vec<(HirId, HirId)>,
}

fn recognise(func: &HirFunction, dt: &DominatorTree, lp: &NaturalLoop) -> Result<Plan, String> {
    let body = &lp.body;
    let header = func.blocks.get(&lp.header).ok_or("no header")?;

    let size: usize = body
        .iter()
        .filter_map(|b| func.blocks.get(b))
        .map(|b| b.instructions.len() + b.phis.len())
        .sum();
    if size > MAX_VERSIONED_BODY {
        return Err("too large".to_string());
    }

    // The unique way in: a plain branch to the header, or one edge of a
    // conditional branch, which gets a block of its own. A branch
    // between the header and another loop's header is a version test
    // already, or a shape this pass does not take.
    let outside: Vec<HirId> = header
        .predecessors
        .iter()
        .copied()
        .filter(|p| !body.contains(p))
        .collect();
    let [preheader] = outside[..] else {
        return Err("no unique preheader".to_string());
    };
    let is_header = |b: HirId| {
        func.blocks
            .get(&b)
            .is_some_and(|blk| blk.predecessors.iter().any(|p| dt.dominates(b, *p)))
    };
    let split = match func.blocks.get(&preheader).map(|b| &b.terminator) {
        Some(HirTerminator::Branch { target }) if *target == lp.header => false,
        Some(HirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        }) if (*true_target == lp.header) != (*false_target == lp.header) => {
            let other = if *true_target == lp.header {
                *false_target
            } else {
                *true_target
            };
            if is_header(other) {
                return Err("preheader chooses between two loops".to_string());
            }
            true
        }
        _ => return Err("preheader does not branch to the header alone".to_string()),
    };

    // Every block and instruction is one the copy can carry.
    for b in body {
        let block = func.blocks.get(b).ok_or("no block")?;
        if !matches!(
            block.terminator,
            HirTerminator::Branch { .. }
                | HirTerminator::CondBranch { .. }
                | HirTerminator::Return { .. }
                | HirTerminator::Unreachable
        ) {
            return Err("a terminator it cannot copy".to_string());
        }
        if !block.instructions.iter().all(copyable) {
            return Err("an instruction it cannot copy".to_string());
        }
        let typed = |r: Option<HirId>| r.is_none_or(|r| func.values.contains_key(&r));
        if !block.instructions.iter().all(|i| typed(i.result_id())) {
            return Err("an untyped result".to_string());
        }
    }

    let mut defined_in: HashMap<HirId, HirId> = HashMap::new();
    let mut defs: HashMap<HirId, &HirInstruction> = HashMap::new();
    for b in body {
        let block = &func.blocks[b];
        for p in &block.phis {
            defined_in.insert(p.result, *b);
        }
        for inst in &block.instructions {
            if let Some(r) = inst.result_id() {
                defined_in.insert(r, *b);
                defs.insert(r, inst);
            }
        }
    }
    let invariant = |v: HirId| !defined_in.contains_key(&v);
    // Casts ahead of the loop, where LICM leaves an invariant one.
    let outside_casts: HashMap<HirId, &HirInstruction> = func
        .blocks
        .iter()
        .filter(|(b, _)| !body.contains(b))
        .flat_map(|(_, block)| block.instructions.iter())
        .filter_map(|inst| match inst {
            HirInstruction::Cast { result, .. } => Some((*result, inst)),
            _ => None,
        })
        .collect();
    let phi_tys: HashMap<HirId, &HirType> = func
        .blocks
        .values()
        .flat_map(|b| b.phis.iter())
        .map(|p| (p.result, &p.ty))
        .collect();
    let is_i64 = |v: HirId| {
        matches!(
            func.values
                .get(&v)
                .map(|v| &v.ty)
                .or(phi_tys.get(&v).copied()),
            Some(HirType::I64)
        )
    };
    let const_i64 = |v: HirId| match func.values.get(&v).map(|v| &v.kind) {
        Some(HirValueKind::Constant(HirConstant::I64(n))) => Some(*n),
        _ => None,
    };

    // The header leaves when `i < N` fails.
    let HirTerminator::CondBranch {
        condition,
        true_target,
        false_target,
    } = header.terminator
    else {
        return Err("header does not end in a branch".to_string());
    };
    let Some(HirInstruction::Binary {
        op, left, right, ..
    }) = header
        .instructions
        .iter()
        .find(|i| i.result_id() == Some(condition))
    else {
        return Err("header branch is not a compare".to_string());
    };
    let t_in = body.contains(&true_target);
    let f_in = body.contains(&false_target);
    let (iv, limit) = match (op, t_in, f_in) {
        (BinaryOp::Lt, true, false) | (BinaryOp::Ge, false, true) => (*left, *right),
        (BinaryOp::Gt, true, false) | (BinaryOp::Le, false, true) => (*right, *left),
        _ => return Err("header compare is not a counted exit".to_string()),
    };
    if !invariant(limit) || !is_i64(limit) || !is_i64(iv) {
        return Err("bound or counter is not i64 or varies".to_string());
    }
    let iv_phi = header
        .phis
        .iter()
        .find(|p| p.result == iv)
        .ok_or("no counter phi")?;
    let mut start = None;
    let mut step = 0i64;
    for (v, from) in &iv_phi.incoming {
        if *from == preheader {
            if start.replace(*v).is_some() {
                return Err("two entries to the counter".to_string());
            }
            continue;
        }
        if !body.contains(from) {
            return Err("counter phi from outside".to_string());
        }
        let c = match defs.get(v) {
            Some(HirInstruction::Binary {
                op: BinaryOp::Add,
                left,
                right,
                ..
            }) if *left == iv => const_i64(*right),
            Some(HirInstruction::Binary {
                op: BinaryOp::Add,
                left,
                right,
                ..
            }) if *right == iv => const_i64(*left),
            _ => None,
        };
        match c {
            Some(c) if c > 0 => step = step.max(c),
            _ => return Err("counter step is not a positive constant".to_string()),
        }
    }
    let start = start.ok_or("no counter phi")?;

    // Compares the copy decides, to a fixed point: a decided branch
    // leaves the copy fewer edges, a phi left with one value there is
    // that value, and that can make another compare one on an index.
    // The header's own compares run once more with `i == N` and are left.
    let mut folds: HashMap<HirId, bool> = HashMap::new();
    let mut facts: Vec<Fact> = Vec::new();
    let order: Vec<HirId> = func
        .blocks
        .keys()
        .copied()
        .filter(|b| body.contains(b))
        .collect();
    loop {
        let taken = |from: HirId, to: HirId| match &func.blocks[&from].terminator {
            HirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => match folds.get(condition) {
                Some(true) => *true_target == to,
                Some(false) => *false_target == to,
                None => true,
            },
            _ => true,
        };
        let mut live: HashSet<HirId> = HashSet::from([lp.header]);
        let mut stack = vec![lp.header];
        while let Some(b) = stack.pop() {
            for t in func.blocks[&b].terminator.targets() {
                if body.contains(&t) && taken(b, t) && live.insert(t) {
                    stack.push(t);
                }
            }
        }
        // `v` as `i + offset` in the iteration that computes it.
        let single_phis: HashMap<HirId, HirId> = order
            .iter()
            .filter(|b| **b != lp.header && live.contains(b))
            .flat_map(|b| {
                let block = &func.blocks[b];
                block.phis.iter().map(move |p| (*b, block, p))
            })
            .filter_map(|(bid, block, p)| {
                let mut values = p
                    .incoming
                    .iter()
                    .filter(|(_, from)| {
                        block.predecessors.contains(from)
                            && live.contains(from)
                            && taken(*from, bid)
                    })
                    .map(|(v, _)| *v);
                let first = values.next()?;
                values.all(|v| v == first).then_some((p.result, first))
            })
            .collect();
        let affine = |mut v: HirId| -> Option<i64> {
            let mut offset = 0i64;
            for _ in 0..16 {
                if v == iv {
                    return (offset.abs() <= 1 << 31).then_some(offset);
                }
                if let Some(&from) = single_phis.get(&v) {
                    v = from;
                    continue;
                }
                match defs.get(&v) {
                    // A select whose condition the copy decides is the
                    // value it picks there.
                    Some(HirInstruction::Select {
                        condition,
                        true_val,
                        false_val,
                        ..
                    }) => {
                        v = match folds.get(condition)? {
                            true => *true_val,
                            false => *false_val,
                        };
                    }
                    Some(HirInstruction::Binary {
                        op: BinaryOp::Add,
                        left,
                        right,
                        ..
                    }) => {
                        if let Some(c) = const_i64(*right) {
                            offset = offset.checked_add(c)?;
                            v = *left;
                        } else if let Some(c) = const_i64(*left) {
                            offset = offset.checked_add(c)?;
                            v = *right;
                        } else {
                            return None;
                        }
                    }
                    Some(HirInstruction::Binary {
                        op: BinaryOp::Sub,
                        left,
                        right,
                        ..
                    }) => {
                        offset = offset.checked_sub(const_i64(*right)?)?;
                        v = *left;
                    }
                    _ => return None,
                }
            }
            None
        };
        let mut decided: Vec<(HirId, bool, Fact)> = Vec::new();
        // An i64 value reinterpreted as u64, wherever it is cast.
        let unsigned_of = |v: HirId| match defs.get(&v).copied().or(outside_casts.get(&v).copied())
        {
            Some(HirInstruction::Cast {
                op: CastOp::Bitcast,
                operand,
                ty: HirType::U64,
                ..
            }) if is_i64(*operand) => Some(*operand),
            _ => None,
        };
        for b in &order {
            if *b == lp.header || !live.contains(b) {
                continue;
            }
            for inst in &func.blocks[b].instructions {
                let HirInstruction::Binary {
                    op,
                    result,
                    left,
                    right,
                    ty,
                } = inst
                else {
                    continue;
                };
                // `idx as u64 < len as u64` holds exactly when
                // `0 <= idx < len`: decided on both facts.
                if *ty == HirType::U64
                    && !folds.contains_key(result)
                    && matches!(op, BinaryOp::Lt | BinaryOp::Ge)
                    && let (Some(idx), Some(bound)) = (unsigned_of(*left), unsigned_of(*right))
                    && let Some(offset) = affine(idx)
                    && const_i64(bound).is_none()
                    && invariant(bound)
                {
                    let truth = *op == BinaryOp::Lt;
                    decided.push((*result, truth, Fact { offset, len: None }));
                    decided.push((
                        *result,
                        truth,
                        Fact {
                            offset,
                            len: Some(bound),
                        },
                    ));
                    continue;
                }
                if folds.contains_key(result)
                    || !matches!(
                        op,
                        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge
                    )
                    || !is_i64(*left)
                    || !is_i64(*right)
                {
                    continue;
                }
                // As `idx op bound`.
                let (op, offset, bound) = if let Some(off) = affine(*left) {
                    (*op, off, *right)
                } else if let Some(off) = affine(*right) {
                    let swapped = match op {
                        BinaryOp::Lt => BinaryOp::Gt,
                        BinaryOp::Le => BinaryOp::Ge,
                        BinaryOp::Gt => BinaryOp::Lt,
                        _ => BinaryOp::Le,
                    };
                    (swapped, off, *left)
                } else {
                    continue;
                };
                let (truth, len) = match const_i64(bound) {
                    // `0 <= idx`.
                    Some(0) => match op {
                        BinaryOp::Lt => (false, None),
                        BinaryOp::Ge => (true, None),
                        _ => continue,
                    },
                    Some(_) => continue,
                    // `idx < len`.
                    None if invariant(bound) => match op {
                        BinaryOp::Lt | BinaryOp::Le => (true, Some(bound)),
                        _ => (false, Some(bound)),
                    },
                    None => continue,
                };
                decided.push((*result, truth, Fact { offset, len }));
            }
        }
        if decided.is_empty() {
            break;
        }
        for (result, truth, fact) in decided {
            folds.insert(result, truth);
            if !facts.contains(&fact) {
                facts.push(fact);
            }
        }
    }
    if folds.is_empty() {
        return Err("no compare decided".to_string());
    }
    // A sign check alone is one compare the copy would save, against a
    // second copy of the loop.
    if facts.iter().all(|f| f.len.is_none()) {
        return Err("no bound decided, only signs".to_string());
    }

    // Reads of loop values after the loop, each through the exit that
    // dominates it and that only the loop enters.
    let exit_for = |at: HirId| -> Option<HirId> {
        lp.exits.iter().copied().find(|e| {
            dt.dominates(*e, at) && func.blocks[e].predecessors.iter().all(|p| body.contains(p))
        })
    };
    let mut merges: Vec<(HirId, HirId)> = Vec::new();
    let mut need = |at: HirId, v: HirId| -> Result<(), String> {
        let e = exit_for(at).ok_or("a value it defines is read past an exit it cannot merge")?;
        if !merges.contains(&(e, v)) {
            merges.push((e, v));
        }
        Ok(())
    };
    for (bid, block) in &func.blocks {
        // A block nothing reaches may read what it likes; it runs never.
        if body.contains(bid) || dt.rpo_position(*bid).is_none() {
            continue;
        }
        for p in &block.phis {
            for (v, from) in &p.incoming {
                if defined_in.contains_key(v)
                    && !body.contains(from)
                    && dt.rpo_position(*from).is_some()
                {
                    need(*from, *v)?;
                }
            }
        }
        let mut reads = Vec::new();
        for inst in &block.instructions {
            inst.for_each_operand(|v| reads.push(v));
        }
        block.terminator.for_each_operand(|v| reads.push(v));
        for v in reads {
            if defined_in.contains_key(&v) {
                need(*bid, v)?;
            }
        }
    }

    Ok(Plan {
        preheader,
        split,
        start,
        limit,
        step,
        facts,
        folds,
        merges,
    })
}

/// Instructions whose copy is the same computation.
fn copyable(inst: &HirInstruction) -> bool {
    matches!(
        inst,
        HirInstruction::Binary { .. }
            | HirInstruction::Unary { .. }
            | HirInstruction::Load { .. }
            | HirInstruction::Store { .. }
            | HirInstruction::GetElementPtr { .. }
            | HirInstruction::Cast { .. }
            | HirInstruction::Select { .. }
            | HirInstruction::ExtractValue { .. }
            | HirInstruction::InsertValue { .. }
            | HirInstruction::Call { .. }
    )
}

fn new_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    func.values.insert(
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

fn apply(func: &mut HirFunction, lp: &NaturalLoop, mut plan: Plan, tried: &mut HashSet<HirId>) {
    if plan.split {
        let edge = HirId::new();
        let mut block = HirBlock::new(edge);
        block.terminator = HirTerminator::Branch { target: lp.header };
        func.blocks.insert(edge, block);
        let pre = func.blocks.get_mut(&plan.preheader).unwrap();
        pre.terminator.retarget(lp.header, edge);
        let header = func.blocks.get_mut(&lp.header).unwrap();
        for p in &mut header.phis {
            for (_, from) in &mut p.incoming {
                if *from == plan.preheader {
                    *from = edge;
                }
            }
        }
        plan.preheader = edge;
        rebuild_cfg_edges(func);
    }
    let body: Vec<HirId> = func
        .blocks
        .keys()
        .copied()
        .filter(|b| lp.body.contains(b))
        .collect();

    // Fresh ids for the copy's blocks and values; a decided compare's
    // copy is a constant.
    let block_map: HashMap<HirId, HirId> = body.iter().map(|b| (*b, HirId::new())).collect();
    let mut value_map: IndexMap<HirId, HirId> = IndexMap::new();
    for b in &body {
        let block = &func.blocks[b];
        let results: Vec<(HirId, HirType)> = block
            .phis
            .iter()
            .map(|p| (p.result, p.ty.clone()))
            .chain(block.instructions.iter().filter_map(|i| {
                let r = i.result_id()?;
                Some((r, func.values.get(&r).map(|v| v.ty.clone())?))
            }))
            .collect();
        for (r, ty) in results {
            let kind = match plan.folds.get(&r) {
                Some(truth) => HirValueKind::Constant(HirConstant::Bool(*truth)),
                None => HirValueKind::Instruction,
            };
            let ty = if plan.folds.contains_key(&r) {
                HirType::Bool
            } else {
                ty
            };
            let id = new_value(func, ty, kind);
            value_map.insert(r, id);
        }
    }
    let map_block = |b: HirId| block_map.get(&b).copied().unwrap_or(b);
    let map_value = |v: HirId| value_map.get(&v).copied().unwrap_or(v);

    let mut copies: Vec<HirBlock> = Vec::new();
    for b in &body {
        let block = &func.blocks[b];
        let mut copy = HirBlock::new(block_map[b]);
        for p in &block.phis {
            copy.phis.push(HirPhi {
                result: map_value(p.result),
                ty: p.ty.clone(),
                incoming: p
                    .incoming
                    .iter()
                    .map(|(v, from)| (map_value(*v), map_block(*from)))
                    .collect(),
            });
        }
        for inst in &block.instructions {
            if inst
                .result_id()
                .is_some_and(|r| plan.folds.contains_key(&r))
            {
                continue;
            }
            let mut inst = inst.clone();
            inst.replace_uses(&value_map);
            if let Some(r) = inst.result_id_mut() {
                *r = map_value(*r);
            }
            copy.instructions.push(inst);
        }
        let mut term = block.terminator.clone();
        term.replace_uses(&value_map);
        match &mut term {
            HirTerminator::Branch { target } => *target = map_block(*target),
            HirTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                *true_target = map_block(*true_target);
                *false_target = map_block(*false_target);
            }
            _ => {}
        }
        copy.terminator = term;
        copies.push(copy);
    }
    let header_copy = block_map[&lp.header];
    let mut copies: crate::hir::IdMap<HirId, HirBlock> =
        copies.into_iter().map(|c| (c.id, c)).collect();
    let alias = simplify_copy(func, &mut copies, header_copy, plan.preheader);
    let resolve = |mut v: HirId| {
        while let Some(&n) = alias.get(&v) {
            v = n;
        }
        v
    };
    let map_value = |v: HirId| resolve(map_value(v));
    let edges: HashSet<(HirId, HirId)> = copies
        .iter()
        .flat_map(|(id, b)| b.terminator.targets().into_iter().map(move |t| (*id, t)))
        .collect();
    // The copy's own edge into `to` from the copy of `from`, if it kept one.
    let copy_edge = |from: HirId, to: HirId| {
        let from = map_block(from);
        edges.contains(&(from, to)).then_some(from)
    };

    // Exits gain the copy's edges; a loop value read after the loop is
    // the phi of both at the exit that dominates the read.
    let exits: Vec<HirId> = lp.exits.iter().copied().collect();
    for e in &exits {
        let Some(block) = func.blocks.get_mut(e) else {
            continue;
        };
        for p in &mut block.phis {
            let extra: Vec<(HirId, HirId)> = p
                .incoming
                .iter()
                .filter(|(_, from)| lp.body.contains(from))
                .filter_map(|(v, from)| Some((map_value(*v), copy_edge(*from, *e)?)))
                .collect();
            p.incoming.extend(extra);
        }
    }
    let dt = DominatorTree::new(func);
    for (e, v) in &plan.merges {
        let ty = func
            .values
            .get(v)
            .map(|x| x.ty.clone())
            .unwrap_or(HirType::I64);
        let merged = new_value(func, ty.clone(), HirValueKind::Instruction);
        let mut preds: Vec<HirId> = func.blocks[e].predecessors.clone();
        preds.dedup();
        let mut incoming = Vec::new();
        for p in preds {
            incoming.push((*v, p));
            if let Some(c) = copy_edge(p, *e) {
                incoming.push((map_value(*v), c));
            }
        }
        let subst: IndexMap<HirId, HirId> = [(*v, merged)].into_iter().collect();
        // A phi reads its value at the end of the edge's source, so the
        // phi's own block need not be one the exit dominates.
        let outside: Vec<HirId> = func
            .blocks
            .keys()
            .copied()
            .filter(|b| !lp.body.contains(b))
            .collect();
        for b in outside {
            let dominated = dt.dominates(*e, b);
            let block = func.blocks.get_mut(&b).unwrap();
            for p in &mut block.phis {
                for (val, from) in &mut p.incoming {
                    if *val == *v && !lp.body.contains(from) && dt.dominates(*e, *from) {
                        *val = merged;
                    }
                }
            }
            if dominated {
                for inst in &mut block.instructions {
                    inst.replace_uses(&subst);
                }
                block.terminator.replace_uses(&subst);
            }
        }
        func.blocks.get_mut(e).unwrap().phis.push(HirPhi {
            result: merged,
            ty,
            incoming,
        });
    }

    // The version test, at the end of the preheader.
    let mut test: Vec<HirInstruction> = Vec::new();
    let mut conds: Vec<HirId> = Vec::new();
    let mut compare = |func: &mut HirFunction,
                       test: &mut Vec<HirInstruction>,
                       op: BinaryOp,
                       left: HirId,
                       right: HirId| {
        let result = new_value(func, HirType::Bool, HirValueKind::Instruction);
        test.push(HirInstruction::Binary {
            op,
            result,
            ty: HirType::I64,
            left,
            right,
        });
        conds.push(result);
    };
    let konst = |func: &mut HirFunction, n: i64| {
        new_value(
            func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(n)),
        )
    };
    let mut offsets: Vec<i64> = plan.facts.iter().map(|f| f.offset).collect();
    offsets.sort_unstable();
    offsets.dedup();
    for off in offsets {
        let k = konst(func, -off);
        compare(func, &mut test, BinaryOp::Ge, plan.start, k);
    }
    for fact in &plan.facts {
        let Some(len) = fact.len else { continue };
        if fact.offset <= 0 {
            compare(func, &mut test, BinaryOp::Le, plan.limit, len);
        } else {
            let k = konst(func, fact.offset);
            compare(func, &mut test, BinaryOp::Ge, len, k);
            let room = new_value(func, HirType::I64, HirValueKind::Instruction);
            test.push(HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: room,
                ty: HirType::I64,
                left: len,
                right: k,
            });
            compare(func, &mut test, BinaryOp::Le, plan.limit, room);
        }
    }
    if plan.step > 1 {
        let k = konst(func, i64::MAX - plan.step + 1);
        compare(func, &mut test, BinaryOp::Le, plan.limit, k);
    }
    let mut ok = conds[0];
    for c in conds[1..].to_vec() {
        let both = new_value(func, HirType::Bool, HirValueKind::Instruction);
        test.push(HirInstruction::Binary {
            op: BinaryOp::And,
            result: both,
            ty: HirType::Bool,
            left: ok,
            right: c,
        });
        ok = both;
    }
    let pre = func.blocks.get_mut(&plan.preheader).unwrap();
    pre.instructions.extend(test);
    pre.terminator = HirTerminator::CondBranch {
        condition: ok,
        true_target: block_map[&lp.header],
        false_target: lp.header,
    };
    for (id, copy) in copies {
        tried.insert(id);
        func.blocks.insert(id, copy);
    }
}

/// Decide the copy's branches on its constant conditions, drop the
/// blocks that leaves unreached and the phi edges from them, and read a
/// phi left with one value as that value. Returns those phis, each with
/// the value it stands for, already substituted in the copy.
fn simplify_copy(
    func: &HirFunction,
    copies: &mut crate::hir::IdMap<HirId, HirBlock>,
    header: HirId,
    preheader: HirId,
) -> HashMap<HirId, HirId> {
    let mut alias: HashMap<HirId, HirId> = HashMap::new();
    let resolve = |alias: &HashMap<HirId, HirId>, mut v: HirId| {
        while let Some(&n) = alias.get(&v) {
            v = n;
        }
        v
    };
    let decided = |v: HirId| match func.values.get(&v).map(|x| &x.kind) {
        Some(HirValueKind::Constant(HirConstant::Bool(b))) => Some(*b),
        _ => None,
    };
    loop {
        let mut changed = false;
        for c in copies.values_mut() {
            if let HirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } = c.terminator
            {
                if let Some(b) = decided(resolve(&alias, condition)) {
                    let target = if b { true_target } else { false_target };
                    c.terminator = HirTerminator::Branch { target };
                    changed = true;
                }
            }
        }
        let mut reached: HashSet<HirId> = HashSet::from([header]);
        let mut stack = vec![header];
        while let Some(b) = stack.pop() {
            for t in copies[&b].terminator.targets() {
                if copies.contains_key(&t) && reached.insert(t) {
                    stack.push(t);
                }
            }
        }
        let before = copies.len();
        copies.retain(|id, _| reached.contains(id));
        changed |= before != copies.len();
        let edges: HashSet<(HirId, HirId)> = copies
            .iter()
            .flat_map(|(id, b)| b.terminator.targets().into_iter().map(move |t| (*id, t)))
            .collect();
        for (id, c) in copies.iter_mut() {
            for p in &mut c.phis {
                let n = p.incoming.len();
                p.incoming.retain(|(_, from)| {
                    (*id == header && *from == preheader) || edges.contains(&(*from, *id))
                });
                changed |= n != p.incoming.len();
            }
            if *id == header {
                continue;
            }
            let mut kept = Vec::new();
            for p in std::mem::take(&mut c.phis) {
                let mut values = p.incoming.iter().map(|(v, _)| resolve(&alias, *v));
                let first = values.next();
                match first {
                    Some(first) if first != p.result && values.all(|v| v == first) => {
                        alias.insert(p.result, first);
                        changed = true;
                    }
                    _ => kept.push(p),
                }
            }
            c.phis = kept;
        }
        if !changed {
            break;
        }
    }
    let subst: IndexMap<HirId, HirId> = alias.keys().map(|k| (*k, resolve(&alias, *k))).collect();
    for c in copies.values_mut() {
        for p in &mut c.phis {
            for (v, _) in &mut p.incoming {
                *v = resolve(&alias, *v);
            }
        }
        for inst in &mut c.instructions {
            inst.replace_uses(&subst);
        }
        c.terminator.replace_uses(&subst);
    }
    alias
}

/// Rebuild every block's predecessor and successor lists from the
/// terminators, which are what the passes keep current.
fn rebuild_cfg_edges(func: &mut HirFunction) {
    let mut preds: HashMap<HirId, Vec<HirId>> = HashMap::new();
    let mut succs: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for (id, block) in &func.blocks {
        let mut targets = block.terminator.targets();
        targets.dedup();
        for t in &targets {
            let p = preds.entry(*t).or_default();
            if !p.contains(id) {
                p.push(*id);
            }
        }
        succs.insert(*id, targets);
    }
    for (id, block) in func.blocks.iter_mut() {
        block.predecessors = preds.remove(id).unwrap_or_default();
        block.successors = succs.remove(id).unwrap_or_default();
    }
}
