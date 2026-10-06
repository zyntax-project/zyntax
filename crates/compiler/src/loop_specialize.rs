//! A scalar loop state that has become constant stays constant when
//! every reachable backedge proves it. A guarded copy carries that
//! state implicitly; the generic loop can enter it on any iteration.

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::*;
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

const MAX_BODY: usize = 128;

/// `ZYNTAX_DISABLE_LOOP_SPECIALIZE=1` keeps generic loops; safe to run with.
pub fn run_module(module: &mut HirModule) -> usize {
    if std::env::var_os("ZYNTAX_DISABLE_LOOP_SPECIALIZE").is_some() {
        return 0;
    }
    module.functions_to_optimize().map(run_function).sum()
}

pub fn run_function(func: &mut HirFunction) -> usize {
    if func.is_external {
        return 0;
    }
    rebuild_edges(func);
    let dt = DominatorTree::new(func);
    let forest = LoopForest::detect(func, &dt);
    // Copies are not reconsidered; only disjoint innermost loops from
    // the input can acquire a version in this pass.
    let mut loops: Vec<_> = forest
        .loops()
        .iter()
        .filter(|lp| {
            !forest
                .loops()
                .iter()
                .any(|other| other.header != lp.header && lp.body.contains(&other.header))
        })
        .cloned()
        .collect();
    loops.sort_by_key(|lp| dt.rpo_position(lp.header));
    let mut made = 0;
    for lp in loops {
        if made == 2 {
            break;
        }
        let size: usize = lp
            .body
            .iter()
            .map(|b| {
                let block = &func.blocks[b];
                block.instructions.len() + block.phis.len()
            })
            .sum();
        if size > MAX_BODY
            || lp.exits.len() != 1
            || lp.body.iter().any(|b| {
                *b != lp.header
                    && func.blocks[b]
                        .predecessors
                        .iter()
                        .any(|p| !lp.body.contains(p))
            })
        {
            continue;
        }
        if lp
            .body
            .iter()
            .any(|b| func.blocks[b].instructions.iter().any(|i| !copyable(i)))
        {
            continue;
        }
        let dt = DominatorTree::new(func);
        let Some(merges) = exit_merges(func, &lp, &dt) else {
            continue;
        };
        let mut plan = None;
        for phi in &func.blocks[&lp.header].phis {
            if !matches!(phi.ty, HirType::Bool | HirType::I64 | HirType::I32) {
                continue;
            }
            if !tested(func, &lp, phi.result) {
                continue;
            }
            for constant in candidates(func, &lp, phi) {
                if let Some(body) = specialize(func, &lp, phi.result, &constant) {
                    plan = Some((phi.result, constant, body));
                    break;
                }
            }
            if plan.is_some() {
                break;
            }
        }
        let Some((state, constant, body)) = plan else {
            continue;
        };
        // `ZYNTAX_TRACE_LOOP_SPECIALIZE=1` reports selected states; safe to run with.
        if std::env::var_os("ZYNTAX_TRACE_LOOP_SPECIALIZE").is_some() {
            eprintln!(
                "[loop_specialize] {}: {:?} = {:?}",
                func.name.resolve_global().unwrap_or_default(),
                state,
                constant
            );
        }
        apply(func, &lp, state, constant, body, merges, &dt);
        rebuild_edges(func);
        made += 1;
    }
    made
}

fn copyable(i: &HirInstruction) -> bool {
    matches!(
        i,
        HirInstruction::Binary { .. }
            | HirInstruction::Unary { .. }
            | HirInstruction::Load { .. }
            | HirInstruction::Store { .. }
            | HirInstruction::GetElementPtr { .. }
            | HirInstruction::Cast { .. }
            | HirInstruction::Select { .. }
            | HirInstruction::ExtractValue { .. }
            | HirInstruction::InsertValue { .. }
            | HirInstruction::Call {
                callee: HirCallable::Intrinsic(
                    Intrinsic::Floor | Intrinsic::Sqrt | Intrinsic::Fabs | Intrinsic::Fma
                ),
                ..
            }
    )
}

fn tested(func: &HirFunction, lp: &NaturalLoop, value: HirId) -> bool {
    lp.body.iter().any(|b| {
        func.blocks[b].instructions.iter().any(|i| {
            matches!(i, HirInstruction::Binary { op: BinaryOp::Eq | BinaryOp::Ne, left, right, .. }
            if *left == value || *right == value)
        }) || matches!(func.blocks[b].terminator,
        HirTerminator::CondBranch { condition, .. } if condition == value)
    })
}

fn candidates(func: &HirFunction, lp: &NaturalLoop, phi: &HirPhi) -> Vec<HirConstant> {
    let mut choices = Vec::new();
    let mut work: Vec<_> = phi.incoming.iter().map(|(v, _)| *v).collect();
    let mut seen = HashSet::from([phi.result]);
    while let Some(v) = work.pop() {
        if !seen.insert(v) || seen.len() > 32 {
            continue;
        }
        if let Some(HirValue {
            kind: HirValueKind::Constant(c),
            ty,
            ..
        }) = func.values.get(&v)
        {
            if *ty == phi.ty && !choices.contains(c) && choices.len() < 4 {
                choices.push(c.clone());
            }
            continue;
        }
        for b in func.blocks.values().filter(|b| lp.body.contains(&b.id)) {
            for p in &b.phis {
                if p.result == v {
                    work.extend(p.incoming.iter().map(|(v, _)| *v));
                }
            }
            for i in &b.instructions {
                if let HirInstruction::Select {
                    result,
                    true_val,
                    false_val,
                    ..
                } = i
                {
                    if *result == v {
                        work.extend([*true_val, *false_val]);
                    }
                }
            }
        }
    }
    choices
}

/// An analysis copy. Each header phi's entry value is unknown: the
/// generic loop may enter this version after any number of iterations.
fn specialize(
    func: &HirFunction,
    lp: &NaturalLoop,
    state: HirId,
    c: &HirConstant,
) -> Option<HirFunction> {
    let mut body = func.clone();
    body.blocks.retain(|id, _| lp.body.contains(id));
    if body.blocks.values().any(|b| {
        !matches!(
            b.terminator,
            HirTerminator::Branch { .. } | HirTerminator::CondBranch { .. }
        )
    }) {
        return None;
    }
    let entry = HirId::new();
    let mut start = HirBlock::new(entry);
    start.terminator = HirTerminator::Branch { target: lp.header };
    body.blocks.insert(entry, start);
    body.entry_block = entry;
    for exit in &lp.exits {
        body.blocks.insert(*exit, HirBlock::new(*exit));
    }
    for phi in &mut body.blocks.get_mut(&lp.header)?.phis {
        phi.incoming.retain(|(_, from)| lp.body.contains(from));
        // This self-read is only an unknown input to the analysis.
        // The emitted copy receives the generic header's current value.
        phi.incoming.push((phi.result, entry));
    }
    body.values.get_mut(&state)?.kind = HirValueKind::Constant(c.clone());
    rebuild_edges(&mut body);
    let before: usize = lp
        .body
        .iter()
        .map(|b| func.blocks[b].instructions.len())
        .sum();
    for _ in 0..8 {
        let folded = crate::const_fold::fold_function(&mut body).folded;
        let removed = crate::cfg_simplify::prune_unreachable(&mut body);
        rebuild_edges(&mut body);
        let mut constants = Vec::new();
        for block in body.blocks.values() {
            for phi in &block.phis {
                if phi.result == state || phi.incoming.is_empty() {
                    continue;
                }
                let mut values = phi.incoming.iter().map(|(v, _)| match body.values.get(v) {
                    Some(HirValue {
                        kind: HirValueKind::Constant(c),
                        ..
                    }) => Some(c),
                    _ => None,
                });
                if let Some(Some(first)) = values.next() {
                    if values.all(|v| v == Some(first)) {
                        constants.push((phi.result, first.clone()));
                    }
                }
            }
        }
        for block in body.blocks.values() {
            for inst in &block.instructions {
                if let HirInstruction::Select {
                    result,
                    true_val,
                    false_val,
                    ..
                } = inst
                {
                    if let (Some(HirValueKind::Constant(a)), Some(HirValueKind::Constant(b))) = (
                        body.values.get(true_val).map(|v| &v.kind),
                        body.values.get(false_val).map(|v| &v.kind),
                    ) {
                        if a == b {
                            constants.push((*result, a.clone()));
                        }
                    }
                }
            }
        }
        if folded + removed + constants.len() == 0 {
            break;
        }
        for (id, constant) in &constants {
            body.values.get_mut(id)?.kind = HirValueKind::Constant(constant.clone());
        }
        for block in body.blocks.values_mut() {
            block.instructions.retain(|i| {
                i.result_id()
                    .is_none_or(|r| !constants.iter().any(|(id, _)| *id == r))
            });
            block
                .phis
                .retain(|p| !constants.iter().any(|(id, _)| *id == p.result));
        }
    }
    let phi = body
        .blocks
        .get(&lp.header)?
        .phis
        .iter()
        .find(|p| p.result == state)?;
    let backedges: Vec<_> = phi
        .incoming
        .iter()
        .filter(|(_, from)| *from != entry)
        .collect();
    if backedges.is_empty() || backedges.iter().any(|(v, _)| {
        !matches!(body.values.get(v), Some(HirValue { kind: HirValueKind::Constant(value), .. }) if value == c)
    }) { return None; }
    let after: usize = body.blocks.values().map(|b| b.instructions.len()).sum();
    if before.saturating_sub(after) < 4 {
        return None;
    }
    Some(body)
}

/// Loop-defined values read outside its exit phis need an explicit
/// merge at an exit that dominates each read.
fn exit_merges(func: &HirFunction, lp: &NaturalLoop, dt: &DominatorTree) -> Option<Vec<HirId>> {
    let exit = *lp.exits.iter().next()?;
    let defs: HashSet<_> = lp
        .body
        .iter()
        .flat_map(|b| {
            let block = &func.blocks[b];
            block
                .phis
                .iter()
                .map(|p| p.result)
                .chain(block.instructions.iter().filter_map(|i| i.result_id()))
        })
        .collect();
    let exclusive = func.blocks[&exit]
        .predecessors
        .iter()
        .all(|p| lp.body.contains(p));
    let mut merges = Vec::new();
    let mut need = |at, v| {
        if !defs.contains(&v) {
            return Some(());
        }
        if !exclusive || !dt.dominates(exit, at) {
            return None;
        }
        if !merges.contains(&v) {
            merges.push(v);
        }
        Some(())
    };
    for (id, block) in &func.blocks {
        if lp.body.contains(id) || dt.rpo_position(*id).is_none() {
            continue;
        }
        for phi in &block.phis {
            for (v, from) in &phi.incoming {
                if !lp.body.contains(from) {
                    need(*from, *v)?;
                }
            }
        }
        let mut uses = Vec::new();
        for i in &block.instructions {
            i.for_each_operand(|v| uses.push(v));
        }
        block.terminator.for_each_operand(|v| uses.push(v));
        for v in uses {
            need(*id, v)?;
        }
    }
    Some(merges)
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

fn apply(
    func: &mut HirFunction,
    lp: &NaturalLoop,
    state: HirId,
    constant: HirConstant,
    body: HirFunction,
    merges: Vec<HirId>,
    dt: &DominatorTree,
) {
    let order: Vec<_> = func
        .blocks
        .keys()
        .filter(|b| lp.body.contains(b))
        .copied()
        .collect();
    let blocks: HashMap<_, _> = order
        .iter()
        .filter(|b| body.blocks.contains_key(*b))
        .map(|b| (*b, HirId::new()))
        .collect();
    let mut values = IndexMap::new();
    for b in &order {
        let defs: Vec<_> = func.blocks[b]
            .phis
            .iter()
            .map(|p| p.result)
            .chain(
                func.blocks[b]
                    .instructions
                    .iter()
                    .filter_map(|i| i.result_id()),
            )
            .collect();
        for id in defs {
            let v = &body.values[&id];
            let fresh = new_value(func, v.ty.clone(), v.kind.clone());
            values.insert(id, fresh);
        }
    }
    let bv = |b| blocks.get(&b).copied().unwrap_or(b);
    let vv = |v| values.get(&v).copied().unwrap_or(v);
    let mut copies = Vec::new();
    for b in &order {
        let Some(source) = body.blocks.get(b) else {
            continue;
        };
        let mut copy = HirBlock::new(bv(*b));
        for phi in &source.phis {
            if matches!(body.values[&phi.result].kind, HirValueKind::Constant(_)) {
                continue;
            }
            let incoming = phi
                .incoming
                .iter()
                .map(|(v, from)| {
                    if *from == body.entry_block {
                        (phi.result, lp.header)
                    } else {
                        (vv(*v), bv(*from))
                    }
                })
                .collect();
            copy.phis.push(HirPhi {
                result: vv(phi.result),
                ty: phi.ty.clone(),
                incoming,
            });
        }
        for i in &source.instructions {
            let mut copy_i = i.clone();
            copy_i.replace_uses(&values);
            if let Some(r) = copy_i.result_id_mut() {
                *r = vv(*r);
            }
            copy.instructions.push(copy_i);
        }
        copy.terminator = source.terminator.clone();
        copy.terminator.replace_uses(&values);
        for old in source.terminator.targets() {
            copy.terminator.retarget(old, bv(old));
        }
        copies.push(copy);
    }
    let exit = *lp.exits.iter().next().unwrap();
    let extra_edges: HashMap<_, _> = order
        .iter()
        .filter_map(|b| {
            body.blocks
                .get(b)?
                .terminator
                .targets()
                .contains(&exit)
                .then_some((*b, bv(*b)))
        })
        .collect();
    let mut exit_phis = func.blocks[&exit].phis.clone();
    for phi in &mut exit_phis {
        let extra: Vec<_> = phi
            .incoming
            .iter()
            .filter_map(|(v, from)| Some((vv(*v), *extra_edges.get(from)?)))
            .collect();
        phi.incoming.extend(extra);
    }
    func.blocks.get_mut(&exit).unwrap().phis = exit_phis;
    for v in merges {
        let ty = func.values[&v].ty.clone();
        let merged = new_value(func, ty.clone(), HirValueKind::Instruction);
        let incoming = func.blocks[&exit]
            .predecessors
            .iter()
            .flat_map(|p| {
                let mut pairs = vec![(v, *p)];
                if let Some(copy) = extra_edges.get(p) {
                    pairs.push((vv(v), *copy));
                }
                pairs
            })
            .collect();
        let subst = IndexMap::from([(v, merged)]);
        for (b, block) in &mut func.blocks {
            if lp.body.contains(b) {
                continue;
            }
            for phi in &mut block.phis {
                for (value, from) in &mut phi.incoming {
                    if *value == v && !lp.body.contains(from) && dt.dominates(exit, *from) {
                        *value = merged;
                    }
                }
            }
            if dt.dominates(exit, *b) {
                for i in &mut block.instructions {
                    i.replace_uses(&subst);
                }
                block.terminator.replace_uses(&subst);
            }
        }
        func.blocks.get_mut(&exit).unwrap().phis.push(HirPhi {
            result: merged,
            ty,
            incoming,
        });
    }
    // Keep the generic header's phis as the current state at the guard.
    // Its instructions and outgoing edges move to the slow body block.
    let slow = HirId::new();
    let mut slow_block = HirBlock::new(slow);
    let guard_value = new_value(
        func,
        func.values[&state].ty.clone(),
        HirValueKind::Constant(constant),
    );
    let condition = new_value(func, HirType::Bool, HirValueKind::Instruction);
    let header = func.blocks.get_mut(&lp.header).unwrap();
    slow_block.instructions = std::mem::take(&mut header.instructions);
    slow_block.terminator = header.terminator.clone();
    header.instructions.push(HirInstruction::Binary {
        op: BinaryOp::Eq,
        result: condition,
        ty: func.values[&state].ty.clone(),
        left: state,
        right: guard_value,
    });
    header.terminator = HirTerminator::CondBranch {
        condition,
        true_target: bv(lp.header),
        false_target: slow,
    };
    for block in func.blocks.values_mut() {
        for phi in &mut block.phis {
            for (_, from) in &mut phi.incoming {
                if *from == lp.header {
                    *from = slow;
                }
            }
        }
    }
    func.blocks.insert(slow, slow_block);
    for copy in copies {
        func.blocks.insert(copy.id, copy);
    }
}

fn rebuild_edges(func: &mut HirFunction) {
    for b in func.blocks.values_mut() {
        b.predecessors.clear();
        b.successors.clear();
    }
    let edges: Vec<_> = func
        .blocks
        .iter()
        .flat_map(|(id, b)| b.terminator.targets().into_iter().map(move |to| (*id, to)))
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
