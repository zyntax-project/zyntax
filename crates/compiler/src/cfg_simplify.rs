//! HIR control-flow-graph simplification.
//!
//! Walks every function in the module and merges any pair of blocks
//! `pred → succ` where:
//!
//!   * `pred` has exactly one successor (`succ`).
//!   * `succ` has exactly one predecessor (`pred`).
//!   * `succ` has no phi nodes (a phi means another block also
//!     reached `succ` historically; merging would change the
//!     semantics).
//!   * `pred ≠ succ` (no self-loop).
//!   * `succ` is not the function's entry block (would change the
//!     entry-block pointer, an invariant most consumers rely on).
//!
//! When the predicate holds, `succ`'s instructions are appended to
//! `pred`, `pred` inherits `succ`'s terminator, and every block that
//! used to point at `succ` (as a successor of `pred` or as a
//! source-block in a downstream phi) is rewired to point at `pred`.
//! `succ` is then removed from the module.
//!
//! The transformation strictly reduces basic-block count without
//! changing semantics. It typically fires after `const_fold` collapses
//! a `CondBranch` to an unconditional `Branch` — those targets are
//! prime candidates for merging into their predecessor. Repeated
//! application converges fast: the outer fixed-point in
//! `run_interp_safe_opts` calls each pass until none reports work, so
//! a single pass is enough per round.

use crate::hir::IdHashSet;
use crate::hir::{HirBlock, HirFunction, HirId, HirInstruction, HirModule, HirTerminator};
use std::collections::HashSet;

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CfgSimplifyStats {
    /// Number of `succ` blocks absorbed into their predecessor.
    pub merged: usize,
    /// Empty blocks whose predecessors now branch straight to their
    /// target.
    pub threaded: usize,
    /// Blocks with no path from the function entry.
    pub unreachable_removed: usize,
}

/// Drop blocks left disconnected by CFG rewrites, and their stale phi edges.
/// Reachability follows terminators, which are the edges codegen executes.
pub fn prune_unreachable(func: &mut HirFunction) -> usize {
    let mut reachable = HashSet::new();
    let mut pending = vec![func.entry_block];
    while let Some(id) = pending.pop() {
        if !reachable.insert(id) {
            continue;
        }
        if let Some(block) = func.blocks.get(&id) {
            pending.extend(block.terminator.targets());
        }
    }
    let before = func.blocks.len();
    if before == reachable.len() {
        return 0;
    }
    func.blocks.retain(|id, _| reachable.contains(id));
    for block in func.blocks.values_mut() {
        block.predecessors.retain(|id| reachable.contains(id));
        block.successors.retain(|id| reachable.contains(id));
        for phi in &mut block.phis {
            phi.incoming.retain(|(_, pred)| reachable.contains(pred));
        }
    }
    before - func.blocks.len()
}

pub fn prune_unreachable_module(module: &mut HirModule) -> usize {
    module.functions_to_optimize().map(prune_unreachable).sum()
}

/// Run on one function. A merge keeps the merged block where it was and
/// hands it the absorbed block's successor, so one walk in block order
/// merges every chain into its head. Threading one block can let
/// another be threaded, so its walk repeats until one threads nothing.
/// The blocks a walk empties leave the function together at its end.
pub fn run(func: &mut HirFunction) -> CfgSimplifyStats {
    let mut total = CfgSimplifyStats::default();
    let mut gone: IdHashSet<HirId> = IdHashSet::default();
    let order: Vec<HirId> = func.blocks.keys().copied().collect();
    for &pred in &order {
        if gone.contains(&pred) {
            continue;
        }
        while let Some(succ) = mergeable_successor(func, pred) {
            absorb(func, pred, succ);
            gone.insert(succ);
            total.merged += 1;
        }
    }
    drop_blocks(func, &mut gone);
    loop {
        let order: Vec<HirId> = func.blocks.keys().copied().collect();
        let before = total.threaded;
        for &id in &order {
            if threadable(func, id) {
                thread_block(func, id);
                gone.insert(id);
                total.threaded += 1;
            }
        }
        drop_blocks(func, &mut gone);
        if total.threaded == before {
            break;
        }
    }
    total
}

/// Remove the blocks of `gone`, which nothing reaches any more.
fn drop_blocks(func: &mut HirFunction, gone: &mut IdHashSet<HirId>) {
    if !gone.is_empty() {
        func.blocks.retain(|id, _| !gone.contains(id));
        gone.clear();
    }
}

/// Whether `id` is an empty block, holding nothing but a branch, that
/// every predecessor can branch past: a landing pad that costs a taken
/// branch and nothing else. Not the entry, not a block with phis (its
/// target would need them), and not one whose predecessor already
/// reaches the target on another edge, since the target's phis could
/// not tell the two apart.
fn threadable(func: &HirFunction, id: HirId) -> bool {
    let Some(block) = func.blocks.get(&id) else {
        return false;
    };
    if id == func.entry_block || !block.instructions.is_empty() || !block.phis.is_empty() {
        return false;
    }
    let HirTerminator::Branch { target } = block.terminator else {
        return false;
    };
    if target == id || block.predecessors.is_empty() {
        return false;
    }
    block.predecessors.iter().all(|p| {
        func.blocks.get(p).is_some_and(|pb| {
            let targets = pb.terminator.targets();
            targets.iter().filter(|t| **t == id).count() == 1 && !targets.contains(&target)
        })
    })
}

/// Send every predecessor of `empty` to its target. The target's phis
/// take each predecessor in the empty block's place, with the value
/// that came through it. `empty` is left for the caller to remove.
fn thread_block(func: &mut HirFunction, empty: HirId) {
    let (preds, target) = match func.blocks.get(&empty) {
        Some(b) => match b.terminator {
            HirTerminator::Branch { target } => (b.predecessors.clone(), target),
            _ => return,
        },
        None => return,
    };
    for &p in &preds {
        if let Some(pb) = func.blocks.get_mut(&p) {
            pb.terminator.retarget(empty, target);
            for s in pb.successors.iter_mut() {
                if *s == empty {
                    *s = target;
                }
            }
        }
    }
    if let Some(tb) = func.blocks.get_mut(&target) {
        tb.predecessors.retain(|p| *p != empty);
        tb.predecessors.extend(preds.iter().copied());
        for phi in tb.phis.iter_mut() {
            if let Some(pos) = phi.incoming.iter().position(|(_, src)| *src == empty) {
                let (value, _) = phi.incoming.remove(pos);
                phi.incoming.extend(preds.iter().map(|p| (value, *p)));
            }
        }
    }
    if let Some(b) = func.blocks.get_mut(&empty) {
        b.predecessors.clear();
    }
}

/// Run on every function in a module.
pub fn run_module(module: &mut HirModule) -> CfgSimplifyStats {
    let mut total = CfgSimplifyStats::default();
    for func in module.functions_to_optimize() {
        let s = run(func);
        total.merged += s.merged;
        total.threaded += s.threaded;
    }
    total
}

/// The successor `pred` merges with: its only one, reached from it
/// alone, without phis, neither `pred` itself nor the entry.
fn mergeable_successor(func: &HirFunction, pred: HirId) -> Option<HirId> {
    let pred_block = func.blocks.get(&pred)?;
    if pred_block.successors.len() != 1 {
        return None;
    }
    let succ_id = pred_block.successors[0];
    if succ_id == pred || succ_id == func.entry_block {
        return None;
    }
    let succ = func.blocks.get(&succ_id)?;
    if succ.predecessors.len() != 1 || succ.predecessors[0] != pred || !succ.phis.is_empty() {
        return None;
    }
    Some(succ_id)
}

/// Move `succ`'s instructions, terminator and successors into `pred`,
/// and point what `succ` reached back at `pred`. `succ` is left empty
/// for the caller to remove.
fn absorb(func: &mut HirFunction, pred: HirId, succ: HirId) {
    let Some(succ_block) = func.blocks.get_mut(&succ) else {
        return;
    };
    let succ_insts = std::mem::take(&mut succ_block.instructions);
    let succ_term = std::mem::replace(&mut succ_block.terminator, HirTerminator::Unreachable);
    let succ_successors = std::mem::take(&mut succ_block.successors);
    succ_block.predecessors.clear();

    let Some(pred_block) = func.blocks.get_mut(&pred) else {
        return;
    };
    pred_block.instructions.extend(succ_insts);
    pred_block.terminator = succ_term;
    pred_block.successors = succ_successors.clone();

    // Every block reached through `succ` is now reached through `pred`:
    // its predecessor list and its phis' source blocks say so.
    let mut visited = IdHashSet::default();
    for &next_id in &succ_successors {
        if !visited.insert(next_id) {
            continue;
        }
        if let Some(next) = func.blocks.get_mut(&next_id) {
            for p in next.predecessors.iter_mut() {
                if *p == succ {
                    *p = pred;
                }
            }
            for phi in next.phis.iter_mut() {
                for (_, src) in phi.incoming.iter_mut() {
                    if *src == succ {
                        *src = pred;
                    }
                }
            }
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────

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

    fn mk_func() -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global("t"), sig());
        f.blocks.clear();
        f
    }

    fn add_const(f: &mut HirFunction, ty: HirType, c: HirConstant) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Constant(c),
                span: None,
            },
        );
        id
    }

    #[test]
    fn merges_straight_line_pair() {
        // entry → b → exit, where b has no phis and exit is just a
        // return. After merge: entry contains b's insts + exit's
        // return becomes entry's terminator (since b → exit → exit).
        // Actually we have a chain: entry's only succ is b, b's only
        // succ is exit. Both pairs are mergeable. After 2 rounds:
        // entry has all the work, exit and b are gone.
        let mut f = mk_func();
        let entry = HirId::new();
        let b = HirId::new();
        let exit = HirId::new();
        f.entry_block = entry;
        for id in [entry, b, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        let entry_blk = f.blocks.get_mut(&entry).unwrap();
        entry_blk.successors = vec![b];
        entry_blk.terminator = HirTerminator::Branch { target: b };

        let b_blk = f.blocks.get_mut(&b).unwrap();
        b_blk.predecessors = vec![entry];
        b_blk.successors = vec![exit];
        b_blk.terminator = HirTerminator::Branch { target: exit };
        let mid_const = add_const(&mut f, HirType::I64, HirConstant::I64(7));
        let mid_inst = HirId::new();
        f.values.insert(
            mid_inst,
            HirValue {
                id: mid_inst,
                ty: HirType::I64,
                kind: HirValueKind::Instruction,
                span: None,
            },
        );
        f.blocks
            .get_mut(&b)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: mid_inst,
                ty: HirType::I64,
                left: mid_const,
                right: mid_const,
            });

        let exit_blk = f.blocks.get_mut(&exit).unwrap();
        exit_blk.predecessors = vec![b];
        exit_blk.terminator = HirTerminator::Return {
            values: vec![mid_inst],
        };

        let stats = run(&mut f);
        assert_eq!(stats.merged, 2);
        assert_eq!(f.blocks.len(), 1, "everything collapsed into entry");
        assert!(f.blocks.contains_key(&entry));
        let merged_entry = &f.blocks[&entry];
        assert_eq!(merged_entry.instructions.len(), 1, "Add instruction kept");
        assert!(matches!(
            merged_entry.terminator,
            HirTerminator::Return { .. }
        ));
    }

    #[test]
    fn does_not_merge_when_succ_has_phi() {
        // entry → b   (and from elsewhere too)
        //         ↓
        //        exit
        // Even though entry has only one successor, b has a phi — it
        // came from two different historical preds. We must not
        // merge.
        let mut f = mk_func();
        let entry = HirId::new();
        let other = HirId::new();
        let b = HirId::new();
        let exit = HirId::new();
        f.entry_block = entry;
        for id in [entry, other, b, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        f.blocks.get_mut(&entry).unwrap().successors = vec![b];
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: b };
        f.blocks.get_mut(&other).unwrap().successors = vec![b];
        f.blocks.get_mut(&other).unwrap().terminator = HirTerminator::Branch { target: b };
        f.blocks.get_mut(&b).unwrap().predecessors = vec![entry, other];
        f.blocks.get_mut(&b).unwrap().successors = vec![exit];
        f.blocks.get_mut(&b).unwrap().terminator = HirTerminator::Branch { target: exit };
        // Give b a phi node.
        let v = add_const(&mut f, HirType::I64, HirConstant::I64(0));
        f.blocks.get_mut(&b).unwrap().phis.push(crate::hir::HirPhi {
            result: v,
            ty: HirType::I64,
            incoming: vec![(v, entry), (v, other)],
        });
        f.blocks.get_mut(&exit).unwrap().predecessors = vec![b];
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![v] };

        let stats = run(&mut f);
        // b → exit *is* a mergeable pair (b's only succ is exit, exit
        // has no phi). entry → b is NOT mergeable (b has a phi). So
        // exactly 1 merge fires (b absorbs exit).
        assert_eq!(stats.merged, 1);
        // 3 blocks left: entry, other, merged-b.
        assert_eq!(f.blocks.len(), 3);
        assert!(f.blocks.contains_key(&b));
        assert!(!f.blocks.contains_key(&exit));
    }

    #[test]
    fn does_not_merge_entry_block() {
        // entry → entry would never qualify (self-loop check), but
        // succ == entry is also blocked. Confirm.
        let mut f = mk_func();
        let entry = HirId::new();
        let pred = HirId::new();
        f.entry_block = entry;
        for id in [entry, pred] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        f.blocks.get_mut(&pred).unwrap().successors = vec![entry];
        f.blocks.get_mut(&pred).unwrap().terminator = HirTerminator::Branch { target: entry };
        f.blocks.get_mut(&entry).unwrap().predecessors = vec![pred];
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![] };

        let stats = run(&mut f);
        assert_eq!(stats.merged, 0);
        assert_eq!(f.blocks.len(), 2);
    }

    #[test]
    fn does_not_merge_self_loop() {
        let mut f = mk_func();
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.insert(entry, HirBlock::new(entry));
        f.blocks.get_mut(&entry).unwrap().successors = vec![entry];
        f.blocks.get_mut(&entry).unwrap().predecessors = vec![entry];
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: entry };
        let stats = run(&mut f);
        assert_eq!(stats.merged, 0);
    }

    #[test]
    fn prune_unreachable_removes_orphan_phi_edges() {
        let mut f = mk_func();
        let entry = f.entry_block;
        let join = HirId::new();
        let orphan = HirId::new();
        let live_value = add_const(&mut f, HirType::I64, HirConstant::I64(7));
        let dead_value = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let result = HirId::new();
        f.blocks.insert(entry, HirBlock::new(entry));
        f.blocks.insert(join, HirBlock::new(join));
        f.blocks.insert(orphan, HirBlock::new(orphan));
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: join };
        f.blocks.get_mut(&entry).unwrap().successors = vec![join];
        f.blocks.get_mut(&orphan).unwrap().terminator = HirTerminator::Branch { target: join };
        f.blocks.get_mut(&orphan).unwrap().successors = vec![join];
        let join_block = f.blocks.get_mut(&join).unwrap();
        join_block.predecessors = vec![entry, orphan];
        join_block.phis.push(crate::hir::HirPhi {
            result,
            ty: HirType::I64,
            incoming: vec![(live_value, entry), (dead_value, orphan)],
        });
        join_block.terminator = HirTerminator::Return {
            values: vec![result],
        };

        assert_eq!(prune_unreachable(&mut f), 1);
        assert!(!f.blocks.contains_key(&orphan));
        assert_eq!(f.blocks[&join].predecessors, vec![entry]);
        assert_eq!(f.blocks[&join].phis[0].incoming, vec![(live_value, entry)]);
    }
}
