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

use crate::hir::{HirBlock, HirFunction, HirId, HirInstruction, HirModule, HirTerminator};
use std::collections::HashSet;

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CfgSimplifyStats {
    /// Number of `succ` blocks absorbed into their predecessor.
    pub merged: usize,
}

/// Run on one function. Iterates until no merge fires (covers chains
/// of mergeable blocks; e.g. A → B → C where both pairs qualify
/// collapses to a single block in one pass).
pub fn run(func: &mut HirFunction) -> CfgSimplifyStats {
    let mut total = CfgSimplifyStats::default();
    for _ in 0..32 {
        let pair = find_mergeable_pair(func);
        let (pred, succ) = match pair {
            Some(p) => p,
            None => break,
        };
        if !merge_pair(func, pred, succ) {
            break;
        }
        total.merged += 1;
    }
    total
}

/// Run on every function in a module.
pub fn run_module(module: &mut HirModule) -> CfgSimplifyStats {
    let mut total = CfgSimplifyStats::default();
    for func in module.functions.values_mut() {
        let s = run(func);
        total.merged += s.merged;
    }
    total
}

fn find_mergeable_pair(func: &HirFunction) -> Option<(HirId, HirId)> {
    for (&pred_id, pred) in &func.blocks {
        if pred.successors.len() != 1 {
            continue;
        }
        let succ_id = pred.successors[0];
        if succ_id == pred_id {
            continue; // self-loop, skip
        }
        if succ_id == func.entry_block {
            continue; // never merge the entry
        }
        let succ = match func.blocks.get(&succ_id) {
            Some(b) => b,
            None => continue,
        };
        if succ.predecessors.len() != 1 || succ.predecessors[0] != pred_id {
            continue;
        }
        if !succ.phis.is_empty() {
            continue;
        }
        return Some((pred_id, succ_id));
    }
    None
}

/// Merge `succ` into `pred`. Returns false if the merge isn't
/// applicable (e.g., a block disappeared between pair discovery and
/// application).
fn merge_pair(func: &mut HirFunction, pred: HirId, succ: HirId) -> bool {
    // Snapshot what we need from `succ` before removing it.
    let (succ_insts, succ_term, succ_successors) = match func.blocks.get(&succ) {
        Some(b) => (
            b.instructions.clone(),
            b.terminator.clone(),
            b.successors.clone(),
        ),
        None => return false,
    };

    // 1. Append succ's instructions to pred + adopt succ's
    //    terminator + successor list.
    let pred_block = match func.blocks.get_mut(&pred) {
        Some(b) => b,
        None => return false,
    };
    pred_block.instructions.extend(succ_insts);
    pred_block.terminator = succ_term;
    pred_block.successors = succ_successors.clone();

    // 2. Every block previously reached only-via-succ now reaches
    //    only-via-pred. Update their predecessor lists.
    let mut visited = HashSet::new();
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
            // Also rewrite phi-incoming source-block ids.
            for phi in next.phis.iter_mut() {
                for (_, src) in phi.incoming.iter_mut() {
                    if *src == succ {
                        *src = pred;
                    }
                }
            }
        }
    }

    // 3. Remove succ from the function.
    func.blocks.shift_remove(&succ);
    true
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
                uses: Default::default(),
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
                uses: Default::default(),
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
}
