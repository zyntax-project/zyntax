//! Branches on a phi of constants, taken from the predecessors that
//! supply the constants.
//!
//! A short-circuit `a and b` in a condition merges a boolean: `false`
//! from the arm that skipped `b`, the value of `b` from the arm that
//! computed it. The merge block holds the phi and branches on it. A
//! predecessor that supplies a constant already knows which way that
//! branch goes, so it jumps to that target itself, and the merge is left
//! for the predecessors whose value is not known.
//!
//! The payoff is what the straightened paths let other passes see: on
//! the path where `0 <= x < width` held, the same test in the next
//! access is dominated by the first and CSE and `branch_fold` remove it.
//! Through the merged boolean they could not.
//!
//! Only a merge that is nothing but phis and the branch is threaded, so
//! a predecessor skips no work; a loop header is left alone; and the
//! phis of the block jumped to take from the predecessor what they took
//! from the merge, through the merge's own phis.

use crate::analysis::DominatorTree;
use crate::error_flag::dominance_holds;
use crate::hir::IdHashMap as HashMap;
use crate::hir::{HirConstant, HirFunction, HirId, HirModule, HirTerminator, HirValueKind};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JumpThreadStats {
    /// Edges sent past a merge to the target its constant decided.
    pub threaded: usize,
}

pub fn run_module(module: &mut HirModule) -> JumpThreadStats {
    let mut total = JumpThreadStats::default();
    // `ZYNTAX_DISABLE_JUMP_THREAD=1` skips the pass, to bisect a
    // miscompile; safe to run with.
    if std::env::var_os("ZYNTAX_DISABLE_JUMP_THREAD").is_some() {
        return total;
    }
    for func in module.functions_to_optimize() {
        total.threaded += run(func).threaded;
    }
    total
}

/// Most rounds of threading one function gets: a chain of merges,
/// as `a and b and c` makes, is threaded a link per round.
const MAX_ROUNDS: usize = 6;

pub fn run(func: &mut HirFunction) -> JumpThreadStats {
    let mut stats = JumpThreadStats::default();
    for _ in 0..MAX_ROUNDS {
        let n = thread_round(func);
        if n == 0 {
            break;
        }
        stats.threaded += n;
    }
    // `ZYNTAX_TRACE_JUMP_THREAD=1` names each function the pass changed.
    if stats.threaded > 0 && std::env::var_os("ZYNTAX_TRACE_JUMP_THREAD").is_some() {
        eprintln!(
            "[jump_thread] {}: {} edge(s)",
            func.name.resolve_global().unwrap_or_default(),
            stats.threaded
        );
    }
    stats
}

/// One planned rewrite: `pred` jumps to `target` instead of to `merge`.
struct Thread {
    pred: HirId,
    merge: HirId,
    target: HirId,
}

fn bool_constant(func: &HirFunction, id: HirId) -> Option<bool> {
    match func.values.get(&id).map(|v| &v.kind) {
        Some(HirValueKind::Constant(HirConstant::Bool(b))) => Some(*b),
        _ => None,
    }
}

/// The merges worth a look: nothing but phis and a branch on one of
/// them that some predecessor supplies a constant for.
fn candidates(func: &HirFunction) -> Vec<HirId> {
    func.blocks
        .iter()
        .filter(|(id, block)| {
            let HirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } = block.terminator
            else {
                return false;
            };
            **id != func.entry_block
                && true_target != false_target
                && block.instructions.is_empty()
                && block
                    .phis
                    .iter()
                    .find(|p| p.result == condition)
                    .is_some_and(|p| {
                        p.incoming
                            .iter()
                            .any(|(v, _)| bool_constant(func, *v).is_some())
                    })
        })
        .map(|(&id, _)| id)
        .collect()
}

fn thread_round(func: &mut HirFunction) -> usize {
    let merges = candidates(func);
    if merges.is_empty() {
        return 0;
    }
    let dt = DominatorTree::of(func);
    let uses = phi_uses(func, &merges);
    let mut plan: Vec<Thread> = Vec::new();
    let mut planned_preds: Vec<HirId> = Vec::new();
    for merge in merges {
        let block = &func.blocks[&merge];
        let HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = block.terminator
        else {
            continue;
        };
        let Some(phi) = block.phis.iter().find(|p| p.result == condition) else {
            continue;
        };
        // A loop header keeps the predecessors it has.
        if phi
            .incoming
            .iter()
            .any(|(_, pred)| dt.dominates(merge, *pred))
        {
            continue;
        }
        if !phis_used_only_here(func, &uses, merge, true_target, false_target) {
            continue;
        }
        for &(value, pred) in &phi.incoming {
            let Some(taken) = bool_constant(func, value) else {
                continue;
            };
            let target = if taken { true_target } else { false_target };
            if target == merge || pred == merge || planned_preds.contains(&pred) {
                continue;
            }
            let Some(pred_block) = func.blocks.get(&pred) else {
                continue;
            };
            let edges = pred_block.terminator.targets();
            // One edge into the merge, none yet to the target, whose
            // phis could not tell the two apart.
            let into_merge = edges.iter().filter(|t| **t == merge).count();
            if !matches!(
                pred_block.terminator,
                HirTerminator::Branch { .. } | HirTerminator::CondBranch { .. }
            ) || into_merge != 1
                || edges.contains(&target)
            {
                continue;
            }
            planned_preds.push(pred);
            plan.push(Thread {
                pred,
                merge,
                target,
            });
        }
    }
    if plan.is_empty() {
        return 0;
    }

    // A rewrite that broke dominance would be a bug in this pass, since
    // the merge defines only its phis and they are read nowhere else; a
    // debug build checks and restores.
    let saved = cfg!(debug_assertions).then(|| func.blocks.clone());
    let mut threaded = 0;
    for Thread {
        pred,
        merge,
        target,
    } in plan
    {
        // What each of the merge's phis is, coming from `pred`.
        let through: HashMap<HirId, HirId> = func.blocks[&merge]
            .phis
            .iter()
            .filter_map(|phi| {
                phi.incoming
                    .iter()
                    .find(|(_, from)| *from == pred)
                    .map(|(v, _)| (phi.result, *v))
            })
            .collect();
        let mut incoming = Vec::new();
        let mut ok = true;
        for phi in &func.blocks[&target].phis {
            match phi.incoming.iter().find(|(_, from)| *from == merge) {
                Some(&(v, _)) => incoming.push(through.get(&v).copied().unwrap_or(v)),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        let target_block = func.blocks.get_mut(&target).expect("target");
        for (phi, v) in target_block.phis.iter_mut().zip(incoming) {
            phi.incoming.push((v, pred));
        }
        let merge_block = func.blocks.get_mut(&merge).expect("merge");
        for phi in &mut merge_block.phis {
            phi.incoming.retain(|(_, from)| *from != pred);
        }
        func.blocks
            .get_mut(&pred)
            .expect("pred")
            .terminator
            .retarget(merge, target);
        threaded += 1;
    }
    if threaded == 0 {
        // Every planned edge was skipped before anything changed.
        return 0;
    }
    func.rebuild_cfg_edges();
    if let Some(saved) = saved
        && !dominance_holds(func)
    {
        eprintln!(
            "[jump_thread] {}: rolled back {threaded} edge(s)",
            func.name.resolve_global().unwrap_or_default()
        );
        func.blocks = saved;
        func.rebuild_cfg_edges();
        debug_assert!(false, "jump threading broke dominance");
        return 0;
    }
    threaded
}

/// How a block reads a value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Read {
    Phi,
    Instruction,
    Terminator,
}

/// Every read of the phis of `merges`, by value: the block that reads it
/// and how, from one walk of the function.
fn phi_uses(func: &HirFunction, merges: &[HirId]) -> HashMap<HirId, Vec<(HirId, Read)>> {
    let mut uses: HashMap<HirId, Vec<(HirId, Read)>> = HashMap::default();
    for merge in merges {
        for phi in &func.blocks[merge].phis {
            uses.insert(phi.result, Vec::new());
        }
    }
    for (&id, block) in &func.blocks {
        for phi in &block.phis {
            for (v, _) in &phi.incoming {
                if let Some(reads) = uses.get_mut(v) {
                    reads.push((id, Read::Phi));
                }
            }
        }
        for inst in &block.instructions {
            inst.for_each_operand(|v| {
                if let Some(reads) = uses.get_mut(&v) {
                    reads.push((id, Read::Instruction));
                }
            });
        }
        block.terminator.for_each_operand(|v| {
            if let Some(reads) = uses.get_mut(&v) {
                reads.push((id, Read::Terminator));
            }
        });
    }
    uses
}

/// Whether the phis of `merge` are read only by its branch and by the
/// phis of the two targets: anywhere else they would be reached by a
/// path that skips the merge.
fn phis_used_only_here(
    func: &HirFunction,
    uses: &HashMap<HirId, Vec<(HirId, Read)>>,
    merge: HirId,
    yes: HirId,
    no: HirId,
) -> bool {
    func.blocks[&merge].phis.iter().all(|phi| {
        uses.get(&phi.result).is_none_or(|reads| {
            reads.iter().all(|&(block, how)| match how {
                Read::Phi => block == yes || block == no,
                Read::Terminator => block == merge,
                Read::Instruction => false,
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirFunctionSignature, HirInstruction, HirPhi, HirType, HirValue,
    };
    use zyntax_typed_ast::InternedString;

    fn function() -> HirFunction {
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::Void],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        };
        let mut f = HirFunction::new(InternedString::new_global("f"), sig);
        f.blocks.clear();
        f
    }

    fn value(f: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                span: None,
            },
        );
        id
    }

    fn block(f: &mut HirFunction, terminator: HirTerminator) -> HirId {
        let id = HirId::new();
        let mut b = HirBlock::new(id);
        b.terminator = terminator;
        f.blocks.insert(id, b);
        id
    }

    fn constant(f: &mut HirFunction, b: bool) -> HirId {
        value(
            f,
            HirType::Bool,
            HirValueKind::Constant(HirConstant::Bool(b)),
        )
    }

    /// `if a and b: yes else: no`, lowered as the builder does:
    /// entry: brcond a, eval_b, merge; eval_b: br merge (b computed there);
    /// merge: m = phi [b from eval_b], [false from entry]; brcond m, yes, no.
    fn short_circuit() -> (HirFunction, HirId, HirId, HirId, HirId) {
        let mut f = function();
        let a = value(&mut f, HirType::Bool, HirValueKind::Parameter(0));
        let b = value(&mut f, HirType::Bool, HirValueKind::Parameter(1));
        let no_value = constant(&mut f, false);
        let ret = HirTerminator::Return { values: Vec::new() };
        let yes = block(&mut f, ret.clone());
        let no = block(&mut f, ret);
        let m = HirId::new();
        let merge = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: m,
                true_target: yes,
                false_target: no,
            },
        );
        let eval_b = block(&mut f, HirTerminator::Branch { target: merge });
        let entry = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: a,
                true_target: eval_b,
                false_target: merge,
            },
        );
        f.values.insert(
            m,
            HirValue {
                id: m,
                ty: HirType::Bool,
                kind: HirValueKind::Instruction,
                span: None,
            },
        );
        f.blocks.get_mut(&merge).unwrap().phis.push(HirPhi {
            result: m,
            ty: HirType::Bool,
            incoming: vec![(b, eval_b), (no_value, entry)],
        });
        f.entry_block = entry;
        (f, entry, eval_b, merge, no)
    }

    #[test]
    fn a_predecessor_with_the_constant_jumps_to_the_decided_target() {
        let (mut f, entry, eval_b, merge, no) = short_circuit();
        let stats = run(&mut f);
        assert_eq!(stats.threaded, 1);
        // The arm that skipped `b` goes to `no` itself.
        assert_eq!(
            f.blocks[&entry].terminator.targets(),
            vec![eval_b, no],
            "false edge now reaches the else arm"
        );
        // The merge keeps the one predecessor whose value is computed.
        assert_eq!(f.blocks[&merge].predecessors, vec![eval_b]);
        assert_eq!(f.blocks[&merge].phis[0].incoming.len(), 1);
    }

    #[test]
    fn a_merge_that_computes_something_is_left_alone() {
        let (mut f, _, _, merge, _) = short_circuit();
        let x = value(&mut f, HirType::I64, HirValueKind::Instruction);
        let one = value(
            &mut f,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(1)),
        );
        f.blocks
            .get_mut(&merge)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: x,
                ty: HirType::I64,
                left: one,
                right: one,
            });
        assert_eq!(run(&mut f).threaded, 0);
    }

    #[test]
    fn a_phi_read_past_the_branch_is_not_bypassed() {
        // A use of the merged boolean after the merge would be reached
        // without the merge on the threaded path.
        let (mut f, _, _, merge, no) = short_circuit();
        let m = f.blocks[&merge].phis[0].result;
        let copy = value(&mut f, HirType::Bool, HirValueKind::Instruction);
        f.blocks
            .get_mut(&no)
            .unwrap()
            .instructions
            .push(HirInstruction::Unary {
                op: crate::hir::UnaryOp::Not,
                result: copy,
                ty: HirType::Bool,
                operand: m,
            });
        assert_eq!(run(&mut f).threaded, 0);
    }
}
