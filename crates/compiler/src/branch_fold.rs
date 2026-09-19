//! Conditional branches a dominating branch has already decided.
//!
//! A block entered only through the true edge of `brcond %c` runs with
//! `%c` true, and so does everything it dominates; a `brcond %c` there
//! is a jump to its true target. The false edge decides `%c` false the
//! same way, and `xor %c, true` is `%c` the other way round.
//!
//! What makes this fire is CSE having made repeated compares one value:
//! a method that reads `self.x` then `self.y` checks the object for
//! None before each read, and only the first check survives.

use crate::analysis::DominatorTree;
use crate::hir::{
    BinaryOp, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirTerminator,
    HirValueKind,
};
use std::collections::HashMap;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BranchFoldStats {
    /// Conditional branches made unconditional.
    pub folded: usize,
}

pub fn run_module(module: &mut HirModule) -> BranchFoldStats {
    let mut total = BranchFoldStats::default();
    // `ZYNTAX_DISABLE_BRANCH_FOLD=1` skips the pass, to bisect a
    // miscompile; safe to run with.
    if std::env::var_os("ZYNTAX_DISABLE_BRANCH_FOLD").is_some() {
        return total;
    }
    for func in module.functions_to_optimize() {
        total.folded += run(func).folded;
    }
    total
}

pub fn run(func: &mut HirFunction) -> BranchFoldStats {
    if func.blocks.len() < 3 {
        return BranchFoldStats::default();
    }
    func.rebuild_cfg_edges();
    let dt = DominatorTree::new(func);
    // `xor %c, true` results by the `%c` they negate.
    let mut negations: HashMap<HirId, HirId> = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary {
                op: BinaryOp::Xor,
                result,
                left,
                right,
                ..
            } = inst
            {
                if is_true(func, *right) {
                    negations.insert(*result, *left);
                } else if is_true(func, *left) {
                    negations.insert(*result, *right);
                }
            }
        }
    }

    let mut known: Vec<(HirId, bool)> = Vec::new();
    let mut folds: Vec<(HirId, bool)> = Vec::new();
    visit(func, &dt, dt.entry(), &negations, &mut known, &mut folds);

    let mut stats = BranchFoldStats::default();
    for (block_id, value) in folds {
        let Some(block) = func.blocks.get(&block_id) else {
            continue;
        };
        let HirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } = block.terminator
        else {
            continue;
        };
        let (taken, dropped) = if value {
            (true_target, false_target)
        } else {
            (false_target, true_target)
        };
        if let Some(block) = func.blocks.get_mut(&block_id) {
            block.terminator = HirTerminator::Branch { target: taken };
            block.successors = vec![taken];
        }
        if dropped != taken
            && let Some(other) = func.blocks.get_mut(&dropped)
        {
            other.predecessors.retain(|p| *p != block_id);
            for phi in &mut other.phis {
                phi.incoming.retain(|(_, from)| *from != block_id);
            }
        }
        stats.folded += 1;
    }
    stats
}

/// Walk the dominator tree from `block`, `known` holding the conditions
/// decided on the way here, and note each conditional branch among them.
fn visit(
    func: &HirFunction,
    dt: &DominatorTree,
    block: HirId,
    negations: &HashMap<HirId, HirId>,
    known: &mut Vec<(HirId, bool)>,
    folds: &mut Vec<(HirId, bool)>,
) {
    let Some(b) = func.blocks.get(&block) else {
        return;
    };
    let decided = match &b.terminator {
        HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => {
            if let Some(value) = decided_value(*condition, negations, known) {
                folds.push((block, value));
            }
            Some((*condition, *true_target, *false_target))
        }
        _ => None,
    };
    for &child in dt.children(block) {
        // The child runs with the condition decided when this block's
        // branch is its only way in, through one of the two edges.
        let fact = decided.and_then(|(cond, t, f)| {
            let sole = func
                .blocks
                .get(&child)
                .is_some_and(|c| c.predecessors.len() == 1 && c.predecessors[0] == block);
            match (sole, child == t, child == f) {
                (true, true, false) => Some((cond, true)),
                (true, false, true) => Some((cond, false)),
                _ => None,
            }
        });
        let depth = known.len();
        if let Some((cond, value)) = fact {
            known.push((cond, value));
            if let Some(negated) = negations.get(&cond) {
                known.push((*negated, !value));
            }
        }
        visit(func, dt, child, negations, known, folds);
        known.truncate(depth);
    }
}

/// What `cond` is here, if a branch on the way decided it, directly or
/// through the value it negates.
fn decided_value(
    cond: HirId,
    negations: &HashMap<HirId, HirId>,
    known: &[(HirId, bool)],
) -> Option<bool> {
    if let Some(v) = known.iter().rev().find(|(c, _)| *c == cond) {
        return Some(v.1);
    }
    let negated = negations.get(&cond)?;
    known
        .iter()
        .rev()
        .find(|(c, _)| c == negated)
        .map(|(_, v)| !v)
}

fn is_true(func: &HirFunction, id: HirId) -> bool {
    matches!(
        func.values.get(&id).map(|v| &v.kind),
        Some(HirValueKind::Constant(HirConstant::Bool(true)))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirType, HirValue};
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
                uses: Default::default(),
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

    /// entry: brcond c, a, b; a: brcond c, x, y; x, y, b: return.
    /// The branch in `a` is decided: `a` is entered only when c is true.
    #[test]
    fn a_branch_on_a_condition_the_dominating_branch_decided_folds() {
        let mut f = function();
        let c = value(&mut f, HirType::Bool, HirValueKind::Parameter(0));
        let ret = HirTerminator::Return { values: Vec::new() };
        let x = block(&mut f, ret.clone());
        let y = block(&mut f, ret.clone());
        let b = block(&mut f, ret.clone());
        let a = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: c,
                true_target: x,
                false_target: y,
            },
        );
        let entry = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: c,
                true_target: a,
                false_target: b,
            },
        );
        f.entry_block = entry;
        let stats = run(&mut f);
        assert_eq!(stats.folded, 1);
        assert!(matches!(f.blocks[&a].terminator, HirTerminator::Branch { target } if target == x));
        assert_eq!(f.blocks[&y].predecessors, Vec::<HirId>::new());
    }

    /// entry: brcond c, a, b; b: brcond n, x, y with n = xor c, true.
    /// In `b` c is false, so n is true.
    #[test]
    fn a_negated_condition_is_decided_too() {
        let mut f = function();
        let c = value(&mut f, HirType::Bool, HirValueKind::Parameter(0));
        let t = value(
            &mut f,
            HirType::Bool,
            HirValueKind::Constant(HirConstant::Bool(true)),
        );
        let n = value(&mut f, HirType::Bool, HirValueKind::Instruction);
        let ret = HirTerminator::Return { values: Vec::new() };
        let x = block(&mut f, ret.clone());
        let y = block(&mut f, ret.clone());
        let a = block(&mut f, ret.clone());
        let b = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: n,
                true_target: x,
                false_target: y,
            },
        );
        f.blocks
            .get_mut(&b)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Xor,
                result: n,
                ty: HirType::Bool,
                left: c,
                right: t,
            });
        let entry = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: c,
                true_target: a,
                false_target: b,
            },
        );
        f.entry_block = entry;
        let stats = run(&mut f);
        assert_eq!(stats.folded, 1);
        assert!(matches!(f.blocks[&b].terminator, HirTerminator::Branch { target } if target == x));
    }

    /// entry: brcond c, a, m; a: br m; m: brcond c, x, y. `m` is reached
    /// both ways, so nothing is decided there.
    #[test]
    fn a_merge_decides_nothing() {
        let mut f = function();
        let c = value(&mut f, HirType::Bool, HirValueKind::Parameter(0));
        let ret = HirTerminator::Return { values: Vec::new() };
        let x = block(&mut f, ret.clone());
        let y = block(&mut f, ret.clone());
        let m = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: c,
                true_target: x,
                false_target: y,
            },
        );
        let a = block(&mut f, HirTerminator::Branch { target: m });
        let entry = block(
            &mut f,
            HirTerminator::CondBranch {
                condition: c,
                true_target: a,
                false_target: m,
            },
        );
        f.entry_block = entry;
        assert_eq!(run(&mut f).folded, 0);
    }
}
