//! Compares against zero decided by the sign of what is compared.
//!
//! A floor division or remainder by a positive constant is a truncating
//! `div`/`rem` and a correction taken when the remainder is negative.
//! The correction's result lies in `[0, d)`, so a value that is the
//! result of one, or a non-negative constant, or a phi of such values
//! round a loop, is never negative, and the next correction on it
//! decides nothing: `x % 7` on an `x` that is itself `y % M` needs no
//! correction at all.
//!
//! The pass finds the values a function can show are non-negative, then
//! folds each `lt v, 0` on one to false and each `ge v, 0` to true, and
//! each `select` on a constant condition to the arm it takes. What is
//! known is decided together, since a loop's phi is non-negative only
//! if what comes round the back edge is.

use crate::hir::{
    BinaryOp, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirValueKind,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SignFoldStats {
    /// Compares folded to a constant.
    pub compares: usize,
    /// Selects replaced by the arm they take.
    pub selects: usize,
}

pub fn run_module(module: &mut HirModule) -> SignFoldStats {
    let mut total = SignFoldStats::default();
    for func in module.functions_to_optimize() {
        if func.is_external {
            continue;
        }
        let s = run_function(func);
        total.compares += s.compares;
        total.selects += s.selects;
    }
    total
}

fn run_function(func: &mut HirFunction) -> SignFoldStats {
    let mut stats = SignFoldStats::default();
    let nonneg = non_negative(func);
    if nonneg.is_empty() {
        return stats;
    }
    let zero_of = |id: HirId| -> bool {
        matches!(
            func.values.get(&id).map(|v| &v.kind),
            Some(HirValueKind::Constant(c)) if const_i128(c) == Some(0)
        )
    };
    // Compares decided by the sign become constants; their results are
    // read wherever they were read.
    let mut decided: Vec<(HirId, bool)> = Vec::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Binary {
                op: op @ (BinaryOp::Lt | BinaryOp::Ge),
                result,
                left,
                right,
                ..
            } = inst
            {
                if nonneg.contains(left) && zero_of(*right) {
                    decided.push((*result, matches!(op, BinaryOp::Ge)));
                }
            }
        }
    }
    for (result, truth) in &decided {
        if let Some(v) = func.values.get_mut(result) {
            v.kind = HirValueKind::Constant(HirConstant::Bool(*truth));
            v.ty = crate::hir::HirType::Bool;
        }
    }
    let decided_ids: HashSet<HirId> = decided.iter().map(|(id, _)| *id).collect();
    stats.compares = decided.len();

    // A select on a constant condition is the arm it takes.
    let mut picks: HashMap<HirId, HirId> = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let HirInstruction::Select {
                result,
                condition,
                true_val,
                false_val,
                ..
            } = inst
            {
                if let Some(HirValueKind::Constant(HirConstant::Bool(b))) =
                    func.values.get(condition).map(|v| &v.kind)
                {
                    picks.insert(*result, if *b { *true_val } else { *false_val });
                }
            }
        }
    }
    stats.selects = picks.len();
    for block in func.blocks.values_mut() {
        block.instructions.retain(|inst| match inst {
            HirInstruction::Binary { result, .. } => !decided_ids.contains(result),
            HirInstruction::Select { result, .. } => !picks.contains_key(result),
            _ => true,
        });
    }
    if !picks.is_empty() {
        crate::cse::apply_substitutions_public(func, &picks);
    }
    stats
}

/// The integer values of `func` that are never negative: non-negative
/// constants, a floor correction's result, a remainder, quotient, sum,
/// product or `and` of non-negatives, a right shift of one, a select
/// between two, and a phi whose every incoming is one. The greatest
/// such set.
fn non_negative(func: &HirFunction) -> HashSet<HirId> {
    // Every candidate to begin with; one whose rule fails drops out,
    // until none does. Only phis and the instructions below can stay.
    let mut set: HashSet<HirId> = HashSet::new();
    let mut defs: HashMap<HirId, &HirInstruction> = HashMap::new();
    for (id, value) in &func.values {
        match &value.kind {
            HirValueKind::Constant(c) => {
                if const_i128(c).is_some_and(|n| n >= 0) {
                    set.insert(*id);
                }
            }
            HirValueKind::Instruction => {
                set.insert(*id);
            }
            _ => {}
        }
    }
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if let Some(result) = inst.result_id() {
                defs.insert(result, inst);
            }
        }
    }
    let phis: HashMap<HirId, Vec<HirId>> = func
        .blocks
        .values()
        .flat_map(|b| b.phis.iter())
        .map(|p| (p.result, p.incoming.iter().map(|(v, _)| *v).collect()))
        .collect();
    loop {
        let before = set.len();
        let holds = |id: &HirId, set: &HashSet<HirId>| -> bool {
            if let Some(incoming) = phis.get(id) {
                return incoming.iter().all(|v| set.contains(v));
            }
            let Some(inst) = defs.get(id) else {
                // A constant kept above, or a parameter and the like,
                // which nothing here can show anything about.
                return matches!(
                    func.values.get(id).map(|v| &v.kind),
                    Some(HirValueKind::Constant(_))
                );
            };
            match inst {
                // A sum or product of non-negatives is one, short of the
                // wrap the integer type would take, which the program
                // was not written for either.
                HirInstruction::Binary {
                    op:
                        BinaryOp::Rem
                        | BinaryOp::And
                        | BinaryOp::Shr
                        | BinaryOp::Div
                        | BinaryOp::Add
                        | BinaryOp::Mul,
                    left,
                    right,
                    ..
                } => set.contains(left) && set.contains(right),
                HirInstruction::Select {
                    condition,
                    true_val,
                    false_val,
                    ..
                } => {
                    (set.contains(true_val) && set.contains(false_val))
                        || is_floor_correction(func, &defs, *condition, *true_val, *false_val)
                }
                _ => false,
            }
        };
        let dropped: Vec<HirId> = set.iter().copied().filter(|id| !holds(id, &set)).collect();
        for id in dropped {
            set.remove(&id);
        }
        if set.len() == before {
            return set;
        }
    }
}

/// Whether `select(condition, true_val, false_val)` is the correction
/// of a truncating remainder by a positive constant toward the floor:
/// `condition = lt r, 0`, `true_val = add r, d`, `false_val = r`, with
/// `r = rem x, d` and `d > 0`. Its result lies in `[0, d)`.
fn is_floor_correction(
    func: &HirFunction,
    defs: &HashMap<HirId, &HirInstruction>,
    condition: HirId,
    true_val: HirId,
    false_val: HirId,
) -> bool {
    let positive = |id: HirId| -> bool {
        matches!(
            func.values.get(&id).map(|v| &v.kind),
            Some(HirValueKind::Constant(c)) if const_i128(c).is_some_and(|n| n > 0)
        )
    };
    let zero = |id: HirId| -> bool {
        matches!(
            func.values.get(&id).map(|v| &v.kind),
            Some(HirValueKind::Constant(c)) if const_i128(c) == Some(0)
        )
    };
    let Some(HirInstruction::Binary {
        op: BinaryOp::Rem,
        left: _,
        right: d,
        ..
    }) = defs.get(&false_val)
    else {
        return false;
    };
    if !positive(*d) {
        return false;
    }
    let Some(HirInstruction::Binary {
        op: BinaryOp::Lt,
        left: cl,
        right: cr,
        ..
    }) = defs.get(&condition)
    else {
        return false;
    };
    if *cl != false_val || !zero(*cr) {
        return false;
    }
    matches!(
        defs.get(&true_val),
        Some(HirInstruction::Binary {
            op: BinaryOp::Add,
            left,
            right,
            ..
        }) if (*left == false_val && *right == *d) || (*right == false_val && *left == *d)
    )
}

fn const_i128(c: &HirConstant) -> Option<i128> {
    Some(match c {
        HirConstant::I8(v) => *v as i128,
        HirConstant::I16(v) => *v as i128,
        HirConstant::I32(v) => *v as i128,
        HirConstant::I64(v) => *v as i128,
        HirConstant::I128(v) => *v,
        HirConstant::ISize(v) => *v as i128,
        HirConstant::U8(v) => *v as i128,
        HirConstant::U16(v) => *v as i128,
        HirConstant::U32(v) => *v as i128,
        HirConstant::U64(v) => *v as i128,
        HirConstant::U128(v) => *v as i128,
        HirConstant::USize(v) => *v as i128,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirTerminator, HirType, HirValue};
    use indexmap::IndexMap;
    use zyntax_typed_ast::InternedString;

    fn value(values: &mut IndexMap<HirId, HirValue>, ty: HirType, kind: HirValueKind) -> HirId {
        let id = HirId::new();
        values.insert(
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

    fn bin(op: BinaryOp, result: HirId, ty: HirType, left: HirId, right: HirId) -> HirInstruction {
        HirInstruction::Binary {
            op,
            result,
            ty,
            left,
            right,
        }
    }

    /// `acc = phi [1, entry], [acc', body]; k = acc % 7` with the floor
    /// correction on `k`, and `acc' = (acc - 3) % M` corrected: the
    /// correction on `k` folds away, the one on `acc'` stays.
    #[test]
    fn a_remainder_of_a_corrected_remainder_needs_no_correction() {
        let i64t = HirType::I64;
        let mut values = IndexMap::new();
        let c = |values: &mut IndexMap<HirId, HirValue>, n: i64| {
            value(
                values,
                HirType::I64,
                HirValueKind::Constant(HirConstant::I64(n)),
            )
        };
        let one = c(&mut values, 1);
        let zero_a = c(&mut values, 0);
        let zero_b = c(&mut values, 0);
        let seven = c(&mut values, 7);
        let three = c(&mut values, 3);
        let m = c(&mut values, 1_000_003);
        let n = value(&mut values, i64t.clone(), HirValueKind::Parameter(0));
        let mut inst = |values: &mut IndexMap<HirId, HirValue>, ty: HirType| {
            value(values, ty, HirValueKind::Instruction)
        };
        let acc = inst(&mut values, i64t.clone());
        let k_rem = inst(&mut values, i64t.clone());
        let k_neg = inst(&mut values, HirType::Bool);
        let k_fix = inst(&mut values, i64t.clone());
        let k = inst(&mut values, i64t.clone());
        let bumped = inst(&mut values, i64t.clone());
        let a_rem = inst(&mut values, i64t.clone());
        let a_neg = inst(&mut values, HirType::Bool);
        let a_fix = inst(&mut values, i64t.clone());
        let acc_next = inst(&mut values, i64t.clone());
        let done = inst(&mut values, HirType::Bool);

        let entry = HirId::new();
        let header = HirId::new();
        let exit = HirId::new();
        let mut eb = HirBlock::new(entry);
        eb.terminator = HirTerminator::Branch { target: header };
        let mut hb = HirBlock::new(header);
        hb.phis.push(crate::hir::HirPhi {
            result: acc,
            ty: i64t.clone(),
            incoming: vec![(one, entry), (acc_next, header)],
        });
        hb.instructions = vec![
            bin(BinaryOp::Rem, k_rem, i64t.clone(), acc, seven),
            bin(BinaryOp::Lt, k_neg, i64t.clone(), k_rem, zero_a),
            bin(BinaryOp::Add, k_fix, i64t.clone(), k_rem, seven),
            HirInstruction::Select {
                result: k,
                ty: i64t.clone(),
                condition: k_neg,
                true_val: k_fix,
                false_val: k_rem,
            },
            bin(BinaryOp::Sub, bumped, i64t.clone(), acc, three),
            bin(BinaryOp::Rem, a_rem, i64t.clone(), bumped, m),
            bin(BinaryOp::Lt, a_neg, i64t.clone(), a_rem, zero_b),
            bin(BinaryOp::Add, a_fix, i64t.clone(), a_rem, m),
            HirInstruction::Select {
                result: acc_next,
                ty: i64t.clone(),
                condition: a_neg,
                true_val: a_fix,
                false_val: a_rem,
            },
            bin(BinaryOp::Lt, done, i64t.clone(), k, n),
        ];
        hb.terminator = HirTerminator::CondBranch {
            condition: done,
            true_target: exit,
            false_target: header,
        };
        let mut xb = HirBlock::new(exit);
        xb.terminator = HirTerminator::Return { values: vec![acc] };

        let mut func = HirFunction::new(
            InternedString::new_global("f"),
            HirFunctionSignature {
                params: vec![],
                returns: vec![i64t],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
                is_fiber: false,
                effects: vec![],
                is_pure: false,
            },
        );
        func.values = values;
        func.blocks.clear();
        func.blocks.insert(entry, eb);
        func.blocks.insert(header, hb);
        func.blocks.insert(exit, xb);
        func.entry_block = entry;

        let stats = run_function(&mut func);
        assert_eq!(stats.compares, 1, "only k's compare is decided");
        assert_eq!(stats.selects, 1, "only k's select is taken");
        let insts = &func.blocks[&header].instructions;
        assert!(
            !insts.iter().any(|i| i.result_id() == Some(k)),
            "k's select is gone"
        );
        assert!(
            insts.iter().any(|i| i.result_id() == Some(acc_next)),
            "acc's correction stays: acc - 3 may be negative"
        );
        // The compare that ended the loop now reads the remainder itself.
        assert!(insts.iter().any(|i| matches!(
            i,
            HirInstruction::Binary { op: BinaryOp::Lt, left, .. } if *left == k_rem
        )));
    }
}
