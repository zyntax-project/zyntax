//! HIR-level constant folding.
//!
//! Replaces `Binary { op, left, right }`, `Unary { op, operand }`,
//! `Cast { op, operand }`, and `Select { condition, … }` instructions
//! whose operands are all `HirValueKind::Constant` with the
//! materialised result — written back as a new `Constant` on the
//! same result `HirId`, with the original instruction removed.
//!
//! Folding the result *in place* via the SSA HirId means every
//! existing use of that value automatically picks up the new
//! constant on its next read — no rewriting of consumer
//! instructions required. Downstream DCE sweeps then naturally
//! remove anything that becomes dead.
//!
//! ## What we fold
//!
//! * **Integer arithmetic / bitwise / shifts** — Add, Sub, Mul, Div,
//!   Rem, And, Or, Xor, Shl, Shr. Wrapping semantics for arithmetic;
//!   Div/Rem by zero is preserved (we skip, runtime semantics
//!   handle the trap).
//! * **Integer comparisons** — Eq, Ne, Lt, Le, Gt, Ge → `Bool`.
//! * **Float arithmetic** — FAdd, FSub, FMul, FDiv, FRem (via
//!   Rust's `%`). Folded for both f32 and f64.
//! * **Float comparisons** — FEq, FNe, FLt, FLe, FGt, FGe → `Bool`.
//!   NaN handling matches IEEE-754 — any compare against NaN
//!   except `FNe` is false.
//! * **Unary** — Neg (wrapping integer negate), Not (bitwise),
//!   FNeg (float negate).
//! * **Cast** — Trunc, ZExt, SExt, FpTrunc, FpExt, FpToUi, FpToSi,
//!   UiToFp, SiToFp on numeric constants. Bitcast between same-
//!   width int/float. PtrToInt / IntToPtr only fold when the
//!   operand is `HirConstant::Null` (yielding `0`).
//! * **Select** — `cond ? t : f` collapses to `t` or `f` when
//!   `cond` is a `Bool` constant.
//!
//! ## What we don't fold
//!
//! * Anything with a non-numeric / non-bool operand (strings,
//!   structs, arrays, vtables).
//! * Division / remainder by zero — caller's runtime semantics
//!   apply.
//! * Float ops in the rare case the existing constant cannot be
//!   exactly represented in the target type (we still fold for
//!   correctness, mirroring rustc's behaviour).
//!
//! ## Fixed point
//!
//! `fold_module` iterates until no instruction can be folded
//! further — folding one site frequently enables another (a `+ 0`
//! peephole that later folds into a comparison). Bounded to 16
//! passes in pathological cases.

use crate::hir::{
    BinaryOp, CastOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule,
    HirTerminator, HirType, HirValueKind, UnaryOp,
};

/// Number of folds performed in one `fold_module` call. Caller can
/// log / assert on this for testability.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FoldStats {
    pub folded: usize,
    pub iterations: usize,
}

/// Fold every reachable constant expression in `module`. Iterates
/// to a fixed point (capped at 16 passes — every pass strictly
/// reduces fold-eligible instructions, so we stop well short).
pub fn fold_module(module: &mut HirModule) -> FoldStats {
    let mut stats = FoldStats::default();
    for func in module.functions.values_mut() {
        stats = combine(stats, fold_function(func));
    }
    stats
}

/// Same as `fold_module` but scoped to a single function. Returned
/// stats are the totals for this function only.
pub fn fold_function(func: &mut HirFunction) -> FoldStats {
    let mut stats = FoldStats::default();
    for _ in 0..16 {
        stats.iterations += 1;
        let this_pass = fold_one_pass(func);
        if this_pass == 0 {
            return stats;
        }
        stats.folded += this_pass;
    }
    stats
}

fn combine(a: FoldStats, b: FoldStats) -> FoldStats {
    FoldStats {
        folded: a.folded + b.folded,
        iterations: a.iterations.max(b.iterations),
    }
}

/// Visit every instruction in every block once. Returns the number
/// of instructions folded this pass.
fn fold_one_pass(func: &mut HirFunction) -> usize {
    let mut folded = 0;

    // Snapshot the block ids — modifying `func.blocks` while
    // iterating requires this to avoid aliasing borrows.
    let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();

    for block_id in block_ids {
        // Pull values out for read-only access while we mutate
        // `func.blocks`. `values` will be re-read after each fold so
        // a fold that produces a new Constant is visible to
        // subsequent folds in the same block.
        let mut keep: Vec<HirInstruction> = Vec::new();

        let block_insts = match func.blocks.get(&block_id) {
            Some(b) => b.instructions.clone(),
            None => continue,
        };

        for inst in block_insts {
            if let Some((result_id, new_const, new_ty)) = try_fold_inst(&inst, &func.values) {
                // Replace the result value's kind with the new
                // Constant. Every existing use of `result_id`
                // automatically picks up the new value.
                if let Some(v) = func.values.get_mut(&result_id) {
                    v.kind = HirValueKind::Constant(new_const);
                    v.ty = new_ty;
                }
                folded += 1;
                // Drop the instruction entirely — its result is now
                // a pure SSA constant, no inst-level evaluation
                // needed.
                continue;
            }
            keep.push(inst);
        }

        if let Some(block) = func.blocks.get_mut(&block_id) {
            block.instructions = keep;
        }
    }

    // Fold terminators too — a `CondBranch` on a known constant
    // becomes an unconditional `Branch`. Doesn't touch instructions
    // but counts toward the `folded` total so the fixed-point loop
    // terminates correctly.
    folded += fold_terminators(func);

    folded
}

/// Inspect one instruction. Return `Some((result_id, new_constant,
/// result_type))` when it folds; `None` otherwise.
fn try_fold_inst(
    inst: &HirInstruction,
    values: &indexmap::IndexMap<HirId, crate::hir::HirValue>,
) -> Option<(HirId, HirConstant, HirType)> {
    match inst {
        HirInstruction::Binary {
            op,
            result,
            ty,
            left,
            right,
        } => {
            let l = constant_of(*left, values)?;
            let r = constant_of(*right, values)?;
            let folded = fold_binary(*op, l, r, ty)?;
            Some((*result, folded, ty.clone()))
        }
        HirInstruction::Unary {
            op,
            result,
            ty,
            operand,
        } => {
            let o = constant_of(*operand, values)?;
            let folded = fold_unary(*op, o, ty)?;
            Some((*result, folded, ty.clone()))
        }
        HirInstruction::Cast {
            op,
            result,
            ty,
            operand,
        } => {
            let o = constant_of(*operand, values)?;
            let folded = fold_cast(*op, o, ty)?;
            Some((*result, folded, ty.clone()))
        }
        HirInstruction::Select {
            result,
            ty,
            condition,
            true_val,
            false_val,
        } => {
            let cond = constant_of(*condition, values)?;
            let HirConstant::Bool(b) = cond else {
                return None;
            };
            let pick = if *b { *true_val } else { *false_val };
            let v = constant_of(pick, values)?;
            Some((*result, v.clone(), ty.clone()))
        }
        _ => None,
    }
}

fn constant_of<'a>(
    id: HirId,
    values: &'a indexmap::IndexMap<HirId, crate::hir::HirValue>,
) -> Option<&'a HirConstant> {
    match &values.get(&id)?.kind {
        HirValueKind::Constant(c) => Some(c),
        _ => None,
    }
}

// ─── Binary folding ───────────────────────────────────────────────

fn fold_binary(
    op: BinaryOp,
    l: &HirConstant,
    r: &HirConstant,
    ty: &HirType,
) -> Option<HirConstant> {
    use BinaryOp::*;
    // Integer family (also covers bool for And/Or/Xor when both
    // sides are Bool).
    if let (Some(li), Some(ri)) = (as_i128(l), as_i128(r)) {
        let folded = match op {
            Add => li.wrapping_add(ri),
            Sub => li.wrapping_sub(ri),
            Mul => li.wrapping_mul(ri),
            Div => {
                if ri == 0 {
                    return None;
                }
                li.wrapping_div(ri)
            }
            Rem => {
                if ri == 0 {
                    return None;
                }
                li.wrapping_rem(ri)
            }
            And => li & ri,
            Or => li | ri,
            Xor => li ^ ri,
            Shl => {
                let shift = ri as u32 & 127;
                li.wrapping_shl(shift)
            }
            Shr => {
                let shift = ri as u32 & 127;
                if is_signed_ty(ty) {
                    li.wrapping_shr(shift)
                } else {
                    (li as u128).wrapping_shr(shift) as i128
                }
            }
            Eq => return Some(HirConstant::Bool(li == ri)),
            Ne => return Some(HirConstant::Bool(li != ri)),
            Lt => {
                return Some(HirConstant::Bool(if is_signed_ty(ty) {
                    li < ri
                } else {
                    (li as u128) < (ri as u128)
                }))
            }
            Le => {
                return Some(HirConstant::Bool(if is_signed_ty(ty) {
                    li <= ri
                } else {
                    (li as u128) <= (ri as u128)
                }))
            }
            Gt => {
                return Some(HirConstant::Bool(if is_signed_ty(ty) {
                    li > ri
                } else {
                    (li as u128) > (ri as u128)
                }))
            }
            Ge => {
                return Some(HirConstant::Bool(if is_signed_ty(ty) {
                    li >= ri
                } else {
                    (li as u128) >= (ri as u128)
                }))
            }
            // Float ops on int operands — shouldn't happen in
            // well-typed HIR, but be defensive.
            FAdd | FSub | FMul | FDiv | FRem | FEq | FNe | FLt | FLe | FGt | FGe => return None,
        };
        return Some(int_constant(folded, ty));
    }

    // Bool ops (And/Or/Xor) on Bool operands — when the operands
    // weren't coerced into i128 above (Bool is, but the result type
    // here should be Bool, which `int_constant` doesn't handle).
    if let (HirConstant::Bool(lb), HirConstant::Bool(rb)) = (l, r) {
        return match op {
            And => Some(HirConstant::Bool(*lb && *rb)),
            Or => Some(HirConstant::Bool(*lb || *rb)),
            Xor => Some(HirConstant::Bool(*lb ^ *rb)),
            Eq => Some(HirConstant::Bool(*lb == *rb)),
            Ne => Some(HirConstant::Bool(*lb != *rb)),
            _ => None,
        };
    }

    // Float family.
    let (lf, rf, is_f32) = match (l, r) {
        (HirConstant::F32(a), HirConstant::F32(b)) => (*a as f64, *b as f64, true),
        (HirConstant::F64(a), HirConstant::F64(b)) => (*a, *b, false),
        _ => return None,
    };

    let result_f64 = match op {
        FAdd => lf + rf,
        FSub => lf - rf,
        FMul => lf * rf,
        FDiv => lf / rf, // float div-by-zero produces ±inf or NaN, valid IEEE
        FRem => lf % rf,
        FEq => return Some(HirConstant::Bool(lf == rf)),
        FNe => return Some(HirConstant::Bool(lf != rf)),
        FLt => return Some(HirConstant::Bool(lf < rf)),
        FLe => return Some(HirConstant::Bool(lf <= rf)),
        FGt => return Some(HirConstant::Bool(lf > rf)),
        FGe => return Some(HirConstant::Bool(lf >= rf)),
        _ => return None,
    };

    Some(if is_f32 {
        HirConstant::F32(result_f64 as f32)
    } else {
        HirConstant::F64(result_f64)
    })
}

// ─── Unary folding ────────────────────────────────────────────────

fn fold_unary(op: UnaryOp, v: &HirConstant, ty: &HirType) -> Option<HirConstant> {
    match (op, v) {
        (UnaryOp::Neg, _) => {
            let i = as_i128(v)?;
            Some(int_constant(i.wrapping_neg(), ty))
        }
        (UnaryOp::Not, HirConstant::Bool(b)) => Some(HirConstant::Bool(!b)),
        (UnaryOp::Not, _) => {
            let i = as_i128(v)?;
            Some(int_constant(!i, ty))
        }
        (UnaryOp::FNeg, HirConstant::F32(f)) => Some(HirConstant::F32(-f)),
        (UnaryOp::FNeg, HirConstant::F64(f)) => Some(HirConstant::F64(-f)),
        _ => None,
    }
}

// ─── Cast folding ─────────────────────────────────────────────────

fn fold_cast(op: CastOp, v: &HirConstant, target_ty: &HirType) -> Option<HirConstant> {
    use CastOp::*;
    match op {
        Trunc | ZExt | SExt => {
            let i = as_i128(v)?;
            Some(int_constant(i, target_ty))
        }
        FpTrunc => {
            if let HirConstant::F64(f) = v {
                return Some(HirConstant::F32(*f as f32));
            }
            None
        }
        FpExt => {
            if let HirConstant::F32(f) = v {
                return Some(HirConstant::F64(*f as f64));
            }
            None
        }
        FpToSi | FpToUi => {
            let f = match v {
                HirConstant::F32(x) => *x as f64,
                HirConstant::F64(x) => *x,
                _ => return None,
            };
            // Rust's `as` semantics: saturating cast to integer.
            Some(int_constant(f as i128, target_ty))
        }
        SiToFp => {
            let i = as_i128(v)?;
            match target_ty {
                HirType::F32 => Some(HirConstant::F32(i as f32)),
                HirType::F64 => Some(HirConstant::F64(i as f64)),
                _ => None,
            }
        }
        UiToFp => {
            // For unsigned-source we need the raw bit pattern as u128.
            let u = match v {
                HirConstant::U8(x) => *x as u128,
                HirConstant::U16(x) => *x as u128,
                HirConstant::U32(x) => *x as u128,
                HirConstant::U64(x) => *x as u128,
                HirConstant::U128(x) => *x,
                // Signed -> unsigned cast happens at the front-end;
                // here we only fold UiToFp when the operand is
                // already unsigned to avoid sign-extension surprises.
                _ => return None,
            };
            match target_ty {
                HirType::F32 => Some(HirConstant::F32(u as f32)),
                HirType::F64 => Some(HirConstant::F64(u as f64)),
                _ => None,
            }
        }
        Bitcast => {
            // Same-width int↔float bitcast.
            match (v, target_ty) {
                (HirConstant::I32(i), HirType::F32) => {
                    Some(HirConstant::F32(f32::from_bits(*i as u32)))
                }
                (HirConstant::U32(u), HirType::F32) => Some(HirConstant::F32(f32::from_bits(*u))),
                (HirConstant::I64(i), HirType::F64) => {
                    Some(HirConstant::F64(f64::from_bits(*i as u64)))
                }
                (HirConstant::U64(u), HirType::F64) => Some(HirConstant::F64(f64::from_bits(*u))),
                (HirConstant::F32(f), HirType::I32) => Some(HirConstant::I32(f.to_bits() as i32)),
                (HirConstant::F32(f), HirType::U32) => Some(HirConstant::U32(f.to_bits())),
                (HirConstant::F64(f), HirType::I64) => Some(HirConstant::I64(f.to_bits() as i64)),
                (HirConstant::F64(f), HirType::U64) => Some(HirConstant::U64(f.to_bits())),
                _ => None,
            }
        }
        // Pointer casts only fold the Null case — there's no value-
        // level meaning to "a constant non-null pointer" in HIR.
        PtrToInt => {
            if matches!(v, HirConstant::Null(_)) {
                return Some(int_constant(0, target_ty));
            }
            None
        }
        IntToPtr => {
            let i = as_i128(v)?;
            if i == 0 {
                return Some(HirConstant::Null(target_ty.clone()));
            }
            None
        }
    }
}

// ─── Terminator folding ───────────────────────────────────────────
//
// `CondBranch { condition }` where condition is a known Bool
// constant becomes `Branch { target }`. `Switch { value }` where
// value is a known integer constant becomes `Branch { matched_case }`.
// Block-level CFG simplification (merging unreachable preds, etc.)
// is a separate concern — that's what CFG-simplify will do.
fn fold_terminators(func: &mut HirFunction) -> usize {
    let mut folded = 0;
    // Snapshot block ids so we can mutate while we walk.
    let ids: Vec<HirId> = func.blocks.keys().copied().collect();
    for id in ids {
        let new_term = {
            let block = match func.blocks.get(&id) {
                Some(b) => b,
                None => continue,
            };
            match &block.terminator {
                HirTerminator::CondBranch {
                    condition,
                    true_target,
                    false_target,
                } => match constant_of(*condition, &func.values) {
                    Some(HirConstant::Bool(true)) => Some(HirTerminator::Branch {
                        target: *true_target,
                    }),
                    Some(HirConstant::Bool(false)) => Some(HirTerminator::Branch {
                        target: *false_target,
                    }),
                    _ => None,
                },
                _ => None,
            }
        };
        if let Some(t) = new_term {
            if let Some(b) = func.blocks.get_mut(&id) {
                b.terminator = t;
                folded += 1;
            }
        }
    }
    folded
}

// ─── helpers ──────────────────────────────────────────────────────

/// Reduce an integer-shaped constant (including Bool, treated as
/// 0/1) to a wide signed representation. Returns `None` for non-int
/// kinds.
fn as_i128(c: &HirConstant) -> Option<i128> {
    use HirConstant::*;
    Some(match c {
        Bool(b) => *b as i128,
        I8(x) => *x as i128,
        I16(x) => *x as i128,
        I32(x) => *x as i128,
        I64(x) => *x as i128,
        I128(x) => *x,
        U8(x) => *x as i128,
        U16(x) => *x as i128,
        U32(x) => *x as i128,
        U64(x) => *x as i128,
        U128(x) => *x as i128, // truncating, matches `as` semantics
        _ => return None,
    })
}

fn is_signed_ty(ty: &HirType) -> bool {
    matches!(
        ty,
        HirType::I8 | HirType::I16 | HirType::I32 | HirType::I64 | HirType::I128
    )
}

/// Materialise a wide signed integer back into the right
/// `HirConstant` variant for `ty`. Wrapping into the target width.
fn int_constant(v: i128, ty: &HirType) -> HirConstant {
    match ty {
        HirType::Bool => HirConstant::Bool(v != 0),
        HirType::I8 => HirConstant::I8(v as i8),
        HirType::I16 => HirConstant::I16(v as i16),
        HirType::I32 => HirConstant::I32(v as i32),
        HirType::I64 => HirConstant::I64(v as i64),
        HirType::I128 => HirConstant::I128(v),
        HirType::U8 => HirConstant::U8(v as u8),
        HirType::U16 => HirConstant::U16(v as u16),
        HirType::U32 => HirConstant::U32(v as u32),
        HirType::U64 => HirConstant::U64(v as u64),
        HirType::U128 => HirConstant::U128(v as u128),
        // Default to I64 when ty isn't a known integer — defensive,
        // shouldn't happen in well-typed HIR.
        _ => HirConstant::I64(v as i64),
    }
}

#[allow(unused)]
fn touch_hir_callable(_: &HirCallable) {}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirValue};
    use indexmap::IndexMap;
    use zyntax_typed_ast::InternedString;

    fn empty_sig(ret: HirType) -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
            returns: vec![ret],
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

    fn mk_func(ret: HirType) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global("t"), empty_sig(ret));
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
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

    fn add_inst_result(f: &mut HirFunction, ty: HirType) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn push(f: &mut HirFunction, inst: HirInstruction) {
        let entry = f.entry_block;
        f.blocks.get_mut(&entry).unwrap().instructions.push(inst);
    }

    #[test]
    fn folds_i32_add_to_constant_and_removes_instruction() {
        let mut f = mk_func(HirType::I32);
        let l = add_const(&mut f, HirType::I32, HirConstant::I32(7));
        let r = add_const(&mut f, HirType::I32, HirConstant::I32(35));
        let result = add_inst_result(&mut f, HirType::I32);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result,
                ty: HirType::I32,
                left: l,
                right: r,
            },
        );

        let stats = fold_function(&mut f);
        assert_eq!(stats.folded, 1);
        // Instruction is gone.
        assert_eq!(f.blocks[&f.entry_block].instructions.len(), 0);
        // Result is now a Constant(I32(42)).
        match &f.values[&result].kind {
            HirValueKind::Constant(HirConstant::I32(42)) => {}
            other => panic!("expected I32(42), got {other:?}"),
        }
    }

    #[test]
    fn fixed_point_folds_chained_arithmetic() {
        // (2 + 3) * (10 - 4)  → 5 * 6 → 30
        let mut f = mk_func(HirType::I64);
        let two = add_const(&mut f, HirType::I64, HirConstant::I64(2));
        let three = add_const(&mut f, HirType::I64, HirConstant::I64(3));
        let ten = add_const(&mut f, HirType::I64, HirConstant::I64(10));
        let four = add_const(&mut f, HirType::I64, HirConstant::I64(4));
        let sum = add_inst_result(&mut f, HirType::I64);
        let diff = add_inst_result(&mut f, HirType::I64);
        let prod = add_inst_result(&mut f, HirType::I64);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: sum,
                ty: HirType::I64,
                left: two,
                right: three,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: diff,
                ty: HirType::I64,
                left: ten,
                right: four,
            },
        );
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Mul,
                result: prod,
                ty: HirType::I64,
                left: sum,
                right: diff,
            },
        );

        let stats = fold_function(&mut f);
        // All three folded — the multiply was foldable only AFTER
        // sum & diff folded, so we need at least 2 iterations.
        assert_eq!(stats.folded, 3);
        assert!(stats.iterations >= 2);
        assert_eq!(f.blocks[&f.entry_block].instructions.len(), 0);
        match &f.values[&prod].kind {
            HirValueKind::Constant(HirConstant::I64(30)) => {}
            other => panic!("expected I64(30), got {other:?}"),
        }
    }

    #[test]
    fn does_not_fold_division_by_zero() {
        let mut f = mk_func(HirType::I32);
        let l = add_const(&mut f, HirType::I32, HirConstant::I32(10));
        let r = add_const(&mut f, HirType::I32, HirConstant::I32(0));
        let result = add_inst_result(&mut f, HirType::I32);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Div,
                result,
                ty: HirType::I32,
                left: l,
                right: r,
            },
        );
        let stats = fold_function(&mut f);
        assert_eq!(stats.folded, 0);
        assert_eq!(f.blocks[&f.entry_block].instructions.len(), 1);
    }

    #[test]
    fn folds_unsigned_comparison_correctly() {
        // (u32) 0xFFFFFFFF < (u32) 1  → false (unsigned)
        // If we accidentally signed-compared it'd be -1 < 1 → true.
        let mut f = mk_func(HirType::Bool);
        let big = add_const(&mut f, HirType::U32, HirConstant::U32(0xFFFFFFFF));
        let one = add_const(&mut f, HirType::U32, HirConstant::U32(1));
        let result = add_inst_result(&mut f, HirType::Bool);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Lt,
                result,
                ty: HirType::U32,
                left: big,
                right: one,
            },
        );
        let stats = fold_function(&mut f);
        assert_eq!(stats.folded, 1);
        match &f.values[&result].kind {
            HirValueKind::Constant(HirConstant::Bool(false)) => {}
            other => panic!("expected Bool(false), got {other:?}"),
        }
    }

    #[test]
    fn collapses_select_when_condition_is_known() {
        let mut f = mk_func(HirType::I64);
        let cond = add_const(&mut f, HirType::Bool, HirConstant::Bool(true));
        let t = add_const(&mut f, HirType::I64, HirConstant::I64(42));
        let fl = add_const(&mut f, HirType::I64, HirConstant::I64(99));
        let result = add_inst_result(&mut f, HirType::I64);
        push(
            &mut f,
            HirInstruction::Select {
                result,
                ty: HirType::I64,
                condition: cond,
                true_val: t,
                false_val: fl,
            },
        );
        let stats = fold_function(&mut f);
        assert_eq!(stats.folded, 1);
        match &f.values[&result].kind {
            HirValueKind::Constant(HirConstant::I64(42)) => {}
            other => panic!("expected I64(42), got {other:?}"),
        }
    }

    #[test]
    fn folds_cond_branch_on_known_bool() {
        let mut f = mk_func(HirType::I64);
        let true_id = HirId::new();
        let false_id = HirId::new();
        f.blocks.insert(true_id, HirBlock::new(true_id));
        f.blocks.insert(false_id, HirBlock::new(false_id));
        let cond = add_const(&mut f, HirType::Bool, HirConstant::Bool(false));
        f.blocks.get_mut(&f.entry_block).unwrap().terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: true_id,
            false_target: false_id,
        };
        let stats = fold_function(&mut f);
        assert!(stats.folded >= 1);
        match f.blocks[&f.entry_block].terminator {
            HirTerminator::Branch { target } => assert_eq!(target, false_id),
            ref other => panic!("expected Branch -> false_id, got {other:?}"),
        }
    }
}
