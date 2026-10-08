//! Strength-reduce bounded signed division and remainder by positive constants.
//! Bounds start unknown and only tighten; arithmetic must prove it cannot overflow.

use crate::hir::*;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Range {
    lo: i64,
    hi: i64,
}
impl Range {
    fn union(self, b: Self) -> Self {
        Self {
            lo: self.lo.min(b.lo),
            hi: self.hi.max(b.hi),
        }
    }
    fn fits_i32(self) -> bool {
        self.lo <= self.hi && self.lo >= i32::MIN as i64 && self.hi <= i32::MAX as i64
    }
}

fn constant(f: &HirFunction, id: HirId) -> Option<i64> {
    match f.values.get(&id)?.kind {
        HirValueKind::Constant(HirConstant::I64(v)) => Some(v),
        _ => None,
    }
}
fn divisor(f: &HirFunction, id: HirId) -> Option<i64> {
    constant(f, id).filter(|d| *d > 1 && *d <= i32::MAX as i64 && !(*d as u64).is_power_of_two())
}

fn reciprocal(r: Range, d: i64) -> Option<(i64, i64)> {
    if !r.fits_i32() {
        return None;
    }
    let magnitude = r.lo.unsigned_abs().max(r.hi.unsigned_abs());
    if magnitude == 0 {
        return None;
    }
    // S > |x|*(d-1) bounds the reciprocal's rounding error below 1/d.
    // With m=ceil(S/d), floor(x*m/S) is x/d for nonnegative x and one
    // less than trunc(x/d) for negative x. Non-powers of two have m*d > S.
    let needed = magnitude.checked_mul(d as u64 - 1)?.checked_add(1)?;
    let shift = 64 - (needed - 1).leading_zeros();
    let scale = 1u64.checked_shl(shift)?;
    let m = i64::try_from(scale.div_ceil(d as u64)).ok()?;
    r.lo.checked_mul(m)?;
    r.hi.checked_mul(m)?;
    Some((m, shift as i64))
}

fn transfer(f: &HirFunction, i: &HirInstruction, facts: &HashMap<HirId, Range>) -> Option<Range> {
    let get = |id| facts.get(&id).copied();
    match i {
        HirInstruction::Binary {
            op: BinaryOp::Rem,
            ty: HirType::I64,
            left,
            right,
            ..
        } => {
            let d = constant(f, *right).filter(|d| *d > 0)?;
            let a = get(*left);
            Some(Range {
                lo: if a.is_some_and(|a| a.lo >= 0) {
                    0
                } else {
                    1 - d
                },
                hi: if a.is_some_and(|a| a.hi <= 0) {
                    0
                } else {
                    d - 1
                },
            })
        }
        HirInstruction::Binary {
            op,
            ty: HirType::I64,
            left,
            right,
            ..
        } => {
            let (a, b) = (get(*left)?, get(*right)?);
            match op {
                BinaryOp::Add => Some(Range {
                    lo: a.lo.checked_add(b.lo)?,
                    hi: a.hi.checked_add(b.hi)?,
                }),
                BinaryOp::Sub => Some(Range {
                    lo: a.lo.checked_sub(b.hi)?,
                    hi: a.hi.checked_sub(b.lo)?,
                }),
                BinaryOp::Mul => {
                    let products = [
                        a.lo.checked_mul(b.lo)?,
                        a.lo.checked_mul(b.hi)?,
                        a.hi.checked_mul(b.lo)?,
                        a.hi.checked_mul(b.hi)?,
                    ];
                    Some(Range {
                        lo: *products.iter().min()?,
                        hi: *products.iter().max()?,
                    })
                }
                _ => None,
            }
        }
        HirInstruction::Select {
            ty: HirType::I64,
            true_val,
            false_val,
            ..
        } => Some(get(*true_val)?.union(get(*false_val)?)),
        HirInstruction::Cast {
            op: CastOp::SExt,
            ty: HirType::I64,
            operand,
            ..
        } => match f.values.get(operand)?.ty {
            HirType::I8 => Some(Range {
                lo: i8::MIN as i64,
                hi: i8::MAX as i64,
            }),
            HirType::I16 => Some(Range {
                lo: i16::MIN as i64,
                hi: i16::MAX as i64,
            }),
            HirType::I32 => Some(Range {
                lo: i32::MIN as i64,
                hi: i32::MAX as i64,
            }),
            _ => None,
        },
        _ => None,
    }
}

pub fn run_function(f: &mut HirFunction) -> usize {
    if !f.blocks.values().flat_map(|b| &b.instructions).any(|i| {
        matches!(i, HirInstruction::Binary {
            op: BinaryOp::Div | BinaryOp::Rem, ty: HirType::I64, right, ..
        } if divisor(f, *right).is_some())
    }) {
        return 0;
    }
    let mut facts: HashMap<_, _> = f
        .values
        .keys()
        .filter_map(|id| constant(f, *id).map(|v| (*id, Range { lo: v, hi: v })))
        .collect();
    // Every intermediate bound is sound, so reaching the work cap loses only precision.
    for _ in 0..16 {
        let previous = facts.clone();
        let mut changed = false;
        let mut update = |id, new: Range| {
            let new = if let Some(old) = facts.get(&id) {
                Range {
                    lo: old.lo.max(new.lo),
                    hi: old.hi.min(new.hi),
                }
            } else {
                new
            };
            if new.lo <= new.hi && facts.get(&id) != Some(&new) {
                facts.insert(id, new);
                changed = true;
            }
        };
        // Apply one round together, including loop-carried phis.
        for b in f.blocks.values() {
            for p in &b.phis {
                if p.ty != HirType::I64 || p.incoming.is_empty() {
                    continue;
                }
                let range = p
                    .incoming
                    .iter()
                    .try_fold(None, |joined: Option<Range>, (v, _)| {
                        let r = *previous.get(v)?;
                        Some(Some(joined.map_or(r, |a| a.union(r))))
                    });
                if let Some(Some(r)) = range {
                    update(p.result, r);
                }
            }
            for i in &b.instructions {
                if let Some(r) = transfer(f, i, &previous) {
                    update(i.result_id().unwrap(), r);
                }
            }
        }
        if !changed {
            break;
        }
    }
    let candidates: HashMap<_, _> = f
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .filter_map(|i| match i {
            HirInstruction::Binary {
                op: op @ (BinaryOp::Div | BinaryOp::Rem),
                ty: HirType::I64,
                result,
                left,
                right,
            } => {
                let range = *facts.get(left)?;
                let (multiplier, shift) = reciprocal(range, divisor(f, *right)?)?;
                Some((
                    *result,
                    (*op, *left, *right, multiplier, shift, range.lo < 0),
                ))
            }
            _ => None,
        })
        .collect();
    if candidates.is_empty() {
        return 0;
    }
    let ids: Vec<_> = f.blocks.keys().copied().collect();
    for bid in ids {
        let mut out = Vec::new();
        for i in std::mem::take(&mut f.blocks.get_mut(&bid).unwrap().instructions) {
            if let Some((result, (op, left, right, multiplier, shift, signed))) = i
                .result_id()
                .and_then(|r| candidates.get(&r).map(|c| (r, *c)))
            {
                let multiplier = f.create_value(
                    HirType::I64,
                    HirValueKind::Constant(HirConstant::I64(multiplier)),
                );
                let shift = f.create_value(
                    HirType::I64,
                    HirValueKind::Constant(HirConstant::I64(shift)),
                );
                let product = f.create_value(HirType::I64, HirValueKind::Instruction);
                let quotient = if op == BinaryOp::Div {
                    result
                } else {
                    f.create_value(HirType::I64, HirValueKind::Instruction)
                };
                let floor = if signed {
                    f.create_value(HirType::I64, HirValueKind::Instruction)
                } else {
                    quotient
                };
                let mut emit = |result, op, left, right| {
                    out.push(HirInstruction::Binary {
                        result,
                        op,
                        ty: HirType::I64,
                        left,
                        right,
                    })
                };
                emit(product, BinaryOp::Mul, left, multiplier);
                emit(floor, BinaryOp::Shr, product, shift);
                if signed {
                    let shift =
                        f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(63)));
                    let sign = f.create_value(HirType::I64, HirValueKind::Instruction);
                    emit(sign, BinaryOp::Shr, left, shift);
                    emit(quotient, BinaryOp::Sub, floor, sign);
                }
                if op == BinaryOp::Rem {
                    let product = f.create_value(HirType::I64, HirValueKind::Instruction);
                    emit(product, BinaryOp::Mul, quotient, right);
                    emit(result, BinaryOp::Sub, left, product);
                }
            } else {
                out.push(i);
            }
        }
        f.blocks.get_mut(&bid).unwrap().instructions = out;
    }
    candidates.len()
}

pub fn run_module(m: &mut HirModule) -> usize {
    // ZYNTAX_DISABLE_BOUNDED_DIV retains native division; safe.
    if std::env::var_os("ZYNTAX_DISABLE_BOUNDED_DIV").is_some() {
        return 0;
    }
    m.functions_to_optimize().map(run_function).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reciprocal_matches_signed_division_at_boundaries_and_random_values() {
        let mut seed = 0x4d59_5df4_d0f3_3173u64;
        for magnitude in [
            1,
            2,
            7,
            100,
            1_000_003,
            i32::MAX as i64,
            1 + i32::MAX as i64,
        ] {
            let range = Range {
                lo: -magnitude,
                hi: magnitude.min(i32::MAX as i64),
            };
            for d in (3..128).chain([1023, 1025, 1_000_003, i32::MAX as i64]) {
                if (d as u64).is_power_of_two() {
                    continue;
                }
                let (m, shift) = reciprocal(range, d).unwrap();
                let mut samples = vec![range.lo, range.lo + 1, range.hi, range.hi - 1, -1, 0, 1];
                for _ in 0..64 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    samples.push(range.lo + (seed % (range.hi - range.lo + 1) as u64) as i64);
                }
                for x in samples {
                    assert_eq!(((x * m) >> shift) - (x >> 63), x / d, "x={x} d={d}");
                }
            }
        }
    }
}
