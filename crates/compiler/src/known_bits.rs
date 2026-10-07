//! Bounded integer bit facts for equality folding. Unknown definitions,
//! including loop phis and memory reads, contribute no facts.

use crate::hir::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Default, Debug)]
struct Bits {
    zero: u128,
    one: u128,
}

fn width(ty: &HirType) -> Option<u32> {
    Some(match ty {
        HirType::I8 | HirType::U8 => 8,
        HirType::I16 | HirType::U16 => 16,
        HirType::I32 | HirType::U32 => 32,
        HirType::I64 | HirType::U64 => 64,
        HirType::I128 | HirType::U128 => 128,
        _ => return None,
    })
}

fn mask(width: u32) -> u128 {
    u128::MAX >> (128 - width)
}

fn constant(c: &HirConstant) -> Option<u128> {
    Some(match c {
        HirConstant::I8(v) => *v as u128,
        HirConstant::I16(v) => *v as u128,
        HirConstant::I32(v) => *v as u128,
        HirConstant::I64(v) => *v as u128,
        HirConstant::I128(v) => *v as u128,
        HirConstant::U8(v) => *v as u128,
        HirConstant::U16(v) => *v as u128,
        HirConstant::U32(v) => *v as u128,
        HirConstant::U64(v) => *v as u128,
        HirConstant::U128(v) => *v,
        _ => return None,
    })
}

impl Bits {
    fn binary(self, op: BinaryOp, b: Self) -> Self {
        match op {
            BinaryOp::And => Self {
                zero: self.zero | b.zero,
                one: self.one & b.one,
            },
            BinaryOp::Or => Self {
                zero: self.zero & b.zero,
                one: self.one | b.one,
            },
            BinaryOp::Xor => Self {
                zero: (self.zero & b.zero) | (self.one & b.one),
                one: (self.zero & b.one) | (self.one & b.zero),
            },
            _ => Self::default(),
        }
    }

    fn equal(self, b: Self, width: u32) -> Option<bool> {
        if (self.one & b.zero) | (self.zero & b.one) != 0 {
            Some(false)
        } else if self.zero | self.one == mask(width) && b.zero | b.one == mask(width) {
            Some(self.one == b.one)
        } else {
            None
        }
    }
}

struct Facts<'a> {
    func: &'a HirFunction,
    defs: HashMap<HirId, &'a HirInstruction>,
    cache: HashMap<HirId, Bits>,
}

impl Facts<'_> {
    fn get(&mut self, id: HirId, depth: u32) -> Bits {
        if let Some(b) = self.cache.get(&id) {
            return *b;
        }
        let Some(value) = self.func.values.get(&id) else {
            return Bits::default();
        };
        let Some(w) = width(&value.ty) else {
            return Bits::default();
        };
        let m = mask(w);
        if let HirValueKind::Constant(c) = &value.kind {
            return constant(c).map_or(Bits::default(), |v| Bits {
                zero: !v & m,
                one: v & m,
            });
        }
        if depth == 0 {
            return Bits::default();
        }
        let same_width = |v| self.func.values.get(&v).and_then(|v| width(&v.ty)) == Some(w);
        let bits = match self.defs.get(&id).copied() {
            Some(HirInstruction::Binary {
                op, left, right, ..
            }) if same_width(*left) && same_width(*right) => {
                let a = self.get(*left, depth - 1);
                let b = self.get(*right, depth - 1);
                a.binary(*op, b)
            }
            Some(HirInstruction::Unary {
                op: UnaryOp::Not,
                operand,
                ..
            }) if same_width(*operand) => {
                let a = self.get(*operand, depth - 1);
                Bits {
                    zero: a.one,
                    one: a.zero,
                }
            }
            Some(HirInstruction::Select {
                true_val,
                false_val,
                ..
            }) if same_width(*true_val) && same_width(*false_val) => {
                let a = self.get(*true_val, depth - 1);
                let b = self.get(*false_val, depth - 1);
                Bits {
                    zero: a.zero & b.zero,
                    one: a.one & b.one,
                }
            }
            Some(HirInstruction::Cast { op, operand, .. }) => {
                let from = self.func.values.get(operand).and_then(|v| width(&v.ty));
                match (op, from) {
                    (CastOp::Trunc, Some(from)) if from > w => self.get(*operand, depth - 1),
                    (CastOp::Bitcast, Some(from)) if from == w => self.get(*operand, depth - 1),
                    (CastOp::ZExt | CastOp::SExt, Some(from)) if from < w => {
                        let mut b = self.get(*operand, depth - 1);
                        let high = m & !mask(from);
                        if *op == CastOp::ZExt || b.zero & (1 << (from - 1)) != 0 {
                            b.zero |= high;
                        } else if b.one & (1 << (from - 1)) != 0 {
                            b.one |= high;
                        }
                        b
                    }
                    _ => Bits::default(),
                }
            }
            _ => Bits::default(),
        };
        let bits = Bits {
            zero: bits.zero & m,
            one: bits.one & m,
        };
        self.cache.insert(id, bits);
        bits
    }
}

pub(crate) fn fold_comparisons(func: &mut HirFunction) -> usize {
    // `ZYNTAX_DISABLE_KNOWN_BITS=1` keeps integer tag checks; safe to run with.
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_KNOWN_BITS").is_some()) {
        return 0;
    }
    let instructions = || func.blocks.values().flat_map(|b| &b.instructions);
    let comparisons: Vec<_> = instructions()
        .filter_map(|i| match i {
            HirInstruction::Binary {
                op: op @ (BinaryOp::Eq | BinaryOp::Ne),
                result,
                left,
                right,
                ..
            } if width(&func.values.get(left)?.ty)? == width(&func.values.get(right)?.ty)? => {
                Some((*result, *op, *left, *right))
            }
            _ => None,
        })
        .collect();
    if comparisons.is_empty() {
        return 0;
    }
    let defs: HashMap<_, _> = instructions()
        .filter_map(|i| match i {
            HirInstruction::Binary {
                op: BinaryOp::And | BinaryOp::Or | BinaryOp::Xor,
                result,
                ..
            }
            | HirInstruction::Unary {
                op: UnaryOp::Not,
                result,
                ..
            }
            | HirInstruction::Select { result, .. }
            | HirInstruction::Cast { result, .. } => Some((*result, i)),
            _ => None,
        })
        .collect();
    if defs.is_empty() {
        return 0;
    }
    let mut facts = Facts {
        func,
        defs,
        cache: HashMap::new(),
    };
    let mut folded = HashMap::new();
    for (result, op, left, right) in comparisons {
        let a = facts.get(left, 16);
        let b = facts.get(right, 16);
        if let Some(eq) = a.equal(b, width(&func.values[&left].ty).unwrap()) {
            folded.insert(result, if op == BinaryOp::Eq { eq } else { !eq });
        }
    }
    // Only comparisons disappear. Effectful producers are still evaluated.
    for (id, value) in &folded {
        let v = func.values.get_mut(id).unwrap();
        v.kind = HirValueKind::Constant(HirConstant::Bool(*value));
        v.ty = HirType::Bool;
    }
    if !folded.is_empty() {
        for block in func.blocks.values_mut() {
            block
                .instructions
                .retain(|i| i.result_id().is_none_or(|r| !folded.contains_key(&r)));
        }
    }
    folded.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::InternedString;

    fn function() -> HirFunction {
        HirFunction::new(
            InternedString::new_global("bits"),
            HirFunctionSignature {
                params: vec![],
                returns: vec![],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
                is_fiber: false,
                effects: vec![],
                is_pure: false,
            },
        )
    }

    fn emit(f: &mut HirFunction, ty: HirType, make: impl FnOnce(HirId) -> HirInstruction) -> HirId {
        let id = f.create_value(ty, HirValueKind::Instruction);
        f.blocks
            .get_mut(&f.entry_block)
            .unwrap()
            .instructions
            .push(make(id));
        id
    }

    fn facts(f: &HirFunction, id: HirId) -> Bits {
        Facts {
            func: f,
            defs: f
                .blocks
                .values()
                .flat_map(|b| &b.instructions)
                .filter_map(|i| Some((i.result_id()?, i)))
                .collect(),
            cache: HashMap::new(),
        }
        .get(id, 16)
    }

    #[test]
    fn exhaustive_four_bit_facts_never_exclude_a_possible_value() {
        let states: Vec<_> = (0..16)
            .flat_map(|zero| {
                (0..16).filter_map(move |one| (zero & one == 0).then_some(Bits { zero, one }))
            })
            .collect();
        let matches = |b: Bits, v: u128| v & b.zero == 0 && v & b.one == b.one;
        for &a in &states {
            for &b in &states {
                for x in (0..16).filter(|&x| matches(a, x)) {
                    for y in (0..16).filter(|&y| matches(b, y)) {
                        for (op, v) in [
                            (BinaryOp::And, x & y),
                            (BinaryOp::Or, x | y),
                            (BinaryOp::Xor, x ^ y),
                        ] {
                            assert!(matches(a.binary(op, b), v), "{a:?} {op:?} {b:?}");
                        }
                        if let Some(eq) = a.equal(b, 4) {
                            assert_eq!(eq, x == y);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sign_and_zero_extensions_and_truncation_keep_their_widths() {
        let mut f = function();
        let x = f.create_value(HirType::I8, HirValueKind::Parameter(0));
        let high = f.create_value(HirType::I8, HirValueKind::Constant(HirConstant::I8(-128)));
        let bits = emit(&mut f, HirType::I8, |result| HirInstruction::Binary {
            result,
            op: BinaryOp::Or,
            ty: HirType::I8,
            left: x,
            right: high,
        });
        for (op, expected_zero, expected_one) in
            [(CastOp::ZExt, 0xff00, 0x80), (CastOp::SExt, 0, 0xff80)]
        {
            let wide = emit(&mut f, HirType::I16, |result| HirInstruction::Cast {
                result,
                op,
                ty: HirType::I16,
                operand: bits,
            });
            let b = facts(&f, wide);
            assert_eq!((b.zero, b.one), (expected_zero, expected_one));
            let narrow = emit(&mut f, HirType::U8, |result| HirInstruction::Cast {
                result,
                op: CastOp::Trunc,
                ty: HirType::U8,
                operand: wide,
            });
            let b = facts(&f, narrow);
            assert_eq!((b.zero, b.one), (0, 0x80));
        }
        for (ty, c, expected) in [
            (HirType::I8, HirConstant::I8(-1), 255),
            (HirType::I64, HirConstant::I64(-1), u64::MAX as u128),
            (HirType::U128, HirConstant::U128(1 << 127), 1 << 127),
            (HirType::I128, HirConstant::I128(-1), u128::MAX),
        ] {
            let id = f.create_value(ty.clone(), HirValueKind::Constant(c));
            let b = facts(&f, id);
            assert_eq!(
                (b.zero, b.one),
                (!expected & mask(width(&ty).unwrap()), expected)
            );
        }
    }

    #[test]
    fn select_keeps_only_bits_common_to_both_arms_and_unknowns_stay_unknown() {
        let mut f = function();
        let yes = f.create_value(HirType::Bool, HirValueKind::Parameter(0));
        let a = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(4)));
        let b = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(7)));
        let pick = emit(&mut f, HirType::I64, |result| HirInstruction::Select {
            result,
            ty: HirType::I64,
            condition: yes,
            true_val: a,
            false_val: b,
        });
        let bits = facts(&f, pick);
        assert_eq!((bits.zero, bits.one), (!7 & mask(64), 4));
        let unknown = f.create_value(HirType::I64, HirValueKind::Instruction);
        assert_eq!(facts(&f, unknown).one | facts(&f, unknown).zero, 0);
        let opaque = emit(&mut f, HirType::I64, |result| HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Symbol("effect".into()),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        });
        assert_eq!(facts(&f, opaque).one | facts(&f, opaque).zero, 0);
        let float = f.create_value(HirType::F64, HirValueKind::Constant(HirConstant::F64(4.0)));
        let cast = emit(&mut f, HirType::I64, |result| HirInstruction::Cast {
            result,
            op: CastOp::Bitcast,
            ty: HirType::I64,
            operand: float,
        });
        assert_eq!(facts(&f, cast).one | facts(&f, cast).zero, 0);
    }

    #[test]
    fn a_deep_expression_and_a_cycle_stop_at_the_budget() {
        let mut f = function();
        let mut v = f.create_value(HirType::I64, HirValueKind::Parameter(0));
        for _ in 0..100 {
            v = emit(&mut f, HirType::I64, |result| HirInstruction::Unary {
                result,
                op: UnaryOp::Not,
                ty: HirType::I64,
                operand: v,
            });
        }
        assert_eq!(facts(&f, v).one | facts(&f, v).zero, 0);
        let cycle = emit(&mut f, HirType::I64, |result| HirInstruction::Unary {
            result,
            op: UnaryOp::Not,
            ty: HirType::I64,
            operand: result,
        });
        assert_eq!(facts(&f, cycle).one | facts(&f, cycle).zero, 0);
    }
}
