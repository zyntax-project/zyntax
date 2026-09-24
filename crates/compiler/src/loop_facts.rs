//! Facts about a function's loops that hold before it runs.
//!
//! A counted loop is one whose header compares an induction phi against
//! a constant bound, where the phi enters at a constant and steps by a
//! constant on every back edge. Its trip count is then known statically.
//! A function holding such a loop, alone or nested in other counted
//! loops, with enough iterations is hot before its first call.

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, HirConstant, HirFunction, HirId, HirInstruction, HirTerminator, HirType, HirValueKind,
};
use std::collections::HashMap;

/// Iterations, counted through the enclosing counted loops, from which a
/// loop makes its function worth compiling before its first call.
pub const STATIC_HOT_TRIPS: u64 = 4096;

/// A loop whose trip count is known statically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountedLoop {
    /// The header phi the exit test compares.
    pub induction: HirId,
    /// The induction's value on entry.
    pub init: i128,
    /// What each back edge adds to the induction.
    pub step: i128,
    /// Times the body runs.
    pub trips: i128,
}

/// Why a loop is not counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotCounted {
    /// The header does not branch on a compare of a header phi.
    Shape,
    /// The induction does not step by a constant towards the bound.
    Step,
    /// The bound or the initial value is not a constant, or the
    /// induction would leave its type's range before the exit.
    Bounds,
}

/// The trip count of `lp`, with `konst` naming the constant each value
/// id stands for. The header branches into the body while `i < bound`
/// (`<=`, or `>` / `>=` with a negative step) and out of it otherwise;
/// `i` is a header phi entering with one constant from outside the body
/// and one value `i + step` from inside it.
pub fn counted_loop(
    func: &HirFunction,
    lp: &NaturalLoop,
    konst: &dyn Fn(HirId) -> Option<i128>,
) -> Result<CountedLoop, NotCounted> {
    let header = func.blocks.get(&lp.header).ok_or(NotCounted::Shape)?;
    let HirTerminator::CondBranch {
        condition,
        true_target,
        false_target,
    } = &header.terminator
    else {
        return Err(NotCounted::Shape);
    };
    if !lp.body.contains(true_target) || lp.body.contains(false_target) {
        return Err(NotCounted::Shape);
    }
    let (op, left, right) = header
        .instructions
        .iter()
        .find_map(|inst| match inst {
            HirInstruction::Binary {
                op,
                left,
                right,
                result,
                ..
            } if result == condition => Some((*op, *left, *right)),
            _ => None,
        })
        .ok_or(NotCounted::Shape)?;
    let (upward, inclusive) = match op {
        BinaryOp::Lt => (true, false),
        BinaryOp::Le => (true, true),
        BinaryOp::Gt => (false, false),
        BinaryOp::Ge => (false, true),
        _ => return Err(NotCounted::Shape),
    };
    let ind = header
        .phis
        .iter()
        .find(|p| p.result == left)
        .ok_or(NotCounted::Shape)?;
    let bound = konst(right).ok_or(NotCounted::Bounds)?;

    // One value from outside the body, one from inside it.
    let mut entering: Option<HirId> = None;
    let mut next: Option<HirId> = None;
    for (val, pred) in &ind.incoming {
        let slot = if lp.body.contains(pred) {
            &mut next
        } else {
            &mut entering
        };
        match slot {
            Some(seen) if seen != val => return Err(NotCounted::Shape),
            _ => *slot = Some(*val),
        }
    }
    let (Some(entering), Some(next)) = (entering, next) else {
        return Err(NotCounted::Shape);
    };

    let step = lp
        .body
        .iter()
        .filter_map(|b| func.blocks.get(b))
        .flat_map(|b| b.instructions.iter())
        .find_map(|inst| match inst {
            HirInstruction::Binary {
                op: BinaryOp::Add,
                left: l,
                right: r,
                result,
                ..
            } if *result == next => {
                if *l == ind.result {
                    Some(konst(*r))
                } else if *r == ind.result {
                    Some(konst(*l))
                } else {
                    Some(None)
                }
            }
            _ => None,
        })
        .flatten()
        .ok_or(NotCounted::Step)?;
    if (upward && step <= 0) || (!upward && step >= 0) {
        return Err(NotCounted::Step);
    }
    let init = konst(entering).ok_or(NotCounted::Bounds)?;

    // Distance to cover and the stride covering it, both positive.
    let (distance, stride) = if upward {
        (bound - init, step)
    } else {
        (init - bound, -step)
    };
    let trips = if inclusive {
        if distance < 0 {
            0
        } else {
            distance / stride + 1
        }
    } else if distance <= 0 {
        0
    } else {
        (distance + stride - 1) / stride
    };
    // Every value the induction takes, the one that fails the test
    // included, is in its type's range, so none wraps.
    let (lo, hi) = int_range(&ind.ty).ok_or(NotCounted::Bounds)?;
    let last = init
        .checked_add(trips.checked_mul(step).ok_or(NotCounted::Bounds)?)
        .ok_or(NotCounted::Bounds)?;
    if init < lo || init > hi || last < lo || last > hi {
        return Err(NotCounted::Bounds);
    }
    Ok(CountedLoop {
        induction: ind.result,
        init,
        step,
        trips,
    })
}

/// The static trip count of the loop headed by `header`, or `None` when
/// `header` heads no counted loop.
pub fn static_trip_count(func: &HirFunction, header: HirId) -> Option<u64> {
    let dt = DominatorTree::new(func);
    let forest = LoopForest::detect(func, &dt);
    let lp = forest.loops().iter().find(|lp| lp.header == header)?;
    let defs = Defs::of(func);
    let trips = counted_loop(func, lp, &|id| defs.constant(func, id))
        .ok()?
        .trips;
    u64::try_from(trips).ok()
}

/// Whether `func` holds a loop hot before its first call: see
/// [`hot_static_loops`].
pub fn hot_static_loop(func: &HirFunction) -> bool {
    !hot_static_loops(func).is_empty()
}

/// The headers of the counted loops of `func`, each with how often it
/// runs: its trip count times those of the counted loops around it. Most
/// often first, and none unless the first runs at least
/// [`STATIC_HOT_TRIPS`] times.
pub fn hot_static_loops(func: &HirFunction) -> Vec<(HirId, u64)> {
    if !has_counted_header(func) {
        return Vec::new();
    }
    let dt = DominatorTree::new(func);
    let forest = LoopForest::detect(func, &dt);
    let defs = Defs::of(func);
    let trips: Vec<Option<u64>> = forest
        .loops()
        .iter()
        .map(|lp| {
            counted_loop(func, lp, &|id| defs.constant(func, id))
                .ok()
                .and_then(|c| u64::try_from(c.trips).ok())
        })
        .collect();
    let mut hot: Vec<(HirId, u64)> = Vec::new();
    for (idx, lp) in forest.loops().iter().enumerate() {
        let Some(own) = trips[idx] else {
            continue;
        };
        let mut runs = own;
        let mut outer = forest.parent_of(idx);
        while let Some(o) = outer {
            if let Some(t) = trips[o] {
                runs = runs.saturating_mul(t);
            }
            outer = forest.parent_of(o);
        }
        hot.push((lp.header, runs));
    }
    hot.sort_by(|a, b| b.1.cmp(&a.1));
    if hot.first().is_none_or(|(_, runs)| *runs < STATIC_HOT_TRIPS) {
        hot.clear();
    }
    hot
}

/// Whether some block could head a counted loop: it has phis and
/// branches on an ordered compare it makes. Spares the loop analysis
/// of a function that has none.
fn has_counted_header(func: &HirFunction) -> bool {
    func.blocks.values().any(|b| {
        let HirTerminator::CondBranch { condition, .. } = &b.terminator else {
            return false;
        };
        !b.phis.is_empty()
            && b.instructions.iter().any(|inst| {
                matches!(
                    inst,
                    HirInstruction::Binary {
                        op: BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge,
                        result,
                        ..
                    } if result == condition
                )
            })
    })
}

/// The integer constant `id` is, when it is one.
pub fn const_int(func: &HirFunction, id: HirId) -> Option<i128> {
    match &func.values.get(&id)?.kind {
        HirValueKind::Constant(c) => match c {
            HirConstant::I8(x) => Some(*x as i128),
            HirConstant::I16(x) => Some(*x as i128),
            HirConstant::I32(x) => Some(*x as i128),
            HirConstant::I64(x) => Some(*x as i128),
            HirConstant::I128(x) => Some(*x),
            HirConstant::U8(x) => Some(*x as i128),
            HirConstant::U16(x) => Some(*x as i128),
            HirConstant::U32(x) => Some(*x as i128),
            HirConstant::U64(x) => Some(*x as i128),
            HirConstant::U128(x) => i128::try_from(*x).ok(),
            _ => None,
        },
        _ => None,
    }
}

/// The values an integer type holds.
fn int_range(ty: &HirType) -> Option<(i128, i128)> {
    Some(match ty {
        HirType::I8 => (i8::MIN as i128, i8::MAX as i128),
        HirType::I16 => (i16::MIN as i128, i16::MAX as i128),
        HirType::I32 => (i32::MIN as i128, i32::MAX as i128),
        HirType::I64 => (i64::MIN as i128, i64::MAX as i128),
        HirType::I128 => (i128::MIN, i128::MAX),
        HirType::U8 => (0, u8::MAX as i128),
        HirType::U16 => (0, u16::MAX as i128),
        HirType::U32 => (0, u32::MAX as i128),
        HirType::U64 => (0, u64::MAX as i128),
        HirType::U128 => (0, i128::MAX),
        _ => return None,
    })
}

/// Each instruction by the value it defines, so a bound a body computes
/// from constants (`35 * size` before the optimiser folds it) counts as
/// the constant it is.
struct Defs<'a>(HashMap<HirId, &'a HirInstruction>);

impl<'a> Defs<'a> {
    fn of(func: &'a HirFunction) -> Self {
        let mut defs = HashMap::new();
        for block in func.blocks.values() {
            for inst in &block.instructions {
                if let HirInstruction::Binary { result, .. } = inst {
                    defs.insert(*result, inst);
                }
            }
        }
        Defs(defs)
    }

    fn constant(&self, func: &HirFunction, id: HirId) -> Option<i128> {
        self.fold(func, id, 8)
    }

    fn fold(&self, func: &HirFunction, id: HirId, depth: u32) -> Option<i128> {
        if let Some(c) = const_int(func, id) {
            return Some(c);
        }
        if depth == 0 {
            return None;
        }
        let HirInstruction::Binary {
            op,
            left,
            right,
            ty,
            ..
        } = self.0.get(&id)?
        else {
            return None;
        };
        let l = self.fold(func, *left, depth - 1)?;
        let r = self.fold(func, *right, depth - 1)?;
        let v = match op {
            BinaryOp::Add => l.checked_add(r)?,
            BinaryOp::Sub => l.checked_sub(r)?,
            BinaryOp::Mul => l.checked_mul(r)?,
            _ => return None,
        };
        let (lo, hi) = int_range(ty)?;
        (lo..=hi).contains(&v).then_some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirPhi, HirValue};
    use zyntax_typed_ast::InternedString;

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

    fn i64_const(f: &mut HirFunction, v: i64) -> HirId {
        value(f, HirType::I64, HirValueKind::Constant(HirConstant::I64(v)))
    }

    fn link(f: &mut HirFunction, from: HirId, to: HirId) {
        f.blocks.get_mut(&from).unwrap().successors.push(to);
        f.blocks.get_mut(&to).unwrap().predecessors.push(from);
    }

    /// The loop of a `for i in range(n)` main as lowered:
    ///
    /// ```text
    /// block2: %i = phi [0, block1], [%next, block7]
    ///         %c = lt %i, %n
    ///         brcond %c, block3, block8
    /// block3: ... br block7
    /// block7: %next = add %i, 1 ; br block2
    /// ```
    ///
    /// Returns the function and its header.
    fn range_loop(bound: impl FnOnce(&mut HirFunction) -> HirId) -> (HirFunction, HirId) {
        let sig = HirFunctionSignature {
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
        };
        let mut f = HirFunction::new(InternedString::new_global("main"), sig);
        let [entry, pre, header, body, latch, exit] = std::array::from_fn(|_| HirId::new());
        f.entry_block = entry;
        f.blocks.clear();
        for id in [entry, pre, header, body, latch, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        let zero = i64_const(&mut f, 0);
        let one = i64_const(&mut f, 1);
        let n = bound(&mut f);
        let i = value(&mut f, HirType::I64, HirValueKind::Instruction);
        let next = value(&mut f, HirType::I64, HirValueKind::Instruction);
        let c = value(&mut f, HirType::Bool, HirValueKind::Instruction);

        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: pre };
        f.blocks.get_mut(&pre).unwrap().terminator = HirTerminator::Branch { target: header };
        {
            let h = f.blocks.get_mut(&header).unwrap();
            h.phis.push(HirPhi {
                result: i,
                ty: HirType::I64,
                incoming: vec![(zero, pre), (next, latch)],
            });
            h.instructions.push(HirInstruction::Binary {
                op: BinaryOp::Lt,
                result: c,
                ty: HirType::I64,
                left: i,
                right: n,
            });
            h.terminator = HirTerminator::CondBranch {
                condition: c,
                true_target: body,
                false_target: exit,
            };
        }
        f.blocks.get_mut(&body).unwrap().terminator = HirTerminator::Branch { target: latch };
        {
            let l = f.blocks.get_mut(&latch).unwrap();
            l.instructions.push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: next,
                ty: HirType::I64,
                left: i,
                right: one,
            });
            l.terminator = HirTerminator::Branch { target: header };
        }
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![i] };
        for (from, to) in [
            (entry, pre),
            (pre, header),
            (header, body),
            (header, exit),
            (body, latch),
            (latch, header),
        ] {
            link(&mut f, from, to);
        }
        (f, header)
    }

    /// `for i in range(outer): for j in range(inner): pass`, with its
    /// outer and inner headers.
    fn nested(outer: i64, inner: i64) -> (HirFunction, HirId, HirId) {
        let (mut f, oh) = range_loop(|f| i64_const(f, outer));
        // Put a second counted loop between the outer header and its
        // body block.
        let obody = match &f.blocks[&oh].terminator {
            HirTerminator::CondBranch { true_target, .. } => *true_target,
            _ => unreachable!(),
        };
        let [ih, ibody, iexit] = std::array::from_fn(|_| HirId::new());
        for id in [ih, ibody, iexit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        let zero = i64_const(&mut f, 0);
        let one = i64_const(&mut f, 1);
        let n = i64_const(&mut f, inner);
        let j = value(&mut f, HirType::I64, HirValueKind::Instruction);
        let next = value(&mut f, HirType::I64, HirValueKind::Instruction);
        let c = value(&mut f, HirType::Bool, HirValueKind::Instruction);
        {
            let h = f.blocks.get_mut(&ih).unwrap();
            h.phis.push(HirPhi {
                result: j,
                ty: HirType::I64,
                incoming: vec![(zero, oh), (next, ibody)],
            });
            h.instructions.push(HirInstruction::Binary {
                op: BinaryOp::Lt,
                result: c,
                ty: HirType::I64,
                left: j,
                right: n,
            });
            h.terminator = HirTerminator::CondBranch {
                condition: c,
                true_target: ibody,
                false_target: iexit,
            };
        }
        {
            let b = f.blocks.get_mut(&ibody).unwrap();
            b.instructions.push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: next,
                ty: HirType::I64,
                left: j,
                right: one,
            });
            b.terminator = HirTerminator::Branch { target: ih };
        }
        f.blocks.get_mut(&iexit).unwrap().terminator = HirTerminator::Branch { target: obody };
        // The outer header enters the inner loop instead of its body.
        let o = f.blocks.get_mut(&oh).unwrap();
        if let HirTerminator::CondBranch { true_target, .. } = &mut o.terminator {
            *true_target = ih;
        }
        o.successors.retain(|s| *s != obody);
        f.blocks
            .get_mut(&obody)
            .unwrap()
            .predecessors
            .retain(|p| *p != oh);
        for (from, to) in [
            (oh, ih),
            (ih, ibody),
            (ih, iexit),
            (ibody, ih),
            (iexit, obody),
        ] {
            link(&mut f, from, to);
        }
        (f, oh, ih)
    }

    #[test]
    fn a_nested_loop_runs_its_trips_times_the_outer_ones() {
        let (f, outer, inner) = nested(4, 2000);
        assert_eq!(static_trip_count(&f, inner), Some(2000));
        assert_eq!(hot_static_loops(&f), vec![(inner, 8000), (outer, 4)]);
        let (f, _, _) = nested(4, 1000);
        assert!(!hot_static_loop(&f));
    }

    #[test]
    fn a_range_loop_with_a_constant_bound_is_counted() {
        let (f, header) = range_loop(|f| i64_const(f, 1_000_000));
        assert_eq!(static_trip_count(&f, header), Some(1_000_000));
        assert!(hot_static_loop(&f));
        assert_eq!(hot_static_loops(&f), vec![(header, 1_000_000)]);
    }

    #[test]
    fn a_bound_computed_from_constants_is_counted() {
        let (f, header) = range_loop(|f| {
            let a = i64_const(f, 35);
            let b = i64_const(f, 25);
            let m = value(f, HirType::I64, HirValueKind::Instruction);
            let entry = f.entry_block;
            f.blocks
                .get_mut(&entry)
                .unwrap()
                .instructions
                .push(HirInstruction::Binary {
                    op: BinaryOp::Mul,
                    result: m,
                    ty: HirType::I64,
                    left: a,
                    right: b,
                });
            m
        });
        assert_eq!(static_trip_count(&f, header), Some(875));
        assert!(!hot_static_loop(&f));
    }

    #[test]
    fn a_parameter_bound_is_not_counted() {
        let (f, header) = range_loop(|f| value(f, HirType::I64, HirValueKind::Parameter(0)));
        assert_eq!(static_trip_count(&f, header), None);
        assert!(!hot_static_loop(&f));
    }

    #[test]
    fn a_short_loop_is_not_hot() {
        let (f, header) = range_loop(|f| i64_const(f, 100));
        assert_eq!(static_trip_count(&f, header), Some(100));
        assert!(!hot_static_loop(&f));
    }

    #[test]
    fn a_bound_past_the_induction_range_is_not_counted() {
        let (mut f, header) = range_loop(|f| i64_const(f, i64::MAX));
        // A step of 2 from 0 leaves i64 before reaching i64::MAX.
        let two = i64_const(&mut f, 2);
        for block in f.blocks.values_mut() {
            for inst in &mut block.instructions {
                if let HirInstruction::Binary {
                    op: BinaryOp::Add,
                    right,
                    ..
                } = inst
                {
                    *right = two;
                }
            }
        }
        assert_eq!(static_trip_count(&f, header), None);
    }
}
