//! Which counted loops have independent iterations.
//!
//! Spreading a loop across cores is safe exactly when no iteration can
//! observe another's writes. Proving it means saying, for every address
//! the body touches, where that address can be relative to the counter.
//! An iteration that writes only inside a band of its own touches
//! nothing another iteration touches, whichever core runs it and in
//! whatever order.
//!
//! The band is what an index says about the counter, and there are four
//! answers:
//!
//! * The same address every time. Fine to read, a race to write.
//! * The counter itself, so iteration `i` touches element `i`.
//! * `i * n` plus something confined to `[0, n)`, so iteration `i`
//!   touches somewhere inside its own run of `n` elements. This is the
//!   row of a matrix, and it is the shape that carries the loops worth
//!   spreading.
//! * Not understood.
//!
//! Where a band starts counts as much as how wide it is. `a[i]` and
//! `a[i + 1]` both advance one element per iteration, but the second
//! sits in the band the next iteration owns, so through one buffer they
//! are a dependence and not a match. A band is therefore a width and a
//! starting point, and two accesses agree only when both agree.
//!
//! Reads and writes are held to different standards, because two reads
//! never conflict. Every write must be banded. A read must be banded
//! only where it goes through storage something also writes.
//!
//! ## What the analysis cannot see, and says instead
//!
//! Two base pointers with different SSA ids may still name the same
//! storage: nothing in a function body rules out its caller having
//! passed one buffer twice. Rather than assume otherwise, a loop that
//! is independent *provided* certain things hold is reported with those
//! things named, as [`Obligation`]s, and counted separately from one
//! that needs nothing. Nothing here decides that a loop is safe on an
//! assumption its caller never agreed to.
//!
//! The agreement the language does offer is a parameter declared `mut`
//! or `own`: nothing else names it, and every caller was held to that
//! by `exclusive_args` before this runs. Two parameters where one is
//! exclusive are therefore two buffers, and the obligation is settled
//! rather than reported. Writing a kernel that way is what turns it
//! from conditionally independent into independent.
//!
//! What disqualifies a loop outright:
//!
//! * A value carried between iterations. A header phi other than the
//!   counter is an accumulator, and an accumulator two cores advance at
//!   once is a race. Reductions need a different shape (per-core
//!   partials combined at the end), so they are refused here rather
//!   than silently split.
//! * A write whose address is not banded, since nothing then bounds
//!   which iteration reaches it.
//! * A read through storage the loop writes, at an address not in the
//!   same band as the write. No obligation fixes that: the two are the
//!   same buffer by construction.
//! * A base pointer the loop itself computes, which may differ per
//!   iteration and so says nothing about disjointness.
//! * A call. Its effects are not visible here, so it could touch
//!   anything. Pure arithmetic intrinsics are the exception, having no
//!   effects to worry about, and so is any instruction on the list of
//!   those that compute from values rather than reach memory. That list
//!   names what is allowed rather than what is not, so an instruction
//!   added later is refused until someone decides it belongs.
//! * An access wider than the counter's step. A band says where an
//!   iteration starts, not where it stops, and a four-lane store at
//!   `a[i]` covers `i` through `i + 4`. A vectorized loop moves its
//!   counter by the lane count, which is what makes the two meet; a
//!   loop that does not is refused.
//!
//! This decides safety only. Whether spreading a given loop is
//! worthwhile is a separate question of trip count and work per
//! iteration, and belongs where the dispatch is chosen.

use std::collections::{HashMap, HashSet};

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, CastOp, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirTerminator,
    HirValueKind, Intrinsic,
};

/// How far apart two iterations' accesses are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stride {
    /// One element: iteration `i` touches element `i`.
    One,
    /// The run of elements named by this value, which the loop does not
    /// change: iteration `i` touches somewhere in its own run.
    Elements(HirId),
}

/// Something a dispatch site must establish before spreading the loop.
///
/// Both are about values already computed before the loop begins, so
/// both can be settled where the dispatch is decided rather than per
/// iteration. `SameCount` is a comparison. `Disjoint` is a claim about
/// storage: it needs either extents to compare or a parameter whose
/// ownership already says no one else holds the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Obligation {
    /// These two base pointers must not reach the same storage. Needed
    /// where one is written and the other is touched at an address the
    /// band does not cover.
    Disjoint(HirId, HirId),
    /// These two counts must be equal. Needed where an iteration's band
    /// is `n` elements wide but the counter walking inside it stops at
    /// a separately named `m`: the band only contains the walk when the
    /// two agree.
    SameCount(HirId, HirId),
}

/// One loop whose iterations do not interfere.
#[derive(Debug, Clone)]
pub struct ParallelLoop {
    /// Header block, which is also the back-edge target.
    pub header: HirId,
    /// The counter's phi result.
    pub induction: HirId,
    /// Base pointers the body reads through.
    pub reads: Vec<HirId>,
    /// Base pointers the body writes through.
    pub writes: Vec<HirId>,
    /// What must hold for the independence to be real. Empty means the
    /// loop is independent as written.
    pub obligations: Vec<Obligation>,
}

impl ParallelLoop {
    /// Whether independence was shown without asking anything of the
    /// caller.
    pub fn is_unconditional(&self) -> bool {
        self.obligations.is_empty()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ParallelStats {
    /// Loops shown independent outright.
    pub independent: usize,
    /// Loops independent provided their obligations hold.
    pub conditional: usize,
    /// Loops refused because a value is carried between iterations.
    pub carried_dependency: usize,
    /// Loops refused because something in the body was not understood.
    pub opaque_body: usize,
}

/// Whether an intrinsic is pure arithmetic, so a call to it says nothing
/// about what memory an iteration touches.
fn is_pure_intrinsic(i: &Intrinsic) -> bool {
    matches!(
        i,
        Intrinsic::Sqrt
            | Intrinsic::Rsqrt
            | Intrinsic::Fabs
            | Intrinsic::Fma
            | Intrinsic::Sin
            | Intrinsic::Cos
            | Intrinsic::Pow
            | Intrinsic::Log
            | Intrinsic::Exp
            | Intrinsic::Ctpop
            | Intrinsic::Ctlz
            | Intrinsic::Cttz
            | Intrinsic::Bswap
            | Intrinsic::SizeOf
            | Intrinsic::AlignOf
    )
}

/// Whether `inst` defines `value`.
///
/// Only the instructions this analysis can meet need naming; anything
/// unrecognised is refused by the caller before it gets here.
fn defines(inst: &HirInstruction, value: HirId) -> bool {
    match inst {
        HirInstruction::Binary { result, .. }
        | HirInstruction::Unary { result, .. }
        | HirInstruction::Cast { result, .. }
        | HirInstruction::GetElementPtr { result, .. }
        | HirInstruction::Load { result, .. }
        | HirInstruction::VectorLoad { result, .. }
        | HirInstruction::VectorSplat { result, .. }
        | HirInstruction::ExtractValue { result, .. }
        | HirInstruction::InsertValue { result, .. }
        | HirInstruction::Select { result, .. }
        | HirInstruction::Alloca { result, .. } => *result == value,
        HirInstruction::Call { result, .. } => *result == Some(value),
        _ => false,
    }
}

/// Everything the loop body defines, so "does this change between
/// iterations" is one lookup rather than a scan per question.
fn values_defined_in(func: &HirFunction, blocks: &HashSet<HirId>) -> HashSet<HirId> {
    let mut defined = HashSet::new();
    for b in blocks {
        let Some(blk) = func.blocks.get(b) else {
            continue;
        };
        for phi in &blk.phis {
            defined.insert(phi.result);
        }
        for inst in &blk.instructions {
            if let Some(r) = result_of(inst) {
                defined.insert(r);
            }
        }
    }
    defined
}

/// The value an instruction defines, where it defines one.
fn result_of(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. }
        | HirInstruction::Unary { result, .. }
        | HirInstruction::Cast { result, .. }
        | HirInstruction::GetElementPtr { result, .. }
        | HirInstruction::Load { result, .. }
        | HirInstruction::VectorLoad { result, .. }
        | HirInstruction::VectorSplat { result, .. }
        | HirInstruction::ExtractValue { result, .. }
        | HirInstruction::InsertValue { result, .. }
        | HirInstruction::Select { result, .. }
        | HirInstruction::Alloca { result, .. } => Some(*result),
        HirInstruction::Call { result, .. } => *result,
        _ => None,
    }
}

/// Where each value in the loop body is defined and by what.
struct Defs<'a> {
    by_value: HashMap<HirId, &'a HirInstruction>,
    defined: HashSet<HirId>,
}

impl<'a> Defs<'a> {
    /// Index every instruction in the function, but record only the
    /// loop body as what changes. A counter's starting value is defined
    /// outside the loop and still has to be read, so where a value is
    /// defined and whether the loop can change it are separate
    /// questions.
    fn build(func: &'a HirFunction, blocks: &HashSet<HirId>) -> Self {
        let mut by_value = HashMap::new();
        for blk in func.blocks.values() {
            for inst in &blk.instructions {
                if let Some(r) = result_of(inst) {
                    by_value.insert(r, inst);
                }
            }
        }
        Defs {
            by_value,
            defined: values_defined_in(func, blocks),
        }
    }

    /// Whether the loop can give this value a different meaning on a
    /// later iteration.
    fn varies(&self, v: HirId) -> bool {
        self.defined.contains(&v)
    }
}

/// Where a band starts, which is as much a part of naming it as how
/// wide it is. Two accesses a fixed distance apart lie in *different*
/// bands, so `a[i]` and `a[i + 1]` are not the same claim even though
/// both advance one element per iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shift {
    /// The band starts where the base does.
    None,
    /// Displaced by one value the loop does not change. Two shifts are
    /// the same when they name the same value, which after common
    /// subexpression elimination is what equal expressions do.
    By(HirId),
    /// Displaced by more than one thing. Never treated as equal to
    /// anything, including another `Several`, since this analysis does
    /// not add them up.
    Several,
}

impl Shift {
    fn plus(self, other: Shift) -> Shift {
        match (self, other) {
            (Shift::None, x) | (x, Shift::None) => x,
            _ => Shift::Several,
        }
    }

    /// Whether two shifts are known to name the same starting point.
    fn same(self, other: Shift) -> bool {
        match (self, other) {
            (Shift::None, Shift::None) => true,
            (Shift::By(a), Shift::By(b)) => a == b,
            _ => false,
        }
    }
}

/// Which run of addresses an iteration owns, as a width and a start.
#[derive(Debug, Clone, Copy)]
struct Band {
    stride: Stride,
    shift: Shift,
}

impl Band {
    /// Whether two accesses land in the band belonging to the same
    /// iteration. Same width and same start; a difference in either
    /// puts iteration `i` of one inside iteration `i + 1` of the other.
    fn same(self, other: Band) -> bool {
        self.stride == other.stride && self.shift.same(other.shift)
    }
}

/// What an address index says about the counter.
#[derive(Debug, Clone, Copy)]
enum Index {
    /// The same value on every iteration.
    Fixed,
    /// Confined to iteration `i`'s own band.
    Banded(Band),
    /// Exactly `shift + i * n`, with nothing yet saying how far past it
    /// the access reaches. A counter confined to `[0, n)` added to this
    /// makes it a band; on its own it is not one, because an `n` of
    /// zero would send every iteration to a single address.
    Scaled { n: HirId, shift: Shift },
    /// Not understood.
    Unknown,
}

impl Index {
    /// The band this index lies in, where it lies in one.
    fn band(self) -> Option<Band> {
        match self {
            Index::Banded(b) => Some(b),
            _ => None,
        }
    }
}

/// A counter an enclosing loop confines, and the count it stops at.
#[derive(Debug, Clone, Copy)]
struct Confined {
    counter: HirId,
    limit: HirId,
}

/// Counters that satisfy `0 <= counter < limit` throughout `block`.
///
/// Each comes from a loop whose body contains `block`: the counter
/// starts at a non-negative constant, advances by a positive one, and
/// the header branches into the body only while it is below the limit.
/// Since a natural loop's body is entered only through its header, and
/// the header's other edge leaves the loop, being in the body means
/// having taken the edge on which the comparison held.
fn confined_counters(
    func: &HirFunction,
    forest: &LoopForest,
    outer: &NaturalLoop,
    outer_defs: &Defs<'_>,
    block: HirId,
) -> Vec<Confined> {
    let mut found = Vec::new();
    for lp in forest.loops() {
        // Only loops properly inside the one being decided, and only
        // where `block` sits in the part the comparison guards.
        if lp.header == outer.header || !lp.body.contains(&block) || block == lp.header {
            continue;
        }
        if !outer.body.contains(&lp.header) {
            continue;
        }
        let Some(header) = func.blocks.get(&lp.header) else {
            continue;
        };
        let HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = &header.terminator
        else {
            continue;
        };
        // The false edge must leave the loop, or the body could also be
        // reached without the comparison having held.
        if lp.body.contains(false_target) || !lp.body.contains(true_target) {
            continue;
        }
        let Some(HirInstruction::Binary {
            op: BinaryOp::Lt,
            left,
            right,
            ..
        }) = outer_defs.by_value.get(condition).copied()
        else {
            continue;
        };
        // The limit has to mean the same thing on every iteration of
        // the loop being decided, or comparing it against a stride
        // says nothing.
        if outer_defs.varies(*right) {
            continue;
        }
        // The counter has to be bounded below as well as above: the
        // comparison at the header only says where it stops, and a
        // band needs to know where it starts.
        if step_of(func, outer_defs, lp, *left).is_none() {
            continue;
        }
        found.push(Confined {
            counter: *left,
            limit: *right,
        });
    }
    found
}

/// How far the counter moves each iteration, where it starts at zero or
/// above and only ever climbs by a fixed amount.
///
/// `None` means one of those could not be shown, in which case nothing
/// bounds the counter from below and it is not a counter this analysis
/// will reason about.
fn step_of(func: &HirFunction, defs: &Defs<'_>, lp: &NaturalLoop, counter: HirId) -> Option<i64> {
    let header = func.blocks.get(&lp.header)?;
    let phi = header.phis.iter().find(|p| p.result == counter)?;
    let mut step = None;
    let mut entered_from_outside = false;
    for (value, from) in &phi.incoming {
        if lp.body.contains(from) {
            // Around the back edge: the counter must have grown.
            let HirInstruction::Binary {
                op: BinaryOp::Add,
                left,
                right,
                ..
            } = defs.by_value.get(value).copied()?
            else {
                return None;
            };
            let amount = if *left == counter {
                *right
            } else if *right == counter {
                *left
            } else {
                return None;
            };
            let c = int_constant(func, defs, amount).filter(|c| *c > 0)?;
            // Two back edges advancing by different amounts leave no
            // single step to compare a vector width against.
            if step.is_some_and(|prev| prev != c) {
                return None;
            }
            step = Some(c);
        } else {
            // From outside: the counter must start at zero or above.
            int_constant(func, defs, *value).filter(|c| *c >= 0)?;
            entered_from_outside = true;
        }
    }
    if entered_from_outside {
        step
    } else {
        None
    }
}

/// The value as a signed integer constant, where it is one.
///
/// A literal reaches the loop through a widening cast before constant
/// folding has run, and a counter that starts at `sext 0` starts at
/// zero. Widening is followed through for that reason; truncation is
/// not, since it can turn a value this reads as positive into one that
/// is not.
fn int_constant(func: &HirFunction, defs: &Defs<'_>, value: HirId) -> Option<i64> {
    let mut value = value;
    for _ in 0..4 {
        if func
            .values
            .get(&value)
            .is_some_and(|v| matches!(v.kind, HirValueKind::Constant(_)))
        {
            break;
        }
        match defs.by_value.get(&value) {
            Some(HirInstruction::Cast {
                op: CastOp::SExt | CastOp::ZExt,
                operand,
                ..
            }) => value = *operand,
            _ => break,
        }
    }
    let v = func.values.get(&value)?;
    let HirValueKind::Constant(c) = &v.kind else {
        return None;
    };
    match c {
        HirConstant::I8(x) => Some(*x as i64),
        HirConstant::I16(x) => Some(*x as i64),
        HirConstant::I32(x) => Some(*x as i64),
        HirConstant::I64(x) => Some(*x),
        HirConstant::U8(x) => Some(*x as i64),
        HirConstant::U16(x) => Some(*x as i64),
        HirConstant::U32(x) => Some(*x as i64),
        HirConstant::U64(x) => i64::try_from(*x).ok(),
        _ => None,
    }
}

/// Read what an index says about the counter.
///
/// `obligations` gathers what the caller would have to establish for
/// the reading to hold; the caller discards them along with the loop if
/// it refuses it for some other reason.
fn classify_index(
    func: &HirFunction,
    defs: &Defs<'_>,
    induction: HirId,
    confined: &[Confined],
    value: HirId,
    obligations: &mut Vec<Obligation>,
    depth: u32,
) -> Index {
    if value == induction {
        return Index::Banded(Band {
            stride: Stride::One,
            shift: Shift::None,
        });
    }
    if !defs.varies(value) {
        return Index::Fixed;
    }
    // An index built from more arithmetic than this is not a shape we
    // recognise, and the recursion has to stop somewhere.
    if depth == 0 {
        return Index::Unknown;
    }
    let Some(HirInstruction::Binary {
        op, left, right, ..
    }) = defs.by_value.get(&value).copied()
    else {
        return Index::Unknown;
    };
    let mut sub = |v: HirId, obs: &mut Vec<Obligation>| {
        classify_index(func, defs, induction, confined, v, obs, depth - 1)
    };
    match op {
        BinaryOp::Mul => {
            // `i * n` for an `n` the loop does not change.
            let scaled = |counter: Index, other: HirId| match counter {
                Index::Banded(Band {
                    stride: Stride::One,
                    shift: Shift::None,
                }) if !defs.varies(other) => Some(Index::Scaled {
                    n: other,
                    shift: Shift::None,
                }),
                _ => None,
            };
            let l = sub(*left, obligations);
            let r = sub(*right, obligations);
            // Recomputing the same product every iteration is not the
            // same as the product changing. Before anything hoists it,
            // `i * cols` sits inside the column loop and is defined in
            // its body, but to that loop it is a fixed offset.
            if matches!((l, r), (Index::Fixed, Index::Fixed)) {
                return Index::Fixed;
            }
            if let Some(idx) = scaled(l, *right) {
                return idx;
            }
            scaled(r, *left).unwrap_or(Index::Unknown)
        }
        BinaryOp::Add => {
            let l = sub(*left, obligations);
            let r = sub(*right, obligations);
            combine_add(l, *left, r, *right, confined, obligations)
        }
        // Taking something away displaces where the band starts. How
        // far is not tracked, so the start stops being comparable, and
        // every access through the base is then held apart by an
        // obligation rather than by a matching band.
        BinaryOp::Sub => {
            let l = sub(*left, obligations);
            if defs.varies(*right) {
                return Index::Unknown;
            }
            match l {
                Index::Banded(b) => Index::Banded(Band {
                    stride: b.stride,
                    shift: Shift::Several,
                }),
                Index::Scaled { n, .. } => Index::Scaled {
                    n,
                    shift: Shift::Several,
                },
                other => other,
            }
        }
        _ => Index::Unknown,
    }
}

/// What adding two index readings gives.
fn combine_add(
    l: Index,
    lv: HirId,
    r: Index,
    rv: HirId,
    confined: &[Confined],
    obligations: &mut Vec<Obligation>,
) -> Index {
    // A fixed amount displaces where the band starts. It does not
    // change how wide the band is or which iteration owns it, but two
    // accesses displaced differently are in different bands, so the
    // displacement is carried along to be compared later.
    let shifted = |idx: Index, by: HirId| match idx {
        Index::Fixed => Index::Fixed,
        Index::Unknown => Index::Unknown,
        Index::Banded(b) => Index::Banded(Band {
            stride: b.stride,
            shift: b.shift.plus(Shift::By(by)),
        }),
        Index::Scaled { n, shift } => Index::Scaled {
            n,
            shift: shift.plus(Shift::By(by)),
        },
    };
    match (l, r) {
        (Index::Fixed, Index::Fixed) => return Index::Fixed,
        (Index::Fixed, other) => return shifted(other, lv),
        (other, Index::Fixed) => return shifted(other, rv),
        _ => {}
    }
    // `i * n` plus a counter that stops before `n` stays inside the run
    // of `n` elements belonging to `i`. Where the run and the counter's
    // limit are named separately, that is exactly the fact the caller
    // has to establish.
    let mut banded = |n: HirId, shift: Shift, counter: HirId| {
        let c = confined.iter().find(|c| c.counter == counter)?;
        if c.limit != n {
            obligations.push(Obligation::SameCount(n, c.limit));
        }
        Some(Index::Banded(Band {
            stride: Stride::Elements(n),
            shift,
        }))
    };
    if let Index::Scaled { n, shift } = l {
        if let Some(idx) = banded(n, shift, rv) {
            return idx;
        }
    }
    if let Index::Scaled { n, shift } = r {
        if let Some(idx) = banded(n, shift, lv) {
            return idx;
        }
    }
    // Anything else added to a band moves the access out of it.
    Index::Unknown
}

/// Resolve an address to the base it was computed from and what its
/// index says about the counter.
fn address_of(
    func: &HirFunction,
    defs: &Defs<'_>,
    induction: HirId,
    confined: &[Confined],
    addr: HirId,
    obligations: &mut Vec<Obligation>,
) -> Option<(HirId, Index)> {
    let (base, index) = match defs.by_value.get(&addr).copied()? {
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            // More than one index is a shape into an aggregate rather
            // than a walk along a buffer, and the band argument does
            // not carry over.
            if indices.len() != 1 {
                return None;
            }
            (*ptr, indices[0])
        }
        // Adding to a pointer names an element the same way indexing
        // does, and the vector intrinsics are written that way, so a
        // buffer walked with `p + i` has to read as the same shape as
        // one walked with `p[i]`.
        HirInstruction::Binary {
            op: BinaryOp::Add,
            left,
            right,
            ..
        } => match (is_pointer(func, *left), is_pointer(func, *right)) {
            (true, false) => (*left, *right),
            (false, true) => (*right, *left),
            _ => return None,
        },
        _ => return None,
    };
    // A base the loop computes may name different storage on different
    // iterations, so it proves nothing.
    if defs.varies(base) {
        return None;
    }
    let idx = classify_index(func, defs, induction, confined, index, obligations, 8);
    Some((base, idx))
}

/// Whether a value names storage rather than a number.
fn is_pointer(func: &HirFunction, value: HirId) -> bool {
    func.values.get(&value).is_some_and(|v| {
        matches!(
            v.ty,
            crate::hir::HirType::Ptr(_) | crate::hir::HirType::Ref { .. }
        )
    })
}

/// Whether an instruction reaches memory only through the addresses
/// this analysis already accounted for.
///
/// Arithmetic, address computation, casts and lane shuffling all
/// compute from values. Anything that can reach a pointer this
/// function did not hand it is not on the list: a closure call, an
/// atomic, an effect. An instruction absent from the list is treated
/// as reaching anywhere.
fn touches_no_memory(inst: &HirInstruction) -> bool {
    matches!(
        inst,
        HirInstruction::Binary { .. }
            | HirInstruction::Unary { .. }
            | HirInstruction::Cast { .. }
            | HirInstruction::GetElementPtr { .. }
            | HirInstruction::Select { .. }
            | HirInstruction::ExtractValue { .. }
            | HirInstruction::InsertValue { .. }
            | HirInstruction::VectorSplat { .. }
            | HirInstruction::VectorExtractLane { .. }
            | HirInstruction::VectorInsertLane { .. }
            | HirInstruction::VectorHorizontalReduce { .. }
            | HirInstruction::VectorUnaryOp { .. }
            | HirInstruction::VectorMinMax { .. }
            | HirInstruction::VectorDot { .. }
    )
}

/// Whether a block ends in a way that keeps control inside the body.
///
/// A terminator can call, and a call whose effects are not visible
/// here could touch anything.
fn terminator_is_plain(term: &HirTerminator) -> bool {
    matches!(
        term,
        HirTerminator::Branch { .. }
            | HirTerminator::CondBranch { .. }
            | HirTerminator::Switch { .. }
            | HirTerminator::Return { .. }
            | HirTerminator::Unreachable
    )
}

/// One memory access the body makes.
struct Access {
    base: HirId,
    index: Index,
    is_write: bool,
    /// How many elements it spans from the address. One for a scalar,
    /// the lane count for a vector.
    lanes: i64,
}

/// How many elements an access covers.
///
/// A band says which address an iteration starts at; it does not by
/// itself say the access stops before the next iteration's. For a
/// scalar the two are the same question, but a vector store at `a[i]`
/// covers `i` through `i + lanes`, and whether that reaches into the
/// next iteration's territory depends on how far the counter moves.
fn lanes_of(func: &HirFunction, inst: &HirInstruction) -> i64 {
    let ty = match inst {
        HirInstruction::VectorLoad { ty, .. } => Some(ty.clone()),
        HirInstruction::VectorStore { value, .. } => func.values.get(value).map(|v| v.ty.clone()),
        _ => None,
    };
    match ty {
        Some(crate::hir::HirType::Vector(_, lanes)) => lanes as i64,
        // A vector access whose width is not on the instruction is not
        // one this can bound, so it is given a width nothing satisfies.
        Some(_) => i64::MAX,
        None => 1,
    }
}

/// Decide one loop.
fn examine(
    func: &HirFunction,
    forest: &LoopForest,
    lp: &NaturalLoop,
    stats: &mut ParallelStats,
) -> Option<ParallelLoop> {
    let header = func.blocks.get(&lp.header)?;

    // Exactly one value may change between iterations, and it has to be
    // the counter. Anything else is carried state.
    if header.phis.len() != 1 {
        stats.carried_dependency += 1;
        return None;
    }
    let induction = header.phis[0].result;

    let defs = Defs::build(func, &lp.body);
    let mut obligations = Vec::new();
    let mut accesses = Vec::new();

    for b in &lp.body {
        let Some(blk) = func.blocks.get(b) else {
            continue;
        };
        if !terminator_is_plain(&blk.terminator) {
            stats.opaque_body += 1;
            return None;
        }
        let confined = confined_counters(func, forest, lp, &defs, *b);
        // A phi anywhere else in the body is control flow joining, not
        // state carried around the back edge, so it is allowed. A phi in
        // the header was already ruled on above.
        for inst in &blk.instructions {
            let (ptr, is_write) = match inst {
                HirInstruction::Load { ptr, .. } | HirInstruction::VectorLoad { ptr, .. } => {
                    (*ptr, false)
                }
                HirInstruction::Store { ptr, .. } | HirInstruction::VectorStore { ptr, .. } => {
                    (*ptr, true)
                }
                HirInstruction::Call { callee, .. } => {
                    let pure = matches!(
                        callee,
                        crate::hir::HirCallable::Intrinsic(i) if is_pure_intrinsic(i)
                    );
                    if !pure {
                        stats.opaque_body += 1;
                        return None;
                    }
                    continue;
                }
                // Everything else is judged by whether it is known to
                // touch no memory of its own. Naming the harmless
                // shapes rather than the harmful ones means an
                // instruction added later is refused until someone
                // decides it belongs, which for a pass that decides
                // what may run concurrently is the direction to be
                // wrong in.
                other => {
                    if !touches_no_memory(other) {
                        stats.opaque_body += 1;
                        return None;
                    }
                    continue;
                }
            };
            match address_of(func, &defs, induction, &confined, ptr, &mut obligations) {
                Some((base, index)) => accesses.push(Access {
                    base,
                    index,
                    is_write,
                    lanes: lanes_of(func, inst),
                }),
                None => {
                    stats.opaque_body += 1;
                    return None;
                }
            }
        }
    }

    // A loop that writes nothing has nothing to race over, but also
    // nothing to gain, so it is not reported.
    if !accesses.iter().any(|a| a.is_write) {
        return None;
    }

    // An access wider than one element has to finish before the next
    // iteration's address begins, or the two overlap however cleanly
    // the bands are drawn. A vectorized loop moves its counter by the
    // lane count, which is exactly what makes that hold; a loop that
    // does not is refused rather than reasoned about.
    let widest = accesses.iter().map(|a| a.lanes).max().unwrap_or(1);
    if widest > 1 {
        let step = step_of(func, &defs, lp, induction).unwrap_or(0);
        let all_unit_stride = accesses
            .iter()
            .all(|a| matches!(a.index.band().map(|b| b.stride), Some(Stride::One)));
        if step < widest || !all_unit_stride {
            stats.opaque_body += 1;
            return None;
        }
    }

    if !bands_are_disjoint(func, &accesses, &mut obligations) {
        stats.opaque_body += 1;
        return None;
    }

    let mut reads: Vec<HirId> = accesses
        .iter()
        .filter(|a| !a.is_write)
        .map(|a| a.base)
        .collect();
    let mut writes: Vec<HirId> = accesses
        .iter()
        .filter(|a| a.is_write)
        .map(|a| a.base)
        .collect();
    reads.sort();
    reads.dedup();
    writes.sort();
    writes.dedup();
    obligations.sort_by_key(|o| match o {
        Obligation::Disjoint(a, b) => (0u8, *a, *b),
        Obligation::SameCount(a, b) => (1u8, *a, *b),
    });
    obligations.dedup();

    if obligations.is_empty() {
        stats.independent += 1;
    } else {
        stats.conditional += 1;
    }
    Some(ParallelLoop {
        header: lp.header,
        induction,
        reads,
        writes,
        obligations,
    })
}

/// Whether no iteration's writes can reach another's accesses, adding
/// what the caller must establish where the answer depends on storage
/// this function cannot tell apart.
fn bands_are_disjoint(
    func: &HirFunction,
    accesses: &[Access],
    obligations: &mut Vec<Obligation>,
) -> bool {
    let mut bases: Vec<HirId> = accesses.iter().map(|a| a.base).collect();
    bases.sort();
    bases.dedup();

    // Within one base pointer the answer is settled here, since every
    // access through it reaches the same storage by construction.
    let mut band_of: HashMap<HirId, Band> = HashMap::new();
    for base in &bases {
        let through: Vec<&Access> = accesses.iter().filter(|a| a.base == *base).collect();
        if !through.iter().any(|a| a.is_write) {
            // Read-only storage: two iterations reading it cannot
            // disagree. The band is recorded only to compare against a
            // write's later.
            if let Some(b) = through.first().and_then(|a| a.index.band()) {
                if through
                    .iter()
                    .all(|a| a.index.band().is_some_and(|x| x.same(b)))
                {
                    band_of.insert(*base, b);
                }
            }
            continue;
        }
        // A write has to be inside a band. `i * n` on its own is not
        // one: an `n` of zero would put every iteration at the same
        // address.
        let Some(band) = through
            .iter()
            .find(|a| a.is_write)
            .and_then(|a| a.index.band())
        else {
            return false;
        };
        // And everything else through the same storage has to be in the
        // band belonging to the same iteration, or one iteration reads
        // or overwrites what another wrote. Nothing the caller could
        // promise changes this.
        if !through
            .iter()
            .all(|a| a.index.band().is_some_and(|x| x.same(band)))
        {
            return false;
        }
        band_of.insert(*base, band);
    }

    // Between base pointers the answer is not this function's to give.
    // Two ids may turn out to be one buffer. Where both sides walk the
    // same band the question does not arise, because each iteration
    // owns band `i` of whatever storage that turns out to be;
    // otherwise the caller has to say they are apart.
    for (a, b) in pairs(&bases) {
        let touched_by_write = accesses
            .iter()
            .any(|x| x.is_write && (x.base == a || x.base == b));
        if !touched_by_write {
            continue;
        }
        if let (Some(x), Some(y)) = (band_of.get(&a), band_of.get(&b)) {
            if x.same(*y) {
                continue;
            }
        }
        // A parameter declared exclusive is one nothing else names, and
        // every caller was held to that before this ran. Two parameters
        // are then two buffers, which is the fact the obligation was
        // asking for, so it does not need asking again.
        if held_apart(func, a, b) {
            continue;
        }
        obligations.push(Obligation::Disjoint(a, b));
    }
    true
}

/// Whether the language already says these two name different storage.
///
/// Only parameters carry the claim. Anything else reached this loop
/// through the body, where two ids saying different things about the
/// same buffer is exactly what cannot be ruled out from here.
fn held_apart(func: &HirFunction, a: HirId, b: HirId) -> bool {
    use crate::exclusive_args::{is_parameter, parameter_is_exclusive};
    if !is_parameter(func, a) || !is_parameter(func, b) {
        return false;
    }
    // One of them being exclusive is enough: nothing else may name it,
    // and the other is something else.
    parameter_is_exclusive(func, a) || parameter_is_exclusive(func, b)
}

/// Every unordered pair of distinct bases.
fn pairs(bases: &[HirId]) -> Vec<(HirId, HirId)> {
    let mut out = Vec::new();
    for (i, a) in bases.iter().enumerate() {
        for b in &bases[i + 1..] {
            out.push((*a, *b));
        }
    }
    out
}

/// Every loop in one function whose iterations are independent.
pub fn analyze(func: &HirFunction) -> (Vec<ParallelLoop>, ParallelStats) {
    let mut stats = ParallelStats::default();
    if func.is_external || func.blocks.is_empty() {
        return (Vec::new(), stats);
    }
    let dt = DominatorTree::new(func);
    let forest = LoopForest::detect(func, &dt);
    let mut found = Vec::new();
    for lp in forest.loops() {
        if let Some(p) = examine(func, &forest, lp, &mut stats) {
            found.push(p);
        }
    }
    (found, stats)
}

/// The same over a whole module, for reporting.
pub fn analyze_module(module: &HirModule) -> ParallelStats {
    let mut total = ParallelStats::default();
    for func in module.functions.values() {
        let (_, s) = analyze(func);
        total.independent += s.independent;
        total.conditional += s.conditional;
        total.carried_dependency += s.carried_dependency;
        total.opaque_body += s.opaque_body;
    }
    total
}
