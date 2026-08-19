//! Which counted loops have independent iterations.
//!
//! Spreading a loop across cores is safe exactly when no iteration can
//! observe another's writes. That is a stronger claim than "it
//! vectorizes", but it rests on the same evidence: every address the
//! body touches is computed from the induction variable, and every base
//! it is computed from is the same on every iteration. Iteration `i`
//! then touches only what belongs to `i`, whichever core runs it and in
//! whatever order.
//!
//! What disqualifies a loop, and why:
//!
//! * A value carried between iterations. A header phi other than the
//!   induction variable is an accumulator, and an accumulator read by
//!   two cores at once is a race. Reductions need a different shape
//!   (per-core partials combined at the end), so they are refused here
//!   rather than silently split.
//! * An address not derived from the induction variable, since nothing
//!   then bounds which iteration reaches it.
//! * A base pointer the loop itself computes, which may differ per
//!   iteration and so says nothing about disjointness.
//! * A call. Its effects are not visible here, so it could touch
//!   anything. Pure arithmetic intrinsics are the exception, having no
//!   effects to worry about.
//!
//! This decides safety only. Whether spreading a given loop is
//! worthwhile is a separate question of trip count and work per
//! iteration, and belongs where the dispatch is chosen.

use std::collections::HashSet;

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{HirFunction, HirId, HirInstruction, HirModule, Intrinsic};

/// One loop whose iterations do not interfere.
#[derive(Debug, Clone)]
pub struct ParallelLoop {
    /// Header block, which is also the back-edge target.
    pub header: HirId,
    /// The induction variable's phi result.
    pub induction: HirId,
    /// Base pointers the body reads through.
    pub reads: Vec<HirId>,
    /// Base pointers the body writes through.
    pub writes: Vec<HirId>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ParallelStats {
    /// Loops whose iterations were shown independent.
    pub independent: usize,
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

/// Whether anything inside the loop defines `value`.
fn defined_in_loop(func: &HirFunction, blocks: &HashSet<HirId>, value: HirId) -> bool {
    blocks.iter().any(|b| {
        func.blocks.get(b).is_some_and(|blk| {
            blk.phis.iter().any(|p| p.result == value)
                || blk.instructions.iter().any(|i| defines(i, value))
        })
    })
}

/// Resolve an address to the base it was computed from, requiring every
/// step to be indexed by the induction variable.
fn base_of_address(
    func: &HirFunction,
    body: &HashSet<HirId>,
    addr: HirId,
    induction: HirId,
) -> Option<HirId> {
    for b in body {
        let Some(blk) = func.blocks.get(b) else {
            continue;
        };
        for inst in &blk.instructions {
            if let HirInstruction::GetElementPtr {
                result,
                ptr,
                indices,
                ..
            } = inst
            {
                if *result != addr {
                    continue;
                }
                // Every index has to be the induction variable itself.
                // An index the loop computes some other way could repeat
                // across iterations, and then two of them collide.
                if indices.len() != 1 || indices[0] != induction {
                    return None;
                }
                // A base the loop computes may name different storage on
                // different iterations, so it proves nothing.
                if defined_in_loop(func, body, *ptr) {
                    return None;
                }
                return Some(*ptr);
            }
        }
    }
    None
}

/// Decide one loop.
fn examine(
    func: &HirFunction,
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

    let mut reads = Vec::new();
    let mut writes = Vec::new();

    for b in &lp.body {
        let Some(blk) = func.blocks.get(b) else {
            continue;
        };
        // A phi anywhere else in the body is control flow joining, not
        // state carried around the back edge, so it is allowed. A phi in
        // the header was already ruled on above.
        for inst in &blk.instructions {
            match inst {
                HirInstruction::Load { ptr, .. } | HirInstruction::VectorLoad { ptr, .. } => {
                    match base_of_address(func, &lp.body, *ptr, induction) {
                        Some(base) => reads.push(base),
                        None => {
                            stats.opaque_body += 1;
                            return None;
                        }
                    }
                }
                HirInstruction::Store { ptr, .. } | HirInstruction::VectorStore { ptr, .. } => {
                    match base_of_address(func, &lp.body, *ptr, induction) {
                        Some(base) => writes.push(base),
                        None => {
                            stats.opaque_body += 1;
                            return None;
                        }
                    }
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
                }
                // Arithmetic, address computation and casts touch no
                // memory of their own.
                _ => {}
            }
        }
    }

    // A loop that writes nothing has nothing to race over, but also
    // nothing to gain, so it is not reported.
    if writes.is_empty() {
        return None;
    }

    reads.sort();
    reads.dedup();
    writes.sort();
    writes.dedup();

    stats.independent += 1;
    Some(ParallelLoop {
        header: lp.header,
        induction,
        reads,
        writes,
    })
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
        if let Some(p) = examine(func, lp, &mut stats) {
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
        total.carried_dependency += s.carried_dependency;
        total.opaque_body += s.opaque_body;
    }
    total
}
