//! Floating-point multiply-add contraction.
//!
//! Recognises the pattern
//!
//! ```text
//!     t = FMul(a, b)        ; HirType::F64
//!     r = FAdd(t, c)        ; HirType::F64    (or FAdd(c, t))
//! ```
//!
//! and rewrites it to a single `Intrinsic::Fma(a, b, c)` instruction
//! producing `r`, removing the now-dead `FMul`. Cranelift lowers
//! `Fma` to a hardware fused-multiply-add (one IEEE-754 round
//! instead of two), and the BC interpreter uses `f64::mul_add` so
//! the two execution paths agree.
//!
//! ## Precision delta
//!
//! Contracting `a*b + c` into `fma(a, b, c)` changes IEEE-754
//! rounding: instead of rounding the product and then the sum, the
//! FMA rounds exactly once on the final result. This is standard
//! fast-math contraction (GCC/Clang/LLVM apply it under
//! `-ffp-contract=fast` and it is implicitly allowed by C/C++ via
//! `STDC FP_CONTRACT`). The numeric result is typically *more*
//! accurate than the two-rounding form, but it is not bit-identical
//! — code that depends on bit-exact reproduction of an intermediate
//! product will observe the difference.
//!
//! ## Safety constraints
//!
//! * Both operations must live in the **same basic block** — cross-
//!   block contraction would require dominance analysis (the fmul
//!   must dominate the fadd and lie on every path to it) and is
//!   left as future work.
//! * The `FMul` result must have **exactly one use** module-wide
//!   (this `FAdd`). Otherwise we would either duplicate the multiply
//!   (defeating the optimisation) or leave a dangling reference.
//!   The use count is collected by scanning every block's
//!   instructions, phis, and terminator in the function.
//! * Only `HirType::F64` is contracted today — the same pattern
//!   would apply to `F32`, but the current SSA pipeline normalises
//!   most float math to `F64` so the F32 path is rare enough to
//!   skip until we see it.
//! * Only `FAdd` is contracted. `FSub(fmul a b, c)` could become
//!   `Fma(a, b, -c)` via an `FNeg`, but that needs an extra
//!   instruction and an extra def-site; deferred.

use crate::hir::{
    BinaryOp, HirCallable, HirFunction, HirId, HirInstruction, HirModule, HirTerminator, HirType,
    Intrinsic,
};
use std::collections::HashMap;

/// Per-call counters surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FmaStats {
    /// Number of `fadd(fmul, _)` patterns rewritten to `Intrinsic::Fma`.
    pub contracted: usize,
}

/// Public entry. Runs FMA contraction on every function in `module`.
pub fn run_module(module: &mut HirModule) -> FmaStats {
    let mut stats = FmaStats::default();
    // Temporary regression-bisection gate (Phase 1 mandelbrot drift).
    // Set ZYNTAX_DISABLE_FMA=1 to disable contraction when bisecting
    // codegen-level IEEE-754 rounding divergences (e.g. mandelbrot
    // checksum drift vs rayzor's HashLink reference).
    if std::env::var("ZYNTAX_DISABLE_FMA").is_ok() {
        return stats;
    }
    for func in module.functions.values_mut() {
        stats.contracted += run_function(func);
    }
    stats
}

/// Run the pass on a single function. Returns the number of
/// `Fma` instructions emitted.
pub fn run_function(func: &mut HirFunction) -> usize {
    // Step 1: build a function-wide use count for every HirId that
    // appears as an operand. We need this to enforce the
    // "exactly one use" guard on the fmul result.
    let use_counts = collect_use_counts(func);

    // Step 2: walk each block in isolation, look for the pattern.
    let block_ids: Vec<HirId> = func.blocks.keys().copied().collect();
    let mut contracted = 0;

    for block_id in block_ids {
        let block = match func.blocks.get_mut(&block_id) {
            Some(b) => b,
            None => continue,
        };
        contracted += contract_block(&mut block.instructions, &use_counts);
    }

    contracted
}

/// Walk `instructions` looking for FAdd(t, c) / FAdd(c, t) where
/// `t` is the result of an earlier FMul in this same block with a
/// single function-wide use. Rewrites the FAdd to `Fma(a, b, c)`
/// and drops the FMul.
fn contract_block(
    instructions: &mut Vec<HirInstruction>,
    use_counts: &HashMap<HirId, usize>,
) -> usize {
    // `defs_in_block` maps each F64 FMul result defined earlier in
    // *this* block to (a, b, position_in_instructions). Limited to
    // the current block by construction — we only insert as we walk.
    let mut fmul_defs: HashMap<HirId, (HirId, HirId, usize)> = HashMap::new();
    // Indices of fmul instructions to remove after we're done
    // (so we don't perturb the iteration).
    let mut to_remove: Vec<usize> = Vec::new();
    let mut contracted = 0;

    for idx in 0..instructions.len() {
        // Record FMul F64 defs so a later FAdd can pick them up.
        if let HirInstruction::Binary {
            op: BinaryOp::FMul,
            ty: HirType::F64,
            left,
            right,
            result,
        } = &instructions[idx]
        {
            fmul_defs.insert(*result, (*left, *right, idx));
            continue;
        }

        // Look for FAdd F64 candidates.
        let HirInstruction::Binary {
            op: BinaryOp::FAdd,
            ty: HirType::F64,
            left: add_left,
            right: add_right,
            result: add_result,
        } = &instructions[idx]
        else {
            continue;
        };

        let add_result = *add_result;

        // Try lhs as the fmul side first, then rhs. Take the first
        // candidate whose fmul result has exactly one function-wide
        // use (this FAdd consuming it).
        let (mul_id, addend) = if let Some(&(_a, _b, _i)) = fmul_defs.get(add_left) {
            if use_counts.get(add_left).copied().unwrap_or(0) == 1 {
                (*add_left, *add_right)
            } else if let Some(&(_a, _b, _i)) = fmul_defs.get(add_right) {
                if use_counts.get(add_right).copied().unwrap_or(0) == 1 {
                    (*add_right, *add_left)
                } else {
                    continue;
                }
            } else {
                continue;
            }
        } else if let Some(&(_a, _b, _i)) = fmul_defs.get(add_right) {
            if use_counts.get(add_right).copied().unwrap_or(0) == 1 {
                (*add_right, *add_left)
            } else {
                continue;
            }
        } else {
            continue;
        };

        let (a, b, mul_idx) = fmul_defs.remove(&mul_id).expect("present by guard");

        // Replace the FAdd in place with a Call to Intrinsic::Fma.
        instructions[idx] = HirInstruction::Call {
            result: Some(add_result),
            callee: HirCallable::Intrinsic(Intrinsic::Fma),
            args: vec![a, b, addend],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        };
        to_remove.push(mul_idx);
        contracted += 1;
    }

    // Remove the consumed FMul instructions from highest to lowest
    // index so each removal doesn't shift later removals.
    to_remove.sort_unstable();
    to_remove.dedup();
    for &idx in to_remove.iter().rev() {
        instructions.remove(idx);
    }

    contracted
}

/// Build a function-wide map `HirId -> number of times it appears
/// as an operand` across all block instructions, phi incomings, and
/// terminators. This is the use-count we check before contracting
/// (single use → safe to fuse; otherwise the FMul still has live
/// downstream consumers).
fn collect_use_counts(func: &HirFunction) -> HashMap<HirId, usize> {
    let mut counts: HashMap<HirId, usize> = HashMap::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            for operand in inst.operands() {
                *counts.entry(operand).or_insert(0) += 1;
            }
        }
        for phi in &block.phis {
            for (value, _block) in &phi.incoming {
                *counts.entry(*value).or_insert(0) += 1;
            }
        }
        terminator_operands(&block.terminator, &mut counts);
    }
    counts
}

/// Add every HirId operand of `term` to `counts`. Mirrors the
/// instruction-level `operands()` helper, kept local to the pass
/// so we don't have to extend `HirTerminator` with a public method
/// just for this counter.
fn terminator_operands(term: &HirTerminator, counts: &mut HashMap<HirId, usize>) {
    match term {
        HirTerminator::Return { values } => {
            for v in values {
                *counts.entry(*v).or_insert(0) += 1;
            }
        }
        HirTerminator::Branch { .. } => {}
        HirTerminator::CondBranch { condition, .. } => {
            *counts.entry(*condition).or_insert(0) += 1;
        }
        HirTerminator::Switch { value, .. } => {
            *counts.entry(*value).or_insert(0) += 1;
        }
        HirTerminator::Unreachable => {}
        HirTerminator::Invoke { args, .. } => {
            for a in args {
                *counts.entry(*a).or_insert(0) += 1;
            }
        }
        HirTerminator::PatternMatch { value, .. } => {
            *counts.entry(*value).or_insert(0) += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirInstruction, HirTerminator};
    use indexmap::IndexMap;
    use zyntax_typed_ast::InternedString;

    fn empty_func(entry: HirId) -> HirFunction {
        HirFunction {
            id: HirId::new(),
            name: InternedString::new_global("test"),
            signature: HirFunctionSignature {
                params: Vec::new(),
                returns: vec![HirType::F64],
                type_params: Vec::new(),
                const_params: Vec::new(),
                lifetime_params: Vec::new(),
                is_variadic: false,
                is_async: false,
                effects: Vec::new(),
                is_pure: true,
            },
            entry_block: entry,
            blocks: IndexMap::new(),
            locals: IndexMap::new(),
            values: IndexMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: crate::hir::CallingConvention::C,
            attributes: Default::default(),
            link_name: None,
        }
    }

    #[test]
    fn contracts_fmul_then_fadd() {
        let entry = HirId::new();
        let mut func = empty_func(entry);

        // Values: a, b, c are inputs; t = fmul; r = fadd
        let a = HirId::new();
        let b = HirId::new();
        let c = HirId::new();
        let t = HirId::new();
        let r = HirId::new();

        let block = HirBlock {
            id: entry,
            label: None,
            phis: Vec::new(),
            instructions: vec![
                HirInstruction::Binary {
                    op: BinaryOp::FMul,
                    result: t,
                    ty: HirType::F64,
                    left: a,
                    right: b,
                },
                HirInstruction::Binary {
                    op: BinaryOp::FAdd,
                    result: r,
                    ty: HirType::F64,
                    left: t,
                    right: c,
                },
            ],
            terminator: HirTerminator::Return { values: vec![r] },
            dominance_frontier: Default::default(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        };
        func.blocks.insert(entry, block);

        let contracted = run_function(&mut func);
        assert_eq!(contracted, 1);

        let block = &func.blocks[&entry];
        assert_eq!(block.instructions.len(), 1);
        match &block.instructions[0] {
            HirInstruction::Call {
                callee: HirCallable::Intrinsic(Intrinsic::Fma),
                args,
                result: Some(res),
                ..
            } => {
                assert_eq!(*res, r);
                assert_eq!(args, &vec![a, b, c]);
            }
            other => panic!("expected Fma call, got {:?}", other),
        }
    }

    #[test]
    fn skips_when_fmul_has_other_use() {
        let entry = HirId::new();
        let mut func = empty_func(entry);

        let a = HirId::new();
        let b = HirId::new();
        let c = HirId::new();
        let t = HirId::new();
        let r1 = HirId::new();
        let r2 = HirId::new(); // second consumer of `t`

        let block = HirBlock {
            id: entry,
            label: None,
            phis: Vec::new(),
            instructions: vec![
                HirInstruction::Binary {
                    op: BinaryOp::FMul,
                    result: t,
                    ty: HirType::F64,
                    left: a,
                    right: b,
                },
                HirInstruction::Binary {
                    op: BinaryOp::FAdd,
                    result: r1,
                    ty: HirType::F64,
                    left: t,
                    right: c,
                },
                // second use of t — fmul result also flows here
                HirInstruction::Binary {
                    op: BinaryOp::FAdd,
                    result: r2,
                    ty: HirType::F64,
                    left: t,
                    right: c,
                },
            ],
            terminator: HirTerminator::Return {
                values: vec![r1, r2],
            },
            dominance_frontier: Default::default(),
            predecessors: Vec::new(),
            successors: Vec::new(),
        };
        func.blocks.insert(entry, block);

        let contracted = run_function(&mut func);
        assert_eq!(contracted, 0);
        // FMul + two FAdds = three instructions unchanged.
        assert_eq!(func.blocks[&entry].instructions.len(), 3);
    }
}
