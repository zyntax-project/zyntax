//! HIR loop vectorization for *reduction* loops — `sum += op(a[i], b[i])`.
//!
//! Complementary to `loop_vectorize` (which handles store-to-array
//! SAXPY shapes): this module recognises the canonical
//!
//! ```text
//! for i in 0..n { sum += op(a[i], b[i]) }
//! ```
//!
//! pattern, where `sum` is a header phi that accumulates per
//! iteration. After vectorization the loop runs at stride 4 over a
//! lane-wide accumulator (`sum_vec : Vector(elem_ty, 4)`), the
//! main-loop body is `sum_vec = sum_vec + Vector(op(a_vec, b_vec))`,
//! and on exit we horizontally reduce `sum_vec` to a scalar via the
//! existing `HirInstruction::VectorHorizontalReduce` op. A scalar
//! epilogue handles the tail `[vec_n, n)` iterations, seeded with
//! the post-reduce scalar from the vector loop.
//!
//! ## Pattern we recognise
//!
//! Two phis in the header:
//!
//! ```text
//! i   = phi [0 from preheader, i_next from body]
//! sum = phi [0 from preheader, sum_next from body]
//! ```
//!
//! Header bound: `i < n`, branch to body / exit on it.
//!
//! Body, exactly:
//!
//! ```text
//! addr_a    = GEP(ptr_a, [i])
//! va        = Load addr_a
//! addr_b    = GEP(ptr_b, [i])
//! vb        = Load addr_b
//! vc        = Binary(elementwise_op, va, vb)    // Mul/Add/Sub
//! sum_next  = Binary(reduce_op, sum, vc)         // typically Add
//! i_next    = i + 1
//! Branch header
//! ```
//!
//! `reduce_op` must be one of `Add` / `FAdd` (the only ops where a
//! lane-parallel reduction is semantically equivalent to the scalar
//! one — `Sub` / `FSub` would order-dependent, `Mul` is fine
//! mathematically but loses precision on float).
//!
//! `elementwise_op` must be one of the SIMD-safe set
//! (`Add`/`Sub`/`Mul`/`FAdd`/`FSub`/`FMul`).
//!
//! ## What we don't do (yet)
//!
//! * Reductions other than `Add` / `FAdd`.
//! * Reductions where the body doesn't compute a single
//!   elementwise op (e.g., dot product with a Cast in between).
//! * Multi-induction or multi-accumulator loops.
//! * Non-unit stride.
//!
//! Any loop that fails any precondition stays untouched and surfaces
//! a `skipped_*` counter so callers can see why.

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, CastOp, HirBlock, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirPhi,
    HirTerminator, HirType, HirValue, HirValueKind,
};
use std::collections::HashSet;

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReductionStats {
    pub vectorized: usize,
    pub loops_visited: usize,
    pub skipped_shape: usize,
    pub skipped_op_unsupported: usize,
}

/// Vectorize every matching reduction loop in `func`.
pub fn run(func: &mut HirFunction) -> ReductionStats {
    let dt = DominatorTree::new(func);
    let lf = LoopForest::detect(func, &dt);
    if lf.loops().is_empty() {
        return ReductionStats::default();
    }

    let mut stats = ReductionStats::default();
    for lp in lf.loops().to_vec() {
        stats.loops_visited += 1;
        match recognise(func, &lp) {
            Recognition::Match(pat) => {
                rewrite(func, &pat);
                stats.vectorized += 1;
            }
            Recognition::SkipShape => stats.skipped_shape += 1,
            Recognition::SkipOp => stats.skipped_op_unsupported += 1,
        }
    }
    stats
}

/// Module-level entry.
pub fn run_module(module: &mut HirModule) -> ReductionStats {
    let mut total = ReductionStats::default();
    for func in module.functions.values_mut() {
        let s = run(func);
        total.vectorized += s.vectorized;
        total.loops_visited += s.loops_visited;
        total.skipped_shape += s.skipped_shape;
        total.skipped_op_unsupported += s.skipped_op_unsupported;
    }
    total
}

// ─── Pattern recognition ──────────────────────────────────────────

enum Recognition {
    Match(ReductionPattern),
    SkipShape,
    SkipOp,
}

#[allow(dead_code)]
struct ReductionPattern {
    header: HirId,
    body: HirId,
    exit: HirId,
    preheader: HirId,
    i_phi: HirId,
    i_next: HirId,
    sum_phi: HirId,
    sum_next: HirId,
    n: HirId,
    reduce_op: BinaryOp,
    elementwise_op: BinaryOp,
    /// Accumulator element type (the reduce type — i32 for the widening dot).
    elem_ty: HirType,
    ptr_a: HirId,
    ptr_b: HirId,
    /// When set, the body is `acc += widen(a[i]) * widen(b[i])` with `i8`
    /// operands widened to `i32` — vectorized to a stride-16 `VectorDot`
    /// (i8x16 · i8x16 → i32x4) instead of a stride-4 same-width mul + reduce.
    widening_dot: bool,
    /// Load element type (i8/u8 for the dot; == elem_ty otherwise).
    load_elem_ty: HirType,
    /// `b` operand is unsigned (u8) → VectorDot's `rhs_unsigned`.
    rhs_unsigned: bool,
    /// Original scalar body instructions — copied into the scalar
    /// epilogue at rewrite time.
    body_insts: Vec<HirInstruction>,
}

fn recognise(func: &HirFunction, lp: &NaturalLoop) -> Recognition {
    if lp.body.len() != 2 {
        return Recognition::SkipShape;
    }
    let header_id = lp.header;
    let body_id = match lp.body.iter().find(|&&b| b != header_id) {
        Some(&b) => b,
        None => return Recognition::SkipShape,
    };
    let header = match func.blocks.get(&header_id) {
        Some(b) => b,
        None => return Recognition::SkipShape,
    };
    let body = match func.blocks.get(&body_id) {
        Some(b) => b,
        None => return Recognition::SkipShape,
    };

    // Header CondBranch(i < n, body, exit).
    let (cond, true_t, false_t) = match &header.terminator {
        HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => (*condition, *true_target, *false_target),
        _ => return Recognition::SkipShape,
    };
    let exit_id = if true_t == body_id {
        false_t
    } else {
        return Recognition::SkipShape;
    };

    // Header has TWO phis: induction i + accumulator sum.
    if header.phis.len() != 2 {
        return Recognition::SkipShape;
    }

    let preheader_id = {
        let outside: Vec<HirId> = header
            .predecessors
            .iter()
            .copied()
            .filter(|p| !lp.body.contains(p))
            .collect();
        if outside.len() != 1 {
            return Recognition::SkipShape;
        }
        outside[0]
    };

    // Walk both phis; classify each as either induction (init 0,
    // operand of `i + 1`) or accumulator (init 0, operand of the
    // sum-update binary).
    let (i_phi, i_next, sum_phi, sum_next) =
        match classify_phis(func, header, body, body_id, preheader_id) {
            Some(t) => t,
            None => return Recognition::SkipShape,
        };

    // Cond = i < n
    let n = match find_inst_by_result_in(header, cond) {
        Some(HirInstruction::Binary {
            op: BinaryOp::Lt,
            left,
            right,
            ..
        }) if *left == i_phi => *right,
        _ => return Recognition::SkipShape,
    };

    // Verify i_next = i + 1
    if !is_increment_by_one(func, body, i_next, i_phi) {
        return Recognition::SkipShape;
    }

    // sum_next must be `Binary(reduce_op, sum, vc)` where vc is the
    // result of a Binary(elementwise_op, va, vb) of loads through
    // GEPs of i.
    let (reduce_op, vc, elem_ty) = match find_inst_by_result_in(body, sum_next) {
        Some(HirInstruction::Binary {
            op,
            left,
            right,
            ty,
            ..
        }) if *left == sum_phi => (*op, *right, ty.clone()),
        _ => return Recognition::SkipShape,
    };
    if !is_reduce_op(reduce_op) {
        return Recognition::SkipOp;
    }
    if !is_vector_elem_type(&elem_ty) {
        return Recognition::SkipOp;
    }

    let (elementwise_op, va, vb) = match find_inst_by_result_in(body, vc) {
        Some(HirInstruction::Binary {
            op, left, right, ..
        }) => (*op, *left, *right),
        _ => return Recognition::SkipShape,
    };
    if !is_elementwise_op(elementwise_op) {
        return Recognition::SkipOp;
    }

    // Widening dot: `acc:i32 += (i32)a[i] * (i32)b[i]` where `a[i]`/`b[i]` are
    // i8 loads sign/zero-extended before the multiply. `va`/`vb` are the
    // widening casts, not direct loads — so the same-width resolver below
    // rejects them (this is the "dot product with a Cast in between" the pass
    // used to skip). Vectorize to a stride-16 `VectorDot` (i8x16·i8x16→i32x4).
    if matches!(elementwise_op, BinaryOp::Mul)
        && matches!(reduce_op, BinaryOp::Add)
        && matches!(elem_ty, HirType::I32)
    {
        if let (Some((pa, la)), Some((pb, lb))) = (
            resolve_widening_load(body, va, i_phi),
            resolve_widening_load(body, vb, i_phi),
        ) {
            let invariant = [pa, pb].iter().all(|&p| {
                !defined_in_block(func, p, body_id) && !defined_in_block(func, p, header_id)
            });
            if invariant {
                return Recognition::Match(ReductionPattern {
                    header: header_id,
                    body: body_id,
                    exit: exit_id,
                    preheader: preheader_id,
                    i_phi,
                    i_next,
                    sum_phi,
                    sum_next,
                    n,
                    reduce_op,
                    elementwise_op,
                    elem_ty, // i32 accumulator
                    ptr_a: pa,
                    ptr_b: pb,
                    widening_dot: true,
                    load_elem_ty: la, // i8 / u8
                    rhs_unsigned: matches!(lb, HirType::U8),
                    body_insts: body.instructions.clone(),
                });
            }
        }
    }

    let ptr_a = match resolve_load_through_gep(body, va, i_phi) {
        Some(p) => p,
        None => return Recognition::SkipShape,
    };
    let ptr_b = match resolve_load_through_gep(body, vb, i_phi) {
        Some(p) => p,
        None => return Recognition::SkipShape,
    };

    // Pointers must be loop-invariant.
    for p in [ptr_a, ptr_b] {
        if defined_in_block(func, p, body_id) || defined_in_block(func, p, header_id) {
            return Recognition::SkipShape;
        }
    }

    Recognition::Match(ReductionPattern {
        header: header_id,
        body: body_id,
        exit: exit_id,
        preheader: preheader_id,
        i_phi,
        i_next,
        sum_phi,
        sum_next,
        n,
        reduce_op,
        elementwise_op,
        load_elem_ty: elem_ty.clone(),
        elem_ty,
        ptr_a,
        ptr_b,
        widening_dot: false,
        rhs_unsigned: false,
        body_insts: body.instructions.clone(),
    })
}

/// Resolve `value = Cast(Load(GEP(ptr, [i])))` where the cast widens an i8/u8
/// load to i32 — the shape of one operand of a widening dot. Returns the base
/// pointer and the load's element type.
fn resolve_widening_load(block: &HirBlock, value: HirId, i: HirId) -> Option<(HirId, HirType)> {
    let cast = find_inst_by_result_in(block, value)?;
    let cast_operand = match cast {
        HirInstruction::Cast {
            operand, ty, op, ..
        } if matches!(op, CastOp::SExt | CastOp::ZExt) && matches!(ty, HirType::I32) => *operand,
        _ => return None,
    };
    let load = find_inst_by_result_in(block, cast_operand)?;
    let (gep_addr, load_ty) = match load {
        HirInstruction::Load { ptr, ty, .. } if matches!(ty, HirType::I8 | HirType::U8) => {
            (*ptr, ty.clone())
        }
        _ => return None,
    };
    let gep = find_inst_by_result_in(block, gep_addr)?;
    match gep {
        HirInstruction::GetElementPtr { ptr, indices, .. }
            if indices.len() == 1 && indices[0] == i =>
        {
            Some((*ptr, load_ty))
        }
        _ => None,
    }
}

fn classify_phis(
    func: &HirFunction,
    header: &HirBlock,
    body: &HirBlock,
    body_id: HirId,
    preheader_id: HirId,
) -> Option<(HirId, HirId, HirId, HirId)> {
    // For each phi, identify init-from-preheader value and
    // next-from-body value. Then identify which is the induction
    // (next == result + 1) and which is the accumulator
    // (next == result + something).
    let mut phi_info: Vec<(HirId, HirId, HirId)> = Vec::new(); // (result, init, next)
    for phi in &header.phis {
        let mut init = None;
        let mut next = None;
        for &(val, blk) in &phi.incoming {
            if blk == preheader_id {
                init = Some(val);
            } else if blk == body_id {
                next = Some(val);
            }
        }
        let init = init?;
        let next = next?;
        // Both inits must be 0 (numeric or float).
        if !matches!(
            func.values.get(&init).map(|v| &v.kind),
            Some(HirValueKind::Constant(HirConstant::I32(0)))
                | Some(HirValueKind::Constant(HirConstant::I64(0)))
                | Some(HirValueKind::Constant(HirConstant::U32(0)))
                | Some(HirValueKind::Constant(HirConstant::U64(0)))
                | Some(HirValueKind::Constant(HirConstant::F32(0.0)))
                | Some(HirValueKind::Constant(HirConstant::F64(0.0)))
        ) {
            return None;
        }
        phi_info.push((phi.result, init, next));
    }
    if phi_info.len() != 2 {
        return None;
    }

    // The induction is whichever phi has next == result + 1.
    for combo in [(0usize, 1usize), (1, 0)] {
        let (ind_idx, acc_idx) = combo;
        let (i_res, _, i_next) = phi_info[ind_idx];
        let (sum_res, _, sum_next) = phi_info[acc_idx];
        if is_increment_by_one(func, body, i_next, i_res) {
            return Some((i_res, i_next, sum_res, sum_next));
        }
    }
    None
}

fn is_increment_by_one(
    func: &HirFunction,
    body: &HirBlock,
    candidate_next: HirId,
    candidate_phi: HirId,
) -> bool {
    let inst = match find_inst_by_result_in(body, candidate_next) {
        Some(i) => i,
        None => return false,
    };
    matches!(
        inst,
        HirInstruction::Binary { op: BinaryOp::Add, left, right, .. }
            if *left == candidate_phi
                && matches!(func.values.get(right).map(|v| &v.kind),
                    Some(HirValueKind::Constant(HirConstant::I32(1)))
                        | Some(HirValueKind::Constant(HirConstant::I64(1))))
    )
}

fn is_reduce_op(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::Add | BinaryOp::FAdd)
}

fn is_elementwise_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::FAdd
            | BinaryOp::FSub
            | BinaryOp::FMul
    )
}

fn is_vector_elem_type(ty: &HirType) -> bool {
    matches!(
        ty,
        HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::F32
            | HirType::F64
    )
}

fn find_inst_by_result_in(block: &HirBlock, target: HirId) -> Option<&HirInstruction> {
    block.instructions.iter().find(|inst| match inst {
        HirInstruction::Binary { result, .. } => *result == target,
        HirInstruction::Unary { result, .. } => *result == target,
        HirInstruction::Cast { result, .. } => *result == target,
        HirInstruction::GetElementPtr { result, .. } => *result == target,
        HirInstruction::ExtractValue { result, .. } => *result == target,
        HirInstruction::Load { result, .. } => *result == target,
        _ => false,
    })
}

fn resolve_load_through_gep(block: &HirBlock, value: HirId, i: HirId) -> Option<HirId> {
    let load = find_inst_by_result_in(block, value)?;
    let gep_addr = match load {
        HirInstruction::Load { ptr, .. } => *ptr,
        _ => return None,
    };
    let gep = find_inst_by_result_in(block, gep_addr)?;
    match gep {
        HirInstruction::GetElementPtr { ptr, indices, .. }
            if indices.len() == 1 && indices[0] == i =>
        {
            Some(*ptr)
        }
        _ => None,
    }
}

fn defined_in_block(func: &HirFunction, value: HirId, block_id: HirId) -> bool {
    let block = match func.blocks.get(&block_id) {
        Some(b) => b,
        None => return false,
    };
    if block.phis.iter().any(|p| p.result == value) {
        return true;
    }
    find_inst_by_result_in(block, value).is_some()
}

// ─── Rewriting ────────────────────────────────────────────────────

/// Materialise the vector reduction loop + scalar epilogue. After:
///
/// ```text
/// preheader: vec_n = n & ~3; vec_init = vector_zero(elem_ty, 4)
///   → header (vec_check)
/// header: i_vec = phi [0, i_vec_next]
///         sum_vec = phi [vec_init, sum_vec_next]
///         cond = i_vec < vec_n
///         cond_branch -> body, post_vec
/// body:   va_vec = VectorLoad ptr_a + i_vec
///         vb_vec = VectorLoad ptr_b + i_vec
///         vc_vec = Binary(elementwise, va_vec, vb_vec)
///         sum_vec_next = Binary(reduce, sum_vec, vc_vec)
///         i_vec_next = i_vec + 4
///         Branch header
/// post_vec: scalar_seed = VectorHorizontalReduce(sum_vec, reduce)
///         → scalar_check
/// scalar_check: i_s = phi [i_vec, i_s_next]
///               sum_s = phi [scalar_seed, sum_s_next]
///               cond2 = i_s < n
///               cond_branch -> scalar_body, exit
/// scalar_body: (original scalar body verbatim, branching back
///               to scalar_check)
/// ```
fn rewrite(func: &mut HirFunction, pat: &ReductionPattern) {
    let n_ty = func
        .values
        .get(&pat.n)
        .map(|v| v.ty.clone())
        .unwrap_or(HirType::I64);
    // Stride + vector types. The widening dot consumes 16 i8 lanes per
    // iteration into an i32x4 accumulator via a single `VectorDot`; the
    // same-width path consumes 4 elements (elementwise op + reduce).
    let stride: i64 = if pat.widening_dot { 16 } else { 4 };
    let mask_const = match n_ty {
        HirType::I32 | HirType::U32 => HirConstant::I32(!((stride as i32) - 1)),
        _ => HirConstant::I64(!(stride - 1)),
    };
    let mask_id = create_value(func, n_ty.clone(), HirValueKind::Constant(mask_const));
    let vec_n_id = create_value(func, n_ty.clone(), HirValueKind::Instruction);

    // Accumulator element/vector type (`i32`/i32x4 for the dot). Synthesise the
    // vector-zero init via `VectorSplat` of a scalar zero (no `HirConstant::Vector`).
    let elem_ty = pat.elem_ty.clone();
    let vec_ty = HirType::Vector(Box::new(elem_ty.clone()), 4);
    // Load element/vector type: i8/i8x16 for the dot, else == accumulator.
    let load_elem_ty = pat.load_elem_ty.clone();
    let load_vec_ty = if pat.widening_dot {
        HirType::Vector(Box::new(load_elem_ty.clone()), 16)
    } else {
        vec_ty.clone()
    };
    let zero_scalar = create_value(
        func,
        elem_ty.clone(),
        HirValueKind::Constant(match elem_ty {
            HirType::F32 => HirConstant::F32(0.0),
            HirType::F64 => HirConstant::F64(0.0),
            HirType::I32 | HirType::U32 => HirConstant::I32(0),
            _ => HirConstant::I64(0),
        }),
    );
    let vec_init = create_value(func, vec_ty.clone(), HirValueKind::Instruction);

    // Preheader instructions: compute vec_n, splat zero into vec_init.
    {
        let ph = func.blocks.get_mut(&pat.preheader).unwrap();
        ph.instructions.push(HirInstruction::Binary {
            op: BinaryOp::And,
            result: vec_n_id,
            ty: n_ty.clone(),
            left: pat.n,
            right: mask_id,
        });
        ph.instructions.push(HirInstruction::VectorSplat {
            result: vec_init,
            ty: vec_ty.clone(),
            scalar: zero_scalar,
        });
    }

    // New sum_vec phi result + i increment-by-stride + new vec body ops.
    let sum_vec_next = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
    let va_vec = create_value(func, load_vec_ty.clone(), HirValueKind::Instruction);
    let vb_vec = create_value(func, load_vec_ty.clone(), HirValueKind::Instruction);
    let vc_vec = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
    let addr_a_v = create_value(
        func,
        HirType::Ptr(Box::new(load_elem_ty.clone())),
        HirValueKind::Instruction,
    );
    let addr_b_v = create_value(
        func,
        HirType::Ptr(Box::new(load_elem_ty.clone())),
        HirValueKind::Instruction,
    );
    let stride_const = create_value(
        func,
        n_ty.clone(),
        HirValueKind::Constant(match n_ty {
            HirType::I32 | HirType::U32 => HirConstant::I32(stride as i32),
            _ => HirConstant::I64(stride),
        }),
    );
    let new_i_next = create_value(func, n_ty.clone(), HirValueKind::Instruction);
    let new_cmp = create_value(func, HirType::Bool, HirValueKind::Instruction);
    let load_align = elem_size_bytes(&load_elem_ty) as u32;

    // Replace header's phis: keep i_phi but rewrite its body
    // incoming to new_i_next; replace sum_phi with a vector-typed phi.
    {
        let header_blk = func.blocks.get_mut(&pat.header).unwrap();
        for phi in &mut header_blk.phis {
            if phi.result == pat.i_phi {
                for (val, blk) in phi.incoming.iter_mut() {
                    if *blk == pat.body && *val == pat.i_next {
                        *val = new_i_next;
                    }
                }
            } else if phi.result == pat.sum_phi {
                phi.ty = vec_ty.clone();
                for (val, blk) in phi.incoming.iter_mut() {
                    if *blk == pat.preheader {
                        *val = vec_init;
                    } else if *blk == pat.body && *val == pat.sum_next {
                        *val = sum_vec_next;
                    }
                }
            }
        }
        // Also rewrite the sum_phi's HirValue to be vector-typed.
        if let Some(v) = func.values.get_mut(&pat.sum_phi) {
            v.ty = vec_ty.clone();
        }

        // Replace the Lt instruction with a fresh one against vec_n.
        let old_cmp = match &header_blk.terminator {
            HirTerminator::CondBranch { condition, .. } => *condition,
            _ => unreachable!(),
        };
        for inst in header_blk.instructions.iter_mut() {
            if let HirInstruction::Binary {
                op: BinaryOp::Lt,
                result,
                right,
                ..
            } = inst
            {
                if *result == old_cmp {
                    *result = new_cmp;
                    *right = vec_n_id;
                }
            }
        }
    }

    // Create post_vec + scalar_check + scalar_body block ids.
    let post_vec = HirId::new();
    let scalar_check = HirId::new();
    let scalar_body = HirId::new();
    let scalar_seed = create_value(func, elem_ty.clone(), HirValueKind::Instruction);

    {
        let header_blk = func.blocks.get_mut(&pat.header).unwrap();
        header_blk.terminator = HirTerminator::CondBranch {
            condition: new_cmp,
            true_target: pat.body,
            false_target: post_vec,
        };
    }

    // Rewrite vector body block.
    {
        let body_blk = func.blocks.get_mut(&pat.body).unwrap();
        body_blk.instructions.clear();
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_a_v,
            ty: HirType::Ptr(Box::new(load_elem_ty.clone())),
            ptr: pat.ptr_a,
            indices: vec![pat.i_phi],
        });
        body_blk.instructions.push(HirInstruction::VectorLoad {
            result: va_vec,
            ty: load_vec_ty.clone(),
            ptr: addr_a_v,
            align: load_align,
        });
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_b_v,
            ty: HirType::Ptr(Box::new(load_elem_ty.clone())),
            ptr: pat.ptr_b,
            indices: vec![pat.i_phi],
        });
        body_blk.instructions.push(HirInstruction::VectorLoad {
            result: vb_vec,
            ty: load_vec_ty.clone(),
            ptr: addr_b_v,
            align: load_align,
        });
        if pat.widening_dot {
            // sum_vec_next = VectorDot(sum_phi, a_i8x16, b_i8x16) — the fused
            // widening dot-accumulate that lowers to SDOT.
            body_blk.instructions.push(HirInstruction::VectorDot {
                result: sum_vec_next,
                acc: pat.sum_phi,
                a: va_vec,
                b: vb_vec,
                rhs_i7: false,
                rhs_unsigned: pat.rhs_unsigned,
            });
        } else {
            body_blk.instructions.push(HirInstruction::Binary {
                op: pat.elementwise_op,
                result: vc_vec,
                ty: vec_ty.clone(),
                left: va_vec,
                right: vb_vec,
            });
            body_blk.instructions.push(HirInstruction::Binary {
                op: pat.reduce_op,
                result: sum_vec_next,
                ty: vec_ty.clone(),
                left: pat.sum_phi,
                right: vc_vec,
            });
        }
        body_blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: new_i_next,
            ty: n_ty.clone(),
            left: pat.i_phi,
            right: stride_const,
        });
        body_blk.terminator = HirTerminator::Branch { target: pat.header };
    }

    // Post-vec: horizontal-reduce sum_vec into a scalar; branch to
    // scalar_check.
    let mut post_vec_blk = HirBlock::new(post_vec);
    post_vec_blk.predecessors = vec![pat.header];
    post_vec_blk.successors = vec![scalar_check];
    post_vec_blk
        .instructions
        .push(HirInstruction::VectorHorizontalReduce {
            result: scalar_seed,
            ty: elem_ty.clone(),
            vector: pat.sum_phi,
            op: pat.reduce_op,
        });
    post_vec_blk.terminator = HirTerminator::Branch {
        target: scalar_check,
    };
    func.blocks.insert(post_vec, post_vec_blk);

    // Scalar epilogue: a check block with two phis (i_s, sum_s),
    // branching to scalar_body / exit on `i_s < n`.
    let scalar_i = create_value(func, n_ty.clone(), HirValueKind::Instruction);
    let scalar_sum = create_value(func, elem_ty.clone(), HirValueKind::Instruction);
    let scalar_cond = create_value(func, HirType::Bool, HirValueKind::Instruction);

    let mut scalar_check_blk = HirBlock::new(scalar_check);
    scalar_check_blk.predecessors = vec![post_vec, scalar_body];
    scalar_check_blk.successors = vec![scalar_body, pat.exit];
    scalar_check_blk.phis.push(HirPhi {
        result: scalar_i,
        ty: n_ty.clone(),
        // Body-incoming is patched after scalar_body's i_next2 is minted.
        incoming: vec![(pat.i_phi, post_vec)],
    });
    scalar_check_blk.phis.push(HirPhi {
        result: scalar_sum,
        ty: elem_ty.clone(),
        incoming: vec![(scalar_seed, post_vec)],
    });
    scalar_check_blk.instructions.push(HirInstruction::Binary {
        op: BinaryOp::Lt,
        result: scalar_cond,
        ty: HirType::Bool,
        left: scalar_i,
        right: pat.n,
    });
    scalar_check_blk.terminator = HirTerminator::CondBranch {
        condition: scalar_cond,
        true_target: scalar_body,
        false_target: pat.exit,
    };
    func.blocks.insert(scalar_check, scalar_check_blk);

    // Clone the original scalar body verbatim, substituting i_phi →
    // scalar_i and sum_phi → scalar_sum.
    let mut subs: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();
    subs.insert(pat.i_phi, scalar_i);
    subs.insert(pat.sum_phi, scalar_sum);

    let mut scalar_body_blk = HirBlock::new(scalar_body);
    scalar_body_blk.predecessors = vec![scalar_check];
    scalar_body_blk.successors = vec![scalar_check];

    let mut scalar_i_next: Option<HirId> = None;
    let mut scalar_sum_next: Option<HirId> = None;
    for inst in &pat.body_insts {
        let res = match inst {
            HirInstruction::Binary { result, .. } => Some(*result),
            HirInstruction::Load { result, .. } => Some(*result),
            HirInstruction::GetElementPtr { result, .. } => Some(*result),
            _ => None,
        };
        if let Some(r) = res {
            let new_res = create_value(
                func,
                func.values
                    .get(&r)
                    .map(|v| v.ty.clone())
                    .unwrap_or(HirType::I64),
                HirValueKind::Instruction,
            );
            subs.insert(r, new_res);
            if r == pat.i_next {
                scalar_i_next = Some(new_res);
            }
            if r == pat.sum_next {
                scalar_sum_next = Some(new_res);
            }
        }
        let mut cloned = inst.clone();
        substitute_operands(&mut cloned, &subs);
        scalar_body_blk.instructions.push(cloned);
    }
    scalar_body_blk.terminator = HirTerminator::Branch {
        target: scalar_check,
    };
    func.blocks.insert(scalar_body, scalar_body_blk);

    // Patch scalar_check's phis with body-incoming edges.
    {
        let sc = func.blocks.get_mut(&scalar_check).unwrap();
        for phi in &mut sc.phis {
            if phi.result == scalar_i {
                if let Some(n) = scalar_i_next {
                    phi.incoming.push((n, scalar_body));
                }
            } else if phi.result == scalar_sum {
                if let Some(n) = scalar_sum_next {
                    phi.incoming.push((n, scalar_body));
                }
            }
        }
    }

    // Wire exit's predecessor list: was header; now scalar_check.
    if let Some(exit_blk) = func.blocks.get_mut(&pat.exit) {
        exit_blk.predecessors.retain(|p| *p != pat.header);
        exit_blk.predecessors.push(scalar_check);
    }

    // Redirect post-loop uses of the reduction result. `sum_phi` is now the
    // VECTOR accumulator — live only inside the loop and the post-vec
    // reduce. Anything downstream of the loop (the exit's `return sum` /
    // store, later blocks) must read the fully-reduced scalar `scalar_sum`
    // instead. Without this the vectorized loop computes the right sum but
    // nothing consumes it — the pass's fixtures all returned `()`, so this
    // was previously untested and silently wrong for any *used* reduction.
    let loop_region: HashSet<HirId> = [pat.header, pat.body, post_vec].into_iter().collect();
    let mut redirect: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();
    redirect.insert(pat.sum_phi, scalar_sum);
    for (bid, blk) in func.blocks.iter_mut() {
        if loop_region.contains(bid) {
            continue;
        }
        for inst in &mut blk.instructions {
            inst.replace_uses(&redirect);
        }
        blk.terminator.replace_uses(&redirect);
        for phi in &mut blk.phis {
            for (val, _pred) in &mut phi.incoming {
                if *val == pat.sum_phi {
                    *val = scalar_sum;
                }
            }
        }
    }
}

fn elem_size_bytes(ty: &HirType) -> u64 {
    match ty {
        HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 => 8,
        _ => 4,
    }
}

fn create_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
    let id = HirId::new();
    func.values.insert(
        id,
        HirValue {
            id,
            ty,
            kind,
            uses: HashSet::new(),
            span: None,
        },
    );
    id
}

fn substitute_operands(inst: &mut HirInstruction, subs: &indexmap::IndexMap<HirId, HirId>) {
    let mut sub = |id: &mut HirId| {
        if let Some(&new) = subs.get(id) {
            *id = new;
        }
    };
    match inst {
        HirInstruction::Binary {
            result,
            left,
            right,
            ..
        } => {
            sub(result);
            sub(left);
            sub(right);
        }
        HirInstruction::Unary {
            result, operand, ..
        } => {
            sub(result);
            sub(operand);
        }
        HirInstruction::Cast {
            result, operand, ..
        } => {
            sub(result);
            sub(operand);
        }
        HirInstruction::GetElementPtr {
            result,
            ptr,
            indices,
            ..
        } => {
            sub(result);
            sub(ptr);
            for i in indices.iter_mut() {
                sub(i);
            }
        }
        HirInstruction::Load { result, ptr, .. } => {
            sub(result);
            sub(ptr);
        }
        HirInstruction::Store { value, ptr, .. } => {
            sub(value);
            sub(ptr);
        }
        _ => {}
    }
}

// ─── tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::HirFunctionSignature;
    use zyntax_typed_ast::InternedString;

    fn sig() -> HirFunctionSignature {
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
        }
    }

    fn add_param(f: &mut HirFunction, ty: HirType, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Parameter(idx),
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }
    fn add_const(f: &mut HirFunction, ty: HirType, c: HirConstant) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Constant(c),
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }
    fn add_inst(f: &mut HirFunction, ty: HirType) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    /// Build `for i in 0..n { sum += op(a[i], b[i]) }` with the given
    /// elem_ty + elementwise_op + reduce_op.
    fn build_reduction(elem_ty: HirType, elementwise: BinaryOp, reduce: BinaryOp) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global("red"), sig());
        let entry = HirId::new();
        let header = HirId::new();
        let body = HirId::new();
        let exit = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        for id in [entry, header, body, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        f.blocks.get_mut(&entry).unwrap().successors = vec![header];
        f.blocks.get_mut(&header).unwrap().predecessors = vec![entry, body];
        f.blocks.get_mut(&header).unwrap().successors = vec![body, exit];
        f.blocks.get_mut(&body).unwrap().predecessors = vec![header];
        f.blocks.get_mut(&body).unwrap().successors = vec![header];
        f.blocks.get_mut(&exit).unwrap().predecessors = vec![header];

        let n_ty = HirType::I64;
        let ptr_ty = HirType::Ptr(Box::new(elem_ty.clone()));
        let ptr_a = add_param(&mut f, ptr_ty.clone(), 0);
        let ptr_b = add_param(&mut f, ptr_ty.clone(), 1);
        let n = add_param(&mut f, n_ty.clone(), 2);
        let zero_int = add_const(&mut f, n_ty.clone(), HirConstant::I64(0));
        let one = add_const(&mut f, n_ty.clone(), HirConstant::I64(1));
        let zero_elem = match elem_ty {
            HirType::F32 => add_const(&mut f, elem_ty.clone(), HirConstant::F32(0.0)),
            HirType::F64 => add_const(&mut f, elem_ty.clone(), HirConstant::F64(0.0)),
            _ => add_const(&mut f, elem_ty.clone(), HirConstant::I64(0)),
        };

        let i_phi = add_inst(&mut f, n_ty.clone());
        let sum_phi = add_inst(&mut f, elem_ty.clone());
        let i_next = add_inst(&mut f, n_ty.clone());
        let sum_next = add_inst(&mut f, elem_ty.clone());
        f.blocks.get_mut(&header).unwrap().phis.push(HirPhi {
            result: i_phi,
            ty: n_ty.clone(),
            incoming: vec![(zero_int, entry), (i_next, body)],
        });
        f.blocks.get_mut(&header).unwrap().phis.push(HirPhi {
            result: sum_phi,
            ty: elem_ty.clone(),
            incoming: vec![(zero_elem, entry), (sum_next, body)],
        });

        let cond = add_inst(&mut f, HirType::Bool);
        f.blocks
            .get_mut(&header)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Lt,
                result: cond,
                ty: n_ty.clone(),
                left: i_phi,
                right: n,
            });
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
        f.blocks.get_mut(&header).unwrap().terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: body,
            false_target: exit,
        };
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![] };

        let addr_a = add_inst(&mut f, ptr_ty.clone());
        let va = add_inst(&mut f, elem_ty.clone());
        let addr_b = add_inst(&mut f, ptr_ty.clone());
        let vb = add_inst(&mut f, elem_ty.clone());
        let vc = add_inst(&mut f, elem_ty.clone());

        let body_blk = f.blocks.get_mut(&body).unwrap();
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_a,
            ty: ptr_ty.clone(),
            ptr: ptr_a,
            indices: vec![i_phi],
        });
        body_blk.instructions.push(HirInstruction::Load {
            result: va,
            ty: elem_ty.clone(),
            ptr: addr_a,
            align: 4,
            volatile: false,
        });
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_b,
            ty: ptr_ty.clone(),
            ptr: ptr_b,
            indices: vec![i_phi],
        });
        body_blk.instructions.push(HirInstruction::Load {
            result: vb,
            ty: elem_ty.clone(),
            ptr: addr_b,
            align: 4,
            volatile: false,
        });
        body_blk.instructions.push(HirInstruction::Binary {
            op: elementwise,
            result: vc,
            ty: elem_ty.clone(),
            left: va,
            right: vb,
        });
        body_blk.instructions.push(HirInstruction::Binary {
            op: reduce,
            result: sum_next,
            ty: elem_ty.clone(),
            left: sum_phi,
            right: vc,
        });
        body_blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: i_next,
            ty: n_ty.clone(),
            left: i_phi,
            right: one,
        });
        body_blk.terminator = HirTerminator::Branch { target: header };
        f
    }

    #[test]
    fn vectorizes_f32_dot_product() {
        let mut f = build_reduction(HirType::F32, BinaryOp::FMul, BinaryOp::FAdd);
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 1, "{stats:?}");
        // VectorHorizontalReduce should be in some block (post_vec).
        let has_reduce = f.blocks.values().any(|b| {
            b.instructions
                .iter()
                .any(|i| matches!(i, HirInstruction::VectorHorizontalReduce { .. }))
        });
        assert!(has_reduce);
    }

    /// `for i in 0..n { sum:i32 += (i32)a[i] * (i32)b[i] }` with `a`,`b : *i8`.
    fn build_widening_dot() -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global("wdot"), sig());
        let entry = HirId::new();
        let header = HirId::new();
        let body = HirId::new();
        let exit = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        for id in [entry, header, body, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }
        f.blocks.get_mut(&entry).unwrap().successors = vec![header];
        f.blocks.get_mut(&header).unwrap().predecessors = vec![entry, body];
        f.blocks.get_mut(&header).unwrap().successors = vec![body, exit];
        f.blocks.get_mut(&body).unwrap().predecessors = vec![header];
        f.blocks.get_mut(&body).unwrap().successors = vec![header];
        f.blocks.get_mut(&exit).unwrap().predecessors = vec![header];

        let n_ty = HirType::I64;
        let i8_ty = HirType::I8;
        let i32_ty = HirType::I32;
        let ptr_ty = HirType::Ptr(Box::new(i8_ty.clone()));
        let ptr_a = add_param(&mut f, ptr_ty.clone(), 0);
        let ptr_b = add_param(&mut f, ptr_ty.clone(), 1);
        let n = add_param(&mut f, n_ty.clone(), 2);
        let zero_int = add_const(&mut f, n_ty.clone(), HirConstant::I64(0));
        let one = add_const(&mut f, n_ty.clone(), HirConstant::I64(1));
        let zero_i32 = add_const(&mut f, i32_ty.clone(), HirConstant::I32(0));

        let i_phi = add_inst(&mut f, n_ty.clone());
        let sum_phi = add_inst(&mut f, i32_ty.clone());
        let i_next = add_inst(&mut f, n_ty.clone());
        let sum_next = add_inst(&mut f, i32_ty.clone());
        f.blocks.get_mut(&header).unwrap().phis.push(HirPhi {
            result: i_phi,
            ty: n_ty.clone(),
            incoming: vec![(zero_int, entry), (i_next, body)],
        });
        f.blocks.get_mut(&header).unwrap().phis.push(HirPhi {
            result: sum_phi,
            ty: i32_ty.clone(),
            incoming: vec![(zero_i32, entry), (sum_next, body)],
        });

        let cond = add_inst(&mut f, HirType::Bool);
        f.blocks
            .get_mut(&header)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Lt,
                result: cond,
                ty: n_ty.clone(),
                left: i_phi,
                right: n,
            });
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };
        f.blocks.get_mut(&header).unwrap().terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: body,
            false_target: exit,
        };
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return { values: vec![] };

        let addr_a = add_inst(&mut f, ptr_ty.clone());
        let va = add_inst(&mut f, i8_ty.clone());
        let va32 = add_inst(&mut f, i32_ty.clone());
        let addr_b = add_inst(&mut f, ptr_ty.clone());
        let vb = add_inst(&mut f, i8_ty.clone());
        let vb32 = add_inst(&mut f, i32_ty.clone());
        let vc = add_inst(&mut f, i32_ty.clone());

        let body_blk = f.blocks.get_mut(&body).unwrap();
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_a,
            ty: ptr_ty.clone(),
            ptr: ptr_a,
            indices: vec![i_phi],
        });
        body_blk.instructions.push(HirInstruction::Load {
            result: va,
            ty: i8_ty.clone(),
            ptr: addr_a,
            align: 1,
            volatile: false,
        });
        body_blk.instructions.push(HirInstruction::Cast {
            result: va32,
            ty: i32_ty.clone(),
            op: CastOp::SExt,
            operand: va,
        });
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_b,
            ty: ptr_ty.clone(),
            ptr: ptr_b,
            indices: vec![i_phi],
        });
        body_blk.instructions.push(HirInstruction::Load {
            result: vb,
            ty: i8_ty.clone(),
            ptr: addr_b,
            align: 1,
            volatile: false,
        });
        body_blk.instructions.push(HirInstruction::Cast {
            result: vb32,
            ty: i32_ty.clone(),
            op: CastOp::SExt,
            operand: vb,
        });
        body_blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Mul,
            result: vc,
            ty: i32_ty.clone(),
            left: va32,
            right: vb32,
        });
        body_blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: sum_next,
            ty: i32_ty.clone(),
            left: sum_phi,
            right: vc,
        });
        body_blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: i_next,
            ty: n_ty.clone(),
            left: i_phi,
            right: one,
        });
        body_blk.terminator = HirTerminator::Branch { target: header };

        // Populate the signature params (matching the Parameter value ids) so
        // the interpreter can bind call args, and return the accumulated sum
        // (a real reduction consumes its result) to exercise the transform's
        // post-loop use redirection.
        let mk_param = |id, ty| crate::hir::HirParam {
            id,
            name: InternedString::new_global("p"),
            ty,
            attributes: crate::hir::ParamAttributes::default(),
            ownership: Default::default(),
        };
        f.signature.params = vec![
            mk_param(ptr_a, ptr_ty.clone()),
            mk_param(ptr_b, ptr_ty.clone()),
            mk_param(n, n_ty.clone()),
        ];
        f.signature.returns = vec![i32_ty.clone()];
        f.blocks.get_mut(&exit).unwrap().terminator = HirTerminator::Return {
            values: vec![sum_phi],
        };
        f
    }

    #[test]
    fn vectorizes_widening_i8_dot_to_vectordot() {
        let mut f = build_widening_dot();
        let stats = run(&mut f);
        assert_eq!(
            stats.vectorized, 1,
            "widening i8 dot should vectorize: {stats:?}"
        );
        // The vectorized body must use the fused `VectorDot` (the SDOT
        // primitive), not a same-width vector Mul + reduce.
        let has_dot = f.blocks.values().any(|b| {
            b.instructions
                .iter()
                .any(|i| matches!(i, HirInstruction::VectorDot { .. }))
        });
        assert!(has_dot, "expected a VectorDot in the vectorized body");
    }

    #[test]
    fn widening_dot_executes_correctly() {
        let mut f = build_widening_dot();
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 1, "{stats:?}");
        // Execute the transformed function through the interpreter with
        // a=[1..=20], b=[1;20], n=20 → Σ(1..=20) = 210. The stride-16 vector
        // loop covers [0,16) via one VectorDot (=136); the scalar epilogue
        // [16,20) adds 17+18+19+20 = 74. Proves stride-16 + i8x16 load +
        // VectorDot + i32 horizontal reduce + scalar epilogue + the post-loop
        // exit redirect all compose to the right value.
        let mut module = HirModule::new(InternedString::new_global("m"));
        module.functions.insert(f.id, f);
        let a: Vec<i8> = (1..=20).collect();
        let b: Vec<i8> = vec![1i8; 20];
        let mut interp = crate::hir_interp::HirInterpreter::new();
        let got = interp
            .call(
                &module,
                "wdot",
                vec![
                    crate::value::ZyntaxValue::Pointer(a.as_ptr() as *mut u8),
                    crate::value::ZyntaxValue::Pointer(b.as_ptr() as *mut u8),
                    crate::value::ZyntaxValue::Int(20),
                ],
            )
            .expect("interp call");
        assert_eq!(
            crate::hir_interp::value_to_i64(&got),
            Some(210),
            "got {got:?}"
        );
    }

    #[test]
    fn skips_unsupported_reduce_op() {
        // FSub isn't a sound reduction (lane ordering matters), so
        // this should be skipped.
        let mut f = build_reduction(HirType::F32, BinaryOp::FMul, BinaryOp::FSub);
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 0);
        assert_eq!(stats.skipped_op_unsupported, 1);
    }

    #[test]
    fn skips_when_no_loop_present() {
        let mut f = HirFunction::new(InternedString::new_global("nop"), sig());
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![] };
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 0);
        assert_eq!(stats.loops_visited, 0);
    }
}
