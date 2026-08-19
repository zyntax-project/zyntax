//! HIR loop vectorization for SAXPY-shape counted loops.
//!
//! Recognises the canonical
//!
//! ```text
//! for i in 0..n { c[i] = op(a[i], b[i]) }
//! ```
//!
//! pattern that's been lowered to HIR and rewrites it to a stride-4
//! `VectorLoad` / vector-binop / `VectorStore` main loop followed by
//! a scalar epilogue for the tail.
//!
//! Same vector shape the existing `emit_elementwise_simd_loop` in
//! `ssa.rs` produces for `@kernel elementwise` directive-tagged
//! source — this pass just generalises so the user doesn't have to
//! tag the loop. The BC interpreter and wasm-tier-1 JIT both already
//! understand `HirInstruction::VectorLoad / VectorStore` since they
//! were wired for the kernel path.
//!
//! ## Pattern we recognise
//!
//! Header (single phi `i` with init 0, increment by 1; loop bound is
//! `i < n`):
//!
//! ```text
//! i = phi [0 from preheader, i_next from body]
//! cond = (i < n)
//! cond_branch cond -> body, exit
//! ```
//!
//! Body (the SAXPY shape — exactly these ops, in any order, with the
//! only producer-consumer chain shown):
//!
//! ```text
//! addr_a = GEP(ptr_a, [i])
//! va     = Load addr_a
//! addr_b = GEP(ptr_b, [i])
//! vb     = Load addr_b
//! vc     = Binary(elementwise_op, va, vb)
//! addr_c = GEP(ptr_c, [i])
//! Store vc -> addr_c
//! i_next = i + 1
//! Branch header
//! ```
//!
//! `ptr_a`, `ptr_b`, `ptr_c` must be loop-invariant. The elementwise
//! op must be one of the SIMD-safe set (`Add`, `Sub`, `Mul`, `FAdd`,
//! `FSub`, `FMul`).
//!
//! ## What we don't yet do
//!
//! * Reductions (`sum += a[i] * b[i]`) — would need
//!   `VectorHorizontalReduce`. Adding this is a focused follow-up.
//! * Non-unit stride (`a[i*2]`) — requires gather/scatter HIR ops
//!   we don't yet have.
//! * Aliasing analysis — we conservatively skip when we can't prove
//!   the three pointers are distinct (right now we assume they are,
//!   matching the `@kernel` contract; a real alias check is a
//!   future pass).
//! * Branches inside the body. Single-block bodies only.
//! * Reverse loops, `step` loops, multi-induction-variable loops.
//!
//! ## What we don't transform
//!
//! Any loop that fails any precondition stays exactly as it was. The
//! stats counters surface the reason so callers (the optimization
//! pipeline harness, tests) can see why a particular loop wasn't
//! vectorized.

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, HirBlock, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirPhi,
    HirTerminator, HirType, HirValue, HirValueKind,
};
use indexmap::IndexMap;
use std::collections::HashSet;

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VectorizeStats {
    /// Loops we transformed.
    pub vectorized: usize,
    /// Loops we examined.
    pub loops_visited: usize,
    /// Skipped because the loop body shape didn't match SAXPY.
    pub skipped_shape: usize,
    /// Skipped because we couldn't recover the induction variable.
    pub skipped_no_induction: usize,
    /// Skipped because the elementwise op isn't in our safe list.
    pub skipped_op_unsupported: usize,
}

/// Vectorize every matching loop in `func`.
pub fn run(func: &mut HirFunction) -> VectorizeStats {
    let dt = DominatorTree::new(func);
    let lf = LoopForest::detect(func, &dt);
    if lf.loops().is_empty() {
        return VectorizeStats::default();
    }

    let mut stats = VectorizeStats::default();

    // Innermost-first — the existing `LoopForest::loops()` order.
    // Each loop is independent so the order isn't load-bearing, but
    // it matches LICM and keeps the trace order deterministic.
    for lp in lf.loops().to_vec() {
        stats.loops_visited += 1;
        match recognise_saxpy(func, &lp) {
            Recognition::Match(pat) => {
                rewrite_to_vector_loop(func, &pat);
                stats.vectorized += 1;
            }
            Recognition::SkipShape => stats.skipped_shape += 1,
            Recognition::SkipNoInduction => stats.skipped_no_induction += 1,
            Recognition::SkipOp => stats.skipped_op_unsupported += 1,
        }
    }

    stats
}

/// Module-level entry.
pub fn run_module(module: &mut HirModule) -> VectorizeStats {
    let mut total = VectorizeStats::default();
    for func in module.functions.values_mut() {
        let s = run(func);
        total.vectorized += s.vectorized;
        total.loops_visited += s.loops_visited;
        total.skipped_shape += s.skipped_shape;
        total.skipped_no_induction += s.skipped_no_induction;
        total.skipped_op_unsupported += s.skipped_op_unsupported;
    }
    total
}

// ─── Pattern recognition ──────────────────────────────────────────

enum Recognition {
    Match(SaxpyPattern),
    SkipShape,
    SkipNoInduction,
    SkipOp,
}

/// Everything the rewriter needs to know about a matched SAXPY loop.
struct SaxpyPattern {
    /// Loop header block id.
    header: HirId,
    /// The single body block (a successor of header that branches
    /// back to header).
    body: HirId,
    /// Loop exit block id (the false-target of the header's
    /// CondBranch).
    exit: HirId,
    /// Unique outside-the-loop predecessor of the header — where we
    /// stash the vector-bound computation and the trip-count
    /// pre-roll.
    preheader: HirId,
    /// Induction variable phi result id (the `i` in `for i in 0..n`).
    i_phi: HirId,
    /// `i_next = i + 1` instruction's result id in the body.
    i_next: HirId,
    /// Loop bound (the `n` in `i < n`).
    n: HirId,
    /// The elementwise expression whose value is stored to `c[i]`.
    expr: VecExpr,
    /// Element type, vectorized at lanes=4.
    elem_ty: HirType,
    /// Pointer to the output array.
    ptr_c: HirId,
    /// GEP, Load, Binary, GEP, Store instructions in the body — we
    /// snapshot them so the rewriter can copy them into the scalar
    /// epilogue verbatim.
    body_insts: Vec<HirInstruction>,
}

/// One node of the elementwise expression a vectorized body computes.
///
/// A leaf is either an element read at the induction variable, which
/// becomes a vector load, or a value the loop does not change, which
/// becomes one broadcast hoisted out of the loop. That second leaf is
/// what lets a scaled kernel such as `y[i] = alpha * x[i] + y[i]` match:
/// `alpha` is the same in every lane and on every iteration.
#[derive(Debug, Clone)]
enum VecExpr {
    /// An element read through `GEP(ptr, [i])`.
    Load(HirId),
    /// A loop-invariant scalar, broadcast to every lane.
    Splat(HirId),
    /// An elementwise operation over two sub-expressions.
    Binary(BinaryOp, Box<VecExpr>, Box<VecExpr>),
}

impl VecExpr {
    /// Whether anything in the tree is read from memory. An expression
    /// of nothing but invariants is the same on every iteration and
    /// wants hoisting, not vectorizing.
    fn reads_memory(&self) -> bool {
        match self {
            VecExpr::Load(_) => true,
            VecExpr::Splat(_) => false,
            VecExpr::Binary(_, l, r) => l.reads_memory() || r.reads_memory(),
        }
    }
}

/// Depth of expression tree worth matching. Elementwise kernels are
/// shallow, and the cap keeps a pathological body from costing compile
/// time.
const MAX_EXPR_DEPTH: usize = 8;

/// Why an expression could not be matched. The two are reported
/// separately: a shape nothing can be done about, against an operation
/// that would vectorize if it were supported.
enum VecReject {
    Shape,
    Op,
}

/// Build the expression the store's value computes.
///
/// Every instruction the tree consumes is pushed to `accounted`, so the
/// caller can confirm the body holds nothing else before replacing it.
#[allow(clippy::too_many_arguments)]
fn build_vec_expr(
    func: &HirFunction,
    body: &HirBlock,
    body_id: HirId,
    header_id: HirId,
    value: HirId,
    i: HirId,
    elem_ty: &HirType,
    depth: usize,
    accounted: &mut Vec<HirId>,
) -> Result<VecExpr, VecReject> {
    if depth > MAX_EXPR_DEPTH {
        return Err(VecReject::Shape);
    }

    // An element read at the induction variable.
    if let Some(ptr) = resolve_load_through_gep(body, value, i) {
        // The base has to name the same array on every iteration.
        if defined_in_block(func, ptr, body_id) || defined_in_block(func, ptr, header_id) {
            return Err(VecReject::Shape);
        }
        if value_type(func, value).as_ref() != Some(elem_ty) {
            return Err(VecReject::Shape);
        }
        accounted.push(value);
        if let Some(HirInstruction::Load { ptr: gep, .. }) = find_inst_by_result_in(body, value) {
            accounted.push(*gep);
        }
        return Ok(VecExpr::Load(ptr));
    }

    match find_inst_by_result_in(body, value) {
        Some(HirInstruction::Binary {
            op,
            left,
            right,
            ty,
            ..
        }) => {
            if ty != elem_ty {
                return Err(VecReject::Shape);
            }
            if !is_vectorizable_op(*op) {
                return Err(VecReject::Op);
            }
            let (op, left, right) = (*op, *left, *right);
            accounted.push(value);
            let l = build_vec_expr(
                func,
                body,
                body_id,
                header_id,
                left,
                i,
                elem_ty,
                depth + 1,
                accounted,
            )?;
            let r = build_vec_expr(
                func,
                body,
                body_id,
                header_id,
                right,
                i,
                elem_ty,
                depth + 1,
                accounted,
            )?;
            Ok(VecExpr::Binary(op, Box::new(l), Box::new(r)))
        }
        // Defined by no instruction in the body. If the loop does not
        // define it at all then it holds the same value throughout, and
        // every lane wants that value.
        None => {
            if defined_in_block(func, value, body_id) || defined_in_block(func, value, header_id) {
                return Err(VecReject::Shape);
            }
            if value_type(func, value).as_ref() != Some(elem_ty) {
                return Err(VecReject::Shape);
            }
            Ok(VecExpr::Splat(value))
        }
        _ => Err(VecReject::Shape),
    }
}

fn recognise_saxpy(func: &HirFunction, lp: &NaturalLoop) -> Recognition {
    // 1. Body must be a single block, and the header must have exactly
    //    one body block among its successors.
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

    // 2. Header has CondBranch(cond, body, exit).
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
    } else if false_t == body_id {
        // Loop runs while the condition is FALSE — we don't handle
        // this orientation yet (would need to invert the bound
        // check at vectorize time).
        return Recognition::SkipShape;
    } else {
        return Recognition::SkipShape;
    };

    // 3. Header has exactly one phi (the induction variable) with
    //    init 0 from preheader, increment from body.
    if header.phis.len() != 1 {
        return Recognition::SkipNoInduction;
    }
    let phi = &header.phis[0];

    // Find preheader: the unique outside-the-loop predecessor.
    let outside: Vec<HirId> = header
        .predecessors
        .iter()
        .copied()
        .filter(|p| !lp.body.contains(p))
        .collect();
    if outside.len() != 1 {
        return Recognition::SkipShape;
    }
    let preheader_id = outside[0];

    // Phi incoming: should have (value_from_preheader, preheader_id)
    // and (i_next_from_body, body_id).
    let mut init_val = None;
    let mut i_next = None;
    for &(val, blk) in &phi.incoming {
        if blk == preheader_id {
            init_val = Some(val);
        } else if blk == body_id {
            i_next = Some(val);
        }
    }
    let init_val = match init_val {
        Some(v) => v,
        None => return Recognition::SkipNoInduction,
    };
    let i_next = match i_next {
        Some(v) => v,
        None => return Recognition::SkipNoInduction,
    };

    // init_val must be constant 0.
    match func.values.get(&init_val).map(|v| &v.kind) {
        Some(HirValueKind::Constant(HirConstant::I32(0)))
        | Some(HirValueKind::Constant(HirConstant::I64(0)))
        | Some(HirValueKind::Constant(HirConstant::U32(0)))
        | Some(HirValueKind::Constant(HirConstant::U64(0))) => {}
        _ => return Recognition::SkipNoInduction,
    }

    // 4. Cond should be `i < n` for some n.
    let n = match func
        .blocks
        .get(&header_id)
        .and_then(|_| find_inst_by_result(func, header_id, cond))
    {
        Some(HirInstruction::Binary {
            op: BinaryOp::Lt,
            left,
            right,
            ..
        }) if *left == phi.result => *right,
        // The compare might also live in the body — we only check
        // the header here. Strict matching keeps the v1 small.
        _ => return Recognition::SkipShape,
    };

    // 5. Body shape: i_next = i + 1 (somewhere), and a SAXPY
    //    sub-sequence we can identify by following Store backwards.
    let i_next_inst = match find_inst_by_result_in(body, i_next) {
        Some(inst) => inst,
        None => return Recognition::SkipShape,
    };
    let one_is_const_1 = matches!(
        i_next_inst,
        HirInstruction::Binary { op: BinaryOp::Add, left, right, .. }
            if *left == phi.result
                && matches!(func.values.get(right).map(|v| &v.kind),
                    Some(HirValueKind::Constant(HirConstant::I32(1)))
                        | Some(HirValueKind::Constant(HirConstant::I64(1))))
    );
    if !one_is_const_1 {
        return Recognition::SkipShape;
    }

    // 6. Find the Store. There must be exactly one Store. Its
    //    pointer must come from a GEP indexed by phi.result. Its
    //    value must come from a Binary whose operands are two Loads
    //    each indexed by GEP(_, phi.result).
    let store = match body
        .instructions
        .iter()
        .find(|inst| matches!(inst, HirInstruction::Store { .. }))
    {
        Some(s) => s,
        None => return Recognition::SkipShape,
    };
    let (store_value, store_ptr) = match store {
        HirInstruction::Store { value, ptr, .. } => (*value, *ptr),
        _ => unreachable!(),
    };
    let n_stores = body
        .instructions
        .iter()
        .filter(|i| matches!(i, HirInstruction::Store { .. }))
        .count();
    if n_stores != 1 {
        return Recognition::SkipShape;
    }

    // store_ptr should be GEP(ptr_c, [phi.result]).
    let ptr_c = match find_inst_by_result_in(body, store_ptr) {
        Some(HirInstruction::GetElementPtr { ptr, indices, .. })
            if indices.len() == 1 && indices[0] == phi.result =>
        {
            *ptr
        }
        _ => return Recognition::SkipShape,
    };

    // The stored value is an elementwise expression over element reads
    // at `i` and values the loop does not change.
    let elem_ty = match value_type(func, store_value) {
        Some(t) => t,
        None => return Recognition::SkipShape,
    };
    if !is_vector_elem_type(&elem_ty) {
        return Recognition::SkipOp;
    }

    let mut accounted: Vec<HirId> = vec![store_ptr];
    let expr = match build_vec_expr(
        func,
        body,
        body_id,
        header_id,
        store_value,
        phi.result,
        &elem_ty,
        0,
        &mut accounted,
    ) {
        Ok(e) => e,
        Err(VecReject::Op) => return Recognition::SkipOp,
        Err(VecReject::Shape) => return Recognition::SkipShape,
    };

    // An expression of nothing but invariants belongs outside the loop.
    if !expr.reads_memory() {
        return Recognition::SkipShape;
    }

    // 7. The output base must name the same array on every iteration.
    if defined_in_block(func, ptr_c, body_id) || defined_in_block(func, ptr_c, header_id) {
        return Recognition::SkipShape;
    }

    // 8. The body is about to be replaced wholesale, so everything in
    //    it has to be part of what the vector form recomputes. Anything
    //    else, a second store or a call, would simply be dropped.
    accounted.push(i_next);
    for inst in &body.instructions {
        if matches!(inst, HirInstruction::Store { .. }) {
            continue;
        }
        match instruction_result(inst) {
            Some(r) if accounted.contains(&r) => {}
            _ => return Recognition::SkipShape,
        }
    }

    Recognition::Match(SaxpyPattern {
        header: header_id,
        body: body_id,
        exit: exit_id,
        preheader: preheader_id,
        i_phi: phi.result,
        i_next,
        n,
        expr,
        elem_ty,
        ptr_c,
        body_insts: body.instructions.clone(),
    })
}

fn is_vectorizable_op(op: BinaryOp) -> bool {
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

fn find_inst_by_result<'a>(
    func: &'a HirFunction,
    block_id: HirId,
    target: HirId,
) -> Option<&'a HirInstruction> {
    let block = func.blocks.get(&block_id)?;
    find_inst_by_result_in(block, target)
}

fn find_inst_by_result_in(block: &HirBlock, target: HirId) -> Option<&HirInstruction> {
    block.instructions.iter().find(|inst| match inst {
        HirInstruction::Binary { result, .. } => *result == target,
        HirInstruction::Unary { result, .. } => *result == target,
        HirInstruction::Cast { result, .. } => *result == target,
        HirInstruction::GetElementPtr { result, .. } => *result == target,
        HirInstruction::ExtractValue { result, .. } => *result == target,
        HirInstruction::Load { result, .. } => *result == target,
        HirInstruction::Alloca { result, .. } => *result == target,
        HirInstruction::Select { result, .. } => *result == target,
        _ => false,
    })
}

/// Given a value that should be a `Load(GEP(ptr, [i]))`, return the
/// pointer or `None` if the shape doesn't match.
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

/// Materialise the vector main loop + scalar epilogue. After this:
///
/// ```text
/// preheader → vec_check → vec_body → vec_check (back-edge)
///                 ↓
///               scalar_check → scalar_body → scalar_check (back-edge)
///                                   ↓
///                                 exit
/// ```
///
/// where `vec_n = n & ~3` (rounded down to multiple of 4) drives the
/// vector loop and `n` drives the scalar tail. The original header
/// and body blocks are reused — header becomes vec_check, body
/// becomes vec_body (with vector ops); two new blocks are
/// synthesised for the scalar epilogue.
fn rewrite_to_vector_loop(func: &mut HirFunction, pat: &SaxpyPattern) {
    // Allocate ids for the synthesised scalar epilogue blocks.
    let scalar_check = HirId::new();
    let scalar_body = HirId::new();

    // ── 1. Compute vec_n = (n & ~(lanes-1)) in the preheader, before
    //       its terminator: round the trip count down to a whole number
    //       of vectors. Lanes come from the element and the target
    //       width, so the mask and the step below follow from them
    //       rather than assuming four.
    let lanes = match crate::target_vector::lanes_for(&pat.elem_ty) {
        Some(l) => l,
        None => return,
    };
    let n_ty = func
        .values
        .get(&pat.n)
        .map(|v| v.ty.clone())
        .unwrap_or(HirType::I64);
    let lane_mask = !(lanes as i64 - 1);
    let mask_const = match n_ty {
        HirType::I32 | HirType::U32 => HirConstant::I32(lane_mask as i32),
        _ => HirConstant::I64(lane_mask),
    };
    let mask_id = create_value(func, n_ty.clone(), HirValueKind::Constant(mask_const));
    let vec_n_id = create_value(func, n_ty.clone(), HirValueKind::Instruction);
    let preheader_blk = func.blocks.get_mut(&pat.preheader).unwrap();
    preheader_blk.instructions.push(HirInstruction::Binary {
        op: BinaryOp::And,
        result: vec_n_id,
        ty: n_ty.clone(),
        left: pat.n,
        right: mask_id,
    });

    // ── 2. Rewrite the header (= vec_check). Replace the
    //       `i < n` compare with `i < vec_n`, and redirect the
    //       false-target from `exit` to `scalar_check`.
    //
    //       We mint the fresh compare-result id BEFORE re-borrowing
    //       `func.blocks` so the two mutable borrows don't overlap.
    let new_cmp_id = create_value(func, HirType::Bool, HirValueKind::Instruction);
    {
        let header_blk = func.blocks.get_mut(&pat.header).unwrap();
        let old_cmp = match &header_blk.terminator {
            HirTerminator::CondBranch { condition, .. } => *condition,
            _ => unreachable!("verified during recognition"),
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
                    *result = new_cmp_id;
                    *right = vec_n_id;
                }
            }
        }
        header_blk.terminator = HirTerminator::CondBranch {
            condition: new_cmp_id,
            true_target: pat.body,
            false_target: scalar_check,
        };
    }

    // ── 3. Replace the body's scalar Load/Binary/Store with the
    //       vector equivalents, and bump i by 4 instead of 1.
    let elem_ty = pat.elem_ty.clone();
    let vec_ty = HirType::Vector(Box::new(elem_ty.clone()), lanes as u32);

    // Build the vector body ahead of installing it, so the broadcasts
    // the expression needs can be collected for the preheader on the
    // same walk.
    let mut vec_body: Vec<HirInstruction> = Vec::new();
    let mut vec_pre: Vec<HirInstruction> = Vec::new();
    let vc_vec_id = emit_vec_expr(
        func,
        &pat.expr,
        &elem_ty,
        &vec_ty,
        pat.i_phi,
        &mut vec_body,
        &mut vec_pre,
    );

    // The broadcasts go ahead of the loop, after the vector bound the
    // preheader already computes.
    if !vec_pre.is_empty() {
        let pre_blk = func.blocks.get_mut(&pat.preheader).unwrap();
        pre_blk.instructions.append(&mut vec_pre);
    }

    let addr_c_vec = create_value(
        func,
        HirType::Ptr(Box::new(elem_ty.clone())),
        HirValueKind::Instruction,
    );
    let step = create_value(
        func,
        n_ty.clone(),
        HirValueKind::Constant(match n_ty {
            HirType::I32 | HirType::U32 => HirConstant::I32(lanes as i32),
            _ => HirConstant::I64(lanes as i64),
        }),
    );
    let new_i_next = create_value(func, n_ty.clone(), HirValueKind::Instruction);
    let elem_size = elem_size_bytes(&elem_ty) as u32;

    {
        let body_blk = func.blocks.get_mut(&pat.body).unwrap();
        body_blk.instructions.clear();
        body_blk.instructions.append(&mut vec_body);
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_c_vec,
            ty: HirType::Ptr(Box::new(elem_ty.clone())),
            ptr: pat.ptr_c,
            indices: vec![pat.i_phi],
        });
        body_blk.instructions.push(HirInstruction::VectorStore {
            value: vc_vec_id,
            ptr: addr_c_vec,
            align: elem_size,
        });
        // Bump i by 4.
        body_blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::Add,
            result: new_i_next,
            ty: n_ty.clone(),
            left: pat.i_phi,
            right: step,
        });
        body_blk.terminator = HirTerminator::Branch { target: pat.header };
    }

    // Rewrite the header's phi to use new_i_next as the body
    // incoming value instead of the original i_next.
    {
        let header_blk = func.blocks.get_mut(&pat.header).unwrap();
        for phi in &mut header_blk.phis {
            for (val, blk) in phi.incoming.iter_mut() {
                if *blk == pat.body && *val == pat.i_next {
                    *val = new_i_next;
                }
            }
        }
    }

    // ── 4. Materialise the scalar epilogue: scalar_check (`i < n`
    //       → scalar_body : exit) and scalar_body (the original
    //       scalar body verbatim, branching back to scalar_check).
    //
    //       Both blocks need fresh phi/inst ids because the scalar
    //       loop's `i` is a NEW phi (joining the vec loop's final
    //       `i_phi` value with the scalar body's `i_next2`).
    let scalar_i = create_value(func, n_ty.clone(), HirValueKind::Instruction);
    let scalar_cond = create_value(func, HirType::Bool, HirValueKind::Instruction);

    // scalar_check block.
    let mut scalar_check_blk = HirBlock::new(scalar_check);
    scalar_check_blk.predecessors = vec![pat.header, scalar_body];
    scalar_check_blk.successors = vec![scalar_body, pat.exit];
    scalar_check_blk.phis.push(HirPhi {
        result: scalar_i,
        ty: n_ty.clone(),
        // Body incoming (`scalar_body`) is patched after we mint
        // the scalar body's `i_next2` below.
        incoming: vec![(pat.i_phi, pat.header)],
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

    // scalar_body block — clone the original scalar instructions,
    // substituting the phi's `i` reference with our new `scalar_i`
    // and minting fresh result ids so they don't collide with the
    // vector-body ones.
    let mut substitutions: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();
    substitutions.insert(pat.i_phi, scalar_i);

    let mut scalar_body_blk = HirBlock::new(scalar_body);
    scalar_body_blk.predecessors = vec![scalar_check];
    scalar_body_blk.successors = vec![scalar_check];

    let mut scalar_i_next = None;
    for inst in &pat.body_insts {
        if let Some(res) = instruction_result(inst) {
            // The i_next add gets special treatment so we can
            // remember its fresh id and feed it into the scalar
            // check's phi.
            let new_res = create_value(
                func,
                value_type(func, res).unwrap_or(HirType::I64),
                HirValueKind::Instruction,
            );
            substitutions.insert(res, new_res);
            if res == pat.i_next {
                scalar_i_next = Some(new_res);
            }
        }
        let mut cloned = inst.clone();
        substitute_operands(&mut cloned, &substitutions);
        scalar_body_blk.instructions.push(cloned);
    }
    scalar_body_blk.terminator = HirTerminator::Branch {
        target: scalar_check,
    };
    func.blocks.insert(scalar_body, scalar_body_blk);

    // Patch scalar_check's phi to include the body-incoming edge.
    if let Some(scalar_i_next_id) = scalar_i_next {
        let blk = func.blocks.get_mut(&scalar_check).unwrap();
        for phi in &mut blk.phis {
            phi.incoming.push((scalar_i_next_id, scalar_body));
        }
    }

    // ── 5. Wire the exit's predecessor list to include scalar_check
    //       (was: header; now header falls through to scalar_check
    //       on the false-edge, scalar_check falls through to exit).
    if let Some(exit_blk) = func.blocks.get_mut(&pat.exit) {
        exit_blk.predecessors.retain(|p| *p != pat.header);
        exit_blk.predecessors.push(scalar_check);
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

/// Emit the vector form of `expr`, and return the value holding it.
///
/// Element reads and arithmetic append to `body`. A broadcast appends to
/// `pre` instead: the value it splats is the same on every iteration, so
/// repeating it per iteration would be waste the loop cannot remove.
fn emit_vec_expr(
    func: &mut HirFunction,
    expr: &VecExpr,
    elem_ty: &HirType,
    vec_ty: &HirType,
    i_phi: HirId,
    body: &mut Vec<HirInstruction>,
    pre: &mut Vec<HirInstruction>,
) -> HirId {
    match expr {
        VecExpr::Load(ptr) => {
            let addr = create_value(
                func,
                HirType::Ptr(Box::new(elem_ty.clone())),
                HirValueKind::Instruction,
            );
            let out = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
            body.push(HirInstruction::GetElementPtr {
                result: addr,
                ty: HirType::Ptr(Box::new(elem_ty.clone())),
                ptr: *ptr,
                indices: vec![i_phi],
            });
            body.push(HirInstruction::VectorLoad {
                result: out,
                ty: vec_ty.clone(),
                ptr: addr,
                align: elem_size_bytes(elem_ty) as u32,
            });
            out
        }
        VecExpr::Splat(scalar) => {
            let out = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
            pre.push(HirInstruction::VectorSplat {
                result: out,
                ty: vec_ty.clone(),
                scalar: *scalar,
            });
            out
        }
        VecExpr::Binary(op, l, r) => {
            let lv = emit_vec_expr(func, l, elem_ty, vec_ty, i_phi, body, pre);
            let rv = emit_vec_expr(func, r, elem_ty, vec_ty, i_phi, body, pre);
            let out = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
            body.push(HirInstruction::Binary {
                op: *op,
                result: out,
                ty: vec_ty.clone(),
                left: lv,
                right: rv,
            });
            out
        }
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

fn value_type(func: &HirFunction, id: HirId) -> Option<HirType> {
    func.values.get(&id).map(|v| v.ty.clone())
}

fn instruction_result(inst: &HirInstruction) -> Option<HirId> {
    match inst {
        HirInstruction::Binary { result, .. } => Some(*result),
        HirInstruction::Unary { result, .. } => Some(*result),
        HirInstruction::Cast { result, .. } => Some(*result),
        HirInstruction::GetElementPtr { result, .. } => Some(*result),
        HirInstruction::ExtractValue { result, .. } => Some(*result),
        HirInstruction::InsertValue { result, .. } => Some(*result),
        HirInstruction::Load { result, .. } => Some(*result),
        HirInstruction::Alloca { result, .. } => Some(*result),
        HirInstruction::Call { result, .. } => *result,
        HirInstruction::IndirectCall { result, .. } => *result,
        HirInstruction::Select { result, .. } => Some(*result),
        HirInstruction::Atomic { result, .. } => Some(*result),
        _ => None,
    }
}

fn substitute_operands(inst: &mut HirInstruction, subs: &IndexMap<HirId, HirId>) {
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

    fn empty_sig() -> HirFunctionSignature {
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

    /// Build the canonical SAXPY loop:
    ///
    /// ```text
    /// for i in 0..n { c[i] = a[i] + b[i] }
    /// ```
    ///
    /// over the given elem_ty + binary op.
    fn build_saxpy(elem_ty: HirType, op: BinaryOp) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global("saxpy"), empty_sig());
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
        let ptr_c = add_param(&mut f, ptr_ty.clone(), 2);
        let n = add_param(&mut f, n_ty.clone(), 3);
        let zero = add_const(&mut f, n_ty.clone(), HirConstant::I64(0));
        let one = add_const(&mut f, n_ty.clone(), HirConstant::I64(1));

        let i_phi = add_inst(&mut f, n_ty.clone());
        let i_next = add_inst(&mut f, n_ty.clone());
        f.blocks.get_mut(&header).unwrap().phis.push(HirPhi {
            result: i_phi,
            ty: n_ty.clone(),
            incoming: vec![(zero, entry), (i_next, body)],
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
        let addr_c = add_inst(&mut f, ptr_ty.clone());

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
            op,
            result: vc,
            ty: elem_ty.clone(),
            left: va,
            right: vb,
        });
        body_blk.instructions.push(HirInstruction::GetElementPtr {
            result: addr_c,
            ty: ptr_ty.clone(),
            ptr: ptr_c,
            indices: vec![i_phi],
        });
        body_blk.instructions.push(HirInstruction::Store {
            value: vc,
            ptr: addr_c,
            align: 4,
            volatile: false,
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
    fn vectorizes_f32_add_saxpy_loop() {
        let mut f = build_saxpy(HirType::F32, BinaryOp::FAdd);
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 1, "{stats:?}");
        // Body should now contain VectorLoad / VectorStore / vector
        // Binary.
        let body_id = f
            .blocks
            .values()
            .find(|b| {
                b.instructions
                    .iter()
                    .any(|i| matches!(i, HirInstruction::VectorStore { .. }))
            })
            .map(|b| b.id)
            .expect("vector body present");
        let body = &f.blocks[&body_id];
        assert!(body
            .instructions
            .iter()
            .any(|i| matches!(i, HirInstruction::VectorLoad { .. })));
        assert!(body
            .instructions
            .iter()
            .any(|i| matches!(i, HirInstruction::VectorStore { .. })));
    }

    #[test]
    fn skips_loops_with_unsupported_op() {
        // Build a SAXPY loop but with a non-vectorizable op (Div).
        let mut f = build_saxpy(HirType::I32, BinaryOp::Div);
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 0);
        assert_eq!(stats.skipped_op_unsupported, 1);
    }

    #[test]
    fn skips_non_loop_function() {
        let mut f = HirFunction::new(InternedString::new_global("nop"), empty_sig());
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return { values: vec![] };
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 0);
        assert_eq!(stats.loops_visited, 0);
    }

    #[test]
    fn scalar_epilogue_branches_back_to_check() {
        let mut f = build_saxpy(HirType::F32, BinaryOp::FMul);
        run(&mut f);
        // Find a block that ends in Branch back to a check block —
        // the scalar epilogue body.
        let mut found_epilogue_branch = false;
        for blk in f.blocks.values() {
            if let HirTerminator::Branch { target } = blk.terminator {
                if blk.id != target {
                    // Inspect the target: a scalar_check has a Lt
                    // comparison and a CondBranch to exit/scalar_body.
                    if let Some(t) = f.blocks.get(&target) {
                        if matches!(t.terminator, HirTerminator::CondBranch { .. })
                            && t.instructions.iter().any(|i| {
                                matches!(
                                    i,
                                    HirInstruction::Binary {
                                        op: BinaryOp::Lt,
                                        ..
                                    }
                                )
                            })
                        {
                            found_epilogue_branch = true;
                        }
                    }
                }
            }
        }
        assert!(
            found_epilogue_branch,
            "scalar epilogue body should branch back to scalar_check"
        );
    }
}
