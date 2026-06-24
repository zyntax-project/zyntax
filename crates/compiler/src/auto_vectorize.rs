//! Generalised HIR loop vectorization — supersedes `loop_vectorize` and
//! `reduction_vectorize` by recognising both shapes (elementwise stores
//! and reductions) under a single analysis + cost-modelled transform.
//!
//! ## Why a unified pass
//!
//! `loop_vectorize` and `reduction_vectorize` each match a single
//! hard-coded body shape (one elementwise op storing to `c[i]` or
//! `sum += op(a[i], b[i])`). Real loops drift away from those
//! templates — a loop with even one extra scalar arithmetic instruction
//! falls off both pattern matchers. This pass uses a more flexible
//! analyse-then-vectorize structure:
//!
//!   1. analyse_loop walks the body, classifying every instruction as
//!      memory access (Load/Store/GEP indexed by the IV), reduction
//!      update (Binary updating a header phi), scalar arithmetic
//!      (Binary/Unary/Cast on lane-able element types), or hazard
//!      (Call, Atomic, side-effectful op, FDiv/FRem etc.).
//!   2. A cost model decides whether vectorization is worth it given
//!      the work mix and the (possibly known) trip count.
//!   3. The rewriter splits the loop into vec-loop + scalar tail with
//!      the same CFG shape both old passes already produced.
//!
//! ## V1 scope
//!
//! * Single-body-block counted loops with header phi `i = phi[0, i+k]`,
//!   header bound `i < n`, body branching back to header.
//! * Stride-1 (contiguous) GEPs only — no gather/scatter.
//! * Add/FAdd accumulator reductions only (commutative + associative
//!   under the assumption float reordering is acceptable, matching
//!   LLVM `-ffast-math`-like semantics for these loops).
//! * Hazard set rejects: calls (except a small intrinsic allowlist),
//!   atomics, FDiv/FRem/Div/Rem, indirect calls, throw.
//!
//! ## Knobs
//!
//! * `ZYNTAX_DISABLE_AUTO_VECTORIZE=1` — full pass skip.
//! * `ZYNTAX_AUTO_VEC_DUMP=1` — per-loop dump to stderr.
//! * `ZYNTAX_AUTO_VEC_MIN_TRIP_COUNT=<n>` — override min_trip_count.

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, HirBlock, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirPhi,
    HirTerminator, HirType, HirValue, HirValueKind,
};
use indexmap::IndexMap;
use std::collections::HashSet;

/// Per-pass counters returned to callers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AutoVectorizeStats {
    /// Loops we transformed.
    pub vectorized: usize,
    /// Loops we examined.
    pub loops_visited: usize,
    /// Body shape didn't match (multi-block body, weird terminator, etc.).
    pub rejected_shape: usize,
    /// Body contains a hazardous op (Call, FDiv, Atomic, etc.).
    pub rejected_hazard: usize,
    /// Cost model says vectorization isn't worth it.
    pub rejected_cost: usize,
    /// Couldn't find a valid induction variable.
    pub rejected_no_iv: usize,
    /// Trip count is known and below `min_trip_count`.
    pub rejected_trip_count: usize,
}

impl AutoVectorizeStats {
    fn add(&mut self, other: AutoVectorizeStats) {
        self.vectorized += other.vectorized;
        self.loops_visited += other.loops_visited;
        self.rejected_shape += other.rejected_shape;
        self.rejected_hazard += other.rejected_hazard;
        self.rejected_cost += other.rejected_cost;
        self.rejected_no_iv += other.rejected_no_iv;
        self.rejected_trip_count += other.rejected_trip_count;
    }
}

/// Pass configuration.
#[derive(Debug, Clone, Copy)]
pub struct AutoVectorizePass {
    /// Skip loops with statically-known trip count below this.
    pub min_trip_count: usize,
    /// If false, accept every loop that passes hazard/shape checks
    /// regardless of cost. Useful for tests.
    pub use_cost_model: bool,
    /// Default lane count for 32-bit element types. 64-bit types are
    /// vectorized at half the lanes.
    pub simd_lanes: u32,
}

impl Default for AutoVectorizePass {
    fn default() -> Self {
        let min_tc = std::env::var("ZYNTAX_AUTO_VEC_MIN_TRIP_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        Self {
            min_trip_count: min_tc,
            use_cost_model: true,
            simd_lanes: 4,
        }
    }
}

/// Run the pass on a single function. Used by tests and the module
/// entry. Returns the per-pass counters.
pub fn run(func: &mut HirFunction) -> AutoVectorizeStats {
    AutoVectorizePass::default().run(func)
}

/// Run the pass on every function in `module`.
pub fn run_module(module: &mut HirModule) -> AutoVectorizeStats {
    if std::env::var("ZYNTAX_DISABLE_AUTO_VECTORIZE").is_ok() {
        return AutoVectorizeStats::default();
    }
    let pass = AutoVectorizePass::default();
    let mut total = AutoVectorizeStats::default();
    for func in module.functions.values_mut() {
        total.add(pass.run(func));
    }
    total
}

impl AutoVectorizePass {
    pub fn run(&self, func: &mut HirFunction) -> AutoVectorizeStats {
        let dt = DominatorTree::new(func);
        let lf = LoopForest::detect(func, &dt);
        if lf.loops().is_empty() {
            return AutoVectorizeStats::default();
        }

        let dump = std::env::var("ZYNTAX_AUTO_VEC_DUMP").is_ok();
        let mut stats = AutoVectorizeStats::default();

        let func_name = func
            .name
            .resolve_global()
            .unwrap_or_else(|| "<anon>".to_string());
        for lp in lf.loops().to_vec() {
            stats.loops_visited += 1;
            match self.analyze_loop(func, &lp) {
                LoopOutcome::Vectorize(plan) => {
                    if dump {
                        eprintln!(
                            "auto_vectorize: {} header={:?} shape={:?} lanes={} reductions={} loads={} stores={}",
                            func_name,
                            plan.header,
                            plan.shape,
                            plan.lanes,
                            plan.reductions.len(),
                            plan.vec_loads.len(),
                            plan.vec_stores.len(),
                        );
                    }
                    vectorize_loop(func, &plan);
                    stats.vectorized += 1;
                }
                LoopOutcome::RejectShape(reason) => {
                    if dump {
                        eprintln!("auto_vectorize: {} reject shape: {}", func_name, reason);
                    }
                    stats.rejected_shape += 1;
                }
                LoopOutcome::RejectNoIv => stats.rejected_no_iv += 1,
                LoopOutcome::RejectHazard(reason) => {
                    if dump {
                        eprintln!("auto_vectorize: {} reject hazard: {}", func_name, reason);
                    }
                    stats.rejected_hazard += 1;
                }
                LoopOutcome::RejectCost(speedup) => {
                    if dump {
                        eprintln!(
                            "auto_vectorize: {} reject cost: speedup={:.2}",
                            func_name, speedup
                        );
                    }
                    stats.rejected_cost += 1;
                }
                LoopOutcome::RejectTripCount(tc) => {
                    if dump {
                        eprintln!("auto_vectorize: {} reject trip_count: {}", func_name, tc);
                    }
                    stats.rejected_trip_count += 1;
                }
            }
        }

        stats
    }

    fn analyze_loop(&self, func: &HirFunction, lp: &NaturalLoop) -> LoopOutcome {
        // V1: header + single body block.
        if lp.body.len() != 2 {
            return LoopOutcome::RejectShape("multi-block body");
        }
        let header_id = lp.header;
        let body_id = match lp.body.iter().find(|&&b| b != header_id) {
            Some(&b) => b,
            None => return LoopOutcome::RejectShape("no body block"),
        };
        let header = match func.blocks.get(&header_id) {
            Some(b) => b,
            None => return LoopOutcome::RejectShape("no header"),
        };
        let body = match func.blocks.get(&body_id) {
            Some(b) => b,
            None => return LoopOutcome::RejectShape("no body"),
        };

        // Header must be CondBranch on `i < n` with body-then-exit
        // orientation.
        let (cond_id, true_t, false_t) = match &header.terminator {
            HirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => (*condition, *true_target, *false_target),
            _ => return LoopOutcome::RejectShape("header terminator not CondBranch"),
        };
        let exit_id = if true_t == body_id {
            false_t
        } else {
            return LoopOutcome::RejectShape("inverted cond-branch orientation");
        };

        // Body must branch back to header.
        if !matches!(&body.terminator, HirTerminator::Branch { target } if *target == header_id) {
            return LoopOutcome::RejectShape("body doesn't fall back to header");
        }

        // Preheader: exactly one outside-loop predecessor.
        let outside: Vec<HirId> = header
            .predecessors
            .iter()
            .copied()
            .filter(|p| !lp.body.contains(p))
            .collect();
        if outside.len() != 1 {
            return LoopOutcome::RejectShape("not exactly one preheader");
        }
        let preheader_id = outside[0];

        // Classify each header phi: induction (init 0, body-incoming
        // is an Add of self+const-step) or reduction accumulator
        // (init 0, body-incoming is an Add of self+expr).
        let mut iv: Option<InductionVariable> = None;
        let mut reductions: Vec<Reduction> = Vec::new();
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
            let init = match init {
                Some(v) => v,
                None => return LoopOutcome::RejectShape("phi missing preheader-incoming"),
            };
            let next = match next {
                Some(v) => v,
                None => return LoopOutcome::RejectShape("phi missing body-incoming"),
            };
            if !is_zero_constant(func, init) {
                return LoopOutcome::RejectShape("phi init not zero");
            }

            if let Some(step) = is_const_step_increment(func, body, next, phi.result) {
                if iv.is_some() {
                    // Two IVs — V1 only supports one.
                    return LoopOutcome::RejectShape("multiple induction variables");
                }
                iv = Some(InductionVariable {
                    phi: phi.result,
                    next,
                    step,
                });
            } else if let Some(op) = find_reduction_update(body, next, phi.result) {
                if !matches!(op, BinaryOp::Add | BinaryOp::FAdd) {
                    return LoopOutcome::RejectHazard("non-Add reduction op");
                }
                reductions.push(Reduction {
                    phi: phi.result,
                    next,
                    op,
                    elem_ty: phi.ty.clone(),
                });
            } else {
                return LoopOutcome::RejectShape("phi not classifiable");
            }
        }

        let iv = match iv {
            Some(i) => i,
            None => return LoopOutcome::RejectNoIv,
        };

        // Condition must be `iv < n`.
        let n = match find_inst_by_result_in(header, cond_id) {
            Some(HirInstruction::Binary {
                op: BinaryOp::Lt,
                left,
                right,
                ..
            }) if *left == iv.phi => *right,
            _ => return LoopOutcome::RejectShape("header cond not iv<n"),
        };

        // Trip count: known if `n` is a constant.
        let known_trip = constant_int(func, n);
        if let Some(tc) = known_trip {
            if (tc as usize) < self.min_trip_count {
                return LoopOutcome::RejectTripCount(tc as usize);
            }
        }

        // Scan body for hazards + collect mem accesses + scalar
        // arithmetic. Track each load/store and the ptr-base they're
        // indexed off.
        let mut vec_loads: Vec<MemAcc> = Vec::new();
        let mut vec_stores: Vec<MemAcc> = Vec::new();
        let mut n_arith: usize = 0;
        let mut elem_ty_hint: Option<HirType> = None;

        for inst in &body.instructions {
            match inst {
                HirInstruction::Binary { op, ty, result, .. } => {
                    // Skip the IV increment itself + reduction
                    // updates (we already handled those).
                    if *result == iv.next {
                        continue;
                    }
                    if reductions.iter().any(|r| r.next == *result) {
                        continue;
                    }
                    if !is_vectorizable_op(*op) {
                        return LoopOutcome::RejectHazard("non-vectorizable arith op");
                    }
                    if !is_vector_elem_type(ty) {
                        return LoopOutcome::RejectHazard("non-vectorizable element type");
                    }
                    n_arith += 1;
                    if elem_ty_hint.is_none() {
                        elem_ty_hint = Some(ty.clone());
                    }
                }
                HirInstruction::Unary { ty, .. } | HirInstruction::Cast { ty, .. } => {
                    if !is_vector_elem_type(ty) {
                        return LoopOutcome::RejectHazard("non-vectorizable unary/cast type");
                    }
                    n_arith += 1;
                    if elem_ty_hint.is_none() {
                        elem_ty_hint = Some(ty.clone());
                    }
                }
                HirInstruction::GetElementPtr { ptr, indices, .. } => {
                    // Must be stride-1 indexed by the IV.
                    if indices.len() != 1 || indices[0] != iv.phi {
                        return LoopOutcome::RejectShape("non-stride-1 GEP");
                    }
                    // Base must be loop-invariant.
                    if defined_in_block(func, *ptr, body_id)
                        || defined_in_block(func, *ptr, header_id)
                    {
                        return LoopOutcome::RejectShape("GEP base not loop-invariant");
                    }
                }
                HirInstruction::Load {
                    result, ty, ptr, ..
                } => {
                    let base = gep_base_for(body, *ptr, iv.phi);
                    let base = match base {
                        Some(b) => b,
                        None => return LoopOutcome::RejectShape("Load not via stride-1 GEP"),
                    };
                    if !is_vector_elem_type(ty) {
                        return LoopOutcome::RejectHazard("non-vectorizable load type");
                    }
                    vec_loads.push(MemAcc {
                        ptr_base: base,
                        elem_ty: ty.clone(),
                        scalar_result: *result,
                    });
                    if elem_ty_hint.is_none() {
                        elem_ty_hint = Some(ty.clone());
                    }
                }
                HirInstruction::Store { value, ptr, .. } => {
                    let base = gep_base_for(body, *ptr, iv.phi);
                    let base = match base {
                        Some(b) => b,
                        None => return LoopOutcome::RejectShape("Store not via stride-1 GEP"),
                    };
                    let ty = func
                        .values
                        .get(value)
                        .map(|v| v.ty.clone())
                        .unwrap_or(HirType::Void);
                    if !is_vector_elem_type(&ty) {
                        return LoopOutcome::RejectHazard("non-vectorizable store type");
                    }
                    vec_stores.push(MemAcc {
                        ptr_base: base,
                        elem_ty: ty.clone(),
                        scalar_result: *value,
                    });
                    if elem_ty_hint.is_none() {
                        elem_ty_hint = Some(ty);
                    }
                }
                HirInstruction::Call { .. } | HirInstruction::IndirectCall { .. } => {
                    return LoopOutcome::RejectHazard("call inside loop body");
                }
                HirInstruction::Atomic { .. } => {
                    return LoopOutcome::RejectHazard("atomic inside loop body");
                }
                HirInstruction::Alloca { .. } => {
                    return LoopOutcome::RejectHazard("alloca inside loop body");
                }
                _ => {
                    // Other instructions (e.g. ExtractValue,
                    // InsertValue, Select) — conservative reject for V1.
                    return LoopOutcome::RejectShape("unsupported inst kind");
                }
            }
        }

        // Pick the lane element type + lane count.
        let lane_ty = match elem_ty_hint {
            Some(t) => t,
            None => return LoopOutcome::RejectShape("could not infer element type"),
        };
        let lanes = match &lane_ty {
            HirType::I64 | HirType::U64 | HirType::F64 => 2,
            _ => self.simd_lanes,
        };

        // Step must equal `1` (we expand the IV by `lanes` in the
        // vec loop, then 1 in the scalar tail). Steps other than 1
        // aren't supported in V1.
        if iv.step != 1 {
            return LoopOutcome::RejectShape("IV step != 1");
        }

        // Classify shape (purely informational, kept for dump).
        let shape = match (vec_stores.is_empty(), reductions.is_empty()) {
            (true, false) => Shape::Reduction,
            (false, true) => {
                if vec_loads.is_empty() {
                    Shape::Elementwise // store-only is rare
                } else if vec_stores.len() == 1 && vec_loads.len() == 2 && n_arith >= 1 {
                    Shape::Saxpy
                } else {
                    Shape::Elementwise
                }
            }
            (false, false) => Shape::ReductionWithStore,
            (true, true) => Shape::Elementwise,
        };

        // Cost model.
        let speedup = if self.use_cost_model {
            estimate_speedup(
                vec_loads.len() + vec_stores.len(),
                n_arith,
                reductions.len(),
                lanes,
                known_trip,
            )
        } else {
            f64::INFINITY
        };
        if self.use_cost_model && speedup < 1.5 {
            return LoopOutcome::RejectCost(speedup);
        }

        LoopOutcome::Vectorize(LoopAnalysis {
            shape,
            header: header_id,
            body: body_id,
            exit: exit_id,
            preheader: preheader_id,
            iv,
            n,
            reductions,
            vec_loads,
            vec_stores,
            lane_ty,
            lanes,
            body_insts: body.instructions.clone(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Elementwise,
    Saxpy,
    Reduction,
    ReductionWithStore,
}

#[derive(Debug, Clone, Copy)]
struct InductionVariable {
    phi: HirId,
    next: HirId,
    step: i64,
}

#[derive(Debug, Clone)]
struct Reduction {
    phi: HirId,
    next: HirId,
    op: BinaryOp,
    elem_ty: HirType,
}

#[derive(Debug, Clone)]
struct MemAcc {
    /// Loop-invariant pointer base (the param/value the GEP indexes
    /// off).
    ptr_base: HirId,
    elem_ty: HirType,
    /// The Load result / Store value id in the original scalar body.
    /// Currently only used for diagnostics — the actual rewrite
    /// rebuilds the vector body from the analysis info.
    #[allow(dead_code)]
    scalar_result: HirId,
}

struct LoopAnalysis {
    #[allow(dead_code)]
    shape: Shape,
    header: HirId,
    body: HirId,
    exit: HirId,
    preheader: HirId,
    iv: InductionVariable,
    n: HirId,
    reductions: Vec<Reduction>,
    vec_loads: Vec<MemAcc>,
    vec_stores: Vec<MemAcc>,
    lane_ty: HirType,
    lanes: u32,
    /// Snapshot of the original scalar body — copied verbatim (with
    /// operand substitution) into the scalar tail.
    body_insts: Vec<HirInstruction>,
}

enum LoopOutcome {
    Vectorize(LoopAnalysis),
    RejectShape(&'static str),
    RejectNoIv,
    RejectHazard(&'static str),
    RejectCost(f64),
    RejectTripCount(usize),
}

// ─── Recognisers ──────────────────────────────────────────────────────

fn is_zero_constant(func: &HirFunction, val: HirId) -> bool {
    matches!(
        func.values.get(&val).map(|v| &v.kind),
        Some(HirValueKind::Constant(HirConstant::I32(0)))
            | Some(HirValueKind::Constant(HirConstant::I64(0)))
            | Some(HirValueKind::Constant(HirConstant::U32(0)))
            | Some(HirValueKind::Constant(HirConstant::U64(0)))
            | Some(HirValueKind::Constant(HirConstant::F32(0.0)))
            | Some(HirValueKind::Constant(HirConstant::F64(0.0)))
    )
}

fn constant_int(func: &HirFunction, val: HirId) -> Option<i64> {
    match func.values.get(&val).map(|v| &v.kind) {
        Some(HirValueKind::Constant(HirConstant::I32(v))) => Some(*v as i64),
        Some(HirValueKind::Constant(HirConstant::I64(v))) => Some(*v),
        Some(HirValueKind::Constant(HirConstant::U32(v))) => Some(*v as i64),
        Some(HirValueKind::Constant(HirConstant::U64(v))) => Some(*v as i64),
        _ => None,
    }
}

/// If `next` is `phi + const_step` in `body`, return the step.
fn is_const_step_increment(
    func: &HirFunction,
    body: &HirBlock,
    next: HirId,
    phi: HirId,
) -> Option<i64> {
    let inst = find_inst_by_result_in(body, next)?;
    match inst {
        HirInstruction::Binary {
            op: BinaryOp::Add,
            left,
            right,
            ..
        } if *left == phi => constant_int(func, *right),
        _ => None,
    }
}

/// If `next = Binary(Add/FAdd, phi, expr)`, return the op.
fn find_reduction_update(body: &HirBlock, next: HirId, phi: HirId) -> Option<BinaryOp> {
    let inst = find_inst_by_result_in(body, next)?;
    match inst {
        HirInstruction::Binary { op, left, .. }
            if *left == phi && matches!(op, BinaryOp::Add | BinaryOp::FAdd) =>
        {
            Some(*op)
        }
        _ => None,
    }
}

/// Walk the body to confirm `ptr` is a GEP base+iv stride-1 access;
/// return the base.
fn gep_base_for(body: &HirBlock, ptr: HirId, iv: HirId) -> Option<HirId> {
    let inst = find_inst_by_result_in(body, ptr)?;
    match inst {
        HirInstruction::GetElementPtr { ptr, indices, .. }
            if indices.len() == 1 && indices[0] == iv =>
        {
            Some(*ptr)
        }
        _ => None,
    }
}

fn find_inst_by_result_in(block: &HirBlock, target: HirId) -> Option<&HirInstruction> {
    block.instructions.iter().find(|inst| match inst {
        HirInstruction::Binary { result, .. } => *result == target,
        HirInstruction::Unary { result, .. } => *result == target,
        HirInstruction::Cast { result, .. } => *result == target,
        HirInstruction::GetElementPtr { result, .. } => *result == target,
        HirInstruction::ExtractValue { result, .. } => *result == target,
        HirInstruction::InsertValue { result, .. } => *result == target,
        HirInstruction::Load { result, .. } => *result == target,
        HirInstruction::Alloca { result, .. } => *result == target,
        HirInstruction::Select { result, .. } => *result == target,
        _ => false,
    })
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

fn is_vectorizable_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::FAdd
            | BinaryOp::FSub
            | BinaryOp::FMul
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Xor
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

/// Rough speedup estimate. The model is intentionally simple:
///
/// - Each loop iteration does `work = n_mem + n_arith` "ops" worth of
///   scalar work. Cost of the scalar loop ≈ `work * trip`.
/// - Each vector iteration also does `work` ops but covers `vf`
///   scalar iterations, so we run `trip/vf` vec iterations + a tail
///   of up to `vf-1` scalar iterations. Cost ≈
///   `work * trip / vf + work * (vf/2) + overhead`.
/// - `overhead` is a small fixed setup cost (vec_n compute,
///   horizontal reduce per reduction).
fn estimate_speedup(
    n_mem: usize,
    n_arith: usize,
    n_reductions: usize,
    vf: u32,
    known_trip: Option<i64>,
) -> f64 {
    let vf = vf as f64;
    let work = (n_mem + n_arith) as f64;
    if work <= 0.0 {
        return 0.0;
    }
    let overhead = 2.0 + (n_reductions as f64) * 1.5;
    let trip = known_trip.map(|t| t as f64).unwrap_or(64.0);
    if trip < vf {
        return 0.0;
    }
    let baseline = work * trip;
    let tail = work * (vf / 2.0); // expected scalar tail cost
    let new_cost = work * trip / vf + tail + overhead;
    if new_cost <= 0.0 {
        return f64::INFINITY;
    }
    baseline / new_cost
}

// ─── Rewriter ─────────────────────────────────────────────────────────

fn vectorize_loop(func: &mut HirFunction, plan: &LoopAnalysis) {
    let lanes = plan.lanes;
    let lane_ty = plan.lane_ty.clone();
    let vec_ty = HirType::Vector(Box::new(lane_ty.clone()), lanes);

    let n_ty = func
        .values
        .get(&plan.n)
        .map(|v| v.ty.clone())
        .unwrap_or(HirType::I64);

    // ── 1. Preheader: compute `vec_n = n & ~(lanes-1)` and seed
    //       vector accumulators for each reduction.
    let mask_const = match n_ty {
        HirType::I32 | HirType::U32 => HirConstant::I32(!(lanes as i32 - 1)),
        _ => HirConstant::I64(!(lanes as i64 - 1)),
    };
    let mask_id = create_value(func, n_ty.clone(), HirValueKind::Constant(mask_const));
    let vec_n_id = create_value(func, n_ty.clone(), HirValueKind::Instruction);

    // For each reduction we materialise a VectorSplat(0) in the
    // preheader.
    let mut reduction_vec_init: Vec<HirId> = Vec::with_capacity(plan.reductions.len());
    let mut reduction_vec_next: Vec<HirId> = Vec::with_capacity(plan.reductions.len());
    let mut reduction_vec_ty: Vec<HirType> = Vec::with_capacity(plan.reductions.len());

    for red in &plan.reductions {
        let red_vec_ty = HirType::Vector(Box::new(red.elem_ty.clone()), lanes);
        let zero_scalar = create_value(
            func,
            red.elem_ty.clone(),
            HirValueKind::Constant(match red.elem_ty {
                HirType::F32 => HirConstant::F32(0.0),
                HirType::F64 => HirConstant::F64(0.0),
                HirType::I32 | HirType::U32 => HirConstant::I32(0),
                _ => HirConstant::I64(0),
            }),
        );
        let vec_init = create_value(func, red_vec_ty.clone(), HirValueKind::Instruction);
        let vec_next = create_value(func, red_vec_ty.clone(), HirValueKind::Instruction);
        reduction_vec_init.push(vec_init);
        reduction_vec_next.push(vec_next);
        reduction_vec_ty.push(red_vec_ty.clone());

        let ph = func.blocks.get_mut(&plan.preheader).unwrap();
        ph.instructions.push(HirInstruction::VectorSplat {
            result: vec_init,
            ty: red_vec_ty,
            scalar: zero_scalar,
        });
    }

    {
        let ph = func.blocks.get_mut(&plan.preheader).unwrap();
        ph.instructions.push(HirInstruction::Binary {
            op: BinaryOp::And,
            result: vec_n_id,
            ty: n_ty.clone(),
            left: plan.n,
            right: mask_id,
        });
    }

    // ── 2. Rewrite header phis + bound check.
    let new_cmp_id = create_value(func, HirType::Bool, HirValueKind::Instruction);
    let lanes_const = create_value(
        func,
        n_ty.clone(),
        HirValueKind::Constant(match n_ty {
            HirType::I32 | HirType::U32 => HirConstant::I32(lanes as i32),
            _ => HirConstant::I64(lanes as i64),
        }),
    );
    let new_i_next = create_value(func, n_ty.clone(), HirValueKind::Instruction);

    // Reduction phis become vector-typed; their preheader-incoming
    // becomes the splat, body-incoming becomes new vec_next.
    {
        let header_blk = func.blocks.get_mut(&plan.header).unwrap();
        for phi in &mut header_blk.phis {
            if phi.result == plan.iv.phi {
                for (val, blk) in phi.incoming.iter_mut() {
                    if *blk == plan.body && *val == plan.iv.next {
                        *val = new_i_next;
                    }
                }
            } else if let Some(red_idx) = plan.reductions.iter().position(|r| r.phi == phi.result) {
                phi.ty = reduction_vec_ty[red_idx].clone();
                for (val, blk) in phi.incoming.iter_mut() {
                    if *blk == plan.preheader {
                        *val = reduction_vec_init[red_idx];
                    } else if *blk == plan.body {
                        *val = reduction_vec_next[red_idx];
                    }
                }
            }
        }
        // Update the HirValue type for reduction phis.
        for (red_idx, red) in plan.reductions.iter().enumerate() {
            if let Some(v) = func.values.get_mut(&red.phi) {
                v.ty = reduction_vec_ty[red_idx].clone();
            }
        }

        // Rewrite the Lt: old cmp becomes new_cmp_id against vec_n.
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
                    *result = new_cmp_id;
                    *right = vec_n_id;
                }
            }
        }
    }

    // ── 3. Rewrite body: replace scalar loads/binaries/stores with
    //       vector equivalents. Build the new body sequence from the
    //       analysis info, not the original instruction list — that
    //       way we don't have to track every single substitution.
    //
    //       Strategy: walk the original body instructions and translate
    //       each one to its vector form using a substitution map.
    let mut sub: IndexMap<HirId, HirId> = IndexMap::new();
    let mut new_insts: Vec<HirInstruction> = Vec::new();
    let mut reduction_phi_subs: IndexMap<HirId, HirId> = IndexMap::new();

    for (idx, red) in plan.reductions.iter().enumerate() {
        // Any read of the old scalar phi within the vector body now
        // refers to the vector-typed phi (same HirId). We don't need
        // to substitute the phi result itself — its HirValue.ty was
        // updated above. We DO need to map the old `next` value to
        // the new vector-typed next.
        sub.insert(red.next, reduction_vec_next[idx]);
        reduction_phi_subs.insert(red.phi, red.phi);
    }

    for inst in &plan.body_insts {
        match inst {
            HirInstruction::GetElementPtr {
                result,
                ty,
                ptr,
                indices,
            } => {
                // Keep GEPs as-is (their result is the address into
                // the array, indexed by the IV). The vector load/store
                // will use the same address.
                let new_res = create_value(func, ty.clone(), HirValueKind::Instruction);
                sub.insert(*result, new_res);
                new_insts.push(HirInstruction::GetElementPtr {
                    result: new_res,
                    ty: ty.clone(),
                    ptr: subbed(&sub, *ptr),
                    indices: indices.iter().map(|i| subbed(&sub, *i)).collect(),
                });
            }
            HirInstruction::Load {
                result, ty, ptr, ..
            } => {
                // Vector load.
                let vec_load_ty = HirType::Vector(Box::new(ty.clone()), lanes);
                let new_res = create_value(func, vec_load_ty.clone(), HirValueKind::Instruction);
                sub.insert(*result, new_res);
                new_insts.push(HirInstruction::VectorLoad {
                    result: new_res,
                    ty: vec_load_ty,
                    ptr: subbed(&sub, *ptr),
                    align: elem_size_bytes(ty) as u32,
                });
            }
            HirInstruction::Store { value, ptr, .. } => {
                // Vector store. The value should already have been
                // mapped to a vector value by an earlier translation
                // step.
                let vec_val = subbed(&sub, *value);
                new_insts.push(HirInstruction::VectorStore {
                    value: vec_val,
                    ptr: subbed(&sub, *ptr),
                    align: elem_size_bytes(&lane_ty) as u32,
                });
            }
            HirInstruction::Binary {
                op,
                result,
                ty,
                left,
                right,
            } => {
                // Skip the IV increment — we replace it at end.
                if *result == plan.iv.next {
                    continue;
                }
                if let Some(red_idx) = plan.reductions.iter().position(|r| r.next == *result) {
                    // Reduction update: lhs is the vector phi
                    // (already vector-typed), rhs is whatever was
                    // accumulated this iteration (must be a vector
                    // value via prior sub).
                    let red_ty = reduction_vec_ty[red_idx].clone();
                    new_insts.push(HirInstruction::Binary {
                        op: *op,
                        result: reduction_vec_next[red_idx],
                        ty: red_ty,
                        left: plan.reductions[red_idx].phi,
                        right: subbed(&sub, *right),
                    });
                    continue;
                }
                // Generic vector binary.
                let vec_ty = HirType::Vector(Box::new(ty.clone()), lanes);
                let new_res = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
                sub.insert(*result, new_res);
                new_insts.push(HirInstruction::Binary {
                    op: *op,
                    result: new_res,
                    ty: vec_ty,
                    left: subbed(&sub, *left),
                    right: subbed(&sub, *right),
                });
            }
            HirInstruction::Unary {
                op,
                result,
                ty,
                operand,
            } => {
                let vec_ty = HirType::Vector(Box::new(ty.clone()), lanes);
                let new_res = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
                sub.insert(*result, new_res);
                new_insts.push(HirInstruction::Unary {
                    op: *op,
                    result: new_res,
                    ty: vec_ty,
                    operand: subbed(&sub, *operand),
                });
            }
            HirInstruction::Cast {
                op,
                result,
                ty,
                operand,
            } => {
                let vec_ty = HirType::Vector(Box::new(ty.clone()), lanes);
                let new_res = create_value(func, vec_ty.clone(), HirValueKind::Instruction);
                sub.insert(*result, new_res);
                new_insts.push(HirInstruction::Cast {
                    op: *op,
                    result: new_res,
                    ty: vec_ty,
                    operand: subbed(&sub, *operand),
                });
            }
            _ => {
                // Should have been rejected at analysis time.
                debug_assert!(false, "unsupported inst at rewrite time: {:?}", inst);
            }
        }
    }

    // Append the IV increment: new_i_next = iv.phi + lanes
    new_insts.push(HirInstruction::Binary {
        op: BinaryOp::Add,
        result: new_i_next,
        ty: n_ty.clone(),
        left: plan.iv.phi,
        right: lanes_const,
    });

    // Install the new body.
    {
        let body_blk = func.blocks.get_mut(&plan.body).unwrap();
        body_blk.instructions = new_insts;
        body_blk.terminator = HirTerminator::Branch {
            target: plan.header,
        };
    }

    // ── 4. Materialise the scalar epilogue. Differs based on
    //       whether there are reductions or not:
    //         * No reductions → simple scalar_check → scalar_body → exit
    //         * With reductions → post_vec block reduces each vector
    //           accumulator to scalar via VectorHorizontalReduce,
    //           then the scalar_check phi is seeded with that.
    let post_vec_id = if !plan.reductions.is_empty() {
        Some(HirId::new())
    } else {
        None
    };
    let scalar_check = HirId::new();
    let scalar_body = HirId::new();

    // Update header's CondBranch false-target.
    {
        let header_blk = func.blocks.get_mut(&plan.header).unwrap();
        let new_false = post_vec_id.unwrap_or(scalar_check);
        header_blk.terminator = HirTerminator::CondBranch {
            condition: new_cmp_id,
            true_target: plan.body,
            false_target: new_false,
        };
    }

    // post_vec block: horizontally reduce each vector accumulator.
    let mut reduction_scalar_seeds: Vec<HirId> = Vec::with_capacity(plan.reductions.len());
    if let Some(pv) = post_vec_id {
        let mut pv_blk = HirBlock::new(pv);
        pv_blk.predecessors = vec![plan.header];
        pv_blk.successors = vec![scalar_check];
        for red in &plan.reductions {
            let seed = create_value(func, red.elem_ty.clone(), HirValueKind::Instruction);
            reduction_scalar_seeds.push(seed);
            pv_blk
                .instructions
                .push(HirInstruction::VectorHorizontalReduce {
                    result: seed,
                    ty: red.elem_ty.clone(),
                    vector: red.phi,
                    op: red.op,
                });
        }
        pv_blk.terminator = HirTerminator::Branch {
            target: scalar_check,
        };
        func.blocks.insert(pv, pv_blk);
    }

    // scalar_check block.
    let scalar_i = create_value(func, n_ty.clone(), HirValueKind::Instruction);
    let scalar_cond = create_value(func, HirType::Bool, HirValueKind::Instruction);
    let scalar_pred_in = post_vec_id.unwrap_or(plan.header);

    let mut scalar_check_blk = HirBlock::new(scalar_check);
    scalar_check_blk.predecessors = vec![scalar_pred_in, scalar_body];
    scalar_check_blk.successors = vec![scalar_body, plan.exit];
    scalar_check_blk.phis.push(HirPhi {
        result: scalar_i,
        ty: n_ty.clone(),
        incoming: vec![(plan.iv.phi, scalar_pred_in)],
    });

    // Reduction scalar phis (one per reduction).
    let mut scalar_reduction_phi: Vec<HirId> = Vec::with_capacity(plan.reductions.len());
    for (idx, red) in plan.reductions.iter().enumerate() {
        let scalar_acc = create_value(func, red.elem_ty.clone(), HirValueKind::Instruction);
        scalar_reduction_phi.push(scalar_acc);
        scalar_check_blk.phis.push(HirPhi {
            result: scalar_acc,
            ty: red.elem_ty.clone(),
            incoming: vec![(reduction_scalar_seeds[idx], scalar_pred_in)],
        });
    }
    scalar_check_blk.instructions.push(HirInstruction::Binary {
        op: BinaryOp::Lt,
        result: scalar_cond,
        ty: HirType::Bool,
        left: scalar_i,
        right: plan.n,
    });
    scalar_check_blk.terminator = HirTerminator::CondBranch {
        condition: scalar_cond,
        true_target: scalar_body,
        false_target: plan.exit,
    };
    func.blocks.insert(scalar_check, scalar_check_blk);

    // scalar_body — clone the original scalar instructions, with
    // operand substitutions.
    let mut sub2: IndexMap<HirId, HirId> = IndexMap::new();
    sub2.insert(plan.iv.phi, scalar_i);
    for (idx, red) in plan.reductions.iter().enumerate() {
        sub2.insert(red.phi, scalar_reduction_phi[idx]);
    }

    let mut scalar_body_blk = HirBlock::new(scalar_body);
    scalar_body_blk.predecessors = vec![scalar_check];
    scalar_body_blk.successors = vec![scalar_check];

    let mut scalar_i_next: Option<HirId> = None;
    let mut scalar_reduction_next: Vec<Option<HirId>> = vec![None; plan.reductions.len()];

    for inst in &plan.body_insts {
        if let Some(res) = instruction_result(inst) {
            let ty = func
                .values
                .get(&res)
                .map(|v| v.ty.clone())
                .unwrap_or(HirType::I64);
            // Reductions: tail's `next` value needs the scalar elem
            // type (NOT vector — the header phi up there is vector,
            // but here we're in scalar tail).
            let scalar_ty = if let Some(idx) = plan.reductions.iter().position(|r| r.next == res) {
                plan.reductions[idx].elem_ty.clone()
            } else {
                ty
            };
            let new_res = create_value(func, scalar_ty, HirValueKind::Instruction);
            sub2.insert(res, new_res);
            if res == plan.iv.next {
                scalar_i_next = Some(new_res);
            }
            if let Some(idx) = plan.reductions.iter().position(|r| r.next == res) {
                scalar_reduction_next[idx] = Some(new_res);
            }
        }
        let mut cloned = inst.clone();
        substitute_operands(&mut cloned, &sub2);
        scalar_body_blk.instructions.push(cloned);
    }
    scalar_body_blk.terminator = HirTerminator::Branch {
        target: scalar_check,
    };
    func.blocks.insert(scalar_body, scalar_body_blk);

    // Patch scalar_check's phis with body-incoming edges.
    {
        let sc = func.blocks.get_mut(&scalar_check).unwrap();
        if let Some(i_next) = scalar_i_next {
            for phi in &mut sc.phis {
                if phi.result == scalar_i {
                    phi.incoming.push((i_next, scalar_body));
                }
            }
        }
        for (idx, red_next) in scalar_reduction_next.iter().enumerate() {
            if let Some(n) = red_next {
                for phi in &mut sc.phis {
                    if phi.result == scalar_reduction_phi[idx] {
                        phi.incoming.push((*n, scalar_body));
                    }
                }
            }
        }
    }

    // Wire exit predecessors.
    if let Some(exit_blk) = func.blocks.get_mut(&plan.exit) {
        exit_blk.predecessors.retain(|p| *p != plan.header);
        exit_blk.predecessors.push(scalar_check);
    }

    // Suppress unused warnings on vec_ty / lane helper.
    let _ = vec_ty;
    let _ = reduction_phi_subs;
}

fn subbed(sub: &IndexMap<HirId, HirId>, id: HirId) -> HirId {
    sub.get(&id).copied().unwrap_or(id)
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

// ─── tests ────────────────────────────────────────────────────────────

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

    /// Build a SAXPY-shaped loop: `for i in 0..n { c[i] = op(a[i], b[i]) }`.
    fn build_saxpy(elem_ty: HirType, op: BinaryOp) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global("saxpy"), sig());
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
        let n = add_const(&mut f, n_ty.clone(), HirConstant::I64(128));
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

    /// Build `for i in 0..n { sum += op(a[i], b[i]) }`.
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
        let n = add_const(&mut f, n_ty.clone(), HirConstant::I64(128));
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

    /// Build a SAXPY loop but add a `Call(some_function)` in the body
    /// to trip the hazard check.
    fn build_saxpy_with_call() -> HirFunction {
        let mut f = build_saxpy(HirType::F32, BinaryOp::FAdd);
        // Find the body block (the one that's not entry/exit/header
        // — has a Store).
        let body_id = f
            .blocks
            .values()
            .find(|b| {
                b.instructions
                    .iter()
                    .any(|i| matches!(i, HirInstruction::Store { .. }))
            })
            .map(|b| b.id)
            .unwrap();
        let call_res = add_inst(&mut f, HirType::F32);
        let body_blk = f.blocks.get_mut(&body_id).unwrap();
        // Insert a call near the top.
        // Create a fake function id for the call.
        let fn_id = HirId::new();
        body_blk.instructions.insert(
            0,
            HirInstruction::Call {
                result: Some(call_res),
                callee: crate::hir::HirCallable::Function(fn_id),
                args: vec![],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        f
    }

    #[test]
    fn vectorizes_saxpy_shape_f32() {
        let mut f = build_saxpy(HirType::F32, BinaryOp::FAdd);
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 1, "{stats:?}");
        // Body should now contain VectorLoad/VectorStore.
        let body = f
            .blocks
            .values()
            .find(|b| {
                b.instructions
                    .iter()
                    .any(|i| matches!(i, HirInstruction::VectorStore { .. }))
            })
            .expect("vector body present");
        assert!(body
            .instructions
            .iter()
            .any(|i| matches!(i, HirInstruction::VectorLoad { .. })));
    }

    #[test]
    fn vectorizes_reduction_shape_f32_dot() {
        let mut f = build_reduction(HirType::F32, BinaryOp::FMul, BinaryOp::FAdd);
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 1, "{stats:?}");
        // Some block must contain VectorHorizontalReduce.
        let found = f.blocks.values().any(|b| {
            b.instructions
                .iter()
                .any(|i| matches!(i, HirInstruction::VectorHorizontalReduce { .. }))
        });
        assert!(found, "horizontal reduce should be emitted");
    }

    #[test]
    fn vectorizes_pure_elementwise_loop() {
        // Pure elementwise: `c[i] = a[i] * b[i] + a[i]`. This has an
        // extra arith op compared to plain SAXPY but should still
        // vectorize.
        let mut f = HirFunction::new(InternedString::new_global("elw"), sig());
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

        let elem_ty = HirType::F32;
        let n_ty = HirType::I64;
        let ptr_ty = HirType::Ptr(Box::new(elem_ty.clone()));
        let ptr_a = add_param(&mut f, ptr_ty.clone(), 0);
        let ptr_b = add_param(&mut f, ptr_ty.clone(), 1);
        let ptr_c = add_param(&mut f, ptr_ty.clone(), 2);
        let n = add_const(&mut f, n_ty.clone(), HirConstant::I64(128));
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
        let mul = add_inst(&mut f, elem_ty.clone());
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
            op: BinaryOp::FMul,
            result: mul,
            ty: elem_ty.clone(),
            left: va,
            right: vb,
        });
        body_blk.instructions.push(HirInstruction::Binary {
            op: BinaryOp::FAdd,
            result: vc,
            ty: elem_ty.clone(),
            left: mul,
            right: va,
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

        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 1, "{stats:?}");
    }

    #[test]
    fn rejects_loop_with_call_in_body() {
        let mut f = build_saxpy_with_call();
        let stats = run(&mut f);
        assert_eq!(stats.vectorized, 0, "{stats:?}");
        assert_eq!(stats.rejected_hazard, 1, "{stats:?}");
    }

    #[test]
    fn rejects_non_loop_function() {
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
