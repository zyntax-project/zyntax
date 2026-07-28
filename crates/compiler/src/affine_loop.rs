//! Closed-forming of affine reduction loops.
//!
//! Recognises a counted loop whose sole cross-iteration work is an
//! accumulator advanced by a loop-invariant amount, and replaces the
//! whole loop with its closed form. The canonical shape:
//!
//! ```text
//!     acc = 0.0; i = 0
//!     while i < N { acc = acc + b; i = i + step }   // b, N, step invariant
//! ```
//!
//! has closed form `acc_final = acc_init + T*b`, `i_final = i_init +
//! T*step`, where `T` is the (statically computed) trip count. Once we
//! prove the recurrence and the trip count, the loop body no longer
//! needs to run: the exit uses of `acc` / `i` are rewired to the
//! closed-form values and the loop blocks are deleted.
//!
//! The multiplicative coefficient on `acc` must be exactly 1 — this is
//! a *reduction* fold, not a general geometric-series solver. The
//! recurrence is accepted in whatever form the pipeline produces:
//!
//!   * `Add(acc, b)` / `FAdd(acc, b)`
//!   * `Fma(acc, one, b)` with `one == 1`     (post `fma_contract`)
//!   * `FAdd(FMul(acc, one), b)` with `one == 1`  (pre `fma_contract`)
//!
//! ## Soundness
//!
//! For an **integer** accumulator the closed form is always exact:
//! serial two's-complement addition `T` times equals `acc_init + T*b`
//! under wrapping arithmetic, so we compute it with wrapping ops.
//!
//! For a **float** accumulator, reassociating `T` roundings into one
//! multiply changes IEEE-754 results in general, so we fire ONLY when
//! the closed form is provably *bit-exact*. That holds when every
//! partial sum is an exactly-representable integer, which we guarantee
//! by requiring:
//!
//!   * `acc_init` is an integer-valued float constant (`v == trunc(v)`),
//!   * `b` is an integer-valued float constant,
//!   * `|acc_init| + T*|b|` ≤ 2^p, where `p` is the mantissa width
//!     (53 for f64, 24 for f32).
//!
//! Under those bounds every intermediate sum `acc_init + k*b`
//! (0 ≤ k ≤ T) is an integer with magnitude ≤ 2^p, hence exactly
//! representable, so serial addition and the closed form agree to the
//! bit. Anything outside these bounds bails. This is deliberately
//! narrower than a general fast-math reassociation — we introduce no
//! rounding the serial loop would not also produce.
//!
//! Because every input (init, step, bound, b) is required to be a
//! compile-time constant, the closed form is itself a constant, so the
//! transform emits new `Constant` SSA values rather than runtime
//! arithmetic. Constants are self-defining and dominate every use, so
//! rewiring can never break SSA dominance.
//!
//! ## Conservatism
//!
//! The recogniser bails (incrementing a `skipped_*` counter) unless
//! *every* precondition is provably met: a single latch, a single
//! exit block reached only from inside the loop, a two-block body
//! (`header` + `latch`), exactly two loop-carried phis (induction +
//! accumulator, identity phis for invariants excepted), a body free of
//! stores / calls / effects other than the recognised recurrence, and
//! a statically non-negative trip count. Anything unrecognised is left
//! untouched.

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule,
    HirTerminator, HirType, HirValue, HirValueKind, Intrinsic,
};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};

/// Per-run counters surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AffineLoopStats {
    /// Loops rewritten to their closed form.
    pub folded: usize,
    /// Natural loops the recogniser examined.
    pub loops_visited: usize,
    /// Skipped: CFG shape wasn't the simple counted-loop form.
    pub skipped_shape: usize,
    /// Skipped: no unique outside-the-loop predecessor (preheader).
    pub skipped_no_preheader: usize,
    /// Skipped: loop body contained work we don't recognise.
    pub skipped_unrecognized_body: usize,
    /// Skipped: trip count / bound wasn't a static non-negative count.
    pub skipped_trip_count: usize,
    /// Skipped: accumulator recurrence didn't match (coeff ≠ 1, etc.).
    pub skipped_recurrence: usize,
    /// Skipped: float closed form couldn't be proven bit-exact.
    pub skipped_float_inexact: usize,
}

/// Reason a loop was left untouched — maps 1:1 to a `skipped_*` stat.
enum Skip {
    Shape,
    NoPreheader,
    UnrecognizedBody,
    TripCount,
    Recurrence,
    FloatInexact,
}

/// Run the pass on every function in `module`.
pub fn run_module(module: &mut HirModule) -> AffineLoopStats {
    let mut stats = AffineLoopStats::default();
    // Bisection gate for a codegen-semantics-changing pass (mirrors
    // `ZYNTAX_DISABLE_FMA`): set `ZYNTAX_DISABLE_AFFINE=1` to leave every
    // loop intact when isolating a suspected miscompile.
    if std::env::var("ZYNTAX_DISABLE_AFFINE").is_ok() {
        return stats;
    }
    for func in module.functions.values_mut() {
        if func.is_external {
            continue;
        }
        let s = run_function(func);
        stats.folded += s.folded;
        stats.loops_visited += s.loops_visited;
        stats.skipped_shape += s.skipped_shape;
        stats.skipped_no_preheader += s.skipped_no_preheader;
        stats.skipped_unrecognized_body += s.skipped_unrecognized_body;
        stats.skipped_trip_count += s.skipped_trip_count;
        stats.skipped_recurrence += s.skipped_recurrence;
        stats.skipped_float_inexact += s.skipped_float_inexact;
    }
    stats
}

/// Run the pass on a single function. Folds every recognised affine
/// reduction loop, re-analysing after each fold (a fold rewrites the
/// CFG, so dominators / loops are recomputed from scratch).
pub fn run_function(func: &mut HirFunction) -> AffineLoopStats {
    let mut stats = AffineLoopStats::default();

    // Bounded outer loop: each successful fold deletes a loop, so the
    // count strictly decreases. The cap guards against any analysis
    // pathology leaving a loop "foldable" without shrinking the CFG.
    for _ in 0..256 {
        rebuild_cfg_edges(func);
        let dt = DominatorTree::new(func);
        let lf = LoopForest::detect(func, &dt);
        if lf.loops().is_empty() {
            break;
        }

        // Find the first foldable loop this round. We only touch one
        // per round so the mutated CFG is re-analysed cleanly before
        // considering the next (nested / sibling) loop.
        let mut folded_one = false;
        for lp in lf.loops() {
            stats.loops_visited += 1;
            match try_recognize(func, lp) {
                Ok(plan) => {
                    apply(func, lp, &plan);
                    stats.folded += 1;
                    folded_one = true;
                    break;
                }
                Err(reason) => {
                    bump_skip(&mut stats, reason);
                }
            }
        }
        if !folded_one {
            break;
        }
    }

    stats
}

fn bump_skip(stats: &mut AffineLoopStats, reason: Skip) {
    match reason {
        Skip::Shape => stats.skipped_shape += 1,
        Skip::NoPreheader => stats.skipped_no_preheader += 1,
        Skip::UnrecognizedBody => stats.skipped_unrecognized_body += 1,
        Skip::TripCount => stats.skipped_trip_count += 1,
        Skip::Recurrence => stats.skipped_recurrence += 1,
        Skip::FloatInexact => stats.skipped_float_inexact += 1,
    }
}

/// Everything the transform needs, computed during recognition.
struct FoldPlan {
    preheader: HirId,
    exit: HirId,
    ind_phi: HirId,
    acc_phi: HirId,
    i_final: HirConstant,
    i_final_ty: HirType,
    acc_final: HirConstant,
    acc_final_ty: HirType,
}

/// Attempt to recognise `lp` as an affine reduction loop. Returns the
/// `FoldPlan` on success or a `Skip` reason otherwise. Purely
/// analytical — mutates nothing.
fn try_recognize(func: &HirFunction, lp: &NaturalLoop) -> Result<FoldPlan, Skip> {
    // --- CFG shape: single latch, single exit, 2-block body ---------
    if lp.latches.len() != 1 {
        return Err(Skip::Shape);
    }
    let header = lp.header;
    let latch = lp.latches[0];
    if lp.exits.len() != 1 {
        return Err(Skip::Shape);
    }
    let exit = *lp.exits.iter().next().unwrap();
    if lp.body.len() != 2
        || !lp.body.contains(&header)
        || !lp.body.contains(&latch)
        || header == latch
    {
        return Err(Skip::UnrecognizedBody);
    }

    let preheader = unique_outside_predecessor(func, lp).ok_or(Skip::NoPreheader)?;
    // Preheader must branch unconditionally into the header so we can
    // redirect it straight to the exit.
    match &func.blocks[&preheader].terminator {
        HirTerminator::Branch { target } if *target == header => {}
        _ => return Err(Skip::Shape),
    }

    // Exit must be reached only from inside the loop; after we redirect
    // preheader → exit it then has the single predecessor `preheader`.
    let exit_block = func.blocks.get(&exit).ok_or(Skip::Shape)?;
    if exit_block.predecessors.is_empty()
        || !exit_block.predecessors.iter().all(|p| lp.body.contains(p))
    {
        return Err(Skip::Shape);
    }

    let header_block = &func.blocks[&header];
    let latch_block = &func.blocks[&latch];

    // Latch carries no phis (all loop-carried state lives in the header).
    if !latch_block.phis.is_empty() {
        return Err(Skip::UnrecognizedBody);
    }

    // --- Header terminator: cond-branch on `i < bound` --------------
    let (cond, t_target, f_target) = match &header_block.terminator {
        HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } => (*condition, *true_target, *false_target),
        _ => return Err(Skip::Shape),
    };
    // Continue-into-loop on true, exit on false (the `while i < N` shape).
    if !(lp.body.contains(&t_target) && f_target == exit) {
        return Err(Skip::Shape);
    }

    // Header contains exactly the compare defining `cond` — nothing else.
    if header_block.instructions.len() != 1 {
        return Err(Skip::UnrecognizedBody);
    }
    let (cmp_op, cmp_left, cmp_right) = match &header_block.instructions[0] {
        HirInstruction::Binary {
            op,
            left,
            right,
            result,
            ..
        } if *result == cond => (*op, *left, *right),
        _ => return Err(Skip::Shape),
    };
    let is_le = match cmp_op {
        BinaryOp::Lt => false,
        BinaryOp::Le => true,
        _ => return Err(Skip::Shape),
    };

    // --- Identity phis: header phis that merely carry an invariant --
    // e.g. `factor_h = phi(factor[preheader], factor[latch])`. They are
    // not cross-iteration state; record `phi_result → invariant value`
    // so operand checks resolve through them, and exclude them from the
    // induction/accumulator count.
    let invariant = loop_invariant_values(func, lp);
    let ident_map = identity_phi_map(header_block, &invariant);

    // The real loop-carried phis: induction + accumulator, exactly two.
    let carried: Vec<&crate::hir::HirPhi> = header_block
        .phis
        .iter()
        .filter(|p| !ident_map.contains_key(&p.result))
        .collect();
    if carried.len() != 2 {
        return Err(Skip::UnrecognizedBody);
    }

    // Induction phi is the compare's LHS; bound is the RHS constant.
    let ind_phi = cmp_left;
    let ind = carried
        .iter()
        .find(|p| p.result == ind_phi)
        .ok_or(Skip::Shape)?;
    let acc = carried.iter().find(|p| p.result != ind_phi).unwrap();
    let acc_phi = acc.result;
    let bound = resolve_const_int(func, cmp_right, &ident_map).ok_or(Skip::TripCount)?;

    // Each carried phi has exactly two incomings: init [preheader],
    // next [latch].
    let ind_init_id = phi_incoming(ind, preheader).ok_or(Skip::Shape)?;
    let ind_next_id = phi_incoming(ind, latch).ok_or(Skip::Shape)?;
    let acc_init_id = phi_incoming(acc, preheader).ok_or(Skip::Shape)?;
    let acc_next_id = phi_incoming(acc, latch).ok_or(Skip::Shape)?;
    if ind.incoming.len() != 2 || acc.incoming.len() != 2 {
        return Err(Skip::Shape);
    }

    // --- Induction: i_next = Add(i, step), step const int > 0 -------
    let ind_next_inst = find_def_in(latch_block, ind_next_id).ok_or(Skip::Recurrence)?;
    let step = match ind_next_inst {
        HirInstruction::Binary {
            op: BinaryOp::Add,
            left,
            right,
            ..
        } => {
            if *left == ind_phi {
                resolve_const_int(func, *right, &ident_map)
            } else if *right == ind_phi {
                resolve_const_int(func, *left, &ident_map)
            } else {
                None
            }
        }
        _ => None,
    }
    .ok_or(Skip::Recurrence)?;
    if step <= 0 {
        return Err(Skip::Recurrence);
    }
    let i_init = resolve_const_int(func, ind_init_id, &ident_map).ok_or(Skip::TripCount)?;

    // --- Accumulator: recognise the recurrence, coefficient == 1 ----
    let acc_ty = acc.ty.clone();
    let is_float = matches!(acc_ty, HirType::F32 | HirType::F64);
    let acc_next_inst = find_def_in(latch_block, acc_next_id).ok_or(Skip::Recurrence)?;
    // `allowed` = result ids that legitimately appear in the latch as
    // part of the recurrence (the recurrence op, plus a fused FMul).
    let mut allowed: HashSet<HirId> = HashSet::new();
    let b_id = recognize_recurrence(
        func,
        latch_block,
        acc_next_inst,
        acc_phi,
        is_float,
        &ident_map,
        &mut allowed,
    )?;
    allowed.insert(acc_next_id);
    allowed.insert(ind_next_id);

    // --- Body purity: latch holds only recognised recurrence work ---
    for inst in &latch_block.instructions {
        match instruction_result(inst) {
            Some(r) if allowed.contains(&r) => {}
            // Any instruction whose result isn't part of the recurrence
            // — or that has no result at all (Store / Fence / void Call
            // / effect ops) — means unmodelled work in the loop body.
            _ => return Err(Skip::UnrecognizedBody),
        }
    }
    match &latch_block.terminator {
        HirTerminator::Branch { target } if *target == header => {}
        _ => return Err(Skip::Shape),
    }

    // --- Trip count (i128) ------------------------------------------
    // lt: iterations until i >= bound.  le: until i > bound.
    let trip: i128 = if !is_le {
        if bound <= i_init {
            0
        } else {
            (bound - i_init + step - 1) / step
        }
    } else if bound < i_init {
        0
    } else {
        (bound - i_init) / step + 1
    };
    if trip < 0 {
        return Err(Skip::TripCount);
    }

    // i_final = i_init + trip*step (loop-exit value of the induction).
    let i_final_i128 = i_init + trip * step;
    let i_final = int_constant(&acc_ind_int_ty(&ind.ty), i_final_i128).ok_or(Skip::TripCount)?;

    // --- Closed-form accumulator ------------------------------------
    let acc_final = if is_float {
        let acc_init_f =
            resolve_const_float(func, acc_init_id, &ident_map).ok_or(Skip::FloatInexact)?;
        let b_f = resolve_const_float(func, b_id, &ident_map).ok_or(Skip::FloatInexact)?;
        // Bit-exactness gate.
        if !is_integer_valued_float(acc_init_f) || !is_integer_valued_float(b_f) {
            return Err(Skip::FloatInexact);
        }
        let mantissa_max: f64 = match acc_ty {
            HirType::F32 => (1u64 << 24) as f64,
            _ => (1u64 << 53) as f64,
        };
        let max_partial = acc_init_f.abs() + (trip as f64) * b_f.abs();
        if !(max_partial <= mantissa_max) {
            return Err(Skip::FloatInexact);
        }
        let final_f = acc_init_f + (trip as f64) * b_f;
        match acc_ty {
            HirType::F32 => HirConstant::F32(final_f as f32),
            _ => HirConstant::F64(final_f),
        }
    } else {
        let acc_init_i =
            resolve_const_int(func, acc_init_id, &ident_map).ok_or(Skip::Recurrence)?;
        let b_i = resolve_const_int(func, b_id, &ident_map).ok_or(Skip::Recurrence)?;
        // Wrapping closed form matches serial two's-complement addition.
        let final_i = wrapping_affine(&acc_ty, acc_init_i, trip, b_i).ok_or(Skip::Recurrence)?;
        int_constant(&acc_ty, final_i).ok_or(Skip::Recurrence)?
    };

    Ok(FoldPlan {
        preheader,
        exit,
        ind_phi,
        acc_phi,
        i_final,
        i_final_ty: ind.ty.clone(),
        acc_final,
        acc_final_ty: acc_ty,
    })
}

/// Apply the recognised fold: mint closed-form constants, rewire every
/// use of the induction / accumulator phis to them, redirect the
/// preheader straight to the exit, and delete the loop blocks.
fn apply(func: &mut HirFunction, lp: &NaturalLoop, plan: &FoldPlan) {
    // Mint constant SSA values for the closed-form results.
    let i_final_id = HirId::new();
    func.values.insert(
        i_final_id,
        HirValue {
            id: i_final_id,
            ty: plan.i_final_ty.clone(),
            kind: HirValueKind::Constant(plan.i_final.clone()),
            uses: Default::default(),
            span: None,
        },
    );
    let acc_final_id = HirId::new();
    func.values.insert(
        acc_final_id,
        HirValue {
            id: acc_final_id,
            ty: plan.acc_final_ty.clone(),
            kind: HirValueKind::Constant(plan.acc_final.clone()),
            uses: Default::default(),
            span: None,
        },
    );

    // Rewrite every use of the phis to the closed-form constants. Safe
    // everywhere — constants dominate all uses. In-loop uses are about
    // to be deleted; out-of-loop uses (the exit) now read the constant.
    let mut subs: IndexMap<HirId, HirId> = IndexMap::new();
    subs.insert(plan.ind_phi, i_final_id);
    subs.insert(plan.acc_phi, acc_final_id);
    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            inst.replace_uses(&subs);
        }
        block.terminator.replace_uses(&subs);
        for phi in &mut block.phis {
            for (val, _blk) in &mut phi.incoming {
                if let Some(&n) = subs.get(val) {
                    *val = n;
                }
            }
        }
    }

    // Exit phis had incomings labelled with loop blocks; after the
    // redirect their sole predecessor is the preheader.
    if let Some(exit_block) = func.blocks.get_mut(&plan.exit) {
        for phi in &mut exit_block.phis {
            for (_val, blk) in &mut phi.incoming {
                if lp.body.contains(blk) {
                    *blk = plan.preheader;
                }
            }
        }
        exit_block.predecessors = vec![plan.preheader];
    }

    // Redirect the preheader straight to the exit; the loop is now dead.
    if let Some(ph) = func.blocks.get_mut(&plan.preheader) {
        ph.terminator = HirTerminator::Branch { target: plan.exit };
        ph.successors = vec![plan.exit];
    }

    // Delete the loop blocks. Nothing references them any more.
    for b in &lp.body {
        func.blocks.shift_remove(b);
    }
}

// ─── recurrence recognition ─────────────────────────────────────────

/// Recognise `acc_next` as `acc * 1 + b` in one of the accepted forms.
/// Returns the addend `b`'s value id and records any fused-multiply
/// result id into `allowed` (so the body-purity check permits it).
fn recognize_recurrence(
    func: &HirFunction,
    latch: &crate::hir::HirBlock,
    acc_next: &HirInstruction,
    acc_phi: HirId,
    is_float: bool,
    ident_map: &HashMap<HirId, HirId>,
    allowed: &mut HashSet<HirId>,
) -> Result<HirId, Skip> {
    match acc_next {
        // acc + b  /  b + acc  (integer or float add)
        HirInstruction::Binary {
            op, left, right, ..
        } if is_add_op(*op, is_float) => {
            if *left == acc_phi {
                return Ok(*right);
            }
            if *right == acc_phi {
                return Ok(*left);
            }
            // FAdd(FMul(acc, one), b): one operand is a fused multiply
            // of the accumulator by a constant 1; the other is b.
            for (mul_side, b_side) in [(*left, *right), (*right, *left)] {
                if let Some(HirInstruction::Binary {
                    op: BinaryOp::FMul,
                    left: ml,
                    right: mr,
                    result: mres,
                    ..
                }) = find_def_in(latch, mul_side)
                {
                    let one_ok = (*ml == acc_phi && resolve_is_one(func, *mr, ident_map))
                        || (*mr == acc_phi && resolve_is_one(func, *ml, ident_map));
                    if one_ok {
                        allowed.insert(*mres);
                        return Ok(b_side);
                    }
                }
            }
            Err(Skip::Recurrence)
        }
        // Fma(a, m, c) == a*m + c. Coefficient 1 ⇒ {a,m} == {acc, 1}.
        HirInstruction::Call {
            callee: HirCallable::Intrinsic(Intrinsic::Fma),
            args,
            ..
        } if args.len() == 3 => {
            let (a, m, c) = (args[0], args[1], args[2]);
            if (a == acc_phi && resolve_is_one(func, m, ident_map))
                || (m == acc_phi && resolve_is_one(func, a, ident_map))
            {
                Ok(c)
            } else {
                Err(Skip::Recurrence)
            }
        }
        _ => Err(Skip::Recurrence),
    }
}

fn is_add_op(op: BinaryOp, is_float: bool) -> bool {
    if is_float {
        op == BinaryOp::FAdd
    } else {
        op == BinaryOp::Add
    }
}

// ─── constant helpers ───────────────────────────────────────────────

/// Chase `id` through the identity-phi map to its underlying value.
fn resolve(id: HirId, ident_map: &HashMap<HirId, HirId>) -> HirId {
    let mut cur = id;
    for _ in 0..16 {
        match ident_map.get(&cur) {
            Some(&next) if next != cur => cur = next,
            _ => break,
        }
    }
    cur
}

fn resolve_const_int(
    func: &HirFunction,
    id: HirId,
    ident_map: &HashMap<HirId, HirId>,
) -> Option<i128> {
    const_int(func, resolve(id, ident_map))
}

fn resolve_const_float(
    func: &HirFunction,
    id: HirId,
    ident_map: &HashMap<HirId, HirId>,
) -> Option<f64> {
    const_float(func, resolve(id, ident_map))
}

fn resolve_is_one(func: &HirFunction, id: HirId, ident_map: &HashMap<HirId, HirId>) -> bool {
    let r = resolve(id, ident_map);
    if let Some(v) = const_int(func, r) {
        return v == 1;
    }
    if let Some(v) = const_float(func, r) {
        return v == 1.0;
    }
    false
}

fn const_int(func: &HirFunction, id: HirId) -> Option<i128> {
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

fn const_float(func: &HirFunction, id: HirId) -> Option<f64> {
    match &func.values.get(&id)?.kind {
        HirValueKind::Constant(c) => match c {
            HirConstant::F32(x) => Some(*x as f64),
            HirConstant::F64(x) => Some(*x),
            _ => None,
        },
        _ => None,
    }
}

fn is_integer_valued_float(v: f64) -> bool {
    v.is_finite() && v == v.trunc()
}

/// The integer type to emit a closed-form integer constant of `ty`
/// into. For an integer induction/accumulator this is `ty` itself;
/// used only to keep the constant's type tag consistent.
fn acc_ind_int_ty(ty: &HirType) -> HirType {
    ty.clone()
}

/// Build a `HirConstant` of integer type `ty` from an i128, truncating
/// (wrapping) to the type width. Returns `None` for non-integer types.
fn int_constant(ty: &HirType, v: i128) -> Option<HirConstant> {
    Some(match ty {
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
        _ => return None,
    })
}

/// Compute `acc_init + trip*b` with wrapping arithmetic matching the
/// runtime's two's-complement width, returned as an i128 in the type's
/// value range.
fn wrapping_affine(ty: &HirType, acc_init: i128, trip: i128, b: i128) -> Option<i128> {
    macro_rules! do_signed {
        ($t:ty) => {{
            let a = acc_init as $t;
            let t = trip as $t;
            let bb = b as $t;
            Some(a.wrapping_add(t.wrapping_mul(bb)) as i128)
        }};
    }
    macro_rules! do_unsigned {
        ($t:ty) => {{
            let a = acc_init as $t;
            let t = trip as $t;
            let bb = b as $t;
            Some(a.wrapping_add(t.wrapping_mul(bb)) as i128)
        }};
    }
    match ty {
        HirType::I8 => do_signed!(i8),
        HirType::I16 => do_signed!(i16),
        HirType::I32 => do_signed!(i32),
        HirType::I64 => do_signed!(i64),
        HirType::I128 => Some(acc_init.wrapping_add(trip.wrapping_mul(b))),
        HirType::U8 => do_unsigned!(u8),
        HirType::U16 => do_unsigned!(u16),
        HirType::U32 => do_unsigned!(u32),
        HirType::U64 => do_unsigned!(u64),
        HirType::U128 => {
            let a = acc_init as u128;
            let t = trip as u128;
            let bb = b as u128;
            Some(a.wrapping_add(t.wrapping_mul(bb)) as i128)
        }
        _ => None,
    }
}

// ─── phi / block helpers ────────────────────────────────────────────

/// Value a phi carries in from predecessor `pred`, if present exactly.
fn phi_incoming(phi: &crate::hir::HirPhi, pred: HirId) -> Option<HirId> {
    let mut found = None;
    for (val, blk) in &phi.incoming {
        if *blk == pred {
            if found.is_some() {
                return None; // duplicate edge — bail
            }
            found = Some(*val);
        }
    }
    found
}

/// Find the instruction in `block` that defines `id`.
fn find_def_in(block: &crate::hir::HirBlock, id: HirId) -> Option<&HirInstruction> {
    block
        .instructions
        .iter()
        .find(|i| instruction_result(i) == Some(id))
}

/// Values that are loop-invariant for `lp`: constants / params /
/// globals / undef, plus any instruction or phi defined outside the
/// loop body.
fn loop_invariant_values(func: &HirFunction, lp: &NaturalLoop) -> HashSet<HirId> {
    let mut inv: HashSet<HirId> = HashSet::new();
    for (id, val) in &func.values {
        match val.kind {
            HirValueKind::Constant(_)
            | HirValueKind::Parameter(_)
            | HirValueKind::Global(_)
            | HirValueKind::Undef => {
                inv.insert(*id);
            }
            HirValueKind::Instruction => {}
        }
    }
    for (block_id, block) in &func.blocks {
        if lp.body.contains(block_id) {
            continue;
        }
        for inst in &block.instructions {
            if let Some(r) = instruction_result(inst) {
                inv.insert(r);
            }
        }
        for phi in &block.phis {
            inv.insert(phi.result);
        }
    }
    inv
}

/// Map each identity header-phi result to the invariant value it
/// always carries. An identity phi's non-self incomings are all equal
/// to a single loop-invariant value.
fn identity_phi_map(
    header: &crate::hir::HirBlock,
    invariant: &HashSet<HirId>,
) -> HashMap<HirId, HirId> {
    let mut map: HashMap<HirId, HirId> = HashMap::new();
    // A couple of passes let an identity phi that carries another
    // identity phi resolve transitively.
    for _ in 0..8 {
        let mut changed = false;
        for phi in &header.phis {
            if map.contains_key(&phi.result) {
                continue;
            }
            let mut consensus: Option<HirId> = None;
            let mut ok = true;
            for (val, _pred) in &phi.incoming {
                if *val == phi.result {
                    continue; // self-edge
                }
                let resolved = *map.get(val).unwrap_or(val);
                if !invariant.contains(&resolved) {
                    ok = false;
                    break;
                }
                match consensus {
                    None => consensus = Some(resolved),
                    Some(prev) if prev == resolved => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                if let Some(v) = consensus {
                    map.insert(phi.result, v);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    map
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

/// Header's unique outside-the-loop predecessor (the preheader), or
/// `None` when there is zero or more than one.
fn unique_outside_predecessor(func: &HirFunction, lp: &NaturalLoop) -> Option<HirId> {
    let header = func.blocks.get(&lp.header)?;
    let outside: Vec<HirId> = header
        .predecessors
        .iter()
        .copied()
        .filter(|p| !lp.body.contains(p))
        .collect();
    if outside.len() == 1 {
        Some(outside[0])
    } else {
        None
    }
}

/// Re-derive `successors` / `predecessors` from each block's
/// terminator. Same shape as `licm::rebuild_cfg_edges`; kept local to
/// avoid a cross-module dependency.
fn rebuild_cfg_edges(func: &mut HirFunction) {
    let mut succ_map: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for (&id, block) in &func.blocks {
        let succs = match &block.terminator {
            HirTerminator::Branch { target } => vec![*target],
            HirTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => vec![*true_target, *false_target],
            HirTerminator::Switch { default, cases, .. } => {
                let mut v = vec![*default];
                for (_, t) in cases {
                    v.push(*t);
                }
                v
            }
            HirTerminator::Invoke { normal, unwind, .. } => vec![*normal, *unwind],
            HirTerminator::PatternMatch { .. }
            | HirTerminator::Return { .. }
            | HirTerminator::Unreachable => vec![],
        };
        succ_map.insert(id, succs);
    }
    let mut pred_map: HashMap<HirId, Vec<HirId>> = HashMap::new();
    for (&src, succs) in &succ_map {
        for &t in succs {
            pred_map.entry(t).or_default().push(src);
        }
    }
    for (id, block) in func.blocks.iter_mut() {
        block.successors = succ_map.remove(id).unwrap_or_default();
        block.predecessors = pred_map.remove(id).unwrap_or_default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunctionSignature, HirPhi};
    use zyntax_typed_ast::InternedString;

    fn sig(ret: HirType) -> HirFunctionSignature {
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

    fn add_inst_val(f: &mut HirFunction, ty: HirType) -> HirId {
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

    /// Build the canonical counted-loop skeleton and return the ids the
    /// caller needs to fill in the recurrence:
    ///   entry(preheader) → header ⇄ latch ;  header → exit
    struct Skeleton {
        f: HirFunction,
        entry: HirId,
        header: HirId,
        latch: HirId,
        exit: HirId,
        ind_phi: HirId,
        acc_phi: HirId,
    }

    /// `acc_ty` selects int/float accumulator. `bound`/`i_init`/`step`
    /// are the induction constants (as i128); `acc_init` is added by the
    /// caller. Returns the skeleton with header phis + compare wired,
    /// leaving the latch recurrence for the caller.
    fn skeleton(acc_ty: HirType, i_init: HirId, bound: HirId, ind_ty: HirType) -> Skeleton {
        let mut f = HirFunction::new(InternedString::new_global("t"), sig(HirType::I64));
        let entry = HirId::new();
        let header = HirId::new();
        let latch = HirId::new();
        let exit = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        for id in [entry, header, latch, exit] {
            f.blocks.insert(id, HirBlock::new(id));
        }

        let ind_phi = add_inst_val(&mut f, ind_ty.clone());
        let acc_phi = add_inst_val(&mut f, acc_ty.clone());
        let cond = add_inst_val(&mut f, HirType::Bool);

        // entry: Branch header
        f.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Branch { target: header };

        // header: phis (filled with init now, latch value later), compare, condbranch
        {
            let h = f.blocks.get_mut(&header).unwrap();
            h.phis.push(HirPhi {
                result: ind_phi,
                ty: ind_ty.clone(),
                incoming: vec![(i_init, entry)], // latch edge appended by caller
            });
            h.phis.push(HirPhi {
                result: acc_phi,
                ty: acc_ty.clone(),
                incoming: vec![], // filled by caller (both edges)
            });
            h.instructions.push(HirInstruction::Binary {
                op: BinaryOp::Lt,
                result: cond,
                ty: HirType::Bool,
                left: ind_phi,
                right: bound,
            });
            h.terminator = HirTerminator::CondBranch {
                condition: cond,
                true_target: latch,
                false_target: exit,
            };
        }

        // latch terminator → header (instructions added by caller)
        f.blocks.get_mut(&latch).unwrap().terminator = HirTerminator::Branch { target: header };

        // exit: ret = acc_phi as i64 (reads the accumulator's exit value)
        let ret = add_inst_val(&mut f, HirType::I64);
        {
            let x = f.blocks.get_mut(&exit).unwrap();
            x.instructions.push(HirInstruction::Cast {
                op: crate::hir::CastOp::FpToSi,
                result: ret,
                ty: HirType::I64,
                operand: acc_phi,
            });
            x.terminator = HirTerminator::Return { values: vec![ret] };
        }

        Skeleton {
            f,
            entry,
            header,
            latch,
            exit,
            ind_phi,
            acc_phi,
        }
    }

    /// Wire the standard induction advance `i_next = i + step` into the
    /// latch and complete the induction phi's back-edge.
    fn wire_induction(s: &mut Skeleton, step: HirId, ind_ty: HirType) {
        let i_next = add_inst_val(&mut s.f, ind_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: i_next,
                ty: ind_ty,
                left: s.ind_phi,
                right: step,
            });
        // append back-edge to induction phi
        let ind_phi = s.ind_phi;
        let latch = s.latch;
        for phi in &mut s.f.blocks.get_mut(&s.header).unwrap().phis {
            if phi.result == ind_phi {
                phi.incoming.push((i_next, latch));
            }
        }
    }

    /// Complete the accumulator phi's incomings `(acc_init[entry],
    /// acc_next[latch])`.
    fn wire_acc_phi(s: &mut Skeleton, acc_init: HirId, acc_next: HirId) {
        let acc_phi = s.acc_phi;
        let (entry, latch) = (s.entry, s.latch);
        for phi in &mut s.f.blocks.get_mut(&s.header).unwrap().phis {
            if phi.result == acc_phi {
                phi.incoming = vec![(acc_init, entry), (acc_next, latch)];
            }
        }
    }

    fn exit_cast_operand(f: &HirFunction, exit: HirId) -> HirId {
        for inst in &f.blocks[&exit].instructions {
            if let HirInstruction::Cast { operand, .. } = inst {
                return *operand;
            }
        }
        panic!("no cast in exit");
    }

    // ── positive: float affine loop folds to exact closed form ──────

    #[test]
    fn folds_float_reduction() {
        // sum: f64 = 0; while i < 100 { sum = sum + 1.0; i += 1 }
        // closed form: sum = 100.0
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(100));
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        // carry the constants into the skeleton's function
        for id in [i_init, bound, step, acc_init, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        // sum_next = FAdd(sum, b)
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::FAdd,
                result: acc_next,
                ty: acc_ty.clone(),
                left: s.acc_phi,
                right: b,
            });
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 1, "float reduction should fold");
        // header/latch deleted
        assert!(!s.f.blocks.contains_key(&s.header));
        assert!(!s.f.blocks.contains_key(&s.latch));
        // exit cast now reads a constant == 100.0
        let op = exit_cast_operand(&s.f, s.exit);
        assert_eq!(const_float(&s.f, op), Some(100.0));
    }

    // ── positive: integer affine loop ───────────────────────────────

    #[test]
    fn folds_int_reduction() {
        // sum: i64 = 5; while i < 10 { sum = sum + 2; i += 1 } → 5+10*2=25
        let ind_ty = HirType::I64;
        let acc_ty = HirType::I64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(10));
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::I64(5));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::I64(2));

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::Add,
                result: acc_next,
                ty: acc_ty.clone(),
                left: s.acc_phi,
                right: b,
            });
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 1);
        let op = exit_cast_operand(&s.f, s.exit);
        assert_eq!(const_int(&s.f, op), Some(25));
    }

    // ── positive: Fma recurrence form (post fma_contract) ───────────

    #[test]
    fn folds_fma_form() {
        // sum = fma(sum, 1.0, 1.0); i<50 → sum = 50.0
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(50));
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let one = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, one, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Call {
                result: Some(acc_next),
                callee: HirCallable::Intrinsic(Intrinsic::Fma),
                args: vec![s.acc_phi, one, b],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 1);
        let op = exit_cast_operand(&s.f, s.exit);
        assert_eq!(const_float(&s.f, op), Some(50.0));
    }

    // ── negative: a Store in the body blocks the fold ───────────────

    #[test]
    fn bails_on_store_in_body() {
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(100));
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));
        let ptr = add_const(
            &mut f0,
            HirType::Ptr(Box::new(HirType::F64)),
            HirConstant::Null(HirType::Ptr(Box::new(HirType::F64))),
        );

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, b, ptr] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        {
            let l = s.f.blocks.get_mut(&s.latch).unwrap();
            l.instructions.push(HirInstruction::Binary {
                op: BinaryOp::FAdd,
                result: acc_next,
                ty: acc_ty.clone(),
                left: s.acc_phi,
                right: b,
            });
            // extra side effect: Store b -> ptr
            l.instructions.push(HirInstruction::Store {
                value: b,
                ptr,
                align: 8,
                volatile: false,
            });
        }
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 0);
        assert_eq!(stats.skipped_unrecognized_body, 1);
        assert!(s.f.blocks.contains_key(&s.header));
    }

    // ── negative: multiplicative coefficient ≠ 1 ────────────────────

    #[test]
    fn bails_on_non_unit_coefficient() {
        // sum = fma(sum, 2.0, 1.0) — coefficient 2, geometric not affine
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(50));
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let two = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(2.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, two, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Call {
                result: Some(acc_next),
                callee: HirCallable::Intrinsic(Intrinsic::Fma),
                args: vec![s.acc_phi, two, b],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            });
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 0);
        assert_eq!(stats.skipped_recurrence, 1);
    }

    // ── negative: a second accumulator update in the body ───────────

    #[test]
    fn bails_on_second_update() {
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(100));
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        // second (unrecognised) update: acc_next2 = acc_next + b, not fed back
        let acc_next2 = add_inst_val(&mut s.f, acc_ty.clone());
        {
            let l = s.f.blocks.get_mut(&s.latch).unwrap();
            l.instructions.push(HirInstruction::Binary {
                op: BinaryOp::FAdd,
                result: acc_next,
                ty: acc_ty.clone(),
                left: s.acc_phi,
                right: b,
            });
            l.instructions.push(HirInstruction::Binary {
                op: BinaryOp::FAdd,
                result: acc_next2,
                ty: acc_ty.clone(),
                left: acc_next,
                right: b,
            });
        }
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 0);
        assert_eq!(stats.skipped_unrecognized_body, 1);
    }

    // ── negative: non-constant (unknown) trip count ─────────────────

    #[test]
    fn bails_on_unknown_trip_count() {
        // bound is a Parameter, not a constant.
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = HirId::new();
        f0.values.insert(
            bound,
            HirValue {
                id: bound,
                ty: ind_ty.clone(),
                kind: HirValueKind::Parameter(0),
                uses: Default::default(),
                span: None,
            },
        );
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::FAdd,
                result: acc_next,
                ty: acc_ty.clone(),
                left: s.acc_phi,
                right: b,
            });
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 0);
        assert_eq!(stats.skipped_trip_count, 1);
    }

    // ── negative: non-integer float b (not bit-exact) ───────────────

    #[test]
    fn bails_on_non_integer_float_step() {
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(100));
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.5)); // not integer-valued

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::FAdd,
                result: acc_next,
                ty: acc_ty.clone(),
                left: s.acc_phi,
                right: b,
            });
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 0);
        assert_eq!(stats.skipped_float_inexact, 1);
    }

    // ── negative: oversized trip count breaks float exactness ───────

    #[test]
    fn bails_on_oversized_float_trip() {
        // b = 1.0 but trip > 2^53 → partial sums lose exactness.
        let ind_ty = HirType::I64;
        let acc_ty = HirType::F64;
        let mut f0 = HirFunction::new(InternedString::new_global("seed"), sig(HirType::I64));
        let i_init = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(0));
        let bound = add_const(
            &mut f0,
            ind_ty.clone(),
            HirConstant::I64((1i64 << 53) + 100),
        );
        let step = add_const(&mut f0, ind_ty.clone(), HirConstant::I64(1));
        let acc_init = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(0.0));
        let b = add_const(&mut f0, acc_ty.clone(), HirConstant::F64(1.0));

        let mut s = skeleton(acc_ty.clone(), i_init, bound, ind_ty.clone());
        for id in [i_init, bound, step, acc_init, b] {
            let v = f0.values.get(&id).unwrap().clone();
            s.f.values.insert(id, v);
        }
        wire_induction(&mut s, step, ind_ty.clone());
        let acc_next = add_inst_val(&mut s.f, acc_ty.clone());
        s.f.blocks
            .get_mut(&s.latch)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                op: BinaryOp::FAdd,
                result: acc_next,
                ty: acc_ty.clone(),
                left: s.acc_phi,
                right: b,
            });
        wire_acc_phi(&mut s, acc_init, acc_next);

        let stats = run_function(&mut s.f);
        assert_eq!(stats.folded, 0);
        assert_eq!(stats.skipped_float_inexact, 1);
    }
}
