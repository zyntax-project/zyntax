//! Spread a loop whose iterations cannot interfere across cores.
//!
//! `parallel_safe` decides which loops may be spread and `zyntax_parallel_for`
//! runs the bands. This is the piece between them: it takes a loop the
//! analysis proved independent outright and rewrites it into a call that
//! hands bands of the range to the runtime.
//!
//! The loop leaves the function it was in. Its blocks move to a new one
//! taking `(lo, hi, env)`, running the original body over `[lo, hi)`
//! instead of the whole range, and the call site keeps only the values
//! the body needs, packed into a buffer the band reads back. Nothing is
//! renamed on the way: SSA values belong to a function, so a body moved
//! whole into a fresh function keeps every id it had, and only the three
//! things that genuinely changed are touched -- where the counter starts,
//! where it stops, and where the loop leaves.
//!
//! ## What is refused, and why
//!
//! * A loop the analysis would only offer against an obligation. What a
//!   caller has to establish is not something this can establish for it.
//! * A loop with no loop inside it. Measured on this machine at ten
//!   cores, spreading an elementwise pass over a large buffer returns
//!   1.1x to 2.4x because it is waiting on memory rather than on
//!   arithmetic, while a matrix multiply returns 4.3x at 512 and 6.0x at
//!   1024. Nested work is the cheap way to tell those apart, and paying
//!   a dispatch for the first kind would spend the win on overhead.
//! * A counter that does not step by one, or a loop whose shape is not
//!   a single entry, a single exit and a test at the top.
//! * A value the loop defines and something after it reads. The bands
//!   run in an order nobody chose, so there is no last iteration to take
//!   that value from.
//! * A captured value wider than a machine word, which the buffer this
//!   packs into has no slot for.
//!
//! The runtime declines a range too small to be worth splitting, so the
//! trip count is its decision rather than one made here, where it is
//! usually not known.
//!
//! Off unless `ZYNTAX_PARALLEL_LOOPS=1`.

use std::collections::HashSet;

use crate::analysis::{DominatorTree, LoopForest, NaturalLoop};
use crate::hir::{
    BinaryOp, CastOp, HirBlock, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirModule, HirParam, HirTerminator, HirType, HirValue, HirValueKind,
    ParamAttributes, ParamOwnership,
};
use zyntax_typed_ast::InternedString;

/// The runtime entry that hands bands of a range to worker threads.
const DISPATCH_SYMBOL: &str = "zyntax_parallel_for";

/// The smallest run of iterations worth handing to a worker, for the
/// only loops this dispatches: ones with a loop inside them. A row of a
/// matrix multiply is thousands of operations, so a few rows is already
/// more than the handover costs.
const NESTED_GRAIN: i64 = 4;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DispatchStats {
    /// Loops rewritten into a dispatch.
    pub dispatched: usize,
    /// Loops the analysis would only offer against an obligation.
    pub conditional: usize,
    /// Loops with no loop inside them, where a dispatch costs more than
    /// it returns.
    pub flat: usize,
    /// Loops whose shape this does not rewrite.
    pub shape: usize,
}

impl DispatchStats {
    fn add(&mut self, other: DispatchStats) {
        self.dispatched += other.dispatched;
        self.conditional += other.conditional;
        self.flat += other.flat;
        self.shape += other.shape;
    }
}

/// Whether the pass is switched on.
pub fn enabled() -> bool {
    std::env::var("ZYNTAX_PARALLEL_LOOPS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Everything one rewrite needs, gathered before anything is changed.
struct Plan {
    /// Blocks that move, header first in `header`.
    header: HirId,
    body: HashSet<HirId>,
    /// Where the loop was entered from, and where it leaves to.
    preheader: HirId,
    exit: HirId,
    /// The counter, what it starts at, and what it stops before.
    counter: HirId,
    lo: HirId,
    hi: HirId,
    /// The comparison at the top of the loop, whose right side becomes
    /// the band's upper bound.
    guard: HirId,
    /// Values the body reads that the loop does not define, in the order
    /// they are packed.
    captured: Vec<HirId>,
}

/// Run over every function, rewriting what qualifies.
pub fn run_module(module: &mut HirModule) -> DispatchStats {
    let mut stats = DispatchStats::default();
    if !enabled() {
        return stats;
    }
    // A rewrite adds a function, so the set to walk is fixed first.
    let ids: Vec<HirId> = module.functions.keys().copied().collect();
    let mut new_functions: Vec<HirFunction> = Vec::new();
    for id in ids {
        let Some(func) = module.functions.get(&id) else {
            continue;
        };
        let (plans, s) = plan_function(func);
        stats.add(s);
        if plans.is_empty() {
            continue;
        }
        let mut func = func.clone();
        for plan in plans {
            if let Some(band) = rewrite(&mut func, &plan) {
                new_functions.push(band);
                stats.dispatched += 1;
            }
        }
        module.functions.insert(id, func);
    }
    for band in new_functions {
        module.functions.insert(band.id, band);
    }
    stats
}

/// Which loops in one function qualify.
fn plan_function(func: &HirFunction) -> (Vec<Plan>, DispatchStats) {
    let mut stats = DispatchStats::default();
    if func.is_external || func.blocks.is_empty() {
        return (Vec::new(), stats);
    }
    let (independent, _) = crate::parallel_safe::analyze(func);
    if independent.is_empty() {
        return (Vec::new(), stats);
    }
    let dt = DominatorTree::new(func);
    let forest = LoopForest::detect(func, &dt);

    let mut plans = Vec::new();
    for found in &independent {
        if !found.is_unconditional() {
            stats.conditional += 1;
            continue;
        }
        let Some(lp) = forest.loops().iter().find(|l| l.header == found.header) else {
            stats.shape += 1;
            continue;
        };
        // Nested work is what makes the dispatch worth its cost.
        if !contains_inner_loop(&forest, lp) {
            stats.flat += 1;
            continue;
        }
        match plan_loop(func, lp, found.induction) {
            Some(p) => plans.push(p),
            None => stats.shape += 1,
        }
    }
    // Rewriting an outer loop moves an inner one with it, so only the
    // outermost of any nest is taken. Sorting by size and skipping
    // anything already inside a chosen one leaves exactly those.
    plans.sort_by_key(|p| std::cmp::Reverse(p.body.len()));
    let mut taken: Vec<Plan> = Vec::new();
    for plan in plans {
        if taken.iter().any(|t| t.body.contains(&plan.header)) {
            continue;
        }
        taken.push(plan);
    }
    (taken, stats)
}

/// Whether any other loop sits inside this one.
fn contains_inner_loop(forest: &LoopForest, lp: &NaturalLoop) -> bool {
    forest
        .loops()
        .iter()
        .any(|other| other.header != lp.header && lp.body.contains(&other.header))
}

/// Read one loop into a plan, or refuse it.
fn plan_loop(func: &HirFunction, lp: &NaturalLoop, counter: HirId) -> Option<Plan> {
    let header = func.blocks.get(&lp.header)?;

    // A test at the top, leaving the loop on the false edge.
    let HirTerminator::CondBranch {
        condition,
        true_target,
        false_target,
    } = &header.terminator
    else {
        return None;
    };
    if !lp.body.contains(true_target) || lp.body.contains(false_target) {
        return None;
    }
    let exit = *false_target;
    // And exactly one way out, so there is one place to resume.
    if lp.exits.len() != 1 || !lp.exits.contains(&exit) {
        return None;
    }

    // `counter < hi`, with `hi` computed before the loop.
    let HirInstruction::Binary {
        op: BinaryOp::Lt,
        left,
        right,
        ..
    } = header.instructions.iter().find_map(|i| match i {
        HirInstruction::Binary { result, .. } if *result == *condition => Some(i),
        _ => None,
    })?
    else {
        return None;
    };
    if *left != counter || defined_in(func, &lp.body, *right) {
        return None;
    }
    let hi = *right;

    // One way in, and the counter starting from a value computed there.
    let phi = header.phis.iter().find(|p| p.result == counter)?;
    if phi.incoming.len() != 2 {
        return None;
    }
    let (lo, preheader) = phi
        .incoming
        .iter()
        .find(|(_, b)| !lp.body.contains(b))
        .copied()?;
    let (next, _latch) = phi
        .incoming
        .iter()
        .find(|(_, b)| lp.body.contains(b))
        .copied()?;
    if header
        .predecessors
        .iter()
        .any(|p| *p != preheader && !lp.body.contains(p))
    {
        return None;
    }

    // Stepping by one, so a band of the range is a run of iterations.
    if !steps_by_one(func, &lp.body, counter, next) {
        return None;
    }

    // Nothing the loop computes may outlive it, since the bands finish
    // in no particular order.
    let defined = values_defined_in(func, &lp.body);
    if used_outside(func, &lp.body, &defined) {
        return None;
    }

    // What the body reads from outside has to travel with it.
    let mut captured: Vec<HirId> = Vec::new();
    let mut seen = HashSet::new();
    for b in &lp.body {
        let Some(blk) = func.blocks.get(b) else {
            continue;
        };
        let mut note = |v: HirId, captured: &mut Vec<HirId>| {
            if defined.contains(&v) || v == counter || !seen.insert(v) {
                return;
            }
            // A constant travels as itself; only what a caller computed
            // needs a slot.
            match func.values.get(&v).map(|x| &x.kind) {
                Some(HirValueKind::Parameter(_)) | Some(HirValueKind::Instruction) => {
                    captured.push(v)
                }
                _ => {}
            }
        };
        for phi in &blk.phis {
            for (v, _) in &phi.incoming {
                note(*v, &mut captured);
            }
        }
        for inst in &blk.instructions {
            for v in operands(inst) {
                note(v, &mut captured);
            }
        }
        for v in terminator_operands(&blk.terminator) {
            note(v, &mut captured);
        }
    }
    // `lo` reaches the band as a parameter, and `hi` as the other one,
    // but `hi` may also be read in the body, so it keeps its slot.
    captured.retain(|v| *v != lo);

    // The buffer holds one machine word per value.
    for v in &captured {
        let ty = func.values.get(v).map(|x| &x.ty)?;
        if !fits_a_word(ty) {
            return None;
        }
    }

    Some(Plan {
        header: lp.header,
        body: lp.body.clone(),
        preheader,
        exit,
        counter,
        lo,
        hi,
        guard: *condition,
        captured,
    })
}

/// Whether `next` is `counter + 1`.
fn steps_by_one(func: &HirFunction, body: &HashSet<HirId>, counter: HirId, next: HirId) -> bool {
    for b in body {
        let Some(blk) = func.blocks.get(b) else {
            continue;
        };
        for inst in &blk.instructions {
            if let HirInstruction::Binary {
                op: BinaryOp::Add,
                result,
                left,
                right,
                ..
            } = inst
            {
                if *result != next {
                    continue;
                }
                let other = if *left == counter {
                    *right
                } else if *right == counter {
                    *left
                } else {
                    return false;
                };
                return matches!(
                    func.values.get(&other).map(|v| &v.kind),
                    Some(HirValueKind::Constant(
                        HirConstant::I64(1) | HirConstant::I32(1)
                    ))
                );
            }
        }
    }
    false
}

/// A type the dispatch buffer has a slot for.
fn fits_a_word(ty: &HirType) -> bool {
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
            | HirType::Bool
            | HirType::Ptr(_)
            | HirType::Ref { .. }
    )
}

fn values_defined_in(func: &HirFunction, body: &HashSet<HirId>) -> HashSet<HirId> {
    let mut defined = HashSet::new();
    for b in body {
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

fn defined_in(func: &HirFunction, body: &HashSet<HirId>, value: HirId) -> bool {
    values_defined_in(func, body).contains(&value)
}

/// Whether anything outside the loop reads what the loop defined.
fn used_outside(func: &HirFunction, body: &HashSet<HirId>, defined: &HashSet<HirId>) -> bool {
    for (id, blk) in &func.blocks {
        if body.contains(id) {
            continue;
        }
        for phi in &blk.phis {
            if phi.incoming.iter().any(|(v, _)| defined.contains(v)) {
                return true;
            }
        }
        for inst in &blk.instructions {
            if operands(inst).iter().any(|v| defined.contains(v)) {
                return true;
            }
        }
        if terminator_operands(&blk.terminator)
            .iter()
            .any(|v| defined.contains(v))
        {
            return true;
        }
    }
    false
}

/// Every value an instruction reads.
///
/// Read off the debug rendering, which names every id the instruction
/// holds. Enumerating them by hand would mean a shape added later
/// quietly reading something this thinks it does not, and here that
/// would leave a value behind when the body moves.
fn operands(inst: &HirInstruction) -> Vec<HirId> {
    let mut found = ids_in(&format!("{inst:?}"));
    if let Some(r) = result_of(inst) {
        found.retain(|v| *v != r);
    }
    found
}

fn terminator_operands(term: &HirTerminator) -> Vec<HirId> {
    // Block labels are ids too, and naming one as a value is harmless:
    // a block is never in `values`, so it is never captured.
    ids_in(&format!("{term:?}"))
}

fn ids_in(text: &str) -> Vec<HirId> {
    let mut found = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find("HirId(") {
        rest = &rest[at + "HirId(".len()..];
        let Some(end) = rest.find(')') else { break };
        if let Ok(n) = rest[..end].parse::<u32>() {
            found.push(HirId::from_raw(n));
        }
        rest = &rest[end..];
    }
    found
}

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

/// Move the loop out and leave a dispatch behind.
fn rewrite(func: &mut HirFunction, plan: &Plan) -> Option<HirFunction> {
    let band = build_band(func, plan)?;
    install_dispatch(func, plan, band.id);
    Some(band)
}

fn new_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
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

/// The function the loop becomes: `band(lo, hi, env)`.
fn build_band(func: &HirFunction, plan: &Plan) -> Option<HirFunction> {
    let name = InternedString::new_global(&format!(
        "{}$band${}",
        func.name.resolve_global().unwrap_or_else(|| "fn".into()),
        format!("{:?}", plan.header)
            .trim_start_matches("HirId(")
            .trim_end_matches(')')
            .to_string()
    ));
    let sig = HirFunctionSignature {
        params: vec![
            word_param("lo", HirType::I64),
            word_param("hi", HirType::I64),
            word_param("env", HirType::Ptr(Box::new(HirType::I64))),
        ],
        returns: vec![],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    let mut band = HirFunction::new(name, sig);
    band.calling_convention = crate::hir::CallingConvention::C;

    // The body keeps every id it had, so the values it reads come across
    // whole. Ids belong to a function, so nothing can collide.
    for (id, value) in &func.values {
        band.values.insert(*id, value.clone());
    }

    let lo = new_value(&mut band, HirType::I64, HirValueKind::Parameter(0));
    let hi = new_value(&mut band, HirType::I64, HirValueKind::Parameter(1));
    let env = new_value(
        &mut band,
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Parameter(2),
    );
    band.signature.params[0].id = lo;
    band.signature.params[1].id = hi;
    band.signature.params[2].id = env;

    // Whatever the caller passed in arrives from the buffer instead, so
    // a value that was a parameter there is an instruction result here.
    for v in &plan.captured {
        if let Some(slot) = band.values.get_mut(v) {
            slot.kind = HirValueKind::Instruction;
        }
    }

    // Entry: read the captured values back, then fall into the loop.
    let entry = band.entry_block;
    let mut reads: Vec<HirInstruction> = Vec::new();
    for (slot, v) in plan.captured.iter().enumerate() {
        let ty = band.values.get(v)?.ty.clone();
        let index = new_value(
            &mut band,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(slot as i64)),
        );
        let addr = new_value(
            &mut band,
            HirType::Ptr(Box::new(HirType::I64)),
            HirValueKind::Instruction,
        );
        let typed = new_value(
            &mut band,
            HirType::Ptr(Box::new(ty.clone())),
            HirValueKind::Instruction,
        );
        reads.push(HirInstruction::GetElementPtr {
            result: addr,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            ptr: env,
            indices: vec![index],
        });
        reads.push(HirInstruction::Cast {
            op: CastOp::Bitcast,
            result: typed,
            ty: HirType::Ptr(Box::new(ty.clone())),
            operand: addr,
        });
        reads.push(HirInstruction::Load {
            result: *v,
            ty,
            ptr: typed,
            align: 8,
            volatile: false,
        });
    }
    {
        let blk = band.blocks.get_mut(&entry)?;
        blk.instructions = reads;
        blk.terminator = HirTerminator::Branch {
            target: plan.header,
        };
        blk.successors = vec![plan.header];
    }

    // The loop itself, verbatim.
    for b in &plan.body {
        let blk = func.blocks.get(b)?.clone();
        band.blocks.insert(*b, blk);
    }

    // Leaving the loop is leaving the function.
    let done = HirId::new();
    let mut done_blk = HirBlock::new(done);
    done_blk.terminator = HirTerminator::Return { values: vec![] };
    done_blk.predecessors = vec![plan.header];
    band.blocks.insert(done, done_blk);

    {
        let header = band.blocks.get_mut(&plan.header)?;
        // The counter starts where the band starts.
        for phi in &mut header.phis {
            if phi.result == plan.counter {
                for (value, from) in &mut phi.incoming {
                    if *from == plan.preheader {
                        *value = lo;
                        *from = entry;
                    }
                }
            }
        }
        // And stops where the band stops.
        for inst in &mut header.instructions {
            if let HirInstruction::Binary { result, right, .. } = inst {
                if *result == plan.guard {
                    *right = hi;
                }
            }
        }
        if let HirTerminator::CondBranch { false_target, .. } = &mut header.terminator {
            *false_target = done;
        }
        for p in &mut header.predecessors {
            if *p == plan.preheader {
                *p = entry;
            }
        }
        for s in &mut header.successors {
            if *s == plan.exit {
                *s = done;
            }
        }
    }
    Some(band)
}

fn word_param(name: &str, ty: HirType) -> HirParam {
    HirParam {
        id: HirId::new(),
        name: InternedString::new_global(name),
        ty,
        attributes: ParamAttributes::default(),
        ownership: ParamOwnership::Copied,
    }
}

/// Replace the loop with a call handing its range to the runtime.
fn install_dispatch(func: &mut HirFunction, plan: &Plan, band_id: HirId) {
    let slots = plan.captured.len().max(1);

    // The buffer is a frame of the caller, which is sound because the
    // dispatch does not return until every band has finished with it.
    let count = new_value(
        func,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(slots as i64)),
    );
    let env = new_value(
        func,
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    let entry = func.entry_block;
    if let Some(blk) = func.blocks.get_mut(&entry) {
        blk.instructions.insert(
            0,
            HirInstruction::Alloca {
                result: env,
                ty: HirType::I64,
                count: Some(count),
                align: 8,
            },
        );
    }

    // Pack what the body needs, then hand over the range.
    let mut writes: Vec<HirInstruction> = Vec::new();
    for (slot, v) in plan.captured.iter().enumerate() {
        let ty = match func.values.get(v) {
            Some(x) => x.ty.clone(),
            None => continue,
        };
        let index = new_value(
            func,
            HirType::I64,
            HirValueKind::Constant(HirConstant::I64(slot as i64)),
        );
        let addr = new_value(
            func,
            HirType::Ptr(Box::new(HirType::I64)),
            HirValueKind::Instruction,
        );
        let typed = new_value(
            func,
            HirType::Ptr(Box::new(ty.clone())),
            HirValueKind::Instruction,
        );
        writes.push(HirInstruction::GetElementPtr {
            result: addr,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            ptr: env,
            indices: vec![index],
        });
        writes.push(HirInstruction::Cast {
            op: CastOp::Bitcast,
            result: typed,
            ty: HirType::Ptr(Box::new(ty)),
            operand: addr,
        });
        writes.push(HirInstruction::Store {
            value: *v,
            ptr: typed,
            align: 8,
            volatile: false,
        });
    }

    let band_ptr = new_value(
        func,
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    writes.push(HirInstruction::Call {
        result: Some(band_ptr),
        callee: HirCallable::FuncRef(band_id),
        args: vec![],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    // One iteration of a loop with a loop inside it is already a
    // substantial amount of work, so a handful of them is worth a
    // worker. A range shorter than twice this runs where it is.
    let grain = new_value(
        func,
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(NESTED_GRAIN)),
    );
    writes.push(HirInstruction::Call {
        result: None,
        callee: HirCallable::Symbol(DISPATCH_SYMBOL.to_string()),
        args: vec![plan.lo, plan.hi, grain, band_ptr, env],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });

    if let Some(pre) = func.blocks.get_mut(&plan.preheader) {
        pre.instructions.extend(writes);
        pre.terminator = HirTerminator::Branch { target: plan.exit };
        pre.successors = vec![plan.exit];
    }

    // The loop is gone; what followed it now follows the dispatch.
    for b in &plan.body {
        func.blocks.shift_remove(b);
    }
    if let Some(exit) = func.blocks.get_mut(&plan.exit) {
        for p in &mut exit.predecessors {
            if *p == plan.header {
                *p = plan.preheader;
            }
        }
        exit.predecessors.retain(|p| !plan.body.contains(p));
        if !exit.predecessors.contains(&plan.preheader) {
            exit.predecessors.push(plan.preheader);
        }
        for phi in &mut exit.phis {
            for (_, from) in &mut phi.incoming {
                if *from == plan.header {
                    *from = plan.preheader;
                }
            }
        }
    }
}
