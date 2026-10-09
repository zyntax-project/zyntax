//! Pending-error checks the error flag's state has already decided.
//!
//! A language that reports errors through a sticky global (see
//! `HirGlobal::error_flag`) tests it after every fallible call. The flag
//! is null at every function entry, only stores and calls change it, and
//! a callee's attributes say what a call does to it: `sets_error_flag`
//! leaves it set, `nothrow` leaves a null flag null, and a body that
//! stores nothing to it and calls only such bodies leaves it as found.
//! Those body summaries are read once per module ([`summaries`]) and
//! reused across rounds and compiles: they describe what a call does,
//! which no semantics-preserving pass changes.
//!
//! A forward dataflow over {Clear, Set, Unknown} gives each block its
//! entry state, a branch on the flag refining its two edges. Two
//! rewrites follow:
//!
//! - thread: a predecessor that reaches a check block with the flag Set
//!   jumps straight to the check's raising arm, when the check block does
//!   nothing else that the arm or anything after it needs;
//! - fold: a compare of the flag against null, where the state is Clear
//!   or Set, becomes a constant, and `const_fold` removes the branch.
//!
//! After an inlined callee whose only raising path is a cold call to a
//! raiser, the raiser's arm threads to the exit and the normal path's
//! check folds, so the load and branch leave the hot path.

use crate::analysis::DominatorTree;
use crate::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule,
    HirTerminator, HirType, HirValueKind, Intrinsic,
};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ErrorFlagStats {
    /// Compares of the flag made constant.
    pub folded: usize,
    /// Edges sent past a check block to its raising arm.
    pub threaded: usize,
}

/// What the flag holds at a point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// No path reaches the point.
    Unreached,
    Clear,
    Set,
    Unknown,
}

impl State {
    fn join(self, other: State) -> State {
        match (self, other) {
            (State::Unreached, s) | (s, State::Unreached) => s,
            (a, b) if a == b => a,
            _ => State::Unknown,
        }
    }

    /// The state on an edge that runs only when the flag is `known`.
    fn refine(self, known: State) -> State {
        match self {
            State::Unknown => known,
            s if s == known => s,
            _ => State::Unreached,
        }
    }
}

/// What executing an instruction does to the flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Effect {
    /// Leaves it as found.
    Keeps,
    /// Leaves a null flag null; says nothing of a set one.
    KeepsClear,
    /// Leaves it set.
    Sets,
    Unknown,
}

impl Effect {
    fn apply(self, s: State) -> State {
        match (self, s) {
            (_, State::Unreached) => State::Unreached,
            (Effect::Keeps, s) => s,
            (Effect::KeepsClear, State::Clear) => State::Clear,
            (Effect::Sets, _) => State::Set,
            _ => State::Unknown,
        }
    }
}

/// What calling each function of a module does to the error flag.
#[derive(Debug, Default)]
pub struct Summaries(HashMap<HirId, Effect>);

/// The module's error flag, unless it names none or the pass is off.
fn flag_of(module: &HirModule) -> Option<HirId> {
    // `ZYNTAX_DISABLE_ERROR_FLAG=1` keeps every error-flag check as
    // lowered; safe to run with.
    if std::env::var_os("ZYNTAX_DISABLE_ERROR_FLAG").is_some() {
        return None;
    }
    module
        .globals
        .iter()
        .find(|(_, g)| g.error_flag)
        .map(|(id, _)| *id)
}

/// Every function's effect on the flag, each body read once. A body
/// that keeps the flag by itself stops keeping it when a callee does
/// not, which a worklist carries back along the call edges.
pub fn summaries(module: &HirModule) -> Summaries {
    let Some(flag) = flag_of(module) else {
        return Summaries::default();
    };
    let demoted = |f: &HirFunction| {
        if f.attributes.nothrow {
            Effect::KeepsClear
        } else {
            Effect::Unknown
        }
    };
    let mut effects: HashMap<HirId, Effect> = HashMap::with_capacity(module.functions.len());
    let mut callers: HashMap<HirId, Vec<HirId>> = HashMap::new();
    let mut work: Vec<HirId> = Vec::new();
    for (&id, f) in &module.functions {
        let e = if f.attributes.sets_error_flag {
            Effect::Sets
        } else if f.blocks.is_empty() && !f.is_external {
            // An encoded body has no instructions here to summarise.
            demoted(f)
        } else if f.is_external {
            // A foreign `nothrow` body reaches no hook, so it has no way
            // to the flag.
            if f.attributes.nothrow {
                Effect::Keeps
            } else {
                Effect::Unknown
            }
        } else if body_keeps_locally(f, flag, |callee| {
            callers.entry(callee).or_default().push(id)
        }) {
            Effect::Keeps
        } else {
            demoted(f)
        };
        if e != Effect::Keeps {
            work.push(id);
        }
        effects.insert(id, e);
    }
    // A callee the module does not define keeps nothing.
    for (callee, list) in &callers {
        if !effects.contains_key(callee) {
            for c in list {
                if effects.get(c) == Some(&Effect::Keeps) {
                    effects.insert(*c, demoted(&module.functions[c]));
                    work.push(*c);
                }
            }
        }
    }
    while let Some(id) = work.pop() {
        let Some(list) = callers.get(&id) else {
            continue;
        };
        for c in list {
            if effects.get(c) == Some(&Effect::Keeps) {
                effects.insert(*c, demoted(&module.functions[c]));
                work.push(*c);
            }
        }
    }
    Summaries(effects)
}

/// Whether `f` keeps the flag provided every function it calls does,
/// naming each such callee to `callee`.
fn body_keeps_locally(f: &HirFunction, flag: HirId, mut callee: impl FnMut(HirId)) -> bool {
    let ptrs = flag_pointers(f, flag);
    if !ptrs.is_empty() && address_escapes(f, &ptrs) {
        return false;
    }
    let none = HashMap::new();
    let mut keeps = true;
    for b in f.blocks.values() {
        if matches!(b.terminator, HirTerminator::Invoke { .. }) {
            return false;
        }
        for inst in &b.instructions {
            match inst {
                HirInstruction::Store { ptr, .. } if ptrs.contains(ptr) => return false,
                HirInstruction::Call {
                    callee: HirCallable::Function(id),
                    ..
                } => callee(*id),
                _ => keeps &= effect_of(inst, &none) == Effect::Keeps,
            }
        }
    }
    keeps
}

/// Fold and thread the flag checks of the functions still to optimise,
/// calls read through `summaries` (built by [`summaries`] on this module
/// or one it was copied from).
pub fn run_module(module: &mut HirModule, summaries: &Summaries) -> ErrorFlagStats {
    let mut total = ErrorFlagStats::default();
    let Some(flag) = flag_of(module) else {
        return total;
    };
    // Only bodies that name the flag have a check to decide.
    let candidates: HashSet<HirId> = module
        .functions
        .iter()
        .filter(|(_, f)| {
            !f.attributes.optimized
                && !f.is_external
                && f.values
                    .values()
                    .any(|v| matches!(v.kind, HirValueKind::Global(g) if g == flag))
        })
        .map(|(id, _)| *id)
        .collect();
    if candidates.is_empty() {
        return total;
    }
    let effects = call_effects(module, &candidates, summaries);
    for func in module.functions_to_optimize() {
        if !candidates.contains(&func.id) {
            continue;
        }
        let s = run(func, flag, &effects);
        total.folded += s.folded;
        total.threaded += s.threaded;
    }
    total
}

/// What calling each function `bodies` call does to the flag: its
/// summary, or for a function added since the summaries were read, what
/// its attributes say.
fn call_effects(
    module: &HirModule,
    bodies: &HashSet<HirId>,
    summaries: &Summaries,
) -> HashMap<HirId, Effect> {
    let mut effects = HashMap::new();
    for id in bodies {
        for b in module.functions[id].blocks.values() {
            for inst in &b.instructions {
                if let HirInstruction::Call {
                    callee: HirCallable::Function(c),
                    ..
                } = inst
                {
                    effects.entry(*c).or_insert_with(|| {
                        if let Some(e) = summaries.0.get(c) {
                            return *e;
                        }
                        module.functions.get(c).map_or(Effect::Unknown, |f| {
                            if f.attributes.sets_error_flag {
                                Effect::Sets
                            } else if f.attributes.nothrow && f.is_external {
                                Effect::Keeps
                            } else if f.attributes.nothrow {
                                Effect::KeepsClear
                            } else {
                                Effect::Unknown
                            }
                        })
                    });
                }
            }
        }
    }
    effects
}

/// The effect of an instruction other than a store to the flag.
fn effect_of(inst: &HirInstruction, effects: &HashMap<HirId, Effect>) -> Effect {
    use HirInstruction as I;
    match inst {
        I::Call { callee, .. } => match callee {
            HirCallable::Function(id) => effects.get(id).copied().unwrap_or(Effect::Unknown),
            // A suspension or a destructor runs code this body cannot see.
            HirCallable::Intrinsic(Intrinsic::Yield | Intrinsic::Await | Intrinsic::Drop) => {
                Effect::Unknown
            }
            HirCallable::Intrinsic(_) => Effect::Keeps,
            _ => Effect::Unknown,
        },
        // The flag's address is never taken (checked per function), so
        // no store or read through another pointer reaches it.
        I::Binary { .. }
        | I::Unary { .. }
        | I::Alloca { .. }
        | I::Load { .. }
        | I::Store { .. }
        | I::GetElementPtr { .. }
        | I::Cast { .. }
        | I::Select { .. }
        | I::ExtractValue { .. }
        | I::InsertValue { .. }
        | I::Atomic { .. }
        | I::Fence { .. }
        | I::CreateUnion { .. }
        | I::GetUnionDiscriminant { .. }
        | I::ExtractUnionValue { .. }
        | I::CreateTraitObject { .. }
        | I::UpcastTraitObject { .. }
        | I::CreateClosure { .. }
        | I::CreateRef { .. }
        | I::Deref { .. }
        | I::Move { .. }
        | I::Copy { .. }
        | I::BeginLifetime { .. }
        | I::EndLifetime { .. }
        | I::LifetimeConstraint { .. }
        | I::VectorSplat { .. }
        | I::VectorExtractLane { .. }
        | I::VectorInsertLane { .. }
        | I::VectorHorizontalReduce { .. }
        | I::VectorLoad { .. }
        | I::VectorStore { .. }
        | I::VectorUnaryOp { .. }
        | I::VectorMinMax { .. }
        | I::VectorDot { .. } => Effect::Keeps,
        _ => Effect::Unknown,
    }
}

/// The values naming the flag's address.
fn flag_pointers(f: &HirFunction, flag: HirId) -> HashSet<HirId> {
    f.values
        .iter()
        .filter(|(_, v)| matches!(v.kind, HirValueKind::Global(g) if g == flag))
        .map(|(id, _)| *id)
        .collect()
}

/// Whether the flag's address is used other than to load or store it.
fn address_escapes(f: &HirFunction, ptrs: &HashSet<HirId>) -> bool {
    f.blocks.values().any(|b| {
        b.phis
            .iter()
            .any(|p| p.incoming.iter().any(|(v, _)| ptrs.contains(v)))
            || {
                let mut escapes = false;
                b.terminator
                    .for_each_operand(|id| escapes |= ptrs.contains(&id));
                escapes
            }
            || b.instructions.iter().any(|inst| match inst {
                HirInstruction::Load { .. } => false,
                HirInstruction::Store { value, .. } => ptrs.contains(value),
                _ => {
                    let mut escapes = false;
                    inst.for_each_operand(|id| escapes |= ptrs.contains(&id));
                    escapes
                }
            })
    })
}

fn is_null(f: &HirFunction, id: HirId) -> bool {
    matches!(
        f.values.get(&id).map(|v| &v.kind),
        Some(HirValueKind::Constant(
            HirConstant::Null(_)
                | HirConstant::I64(0)
                | HirConstant::U64(0)
                | HirConstant::USize(0)
                | HirConstant::ISize(0)
                | HirConstant::I32(0)
                | HirConstant::U32(0)
        ))
    )
}

/// One function's view of the flag.
struct Flow<'a> {
    ptrs: HashSet<HirId>,
    effects: &'a HashMap<HirId, Effect>,
    /// Flag loads.
    loads: HashSet<HirId>,
    /// Compare result -> (the flag load it tests, whether it is `eq`).
    compares: HashMap<HirId, (HirId, bool)>,
}

/// The dataflow's answer.
struct Solution {
    /// State on each edge.
    edges: HashMap<(HirId, HirId), State>,
    /// State at each flag load.
    at_load: HashMap<HirId, State>,
}

impl Flow<'_> {
    fn new<'a>(f: &HirFunction, flag: HirId, effects: &'a HashMap<HirId, Effect>) -> Flow<'a> {
        let ptrs = flag_pointers(f, flag);
        let mut loads = HashSet::new();
        for b in f.blocks.values() {
            for inst in &b.instructions {
                if let HirInstruction::Load { result, ptr, .. } = inst
                    && ptrs.contains(ptr)
                {
                    loads.insert(*result);
                }
            }
        }
        let mut compares = HashMap::new();
        for b in f.blocks.values() {
            for inst in &b.instructions {
                if let HirInstruction::Binary {
                    op: op @ (BinaryOp::Eq | BinaryOp::Ne),
                    result,
                    left,
                    right,
                    ..
                } = inst
                {
                    let load = if loads.contains(left) && is_null(f, *right) {
                        *left
                    } else if loads.contains(right) && is_null(f, *left) {
                        *right
                    } else {
                        continue;
                    };
                    compares.insert(*result, (load, *op == BinaryOp::Eq));
                }
            }
        }
        Flow {
            ptrs,
            effects,
            loads,
            compares,
        }
    }

    /// Run `block` from `state`: the state at its end, and the flag load
    /// that nothing after it in the block could have changed.
    fn transfer(
        &self,
        f: &HirFunction,
        block: HirId,
        mut state: State,
        mut at_load: Option<&mut HashMap<HirId, State>>,
    ) -> (State, Option<HirId>) {
        let mut last_load = None;
        for inst in &f.blocks[&block].instructions {
            match inst {
                HirInstruction::Load { result, .. } if self.loads.contains(result) => {
                    if let Some(at) = at_load.as_deref_mut() {
                        at.insert(*result, state);
                    }
                    last_load = Some(*result);
                }
                HirInstruction::Store { ptr, value, .. } if self.ptrs.contains(ptr) => {
                    state = if state == State::Unreached {
                        state
                    } else if is_null(f, *value) {
                        State::Clear
                    } else {
                        State::Unknown
                    };
                    last_load = None;
                }
                _ => {
                    let e = effect_of(inst, self.effects);
                    if e != Effect::Keeps {
                        state = e.apply(state);
                        last_load = None;
                    }
                }
            }
        }
        (state, last_load)
    }

    fn solve(&self, f: &HirFunction) -> Solution {
        let mut entry: HashMap<HirId, State> = HashMap::new();
        let mut edges: HashMap<(HirId, HirId), State> = HashMap::new();
        let rpo = &reverse_postorder(f);
        let mut changed = true;
        while changed {
            changed = false;
            for &b in rpo {
                let mut s = if b == f.entry_block {
                    State::Clear
                } else {
                    State::Unreached
                };
                for p in &f.blocks[&b].predecessors {
                    s = s.join(edges.get(&(*p, b)).copied().unwrap_or(State::Unreached));
                }
                entry.insert(b, s);
                let (out, last_load) = self.transfer(f, b, s, None);
                for (to, state) in self.out_edges(f, b, out, last_load) {
                    let old = edges.get(&(b, to)).copied().unwrap_or(State::Unreached);
                    let new = old.join(state);
                    if new != old {
                        edges.insert((b, to), new);
                        changed = true;
                    }
                }
            }
        }
        let mut at_load = HashMap::new();
        for &b in rpo {
            self.transfer(f, b, entry[&b], Some(&mut at_load));
        }
        Solution { edges, at_load }
    }

    /// The state along each edge out of `block`, given its end state.
    fn out_edges(
        &self,
        f: &HirFunction,
        block: HirId,
        out: State,
        last_load: Option<HirId>,
    ) -> Vec<(HirId, State)> {
        let term = &f.blocks[&block].terminator;
        if let HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = term
            && true_target != false_target
            && let Some(&(load, is_eq)) = self.compares.get(condition)
            && last_load == Some(load)
        {
            let (on_true, on_false) = if is_eq {
                (State::Clear, State::Set)
            } else {
                (State::Set, State::Clear)
            };
            return vec![
                (*true_target, out.refine(on_true)),
                (*false_target, out.refine(on_false)),
            ];
        }
        let out = if matches!(term, HirTerminator::Invoke { .. }) {
            Effect::Unknown.apply(out)
        } else {
            out
        };
        term.targets().into_iter().map(|t| (t, out)).collect()
    }
}

fn run(func: &mut HirFunction, flag: HirId, effects: &HashMap<HirId, Effect>) -> ErrorFlagStats {
    let mut stats = ErrorFlagStats::default();
    // Entered with the flag set on purpose: nothing holds at its entry.
    if func.is_external || func.attributes.queries_error_flag {
        return stats;
    }
    let ptrs = flag_pointers(func, flag);
    if ptrs.is_empty() || address_escapes(func, &ptrs) {
        return stats;
    }
    let flow = Flow::new(func, flag, effects);
    if flow.compares.is_empty() {
        return stats;
    }
    func.rebuild_cfg_edges();
    let mut sol = flow.solve(func);
    stats.threaded = thread(func, &flow, &sol);
    if stats.threaded > 0 {
        func.rebuild_cfg_edges();
        sol = flow.solve(func);
    }
    stats.folded = fold(func, &flow, &sol);
    stats
}

/// Blocks reachable from the entry, each before its successors except
/// along back edges.
fn reverse_postorder(f: &HirFunction) -> Vec<HirId> {
    let mut seen: HashSet<HirId> = HashSet::new();
    let mut post: Vec<HirId> = Vec::new();
    let mut stack: Vec<(HirId, usize)> = vec![(f.entry_block, 0)];
    seen.insert(f.entry_block);
    while let Some((b, i)) = stack.pop() {
        let succs = f.blocks.get(&b).map_or(&[][..], |bb| &bb.successors[..]);
        if let Some(&next) = succs.get(i) {
            stack.push((b, i + 1));
            if f.blocks.contains_key(&next) && seen.insert(next) {
                stack.push((next, 0));
            }
        } else {
            post.push(b);
        }
    }
    post.reverse();
    post
}

/// Send each predecessor that reaches a check block with the flag Set
/// straight to the check's raising arm.
fn thread(func: &mut HirFunction, flow: &Flow, sol: &Solution) -> usize {
    // (pred, check block, raising arm)
    let mut plan: Vec<(HirId, HirId, HirId)> = Vec::new();
    for (&c, block) in &func.blocks {
        let HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = block.terminator
        else {
            continue;
        };
        let Some(&(load, is_eq)) = flow.compares.get(&condition) else {
            continue;
        };
        let arm = if is_eq { false_target } else { true_target };
        if arm == c || true_target == false_target {
            continue;
        }
        let preds: Vec<HirId> = block
            .predecessors
            .iter()
            .copied()
            .filter(|p| sol.edges.get(&(*p, c)) == Some(&State::Set))
            .collect();
        if preds.is_empty() || !skippable_check(func, c, load) {
            continue;
        }
        if uses_escape_past(func, c, arm) {
            continue;
        }
        let arm_preds = &func.blocks[&arm].predecessors;
        for p in preds {
            if p == c || arm_preds.contains(&p) || plan.iter().any(|(q, _, _)| *q == p) {
                continue;
            }
            if !matches!(
                func.blocks[&p].terminator,
                HirTerminator::Branch { .. } | HirTerminator::CondBranch { .. }
            ) {
                continue;
            }
            plan.push((p, c, arm));
        }
    }
    if plan.is_empty() {
        return 0;
    }

    let saved = func.blocks.clone();
    let mut threaded = 0;
    for (p, c, arm) in plan {
        // The arm's phis take from `p` what they took from `c`, through
        // `c`'s own phis.
        let c_phis: HashMap<HirId, HirId> = func.blocks[&c]
            .phis
            .iter()
            .filter_map(|phi| {
                phi.incoming
                    .iter()
                    .find(|(_, from)| *from == p)
                    .map(|(v, _)| (phi.result, *v))
            })
            .collect();
        let c_defs: HashSet<HirId> = defs_of(func, c);
        let mut incoming = Vec::new();
        let mut ok = true;
        for phi in &func.blocks[&arm].phis {
            let Some(&(v, _)) = phi.incoming.iter().find(|(_, from)| *from == c) else {
                ok = false;
                break;
            };
            let v = match c_phis.get(&v) {
                Some(&from_p) => from_p,
                None if c_defs.contains(&v) => {
                    ok = false;
                    break;
                }
                None => v,
            };
            incoming.push(v);
        }
        if !ok {
            continue;
        }
        let arm_block = func.blocks.get_mut(&arm).expect("arm");
        for (phi, v) in arm_block.phis.iter_mut().zip(incoming) {
            phi.incoming.push((v, p));
        }
        let c_block = func.blocks.get_mut(&c).expect("check");
        for phi in &mut c_block.phis {
            phi.incoming.retain(|(_, from)| *from != p);
        }
        let p_block = func.blocks.get_mut(&p).expect("pred");
        match &mut p_block.terminator {
            HirTerminator::Branch { target } => *target = arm,
            HirTerminator::CondBranch {
                true_target,
                false_target,
                ..
            } => {
                if *true_target == c {
                    *true_target = arm;
                }
                if *false_target == c {
                    *false_target = arm;
                }
            }
            _ => unreachable!("planned only for branches"),
        }
        threaded += 1;
        func.rebuild_cfg_edges();
    }
    if threaded > 0 && !dominance_holds(func) {
        func.blocks = saved;
        return 0;
    }
    threaded
}

/// The values `block` defines.
fn defs_of(func: &HirFunction, block: HirId) -> HashSet<HirId> {
    let b = &func.blocks[&block];
    b.phis
        .iter()
        .map(|p| p.result)
        .chain(b.instructions.iter().filter_map(|i| i.result_id()))
        .collect()
}

/// Whether the check block `c` can be skipped: it does nothing but
/// compute values, and nothing before `load` touches the flag.
fn skippable_check(func: &HirFunction, c: HirId, load: HirId) -> bool {
    let mut seen_load = false;
    for inst in &func.blocks[&c].instructions {
        match inst {
            HirInstruction::Load {
                result, volatile, ..
            } => {
                if *volatile {
                    return false;
                }
                if *result == load {
                    seen_load = true;
                }
            }
            HirInstruction::Binary { .. }
            | HirInstruction::Unary { .. }
            | HirInstruction::Cast { .. }
            | HirInstruction::GetElementPtr { .. }
            | HirInstruction::Select { .. }
            | HirInstruction::ExtractValue { .. }
            | HirInstruction::InsertValue { .. } => {}
            _ => return false,
        }
    }
    seen_load
}

/// Whether a value `c` defines is used on a path from `arm` that does
/// not pass through `c`, which a threaded edge would reach without it.
fn uses_escape_past(func: &HirFunction, c: HirId, arm: HirId) -> bool {
    let defs = defs_of(func, c);
    let mut reach: HashSet<HirId> = HashSet::new();
    let mut stack = vec![arm];
    while let Some(b) = stack.pop() {
        if b == c || !reach.insert(b) {
            continue;
        }
        if let Some(block) = func.blocks.get(&b) {
            stack.extend(block.successors.iter().copied());
        }
    }
    reach.iter().any(|b| {
        let block = &func.blocks[b];
        let mut used = false;
        for inst in &block.instructions {
            inst.for_each_operand(|id| used |= defs.contains(&id));
        }
        block
            .terminator
            .for_each_operand(|id| used |= defs.contains(&id));
        used || block.phis.iter().any(|p| {
            p.incoming
                .iter()
                .any(|(v, from)| defs.contains(v) && reach.contains(from))
        })
    })
}

/// Every use in a reachable block is dominated by its definition.
fn dominance_holds(func: &HirFunction) -> bool {
    let dt = DominatorTree::new(func);
    let mut def_at: HashMap<HirId, (HirId, usize)> = HashMap::new();
    for (&b, block) in &func.blocks {
        for p in &block.phis {
            def_at.insert(p.result, (b, 0));
        }
        for (i, inst) in block.instructions.iter().enumerate() {
            if let Some(r) = inst.result_id() {
                def_at.insert(r, (b, i + 1));
            }
        }
    }
    let reachable: HashSet<HirId> = dt.rpo().iter().copied().collect();
    let ok = |id: HirId, b: HirId, pos: usize| match def_at.get(&id) {
        None => true,
        Some(&(db, dpos)) => {
            if db == b {
                dpos < pos
            } else {
                dt.dominates(db, b)
            }
        }
    };
    for &b in &reachable {
        let block = &func.blocks[&b];
        for p in &block.phis {
            for (v, from) in &p.incoming {
                if reachable.contains(from) && !ok(*v, *from, usize::MAX) {
                    return false;
                }
            }
        }
        for (i, inst) in block.instructions.iter().enumerate() {
            let mut good = true;
            inst.for_each_operand(|id| good &= ok(id, b, i + 1));
            if !good {
                return false;
            }
        }
        let mut good = true;
        block
            .terminator
            .for_each_operand(|id| good &= ok(id, b, usize::MAX));
        if !good {
            return false;
        }
    }
    true
}

/// Make each compare of the flag against null whose state is known a
/// constant, and drop the loads that leaves unused.
fn fold(func: &mut HirFunction, flow: &Flow, sol: &Solution) -> usize {
    let mut decided: Vec<(HirId, bool)> = Vec::new();
    for (&cmp, &(load, is_eq)) in &flow.compares {
        let value = match sol.at_load.get(&load) {
            Some(State::Clear) => is_eq,
            Some(State::Set) => !is_eq,
            _ => continue,
        };
        decided.push((cmp, value));
    }
    if decided.is_empty() {
        return 0;
    }
    let mut constants: [Option<HirId>; 2] = [None, None];
    let mut subs: HashMap<HirId, HirId> = HashMap::new();
    for (cmp, value) in &decided {
        let c = *constants[*value as usize].get_or_insert_with(|| {
            func.create_value(
                HirType::Bool,
                HirValueKind::Constant(HirConstant::Bool(*value)),
            )
        });
        subs.insert(*cmp, c);
    }
    crate::cse::apply_substitutions_public(func, &subs);
    crate::cse::remove_redundant_instructions_public(func, &subs);

    // A branch on a decided compare is a jump, made here so that the
    // round's cfg_simplify finishes the job without another round.
    let mut jumps: Vec<(HirId, HirId, HirId)> = Vec::new();
    for (&id, block) in &func.blocks {
        if let HirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = block.terminator
            && let Some(value) = constants.iter().position(|c| *c == Some(condition))
        {
            let (taken, dropped) = if value == 1 {
                (true_target, false_target)
            } else {
                (false_target, true_target)
            };
            jumps.push((id, taken, dropped));
        }
    }
    for (id, taken, dropped) in jumps {
        func.blocks[&id].terminator = HirTerminator::Branch { target: taken };
        if dropped != taken
            && let Some(other) = func.blocks.get_mut(&dropped)
        {
            for phi in &mut other.phis {
                phi.incoming.retain(|(_, from)| *from != id);
            }
        }
    }
    func.rebuild_cfg_edges();

    // Loads whose every compare folded.
    let mut used: HashSet<HirId> = HashSet::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            inst.for_each_operand(|id| {
                used.insert(id);
            });
        }
        block.terminator.for_each_operand(|id| {
            used.insert(id);
        });
        for p in &block.phis {
            used.extend(p.incoming.iter().map(|(v, _)| *v));
        }
    }
    for block in func.blocks.values_mut() {
        block.instructions.retain(|inst| match inst {
            HirInstruction::Load { result, .. } => {
                !flow.loads.contains(result) || used.contains(result)
            }
            _ => true,
        });
    }
    decided.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        FunctionAttributes, HirBlock, HirFunctionSignature, HirGlobal, HirPhi, HirValue, Linkage,
        Visibility,
    };
    use zyntax_typed_ast::InternedString;

    fn flag_ty() -> HirType {
        HirType::Ptr(Box::new(HirType::I8))
    }

    fn sig(params: Vec<HirType>) -> HirFunctionSignature {
        HirFunctionSignature {
            params: params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| crate::hir::HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global(&format!("p{i}")),
                    ty,
                    attributes: Default::default(),
                    ownership: Default::default(),
                })
                .collect(),
            returns: vec![HirType::I64],
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

    struct Module {
        module: HirModule,
        flag: HirId,
    }

    impl Module {
        fn new(with_flag: bool) -> Self {
            let mut module = HirModule::new(InternedString::new_global("m"));
            let flag = HirId::new();
            module.globals.insert(
                flag,
                HirGlobal {
                    id: flag,
                    name: InternedString::new_global("flag"),
                    ty: flag_ty(),
                    initializer: None,
                    is_const: false,
                    is_thread_local: false,
                    linkage: Linkage::Internal,
                    visibility: Visibility::Default,
                    error_flag: with_flag,
                },
            );
            Module { module, flag }
        }

        /// A bodiless callee with `attributes`.
        fn callee(&mut self, name: &str, attributes: FunctionAttributes) -> HirId {
            let mut f = HirFunction::new(InternedString::new_global(name), sig(vec![]));
            f.is_external = true;
            f.attributes = attributes;
            let id = f.id;
            self.module.functions.insert(id, f);
            id
        }
    }

    /// A function under construction.
    struct Body {
        f: HirFunction,
        flag_ptr: HirId,
        null: HirId,
    }

    impl Body {
        fn new(m: &Module) -> Self {
            let mut f = HirFunction::new(
                InternedString::new_global("caller"),
                sig(vec![HirType::Bool, HirType::Ptr(Box::new(HirType::I64))]),
            );
            f.blocks.clear();
            let flag_ptr = f.create_value(
                HirType::Ptr(Box::new(flag_ty())),
                HirValueKind::Global(m.flag),
            );
            let null = f.create_value(flag_ty(), HirValueKind::Constant(HirConstant::I64(0)));
            Body { f, flag_ptr, null }
        }

        fn param(&mut self, i: u32, ty: HirType) -> HirId {
            let id = HirId::new();
            self.f.values.insert(
                id,
                HirValue {
                    id,
                    ty,
                    kind: HirValueKind::Parameter(i),
                    uses: Default::default(),
                    span: None,
                },
            );
            id
        }

        fn int(&mut self, v: i64) -> HirId {
            self.f
                .create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(v)))
        }

        fn block(&mut self) -> HirId {
            let id = HirId::new();
            let mut b = HirBlock::new(id);
            b.terminator = HirTerminator::Unreachable;
            self.f.blocks.insert(id, b);
            if self.f.blocks.len() == 1 {
                self.f.entry_block = id;
            }
            id
        }

        fn push(&mut self, b: HirId, inst: HirInstruction) {
            self.f.blocks[&b].instructions.push(inst);
        }

        fn term(&mut self, b: HirId, t: HirTerminator) {
            self.f.blocks[&b].terminator = t;
        }

        fn call(&mut self, b: HirId, callee: HirId) {
            self.push(
                b,
                HirInstruction::Call {
                    result: None,
                    callee: HirCallable::Function(callee),
                    args: vec![],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                },
            );
        }

        /// `load flag; ne null; brcond -> raise, ok`, returning the compare.
        fn check(&mut self, b: HirId, raise: HirId, ok: HirId) -> HirId {
            let load = self.f.create_value(flag_ty(), HirValueKind::Instruction);
            let cmp = self
                .f
                .create_value(HirType::Bool, HirValueKind::Instruction);
            let ptr = self.flag_ptr;
            let null = self.null;
            self.push(
                b,
                HirInstruction::Load {
                    result: load,
                    ty: flag_ty(),
                    ptr,
                    align: 8,
                    volatile: false,
                },
            );
            self.push(
                b,
                HirInstruction::Binary {
                    op: BinaryOp::Ne,
                    result: cmp,
                    ty: HirType::Bool,
                    left: load,
                    right: null,
                },
            );
            self.term(
                b,
                HirTerminator::CondBranch {
                    condition: cmp,
                    true_target: raise,
                    false_target: ok,
                },
            );
            cmp
        }

        fn ret(&mut self, b: HirId, v: i64) {
            let v = self.int(v);
            self.term(b, HirTerminator::Return { values: vec![v] });
        }

        fn finish(mut self, m: &mut Module) -> HirId {
            self.f.rebuild_cfg_edges();
            let id = self.f.id;
            m.module.functions.insert(id, self.f);
            id
        }
    }

    fn run_all(module: &mut HirModule) -> ErrorFlagStats {
        let s = summaries(module);
        run_module(module, &s)
    }

    fn body(m: &Module, id: HirId) -> &HirFunction {
        &m.module.functions[&id]
    }

    /// Whether the check ending `block` was decided: `Some(true)` when
    /// it now jumps to `raise`, `Some(false)` when it jumps elsewhere.
    fn branch_constant(f: &HirFunction, block: HirId, raise: HirId) -> Option<bool> {
        match f.blocks[&block].terminator {
            HirTerminator::Branch { target } => Some(target == raise),
            _ => None,
        }
    }

    fn flag_loads(f: &HirFunction, block: HirId) -> usize {
        f.blocks[&block]
            .instructions
            .iter()
            .filter(|i| matches!(i, HirInstruction::Load { ty, .. } if *ty == flag_ty()))
            .count()
    }

    /// entry: call callee; check -> raise, ok.
    fn after_call(attributes: FunctionAttributes) -> (Module, HirId, HirId, HirId) {
        let mut m = Module::new(true);
        let callee = m.callee("callee", attributes);
        let mut b = Body::new(&m);
        let entry = b.block();
        let raise = b.block();
        let ok = b.block();
        b.call(entry, callee);
        b.check(entry, raise, ok);
        b.ret(raise, -1);
        b.ret(ok, 0);
        let f = b.finish(&mut m);
        run_all(&mut m.module);
        (m, f, entry, raise)
    }

    #[test]
    fn the_check_after_a_nothrow_call_folds_to_no_error() {
        let (m, f, entry, raise) = after_call(FunctionAttributes {
            nothrow: true,
            ..Default::default()
        });
        assert_eq!(branch_constant(body(&m, f), entry, raise), Some(false));
        assert_eq!(flag_loads(body(&m, f), entry), 0);
    }

    #[test]
    fn the_check_after_a_raiser_takes_the_raising_arm() {
        let (m, f, entry, raise) = after_call(FunctionAttributes {
            sets_error_flag: true,
            ..Default::default()
        });
        assert_eq!(branch_constant(body(&m, f), entry, raise), Some(true));
    }

    #[test]
    fn the_check_after_a_call_that_may_raise_is_kept() {
        let (m, f, entry, raise) = after_call(FunctionAttributes::default());
        assert_eq!(branch_constant(body(&m, f), entry, raise), None);
        assert_eq!(flag_loads(body(&m, f), entry), 1);
    }

    #[test]
    fn an_encoded_callee_uses_its_contract_until_its_body_is_read() {
        for (attributes, expected) in [
            (FunctionAttributes::default(), None),
            (
                FunctionAttributes {
                    nothrow: true,
                    ..Default::default()
                },
                Some(false),
            ),
            (
                FunctionAttributes {
                    sets_error_flag: true,
                    ..Default::default()
                },
                Some(true),
            ),
        ] {
            let mut m = Module::new(true);
            let callee = m.callee("encoded", attributes);
            let encoded = m.module.functions.get_mut(&callee).unwrap();
            encoded.is_external = false;
            encoded.attributes.deferred = true;
            encoded.blocks.clear();
            let mut b = Body::new(&m);
            let entry = b.block();
            let raise = b.block();
            let ok = b.block();
            b.call(entry, callee);
            b.check(entry, raise, ok);
            b.ret(raise, -1);
            b.ret(ok, 0);
            let f = b.finish(&mut m);
            run_all(&mut m.module);
            assert_eq!(branch_constant(body(&m, f), entry, raise), expected);
        }
    }

    /// A body that raises and catches its own error is neither a raiser
    /// nor free of the flag: the check after it stays.
    #[test]
    fn a_callee_that_stores_the_flag_keeps_the_callers_check() {
        let mut m = Module::new(true);
        let mut inner = Body::new(&m);
        let e = inner.block();
        let one = inner.int(1);
        let (ptr, null) = (inner.flag_ptr, inner.null);
        inner.push(
            e,
            HirInstruction::Store {
                value: one,
                ptr,
                align: 8,
                volatile: false,
            },
        );
        inner.push(
            e,
            HirInstruction::Store {
                value: null,
                ptr,
                align: 8,
                volatile: false,
            },
        );
        inner.ret(e, 0);
        let catches = inner.finish(&mut m);
        let mut b = Body::new(&m);
        let entry = b.block();
        let raise = b.block();
        let ok = b.block();
        b.call(entry, catches);
        b.check(entry, raise, ok);
        b.ret(raise, -1);
        b.ret(ok, 0);
        let f = b.finish(&mut m);
        run_all(&mut m.module);
        assert_eq!(branch_constant(body(&m, f), entry, raise), None);
    }

    /// A store through another pointer between two checks leaves the
    /// flag as it was: the second check folds too.
    #[test]
    fn an_element_store_between_checks_keeps_the_state() {
        let mut m = Module::new(true);
        let nothrow = m.callee(
            "leaf",
            FunctionAttributes {
                nothrow: true,
                ..Default::default()
            },
        );
        let mut b = Body::new(&m);
        let p = b.param(1, HirType::Ptr(Box::new(HirType::I64)));
        let entry = b.block();
        let mid = b.block();
        let raise = b.block();
        let ok = b.block();
        b.call(entry, nothrow);
        b.check(entry, raise, mid);
        let v = b.int(7);
        b.push(
            mid,
            HirInstruction::Store {
                value: v,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        b.check(mid, raise, ok);
        b.ret(raise, -1);
        b.ret(ok, 0);
        let f = b.finish(&mut m);
        run_all(&mut m.module);
        assert_eq!(branch_constant(body(&m, f), entry, raise), Some(false));
        assert_eq!(branch_constant(body(&m, f), mid, raise), Some(false));
    }

    /// entry: brcond p, a, b; a: call raiser; b: nothing; both reach a
    /// check whose raising arm returns a phi of the check block's phi.
    /// The raiser's edge goes straight to the arm with its own value, and
    /// the check, now reached only with the flag clear, folds.
    #[test]
    fn a_set_predecessor_threads_to_the_raising_arm() {
        let mut m = Module::new(true);
        let raiser = m.callee(
            "raiser",
            FunctionAttributes {
                sets_error_flag: true,
                ..Default::default()
            },
        );
        let mut b = Body::new(&m);
        let cond = b.param(0, HirType::Bool);
        let entry = b.block();
        let a = b.block();
        let other = b.block();
        let check = b.block();
        let raise = b.block();
        let ok = b.block();
        b.term(
            entry,
            HirTerminator::CondBranch {
                condition: cond,
                true_target: a,
                false_target: other,
            },
        );
        b.call(a, raiser);
        b.term(a, HirTerminator::Branch { target: check });
        b.term(other, HirTerminator::Branch { target: check });
        let (one, two) = (b.int(1), b.int(2));
        let merged = b.f.create_value(HirType::I64, HirValueKind::Instruction);
        b.f.blocks[&check].phis.push(HirPhi {
            result: merged,
            ty: HirType::I64,
            incoming: vec![(one, a), (two, other)],
        });
        b.check(check, raise, ok);
        let out = b.f.create_value(HirType::I64, HirValueKind::Instruction);
        b.f.blocks[&raise].phis.push(HirPhi {
            result: out,
            ty: HirType::I64,
            incoming: vec![(merged, check)],
        });
        b.term(raise, HirTerminator::Return { values: vec![out] });
        b.term(
            ok,
            HirTerminator::Return {
                values: vec![merged],
            },
        );
        let f = b.finish(&mut m);
        let stats = run_all(&mut m.module);
        assert_eq!(stats.threaded, 1);
        let func = body(&m, f);
        assert!(
            matches!(func.blocks[&a].terminator, HirTerminator::Branch { target } if target == raise)
        );
        assert!(func.blocks[&raise].phis[0].incoming.contains(&(one, a)));
        assert_eq!(func.blocks[&check].predecessors, vec![other]);
        assert_eq!(func.blocks[&check].phis[0].incoming, vec![(two, other)]);
        assert_eq!(branch_constant(func, check, raise), Some(false));
        assert!(dominance_holds(func));
    }

    /// A module that names no error flag is left alone.
    #[test]
    fn a_module_without_an_error_flag_is_untouched() {
        let mut m = Module::new(false);
        let callee = m.callee(
            "leaf",
            FunctionAttributes {
                nothrow: true,
                ..Default::default()
            },
        );
        let mut b = Body::new(&m);
        let entry = b.block();
        let raise = b.block();
        let ok = b.block();
        b.call(entry, callee);
        b.check(entry, raise, ok);
        b.ret(raise, -1);
        b.ret(ok, 0);
        let f = b.finish(&mut m);
        assert_eq!(run_all(&mut m.module), ErrorFlagStats::default());
        assert_eq!(flag_loads(body(&m, f), entry), 1);
    }

    /// A callee with no attributes whose body never touches the flag
    /// keeps it: the check after it folds.
    #[test]
    fn a_callee_whose_body_keeps_the_flag_folds_the_check() {
        let mut m = Module::new(true);
        let mut leaf = Body::new(&m);
        let e = leaf.block();
        leaf.ret(e, 0);
        let leaf = leaf.finish(&mut m);
        let mut b = Body::new(&m);
        let entry = b.block();
        let raise = b.block();
        let ok = b.block();
        b.call(entry, leaf);
        b.check(entry, raise, ok);
        b.ret(raise, -1);
        b.ret(ok, 0);
        let f = b.finish(&mut m);
        run_all(&mut m.module);
        assert_eq!(branch_constant(body(&m, f), entry, raise), Some(false));
    }
}
