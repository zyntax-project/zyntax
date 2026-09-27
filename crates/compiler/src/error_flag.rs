//! Pending-error checks the error flag's state has already decided.
//!
//! A language that reports errors through a sticky global (see
//! `HirGlobal::error_flag`) tests it after every fallible call. The flag
//! is null at every function entry, only stores and calls change it, and
//! a callee's attributes say what a call does to it: `sets_error_flag`
//! leaves it set, `nothrow` leaves a null flag null, and a body that
//! stores nothing to it and calls only such bodies leaves it as found.
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

pub fn run_module(module: &mut HirModule) -> ErrorFlagStats {
    let mut total = ErrorFlagStats::default();
    // `ZYNTAX_DISABLE_ERROR_FLAG=1` keeps every error-flag check as
    // lowered; safe to run with.
    if std::env::var_os("ZYNTAX_DISABLE_ERROR_FLAG").is_some() {
        return total;
    }
    let Some(flag) = module
        .globals
        .iter()
        .find(|(_, g)| g.error_flag)
        .map(|(id, _)| *id)
    else {
        return total;
    };
    let effects = call_effects(module, flag);
    for func in module.functions_to_optimize() {
        let s = run(func, flag, &effects);
        total.folded += s.folded;
        total.threaded += s.threaded;
    }
    total
}

/// Each function's effect on the flag when called.
fn call_effects(module: &HirModule, flag: HirId) -> HashMap<HirId, Effect> {
    let mut effects: HashMap<HirId, Effect> = module
        .functions
        .iter()
        .map(|(id, f)| {
            let e = if f.attributes.sets_error_flag {
                Effect::Sets
            } else if f.is_external {
                // A foreign `nothrow` body reaches no hook, so it has no
                // way to the flag.
                if f.attributes.nothrow {
                    Effect::Keeps
                } else {
                    Effect::Unknown
                }
            } else {
                Effect::Keeps
            };
            (*id, e)
        })
        .collect();
    // Greatest fixed point: a body keeps the flag while it stores nothing
    // to it and every call it makes keeps it.
    loop {
        let mut changed = false;
        for (id, f) in &module.functions {
            if f.is_external || effects[id] != Effect::Keeps {
                continue;
            }
            if !body_keeps(f, flag, &effects) {
                let e = if f.attributes.nothrow {
                    Effect::KeepsClear
                } else {
                    Effect::Unknown
                };
                effects.insert(*id, e);
                changed = true;
            }
        }
        if !changed {
            return effects;
        }
    }
}

fn body_keeps(f: &HirFunction, flag: HirId, effects: &HashMap<HirId, Effect>) -> bool {
    let ptrs = flag_pointers(f, flag);
    if !ptrs.is_empty() && address_escapes(f, &ptrs) {
        return false;
    }
    f.blocks.values().all(|b| {
        !matches!(b.terminator, HirTerminator::Invoke { .. })
            && b.instructions.iter().all(|inst| match inst {
                HirInstruction::Store { ptr, .. } => !ptrs.contains(ptr),
                _ => effect_of(inst, effects) == Effect::Keeps,
            })
    })
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

    fn solve(&self, f: &HirFunction, dt: &DominatorTree) -> Solution {
        let mut entry: HashMap<HirId, State> = HashMap::new();
        let mut edges: HashMap<(HirId, HirId), State> = HashMap::new();
        let rpo = dt.rpo();
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
    func.rebuild_cfg_edges();
    stats.threaded = thread(func, flag, effects);
    if stats.threaded > 0 {
        func.rebuild_cfg_edges();
    }
    stats.folded = fold(func, flag, effects);
    stats
}

/// Send each predecessor that reaches a check block with the flag Set
/// straight to the check's raising arm.
fn thread(func: &mut HirFunction, flag: HirId, effects: &HashMap<HirId, Effect>) -> usize {
    let flow = Flow::new(func, flag, effects);
    if flow.compares.is_empty() {
        return 0;
    }
    let dt = DominatorTree::new(func);
    let sol = flow.solve(func, &dt);

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
fn fold(func: &mut HirFunction, flag: HirId, effects: &HashMap<HirId, Effect>) -> usize {
    let flow = Flow::new(func, flag, effects);
    if flow.compares.is_empty() {
        return 0;
    }
    let dt = DominatorTree::new(func);
    let sol = flow.solve(func, &dt);
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
