//! Conservative function-purity inference.
//!
//! A function is *pure* when its return value depends only on its
//! arguments and evaluating it produces no observable effect — no
//! stores, no memory reads that could alias mutable state, no algebraic
//! effects, no I/O, no dynamic dispatch. Purity is what lets later
//! passes treat two identical calls as the same value (call CSE) and
//! reason about redundancy across the control-flow graph.
//!
//! ## Why we compute it here (not lean on the backend)
//!
//! LLVM infers `memory(none)` and CSEs duplicate pure calls in its
//! optimiser; the BC interpreter, the wasm tier, and Cranelift's
//! single-pass backend don't. Computing purity at the HIR level lets
//! *every* backend share the result — see the call-CSE consumer in
//! [`crate::cse`].
//!
//! ## The lattice + fixpoint
//!
//! Purity is a greatest-fixpoint property over the call graph: a
//! function is pure iff every instruction it contains is value-pure
//! *and* every function it calls is pure. Mutual/self recursion means
//! we can't decide one function in isolation, so we iterate:
//!
//!   1. Optimistically assume every locally-clean function is pure.
//!   2. Repeatedly demote any function that calls a known-impure one.
//!   3. Stop when a full sweep demotes nothing.
//!
//! Self-recursion (like `fib` calling `fib`) is handled naturally: the
//! optimistic seed keeps `fib` pure through the fixpoint because its
//! only callee — itself — never gets demoted.
//!
//! ## Conservatism
//!
//! Every classification errs toward *impure*. Anything we don't model
//! precisely (loads, allocations, closures, trait dispatch, effects,
//! fibers, async, vector memory ops, non-`Function` callees) makes the
//! enclosing function impure. Marking a truly-impure function pure would
//! be a miscompile once call-CSE trusts the flag, so the whitelist below
//! is deliberately narrow: only instructions whose result is a pure
//! function of their SSA operands are allowed.

use crate::hir::{HirCallable, HirFunction, HirId, HirInstruction, HirModule};
use std::collections::{HashMap, HashSet};

/// Stats surfaced for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PurityStats {
    /// Functions marked pure this run.
    pub pure: usize,
    /// Functions left impure.
    pub impure: usize,
    /// Fixpoint iterations taken.
    pub iterations: usize,
}

/// Infer purity for every function in `module` and write the result
/// back into each function's `signature.is_pure`. Returns stats.
pub fn infer_module(module: &mut HirModule) -> PurityStats {
    // Step 1: local screen. A function that contains an instruction we
    // can't prove value-pure is impure regardless of its callees, and
    // never needs revisiting. Functions that pass the local screen are
    // *candidates* whose purity still hinges on their callees.
    let mut candidate: HashSet<HirId> = HashSet::new();
    // callee_fns[f] = the set of directly-called HirIds for a candidate.
    let mut callee_fns: HashMap<HirId, Vec<HirId>> = HashMap::new();

    for (id, func) in &module.functions {
        // External / declaration-only functions have no body we can
        // inspect — treat as impure (we can't see what they do).
        if func.is_external || func.blocks.is_empty() {
            continue;
        }
        if let Some(callees) = locally_pure_callees(func) {
            candidate.insert(*id);
            callee_fns.insert(*id, callees);
        }
    }

    // Step 2: greatest-fixpoint demotion. Start optimistic (all
    // candidates pure) and demote any whose callee is not a pure
    // candidate, until stable.
    let mut pure: HashSet<HirId> = candidate.clone();
    let mut iterations = 0;
    loop {
        iterations += 1;
        let mut changed = false;
        // Collect demotions first so we don't mutate `pure` mid-scan.
        let mut demote: Vec<HirId> = Vec::new();
        for id in &pure {
            let callees = &callee_fns[id];
            // Pure only if every callee is itself currently pure. A
            // callee not in `pure` (impure, external, or a non-candidate)
            // taints this function.
            if !callees.iter().all(|c| pure.contains(c)) {
                demote.push(*id);
            }
        }
        for id in demote {
            pure.remove(&id);
            changed = true;
        }
        if !changed {
            break;
        }
    }

    // Step 3: write back.
    let mut stats = PurityStats {
        iterations,
        ..Default::default()
    };
    for (id, func) in &mut module.functions {
        let is_pure = pure.contains(id);
        func.signature.is_pure = is_pure;
        if func.is_external || func.blocks.is_empty() {
            // Leave the flag as written (false); don't count declarations.
            continue;
        }
        if is_pure {
            stats.pure += 1;
        } else {
            stats.impure += 1;
        }
    }
    stats
}

/// Screen one function for *local* purity: every instruction must be
/// value-pure. Returns `Some(callees)` — the HirIds of every directly
/// called `Function` — when the body is locally clean, or `None` if any
/// instruction disqualifies it outright.
///
/// The returned callee list is what the fixpoint uses to propagate
/// impurity across the call graph. A call to anything other than a
/// direct `Function` (symbol, intrinsic, indirect) is treated as
/// disqualifying here rather than as a callee, because we can't name a
/// HirId to check its purity.
fn locally_pure_callees(func: &HirFunction) -> Option<Vec<HirId>> {
    let mut callees = Vec::new();
    for block in func.blocks.values() {
        for inst in &block.instructions {
            match inst {
                // Value-pure: result is a deterministic function of the
                // SSA operands, no memory or control effect.
                HirInstruction::Binary { .. }
                | HirInstruction::Unary { .. }
                | HirInstruction::Cast { .. }
                | HirInstruction::Select { .. }
                | HirInstruction::ExtractValue { .. }
                | HirInstruction::InsertValue { .. }
                | HirInstruction::GetElementPtr { .. }
                | HirInstruction::CreateUnion { .. }
                | HirInstruction::GetUnionDiscriminant { .. }
                | HirInstruction::ExtractUnionValue { .. }
                | HirInstruction::VectorSplat { .. }
                | HirInstruction::VectorExtractLane { .. }
                | HirInstruction::VectorInsertLane { .. }
                | HirInstruction::VectorHorizontalReduce { .. }
                | HirInstruction::VectorUnaryOp { .. }
                | HirInstruction::VectorMinMax { .. }
                | HirInstruction::VectorDot { .. } => {}

                // A direct call is pure-*iff* its callee is; record the
                // callee for the fixpoint. Any other callable shape
                // (indirect, intrinsic, symbol) is opaque → impure.
                HirInstruction::Call { callee, .. } => match callee {
                    HirCallable::Function(fid) => callees.push(*fid),
                    _ => return None,
                },

                // Everything else is potentially effectful or reads
                // mutable state — conservatively disqualifying.
                _ => return None,
            }
        }
    }
    Some(callees)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        BinaryOp, HirBlock, HirConstant, HirFunctionSignature, HirInstruction, HirTerminator,
        HirType, HirValue, HirValueKind,
    };
    use zyntax_typed_ast::InternedString;

    fn sig() -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![],
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

    fn mk(name: &str) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global(name), sig());
        let entry = HirId::new();
        f.id = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f
    }

    fn param(f: &mut HirFunction, idx: u32) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(idx),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn konst(f: &mut HirFunction, c: HirConstant) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Constant(c),
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn result(f: &mut HirFunction) -> HirId {
        let id = HirId::new();
        f.values.insert(
            id,
            HirValue {
                id,
                ty: HirType::I64,
                kind: HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        id
    }

    fn push(f: &mut HirFunction, inst: HirInstruction) {
        let e = f.entry_block;
        f.blocks.get_mut(&e).unwrap().instructions.push(inst);
    }

    fn ret(f: &mut HirFunction, v: HirId) {
        let e = f.entry_block;
        f.blocks.get_mut(&e).unwrap().terminator = HirTerminator::Return { values: vec![v] };
    }

    /// A leaf arithmetic function is pure.
    #[test]
    fn arithmetic_leaf_is_pure() {
        let mut m = HirModule::new(InternedString::new_global("m"));
        let mut f = mk("add1");
        let n = param(&mut f, 0);
        let one = konst(&mut f, HirConstant::I64(1));
        let r = result(&mut f);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Add,
                result: r,
                ty: HirType::I64,
                left: n,
                right: one,
            },
        );
        ret(&mut f, r);
        let fid = f.id;
        m.functions.insert(fid, f);

        let stats = infer_module(&mut m);
        assert_eq!(stats.pure, 1);
        assert!(m.functions[&fid].signature.is_pure);
    }

    /// Self-recursion stays pure through the fixpoint (the `fib` case).
    #[test]
    fn self_recursive_is_pure() {
        let mut m = HirModule::new(InternedString::new_global("m"));
        let mut f = mk("fib");
        let fid = f.id;
        let n = param(&mut f, 0);
        let c = konst(&mut f, HirConstant::I64(1));
        let sub = result(&mut f);
        push(
            &mut f,
            HirInstruction::Binary {
                op: BinaryOp::Sub,
                result: sub,
                ty: HirType::I64,
                left: n,
                right: c,
            },
        );
        let call = result(&mut f);
        push(
            &mut f,
            HirInstruction::Call {
                result: Some(call),
                callee: HirCallable::Function(fid),
                args: vec![sub],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        ret(&mut f, call);
        m.functions.insert(fid, f);

        let stats = infer_module(&mut m);
        assert!(
            m.functions[&fid].signature.is_pure,
            "self-recursive pure fn should stay pure, stats={stats:?}"
        );
    }

    /// A store makes the function impure, and any pure caller of it is
    /// demoted transitively.
    #[test]
    fn store_taints_transitively() {
        let mut m = HirModule::new(InternedString::new_global("m"));

        // impure: writes memory.
        let mut writer = mk("writer");
        let wid = writer.id;
        let p = param(&mut writer, 0);
        let v = konst(&mut writer, HirConstant::I64(7));
        push(
            &mut writer,
            HirInstruction::Store {
                value: v,
                ptr: p,
                align: 8,
                volatile: false,
            },
        );
        ret(&mut writer, v);
        m.functions.insert(wid, writer);

        // caller: locally clean but calls the impure writer.
        let mut caller = mk("caller");
        let cid = caller.id;
        let a = param(&mut caller, 0);
        let r = result(&mut caller);
        push(
            &mut caller,
            HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Function(wid),
                args: vec![a],
                type_args: vec![],
                const_args: vec![],
                is_tail: false,
            },
        );
        ret(&mut caller, r);
        m.functions.insert(cid, caller);

        infer_module(&mut m);
        assert!(!m.functions[&wid].signature.is_pure, "writer is impure");
        assert!(
            !m.functions[&cid].signature.is_pure,
            "caller of impure fn must be demoted"
        );
    }
}
