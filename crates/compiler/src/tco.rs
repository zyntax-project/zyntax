//! # Tail-call optimisation marker
//!
//! Walks every function and flags `HirInstruction::Call` instructions
//! whose result flows directly into a [`HirTerminator::Return`] with
//! no intervening work, by setting `is_tail = true` on the call. The
//! LLVM backend emits the `tail` flag on the corresponding
//! `CallInst` so LLVM's sibling-call optimisation can take over.
//!
//! Two shapes match:
//!   * Value return — `%r = Call f(args); Return [%r]`
//!   * Void return — `Call f(args); Return []`
//!
//! ## Scope: self-recursive only
//!
//! The marker only flips `is_tail = true` for **direct self-recursive
//! calls** (the callee is the enclosing function). Three reasons:
//!
//!   1. Self-recursive tail calls are the canonical payoff case —
//!      they let the backend turn unbounded recursion into a single
//!      stack frame (`accumulator` / `factorial` / `foldl` patterns).
//!   2. Sibling tail calls (a call to a different function in tail
//!      position) are also valid TCO candidates, but on small
//!      programs LLVM's IPA inliner can react to the `tail` marker
//!      in counter-productive ways. The fib bench regressed ~45 %
//!      exec when the single `main → fib(40)` call was marked
//!      (main was already a one-shot wrapper; the marker
//!      perturbed LLVM's inlining decisions and slowed the
//!      hot-loop body for reasons unrelated to TCO). Until we have
//!      a per-callsite heuristic that distinguishes "worth marking"
//!      from "leave alone", the safest scope is the case that
//!      can't trigger that interaction.
//!   3. Self-recursion is also the only case where a future
//!      HIR-level loop-rewrite pass could fully eliminate the call
//!      (turning a tail-recursive function into a `while` loop).
//!      Marking sibling tails has no HIR-level transformation
//!      payoff — only the backend hint.
//!
//! ## Soundness
//!
//! The marked call is the last instruction in the block, and the
//! block's terminator is `Return`. Under SSA, the call's result
//! cannot have a "later" reachable use — every use must be
//! dominated by the definition, and the block has no successors.
//!
//! `drop_insert` runs AFTER `run_interp_safe_opts`'s fixed-point, so
//! at TCO time there are no `Free` calls between a Call and the
//! terminating Return. Once drop-insert lands, a Free between the
//! Call and Return would make the Call no longer the last
//! instruction — the marker correctly skips that shape.

use crate::hir::{HirCallable, HirInstruction, HirModule, HirTerminator};

/// Diagnostics for callers / tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TcoStats {
    /// Number of `Return`-terminated blocks whose last instruction
    /// was a `Call` (i.e. shape-eligible candidates).
    pub candidates_visited: usize,
    /// Number of calls we actually flipped `is_tail = false → true`.
    /// A second run of the pass would visit the same candidates but
    /// flip nothing.
    pub marked: usize,
}

/// Mark tail-position calls across every function in `module`.
pub fn run_module(module: &mut HirModule) -> TcoStats {
    let mut stats = TcoStats::default();

    let func_ids: Vec<_> = module.functions.keys().copied().collect();
    for fid in func_ids {
        let func = match module.functions.get_mut(&fid) {
            Some(f) => f,
            None => continue,
        };
        if func.is_external {
            continue;
        }

        let block_ids: Vec<_> = func.blocks.keys().copied().collect();
        for bid in block_ids {
            let block = match func.blocks.get_mut(&bid) {
                Some(b) => b,
                None => continue,
            };

            // Terminator must be Return — capture its values first so
            // the borrow doesn't conflict with the mutable last-inst
            // borrow below.
            let return_values: Vec<_> = match &block.terminator {
                HirTerminator::Return { values } => values.clone(),
                _ => continue,
            };

            // Last instruction must be a Call. `last_mut` is what
            // lets us flip `is_tail` in place.
            let Some(last) = block.instructions.last_mut() else {
                continue;
            };
            let HirInstruction::Call {
                result,
                is_tail,
                callee,
                ..
            } = last
            else {
                continue;
            };

            // Self-recursive only. See the module docs — sibling-tail
            // hints to LLVM regressed the fib bench when the marker
            // fired on `main → fib(40)`; the safe payoff scope is
            // direct self-recursion.
            let HirCallable::Function(target_fid) = callee else {
                continue;
            };
            if *target_fid != fid {
                continue;
            }

            // Match: void return takes no values and a void Call has
            // result=None; a value return passes the Call's single
            // result through.
            let matches_tail = match (result.as_ref(), return_values.as_slice()) {
                (None, []) => true,
                (Some(r), [v]) if r == v => true,
                _ => false,
            };
            if !matches_tail {
                continue;
            }

            stats.candidates_visited += 1;
            if !*is_tail {
                *is_tail = true;
                stats.marked += 1;
            }
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirCallable, HirFunction, HirFunctionSignature, HirId, HirType};
    use zyntax_typed_ast::InternedString;

    fn empty_sig() -> HirFunctionSignature {
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

    fn mk_func(name: &str) -> HirFunction {
        let mut f = HirFunction::new(InternedString::new_global(name), empty_sig());
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        f
    }

    fn mk_module(func: HirFunction) -> HirModule {
        let mut module = HirModule::new(InternedString::new_global("test_module"));
        module.functions.insert(func.id, func);
        module
    }

    fn make_call(result: Option<HirId>, callee: HirId) -> HirInstruction {
        HirInstruction::Call {
            result,
            callee: HirCallable::Function(callee),
            args: Vec::new(),
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        }
    }

    #[test]
    fn self_recursive_value_return_is_marked() {
        let mut func = mk_func("f");
        let self_id = func.id;
        let result_id = HirId::new();
        let entry = func.entry_block;
        let block = func.blocks.get_mut(&entry).unwrap();
        block.instructions.push(make_call(Some(result_id), self_id));
        block.terminator = HirTerminator::Return {
            values: vec![result_id],
        };
        let mut module = mk_module(func);

        let stats = run_module(&mut module);

        assert_eq!(stats.candidates_visited, 1);
        assert_eq!(stats.marked, 1);
        let func = module.functions.values().next().unwrap();
        let block = func.blocks.values().next().unwrap();
        let HirInstruction::Call { is_tail, .. } = &block.instructions[0] else {
            panic!("expected Call");
        };
        assert!(*is_tail);
    }

    #[test]
    fn self_recursive_void_return_is_marked() {
        let mut func = mk_func("f");
        let self_id = func.id;
        let entry = func.entry_block;
        let block = func.blocks.get_mut(&entry).unwrap();
        block.instructions.push(make_call(None, self_id));
        block.terminator = HirTerminator::Return { values: Vec::new() };
        let mut module = mk_module(func);

        let stats = run_module(&mut module);
        assert_eq!(stats.marked, 1);
    }

    #[test]
    fn sibling_tail_call_is_not_marked() {
        // Tail-position call to a DIFFERENT function. By design we
        // restrict the marker to self-recursive direct calls.
        let mut func = mk_func("f");
        let other = HirId::new();
        let result_id = HirId::new();
        let entry = func.entry_block;
        let block = func.blocks.get_mut(&entry).unwrap();
        block.instructions.push(make_call(Some(result_id), other));
        block.terminator = HirTerminator::Return {
            values: vec![result_id],
        };
        let mut module = mk_module(func);

        let stats = run_module(&mut module);
        assert_eq!(stats.marked, 0);
    }

    #[test]
    fn return_of_unrelated_value_is_not_marked() {
        // Call produces r1; Return uses r2 — Call result doesn't flow
        // into the Return, so the call isn't in tail position.
        let mut func = mk_func("f");
        let self_id = func.id;
        let r1 = HirId::new();
        let r2 = HirId::new();
        let entry = func.entry_block;
        let block = func.blocks.get_mut(&entry).unwrap();
        block.instructions.push(make_call(Some(r1), self_id));
        block.terminator = HirTerminator::Return { values: vec![r2] };
        let mut module = mk_module(func);

        let stats = run_module(&mut module);
        assert_eq!(stats.marked, 0);
    }

    #[test]
    fn branch_terminator_is_not_marked() {
        let mut func = mk_func("f");
        let self_id = func.id;
        let r1 = HirId::new();
        let entry = func.entry_block;
        let block = func.blocks.get_mut(&entry).unwrap();
        block.instructions.push(make_call(Some(r1), self_id));
        block.terminator = HirTerminator::Branch {
            target: HirId::new(),
        };
        let mut module = mk_module(func);

        let stats = run_module(&mut module);
        assert_eq!(stats.candidates_visited, 0);
    }

    #[test]
    fn external_functions_are_skipped() {
        let mut func = mk_func("ext");
        let self_id = func.id;
        let result_id = HirId::new();
        let entry = func.entry_block;
        let block = func.blocks.get_mut(&entry).unwrap();
        block.instructions.push(make_call(Some(result_id), self_id));
        block.terminator = HirTerminator::Return {
            values: vec![result_id],
        };
        func.is_external = true;
        let mut module = mk_module(func);

        let stats = run_module(&mut module);
        assert_eq!(stats.candidates_visited, 0);
    }

    #[test]
    fn second_run_is_idempotent() {
        let mut func = mk_func("f");
        let self_id = func.id;
        let result_id = HirId::new();
        let entry = func.entry_block;
        let block = func.blocks.get_mut(&entry).unwrap();
        block.instructions.push(make_call(Some(result_id), self_id));
        block.terminator = HirTerminator::Return {
            values: vec![result_id],
        };
        let mut module = mk_module(func);

        let first = run_module(&mut module);
        let second = run_module(&mut module);
        assert_eq!(first.marked, 1);
        assert_eq!(second.marked, 0);
        assert_eq!(second.candidates_visited, 1);
    }
}
