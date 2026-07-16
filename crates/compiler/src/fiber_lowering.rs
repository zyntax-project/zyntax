//! Fiber HIR lowering — rewrites first-class fiber ops to
//! `Call::Symbol("krio_fiber_*")` before any backend sees them.
//!
//! Mirrors the role `apply_krio_async_lowering` plays for async: the
//! abstract opcodes capture frontend intent, this pass collapses them
//! into runtime symbol calls the BC interp + Cranelift + LLVM
//! backends already know how to dispatch. The runtime signatures are
//! pre-registered (see `effect_runtime::register_fiber_runtime_symbols`
//! and `zrtl::fiber_runtime_symbols`) so link resolution is clean.
//!
//! Lowering lives in this pass, not in per-backend arms. Three
//! backends × six ops would be eighteen arms maintained in lockstep;
//! one pass with six rules is the single point of truth. The pass
//! also keeps the substrate swappable — a future stackless-fallback
//! path for wasm targets without stack switching replaces this pass
//! with one that emits a krio-stackless state machine, without
//! touching any backend.
//!
//! ## ABI
//!
//! Each op is rewritten as follows (operand types listed left-to-
//! right matching the registered runtime signatures):
//!
//! | HIR op                | symbol                    | args                           |
//! |-----------------------|---------------------------|--------------------------------|
//! | `FiberNew`            | `krio_fiber_new`          | `(closure: ptr, stack: i64)`   |
//! | `FiberResume`         | `krio_fiber_resume`       | `(fiber: ptr)`                 |
//! | `FiberResumeWith`     | `krio_fiber_resume_with`  | `(fiber: ptr, value: i64)`     |
//! | `FiberYield`          | `krio_fiber_yield`        | `(value: i64)`                 |
//! | `FiberTransfer`       | `krio_fiber_transfer`     | `(target: ptr, value: i64)`    |
//! | `FiberCancel`         | `krio_fiber_cancel`       | `(fiber: ptr)`                 |
//!
//! `result` values are preserved across the rewrite — SSA uses
//! downstream of a fiber op continue to reference the same `HirId`,
//! now produced by the `Call::Symbol` invocation instead. No use-list
//! fixup needed because the result id is reused verbatim.

use crate::hir::{HirCallable, HirId, HirInstruction, HirModule, HirType, HirValue, HirValueKind};

/// Rewrite every fiber op in the module into its `Call::Symbol`
/// equivalent. Idempotent: running the pass twice is a no-op
/// (subsequent runs see only `Call` instructions and skip).
///
/// Most ops map 1:1. `FiberResume`/`FiberResumeWith` expand to three
/// calls — the resume is bracketed by `__zyntax_effect_fiber_enter` /
/// `__zyntax_effect_fiber_leave` so the fiber gets its own handler-stack
/// segment across the stack switch (a `perform` inside the fiber sees the
/// handlers active when it was resumed, and its own open handlers don't
/// leak into the caller). `FiberDrop` gains a `__zyntax_effect_fiber_forget`
/// so the freed fiber's saved segment is dropped.
///
/// Call this immediately before backend compilation, after any
/// frontend-side fiber-shape transforms have run.
pub fn apply_krio_fiber_lowering(module: &mut HirModule) {
    for func in module.functions.values_mut() {
        let mut minted: Vec<HirId> = Vec::new();
        for block in func.blocks.values_mut() {
            let mut out = Vec::with_capacity(block.instructions.len());
            for inst in std::mem::take(&mut block.instructions) {
                match &inst {
                    // Bracket the resume with enter/leave. `enter` returns
                    // the caller's stack depth (baseline) and re-pushes the
                    // fiber's saved frames; `leave` lifts the fiber's open
                    // frames back out and truncates to baseline.
                    HirInstruction::FiberResume { fiber, .. }
                    | HirInstruction::FiberResumeWith { fiber, .. } => {
                        let fiber = *fiber;
                        let baseline = HirId::new();
                        minted.push(baseline);
                        out.push(sym_call(
                            "__zyntax_effect_fiber_enter",
                            Some(baseline),
                            vec![fiber],
                        ));
                        out.push(rewrite(&inst).expect("resume rewrites to a krio_fiber_* call"));
                        out.push(sym_call(
                            "__zyntax_effect_fiber_leave",
                            None,
                            vec![fiber, baseline],
                        ));
                    }
                    // Forget the fiber's saved segment before freeing it, so
                    // an address-reused fiber can't inherit stale frames.
                    HirInstruction::FiberDrop { fiber } => {
                        out.push(sym_call("__zyntax_effect_fiber_forget", None, vec![*fiber]));
                        out.push(rewrite(&inst).expect("FiberDrop rewrites to krio_fiber_free"));
                    }
                    _ => match rewrite(&inst) {
                        Some(replacement) => out.push(replacement),
                        None => out.push(inst),
                    },
                }
            }
            block.instructions = out;
        }
        // Register the minted baseline values (i64 results of `fiber_enter`)
        // so downstream sees a typed SSA value for each.
        for id in minted {
            func.values.insert(
                id,
                HirValue {
                    id,
                    ty: HirType::I64,
                    kind: HirValueKind::Instruction,
                    uses: std::collections::HashSet::new(),
                    span: None,
                },
            );
        }
    }
}

/// Build a `Call::Symbol` instruction (the runtime-extern call shape used
/// throughout this pass).
fn sym_call(name: &str, result: Option<HirId>, args: Vec<HirId>) -> HirInstruction {
    HirInstruction::Call {
        result,
        callee: HirCallable::Symbol(name.to_string()),
        args,
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    }
}

/// Rewrite a single fiber op. Returns `Some(call)` when the op is a
/// fiber op the pass owns; `None` when it's anything else (no
/// replacement needed). The caller swaps `inst` for the returned
/// value in place.
fn rewrite(inst: &HirInstruction) -> Option<HirInstruction> {
    match inst {
        HirInstruction::FiberNew {
            result,
            ty: _,
            closure,
            stack_size,
        } => Some(HirInstruction::Call {
            result: Some(*result),
            callee: HirCallable::Symbol("krio_fiber_new".to_string()),
            args: vec![*closure, *stack_size],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }),

        HirInstruction::FiberResume {
            result,
            ty: _,
            fiber,
        } => Some(HirInstruction::Call {
            result: Some(*result),
            callee: HirCallable::Symbol("krio_fiber_resume".to_string()),
            args: vec![*fiber],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }),

        HirInstruction::FiberResumeWith {
            result,
            ty: _,
            fiber,
            value,
        } => Some(HirInstruction::Call {
            result: Some(*result),
            callee: HirCallable::Symbol("krio_fiber_resume_with".to_string()),
            args: vec![*fiber, *value],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }),

        HirInstruction::FiberYield { value } => Some(HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("krio_fiber_yield".to_string()),
            args: vec![*value],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }),

        HirInstruction::FiberTransfer {
            result,
            ty: _,
            target,
            value,
        } => Some(HirInstruction::Call {
            result: Some(*result),
            callee: HirCallable::Symbol("krio_fiber_transfer".to_string()),
            args: vec![*target, *value],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }),

        HirInstruction::FiberCancel { fiber } => Some(HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("krio_fiber_cancel".to_string()),
            args: vec![*fiber],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }),

        HirInstruction::FiberDrop { fiber } => Some(HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("krio_fiber_free".to_string()),
            args: vec![*fiber],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        }),

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirId, HirType};

    fn call_sym(inst: &HirInstruction) -> Option<(&str, usize, bool)> {
        match inst {
            HirInstruction::Call {
                callee: HirCallable::Symbol(name),
                args,
                result,
                ..
            } => Some((name.as_str(), args.len(), result.is_some())),
            _ => None,
        }
    }

    #[test]
    fn rewrites_fiber_new() {
        let inst = HirInstruction::FiberNew {
            result: HirId::new(),
            ty: HirType::I64,
            closure: HirId::new(),
            stack_size: HirId::new(),
        };
        let lowered = rewrite(&inst).expect("FiberNew should rewrite");
        assert_eq!(call_sym(&lowered), Some(("krio_fiber_new", 2, true)));
    }

    #[test]
    fn rewrites_fiber_resume() {
        let inst = HirInstruction::FiberResume {
            result: HirId::new(),
            ty: HirType::I64,
            fiber: HirId::new(),
        };
        let lowered = rewrite(&inst).expect("FiberResume should rewrite");
        assert_eq!(call_sym(&lowered), Some(("krio_fiber_resume", 1, true)));
    }

    #[test]
    fn rewrites_fiber_resume_with() {
        let inst = HirInstruction::FiberResumeWith {
            result: HirId::new(),
            ty: HirType::I64,
            fiber: HirId::new(),
            value: HirId::new(),
        };
        let lowered = rewrite(&inst).expect("FiberResumeWith should rewrite");
        assert_eq!(
            call_sym(&lowered),
            Some(("krio_fiber_resume_with", 2, true))
        );
    }

    #[test]
    fn rewrites_fiber_yield_no_result() {
        let inst = HirInstruction::FiberYield {
            value: HirId::new(),
        };
        let lowered = rewrite(&inst).expect("FiberYield should rewrite");
        assert_eq!(call_sym(&lowered), Some(("krio_fiber_yield", 1, false)));
    }

    #[test]
    fn rewrites_fiber_transfer() {
        let inst = HirInstruction::FiberTransfer {
            result: HirId::new(),
            ty: HirType::I64,
            target: HirId::new(),
            value: HirId::new(),
        };
        let lowered = rewrite(&inst).expect("FiberTransfer should rewrite");
        assert_eq!(call_sym(&lowered), Some(("krio_fiber_transfer", 2, true)));
    }

    #[test]
    fn rewrites_fiber_cancel_no_result() {
        let inst = HirInstruction::FiberCancel {
            fiber: HirId::new(),
        };
        let lowered = rewrite(&inst).expect("FiberCancel should rewrite");
        assert_eq!(call_sym(&lowered), Some(("krio_fiber_cancel", 1, false)));
    }

    #[test]
    fn preserves_result_id() {
        // SSA users downstream reference the result id verbatim; the
        // rewrite must reuse it so no use-list fixup is needed.
        let result = HirId::new();
        let inst = HirInstruction::FiberNew {
            result,
            ty: HirType::I64,
            closure: HirId::new(),
            stack_size: HirId::new(),
        };
        let lowered = rewrite(&inst).expect("rewrite");
        match lowered {
            HirInstruction::Call {
                result: Some(r), ..
            } => assert_eq!(r, result),
            _ => panic!("expected Call with same result id"),
        }
    }

    #[test]
    fn skips_non_fiber_ops() {
        // Sanity: rewrite returns None for instructions outside the
        // fiber family, so `apply_krio_fiber_lowering` leaves them
        // untouched.
        let inst = HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("some_other".to_string()),
            args: vec![],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        };
        assert!(rewrite(&inst).is_none());
    }
}
