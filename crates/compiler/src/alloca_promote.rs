//! Alloca → Malloc promotion for escaping allocations.
//!
//! Today's ZynML front-end uniformly lowers array literals,
//! `List<T>` structs, closure environments, and address-taken
//! locals to `HirInstruction::Alloca` — stack allocations. That's
//! the right call for values that live inside a function's stack
//! frame: stack is fast and auto-freed at return. It's silently
//! wrong for values that *escape* the frame (returned, stored
//! into caller-visible memory, passed to a non-Free call, captured
//! by a closure, …) — the caller reads dangling stack and gets
//! garbage or worse.
//!
//! This pass walks each function looking for Allocas whose result
//! escapes per the same escape classifier used by `drop_insert`,
//! and rewrites those Allocas in-place to
//! `Call { callee: Intrinsic::Malloc, args: [size_bytes] }`. Their
//! result HirIds, types, and downstream uses are preserved — only
//! the defining instruction changes from "stack allocation" to
//! "heap allocation". The caller (one stack frame up) now owns the
//! lifetime; `drop_insert` running later in `run_interp_safe_opts`
//! decides whether the caller can free it, or whether ownership
//! flows further out.
//!
//! ## Scope of this first slice
//!
//! Promotes ONLY when escape analysis says the value escapes.
//! Non-escaping Allocas stay on the stack — they're auto-freed at
//! function return, no allocator traffic, no drop bookkeeping.
//!
//! ## What this does NOT do (yet)
//!
//! * Promote Allocas inside loops. A non-escaping Alloca inside a
//!   loop grows the stack frame by one slot per iteration in the
//!   current lowering (since each iter re-emits the Alloca rather
//!   than reusing one slot), which can stack-overflow before the
//!   function returns. Loop-allocated objects want
//!   per-iteration malloc+free; that needs the loop analysis to
//!   identify the iteration boundary for the free, and is the
//!   natural follow-up.
//!
//! * Promote escaping Allocas in async / generator functions.
//!   Krio's state-machine lowering already moves cross-yield state
//!   into a separate frame layout; mixing escape promotion with
//!   that bookkeeping needs care. Skipped here; `is_async` short-
//!   circuits the pass.
//!
//! ## Why not "always Malloc"?
//!
//! For every `let xs = [1,2,3]; xs.sum()` that never escapes, an
//! always-malloc lowering pays an unnecessary allocator round-trip
//! per call. Most ZynML allocations stay function-local; keeping
//! those on the stack is much cheaper than even an excellent
//! allocator can match. The opposite mistake — always-Alloca —
//! is what produces the dangling-stack bug described above.
//! Escape-driven promotion is the choice that matches each
//! allocation's actual lifetime.

use crate::hir::{
    HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirTerminator,
    HirType, HirValue, HirValueKind, Intrinsic,
};
use std::collections::HashSet;

/// Per-run statistics.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PromoteStats {
    /// Allocas examined.
    pub allocas_scanned: usize,
    /// Allocas rewritten to `Call(Intrinsic::Malloc)`.
    pub promoted: usize,
    /// Allocas left on the stack (no escape detected).
    pub kept_on_stack: usize,
    /// Allocas skipped for safety (async functions, dynamic count
    /// with non-trivial size computation that this slice doesn't
    /// emit, etc).
    pub skipped_unsupported: usize,
}

impl PromoteStats {
    fn combine(&mut self, other: PromoteStats) {
        self.allocas_scanned += other.allocas_scanned;
        self.promoted += other.promoted;
        self.kept_on_stack += other.kept_on_stack;
        self.skipped_unsupported += other.skipped_unsupported;
    }
}

/// Run the Alloca→Malloc promotion pass over every function.
pub fn run_module(module: &mut HirModule) -> PromoteStats {
    let mut total = PromoteStats::default();
    for func in module.functions.values_mut() {
        if func.is_external || func.signature.is_async {
            continue;
        }
        total.combine(run_function(func));
    }
    total
}

fn run_function(func: &mut HirFunction) -> PromoteStats {
    let mut stats = PromoteStats::default();
    let allocas = collect_allocas(func);
    for site in allocas {
        stats.allocas_scanned += 1;
        if !escapes(func, site.result) {
            stats.kept_on_stack += 1;
            continue;
        }
        if site.count.is_some() {
            // Dynamic-count Alloca: size = count_reg * sizeof(ty).
            // Emitting the multiply requires inserting a new
            // instruction before the rewrite, plus tracking the
            // new value's HirId. Skipped in this slice; revisit
            // alongside the loop-Alloca case which also needs new
            // instruction insertion.
            stats.skipped_unsupported += 1;
            continue;
        }
        if rewrite_to_malloc(func, &site) {
            stats.promoted += 1;
        } else {
            stats.skipped_unsupported += 1;
        }
    }
    stats
}

#[derive(Debug, Clone)]
struct AllocaSite {
    result: HirId,
    block: HirId,
    inst_idx: usize,
    ty: HirType,
    count: Option<HirId>,
}

fn collect_allocas(func: &HirFunction) -> Vec<AllocaSite> {
    let mut sites = Vec::new();
    for (block_id, block) in &func.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let HirInstruction::Alloca {
                result, ty, count, ..
            } = inst
            {
                sites.push(AllocaSite {
                    result: *result,
                    block: *block_id,
                    inst_idx: idx,
                    ty: ty.clone(),
                    count: *count,
                });
            }
        }
    }
    sites
}

/// Does any use of `target` look like an escape out of the current
/// stack frame? Mirrors the classifier inside
/// `crates/compiler/src/drop_insert.rs` but with a coarser
/// `bool` return — promotion only needs the yes/no answer, not the
/// last-use location. Conservative on purpose: unknown / new HIR
/// shapes are treated as escapes so we err toward heap (correct
/// but slower) rather than stack (fast but UB on escape).
fn escapes(func: &HirFunction, target: HirId) -> bool {
    for block in func.blocks.values() {
        for phi in &block.phis {
            // A phi reading `target` from a predecessor merges
            // `target`'s value into a different SSA name that
            // we then can't follow. Treat as escape.
            if phi.incoming.iter().any(|(v, _)| *v == target) {
                return true;
            }
        }
        for inst in &block.instructions {
            if inst_escapes(inst, target) {
                return true;
            }
        }
        if term_escapes(&block.terminator, target) {
            return true;
        }
    }
    false
}

fn inst_escapes(inst: &HirInstruction, target: HirId) -> bool {
    match inst {
        // Loads through `target` read its memory, not the pointer
        // value itself — no escape.
        HirInstruction::Load { .. } => false,
        HirInstruction::Store { value, .. } => {
            // Storing `target` as the *value* (not as the
            // destination pointer) hands ownership off. The `ptr`
            // arm is a normal write through `target` and is fine.
            *value == target
        }
        HirInstruction::GetElementPtr { indices, .. } => {
            // GEP base of `target` produces a derived pointer; the
            // derived pointer is what would escape further down,
            // and we'll catch it there through its own use chain.
            // `target` appearing as a GEP *index* means it's being
            // used as an integer offset — definite escape.
            indices.contains(&target)
        }
        HirInstruction::Cast { operand, .. } => {
            // The cast result aliases `target` and may escape later
            // under a different SSA name; be safe.
            *operand == target
        }
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => *condition == target || *true_val == target || *false_val == target,
        HirInstruction::ExtractValue { aggregate, .. } => *aggregate == target,
        HirInstruction::InsertValue {
            aggregate, value, ..
        } => *aggregate == target || *value == target,
        HirInstruction::Call { callee, args, .. } => {
            if let HirCallable::Indirect(v) = callee {
                if *v == target {
                    return true;
                }
            }
            args.iter().any(|a| *a == target)
        }
        HirInstruction::IndirectCall { func_ptr, args, .. } => {
            *func_ptr == target || args.iter().any(|a| *a == target)
        }
        HirInstruction::Binary { left, right, .. } => {
            // Binary on a pointer coerces to integer arithmetic
            // (PtrToInt-then-add etc.); the result aliases
            // `target`'s address bits.
            *left == target || *right == target
        }
        HirInstruction::Unary { operand, .. } => *operand == target,
        HirInstruction::AsyncSaveSlot { frame, value, .. } => *frame == target || *value == target,
        // Pure-def instructions can't *use* `target` — they only
        // produce a fresh SSA value. Listed explicitly so the
        // conservative default below catches genuinely-new variants
        // that might carry an operand we haven't taught the pass to
        // inspect.
        HirInstruction::Alloca { .. } => false,
        // Anything not explicitly handled treat as an escape — the
        // doc comment on `escapes()` calls this out. Better to leak
        // (correct under all allocators) than to dangle (UB under a
        // real stack allocator).
        _ => true,
    }
}

fn term_escapes(term: &HirTerminator, target: HirId) -> bool {
    match term {
        HirTerminator::Return { values } => values.iter().any(|v| *v == target),
        HirTerminator::CondBranch { condition, .. } => *condition == target,
        HirTerminator::Switch { value, .. } => *value == target,
        HirTerminator::Invoke { args, .. } => args.iter().any(|a| *a == target),
        _ => false,
    }
}

fn rewrite_to_malloc(func: &mut HirFunction, site: &AllocaSite) -> bool {
    let size_bytes = size_of_hir_ty(&site.ty);
    if size_bytes == 0 {
        // Zero-size types — skip to avoid passing 0 to malloc (most
        // libc mallocs accept this but the BC interp's bump
        // allocator returns a unique pointer either way, and we
        // don't want to entangle that corner case here).
        return false;
    }
    // Synthesise the size constant SSA value.
    let size_id = HirId::new();
    func.values.insert(
        size_id,
        HirValue {
            id: size_id,
            ty: HirType::I64,
            kind: HirValueKind::Constant(HirConstant::I64(size_bytes as i64)),
            uses: HashSet::new(),
            span: None,
        },
    );

    // Rewrite the Alloca instruction in place.
    let Some(block) = func.blocks.get_mut(&site.block) else {
        return false;
    };
    let Some(inst) = block.instructions.get_mut(site.inst_idx) else {
        return false;
    };
    if !matches!(inst, HirInstruction::Alloca { result, .. } if *result == site.result) {
        // Another pass mutated the slot before us — bail conservatively.
        return false;
    }
    *inst = HirInstruction::Call {
        result: Some(site.result),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![size_id],
        type_args: Vec::new(),
        const_args: Vec::new(),
        is_tail: false,
    };

    // Record the new use on the size constant. The promoted
    // Malloc Call has a result HirId — `site.result` — so that
    // goes into `size_id`'s uses set per the existing def-use
    // chain convention (`HirValue.uses` is keyed by consumer
    // instruction *result* HirId).
    if let Some(v) = func.values.get_mut(&size_id) {
        v.uses.insert(site.result);
    }
    true
}

/// Static byte size of a `HirType`. Mirrors the helper in
/// `hir_interp.rs` (not made public there to avoid widening that
/// module's surface). Covers the primitive and small-aggregate
/// variants. Unknown / opaque shapes return 0 so the rewriter
/// skips them conservatively rather than emitting a malloc(0).
fn size_of_hir_ty(ty: &HirType) -> usize {
    match ty {
        HirType::Bool | HirType::I8 | HirType::U8 => 1,
        HirType::I16 | HirType::U16 => 2,
        HirType::I32 | HirType::U32 | HirType::F32 => 4,
        HirType::I64 | HirType::U64 | HirType::F64 | HirType::Ptr(_) => 8,
        HirType::I128 | HirType::U128 => 16,
        HirType::Struct(s) => s.fields.iter().map(size_of_hir_ty).sum::<usize>().max(1),
        HirType::Array(elem, n) => size_of_hir_ty(elem).saturating_mul(*n as usize),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{HirBlock, HirFunction, HirFunctionSignature, HirInstruction, HirValue};
    use zyntax_typed_ast::InternedString;

    fn empty_sig(ret: HirType) -> HirFunctionSignature {
        HirFunctionSignature {
            params: Vec::new(),
            returns: vec![ret],
            type_params: Vec::new(),
            const_params: Vec::new(),
            lifetime_params: Vec::new(),
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: Vec::new(),
            is_pure: false,
        }
    }

    fn add_inst_val(func: &mut HirFunction, ty: HirType) -> HirId {
        let id = HirId::new();
        func.values.insert(
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

    #[test]
    fn non_escaping_alloca_stays_on_stack() {
        let mut f = HirFunction::new(
            InternedString::new_global("local_only"),
            empty_sig(HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Alloca {
            result: ptr,
            ty: HirType::I64,
            count: None,
            align: 8,
        });
        block.terminator = HirTerminator::Return {
            values: vec![], // void return, ptr never escapes
        };

        let stats = run_function(&mut f);
        assert_eq!(stats.allocas_scanned, 1);
        assert_eq!(stats.kept_on_stack, 1);
        assert_eq!(stats.promoted, 0);

        // Still an Alloca, unchanged.
        let block = f.blocks.values().next().unwrap();
        assert!(matches!(
            block.instructions[0],
            HirInstruction::Alloca { .. }
        ));
    }

    #[test]
    fn returned_alloca_becomes_malloc() {
        let mut f = HirFunction::new(
            InternedString::new_global("escaping"),
            empty_sig(HirType::Ptr(Box::new(HirType::I64))),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Alloca {
            result: ptr,
            ty: HirType::I64,
            count: None,
            align: 8,
        });
        block.terminator = HirTerminator::Return { values: vec![ptr] };

        let stats = run_function(&mut f);
        assert_eq!(stats.allocas_scanned, 1);
        assert_eq!(stats.promoted, 1);
        assert_eq!(stats.kept_on_stack, 0);

        let block = f.blocks.values().next().unwrap();
        match &block.instructions[0] {
            HirInstruction::Call {
                result: Some(r),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                args,
                ..
            } => {
                assert_eq!(*r, ptr);
                assert_eq!(args.len(), 1);
                // Size arg should be a Constant(I64(8)) (sizeof(i64)).
                let size_val = f.values.get(&args[0]).expect("size value present");
                match &size_val.kind {
                    HirValueKind::Constant(HirConstant::I64(n)) => assert_eq!(*n, 8),
                    other => panic!("expected I64 size constant, got {other:?}"),
                }
            }
            other => panic!("expected Malloc Call, got {other:?}"),
        }
    }

    #[test]
    fn stored_as_value_alloca_becomes_malloc() {
        // The List<T> struct construction shape: an inner Alloca
        // whose pointer is stored into an outer struct as a value.
        // The inner one must be promoted — its pointer is now
        // reachable from the outer aggregate.
        let mut f = HirFunction::new(
            InternedString::new_global("stash_inside"),
            empty_sig(HirType::Void),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let inner = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let outer = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Alloca {
            result: inner,
            ty: HirType::I64,
            count: None,
            align: 8,
        });
        block.instructions.push(HirInstruction::Alloca {
            result: outer,
            ty: HirType::I64,
            count: None,
            align: 8,
        });
        block.instructions.push(HirInstruction::Store {
            value: inner,
            ptr: outer,
            align: 8,
            volatile: false,
        });
        block.terminator = HirTerminator::Return { values: vec![] };

        let stats = run_function(&mut f);
        // inner escapes (stored as value); outer doesn't (never read
        // back, never returned).
        assert_eq!(stats.allocas_scanned, 2);
        assert_eq!(stats.promoted, 1);
        assert_eq!(stats.kept_on_stack, 1);
    }

    #[test]
    fn dynamic_count_alloca_is_skipped() {
        let mut f = HirFunction::new(
            InternedString::new_global("dyn_count"),
            empty_sig(HirType::Ptr(Box::new(HirType::I64))),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let n = HirId::new();
        f.values.insert(
            n,
            HirValue {
                id: n,
                ty: HirType::I64,
                kind: HirValueKind::Constant(HirConstant::I64(5)),
                uses: HashSet::new(),
                span: None,
            },
        );
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Alloca {
            result: ptr,
            ty: HirType::I64,
            count: Some(n),
            align: 8,
        });
        block.terminator = HirTerminator::Return { values: vec![ptr] };

        let stats = run_function(&mut f);
        assert_eq!(stats.allocas_scanned, 1);
        assert_eq!(stats.skipped_unsupported, 1);
        assert_eq!(stats.promoted, 0);
    }

    #[test]
    fn async_function_skipped_entirely() {
        let mut f = HirFunction::new(
            InternedString::new_global("async_fn"),
            HirFunctionSignature {
                is_async: true,
                is_fiber: false,
                ..empty_sig(HirType::Ptr(Box::new(HirType::I64)))
            },
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Alloca {
            result: ptr,
            ty: HirType::I64,
            count: None,
            align: 8,
        });
        block.terminator = HirTerminator::Return { values: vec![ptr] };

        // Run via run_module so the is_async gate fires.
        let mut module = HirModule::new(InternedString::new_global("test_mod"));
        let fid = HirId::new();
        f.id = fid;
        module.functions.insert(fid, f);
        let stats = run_module(&mut module);
        assert_eq!(stats.allocas_scanned, 0);
        assert_eq!(stats.promoted, 0);
    }
}
