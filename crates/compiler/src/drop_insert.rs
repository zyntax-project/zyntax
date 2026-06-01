//! Compile-time speculative drop-site analysis.
//!
//! Pairs each `Call(Intrinsic::Malloc)` with a matching
//! `Call(Intrinsic::Free)` inserted at the allocation's last use,
//! when static analysis can prove the allocation does not escape
//! the function. Implements the *speculative drop-site* memory
//! strategy that's the default in
//! `CompilationConfig.memory_strategy` (the opt-in GC variants
//! sit alongside it as alternatives selected per program).
//!
//! ## Scope of this first slice
//!
//! This pass is intentionally conservative. It only inserts a free
//! when *all* of the following hold:
//!
//! 1. The allocation is `HirInstruction::Call { callee:
//!    HirCallable::Intrinsic(Intrinsic::Malloc), .. }` and has a
//!    result HirId (otherwise nothing to free).
//! 2. Every use of that result lives in the *same* block as the
//!    malloc itself. Multi-block escapes need cross-block liveness
//!    and a worklist; not in this slice.
//! 3. No use looks like an *escape* — being returned, stored as a
//!    `value` (vs. as a `ptr`) into someone else's pointer, passed
//!    as an arg to a non-Free `Call`/`IndirectCall`, cast away, or
//!    captured by an `AsyncSaveSlot`/`CreateClosure`.
//!
//! When all three hold, we walk the block's instruction list,
//! identify the position of the last use, and splice in a
//! `Call(Intrinsic::Free)` immediately after that index.
//!
//! ## Why not free at end-of-block unconditionally
//!
//! Two reasons. First, the post-use slot might be in a sub-region
//! that doesn't dominate every exit (think early-return branches
//! that jump to a different block). Second, lifetimes that are
//! actually shorter than the block let downstream passes reuse the
//! freed bytes — putting the free as early as possible matters
//! once we add a pooling allocator later.
//!
//! ## What this does NOT do (yet)
//!
//! * Cross-block liveness — needed for malloc-in-loop,
//!   malloc-in-branch-merged-via-phi, malloc-then-conditionally-
//!   returned. The existing [`crate::analysis::LivenessAnalysis`]
//!   computes the block-level live-in/live-out sets, but
//!   `compute_instruction_uses` over there is incomplete (no GEP,
//!   no Cast, no ExtractValue, …) so its results would silently
//!   undercount uses for our purpose. Building on it carries risk
//!   of double-free; we'd rather miss frees than insert wrong
//!   ones. A standalone full-fidelity dataflow lives next.
//!
//! * Escape through stores — `Store { value: M, ptr: P }` where
//!   `P` is itself a local allocation that doesn't escape is
//!   technically still safe to free transitively, but tracking the
//!   alias chain takes a points-to analysis. We treat any such
//!   Store as an escape.
//!
//! * Stack→heap promotion. Today's ZynML lowering emits `Alloca`
//!   for array literals and List structs (stack); only explicit
//!   `Call(Intrinsic::Malloc)` lands in this pass's scope. When
//!   Alloca→Malloc promotion ships (so allocations can outlive
//!   their stack frame), it'll feed into this pass naturally.

use std::collections::HashSet;

use crate::hir::{
    HirCallable, HirFunction, HirId, HirInstruction, HirModule, HirTerminator, Intrinsic,
};

/// Per-run statistics. Mainly for telemetry + test assertions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct DropStats {
    /// Total `Call(Intrinsic::Malloc)` results examined.
    pub mallocs_scanned: usize,
    /// Frees actually inserted into the HIR.
    pub frees_inserted: usize,
    /// Allocations that escape (returned, stored, passed to a call,
    /// captured by closures/async). Skipped — the caller (or a
    /// future garbage collector) owns the lifetime.
    pub escapes_skipped: usize,
    /// Allocations whose uses cross more than one block. Skipped
    /// pending the full-fidelity cross-block dataflow.
    pub multi_block_skipped: usize,
    /// Allocations with zero uses anywhere — likely dead but we
    /// don't insert a free either (something upstream was supposed
    /// to DCE them).
    pub no_use_skipped: usize,
}

impl DropStats {
    fn combine(&mut self, other: DropStats) {
        self.mallocs_scanned += other.mallocs_scanned;
        self.frees_inserted += other.frees_inserted;
        self.escapes_skipped += other.escapes_skipped;
        self.multi_block_skipped += other.multi_block_skipped;
        self.no_use_skipped += other.no_use_skipped;
    }
}

/// Run the drop-site pass over every function in `module`.
pub fn run_module(module: &mut HirModule) -> DropStats {
    let mut total = DropStats::default();
    for func in module.functions.values_mut() {
        if func.is_external {
            continue;
        }
        total.combine(run_function(func));
    }
    total
}

fn run_function(func: &mut HirFunction) -> DropStats {
    let mut stats = DropStats::default();
    let mallocs: Vec<MallocSite> = collect_malloc_sites(func);
    for site in mallocs {
        stats.mallocs_scanned += 1;
        match analyze_site(func, &site) {
            SiteOutcome::SingleBlockDrop { block, after_idx } => {
                insert_free_after(func, block, after_idx, site.result);
                stats.frees_inserted += 1;
            }
            SiteOutcome::Escaped => stats.escapes_skipped += 1,
            SiteOutcome::MultiBlock => stats.multi_block_skipped += 1,
            SiteOutcome::NoUse => stats.no_use_skipped += 1,
        }
    }
    stats
}

/// One malloc call we're considering for drop insertion.
#[derive(Debug, Clone, Copy)]
struct MallocSite {
    /// Result HirId — the pointer the malloc produced.
    result: HirId,
    /// Block containing the malloc Call.
    block: HirId,
    /// Index of the malloc Call inside `block.instructions`.
    inst_idx: usize,
}

enum SiteOutcome {
    SingleBlockDrop { block: HirId, after_idx: usize },
    Escaped,
    MultiBlock,
    NoUse,
}

fn collect_malloc_sites(func: &HirFunction) -> Vec<MallocSite> {
    let mut sites = Vec::new();
    for (block_id, block) in &func.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let HirInstruction::Call {
                result: Some(result),
                callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                ..
            } = inst
            {
                sites.push(MallocSite {
                    result: *result,
                    block: *block_id,
                    inst_idx: idx,
                });
            }
        }
    }
    sites
}

fn analyze_site(func: &HirFunction, site: &MallocSite) -> SiteOutcome {
    // Walk every block, every instruction, every terminator, and
    // for each use of `site.result`:
    //   - record the (block, idx) location and the use *kind*
    //   - bail to `Escaped` immediately on any escape-classified use
    //   - if any use is outside the malloc's own block, bail to
    //     `MultiBlock`
    //
    // Indices use `usize::MAX` as a sentinel for "terminator use".
    let target = site.result;
    let mut last_idx_in_block: Option<usize> = None;
    let mut had_any_use = false;

    for (block_id, block) in &func.blocks {
        // Phis: any phi reading `target` from a predecessor implies
        // cross-block flow → multi-block.
        for phi in &block.phis {
            if phi.incoming.iter().any(|(v, _)| *v == target) {
                return SiteOutcome::MultiBlock;
            }
            if phi.result == target {
                // The malloc itself isn't a phi result, but defend
                // against unusual lowerings — if it is, treat as
                // multi-block to avoid surprising the rewriter.
                return SiteOutcome::MultiBlock;
            }
        }

        for (idx, inst) in block.instructions.iter().enumerate() {
            // Skip the malloc instruction itself.
            if *block_id == site.block && idx == site.inst_idx {
                continue;
            }
            match classify_inst_use(inst, target) {
                UseKind::None => {}
                UseKind::Use => {
                    had_any_use = true;
                    if *block_id != site.block {
                        return SiteOutcome::MultiBlock;
                    }
                    last_idx_in_block = Some(match last_idx_in_block {
                        Some(prev) => prev.max(idx),
                        None => idx,
                    });
                }
                UseKind::Escape => return SiteOutcome::Escaped,
            }
        }

        // Terminator. A `Return` carrying `target` is an escape.
        // A `CondBranch` / `Switch` using it as the discriminator
        // is a normal use, but the value flows into successor
        // blocks → multi-block.
        match classify_terminator_use(&block.terminator, target) {
            UseKind::None => {}
            UseKind::Use => {
                had_any_use = true;
                if *block_id != site.block {
                    return SiteOutcome::MultiBlock;
                }
                // Terminator-use in the malloc's own block — the
                // value flows to successors via the branch.
                return SiteOutcome::MultiBlock;
            }
            UseKind::Escape => return SiteOutcome::Escaped,
        }
    }

    if !had_any_use {
        return SiteOutcome::NoUse;
    }

    match last_idx_in_block {
        Some(idx) => SiteOutcome::SingleBlockDrop {
            block: site.block,
            after_idx: idx,
        },
        None => SiteOutcome::NoUse,
    }
}

/// Use-kind classification for a single instruction against one
/// target value.
enum UseKind {
    None,
    Use,
    Escape,
}

fn classify_inst_use(inst: &HirInstruction, target: HirId) -> UseKind {
    match inst {
        HirInstruction::Binary { left, right, .. } => {
            if *left == target || *right == target {
                UseKind::Use
            } else {
                UseKind::None
            }
        }
        HirInstruction::Unary { operand, .. } => {
            if *operand == target {
                UseKind::Use
            } else {
                UseKind::None
            }
        }
        HirInstruction::Load { ptr, .. } => {
            if *ptr == target {
                UseKind::Use
            } else {
                UseKind::None
            }
        }
        HirInstruction::Store { value, ptr, .. } => {
            // Storing `target` as the *value* into someone else's
            // pointer hands ownership off to whoever owns `ptr` —
            // treat as escape.
            if *value == target {
                return UseKind::Escape;
            }
            // Storing into `target` (as ptr) is a normal use.
            if *ptr == target {
                UseKind::Use
            } else {
                UseKind::None
            }
        }
        HirInstruction::GetElementPtr { ptr, indices, .. } => {
            if *ptr == target {
                return UseKind::Use;
            }
            if indices.contains(&target) {
                UseKind::Use
            } else {
                UseKind::None
            }
        }
        HirInstruction::Cast { operand, .. } => {
            // A cast result is a new SSA value, but it aliases the
            // input pointer-wise. To stay safe we treat the cast as
            // an escape — the cast result could be stored, passed,
            // or returned downstream and we'd lose track without an
            // alias chain.
            if *operand == target {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        HirInstruction::Call { callee, args, .. } => {
            // Free of `target` is exactly what this pass would have
            // inserted — refuse to double-insert.
            let is_free_call = matches!(callee, HirCallable::Intrinsic(Intrinsic::Free));
            if is_free_call && args.iter().any(|a| *a == target) {
                return UseKind::Escape;
            }
            if let HirCallable::Indirect(v) = callee {
                if *v == target {
                    return UseKind::Escape;
                }
            }
            if args.iter().any(|a| *a == target) {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        HirInstruction::IndirectCall { func_ptr, args, .. } => {
            if *func_ptr == target || args.iter().any(|a| *a == target) {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        HirInstruction::Select {
            condition,
            true_val,
            false_val,
            ..
        } => {
            // The select result aliases either branch — escape to
            // stay safe (matches the Cast treatment).
            if *condition == target || *true_val == target || *false_val == target {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        HirInstruction::ExtractValue { aggregate, .. } => {
            // Extracting from `target` (if target is an aggregate
            // value) hands a piece of it to a new SSA value — same
            // escape concern as Cast/Select. The malloc results
            // we're tracking are pointers though, not aggregates,
            // so this arm is mostly defensive.
            if *aggregate == target {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        HirInstruction::InsertValue {
            aggregate, value, ..
        } => {
            if *aggregate == target || *value == target {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        // Anything else that could plausibly use the value treat as
        // escape — better to leak than to double-free. The explicit
        // list above covers every variant we know about today; if a
        // new variant lands and we forget to update this match,
        // we'd fall through to `UseKind::None` (no escape, no use)
        // which can lead to a use-after-free if the new variant
        // *is* a real use. Surface that as `Escape` so the worst
        // case is a memory leak — debuggable, not corrupting.
        _ => UseKind::Escape,
    }
}

fn classify_terminator_use(term: &HirTerminator, target: HirId) -> UseKind {
    match term {
        HirTerminator::Return { values } => {
            if values.iter().any(|v| *v == target) {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        HirTerminator::CondBranch { condition, .. } => {
            if *condition == target {
                UseKind::Use
            } else {
                UseKind::None
            }
        }
        HirTerminator::Switch { value, .. } => {
            if *value == target {
                UseKind::Use
            } else {
                UseKind::None
            }
        }
        HirTerminator::Branch { .. } => UseKind::None,
        HirTerminator::Invoke { args, .. } => {
            // Invoke = call-with-resume-edge; any arg use is an escape.
            if args.iter().any(|a| *a == target) {
                UseKind::Escape
            } else {
                UseKind::None
            }
        }
        _ => UseKind::None,
    }
}

fn insert_free_after(func: &mut HirFunction, block_id: HirId, after_idx: usize, target: HirId) {
    let free_inst = HirInstruction::Call {
        result: None,
        callee: HirCallable::Intrinsic(Intrinsic::Free),
        args: vec![target],
        type_args: Vec::new(),
        const_args: Vec::new(),
        is_tail: false,
    };
    if let Some(block) = func.blocks.get_mut(&block_id) {
        let insert_at = (after_idx + 1).min(block.instructions.len());
        block.instructions.insert(insert_at, free_inst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::{
        HirBlock, HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule,
        HirType, HirValue, HirValueKind,
    };
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
            effects: Vec::new(),
            is_pure: false,
        }
    }

    fn add_const(func: &mut HirFunction, ty: HirType, c: HirConstant) -> HirId {
        let id = HirId::new();
        func.values.insert(
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

    /// A function that mallocs an i64 worth of bytes, stores into
    /// it, loads back, and returns the *load* (not the pointer).
    /// The pointer should be drop-inserted.
    fn build_alloc_store_load_return() -> (HirFunction, HirId) {
        let mut f = HirFunction::new(
            InternedString::new_global("alloc_local"),
            empty_sig(HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));

        let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let val_to_store = add_const(&mut f, HirType::I64, HirConstant::I64(42));
        let loaded = add_inst_val(&mut f, HirType::I64);

        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::Store {
            value: val_to_store,
            ptr,
            align: 8,
            volatile: false,
        });
        block.instructions.push(HirInstruction::Load {
            result: loaded,
            ty: HirType::I64,
            ptr,
            align: 8,
            volatile: false,
        });
        block.terminator = HirTerminator::Return {
            values: vec![loaded],
        };
        (f, ptr)
    }

    #[test]
    fn inserts_free_after_last_use_in_single_block() {
        let (mut f, ptr) = build_alloc_store_load_return();
        let stats = run_function(&mut f);
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 1);
        assert_eq!(stats.escapes_skipped, 0);

        let block = f.blocks.values().next().unwrap();
        // Expect: malloc, store, load, FREE, (Return is terminator)
        assert_eq!(block.instructions.len(), 4);
        match &block.instructions[3] {
            HirInstruction::Call { callee, args, .. } => {
                assert!(matches!(callee, HirCallable::Intrinsic(Intrinsic::Free)));
                assert_eq!(args, &vec![ptr]);
            }
            other => panic!("expected Free call, got {other:?}"),
        }
    }

    #[test]
    fn skips_when_pointer_is_returned() {
        let mut f = HirFunction::new(
            InternedString::new_global("alloc_and_return"),
            empty_sig(HirType::Ptr(Box::new(HirType::I64))),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return { values: vec![ptr] };

        let stats = run_function(&mut f);
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 0);
        assert_eq!(stats.escapes_skipped, 1);
    }

    #[test]
    fn skips_when_pointer_stored_as_value_into_another_pointer() {
        // Models the List<T> struct construction: malloc data
        // buffer, then store its pointer into the list struct.
        // Conservative — we treat this as escape.
        let mut f = HirFunction::new(
            InternedString::new_global("alloc_and_stash"),
            empty_sig(HirType::Void),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let data_ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let stash_slot = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Alloca {
            result: stash_slot,
            ty: HirType::I64,
            count: None,
            align: 8,
        });
        block.instructions.push(HirInstruction::Call {
            result: Some(data_ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::Store {
            value: data_ptr,
            ptr: stash_slot,
            align: 8,
            volatile: false,
        });
        block.terminator = HirTerminator::Return { values: vec![] };

        let stats = run_function(&mut f);
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 0);
        assert_eq!(stats.escapes_skipped, 1);
    }

    #[test]
    fn skips_when_pointer_passed_to_another_call() {
        let mut f = HirFunction::new(
            InternedString::new_global("alloc_and_pass"),
            empty_sig(HirType::Void),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        // Hand the pointer to some other call — unknown ownership
        // semantics, must conservatively treat as escape.
        block.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Symbol("opaque_consumer".to_string()),
            args: vec![ptr],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return { values: vec![] };

        let stats = run_function(&mut f);
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 0);
        assert_eq!(stats.escapes_skipped, 1);
    }

    #[test]
    fn no_use_alloc_does_not_get_free_inserted() {
        // Malloc with zero downstream uses — odd shape, possibly
        // dead. We don't insert a free either; let DCE handle it.
        let mut f = HirFunction::new(
            InternedString::new_global("dead_alloc"),
            empty_sig(HirType::Void),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return { values: vec![] };

        let stats = run_function(&mut f);
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 0);
        assert_eq!(stats.no_use_skipped, 1);
    }

    #[test]
    fn does_not_double_free_an_existing_explicit_free() {
        // If someone already emitted a Free for the pointer, the
        // Free Call counts as the escape path — we won't insert a
        // second one. (Same outcome whether you call it "escape"
        // or "user already freed it" — either way, no double-free.)
        let mut f = HirFunction::new(
            InternedString::new_global("explicit_free"),
            empty_sig(HirType::Void),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::Call {
            result: None,
            callee: HirCallable::Intrinsic(Intrinsic::Free),
            args: vec![ptr],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return { values: vec![] };

        let stats = run_function(&mut f);
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 0);
        // Either Escape or NoUse outcome — Escape because the Free
        // Call's args include `ptr` and our classifier treats that
        // as an escape (refusing to double-insert).
        assert_eq!(stats.escapes_skipped, 1);
    }
}
