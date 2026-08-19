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
    let facts = ModuleFacts::build(module);
    for func in module.functions.values_mut() {
        if func.is_external {
            continue;
        }
        total.combine(run_function(func, &facts));
    }
    total
}

/// What this pass knows about the other functions in the module.
#[derive(Default)]
struct ModuleFacts {
    returns_owned: std::collections::HashSet<HirId>,
    /// Per callee, which parameters are only borrowed. Keyed on the
    /// module's own key for the function, which is what a call names and
    /// is not always the function's `id` field.
    borrowed_params: std::collections::HashMap<HirId, Vec<bool>>,
}

impl ModuleFacts {
    fn build(module: &HirModule) -> Self {
        let mut borrowed_params = std::collections::HashMap::new();
        for (key, func) in module.functions.iter() {
            borrowed_params.insert(
                *key,
                func.signature
                    .params
                    .iter()
                    .map(|p| {
                        matches!(
                            p.ownership,
                            crate::hir::ParamOwnership::Borrowed
                                | crate::hir::ParamOwnership::BorrowedMut
                        )
                    })
                    .collect(),
            );
        }
        // Borrow facts first: deciding whether an allocation leaves a
        // function needs to know what its calls do with a pointer.
        let mut facts = Self {
            returns_owned: std::collections::HashSet::new(),
            borrowed_params,
        };
        facts.returns_owned = functions_returning_owned_storage(module, &facts);
        facts
    }

    /// Whether passing `target` here leaves the caller holding it.
    fn call_only_borrows(&self, callee: &HirCallable, args: &[HirId], target: HirId) -> bool {
        let HirCallable::Function(id) = callee else {
            return false;
        };
        let Some(borrows) = self.borrowed_params.get(id) else {
            return false;
        };
        args.iter()
            .enumerate()
            .filter(|(_, a)| **a == target)
            .all(|(i, _)| borrows.get(i).copied().unwrap_or(false))
    }
}

/// Release storage an accumulator replaces each time round a loop.
///
/// `sum = sum + a + b` keeps one object per iteration. The header phi
/// holds the previous one, the body builds a new one, and the previous
/// is unreachable the moment the body has read it. Nothing released it,
/// because a phi is exactly what the per-site analysis turns down.
///
/// The release goes after the phi's last use in the body, so it frees
/// only what the body actually consumed. What leaves through the exit is
/// the value the last body execution produced, which no body execution
/// ever read, so code after the loop still holds live storage. That last
/// object is not released here: one outlives the loop, rather than one
/// per iteration.
///
/// The conditions are narrow deliberately. Releasing the wrong incoming
/// is a use after free, not a slow program.
fn release_loop_carried(func: &mut HirFunction, facts: &ModuleFacts) -> usize {
    use crate::analysis::{DominatorTree, LoopForest};

    let owned: std::collections::HashSet<HirId> = collect_owned_sites(func, facts)
        .iter()
        .map(|s| s.result)
        .collect();
    if owned.is_empty() {
        return 0;
    }

    let dt = DominatorTree::new(func);
    let forest = LoopForest::detect(func, &dt);
    let mut plan: Vec<(HirId, usize, HirId)> = Vec::new();

    for lp in forest.loops() {
        // One body block, so "the last use in the body" is unambiguous
        // and every way round the loop passes through it.
        if lp.body.len() != 2 {
            continue;
        }
        let Some(&body_id) = lp.body.iter().find(|&&b| b != lp.header) else {
            continue;
        };
        let (Some(header), Some(body)) = (func.blocks.get(&lp.header), func.blocks.get(&body_id))
        else {
            continue;
        };
        // The body must close the loop itself, or what reaches the phi
        // next is not what this body produced.
        if !matches!(body.terminator, HirTerminator::Branch { target } if target == lp.header) {
            continue;
        }

        for phi in &header.phis {
            if phi.incoming.len() != 2 {
                continue;
            }
            // Every incoming must be storage this function owns, and the
            // one arriving round the back edge must be built by the
            // body. Otherwise the phi can carry one object twice and the
            // release would run on it after it was already gone.
            let mut all_owned = true;
            let mut back_edge_fresh = false;
            for (val, pred) in &phi.incoming {
                if !owned.contains(val) {
                    all_owned = false;
                    break;
                }
                if *pred == body_id {
                    back_edge_fresh = defines_value(body, *val);
                }
            }
            if !all_owned || !back_edge_fresh {
                continue;
            }

            // Nothing anywhere may keep the value, or releasing it here
            // pulls the storage out from under whatever kept it.
            let derived = derived_values(func, phi.result);
            if !uses_are_all_borrows(func, &derived, facts) {
                continue;
            }

            let Some(last) = last_use_index(body, &derived, facts) else {
                continue;
            };
            plan.push((body_id, last, phi.result));
        }
    }

    let inserted = plan.len();
    // Back to front, so earlier indices stay valid.
    plan.sort_by_key(|(_, idx, _)| std::cmp::Reverse(*idx));
    for (block_id, idx, value) in plan {
        insert_free_after(func, block_id, idx, value, Release::Intrinsic);
    }
    inserted
}

/// Whether this block defines `value`.
fn defines_value(block: &crate::hir::HirBlock, value: HirId) -> bool {
    block
        .instructions
        .iter()
        .any(|i| instruction_defines(i, value))
}

fn instruction_defines(inst: &HirInstruction, value: HirId) -> bool {
    match inst {
        HirInstruction::Binary { result, .. }
        | HirInstruction::Unary { result, .. }
        | HirInstruction::Cast { result, .. }
        | HirInstruction::GetElementPtr { result, .. }
        | HirInstruction::Load { result, .. }
        | HirInstruction::ExtractValue { result, .. }
        | HirInstruction::InsertValue { result, .. }
        | HirInstruction::Select { result, .. }
        | HirInstruction::Alloca { result, .. } => *result == value,
        HirInstruction::Call { result, .. } => *result == Some(value),
        _ => false,
    }
}

/// Whether every use of the set, anywhere in the function, leaves the
/// storage to us.
fn uses_are_all_borrows(
    func: &HirFunction,
    derived: &std::collections::HashSet<HirId>,
    facts: &ModuleFacts,
) -> bool {
    for block in func.blocks.values() {
        for inst in &block.instructions {
            if matches!(classify_derived_use(inst, derived, facts), UseKind::Escape) {
                return false;
            }
        }
        for d in derived {
            if matches!(
                classify_terminator_use(&block.terminator, *d),
                UseKind::Escape
            ) {
                return false;
            }
        }
    }
    true
}

/// Index of the last instruction in `block` that reads the set.
fn last_use_index(
    block: &crate::hir::HirBlock,
    derived: &std::collections::HashSet<HirId>,
    facts: &ModuleFacts,
) -> Option<usize> {
    let mut last = None;
    for (idx, inst) in block.instructions.iter().enumerate() {
        if matches!(classify_derived_use(inst, derived, facts), UseKind::Use) {
            last = Some(idx);
        }
    }
    last
}

/// Functions whose result is storage the caller owns.
///
/// Deliberately strict, because being wrong here releases something the
/// callee still refers to. A function qualifies only when it holds
/// exactly one allocation, that allocation leaves solely by being
/// returned, and every return hands it back. A function that sometimes
/// returns a fresh object and sometimes one it was given fails the last
/// condition and is left alone.
fn functions_returning_owned_storage(
    module: &HirModule,
    facts: &ModuleFacts,
) -> std::collections::HashSet<HirId> {
    let mut owned = std::collections::HashSet::new();
    for (key, func) in module.functions.iter() {
        if func.is_external {
            continue;
        }
        let sites = collect_malloc_sites(func);
        if sites.len() != 1 || sites[0].release != Release::Intrinsic {
            continue;
        }
        let site = sites[0];
        let derived = derived_values(func, site.result);
        if !escapes_only_by_return(func, &derived, &site, facts) {
            continue;
        }
        let mut returns = 0usize;
        let mut all_return_it = true;
        for block in func.blocks.values() {
            if let HirTerminator::Return { values } = &block.terminator {
                returns += 1;
                if !values.iter().any(|v| derived.contains(v)) {
                    all_return_it = false;
                }
            }
        }
        if returns > 0 && all_return_it {
            owned.insert(*key);
        }
    }
    owned
}

/// Whether the allocation leaves this function only by being returned.
fn escapes_only_by_return(
    func: &HirFunction,
    derived: &std::collections::HashSet<HirId>,
    site: &MallocSite,
    facts: &ModuleFacts,
) -> bool {
    for (block_id, block) in &func.blocks {
        for phi in &block.phis {
            if phi.incoming.iter().any(|(v, _)| derived.contains(v)) {
                return false;
            }
        }
        for (idx, inst) in block.instructions.iter().enumerate() {
            if *block_id == site.block && idx == site.inst_idx {
                continue;
            }
            if matches!(classify_derived_use(inst, derived, facts), UseKind::Escape) {
                return false;
            }
        }
        // Returning the allocation is the transfer itself; any other
        // escaping terminator is not.
        if let HirTerminator::Return { values } = &block.terminator {
            if values.iter().any(|v| derived.contains(v)) {
                continue;
            }
        }
        for d in derived {
            if matches!(
                classify_terminator_use(&block.terminator, *d),
                UseKind::Escape
            ) {
                return false;
            }
        }
    }
    true
}

fn run_function(func: &mut HirFunction, facts: &ModuleFacts) -> DropStats {
    let mut stats = DropStats::default();
    let mallocs: Vec<MallocSite> = collect_owned_sites(func, facts);
    for site in mallocs {
        stats.mallocs_scanned += 1;
        match analyze_site(func, &site, facts) {
            SiteOutcome::SingleBlockDrop { block, after_idx } => {
                insert_free_after(func, block, after_idx, site.result, site.release);
                stats.frees_inserted += 1;
            }
            SiteOutcome::Escaped => stats.escapes_skipped += 1,
            SiteOutcome::MultiBlock => stats.multi_block_skipped += 1,
            SiteOutcome::NoUse => stats.no_use_skipped += 1,
        }
    }
    stats.frees_inserted += release_loop_carried(func, facts);
    stats
}

/// What a runtime entry point does with a pointer.
///
/// Storage is not only handed out by the allocation intrinsic. A value
/// widened to the dynamic type is put in a box by a runtime call, and
/// that box is storage the caller owns exactly as a malloc's result is.
/// Without naming those calls the pass never sees them, so a program
/// that boxes in a loop allocates once per iteration and releases
/// nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SymbolRole {
    /// Returns storage the caller owns, released by the named symbol.
    Allocates(&'static str),
    /// Reads through a pointer argument without keeping it, so passing
    /// one here does not end the caller's claim.
    Borrows,
}

/// The role of a runtime symbol, or `None` where the pass knows nothing
/// about it and must assume the worst.
fn symbol_role(name: &str) -> Option<SymbolRole> {
    match name {
        "zyntax_box_bool" | "zyntax_box_f32" | "zyntax_box_f64" | "zyntax_box_i32"
        | "zyntax_box_i64" | "zyntax_box_opaque" => Some(SymbolRole::Allocates("zyntax_box_free")),
        "zyntax_box_get_bool"
        | "zyntax_box_get_f32"
        | "zyntax_box_get_f64"
        | "zyntax_box_get_i32"
        | "zyntax_box_get_i64"
        | "zyntax_box_get_opaque"
        | "zyntax_box_get_tag" => Some(SymbolRole::Borrows),
        _ => None,
    }
}

/// How the storage at a site is released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Release {
    /// `Call(Intrinsic::Free)`.
    Intrinsic,
    /// A named runtime call taking the pointer.
    Symbol(&'static str),
}

/// One allocation we're considering for drop insertion.
#[derive(Debug, Clone, Copy)]
struct MallocSite {
    /// Result HirId — the pointer the allocation produced.
    result: HirId,
    /// Block containing the allocating Call.
    block: HirId,
    /// Index of the allocating Call inside `block.instructions`.
    inst_idx: usize,
    /// What releases this storage.
    release: Release,
}

enum SiteOutcome {
    SingleBlockDrop { block: HirId, after_idx: usize },
    Escaped,
    MultiBlock,
    NoUse,
}

/// Allocation sites, counting a call whose callee hands back owned
/// storage: the caller owns that result and is the only one able to
/// release it.
fn collect_owned_sites(func: &HirFunction, facts: &ModuleFacts) -> Vec<MallocSite> {
    let mut sites = collect_malloc_sites(func);
    for (block_id, block) in &func.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let HirInstruction::Call {
                result: Some(result),
                callee: HirCallable::Function(callee_id),
                ..
            } = inst
            {
                if facts.returns_owned.contains(callee_id) {
                    sites.push(MallocSite {
                        result: *result,
                        block: *block_id,
                        inst_idx: idx,
                        release: Release::Intrinsic,
                    });
                }
            }
        }
    }
    sites
}

fn collect_malloc_sites(func: &HirFunction) -> Vec<MallocSite> {
    let mut sites = Vec::new();
    for (block_id, block) in &func.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            let (result, release) = match inst {
                HirInstruction::Call {
                    result: Some(result),
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                    ..
                } => (*result, Release::Intrinsic),
                HirInstruction::Call {
                    result: Some(result),
                    callee: HirCallable::Symbol(name),
                    ..
                } => match symbol_role(name) {
                    Some(SymbolRole::Allocates(free_name)) => (*result, Release::Symbol(free_name)),
                    _ => continue,
                },
                _ => continue,
            };
            sites.push(MallocSite {
                result,
                block: *block_id,
                inst_idx: idx,
                release,
            });
        }
    }
    sites
}

fn analyze_site(func: &HirFunction, site: &MallocSite, facts: &ModuleFacts) -> SiteOutcome {
    // Walk every block, every instruction, every terminator, and
    // for each use of `site.result`:
    //   - record the (block, idx) location and the use *kind*
    //   - bail to `Escaped` immediately on any escape-classified use
    //   - if any use is outside the malloc's own block, bail to
    //     `MultiBlock`
    //
    // Indices use `usize::MAX` as a sentinel for "terminator use".
    // Everything the allocation flows into without leaving the
    // function. A pointer put into a local aggregate and read back out
    // is the same pointer, so the aggregate and what comes out of it
    // hold the claim too and their uses keep the storage live. Without
    // this, storing a box in a struct forfeits it: the store reads as
    // an escape even where the struct never leaves.
    let derived = derived_values(func, site.result);
    let target = site.result;
    let mut last_idx_in_block: Option<usize> = None;
    let mut had_any_use = false;

    for (block_id, block) in &func.blocks {
        // Phis: any phi reading a derived value from a predecessor
        // implies cross-block flow → multi-block.
        for phi in &block.phis {
            if phi.incoming.iter().any(|(v, _)| derived.contains(v)) {
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
            match classify_derived_use(inst, &derived, facts) {
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
        match derived
            .iter()
            .map(|d| classify_terminator_use(&block.terminator, *d))
            .fold(UseKind::None, strongest)
        {
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

/// Everything `root` flows into inside this function, `root` included.
///
/// Only the shapes that carry the same pointer are followed: putting it
/// into an aggregate, taking it back out, and casting it. Following more
/// would widen the live range without making anything reclaimable.
fn derived_values(func: &HirFunction, root: HirId) -> std::collections::HashSet<HirId> {
    let mut set = std::collections::HashSet::new();
    set.insert(root);
    // Blocks are unordered here, so a single sweep can miss a chain
    // that runs backwards through the map. Repeat until nothing new
    // appears; the set only grows and is bounded by the value count.
    loop {
        let before = set.len();
        for block in func.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    HirInstruction::InsertValue {
                        result,
                        aggregate,
                        value,
                        ..
                    } => {
                        if set.contains(value) || set.contains(aggregate) {
                            set.insert(*result);
                        }
                    }
                    HirInstruction::ExtractValue {
                        result, aggregate, ..
                    } => {
                        if set.contains(aggregate) {
                            set.insert(*result);
                        }
                    }
                    HirInstruction::Cast {
                        result, operand, ..
                    } => {
                        if set.contains(operand) {
                            set.insert(*result);
                        }
                    }
                    _ => {}
                }
            }
        }
        if set.len() == before {
            return set;
        }
    }
}

/// The more restrictive of two classifications.
fn strongest(a: UseKind, b: UseKind) -> UseKind {
    match (a, b) {
        (UseKind::Escape, _) | (_, UseKind::Escape) => UseKind::Escape,
        (UseKind::Use, _) | (_, UseKind::Use) => UseKind::Use,
        _ => UseKind::None,
    }
}

/// Classify an instruction against every value the allocation reaches.
///
/// Moving a derived value into or out of an aggregate that is itself
/// derived keeps the pointer inside the set, so it is a use rather than
/// an escape: the aggregate's own uses are classified in turn, and an
/// aggregate that does leave is caught there.
fn classify_derived_use(
    inst: &HirInstruction,
    derived: &std::collections::HashSet<HirId>,
    facts: &ModuleFacts,
) -> UseKind {
    match inst {
        HirInstruction::InsertValue {
            result,
            aggregate,
            value,
            ..
        } if (derived.contains(value) || derived.contains(aggregate))
            && derived.contains(result) =>
        {
            return UseKind::Use;
        }
        HirInstruction::ExtractValue {
            result, aggregate, ..
        } if derived.contains(aggregate) && derived.contains(result) => {
            return UseKind::Use;
        }
        HirInstruction::Cast {
            result, operand, ..
        } if derived.contains(operand) && derived.contains(result) => {
            return UseKind::Use;
        }
        _ => {}
    }
    derived
        .iter()
        .map(|d| classify_inst_use(inst, *d, facts))
        .fold(UseKind::None, strongest)
}

/// Use-kind classification for a single instruction against one
/// target value.
#[derive(Clone, Copy)]
enum UseKind {
    None,
    Use,
    Escape,
}

fn classify_inst_use(inst: &HirInstruction, target: HirId, facts: &ModuleFacts) -> UseKind {
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
            if !args.iter().any(|a| *a == target) {
                return UseKind::None;
            }
            // A call the pass knows only reads through the pointer does
            // not end the caller's claim, so it extends the live range
            // rather than forfeiting it. Reading a boxed value is such a
            // call, and treating it as an escape would mean no box that
            // is ever read could be released.
            match callee {
                HirCallable::Symbol(name) if symbol_role(name) == Some(SymbolRole::Borrows) => {
                    UseKind::Use
                }
                // A parameter the callee only borrows leaves the caller
                // holding the storage. Handing a pointer to one extends
                // the live range rather than forfeiting it, which is
                // what lets a temporary passed straight into the next
                // call still be released afterwards.
                _ if facts.call_only_borrows(callee, args, target) => UseKind::Use,
                _ => UseKind::Escape,
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

fn insert_free_after(
    func: &mut HirFunction,
    block_id: HirId,
    after_idx: usize,
    target: HirId,
    release: Release,
) {
    let free_inst = HirInstruction::Call {
        result: None,
        callee: match release {
            Release::Intrinsic => HirCallable::Intrinsic(Intrinsic::Free),
            Release::Symbol(name) => HirCallable::Symbol(name.to_string()),
        },
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
            is_fiber: false,
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
        let stats = run_function(&mut f, &ModuleFacts::default());
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

        let stats = run_function(&mut f, &ModuleFacts::default());
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

        let stats = run_function(&mut f, &ModuleFacts::default());
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

        let stats = run_function(&mut f, &ModuleFacts::default());
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

        let stats = run_function(&mut f, &ModuleFacts::default());
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 0);
        assert_eq!(stats.no_use_skipped, 1);
    }

    /// `fn make() -> *i64 { let p = malloc(8); return p }` — a
    /// constructor, which hands its allocation to whoever called it.
    fn build_constructor() -> HirFunction {
        let mut f = HirFunction::new(
            InternedString::new_global("make"),
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
        f
    }

    /// `fn use_it() -> i64 { let p = make(); return load(p) }`
    fn build_caller(callee_key: HirId) -> HirFunction {
        let mut f = HirFunction::new(
            InternedString::new_global("use_it"),
            empty_sig(HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let got = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let loaded = add_inst_val(&mut f, HirType::I64);
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(got),
            callee: HirCallable::Function(callee_key),
            args: vec![],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::Load {
            result: loaded,
            ty: HirType::I64,
            ptr: got,
            align: 8,
            volatile: false,
        });
        block.terminator = HirTerminator::Return {
            values: vec![loaded],
        };
        f
    }

    /// A constructor's result belongs to the caller, so the caller is
    /// where it gets released. Nothing inside the constructor can do
    /// it: the allocation leaves by being returned.
    #[test]
    fn a_caller_releases_what_a_constructor_returned() {
        let ctor = build_constructor();
        let ctor_key = HirId::new();
        let caller = build_caller(ctor_key);
        let caller_key = HirId::new();

        let mut m = HirModule::new(InternedString::new_global("m"));
        m.functions.insert(ctor_key, ctor);
        m.functions.insert(caller_key, caller);

        let stats = run_module(&mut m);
        assert!(stats.frees_inserted >= 1, "the caller should release it");

        let caller = m.functions.get(&caller_key).unwrap();
        let block = caller.blocks.values().next().unwrap();
        let freed = block.instructions.iter().any(|i| {
            matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    ..
                }
            )
        });
        assert!(
            freed,
            "expected a release in the caller, got {:?}",
            block.instructions
        );
    }

    /// A function handing back something it was given owns nothing, so
    /// its caller must not release the result.
    #[test]
    fn a_caller_does_not_release_what_was_merely_passed_through() {
        let mut passthrough = HirFunction::new(
            InternedString::new_global("passthrough"),
            HirFunctionSignature {
                params: vec![crate::hir::HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global("p"),
                    ty: HirType::Ptr(Box::new(HirType::I64)),
                    attributes: Default::default(),
                    ownership: crate::hir::ParamOwnership::Borrowed,
                }],
                returns: vec![HirType::Ptr(Box::new(HirType::I64))],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
                is_fiber: false,
                effects: vec![],
                is_pure: false,
            },
        );
        let entry = HirId::new();
        passthrough.entry_block = entry;
        passthrough.blocks.clear();
        passthrough.blocks.insert(entry, HirBlock::new(entry));
        let param_id = passthrough.signature.params[0].id;
        passthrough.blocks.get_mut(&entry).unwrap().terminator = HirTerminator::Return {
            values: vec![param_id],
        };

        let key = HirId::new();
        let caller = build_caller(key);
        let caller_key = HirId::new();
        let mut m = HirModule::new(InternedString::new_global("m2"));
        m.functions.insert(key, passthrough);
        m.functions.insert(caller_key, caller);

        run_module(&mut m);
        let caller = m.functions.get(&caller_key).unwrap();
        let block = caller.blocks.values().next().unwrap();
        let freed = block.instructions.iter().any(|i| {
            matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    ..
                }
            )
        });
        assert!(
            !freed,
            "a pass-through returns storage it does not own; releasing it would \
             free the caller's own pointer"
        );
    }

    /// A loop whose header phi carries an allocation replaced each
    /// iteration, which is what an accumulator is:
    ///
    /// ```text
    /// entry:  p0 = malloc; -> header
    /// header: phi = [p0 from entry, p1 from body]; -> body | exit
    /// body:   load phi; p1 = malloc; -> header
    /// exit:   load phi; return
    /// ```
    fn build_accumulator_loop(escaping: bool) -> (HirFunction, HirId) {
        let mut f = HirFunction::new(
            InternedString::new_global("accumulate"),
            empty_sig(HirType::I64),
        );
        let entry = f.entry_block;
        let header = HirId::new();
        let body = HirId::new();
        let exit = HirId::new();
        f.blocks.insert(header, HirBlock::new(header));
        f.blocks.insert(body, HirBlock::new(body));
        f.blocks.insert(exit, HirBlock::new(exit));

        let ptr_ty = HirType::Ptr(Box::new(HirType::I64));
        let size = add_const(&mut f, HirType::I64, HirConstant::I64(8));
        let cond = add_const(&mut f, HirType::Bool, HirConstant::Bool(true));
        let p0 = add_inst_val(&mut f, ptr_ty.clone());
        let p1 = add_inst_val(&mut f, ptr_ty.clone());
        let acc = add_inst_val(&mut f, ptr_ty.clone());
        let l_body = add_inst_val(&mut f, HirType::I64);
        let l_exit = add_inst_val(&mut f, HirType::I64);

        let alloc = |r: HirId, sz: HirId| HirInstruction::Call {
            result: Some(r),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![sz],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        };

        let b = f.blocks.get_mut(&entry).unwrap();
        b.instructions.push(alloc(p0, size));
        b.terminator = HirTerminator::Branch { target: header };

        let h = f.blocks.get_mut(&header).unwrap();
        h.phis.push(crate::hir::HirPhi {
            result: acc,
            ty: ptr_ty.clone(),
            incoming: vec![(p0, entry), (p1, body)],
        });
        h.terminator = HirTerminator::CondBranch {
            condition: cond,
            true_target: body,
            false_target: exit,
        };

        let bd = f.blocks.get_mut(&body).unwrap();
        bd.instructions.push(HirInstruction::Load {
            result: l_body,
            ty: HirType::I64,
            ptr: acc,
            align: 8,
            volatile: false,
        });
        if escaping {
            // Hand the accumulator to something that might keep it.
            bd.instructions.push(HirInstruction::Call {
                result: None,
                callee: HirCallable::Symbol("keeps_it".to_string()),
                args: vec![acc],
                type_args: Vec::new(),
                const_args: Vec::new(),
                is_tail: false,
            });
        }
        bd.instructions.push(alloc(p1, size));
        bd.terminator = HirTerminator::Branch { target: header };

        // Loop detection reads the edge lists rather than deriving
        // them from terminators, so a hand-built function fills them in.
        f.blocks.get_mut(&entry).unwrap().successors = vec![header];
        f.blocks.get_mut(&header).unwrap().successors = vec![body, exit];
        f.blocks.get_mut(&body).unwrap().successors = vec![header];
        f.blocks.get_mut(&header).unwrap().predecessors = vec![entry, body];
        f.blocks.get_mut(&body).unwrap().predecessors = vec![header];
        f.blocks.get_mut(&exit).unwrap().predecessors = vec![header];

        let ex = f.blocks.get_mut(&exit).unwrap();
        ex.instructions.push(HirInstruction::Load {
            result: l_exit,
            ty: HirType::I64,
            ptr: acc,
            align: 8,
            volatile: false,
        });
        ex.terminator = HirTerminator::Return {
            values: vec![l_exit],
        };
        (f, acc)
    }

    /// The object the accumulator replaces is released in the body,
    /// after the body has read it. What leaves through the exit is the
    /// one the last iteration built, which the body never read.
    #[test]
    fn an_accumulator_releases_what_it_replaces() {
        let (mut f, acc) = build_accumulator_loop(false);
        let body_id = f
            .blocks
            .iter()
            .find(|(_, blk)| {
                blk.instructions
                    .iter()
                    .any(|i| matches!(i, HirInstruction::Load { .. }))
                    && matches!(blk.terminator, HirTerminator::Branch { .. })
            })
            .map(|(id, _)| *id)
            .expect("the body");

        assert_eq!(release_loop_carried(&mut f, &ModuleFacts::default()), 1);

        let body = &f.blocks[&body_id];
        let free_at = body.instructions.iter().position(|i| {
            matches!(i, HirInstruction::Call { callee: HirCallable::Intrinsic(Intrinsic::Free), args, .. } if args == &vec![acc])
        });
        let read_at = body
            .instructions
            .iter()
            .position(|i| matches!(i, HirInstruction::Load { .. }));
        let free_at = free_at.expect("the accumulator should be released in the body");
        assert!(
            free_at > read_at.expect("the read"),
            "releasing before the body reads it would be a use after free"
        );
    }

    /// If anything might keep the accumulator, it is not ours to
    /// release.
    #[test]
    fn an_accumulator_that_might_be_kept_is_left_alone() {
        let (mut f, _) = build_accumulator_loop(true);
        assert_eq!(
            release_loop_carried(&mut f, &ModuleFacts::default()),
            0,
            "a call that might keep the pointer forfeits the release"
        );
    }

    /// The shape a boxed value takes: a runtime call hands back
    /// storage, the pointer goes into a local aggregate, comes back
    /// out, and is read through. Every step used to forfeit the claim,
    /// so nothing was ever released and a loop that boxed leaked once
    /// per iteration.
    fn build_box_into_struct_and_read() -> (HirFunction, HirId) {
        let mut f = HirFunction::new(
            InternedString::new_global("box_local"),
            empty_sig(HirType::F64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));

        let bag_ty = HirType::Struct(crate::hir::HirStructType {
            name: None,
            fields: vec![HirType::I64],
            packed: false,
        });
        let scalar = add_const(&mut f, HirType::F64, HirConstant::F64(1.5));
        let boxed = add_inst_val(&mut f, HirType::I64);
        let undef_bag = add_inst_val(&mut f, bag_ty.clone());
        let bag = add_inst_val(&mut f, bag_ty.clone());
        let back = add_inst_val(&mut f, HirType::I64);
        let read = add_inst_val(&mut f, HirType::F64);

        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(boxed),
            callee: HirCallable::Symbol("zyntax_box_f64".to_string()),
            args: vec![scalar],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::InsertValue {
            result: bag,
            ty: bag_ty.clone(),
            aggregate: undef_bag,
            value: boxed,
            indices: vec![0],
        });
        block.instructions.push(HirInstruction::ExtractValue {
            result: back,
            ty: HirType::I64,
            aggregate: bag,
            indices: vec![0],
        });
        block.instructions.push(HirInstruction::Call {
            result: Some(read),
            callee: HirCallable::Symbol("zyntax_box_get_f64".to_string()),
            args: vec![back],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return { values: vec![read] };
        (f, boxed)
    }

    #[test]
    fn a_boxed_value_is_released() {
        let (mut f, boxed) = build_box_into_struct_and_read();
        let stats = run_function(&mut f, &ModuleFacts::default());
        assert_eq!(stats.mallocs_scanned, 1, "the box call is an allocation");
        assert_eq!(stats.frees_inserted, 1, "and it should be released");

        let block = f.blocks.values().next().unwrap();
        let released = block.instructions.iter().any(|i| match i {
            HirInstruction::Call { callee, args, .. } => {
                matches!(callee, HirCallable::Symbol(n) if n == "zyntax_box_free")
                    && args == &vec![boxed]
            }
            _ => false,
        });
        assert!(released, "expected a zyntax_box_free of the boxed pointer");
    }

    /// The release must come after the read, not before it.
    #[test]
    fn a_boxed_value_is_released_after_its_last_read() {
        let (mut f, _) = build_box_into_struct_and_read();
        run_function(&mut f, &ModuleFacts::default());
        let block = f.blocks.values().next().unwrap();
        let idx = |name: &str| {
            block.instructions.iter().position(|i| match i {
                HirInstruction::Call { callee, .. } => {
                    matches!(callee, HirCallable::Symbol(n) if n == name)
                }
                _ => false,
            })
        };
        let read = idx("zyntax_box_get_f64").expect("the read");
        let free = idx("zyntax_box_free").expect("the release");
        assert!(
            free > read,
            "releasing before the read would be a use after free"
        );
    }

    /// A box handed out of the function is not ours to release.
    #[test]
    fn a_boxed_value_that_escapes_is_left_alone() {
        let mut f = HirFunction::new(
            InternedString::new_global("box_escapes"),
            empty_sig(HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let scalar = add_const(&mut f, HirType::F64, HirConstant::F64(1.5));
        let boxed = add_inst_val(&mut f, HirType::I64);
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(boxed),
            callee: HirCallable::Symbol("zyntax_box_f64".to_string()),
            args: vec![scalar],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return {
            values: vec![boxed],
        };

        let stats = run_function(&mut f, &ModuleFacts::default());
        assert_eq!(
            stats.frees_inserted, 0,
            "a returned box belongs to the caller"
        );
        assert_eq!(stats.escapes_skipped, 1);
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

        let stats = run_function(&mut f, &ModuleFacts::default());
        assert_eq!(stats.mallocs_scanned, 1);
        assert_eq!(stats.frees_inserted, 0);
        // Either Escape or NoUse outcome — Escape because the Free
        // Call's args include `ptr` and our classifier treats that
        // as an escape (refusing to double-insert).
        assert_eq!(stats.escapes_skipped, 1);
    }
}
