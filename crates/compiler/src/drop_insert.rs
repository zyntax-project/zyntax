//! Compile-time speculative drop-site analysis.
//!
//! Pairs each allocation with a release at the allocation's last use,
//! when static analysis can prove the allocation does not escape the
//! function. Implements the *speculative drop-site* memory strategy
//! that's the default in `CompilationConfig.memory_strategy` (the
//! opt-in GC variants sit alongside it as alternatives selected per
//! program).
//!
//! ## What counts as an allocation
//!
//! * `Call(Intrinsic::Malloc)`, released by `Free`.
//! * A runtime symbol that hands back storage: a box (`zyntax_box_*`,
//!   released by `zyntax_box_free`) and a string from the IO or string
//!   plugins (released by `$IO$string_free`). An extern function
//!   standing for such a symbol counts the same.
//! * A call to a function of this module that returns storage it
//!   allocated and lets leave no other way
//!   ([`functions_returning_owned_storage`]); the caller owns the
//!   result and releases it by its type.
//!
//! ## Where the release goes
//!
//! Every use of the allocation is classified: a read, a borrow by a
//! callee that keeps nothing, a store through the pointer are uses; a
//! store of the pointer, a hand to a callee that keeps it, a return, a
//! capture are escapes, and any escape forfeits the release. The
//! aliases of the storage (casts, aggregates it is put into and taken
//! out of, the result of a callee that hands its argument back) are
//! tracked as further names for it.
//!
//! Within one block the release follows the last use. Across blocks a
//! backward liveness places it wherever the value is live and no
//! successor keeps it: after a last use, or on an edge into a block that
//! reads it nowhere (an entry insertion, or a block spliced into the
//! edge). This cross-block placement and the transfer through returned
//! storage are on for a module that asks (`HirModule::automatic_release`),
//! for a language whose programs never release anything by hand.
//!
//! A value that reaches a phi is not released by its own name past the
//! merge; the phi owns it instead when every incoming is owned storage
//! nothing else keeps ([`release_owned_phis`]). A loop header's phi is
//! the accumulator, released after the body has read it; a string
//! accumulator seeded from anywhere else is copied on the way in.
//!
//! `ZYNTAX_TRACE_DROP=1` prints what was decided and why. The pass may
//! run more than once over a function: a release it inserted is
//! recognised and never doubled.
//!
//! ## What this does NOT do
//!
//! * Escape through stores: `Store { value: M, ptr: P }` where `P` is
//!   itself a local allocation that doesn't escape is technically still
//!   safe to free transitively, but tracking the alias chain takes a
//!   points-to analysis. We treat any such Store as an escape.
//! * Storage handed to an owning parameter is the callee's; a callee
//!   that stores or returns its parameter must say so with `Owned`.

use std::collections::HashSet;

use crate::hir::{
    HirCallable, HirConstant, HirFunction, HirId, HirInstruction, HirModule, HirTerminator,
    HirType, Intrinsic,
};

/// Releases a string the IO plugin's string functions hand out.
const STRING_FREE: &str = "$IO$string_free";
/// A fresh copy of a string, owned by the caller.
const STRING_COPY: &str = "$IO$string_copy";
/// Releases a dynamic box and whatever it owns.
const BOX_FREE: &str = "zyntax_box_free";

/// Whether storage is released across blocks and through returned
/// storage: asked for by the module (`HirModule::automatic_release`), or
/// by `ZYNTAX_DROP_GLUE=1`, which turns on the type release glue as well.
/// `ZYNTAX_DISABLE_AUTOMATIC_RELEASE=1` overrides both; safe, and what to
/// try first when a program reads freed memory.
fn automatic_release_for(module: &HirModule) -> bool {
    if std::env::var_os("ZYNTAX_DISABLE_AUTOMATIC_RELEASE").is_some() {
        return false;
    }
    module.automatic_release || crate::drop_glue::enabled()
}

/// Runtime symbols whose result is what the box they were given holds: a
/// name for storage the box owns, good for as long as the box is.
fn symbol_result_aliases_arg(symbol: &str) -> bool {
    matches!(
        symbol,
        "zyntax_box_get_str" | "zyntax_box_get_opaque" | "zyntax_box_data"
    )
}

/// A dynamic box.
fn is_box(ty: &HirType) -> bool {
    crate::zrtl::is_dynamic_box_pointer(ty)
}

/// A string: the plugins hand these out as fresh storage.
fn is_string(ty: &HirType) -> bool {
    matches!(ty, HirType::Ptr(inner) if **inner == HirType::I8)
}

/// Whether a plugin symbol's string result is fresh storage the caller
/// owns. Every string the IO and string plugins return is one.
fn makes_strings(symbol: &str) -> bool {
    symbol.starts_with("$IO$") || symbol.starts_with("$String$")
}

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
    /// See [`automatic_release_for`].
    automatic_release: bool,
    returns_owned: std::collections::HashSet<HirId>,
    /// Externs whose result is a fresh string: a call to one is an
    /// allocation the caller releases with the string free.
    string_makers: std::collections::HashSet<HirId>,
    /// Per function, which parameters it may hand back as its result. The
    /// result of such a call is another name for the argument, so the
    /// argument lives as long as the result does.
    returns_param: std::collections::HashMap<HirId, Vec<bool>>,
    /// The runtime symbol behind each extern function.
    extern_links: std::collections::HashMap<HirId, String>,
    /// Function names by key, for the trace.
    names: std::collections::HashMap<HirId, String>,
    /// Per callee, which parameters are only borrowed. Keyed on the
    /// module's own key for the function, which is what a call names and
    /// is not always the function's `id` field.
    borrowed_params: std::collections::HashMap<HirId, Vec<bool>>,
    /// The release function for a named type that owns another, by the
    /// type's name. An allocation of such a type is released through
    /// this rather than by a bare free.
    glue: std::collections::HashMap<zyntax_typed_ast::InternedString, HirId>,
}

impl ModuleFacts {
    fn build(module: &HirModule) -> Self {
        let mut borrowed_params = std::collections::HashMap::new();
        for (key, func) in module.functions.iter() {
            // What a host does with a pointer is unknown here, so an
            // extern borrows only when its symbol is one the pass knows
            // reads without keeping; any other extern is an escape.
            let extern_borrows = func.is_external
                && func
                    .link_name
                    .as_deref()
                    .is_some_and(|name| symbol_role(name).is_some_and(|r| r.borrows_args));
            borrowed_params.insert(
                *key,
                func.signature
                    .params
                    .iter()
                    .map(|p| {
                        if func.is_external {
                            return extern_borrows;
                        }
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
        let mut glue = std::collections::HashMap::new();
        for (key, func) in module.functions.iter() {
            if let Some(ty) = crate::drop_glue::glue_target(func) {
                glue.insert(ty, *key);
            }
        }
        let string_makers = module
            .functions
            .iter()
            .filter(|(_, f)| {
                f.is_external
                    && f.link_name.as_deref().is_some_and(makes_strings)
                    && matches!(f.signature.returns.as_slice(), [ty] if is_string(ty))
            })
            .map(|(key, _)| *key)
            .collect();
        let extern_links = module
            .functions
            .iter()
            .filter(|(_, f)| f.is_external)
            .filter_map(|(key, f)| f.link_name.clone().map(|n| (*key, n)))
            .collect();
        let names = module
            .functions
            .iter()
            .map(|(key, f)| (*key, f.name.resolve_global().unwrap_or_default()))
            .collect();
        let mut facts = Self {
            automatic_release: automatic_release_for(module),
            returns_owned: std::collections::HashSet::new(),
            string_makers,
            returns_param: std::collections::HashMap::new(),
            extern_links,
            names,
            borrowed_params,
            glue,
        };
        // A function returning a callee's result that is itself an
        // argument returns its own parameter, so this grows until it
        // stops; owned-returning functions likewise.
        loop {
            let returned = functions_returning_params(module, &facts);
            if returned == facts.returns_param {
                break;
            }
            facts.returns_param = returned;
        }
        loop {
            let owned = functions_returning_owned_storage(module, &facts);
            if owned.len() == facts.returns_owned.len() {
                break;
            }
            facts.returns_owned = owned;
        }
        if trace_enabled() {
            let mut names: Vec<String> = facts
                .returns_owned
                .iter()
                .filter_map(|k| module.functions.get(k))
                .map(|f| f.name.resolve_global().unwrap_or_default())
                .collect();
            names.sort();
            eprintln!("[drop] returning owned storage: {}", names.join(" "));
        }
        facts
    }

    /// The result of `callee` that is another name for `args[i]`: the
    /// callee may return that parameter, or hands back what the box it
    /// was given holds.
    fn result_aliases_arg(&self, callee: &HirCallable, args: &[HirId], value: HirId) -> bool {
        match callee {
            HirCallable::Symbol(name) => {
                symbol_result_aliases_arg(name) && args.iter().any(|a| *a == value)
            }
            HirCallable::Function(id) => {
                if let Some(link) = self.extern_links.get(id) {
                    return symbol_result_aliases_arg(link) && args.iter().any(|a| *a == value);
                }
                let Some(returned) = self.returns_param.get(id) else {
                    return false;
                };
                args.iter()
                    .enumerate()
                    .any(|(i, a)| *a == value && returned.get(i).copied().unwrap_or(false))
            }
            _ => false,
        }
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

/// Release storage that reaches a phi.
///
/// A phi is another name for whatever arrives on the edge taken, so the
/// per-site analysis leaves anything that flows into one alone: a
/// release placed by the merged name's liveness would name one incoming
/// where another was live. The phi itself can own the storage instead.
/// Then the phi is the definition, and its value is released wherever it
/// has been read for the last time and no path keeps it; an edge into
/// another owning phi hands the value on rather than releasing it.
///
/// A phi owns its value when every incoming is storage this function
/// owns, a site or another owning phi, that nothing reads once it has
/// arrived and no other phi keeps. Ownership is decided for all phis
/// together, since one hands on to the next: every phi starts as a
/// candidate and those with an incoming that fails drop out until none
/// do.
///
/// A loop header's phi is the accumulator `sum = sum + a`: the value
/// round the back edge must be built by the body (or be the phi itself,
/// carried unchanged), so the phi never carries one object twice, and it
/// may be read in the body since the release frees only what the body
/// consumed. A string arriving on an entry edge from anywhere else is
/// copied there first, which is what lets an accumulator start from a
/// literal.
fn release_owned_phis(func: &mut HirFunction, facts: &ModuleFacts) -> usize {
    use crate::analysis::{DominatorTree, LoopForest};

    let sites: std::collections::HashMap<HirId, Release> = collect_owned_sites(func, facts)
        .iter()
        .map(|s| (s.result, s.release))
        .collect();
    if sites.is_empty() {
        return 0;
    }
    let dt = DominatorTree::new(func);
    let forest = LoopForest::detect(func, &dt);
    let bodies: std::collections::HashMap<HirId, std::collections::HashSet<HirId>> = forest
        .loops()
        .iter()
        .map(|lp| (lp.header, lp.body.iter().copied().collect()))
        .collect();
    let phi_blocks: std::collections::HashMap<HirId, HirId> = func
        .blocks
        .iter()
        .flat_map(|(b, block)| block.phis.iter().map(move |p| (p.result, *b)))
        .collect();

    // Copies to make on entry edges: the predecessor block, the phi, the
    // position of its incoming, and the value to copy.
    let mut copies: Vec<(HirId, HirId, usize, HirId)> = Vec::new();
    // Every phi to begin with; a join's phi only under automatic release,
    // since a program releasing by hand may release what arrives at one.
    let mut candidates: std::collections::HashSet<HirId> = phi_blocks
        .iter()
        .filter(|(_, block)| facts.automatic_release || bodies.contains_key(block))
        .map(|(p, _)| *p)
        .collect();
    loop {
        let before = candidates.len();
        copies.clear();
        for (block_id, block) in &func.blocks {
            let body = bodies.get(block_id);
            for phi in &block.phis {
                if !candidates.contains(&phi.result) {
                    continue;
                }
                match phi_incomings_owned(func, facts, &sites, &candidates, body, phi) {
                    Some(seed_copies) => copies.extend(seed_copies),
                    None => {
                        candidates.remove(&phi.result);
                    }
                }
            }
        }
        // The phi's own value: read only by borrowers, and handed on only
        // to phis that own what they are handed.
        let dropped: Vec<HirId> = candidates
            .iter()
            .copied()
            .filter(|p| {
                let derived = derived_values_local(func, *p, facts);
                !uses_are_all_borrows(func, &derived, facts)
                    || phis_using_any(func, &derived)
                        .iter()
                        .any(|other| !candidates.contains(other))
            })
            .collect();
        for p in dropped {
            candidates.remove(&p);
        }
        if candidates.len() == before {
            break;
        }
    }
    if candidates.is_empty() {
        return 0;
    }

    // What releases each owning phi's value: what releases what reaches
    // it, agreed on by every incoming or the phi is not owned.
    let mut releases: std::collections::HashMap<HirId, Release> = std::collections::HashMap::new();
    for _ in 0..candidates.len() {
        for (_, block) in &func.blocks {
            for phi in &block.phis {
                if !candidates.contains(&phi.result) || releases.contains_key(&phi.result) {
                    continue;
                }
                let mut agreed: Option<Release> = None;
                let mut known = true;
                for (val, _) in &phi.incoming {
                    let r = if *val == phi.result {
                        continue;
                    } else if let Some(r) = sites.get(val) {
                        *r
                    } else if let Some(r) = releases.get(val) {
                        *r
                    } else if copies
                        .iter()
                        .any(|(_, p, _, v)| *p == phi.result && *v == *val)
                    {
                        Release::Symbol(STRING_FREE)
                    } else {
                        known = false;
                        break;
                    };
                    if agreed.is_none_or(|have| have == r) {
                        agreed = Some(r);
                    } else {
                        known = false;
                        break;
                    }
                }
                if let (true, Some(r)) = (known, agreed) {
                    releases.insert(phi.result, r);
                }
            }
        }
    }
    candidates.retain(|p| releases.contains_key(p));
    copies.retain(|(_, p, _, _)| candidates.contains(p));

    for (pred, phi_result, index, val) in copies {
        let copy = HirId::new();
        let ty = func
            .values
            .get(&val)
            .map(|v| v.ty.clone())
            .unwrap_or(HirType::Ptr(Box::new(HirType::I8)));
        func.values.insert(
            copy,
            crate::hir::HirValue {
                id: copy,
                ty,
                kind: crate::hir::HirValueKind::Instruction,
                uses: Default::default(),
                span: None,
            },
        );
        if let Some(block) = func.blocks.get_mut(&pred) {
            block.instructions.push(HirInstruction::Call {
                result: Some(copy),
                callee: HirCallable::Symbol(STRING_COPY.to_string()),
                args: vec![val],
                type_args: Vec::new(),
                const_args: Vec::new(),
                is_tail: false,
            });
        }
        for block in func.blocks.values_mut() {
            for phi in block.phis.iter_mut() {
                if phi.result == phi_result {
                    if let Some(incoming) = phi.incoming.get_mut(index) {
                        incoming.0 = copy;
                    }
                }
            }
        }
    }

    // One phi at a time, each analysed against the blocks as the last
    // one left them: a release placed at a block's entry or on a split
    // edge moves what an earlier analysis counted.
    let owned_phis: Vec<(HirId, HirId, Release)> = func
        .blocks
        .iter()
        .flat_map(|(b, block)| block.phis.iter().map(move |p| (*b, p.result)))
        .filter_map(|(b, p)| releases.get(&p).map(|r| (b, p, *r)))
        .collect();
    let mut inserted = 0;
    for (block_id, phi_result, release) in owned_phis {
        let derived = derived_values_local(func, phi_result, facts);
        // Leaving a block along an edge into an owning phi hands the
        // value on.
        let transfer_out: std::collections::HashSet<HirId> = func
            .blocks
            .values()
            .flat_map(|b| b.phis.iter())
            .filter(|other| releases.contains_key(&other.result))
            .flat_map(|other| other.incoming.iter())
            .filter(|(v, _)| derived.contains(v))
            .map(|(_, pred)| *pred)
            .collect();
        let site = MallocSite {
            result: phi_result,
            block: block_id,
            inst_idx: usize::MAX,
            release,
        };
        if let Some(points) = drop_points_transferring(func, &site, &derived, facts, &transfer_out)
        {
            inserted += apply_points(func, points, phi_result, release);
        }
    }
    inserted
}

/// Whether every incoming of `phi` is storage the phi may own, given the
/// sites and the phis still candidates. Returns the string seeds to copy
/// on their entry edges, or `None` where an incoming fails.
fn phi_incomings_owned(
    func: &HirFunction,
    facts: &ModuleFacts,
    sites: &std::collections::HashMap<HirId, Release>,
    candidates: &std::collections::HashSet<HirId>,
    body: Option<&std::collections::HashSet<HirId>>,
    phi: &crate::hir::HirPhi,
) -> Option<Vec<(HirId, HirId, usize, HirId)>> {
    let merge = phi_block_of(func, phi.result)?;
    let mut copies = Vec::new();
    for (index, (val, pred)) in phi.incoming.iter().enumerate() {
        // Carried round unchanged: one object, handed back to itself.
        if *val == phi.result {
            continue;
        }
        let round_back_edge = body.is_some_and(|b| b.contains(pred));
        let owned = sites.contains_key(val) || candidates.contains(val);
        // Nothing but owning phis may keep the incoming.
        let kept_elsewhere = phis_using(func, *val)
            .iter()
            .any(|p| *p != phi.result && !candidates.contains(p));
        let derived = derived_values_local(func, *val, facts);
        let borrowed_only = uses_are_all_borrows(func, &derived, facts);
        let fine = if round_back_edge {
            // Built by the body, so the phi never carries one object
            // twice.
            let fresh = body.is_some_and(|b| {
                b.iter()
                    .filter(|id| **id != merge)
                    .filter_map(|id| func.blocks.get(id))
                    .any(|blk| {
                        defines_value(blk, *val) || blk.phis.iter().any(|p| p.result == *val)
                    })
            });
            owned && !kept_elsewhere && fresh && borrowed_only
        } else {
            // Read by nothing once it has arrived: the phi's release
            // would free it under another name.
            owned
                && !kept_elsewhere
                && borrowed_only
                && !used_past_merge(func, &derived, def_block_of(func, *val), merge)
        };
        if fine {
            continue;
        }
        if trace_enabled() {
            eprintln!(
                "[drop] {}: phi {:?} cannot own incoming {:?} (owned {owned}, kept elsewhere {kept_elsewhere}, borrowed only {borrowed_only}, back edge {round_back_edge})",
                func.name.resolve_global().unwrap_or_default(),
                phi.result,
                val
            );
        }
        // An accumulator's seed a string from anywhere: a copy is ours.
        if body.is_some() && !round_back_edge && is_string(&phi.ty) && is_string_value(func, *val) {
            copies.push((*pred, phi.result, index, *val));
            continue;
        }
        return None;
    }
    Some(copies)
}

/// The block holding the phi `result`.
fn phi_block_of(func: &HirFunction, result: HirId) -> Option<HirId> {
    func.blocks
        .iter()
        .find(|(_, b)| b.phis.iter().any(|p| p.result == result))
        .map(|(id, _)| *id)
}

/// The block defining `value`, by instruction or phi.
fn def_block_of(func: &HirFunction, value: HirId) -> Option<HirId> {
    func.blocks
        .iter()
        .find(|(_, b)| defines_value(b, value) || b.phis.iter().any(|p| p.result == value))
        .map(|(id, _)| *id)
}

/// Whether any of `names` is read on a path from `merge` that does not
/// pass the block defining them again: the merged storage is still
/// referred to under its old name past the merge.
fn used_past_merge(
    func: &HirFunction,
    names: &std::collections::HashSet<HirId>,
    def_block: Option<HirId>,
    merge: HirId,
) -> bool {
    let Some(def_block) = def_block else {
        // Defined by nothing this walks: a parameter, read anywhere.
        return true;
    };
    if def_block == merge {
        return false;
    }
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![merge];
    while let Some(b) = stack.pop() {
        if b == def_block || !seen.insert(b) {
            continue;
        }
        let Some(block) = func.blocks.get(&b) else {
            continue;
        };
        let reads = block
            .instructions
            .iter()
            .any(|i| i.operands().iter().any(|o| names.contains(o)))
            || terminator_operands(&block.terminator)
                .iter()
                .any(|o| names.contains(o));
        if reads {
            return true;
        }
        stack.extend(successors_of(block));
    }
    false
}

/// The phis that read `value` on some edge.
fn phis_using(func: &HirFunction, value: HirId) -> Vec<HirId> {
    let mut out = Vec::new();
    for block in func.blocks.values() {
        for phi in &block.phis {
            if phi.incoming.iter().any(|(v, _)| *v == value) {
                out.push(phi.result);
            }
        }
    }
    out
}

/// The phis that read any of `values` on some edge.
fn phis_using_any(func: &HirFunction, values: &std::collections::HashSet<HirId>) -> Vec<HirId> {
    let mut out = Vec::new();
    for block in func.blocks.values() {
        for phi in &block.phis {
            if phi.incoming.iter().any(|(v, _)| values.contains(v)) {
                out.push(phi.result);
            }
        }
    }
    out
}

/// Whether `value` is read by nothing but the phi `phi_result`.
fn only_use_is_phi(func: &HirFunction, value: HirId, phi_result: HirId) -> bool {
    for block in func.blocks.values() {
        for phi in &block.phis {
            if phi.result != phi_result && phi.incoming.iter().any(|(v, _)| *v == value) {
                return false;
            }
        }
        for inst in &block.instructions {
            if inst.operands().contains(&value) {
                return false;
            }
        }
        if terminator_operands(&block.terminator).contains(&value) {
            return false;
        }
    }
    true
}

fn is_string_value(func: &HirFunction, value: HirId) -> bool {
    func.values.get(&value).is_some_and(|v| is_string(&v.ty))
}

/// The values a terminator reads.
fn terminator_operands(term: &HirTerminator) -> Vec<HirId> {
    match term {
        HirTerminator::Return { values } => values.clone(),
        HirTerminator::CondBranch { condition, .. } => vec![*condition],
        HirTerminator::Switch { value, .. } => vec![*value],
        HirTerminator::Invoke { args, .. } => args.clone(),
        HirTerminator::PatternMatch { value, .. } => vec![*value],
        _ => Vec::new(),
    }
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
                if trace_enabled() {
                    let callee = match inst {
                        HirInstruction::Call {
                            callee: HirCallable::Function(id),
                            ..
                        } => facts
                            .names
                            .get(id)
                            .cloned()
                            .unwrap_or_else(|| format!("{id:?}")),
                        _ => String::new(),
                    };
                    eprintln!("[drop]   escapes through {inst:?} {callee}");
                }
                return false;
            }
        }
        for d in derived {
            if matches!(
                classify_terminator_use(&block.terminator, *d),
                UseKind::Escape
            ) {
                if trace_enabled() {
                    eprintln!("[drop]   escapes through terminator {:?}", block.terminator);
                }
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

/// Per function, which parameters may come back as its result, by
/// position. A parameter that reaches a return through phis, casts, or a
/// call to a function that returns its own parameter counts.
fn functions_returning_params(
    module: &HirModule,
    facts: &ModuleFacts,
) -> std::collections::HashMap<HirId, Vec<bool>> {
    let mut out = std::collections::HashMap::new();
    for (key, func) in module.functions.iter() {
        if func.is_external {
            continue;
        }
        let returned: Vec<HirId> = func
            .blocks
            .values()
            .filter_map(|b| match &b.terminator {
                HirTerminator::Return { values } => Some(values.iter().copied()),
                _ => None,
            })
            .flatten()
            .collect();
        // A parameter's value is the one of `Parameter` kind at its
        // position; the signature's own ids name nothing in the body.
        let flags: Vec<bool> = (0..func.signature.params.len())
            .map(|i| {
                func.values
                    .values()
                    .filter(|v| matches!(v.kind, crate::hir::HirValueKind::Parameter(n) if n as usize == i))
                    .any(|v| {
                        let names = derived_values_with(func, v.id, true, Some(facts));
                        returned.iter().any(|r| names.contains(r))
                    })
            })
            .collect();
        out.insert(*key, flags);
    }
    out
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
        let sites = collect_owned_sites(func, facts);
        if sites.is_empty() {
            continue;
        }
        // More than one allocation transfers only under automatic
        // release. Off, a constructor with a branch stays untransferred,
        // which is what a program releasing by hand depends on.
        if sites.len() > 1 && !facts.automatic_release {
            continue;
        }
        let name = || func.name.resolve_global().unwrap_or_default();
        // A box is released by a named symbol the caller would have to
        // know, and what the caller picks is decided by the returned
        // type. Only storage the caller can release from the type alone
        // transfers: plain storage, a type with its own release, and a
        // string.
        if sites.iter().any(|s| {
            !matches!(
                s.release,
                Release::Intrinsic
                    | Release::Glue(_)
                    | Release::Symbol(STRING_FREE)
                    | Release::Symbol(BOX_FREE)
            )
        }) {
            if trace_enabled() {
                eprintln!(
                    "[drop] {}: a site's release is not known from its type",
                    name()
                );
            }
            continue;
        }
        // Every allocation that reaches a return, not one. A constructor
        // with a branch allocates in each arm and returns whichever it
        // took: `sites.len() != 1` refused all of them, so a type built
        // by anything more than a single unconditional allocation never
        // transferred and its callers never released it. The rule is
        // the same for each returned allocation: it may leave only by
        // being returned, or the caller and whoever else kept it would
        // both hold it. An allocation that never reaches a return is the
        // function's own affair.
        let returned: Vec<HirId> = func
            .blocks
            .values()
            .filter_map(|b| match &b.terminator {
                HirTerminator::Return { values } => Some(values.iter().copied()),
                _ => None,
            })
            .flatten()
            .collect();
        let mut derived = std::collections::HashSet::new();
        let mut all_transfer = true;
        for site in &sites {
            let d = derived_values_in(func, site.result, facts);
            if !returned.iter().any(|r| d.contains(r)) {
                continue;
            }
            if !escapes_only_by_return(func, &d, site, facts) {
                all_transfer = false;
                break;
            }
            derived.extend(d);
        }
        if !all_transfer {
            if trace_enabled() {
                eprintln!(
                    "[drop] {}: a returned allocation also leaves another way",
                    name()
                );
            }
            continue;
        }
        // Every return hands back an allocation, or nothing: a null is
        // what an error path returns in place of one, and releasing
        // nothing is a no-op.
        let returns_nothing = |v: &HirId| {
            func.values.get(v).is_some_and(|value| {
                matches!(
                    value.kind,
                    crate::hir::HirValueKind::Constant(HirConstant::Null(_))
                )
            })
        };
        let mut returns = 0usize;
        let mut all_return_it = true;
        for block in func.blocks.values() {
            if let HirTerminator::Return { values } = &block.terminator {
                returns += 1;
                if !values
                    .iter()
                    .any(|v| derived.contains(v) || returns_nothing(v))
                {
                    all_return_it = false;
                    if trace_enabled() {
                        eprintln!(
                            "[drop] {}: returns {:?}, which is not an allocation of its own",
                            name(),
                            values
                                .first()
                                .and_then(|v| block
                                    .instructions
                                    .iter()
                                    .find(|i| i.result_id() == Some(*v))
                                    .map(|i| format!("{i:?}")))
                                .unwrap_or_default()
                        );
                    }
                }
            }
        }
        if returns > 0 && all_return_it {
            owned.insert(*key);
        } else if trace_enabled() {
            eprintln!(
                "[drop] {} does not return owned storage: {} returns, all owned: {}",
                func.name.resolve_global().unwrap_or_default(),
                returns,
                all_return_it
            );
        }
    }
    owned
}

/// `ZYNTAX_TRACE_DROP=1` prints what the pass decided and why.
fn trace_enabled() -> bool {
    std::env::var_os("ZYNTAX_TRACE_DROP").is_some()
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
            SiteOutcome::MultiBlockDrop { points } => {
                stats.frees_inserted += apply_points(func, points, site.result, site.release);
            }
            SiteOutcome::Escaped => stats.escapes_skipped += 1,
            SiteOutcome::MultiBlock => stats.multi_block_skipped += 1,
            SiteOutcome::NoUse => stats.no_use_skipped += 1,
        }
    }
    stats.frees_inserted += release_owned_phis(func, facts);
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
pub(crate) struct SymbolRole {
    /// Hands back storage the caller owns, released by the named symbol.
    pub(crate) allocates: Option<&'static str>,
    /// Reads through its pointer arguments without keeping them, so
    /// passing one here does not end the caller's claim. A box that
    /// keeps the pointer it is given does not borrow it.
    pub(crate) borrows_args: bool,
}

impl SymbolRole {
    const BORROWS: SymbolRole = SymbolRole {
        allocates: None,
        borrows_args: true,
    };
    const KEEPS_INTO_BOX: SymbolRole = SymbolRole {
        allocates: Some(BOX_FREE),
        borrows_args: false,
    };
    const COPIES_INTO_BOX: SymbolRole = SymbolRole {
        allocates: Some(BOX_FREE),
        borrows_args: true,
    };
}

/// The role of a runtime symbol, or `None` where the pass knows nothing
/// about it and must assume the worst.
pub(crate) fn symbol_role(name: &str) -> Option<SymbolRole> {
    match name {
        "zyntax_box_bool" | "zyntax_box_f32" | "zyntax_box_f64" | "zyntax_box_i32"
        | "zyntax_box_i64" | "zyntax_box_str" | "zyntax_box_ptr" | "zyntax_box_opaque" => {
            Some(SymbolRole::KEEPS_INTO_BOX)
        }
        "zyntax_box_get_bool"
        | "zyntax_box_get_f32"
        | "zyntax_box_get_f64"
        | "zyntax_box_get_i32"
        | "zyntax_box_get_i64"
        | "zyntax_box_get_str"
        | "zyntax_box_get_opaque"
        | "zyntax_box_get_tag"
        | "zyntax_box_header_tag"
        | "zyntax_box_data"
        | "zyntax_box_payload_i64"
        | "zyntax_box_payload_f64"
        | "zyntax_box_payload_bool" => Some(SymbolRole::BORROWS),
        // A box holding its own copy of a string, released with the box.
        "$IO$string_to_dynamic" => Some(SymbolRole::COPIES_INTO_BOX),
        // The IO, string and math plugins read their arguments and hand
        // back fresh storage; none keeps a pointer it was given.
        _ if name.starts_with("$IO$")
            || name.starts_with("$String$")
            || name.starts_with("$Math$") =>
        {
            Some(SymbolRole::BORROWS)
        }
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
    /// The type's own release, which frees what its fields own before
    /// freeing the object. Synthesised by [`crate::drop_glue`]; freeing
    /// such an object with a bare `Free` would leak everything it holds.
    Glue(HirId),
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
    SingleBlockDrop {
        block: HirId,
        after_idx: usize,
    },
    /// Released in each block where it is live and no successor keeps
    /// it so. More than one point because a value can die on two paths
    /// out of a branch, and each has to release it exactly once.
    MultiBlockDrop {
        points: Vec<Point>,
    },
    Escaped,
    MultiBlock,
    NoUse,
}

/// Where a release goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Point {
    /// After the instruction at this index of the block.
    After(HirId, usize),
    /// On the edge from the first block to the second: the value is live
    /// leaving the one and dead entering the other, which reads it
    /// nowhere. Placed at the target's entry when this is its only way
    /// in, else on a block of its own spliced into the edge.
    Edge(HirId, HirId),
}

/// Insert releases of `value` at `points`. Insertions after an index go
/// back to front so earlier indices stay valid; edges are done last,
/// since an entry insertion shifts the block's indices.
fn apply_points(
    func: &mut HirFunction,
    points: Vec<Point>,
    value: HirId,
    release: Release,
) -> usize {
    let mut after: Vec<(HirId, usize)> = Vec::new();
    let mut edges: Vec<(HirId, HirId)> = Vec::new();
    for p in points {
        match p {
            Point::After(b, i) => after.push((b, i)),
            Point::Edge(from, to) => edges.push((from, to)),
        }
    }
    let count = after.len() + edges.len();
    after.sort_by(|a, b| b.1.cmp(&a.1));
    for (block, idx) in after {
        insert_free_after(func, block, idx, value, release);
    }
    for (from, to) in edges {
        insert_free_on_edge(func, from, to, value, release);
    }
    count
}

/// Release `value` on the edge `from -> to`.
fn insert_free_on_edge(
    func: &mut HirFunction,
    from: HirId,
    to: HirId,
    value: HirId,
    release: Release,
) {
    let release_inst = |value| HirInstruction::Call {
        result: None,
        callee: match release {
            Release::Intrinsic => HirCallable::Intrinsic(Intrinsic::Free),
            Release::Symbol(name) => HirCallable::Symbol(name.to_string()),
            Release::Glue(id) => HirCallable::Function(id),
        },
        args: vec![value],
        type_args: Vec::new(),
        const_args: Vec::new(),
        is_tail: false,
    };
    let predecessors: Vec<HirId> = func
        .blocks
        .iter()
        .filter(|(_, b)| successors_of(b).contains(&to))
        .map(|(id, _)| *id)
        .collect();
    if predecessors.len() == 1 && predecessors[0] == from {
        if let Some(block) = func.blocks.get_mut(&to) {
            block.instructions.insert(0, release_inst(value));
        }
        return;
    }
    // A block of its own on the edge, so no other way into `to` runs the
    // release.
    let middle = HirId::new();
    let block = crate::hir::HirBlock {
        id: middle,
        label: None,
        phis: Vec::new(),
        instructions: vec![release_inst(value)],
        terminator: HirTerminator::Branch { target: to },
        dominance_frontier: Default::default(),
        predecessors: vec![from],
        successors: vec![to],
    };
    func.blocks.insert(middle, block);
    if let Some(source) = func.blocks.get_mut(&from) {
        retarget(&mut source.terminator, to, middle);
        for s in source.successors.iter_mut() {
            if *s == to {
                *s = middle;
            }
        }
    }
    if let Some(target) = func.blocks.get_mut(&to) {
        for phi in target.phis.iter_mut() {
            for (_, pred) in phi.incoming.iter_mut() {
                if *pred == from {
                    *pred = middle;
                }
            }
        }
        for p in target.predecessors.iter_mut() {
            if *p == from {
                *p = middle;
            }
        }
    }
}

/// Point every edge of `term` that went to `from` at `to`.
fn retarget(term: &mut HirTerminator, from: HirId, to: HirId) {
    let swap = |t: &mut HirId| {
        if *t == from {
            *t = to;
        }
    };
    match term {
        HirTerminator::Branch { target } => swap(target),
        HirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            swap(true_target);
            swap(false_target);
        }
        HirTerminator::Switch { default, cases, .. } => {
            swap(default);
            for (_, t) in cases.iter_mut() {
                swap(t);
            }
        }
        HirTerminator::Invoke { normal, unwind, .. } => {
            swap(normal);
            swap(unwind);
        }
        HirTerminator::PatternMatch {
            patterns, default, ..
        } => {
            for p in patterns.iter_mut() {
                swap(&mut p.target);
            }
            if let Some(d) = default {
                swap(d);
            }
        }
        HirTerminator::Return { .. } | HirTerminator::Unreachable => {}
    }
}

/// Allocation sites, counting a call whose callee hands back owned
/// storage: the caller owns that result and is the only one able to
/// release it.
/// How storage of the type `result` holds is released.
///
/// A type owning nothing is freed outright. One owning another is
/// released through its own function, which frees what its fields hold
/// first: freeing such an object with a bare `Free` would drop the only
/// reference to everything under it.
fn release_for(func: &HirFunction, result: HirId, facts: &ModuleFacts) -> Release {
    let Some(ty) = func.values.get(&result).map(|v| &v.ty) else {
        return Release::Intrinsic;
    };
    if is_string(ty) {
        return Release::Symbol(STRING_FREE);
    }
    if is_box(ty) {
        return Release::Symbol(BOX_FREE);
    }
    if let HirType::Ptr(inner) = ty {
        if let HirType::Struct(s) = &**inner {
            if let Some(id) = s.name.and_then(|n| facts.glue.get(&n)) {
                return Release::Glue(*id);
            }
        }
    }
    Release::Intrinsic
}

fn collect_owned_sites(func: &HirFunction, facts: &ModuleFacts) -> Vec<MallocSite> {
    let mut sites = collect_malloc_sites(func);
    // A malloc whose type owns another is released through that type's
    // own function. Applied after collection so both kinds of site,
    // the intrinsic and the owned-returning call, pick it up.
    for site in sites.iter_mut() {
        if site.release == Release::Intrinsic {
            site.release = release_for(func, site.result, facts);
        }
    }
    // Calls handing back storage the caller owns: a function that
    // returns what it allocated, an extern that makes a string, and an
    // extern standing for a runtime symbol that allocates.
    for (block_id, block) in &func.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let HirInstruction::Call {
                result: Some(result),
                callee: HirCallable::Function(callee_id),
                ..
            } = inst
            {
                let allocating_extern = facts
                    .extern_links
                    .get(callee_id)
                    .and_then(|link| symbol_role(link).and_then(|r| r.allocates));
                if let Some(free) = allocating_extern {
                    sites.push(MallocSite {
                        result: *result,
                        block: *block_id,
                        inst_idx: idx,
                        release: Release::Symbol(free),
                    });
                } else if facts.returns_owned.contains(callee_id)
                    || facts.string_makers.contains(callee_id)
                {
                    sites.push(MallocSite {
                        result: *result,
                        block: *block_id,
                        inst_idx: idx,
                        release: release_for(func, *result, facts),
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
                    Some(SymbolRole {
                        allocates: Some(free_name),
                        ..
                    }) => (*result, Release::Symbol(free_name)),
                    // A plugin's string result is fresh storage.
                    Some(_)
                        if makes_strings(name)
                            && func.values.get(result).is_some_and(|v| is_string(&v.ty)) =>
                    {
                        (*result, Release::Symbol(STRING_FREE))
                    }
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

/// Where an allocation dies, when it outlives the block it was made in.
///
/// A value is live at a point when some use is still ahead of it. Read
/// backwards: a block's exit is live if any successor's entry is, and a
/// block's entry is live if it uses the value or its own exit is. The
/// definition kills it, so nothing above the allocation is ever called
/// live and a loop's back edge cannot carry a claim into the iteration
/// that made it.
///
/// The release goes wherever the value is live and no successor keeps
/// it so. That is one point for a value dying at the end of a loop
/// body, and one per arm for a value dying inside a branch.
///
/// Returns `None` where no such point exists, which is a value that is
/// live on every exit and so is not this function's to release.
fn drop_points(
    func: &HirFunction,
    site: &MallocSite,
    derived: &std::collections::HashSet<HirId>,
    facts: &ModuleFacts,
) -> Option<Vec<Point>> {
    drop_points_transferring(
        func,
        site,
        derived,
        facts,
        &std::collections::HashSet::new(),
    )
}

/// [`drop_points`] where leaving `transfer_out` blocks hands the storage
/// to whoever owns it past the edge, a phi that releases it later: the
/// value is live out of such a block whatever its successors do, so no
/// release is placed in it.
fn drop_points_transferring(
    func: &HirFunction,
    site: &MallocSite,
    derived: &std::collections::HashSet<HirId>,
    facts: &ModuleFacts,
    transfer_out: &std::collections::HashSet<HirId>,
) -> Option<Vec<Point>> {
    // Only blocks the entry can get to. An unreachable one has no
    // bearing on where the value dies and its successors would drag
    // liveness around the graph for nothing.
    let mut reachable = std::collections::HashSet::new();
    let mut stack = vec![func.entry_block];
    while let Some(b) = stack.pop() {
        if !reachable.insert(b) {
            continue;
        }
        if let Some(block) = func.blocks.get(&b) {
            stack.extend(successors_of(block));
        }
    }

    // What each block does with the value: the last instruction that
    // uses it, and whether the terminator does.
    let mut last_use: std::collections::HashMap<HirId, usize> = std::collections::HashMap::new();
    let mut uses_block: std::collections::HashSet<HirId> = std::collections::HashSet::new();
    for (block_id, block) in &func.blocks {
        if !reachable.contains(block_id) {
            continue;
        }
        for (idx, inst) in block.instructions.iter().enumerate() {
            if *block_id == site.block && idx == site.inst_idx {
                continue;
            }
            match classify_derived_use(inst, derived, facts) {
                UseKind::Use => {
                    uses_block.insert(*block_id);
                    let e = last_use.entry(*block_id).or_insert(idx);
                    *e = (*e).max(idx);
                }
                UseKind::Escape => return None,
                UseKind::None => {}
            }
        }
        match derived
            .iter()
            .map(|d| classify_terminator_use(&block.terminator, *d))
            .fold(UseKind::None, strongest)
        {
            UseKind::Use => {
                uses_block.insert(*block_id);
            }
            UseKind::Escape => return None,
            UseKind::None => {}
        }
    }
    if uses_block.is_empty() {
        return None;
    }

    // live_in[B] = B uses it, or its exit is live. The block holding
    // the allocation kills it: nothing before the call can be holding
    // what the call has not produced.
    let mut live_in: std::collections::HashSet<HirId> = std::collections::HashSet::new();
    let mut live_out: std::collections::HashSet<HirId> = std::collections::HashSet::new();
    loop {
        let mut changed = false;
        for (block_id, block) in &func.blocks {
            if !reachable.contains(block_id) {
                continue;
            }
            let out = transfer_out.contains(block_id)
                || successors_of(block).iter().any(|sc| live_in.contains(sc));
            if out && live_out.insert(*block_id) {
                changed = true;
            }
            let inn = *block_id != site.block && (uses_block.contains(block_id) || out);
            if inn && live_in.insert(*block_id) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Where it is live and nothing after it is: after the last use in a
    // block no successor of which keeps it, and on each edge out of a
    // block that keeps it into one that does not.
    let mut points = Vec::new();
    for (block_id, block) in &func.blocks {
        if !reachable.contains(block_id) {
            continue;
        }
        let holds = *block_id == site.block || live_in.contains(block_id);
        if !holds {
            continue;
        }
        if live_out.contains(block_id) {
            // Handed on along a transfer edge, or read further on along
            // some successor; the others end its life as they are taken.
            if transfer_out.contains(block_id) {
                continue;
            }
            for succ in successors_of(block) {
                if !live_in.contains(&succ) && succ != site.block && reachable.contains(&succ) {
                    points.push(Point::Edge(*block_id, succ));
                }
            }
            continue;
        }
        if !uses_block.contains(block_id) && *block_id != site.block {
            continue;
        }
        // After the last use, or ahead of a terminator that is the only
        // thing using it. A terminator reads the address rather than
        // what is at it, so releasing first is still the right order.
        let at = match last_use.get(block_id) {
            Some(idx) => idx + 1,
            None if *block_id == site.block && !uses_block.contains(block_id) => return None,
            None => block.instructions.len(),
        };
        points.push(Point::After(*block_id, at.saturating_sub(1)));
    }
    (!points.is_empty()).then_some(points)
}

/// A block's successors, read off its terminator rather than off the
/// cached list, which a pass that rewrote control flow may not have
/// kept up to date.
fn successors_of(block: &crate::hir::HirBlock) -> Vec<HirId> {
    match &block.terminator {
        HirTerminator::Branch { target } => vec![*target],
        HirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        HirTerminator::Switch { default, cases, .. } => {
            let mut v = vec![*default];
            v.extend(cases.iter().map(|(_, b)| *b));
            v
        }
        HirTerminator::Invoke { normal, unwind, .. } => vec![*normal, *unwind],
        HirTerminator::PatternMatch {
            patterns, default, ..
        } => {
            let mut v: Vec<HirId> = patterns.iter().map(|p| p.target).collect();
            v.extend(default.iter().copied());
            v
        }
        HirTerminator::Return { .. } | HirTerminator::Unreachable => Vec::new(),
    }
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
    let derived = derived_values_in(func, site.result, facts);
    let target = site.result;
    let mut last_idx_in_block: Option<usize> = None;
    let mut had_any_use = false;

    // A value merged through a phi is another name past the merge, and
    // a release placed by that name's liveness would name this one where
    // it was never defined. Such storage is released through the phi
    // (`release_owned_phis`), never here, so this is decided before any
    // block's instructions are read.
    if !phis_using_any(func, &derived).is_empty()
        || func
            .blocks
            .values()
            .any(|b| b.phis.iter().any(|p| p.result == target))
    {
        return SiteOutcome::MultiBlock;
    }

    for (block_id, block) in &func.blocks {
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
                        return match facts
                            .automatic_release
                            .then(|| drop_points(func, site, &derived, facts))
                            .flatten()
                        {
                            Some(points) => SiteOutcome::MultiBlockDrop { points },
                            None => SiteOutcome::MultiBlock,
                        };
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
pub(crate) fn derived_values(func: &HirFunction, root: HirId) -> std::collections::HashSet<HirId> {
    derived_values_with(func, root, true, None)
}

/// [`derived_values`] knowing which callees hand an argument back.
fn derived_values_in(
    func: &HirFunction,
    root: HirId,
    facts: &ModuleFacts,
) -> std::collections::HashSet<HirId> {
    derived_values_with(func, root, true, Some(facts))
}

/// [`derived_values_in`] stopping at phis: the names for the storage on
/// the paths where `root` itself is defined, which is where a release by
/// its name is defined too.
fn derived_values_local(
    func: &HirFunction,
    root: HirId,
    facts: &ModuleFacts,
) -> std::collections::HashSet<HirId> {
    derived_values_with(func, root, false, Some(facts))
}

fn derived_values_with(
    func: &HirFunction,
    root: HirId,
    through_phis: bool,
    facts: Option<&ModuleFacts>,
) -> std::collections::HashSet<HirId> {
    let mut set = std::collections::HashSet::new();
    set.insert(root);
    // Blocks are unordered here, so a single sweep can miss a chain
    // that runs backwards through the map. Repeat until nothing new
    // appears; the set only grows and is bounded by the value count.
    loop {
        let before = set.len();
        for block in func.blocks.values() {
            if through_phis {
                for phi in &block.phis {
                    if phi.incoming.iter().any(|(v, _)| set.contains(v)) {
                        set.insert(phi.result);
                    }
                }
            }
            for inst in &block.instructions {
                // Two shapes produce a value that is not another name
                // for this storage, and following them would put the
                // whole program in the set.
                //
                // A dereference reads a pointer the allocation holds.
                // Releasing this one leaves what it held untouched, so
                // what comes out is a different object with its own
                // life.
                //
                // A call hands back whatever the callee made. Nothing
                // is given up by not following it: handing the pointer
                // to a callee that does not merely borrow it is already
                // an escape, and one that borrows has promised not to
                // keep it. Following it instead made `t.check()`, an
                // integer, an alias of the tree, and printing that
                // integer read as the tree escaping. The exception is a
                // callee known to hand the argument itself back: its
                // result is this storage under another name.
                if let HirInstruction::Call {
                    result: Some(result),
                    callee,
                    args,
                    ..
                } = inst
                {
                    if let Some(facts) = facts {
                        if args
                            .iter()
                            .any(|a| set.contains(a) && facts.result_aliases_arg(callee, args, *a))
                        {
                            set.insert(*result);
                        }
                    }
                    continue;
                }
                if matches!(
                    inst,
                    HirInstruction::Load { .. }
                        | HirInstruction::IndirectCall { .. }
                        | HirInstruction::TraitMethodCall { .. }
                        | HirInstruction::CallClosure { .. }
                ) {
                    continue;
                }
                // Anything else computing a value from one of these is
                // another name for the same storage. Asked through
                // `operands`, which every instruction answers, rather
                // than through a list of the kinds thought of at the
                // time: a kind left out of a list reads as a value the
                // allocation never reaches, and a use through it as no
                // use at all, which puts the release in front of it.
                // The GEP a struct literal takes to reach a field was
                // exactly that, and the store through it landed after
                // the free.
                if inst.operands().iter().any(|o| set.contains(o)) {
                    if let Some(result) = inst.result_id() {
                        set.insert(result);
                    }
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

/// Whether a call to `callee` releases its argument: the free intrinsic,
/// a runtime release symbol, or a type's own release.
fn is_release(callee: &HirCallable, facts: &ModuleFacts) -> bool {
    match callee {
        HirCallable::Intrinsic(Intrinsic::Free) => true,
        HirCallable::Symbol(name) => name == STRING_FREE || name == BOX_FREE,
        HirCallable::Function(id) => facts.glue.values().any(|g| g == id),
        _ => false,
    }
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
            // A release of `target` is what this pass inserts, and the
            // pass may run again over the same function: a released
            // value is forfeit, never released twice.
            if is_release(callee, facts) && args.iter().any(|a| *a == target) {
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
                HirCallable::Symbol(name) if symbol_role(name).is_some_and(|r| r.borrows_args) => {
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
            Release::Glue(id) => HirCallable::Function(id),
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

    /// A pointer reached through a field pointer keeps the object alive.
    ///
    /// This is the shape the struct-literal lowering emits: allocate,
    /// then for each field a byte-offset GEP, a bitcast, and a store
    /// through the result. The allocation is used at the GEP and again,
    /// through what the GEP produced, at the store.
    ///
    /// `derived_values` follows a pointer through `InsertValue`,
    /// `ExtractValue` and `Cast` and stopped there, so the GEP's result
    /// was not one of the allocation's own values and a store through it
    /// was not one of its uses. The last use read as the GEP, and the
    /// release went in ahead of the store that followed it.
    ///
    /// The asymmetry is what makes this worth pinning: an unclassified
    /// *use* is called an escape and costs a missed release, while an
    /// untracked *alias* costs a release that is too early. One leaks
    /// and the other writes through freed memory.
    #[test]
    fn a_pointer_reached_through_a_field_pointer_is_still_a_use() {
        let mut f = HirFunction::new(
            InternedString::new_global("through_a_field"),
            empty_sig(HirType::Void),
        );
        let entry = *f.blocks.keys().next().unwrap();
        let size = add_const(&mut f, HirType::I64, HirConstant::I64(24));
        let zero = add_const(&mut f, HirType::I64, HirConstant::I64(0));
        let seven = add_const(&mut f, HirType::I64, HirConstant::I64(7));
        let ptr = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let gep = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::U8)));
        let slot = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));

        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::GetElementPtr {
            result: gep,
            ty: HirType::U8,
            ptr,
            indices: vec![zero],
        });
        block.instructions.push(HirInstruction::Cast {
            op: crate::hir::CastOp::Bitcast,
            result: slot,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            operand: gep,
        });
        block.instructions.push(HirInstruction::Store {
            value: seven,
            ptr: slot,
            align: 8,
            volatile: false,
        });
        block.terminator = HirTerminator::Return { values: vec![] };

        run_function(&mut f, &ModuleFacts::default());

        let block = f.blocks.values().next().unwrap();
        let free_at = block.instructions.iter().position(|i| {
            matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    ..
                }
            )
        });
        let store_at = block
            .instructions
            .iter()
            .position(|i| matches!(i, HirInstruction::Store { .. }))
            .expect("the store should still be there");
        if let Some(free_at) = free_at {
            assert!(
                free_at > store_at,
                "the release went in at {free_at}, ahead of the store at \
                 {store_at} that writes through the same allocation"
            );
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
    /// after the body has read it, and the one the last iteration built
    /// is released after the exit has read it.
    #[test]
    fn an_accumulator_releases_what_it_replaces() {
        let (mut f, acc) = build_accumulator_loop(false);
        let is_body = |blk: &HirBlock| {
            blk.instructions
                .iter()
                .any(|i| matches!(i, HirInstruction::Load { .. }))
                && matches!(blk.terminator, HirTerminator::Branch { .. })
        };
        let body_id = f
            .blocks
            .iter()
            .find(|(_, blk)| is_body(blk))
            .map(|(id, _)| *id)
            .expect("the body");
        let exit_id = f
            .blocks
            .iter()
            .find(|(_, blk)| matches!(blk.terminator, HirTerminator::Return { .. }))
            .map(|(id, _)| *id)
            .expect("the exit");

        assert_eq!(release_owned_phis(&mut f, &ModuleFacts::default()), 2);

        let free_of = |blk: &HirBlock| {
            blk.instructions.iter().position(|i| {
                matches!(i, HirInstruction::Call { callee: HirCallable::Intrinsic(Intrinsic::Free), args, .. } if args == &vec![acc])
            })
        };
        let read_of = |blk: &HirBlock| {
            blk.instructions
                .iter()
                .position(|i| matches!(i, HirInstruction::Load { .. }))
        };
        for id in [body_id, exit_id] {
            let blk = &f.blocks[&id];
            let free_at = free_of(blk).expect("the accumulator should be released here");
            assert!(
                free_at > read_of(blk).expect("the read"),
                "releasing before the read would be a use after free"
            );
        }
    }

    /// If anything might keep the accumulator, it is not ours to
    /// release.
    #[test]
    fn an_accumulator_that_might_be_kept_is_left_alone() {
        let (mut f, _) = build_accumulator_loop(true);
        assert_eq!(
            release_owned_phis(&mut f, &ModuleFacts::default()),
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
