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

use std::collections::HashMap;

/// Sets of ids: the pass spends its time asking whether a value is one
/// of a handful, so the hasher is the cheap one.
type IdSet = fnv::FnvHashSet<HirId>;

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
/// Boxes a copy of a string; the box owns the copy.
const STRING_TO_BOX: &str = "$IO$string_to_dynamic";
/// Boxes a string as it is; the box owns it from then on.
const STRING_INTO_BOX: &str = "$IO$string_adopt_dynamic";

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
    let t0 = web_time::Instant::now();
    let facts = ModuleFacts::build(module);
    if std::env::var_os("ZYNTAX_TRACE_DROP_TIME").is_some() {
        eprintln!(
            "[drop-time] facts {:.2} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    run_module_with(module, &facts)
}

/// What the pass knows about every function of `module`, for
/// [`run_module_with`]. The facts are about what a call returns and
/// keeps, which optimising a body does not change, so a module whose
/// functions are optimised one at a time builds them once.
pub fn facts_of(module: &HirModule) -> ModuleFacts {
    ModuleFacts::build(module)
}

/// Write each function's facts into its attributes, for a module about
/// to be stored: a module that links it later starts from them (see
/// `FunctionAttributes::release_facts`).
///
/// A function declared here without a body or a symbol gets its body
/// from the program that links the module, so what is read of it here
/// (an extern that keeps every argument and returns nothing it owns)
/// may not hold there. Nothing is recorded for a function whose facts
/// read such a declaration's, directly or through a callee whose own
/// were not recorded. A function returning no storage reads nothing
/// that matters: both its facts are false whatever it calls, so it is
/// recorded and what calls it is unaffected.
pub fn record_facts(module: &mut HirModule, facts: &ModuleFacts) {
    let supplied_later: Vec<HirId> = module
        .functions
        .iter()
        .filter(|(_, f)| f.is_external && f.link_name.is_none())
        .map(|(key, _)| *key)
        .collect();
    let defined: Vec<HirId> = module
        .functions
        .iter()
        .filter(|(_, f)| !f.is_external)
        .map(|(key, _)| *key)
        .collect();
    let mut callers: std::collections::HashMap<HirId, Vec<HirId>> =
        std::collections::HashMap::new();
    for (key, called) in callees_of(module, &defined) {
        for c in called {
            callers.entry(c).or_default().push(key);
        }
    }
    let returns_storage = |key: &HirId| {
        module.functions[key]
            .signature
            .returns
            .iter()
            .any(may_be_storage)
    };
    let mut program_dependent: IdSet = IdSet::default();
    let mut pending = supplied_later;
    while let Some(key) = pending.pop() {
        for caller in callers.get(&key).into_iter().flatten() {
            if returns_storage(caller) && program_dependent.insert(*caller) {
                pending.push(*caller);
            }
        }
    }
    if trace_enabled() {
        let mut names: Vec<String> = program_dependent
            .iter()
            .filter_map(|k| module.functions.get(k))
            .map(|f| f.name.resolve_global().unwrap_or_default())
            .collect();
        names.sort();
        eprintln!(
            "[drop] recording facts for {} of {} functions; not for the {} reading a body the program supplies: {}",
            defined.len() - program_dependent.len(),
            defined.len(),
            program_dependent.len(),
            names.join(" ")
        );
    }
    for (key, func) in module.functions.iter_mut() {
        if func.is_external || program_dependent.contains(key) {
            continue;
        }
        func.attributes.release_facts = Some(crate::hir::ReleaseFacts {
            returns_owned: facts.returns_owned.contains(key),
            returns_param: facts
                .returns_param
                .get(key)
                .cloned()
                .unwrap_or_else(|| vec![false; func.signature.params.len()]),
            automatic_release: facts.automatic_release,
        });
    }
}

/// The pass over one function with the facts of its module: what a
/// body run before its optimisation needs of the pipeline, so that it
/// frees what it allocates.
pub fn run_function_with(func: &mut HirFunction, facts: &ModuleFacts) -> DropStats {
    run_function(func, facts)
}

/// [`run_module`] with the facts already built.
pub fn run_module_with(module: &mut HirModule, facts: &ModuleFacts) -> DropStats {
    let mut total = DropStats::default();
    let tprof = std::env::var_os("ZYNTAX_TRACE_DROP_TIME").is_some();
    let mut times: Vec<(f64, String)> = Vec::new();
    for func in module.functions_to_optimize() {
        if func.is_external {
            continue;
        }
        let t = web_time::Instant::now();
        total.combine(run_function(func, facts));
        if tprof {
            times.push((
                t.elapsed().as_secs_f64() * 1000.0,
                func.name.resolve_global().unwrap_or_default(),
            ));
        }
    }
    if tprof {
        times.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (t, n) in times.iter().take(8) {
            eprintln!("[drop-time] {t:.2} ms {n}");
        }
    }
    total
}

/// What this pass knows about the other functions in the module.
#[derive(Default)]
pub struct ModuleFacts {
    /// See [`automatic_release_for`].
    automatic_release: bool,
    returns_owned: IdSet,
    /// Externs whose result is a fresh string: a call to one is an
    /// allocation the caller releases with the string free.
    string_makers: IdSet,
    /// Per function, which parameters it may hand back as its result. The
    /// result of such a call is another name for the argument, so the
    /// argument lives as long as the result does.
    returns_param: std::collections::HashMap<HirId, Vec<bool>>,
    /// The runtime symbol behind each extern function.
    extern_links: std::collections::HashMap<HirId, String>,
    /// Functions that are the copying box call on their one parameter
    /// and nothing else: a call to one boxes a string as that call does.
    string_boxers: IdSet,
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
    /// The facts of `module`. A function carrying facts recorded when
    /// its own module was optimised keeps them, and the fixed points run
    /// over the rest: what a stored library recorded holds for any
    /// program that links it (see [`record_facts`]), so the rest is the
    /// program's own.
    /// `ZYNTAX_CHECK_FACTS_SEED=1` also recomputes everything and
    /// panics on a function whose recorded facts disagree; slow, safe to
    /// run with.
    fn build(module: &HirModule) -> Self {
        let facts = Self::build_seeded(module, true);
        static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *CHECK.get_or_init(|| std::env::var_os("ZYNTAX_CHECK_FACTS_SEED").is_some()) {
            let fresh = Self::build_seeded(module, false);
            let mut disagreements: Vec<String> = Vec::new();
            for (key, func) in module.functions.iter() {
                if func.is_external {
                    continue;
                }
                let name = func.name.resolve_global().unwrap_or_default();
                let seeded = if Self::seeded(func, facts.automatic_release) {
                    "recorded"
                } else {
                    "computed"
                };
                let (owned, fresh_owned) = (
                    facts.returns_owned.contains(key),
                    fresh.returns_owned.contains(key),
                );
                if owned != fresh_owned {
                    disagreements.push(format!(
                        "{name} ({seeded}): returns_owned {owned}, fresh fixed point {fresh_owned}"
                    ));
                }
                let (params, fresh_params) =
                    (facts.returns_param.get(key), fresh.returns_param.get(key));
                if params != fresh_params {
                    disagreements.push(format!(
                        "{name} ({seeded}): returns_param {params:?}, fresh fixed point {fresh_params:?}"
                    ));
                }
            }
            assert!(
                disagreements.is_empty(),
                "release facts disagree with a fresh fixed point:\n  {}",
                disagreements.join("\n  ")
            );
        }
        facts
    }

    /// Whether `func`'s recorded facts stand in for computing them here.
    fn seeded(func: &HirFunction, automatic_release: bool) -> bool {
        !func.is_external
            && func
                .attributes
                .release_facts
                .as_ref()
                .is_some_and(|facts| facts.automatic_release == automatic_release)
    }

    fn build_seeded(module: &HirModule, use_seeds: bool) -> Self {
        let automatic_release = automatic_release_for(module);
        let seeded = |func: &HirFunction| use_seeds && Self::seeded(func, automatic_release);
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
        let extern_links: std::collections::HashMap<HirId, String> = module
            .functions
            .iter()
            .filter(|(_, f)| f.is_external)
            .filter_map(|(key, f)| f.link_name.clone().map(|n| (*key, n)))
            .collect();
        let string_boxers = module
            .functions
            .iter()
            .filter(|(_, f)| wraps_string_boxing(f, &extern_links))
            .map(|(key, _)| *key)
            .collect();
        let names = module
            .functions
            .iter()
            .map(|(key, f)| (*key, f.name.resolve_global().unwrap_or_default()))
            .collect();
        let mut facts = Self {
            automatic_release,
            returns_owned: IdSet::default(),
            string_makers,
            returns_param: std::collections::HashMap::new(),
            extern_links,
            string_boxers,
            names,
            borrowed_params,
            glue,
        };
        // Recorded facts first; the fixed points below read them and
        // never ask the functions carrying them again.
        let mut unseeded: Vec<HirId> = Vec::new();
        for (key, func) in module.functions.iter() {
            if func.is_external {
                continue;
            }
            match func
                .attributes
                .release_facts
                .as_ref()
                .filter(|_| seeded(func))
            {
                Some(recorded) => {
                    if recorded.returns_owned {
                        facts.returns_owned.insert(*key);
                    }
                    facts
                        .returns_param
                        .insert(*key, recorded.returns_param.clone());
                }
                None => unseeded.push(*key),
            }
        }
        // A function returning a callee's result that is itself an
        // argument returns its own parameter, so this grows until it
        // stops; owned-returning functions likewise.
        let tprof = std::env::var_os("ZYNTAX_TRACE_DROP_TIME").is_some();
        let t = web_time::Instant::now();
        let mut rounds = 0;
        let callees = callees_of(module, &unseeded);
        // Which functions call each, for asking only the callers of
        // whoever changed again. Only a function computed here is
        // asked again, so only their calls are recorded.
        let mut callers: std::collections::HashMap<HirId, Vec<HirId>> =
            std::collections::HashMap::new();
        for (key, called) in &callees {
            for c in called {
                callers.entry(*c).or_default().push(*key);
            }
        }
        // Every function computed here first, then the callers of
        // whoever changed.
        let mut dirty: Vec<HirId> = unseeded.clone();
        loop {
            rounds += 1;
            let mut changed: Vec<HirId> = Vec::new();
            for key in &dirty {
                let Some(func) = module.functions.get(key) else {
                    continue;
                };
                let flags = params_returned_by(func, &facts);
                if facts.returns_param.get(key) != Some(&flags) {
                    facts.returns_param.insert(*key, flags);
                    changed.push(*key);
                }
            }
            if changed.is_empty() {
                break;
            }
            dirty = changed
                .iter()
                .filter_map(|c| callers.get(c))
                .flatten()
                .copied()
                .collect();
            dirty.sort();
            dirty.dedup();
        }
        if tprof {
            eprintln!(
                "[drop-time] returns_param {:.2} ms in {rounds} rounds over {} of {} functions",
                t.elapsed().as_secs_f64() * 1000.0,
                unseeded.len(),
                module.functions.len()
            );
        }
        // Owned-returning functions, to a fixed point. A function's
        // answer reads only its callees' facts, so after the first pass
        // only the callers of whoever changed are asked again.
        let t = web_time::Instant::now();
        let mut rounds = 0;
        let mut dirty: Vec<HirId> = unseeded;
        loop {
            rounds += 1;
            let mut changed: Vec<HirId> = Vec::new();
            for key in &dirty {
                let Some(func) = module.functions.get(key) else {
                    continue;
                };
                let owned = returns_owned_storage(func, &facts);
                let was = facts.returns_owned.contains(key);
                if owned != was {
                    if owned {
                        facts.returns_owned.insert(*key);
                    } else {
                        facts.returns_owned.remove(key);
                    }
                    changed.push(*key);
                }
            }
            if changed.is_empty() {
                break;
            }
            dirty = changed
                .iter()
                .filter_map(|c| callers.get(c))
                .flatten()
                .copied()
                .filter(|key| module.functions.get(key).is_some_and(|f| !f.is_external))
                .collect();
            dirty.sort();
            dirty.dedup();
        }
        if tprof {
            eprintln!(
                "[drop-time] returns_owned {:.2} ms in {rounds} rounds",
                t.elapsed().as_secs_f64() * 1000.0
            );
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
    let bodies: std::collections::HashMap<HirId, IdSet> = forest
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
    // Every phi that may carry storage to begin with; a join's phi only
    // under automatic release, since a program releasing by hand may
    // release what arrives at one.
    let mut candidates: IdSet = func
        .blocks
        .iter()
        .flat_map(|(b, block)| block.phis.iter().map(move |p| (p, *b)))
        .filter(|(p, block)| {
            may_be_storage(&p.ty) && (facts.automatic_release || bodies.contains_key(block))
        })
        .map(|(p, _)| p.result)
        .collect();
    if candidates.is_empty() {
        return 0;
    }
    // What the rounds below ask of each value does not change between
    // them: indexed once.
    let mut index = PhiIndex::of(func, &phi_blocks);
    loop {
        let before = candidates.len();
        copies.clear();
        for (block_id, block) in &func.blocks {
            let body = bodies.get(block_id);
            for phi in &block.phis {
                if !candidates.contains(&phi.result) {
                    continue;
                }
                match phi_incomings_owned(func, facts, &sites, &candidates, body, phi, &mut index) {
                    Some(seed_copies) => copies.extend(seed_copies),
                    None => {
                        candidates.remove(&phi.result);
                    }
                }
            }
        }
        // The phi's own value: read only by borrowers, and handed on only
        // to phis that own what they are handed.
        let mut dropped: Vec<HirId> = Vec::new();
        for p in candidates.iter().copied() {
            let kept = !index.borrowed_only(func, facts, p);
            let handed_on = index
                .phis_using_derived(func, facts, p)
                .iter()
                .any(|other| !candidates.contains(other));
            if kept || handed_on {
                if trace_enabled() {
                    eprintln!(
                        "[drop] {}: phi {:?} cannot own its value (kept by a use {kept}, handed to a phi that does not own {handed_on})",
                        func.name.resolve_global().unwrap_or_default(),
                        p
                    );
                }
                dropped.push(p);
            }
        }
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
                    let r = if *val == phi.result || is_null_value(func, *val) {
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
        let transfer_out: IdSet = func
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
/// What the rounds deciding which phis own their values ask of each
/// value, indexed once per function: the phis reading it, where it is
/// defined, what is derived from it and whether all its uses borrow.
struct PhiIndex {
    users: Users,
    /// The phis with the value among their incomings.
    phi_users: std::collections::HashMap<HirId, Vec<HirId>>,
    /// The blocks whose terminator reads the value.
    terminator_users: std::collections::HashMap<HirId, Vec<HirId>>,
    /// The block defining the value, by instruction or phi.
    def_blocks: std::collections::HashMap<HirId, HirId>,
    derived: std::collections::HashMap<HirId, IdSet>,
    borrowed: std::collections::HashMap<HirId, bool>,
    past_merge: std::collections::HashMap<(HirId, HirId), bool>,
}

impl PhiIndex {
    fn of(func: &HirFunction, phi_blocks: &std::collections::HashMap<HirId, HirId>) -> Self {
        let mut phi_users: std::collections::HashMap<HirId, Vec<HirId>> =
            std::collections::HashMap::new();
        let mut terminator_users: std::collections::HashMap<HirId, Vec<HirId>> =
            std::collections::HashMap::new();
        let mut def_blocks: std::collections::HashMap<HirId, HirId> = phi_blocks.clone();
        for (b, block) in &func.blocks {
            for phi in &block.phis {
                for (v, _) in &phi.incoming {
                    phi_users.entry(*v).or_default().push(phi.result);
                }
            }
            for inst in &block.instructions {
                if let Some(r) = inst.result_id() {
                    def_blocks.insert(r, *b);
                }
            }
            for v in terminator_operands(&block.terminator) {
                terminator_users.entry(v).or_default().push(*b);
            }
        }
        PhiIndex {
            users: Users::of(func),
            phi_users,
            terminator_users,
            def_blocks,
            derived: std::collections::HashMap::new(),
            borrowed: std::collections::HashMap::new(),
            past_merge: std::collections::HashMap::new(),
        }
    }

    fn derived(&mut self, func: &HirFunction, facts: &ModuleFacts, value: HirId) -> &IdSet {
        let users = &self.users;
        self.derived
            .entry(value)
            .or_insert_with(|| derived_values_using(func, value, false, Some(facts), users))
    }

    /// Whether every use of the value, and of what is derived from it,
    /// borrows: only the instructions naming one of them can say
    /// otherwise.
    fn borrowed_only(&mut self, func: &HirFunction, facts: &ModuleFacts, value: HirId) -> bool {
        if let Some(b) = self.borrowed.get(&value) {
            return *b;
        }
        let derived = self.derived(func, facts, value).clone();
        let mut seen: std::collections::HashSet<(HirId, usize)> = std::collections::HashSet::new();
        let mut borrowed = true;
        'scan: for d in &derived {
            if let Some(sites) = self.users.by_value.get(d) {
                for (b, i) in sites {
                    let Some(inst) = func.blocks.get(b).and_then(|blk| blk.instructions.get(*i))
                    else {
                        continue;
                    };
                    if !seen.insert((*b, *i)) {
                        continue;
                    }
                    if matches!(classify_derived_use(inst, &derived, facts), UseKind::Escape) {
                        borrowed = false;
                        break 'scan;
                    }
                }
            }
            if let Some(blocks) = self.terminator_users.get(d) {
                for b in blocks {
                    let Some(blk) = func.blocks.get(b) else {
                        continue;
                    };
                    if matches!(
                        classify_terminator_use(&blk.terminator, *d),
                        UseKind::Escape
                    ) {
                        borrowed = false;
                        break 'scan;
                    }
                }
            }
        }
        self.borrowed.insert(value, borrowed);
        borrowed
    }

    /// The phis reading the value or anything derived from it.
    fn phis_using_derived(
        &mut self,
        func: &HirFunction,
        facts: &ModuleFacts,
        value: HirId,
    ) -> Vec<HirId> {
        let derived = self.derived(func, facts, value).clone();
        let mut out = Vec::new();
        for d in &derived {
            if let Some(users) = self.phi_users.get(d) {
                out.extend(users.iter().copied());
            }
        }
        out
    }

    fn used_past_merge(
        &mut self,
        func: &HirFunction,
        facts: &ModuleFacts,
        value: HirId,
        merge: HirId,
    ) -> bool {
        if let Some(b) = self.past_merge.get(&(value, merge)) {
            return *b;
        }
        let derived = self.derived(func, facts, value).clone();
        let def_block = self.def_blocks.get(&value).copied();
        let b = used_past_merge(func, &derived, def_block, merge);
        self.past_merge.insert((value, merge), b);
        b
    }
}

fn phi_incomings_owned(
    func: &HirFunction,
    facts: &ModuleFacts,
    sites: &std::collections::HashMap<HirId, Release>,
    candidates: &IdSet,
    body: Option<&IdSet>,
    phi: &crate::hir::HirPhi,
    index: &mut PhiIndex,
) -> Option<Vec<(HirId, HirId, usize, HirId)>> {
    let merge = index.def_blocks.get(&phi.result).copied()?;
    let mut copies = Vec::new();
    for (position, (val, pred)) in phi.incoming.iter().enumerate() {
        // Carried round unchanged: one object, handed back to itself.
        if *val == phi.result {
            continue;
        }
        // Nothing arrives: what an error path hands on in place of
        // storage, and releasing nothing is a no-op.
        if is_null_value(func, *val) {
            continue;
        }
        let round_back_edge = body.is_some_and(|b| b.contains(pred));
        let owned = sites.contains_key(val) || candidates.contains(val);
        // Nothing but owning phis may keep the incoming.
        let kept_elsewhere = index.phi_users.get(val).is_some_and(|users| {
            users
                .iter()
                .any(|p| *p != phi.result && !candidates.contains(p))
        });
        let borrowed_only = index.borrowed_only(func, facts, *val);
        let fine = if round_back_edge {
            // Built by the body, so the phi never carries one object
            // twice.
            let fresh = body.is_some_and(|b| {
                index
                    .def_blocks
                    .get(val)
                    .is_some_and(|d| *d != merge && b.contains(d))
            });
            owned && !kept_elsewhere && fresh && borrowed_only
        } else {
            // Read by nothing once it has arrived: the phi's release
            // would free it under another name.
            owned
                && !kept_elsewhere
                && borrowed_only
                && !index.used_past_merge(func, facts, *val, merge)
        };
        if fine {
            continue;
        }
        if trace_enabled() {
            eprintln!(
                "[drop] {}: phi {:?} cannot own incoming {:?} = {} (owned {owned}, kept elsewhere {kept_elsewhere}, borrowed only {borrowed_only}, back edge {round_back_edge})",
                func.name.resolve_global().unwrap_or_default(),
                phi.result,
                val,
                describe_value(func, *val)
            );
        }
        // An accumulator's seed a string from anywhere: a copy is ours.
        if body.is_some() && !round_back_edge && is_string(&phi.ty) && is_string_value(func, *val) {
            copies.push((*pred, phi.result, position, *val));
            continue;
        }
        return None;
    }
    Some(copies)
}

/// The instruction, phi or constant defining `value`, for the trace.
fn describe_value(func: &HirFunction, value: HirId) -> String {
    for block in func.blocks.values() {
        if let Some(inst) = block
            .instructions
            .iter()
            .find(|i| i.result_id() == Some(value))
        {
            return format!("{inst:?}");
        }
        if let Some(phi) = block.phis.iter().find(|p| p.result == value) {
            return format!("{phi:?}");
        }
    }
    func.values
        .get(&value)
        .map(|v| format!("{:?}: {:?}", v.kind, v.ty))
        .unwrap_or_default()
}

/// Whether `value` is the null pointer, by name or as a zero of pointer
/// type.
fn is_null_value(func: &HirFunction, value: HirId) -> bool {
    func.values.get(&value).is_some_and(|v| match &v.kind {
        crate::hir::HirValueKind::Constant(HirConstant::Null(_)) => true,
        crate::hir::HirValueKind::Constant(
            HirConstant::I64(0) | HirConstant::USize(0) | HirConstant::ISize(0),
        ) => matches!(v.ty, HirType::Ptr(_)),
        _ => false,
    })
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
    names: &IdSet,
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
    let mut seen = IdSet::default();
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
            .any(|i| i.any_operand(|o| names.contains(&o)))
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
fn phis_using_any(func: &HirFunction, values: &IdSet) -> Vec<HirId> {
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
            if inst.any_operand(|o| o == value) {
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
fn uses_are_all_borrows(func: &HirFunction, derived: &IdSet, facts: &ModuleFacts) -> bool {
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
    derived: &IdSet,
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
/// Which of `func`'s parameters it may hand back as its result. Reads
/// only what the facts say of its callees.
fn params_returned_by(func: &HirFunction, facts: &ModuleFacts) -> Vec<bool> {
    {
        let returned: Vec<HirId> = func
            .blocks
            .values()
            .filter_map(|b| match &b.terminator {
                HirTerminator::Return { values } => Some(values.iter().copied()),
                _ => None,
            })
            .flatten()
            .collect();
        let params = func.signature.params.len();
        // A function returning nothing, or only scalars, returns no
        // parameter's storage.
        let returns_storage = returned
            .iter()
            .any(|r| func.values.get(r).is_some_and(|v| may_be_storage(&v.ty)));
        if !returns_storage {
            return vec![false; params];
        }
        // A parameter's value is the one of `Parameter` kind at its
        // position; the signature's own ids name nothing in the body.
        let mut by_position: Vec<Vec<HirId>> = vec![Vec::new(); params];
        for v in func.values.values() {
            if let crate::hir::HirValueKind::Parameter(n) = v.kind {
                if let Some(slot) = by_position.get_mut(n as usize) {
                    slot.push(v.id);
                }
            }
        }
        by_position
            .iter()
            .map(|ids| {
                ids.iter().any(|id| {
                    let names = derived_values_with(func, *id, true, Some(facts));
                    returned.iter().any(|r| names.contains(r))
                })
            })
            .collect()
    }
}

/// Whether `func`'s result is storage the caller owns.
///
/// Deliberately strict, because being wrong here releases something the
/// callee still refers to. A function qualifies only when it holds
/// exactly one allocation, that allocation leaves solely by being
/// returned, and every return hands it back. A function that sometimes
/// returns a fresh object and sometimes one it was given fails the last
/// condition and is left alone.
///
/// Of the module's facts this reads only what is known about the
/// callees, so the answer stands until one of them changes.
fn returns_owned_storage(func: &HirFunction, facts: &ModuleFacts) -> bool {
    {
        // A scalar result is never storage, whatever the body allocates.
        if !func.signature.returns.iter().any(may_be_storage) {
            return false;
        }
        let sites = collect_owned_sites(func, facts);
        if sites.is_empty() {
            return false;
        }
        // More than one allocation transfers only under automatic
        // release. Off, a constructor with a branch stays untransferred,
        // which is what a program releasing by hand depends on.
        if sites.len() > 1 && !facts.automatic_release {
            return false;
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
                    | Release::List
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
            return false;
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
        let mut derived = IdSet::default();
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
            return false;
        }
        // Every return hands back an allocation, or nothing: a null is
        // what an error path returns in place of one, and releasing
        // nothing is a no-op.
        let mut returns = 0usize;
        let mut all_return_it = true;
        for block in func.blocks.values() {
            if let HirTerminator::Return { values } = &block.terminator {
                returns += 1;
                if !values
                    .iter()
                    .any(|v| derived.contains(v) || is_null_value(func, *v))
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
            return true;
        }
        if trace_enabled() {
            eprintln!(
                "[drop] {} does not return owned storage: {} returns, all owned: {}",
                func.name.resolve_global().unwrap_or_default(),
                returns,
                all_return_it
            );
        }
        false
    }
}

/// Whether a value of `ty` can be storage a caller releases: anything
/// but a scalar. An aggregate travels by address here and a pointer is
/// one, so both count.
fn may_be_storage(ty: &HirType) -> bool {
    !matches!(
        ty,
        HirType::Void
            | HirType::Bool
            | HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::I128
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::U128
            | HirType::F32
            | HirType::F64
            | HirType::USize
            | HirType::ISize
    )
}

/// The functions each of `keys` calls directly.
fn callees_of(module: &HirModule, keys: &[HirId]) -> std::collections::HashMap<HirId, Vec<HirId>> {
    keys.iter()
        .filter_map(|key| module.functions.get(key).map(|func| (key, func)))
        .map(|(key, func)| {
            let mut callees: Vec<HirId> = func
                .blocks
                .values()
                .flat_map(|b| b.instructions.iter())
                .filter_map(|inst| match inst {
                    HirInstruction::Call {
                        callee: HirCallable::Function(id),
                        ..
                    } => Some(*id),
                    _ => None,
                })
                .collect();
            callees.sort();
            callees.dedup();
            (*key, callees)
        })
        .collect()
}

/// `ZYNTAX_TRACE_DROP=1` prints what the pass decided and why.
fn trace_enabled() -> bool {
    std::env::var_os("ZYNTAX_TRACE_DROP").is_some()
}

/// Whether the allocation leaves this function only by being returned.
fn escapes_only_by_return(
    func: &HirFunction,
    derived: &IdSet,
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

/// Whether `callee` copies a string into a fresh box.
fn boxes_string(callee: &HirCallable, facts: &ModuleFacts) -> bool {
    match callee {
        HirCallable::Symbol(name) => name == STRING_TO_BOX,
        HirCallable::Function(id) => {
            facts
                .extern_links
                .get(id)
                .is_some_and(|n| n == STRING_TO_BOX)
                || facts.string_boxers.contains(id)
        }
        _ => false,
    }
}

/// Whether `func` is a wrapper of the copying box call: one parameter,
/// one block, whose only instruction boxes that parameter and whose
/// result is returned. Such a call means what the box call means, so it
/// can be turned into the adopting one the same way.
fn wraps_string_boxing(
    func: &HirFunction,
    extern_links: &std::collections::HashMap<HirId, String>,
) -> bool {
    if func.is_external || func.blocks.len() != 1 || func.signature.params.len() != 1 {
        return false;
    }
    let Some(block) = func.blocks.get(&func.entry_block) else {
        return false;
    };
    let [
        HirInstruction::Call {
            result: Some(result),
            callee,
            args,
            ..
        },
    ] = block.instructions.as_slice()
    else {
        return false;
    };
    let boxes = match callee {
        HirCallable::Symbol(name) => name == STRING_TO_BOX,
        HirCallable::Function(id) => extern_links.get(id).is_some_and(|n| n == STRING_TO_BOX),
        _ => false,
    };
    // The body names its parameter by the value of `Parameter` kind,
    // not by the signature's id.
    let [arg] = args.as_slice() else {
        return false;
    };
    let is_param = func
        .values
        .get(arg)
        .is_some_and(|v| matches!(v.kind, crate::hir::HirValueKind::Parameter(0)));
    boxes
        && is_param
        && block.phis.is_empty()
        && matches!(&block.terminator, HirTerminator::Return { values } if values.as_slice() == [*result])
}

/// One string boxed twice on a path, the first box read only by calls
/// that borrow it: the second boxing takes the first box instead of
/// copying the string again. Whatever the second box was to, kept by a
/// container or released, the first now is, and the analysis below
/// decides that from the merged uses. A borrowed box is left as it was,
/// so nothing that read it saw anything but the same string.
fn merge_repeated_boxings(func: &mut HirFunction, facts: &ModuleFacts) -> usize {
    use crate::analysis::DominatorTree;
    // (block, index, result, argument) of every boxing, in block order.
    let mut boxings: Vec<(HirId, usize, HirId, HirId)> = Vec::new();
    for (block_id, block) in &func.blocks {
        for (idx, inst) in block.instructions.iter().enumerate() {
            if let HirInstruction::Call {
                result: Some(result),
                callee,
                args,
                ..
            } = inst
            {
                if boxes_string(callee, facts) {
                    if let [arg] = args.as_slice() {
                        boxings.push((*block_id, idx, *result, *arg));
                    }
                }
            }
        }
    }
    if boxings.len() < 2 {
        return 0;
    }
    let dom = DominatorTree::new(func);
    // A dominator comes before what it dominates in reverse postorder,
    // so the earlier of a pair here is the one that can be kept.
    boxings.sort_by_key(|(block, idx, _, _)| (dom.rpo_position(*block), *idx));
    let mut replacements: indexmap::IndexMap<HirId, HirId> = indexmap::IndexMap::new();
    let mut removed: Vec<(HirId, usize)> = Vec::new();
    for (i, &(block_a, idx_a, box_a, arg)) in boxings.iter().enumerate() {
        if replacements.contains_key(&box_a) {
            continue;
        }
        let mut borrowed_only: Option<bool> = None;
        for &(block_b, idx_b, box_b, arg_b) in boxings.iter().skip(i + 1) {
            if arg_b != arg || replacements.contains_key(&box_b) {
                continue;
            }
            let ahead = if block_a == block_b {
                idx_a < idx_b
            } else {
                dom.strictly_dominates(block_a, block_b)
            };
            if !ahead {
                continue;
            }
            let only_borrows = *borrowed_only.get_or_insert_with(|| {
                let derived = derived_values_in(func, box_a, facts);
                uses_are_all_borrows(func, &derived, facts)
            });
            if !only_borrows {
                if trace_enabled() {
                    eprintln!(
                        "[drop] {}: boxing {box_a:?} is not only borrowed, so {box_b:?} stays",
                        func.name.resolve_global().unwrap_or_default()
                    );
                }
                break;
            }
            replacements.insert(box_b, box_a);
            removed.push((block_b, idx_b));
        }
    }
    if replacements.is_empty() {
        return 0;
    }
    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            inst.replace_uses(&replacements);
        }
        block.terminator.replace_uses(&replacements);
        for phi in block.phis.iter_mut() {
            for (value, _) in phi.incoming.iter_mut() {
                if let Some(&to) = replacements.get(value) {
                    *value = to;
                }
            }
        }
    }
    // Later indices first, so the earlier ones stay right.
    removed.sort_by(|a, b| b.cmp(a));
    for (block, idx) in removed {
        if let Some(b) = func.blocks.get_mut(&block) {
            b.instructions.remove(idx);
        }
    }
    replacements.len()
}

fn run_function(func: &mut HirFunction, facts: &ModuleFacts) -> DropStats {
    let mut stats = DropStats::default();
    if facts.automatic_release {
        let merged = merge_repeated_boxings(func, facts);
        if merged > 0 && trace_enabled() {
            eprintln!(
                "[drop] {}: {merged} repeated boxing(s) take the first box",
                func.name.resolve_global().unwrap_or_default()
            );
        }
    }
    let phases = std::env::var_os("ZYNTAX_TRACE_DROP_PHASES").is_some();
    let started = std::time::Instant::now();
    let mallocs: Vec<MallocSite> = collect_owned_sites(func, facts);
    let sites = mallocs.len();
    let collected = started.elapsed();
    let mut rebuilding = std::time::Duration::ZERO;
    let mut analyzing = std::time::Duration::ZERO;
    let mut applying = std::time::Duration::ZERO;
    let mut rebuilds = 0usize;
    // The use index every site's analysis walks, rebuilt when a release
    // inserted for one site has moved the instructions.
    let mut users = Users::of(func);
    for site in mallocs {
        stats.mallocs_scanned += 1;
        let at = std::time::Instant::now();
        if users.stale(func) {
            users = Users::of(func);
            rebuilds += 1;
        }
        rebuilding += at.elapsed();
        let at = std::time::Instant::now();
        let outcome = analyze_site(func, &site, facts, &users);
        analyzing += at.elapsed();
        let at = std::time::Instant::now();
        if trace_enabled() {
            eprintln!(
                "[drop] {}: site {} -> {}",
                func.name.resolve_global().unwrap_or_default(),
                describe_value(func, site.result),
                match &outcome {
                    SiteOutcome::SingleBlockDrop { .. } => "released in its block",
                    SiteOutcome::MultiBlockDrop { .. } => "released where it dies",
                    SiteOutcome::Escaped => "escapes",
                    SiteOutcome::MultiBlock => "live out of the function or merged",
                    SiteOutcome::NoUse => "never read",
                }
            );
        }
        match outcome {
            SiteOutcome::SingleBlockDrop { block, after_idx } => {
                if !adopt_into_box(func, facts, block, after_idx, site.result, site.release) {
                    insert_free_after(func, block, after_idx, site.result, site.release);
                }
                stats.frees_inserted += 1;
            }
            SiteOutcome::MultiBlockDrop { points } => {
                let sole = match points.as_slice() {
                    [Point::After(block, idx)] => Some((*block, *idx)),
                    _ => None,
                };
                if let Some((block, idx)) = sole {
                    if adopt_into_box(func, facts, block, idx, site.result, site.release) {
                        stats.frees_inserted += 1;
                        continue;
                    }
                }
                stats.frees_inserted += apply_points(func, points, site.result, site.release);
            }
            SiteOutcome::Escaped => stats.escapes_skipped += 1,
            SiteOutcome::MultiBlock => stats.multi_block_skipped += 1,
            SiteOutcome::NoUse => stats.no_use_skipped += 1,
        }
        applying += at.elapsed();
    }
    let at = std::time::Instant::now();
    stats.frees_inserted += release_owned_phis(func, facts);
    let phis = at.elapsed();
    if phases && started.elapsed().as_millis() >= 5 {
        let blocks = func.blocks.len();
        let insts: usize = func.blocks.values().map(|b| b.instructions.len()).sum();
        eprintln!(
            "[drop] {}: {blocks} blocks, {insts} insts, {sites} sites, {rebuilds} rebuilds; collect {:.1} ms, rebuild {:.1} ms, analyze {:.1} ms, apply {:.1} ms, phis {:.1} ms",
            func.name.resolve_global().unwrap_or_default(),
            collected.as_secs_f64() * 1000.0,
            rebuilding.as_secs_f64() * 1000.0,
            analyzing.as_secs_f64() * 1000.0,
            applying.as_secs_f64() * 1000.0,
            phis.as_secs_f64() * 1000.0,
        );
    }
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
        | "zyntax_box_pointer"
        | "zyntax_box_payload_i64"
        | "zyntax_box_payload_f64"
        | "zyntax_box_payload_bool"
        | "zyntax_box_hash"
        | "zyntax_box_set_hash" => Some(SymbolRole::BORROWS),
        // A box holding its own copy of a string, released with the box.
        STRING_TO_BOX => Some(SymbolRole::COPIES_INTO_BOX),
        // A box that took the string itself.
        STRING_INTO_BOX => Some(SymbolRole::KEEPS_INTO_BOX),
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
    /// A list header: the elements, in storage only the header names,
    /// are freed first, then the header.
    List,
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

/// A string released right after the call that boxes a copy of it is
/// given to the box instead: the call becomes the one that takes the
/// string as it is, and no release is placed. Only where that call is
/// the string's one release point, so nothing reads it afterwards.
fn adopt_into_box(
    func: &mut HirFunction,
    facts: &ModuleFacts,
    block: HirId,
    idx: usize,
    target: HirId,
    release: Release,
) -> bool {
    if release != Release::Symbol(STRING_FREE) {
        return false;
    }
    let Some(inst) = func
        .blocks
        .get_mut(&block)
        .and_then(|b| b.instructions.get_mut(idx))
    else {
        return false;
    };
    let HirInstruction::Call { callee, args, .. } = inst else {
        return false;
    };
    if args.as_slice() != [target] {
        return false;
    }
    if !boxes_string(callee, facts) {
        return false;
    }
    *callee = HirCallable::Symbol(STRING_INTO_BOX.to_string());
    true
}

/// Release `value` on the edge `from -> to`.
fn insert_free_on_edge(
    func: &mut HirFunction,
    from: HirId,
    to: HirId,
    value: HirId,
    release: Release,
) {
    let insts = release_instructions(func, value, release);
    let predecessors: Vec<HirId> = func
        .blocks
        .iter()
        .filter(|(_, b)| successors_of(b).contains(&to))
        .map(|(id, _)| *id)
        .collect();
    if predecessors.len() == 1 && predecessors[0] == from {
        if let Some(block) = func.blocks.get_mut(&to) {
            block.instructions.splice(0..0, insts);
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
        instructions: insts,
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
    term.retarget(from, to);
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
    if crate::ssa::is_list_header(ty) {
        return Release::List;
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
    derived: &IdSet,
    facts: &ModuleFacts,
) -> Option<Vec<Point>> {
    drop_points_transferring(func, site, derived, facts, &IdSet::default())
}

/// [`drop_points`] where leaving `transfer_out` blocks hands the storage
/// to whoever owns it past the edge, a phi that releases it later: the
/// value is live out of such a block whatever its successors do, so no
/// release is placed in it.
fn drop_points_transferring(
    func: &HirFunction,
    site: &MallocSite,
    derived: &IdSet,
    facts: &ModuleFacts,
    transfer_out: &IdSet,
) -> Option<Vec<Point>> {
    // Only blocks the entry can get to. An unreachable one has no
    // bearing on where the value dies and its successors would drag
    // liveness around the graph for nothing.
    let mut reachable = IdSet::default();
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
    let mut uses_block: IdSet = IdSet::default();
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
                UseKind::Escape => {
                    if trace_enabled() {
                        eprintln!("[drop]   escapes through {inst:?}");
                    }
                    return None;
                }
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
            UseKind::Escape => {
                if trace_enabled() {
                    eprintln!("[drop]   escapes through terminator {:?}", block.terminator);
                }
                return None;
            }
            UseKind::None => {}
        }
    }
    if uses_block.is_empty() {
        if trace_enabled() {
            eprintln!("[drop]   no block reads it");
        }
        return None;
    }

    // live_in[B] = B uses it, or its exit is live. The block holding
    // the allocation kills it: nothing before the call can be holding
    // what the call has not produced.
    let mut live_in: IdSet = IdSet::default();
    let mut live_out: IdSet = IdSet::default();
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
            None if *block_id == site.block && !uses_block.contains(block_id) => {
                if trace_enabled() {
                    eprintln!("[drop]   its own block neither reads it nor passes it on");
                }
                return None;
            }
            None => block.instructions.len(),
        };
        points.push(Point::After(*block_id, at.saturating_sub(1)));
    }
    if points.is_empty() && trace_enabled() {
        eprintln!("[drop]   live on every path out");
    }
    (!points.is_empty()).then_some(points)
}

/// A block's successors, read off its terminator rather than off the
/// cached list, which a pass that rewrote control flow may not have
/// kept up to date.
fn successors_of(block: &crate::hir::HirBlock) -> Vec<HirId> {
    block.terminator.targets()
}

fn analyze_site(
    func: &HirFunction,
    site: &MallocSite,
    facts: &ModuleFacts,
    users: &Users,
) -> SiteOutcome {
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
    let derived = derived_values_using(func, site.result, true, Some(facts), users);
    let target = site.result;
    let mut last_idx_in_block: Option<usize> = None;
    let mut had_any_use = false;

    // A value merged through a phi is another name past the merge, and
    // a release placed by that name's liveness would name this one where
    // it was never defined. Such storage is released through the phi
    // (`release_owned_phis`), never here, so this is decided before any
    // block's instructions are read.
    let merged = phis_using_any(func, &derived);
    if !merged.is_empty()
        || func
            .blocks
            .values()
            .any(|b| b.phis.iter().any(|p| p.result == target))
    {
        if trace_enabled() {
            eprintln!(
                "[drop] {}: site {:?} reaches a phi and is left to the phi: {}",
                func.name.resolve_global().unwrap_or_default(),
                target,
                merged
                    .iter()
                    .map(|p| describe_value(func, *p))
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
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
        // A `CondBranch` / `Switch` deciding on it (a null check, say)
        // is a normal use, but the value flows into successor blocks,
        // so it dies where the liveness walk says, as any other value
        // read past its block does.
        match derived
            .iter()
            .map(|d| classify_terminator_use(&block.terminator, *d))
            .fold(UseKind::None, strongest)
        {
            UseKind::None => {}
            UseKind::Use => {
                return match facts
                    .automatic_release
                    .then(|| drop_points(func, site, &derived, facts))
                    .flatten()
                {
                    Some(points) => SiteOutcome::MultiBlockDrop { points },
                    None => SiteOutcome::MultiBlock,
                };
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
pub(crate) fn derived_values(func: &HirFunction, root: HirId) -> IdSet {
    derived_values_with(func, root, true, None)
}

/// [`derived_values`] knowing which callees hand an argument back.
fn derived_values_in(func: &HirFunction, root: HirId, facts: &ModuleFacts) -> IdSet {
    derived_values_with(func, root, true, Some(facts))
}

/// [`derived_values_in`] stopping at phis: the names for the storage on
/// the paths where `root` itself is defined, which is where a release by
/// its name is defined too.
fn derived_values_local(func: &HirFunction, root: HirId, facts: &ModuleFacts) -> IdSet {
    derived_values_with(func, root, false, Some(facts))
}

fn derived_values_with(
    func: &HirFunction,
    root: HirId,
    through_phis: bool,
    facts: Option<&ModuleFacts>,
) -> IdSet {
    let users = Users::of(func);
    derived_values_using(func, root, through_phis, facts, &users)
}

/// Which instructions and phis use each value, so a flow from a root is
/// followed along its uses rather than by sweeping the function once
/// per root. Positions go stale when instructions are inserted; the
/// count says when.
pub(crate) struct Users {
    by_value: HashMap<HirId, Vec<(HirId, usize)>>,
    phi_users: HashMap<HirId, Vec<(HirId, usize)>>,
    instructions: usize,
}

impl Users {
    pub(crate) fn of(func: &HirFunction) -> Self {
        let mut by_value: HashMap<HirId, Vec<(HirId, usize)>> = HashMap::new();
        let mut phi_users: HashMap<HirId, Vec<(HirId, usize)>> = HashMap::new();
        let mut instructions = 0;
        for (&bid, block) in &func.blocks {
            instructions += block.instructions.len();
            for (i, inst) in block.instructions.iter().enumerate() {
                let mut seen: smallvec::SmallVec<[HirId; 4]> = smallvec::SmallVec::new();
                for v in inst.operands() {
                    if !seen.contains(&v) {
                        seen.push(v);
                        by_value.entry(v).or_default().push((bid, i));
                    }
                }
            }
            for (i, phi) in block.phis.iter().enumerate() {
                for (v, _) in &phi.incoming {
                    phi_users.entry(*v).or_default().push((bid, i));
                }
            }
        }
        Users {
            by_value,
            phi_users,
            instructions,
        }
    }

    /// Whether `func` has changed shape since this was built.
    pub(crate) fn stale(&self, func: &HirFunction) -> bool {
        func.blocks
            .values()
            .map(|b| b.instructions.len())
            .sum::<usize>()
            != self.instructions
    }
}

/// [`derived_values_with`] over a prepared use index.
fn derived_values_using(
    func: &HirFunction,
    root: HirId,
    through_phis: bool,
    facts: Option<&ModuleFacts>,
    users: &Users,
) -> IdSet {
    let mut set = IdSet::default();
    set.insert(root);
    // A list header owns the element storage its first field names:
    // releasing the header frees that too, so what is read out of the
    // field is another name for what the release ends. `heads` are the
    // names for the header's own address, which the field is read
    // through.
    let list = func
        .values
        .get(&root)
        .is_some_and(|v| crate::ssa::is_list_header(&v.ty));
    let mut heads = IdSet::default();
    if list {
        heads.insert(root);
    }
    let zero = |id: &HirId| {
        func.values.get(id).is_some_and(|v| {
            matches!(
                v.kind,
                crate::hir::HirValueKind::Constant(
                    HirConstant::I64(0)
                        | HirConstant::I32(0)
                        | HirConstant::U64(0)
                        | HirConstant::USize(0)
                )
            )
        })
    };
    // What one instruction makes of the values reached so far: another
    // name for the storage, or a head. The rules are monotone in both
    // sets, so a value is examined once, when it enters.
    let mut work: Vec<HirId> = vec![root];
    while let Some(v) = work.pop() {
        if through_phis {
            for (b, i) in users.phi_users.get(&v).map(|u| u.as_slice()).unwrap_or(&[]) {
                if let Some(phi) = func.blocks.get(b).and_then(|blk| blk.phis.get(*i))
                    && set.insert(phi.result)
                {
                    work.push(phi.result);
                }
            }
        }
        for (b, i) in users.by_value.get(&v).map(|u| u.as_slice()).unwrap_or(&[]) {
            let Some(inst) = func.blocks.get(b).and_then(|blk| blk.instructions.get(*i)) else {
                continue;
            };
            if list {
                match inst {
                    HirInstruction::Cast {
                        result, operand, ..
                    } if heads.contains(operand) => {
                        if heads.insert(*result) {
                            work.push(*result);
                        }
                    }
                    HirInstruction::GetElementPtr {
                        result,
                        ptr,
                        indices,
                        ..
                    } if heads.contains(ptr) && indices.iter().all(zero) => {
                        if heads.insert(*result) {
                            work.push(*result);
                        }
                    }
                    HirInstruction::Load {
                        result, ptr, ty, ..
                    } if heads.contains(ptr) && matches!(ty, HirType::Ptr(_)) => {
                        if set.insert(*result) {
                            work.push(*result);
                        }
                    }
                    // Growing the elements moves them: the result is
                    // the same storage at its new address.
                    HirInstruction::Call {
                        result: Some(result),
                        callee: HirCallable::Intrinsic(Intrinsic::Realloc),
                        args,
                        ..
                    } if args.first().is_some_and(|a| set.contains(a)) => {
                        if set.insert(*result) {
                            work.push(*result);
                        }
                    }
                    _ => {}
                }
            }
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
                if let Some(facts) = facts
                    && args
                        .iter()
                        .any(|a| set.contains(a) && facts.result_aliases_arg(callee, args, *a))
                    && set.insert(*result)
                {
                    work.push(*result);
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
            if inst.operands().iter().any(|o| set.contains(o))
                && let Some(result) = inst.result_id()
                && set.insert(result)
            {
                work.push(result);
            }
        }
    }
    set
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
fn classify_derived_use(inst: &HirInstruction, derived: &IdSet, facts: &ModuleFacts) -> UseKind {
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
        // A name for the storage kept inside the storage itself, as a
        // list header keeps its elements' address.
        HirInstruction::Store { value, ptr, .. }
            if derived.contains(value) && derived.contains(ptr) =>
        {
            return UseKind::Use;
        }
        // Growth whose result is known to be this storage moved: the
        // old address is given up for the new one, not for good.
        HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Intrinsic(Intrinsic::Realloc),
            args,
            ..
        } if derived.contains(result) && args.first().is_some_and(|a| derived.contains(a)) => {
            return UseKind::Use;
        }
        _ => {}
    }
    // Most instructions name none of the values: settled by one walk of
    // the operands before any is classified against each.
    let mut named = false;
    inst.for_each_operand(|v| named |= derived.contains(&v));
    if !named {
        return UseKind::None;
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
                // A copy reads and writes through its pointers and
                // keeps neither.
                HirCallable::Intrinsic(
                    Intrinsic::Memcpy | Intrinsic::Memmove | Intrinsic::Memset,
                ) => UseKind::Use,
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
    let insts = release_instructions(func, target, release);
    if let Some(block) = func.blocks.get_mut(&block_id) {
        let insert_at = (after_idx + 1).min(block.instructions.len());
        block.instructions.splice(insert_at..insert_at, insts);
    }
}

/// The instructions that release `value`, in order.
fn release_instructions(
    func: &mut HirFunction,
    value: HirId,
    release: Release,
) -> Vec<HirInstruction> {
    let call = |callee: HirCallable, arg: HirId| HirInstruction::Call {
        result: None,
        callee,
        args: vec![arg],
        type_args: Vec::new(),
        const_args: Vec::new(),
        is_tail: false,
    };
    match release {
        Release::Intrinsic => vec![call(HirCallable::Intrinsic(Intrinsic::Free), value)],
        Release::Symbol(name) => vec![call(HirCallable::Symbol(name.to_string()), value)],
        Release::Glue(id) => vec![call(HirCallable::Function(id), value)],
        Release::List => {
            // The element storage is whatever the header's first field
            // holds at this point, since growing the list replaces it.
            let data = HirId::new();
            let ty = HirType::Ptr(Box::new(HirType::U8));
            func.values.insert(
                data,
                crate::hir::HirValue {
                    id: data,
                    ty: ty.clone(),
                    kind: crate::hir::HirValueKind::Instruction,
                    uses: Default::default(),
                    span: None,
                },
            );
            vec![
                HirInstruction::Load {
                    result: data,
                    ty,
                    ptr: value,
                    align: 8,
                    volatile: false,
                },
                call(HirCallable::Intrinsic(Intrinsic::Free), data),
                call(HirCallable::Intrinsic(Intrinsic::Free), value),
            ]
        }
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
                uses: Default::default(),
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
                uses: Default::default(),
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

    /// Releasing a list header frees the elements first.
    ///
    /// `data = malloc; header = malloc List; store data -> header;
    /// d = load header; e = load d[0]; return e`. The element storage
    /// is stored into the header, so its own site escapes; the header's
    /// release reads it back out and frees it before the header. The
    /// element read through the loaded data pointer is a use of the
    /// header, so the release lands after it.
    #[test]
    fn a_list_header_release_frees_the_elements_first() {
        let mut f = HirFunction::new(
            InternedString::new_global("list_literal"),
            empty_sig(HirType::I64),
        );
        let entry = *f.blocks.keys().next().unwrap();
        let data_bytes = add_const(&mut f, HirType::I64, HirConstant::I64(32));
        let header_bytes = add_const(&mut f, HirType::I64, HirConstant::I64(24));
        let data = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let header = add_inst_val(
            &mut f,
            HirType::Ptr(Box::new(crate::ssa::list_header_type())),
        );
        let loaded = add_inst_val(&mut f, HirType::Ptr(Box::new(HirType::I64)));
        let elem = add_inst_val(&mut f, HirType::I64);
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(data),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![data_bytes],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::Call {
            result: Some(header),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![header_bytes],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::Store {
            value: data,
            ptr: header,
            align: 8,
            volatile: false,
        });
        block.instructions.push(HirInstruction::Load {
            result: loaded,
            ty: HirType::Ptr(Box::new(HirType::I64)),
            ptr: header,
            align: 8,
            volatile: false,
        });
        block.instructions.push(HirInstruction::Load {
            result: elem,
            ty: HirType::I64,
            ptr: loaded,
            align: 8,
            volatile: false,
        });
        block.terminator = HirTerminator::Return { values: vec![elem] };

        let stats = run_function(&mut f, &ModuleFacts::default());
        assert_eq!(stats.mallocs_scanned, 2);
        assert_eq!(
            stats.escapes_skipped, 1,
            "the element storage is the header's"
        );
        assert_eq!(stats.frees_inserted, 1);

        let block = f.blocks.values().next().unwrap();
        let freed: Vec<HirId> = block
            .instructions
            .iter()
            .filter_map(|i| match i {
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    args,
                    ..
                } => Some(args[0]),
                _ => None,
            })
            .collect();
        assert_eq!(freed.len(), 2, "elements and header");
        assert_eq!(freed[1], header, "the header goes last");
        let read_back = block
            .instructions
            .iter()
            .position(|i| matches!(i, HirInstruction::Load { result, .. } if *result == freed[0]));
        let last_use = block
            .instructions
            .iter()
            .position(|i| matches!(i, HirInstruction::Load { result, .. } if *result == elem));
        assert!(
            read_back > last_use,
            "the release reads the elements' address after the last element read"
        );
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

    fn caller_frees(m: &HirModule, caller_key: HirId) -> bool {
        let caller = m.functions.get(&caller_key).unwrap();
        let block = caller.blocks.values().next().unwrap();
        block.instructions.iter().any(|i| {
            matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    ..
                }
            )
        })
    }

    /// Facts recorded on a module's functions are what a later build
    /// reads: the same answers, and no fixed point over those bodies.
    #[test]
    fn recorded_facts_are_read_back() {
        let ctor_key = HirId::new();
        let caller_key = HirId::new();
        let mut m = HirModule::new(InternedString::new_global("m"));
        m.functions.insert(ctor_key, build_constructor());
        m.functions.insert(caller_key, build_caller(ctor_key));

        let fresh = facts_of(&m);
        assert!(fresh.returns_owned.contains(&ctor_key));
        record_facts(&mut m, &fresh);
        let recorded = m.functions[&ctor_key]
            .attributes
            .release_facts
            .as_ref()
            .expect("the constructor carries its facts");
        assert!(recorded.returns_owned);
        assert!(recorded.returns_param.is_empty());

        let seeded = facts_of(&m);
        assert!(seeded.returns_owned.contains(&ctor_key));
        assert!(!seeded.returns_owned.contains(&caller_key));
        assert_eq!(seeded.returns_param.get(&caller_key), Some(&Vec::new()));
    }

    /// A recorded fact stands in for the body: a constructor recorded as
    /// returning nothing it owns is not released by its caller, whatever
    /// its body does.
    #[test]
    fn a_recorded_fact_is_read_instead_of_the_body() {
        let ctor_key = HirId::new();
        let caller_key = HirId::new();
        let mut ctor = build_constructor();
        ctor.attributes.release_facts = Some(crate::hir::ReleaseFacts {
            returns_owned: false,
            returns_param: Vec::new(),
            automatic_release: false,
        });
        let mut m = HirModule::new(InternedString::new_global("m"));
        m.functions.insert(ctor_key, ctor);
        m.functions.insert(caller_key, build_caller(ctor_key));
        run_module(&mut m);
        assert!(!caller_frees(&m, caller_key));

        // Recorded under the other release setting, the fact is not
        // for this module, and the body decides.
        m.functions
            .get_mut(&ctor_key)
            .unwrap()
            .attributes
            .release_facts
            .as_mut()
            .unwrap()
            .automatic_release = true;
        run_module(&mut m);
        assert!(caller_frees(&m, caller_key));
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

    /// A string boxed, read through, and boxed again on the same path:
    /// `b1 = box(s); read(b1); b2 = box(s); return b2`.
    fn build_string_boxed_twice(first_escapes: bool) -> (HirFunction, HirId, HirId) {
        let mut f = HirFunction::new(
            InternedString::new_global("boxed_twice"),
            empty_sig(HirType::I64),
        );
        let entry = HirId::new();
        f.entry_block = entry;
        f.blocks.clear();
        f.blocks.insert(entry, HirBlock::new(entry));
        let s = add_const(&mut f, HirType::I64, HirConstant::I64(0));
        let first = add_inst_val(&mut f, HirType::I64);
        let read = add_inst_val(&mut f, HirType::F64);
        let second = add_inst_val(&mut f, HirType::I64);
        let boxing = |result| HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Symbol(STRING_TO_BOX.to_string()),
            args: vec![s],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        };
        let block = f.blocks.get_mut(&entry).unwrap();
        block.instructions.push(boxing(first));
        block.instructions.push(HirInstruction::Call {
            result: Some(read),
            callee: HirCallable::Symbol(
                if first_escapes {
                    "zyntax_list_push"
                } else {
                    "zyntax_box_get_f64"
                }
                .to_string(),
            ),
            args: vec![first],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(boxing(second));
        block.terminator = HirTerminator::Return {
            values: vec![second],
        };
        (f, first, second)
    }

    #[test]
    fn a_second_boxing_of_a_borrowed_box_takes_the_first() {
        let (mut f, first, second) = build_string_boxed_twice(false);
        let facts = ModuleFacts {
            automatic_release: true,
            ..ModuleFacts::default()
        };
        let stats = run_function(&mut f, &facts);
        let block = f.blocks.values().next().unwrap();
        let boxings = block
            .instructions
            .iter()
            .filter(|i| matches!(i, HirInstruction::Call { callee: HirCallable::Symbol(n), .. } if n == STRING_TO_BOX))
            .count();
        assert_eq!(boxings, 1, "the second boxing is gone");
        assert!(
            matches!(&block.terminator, HirTerminator::Return { values } if values == &vec![first]),
            "what was returned is the first box"
        );
        assert!(
            !block
                .instructions
                .iter()
                .any(|i| i.result_id() == Some(second)),
            "the second box is not defined any more"
        );
        assert_eq!(
            stats.frees_inserted, 0,
            "the first box is returned now, so it is not released"
        );
    }

    #[test]
    fn a_second_boxing_stays_when_the_first_box_is_kept() {
        let (mut f, _, second) = build_string_boxed_twice(true);
        let facts = ModuleFacts {
            automatic_release: true,
            ..ModuleFacts::default()
        };
        run_function(&mut f, &facts);
        let block = f.blocks.values().next().unwrap();
        let boxings = block
            .instructions
            .iter()
            .filter(|i| matches!(i, HirInstruction::Call { callee: HirCallable::Symbol(n), .. } if n == STRING_TO_BOX))
            .count();
        assert_eq!(boxings, 2, "a kept box is not shared with a later one");
        assert!(
            matches!(&block.terminator, HirTerminator::Return { values } if values == &vec![second])
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

    fn string_param() -> crate::hir::HirParam {
        crate::hir::HirParam {
            id: HirId::new(),
            name: InternedString::new_global("v"),
            ty: HirType::Ptr(Box::new(HirType::I8)),
            attributes: Default::default(),
            ownership: crate::hir::ParamOwnership::Borrowed,
        }
    }

    fn string_to_box_sig() -> HirFunctionSignature {
        HirFunctionSignature {
            params: vec![string_param()],
            ..empty_sig(HirType::I64)
        }
    }

    /// A module with the extern for the copying box call, the wrapper
    /// `wrap(v) { return to_dyn(v) }` around it, and
    /// `boxed() { s = concat(a, b); return wrap(s) }`.
    fn build_boxing_through_wrapper() -> (HirModule, HirId, HirId) {
        let extern_key = HirId::new();
        let mut to_dyn =
            HirFunction::new(InternedString::new_global("to_dyn"), string_to_box_sig());
        to_dyn.is_external = true;
        to_dyn.link_name = Some(STRING_TO_BOX.to_string());

        let wrap_key = HirId::new();
        let mut wrap = HirFunction::new(InternedString::new_global("wrap"), string_to_box_sig());
        let entry = HirId::new();
        wrap.entry_block = entry;
        wrap.blocks.clear();
        wrap.blocks.insert(entry, HirBlock::new(entry));
        let v = HirId::new();
        wrap.values.insert(
            v,
            HirValue {
                id: v,
                ty: HirType::Ptr(Box::new(HirType::I8)),
                kind: HirValueKind::Parameter(0),
                uses: Default::default(),
                span: None,
            },
        );
        let boxed = add_inst_val(&mut wrap, HirType::I64);
        let block = wrap.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(boxed),
            callee: HirCallable::Function(extern_key),
            args: vec![v],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return {
            values: vec![boxed],
        };

        let caller_key = HirId::new();
        let mut caller =
            HirFunction::new(InternedString::new_global("boxed"), empty_sig(HirType::I64));
        let entry = HirId::new();
        caller.entry_block = entry;
        caller.blocks.clear();
        caller.blocks.insert(entry, HirBlock::new(entry));
        let a = add_const(
            &mut caller,
            HirType::Ptr(Box::new(HirType::I8)),
            HirConstant::I64(0),
        );
        let s = add_inst_val(&mut caller, HirType::Ptr(Box::new(HirType::I8)));
        let result = add_inst_val(&mut caller, HirType::I64);
        let block = caller.blocks.get_mut(&entry).unwrap();
        block.instructions.push(HirInstruction::Call {
            result: Some(s),
            callee: HirCallable::Symbol("$IO$string_concat".to_string()),
            args: vec![a, a],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.instructions.push(HirInstruction::Call {
            result: Some(result),
            callee: HirCallable::Function(wrap_key),
            args: vec![s],
            type_args: Vec::new(),
            const_args: Vec::new(),
            is_tail: false,
        });
        block.terminator = HirTerminator::Return {
            values: vec![result],
        };

        let mut m = HirModule::new(InternedString::new_global("m"));
        m.functions.insert(extern_key, to_dyn);
        m.functions.insert(wrap_key, wrap);
        m.functions.insert(caller_key, caller);
        (m, caller_key, s)
    }

    /// A fresh string whose last use is the wrapper around the copying
    /// box call is given to the box instead of being copied and freed.
    #[test]
    fn a_string_boxed_through_the_wrapper_is_adopted() {
        let (mut m, caller_key, s) = build_boxing_through_wrapper();
        run_module(&mut m);
        let caller = m.functions.get(&caller_key).unwrap();
        let block = caller.blocks.values().next().unwrap();
        let adopted = block.instructions.iter().any(|i| {
            matches!(
                i,
                HirInstruction::Call { callee: HirCallable::Symbol(n), args, .. }
                    if n == STRING_INTO_BOX && args == &vec![s]
            )
        });
        assert!(
            adopted,
            "expected the adopting box call, got {:?}",
            block.instructions
        );
        let freed = block.instructions.iter().any(|i| {
            matches!(
                i,
                HirInstruction::Call { callee: HirCallable::Symbol(n), .. } if n == STRING_FREE
            )
        });
        assert!(
            !freed,
            "the box owns the string now, got {:?}",
            block.instructions
        );
    }
}
