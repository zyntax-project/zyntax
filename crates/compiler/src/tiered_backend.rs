//! # Tiered Compilation Backend (beadie-driven)
//!
//! Multi-tier JIT compilation with hot-function promotion. Uses
//! [`beadie::TieredAdapter`] under the hood — it owns the per-tier broker
//! threads, atomic code-pointer swap, generations, and (later) OSR / deopt
//! infrastructure.
//!
//! ## Optimization tiers
//! - **Tier 0 (Interpreter)** — HIR bytecode interpreter, the cold-start
//!   path before any JIT touches a function. Lives in `hir_interp`; not
//!   driven by beadie. Promotes to Baseline on the first hotness sample.
//! - **Tier 1 (Baseline)** — Cranelift, eagerly compiled at module load
//!   on native or wasm-emitted on wasm targets. Beadie generation 0.
//! - **Tier 2 (Standard)** — Cranelift recompile, promoted at the warm
//!   threshold from `ProfileConfig`. Beadie generation 1.
//! - **Tier 3 (Optimized)** — Cranelift or LLVM recompile, promoted at
//!   the hot threshold. Beadie generation 2.
//!
//! Note: the variants below are the JIT-tier ladder only — they're
//! what beadie's broker schedules. The `Interpreter` tier is OUTSIDE
//! this enum because the JIT broker never schedules into it (it's the
//! starting point). Callers ask `function_tier()` and get back one of
//! the JIT tiers once a function has been baselined; before that, the
//! function is implicitly in the `Interpreter` tier.
//!
//! ## Public API
//! Mirrors the previous hand-rolled implementation 1:1 so embedders
//! (`zyntax_embed::TieredRuntime`) keep working without changes.
//!
//! ## Phase boundaries
//! Phase 1 (this file): swap implementation, keep behavior.
//! Phase 2/3 will add OSR; phase 4 will add deopt-on-speculation.
//! See `crates/compiler/BEADIE_INTEGRATION.md`.

use std::collections::HashMap;
use std::ptr;
use std::sync::{Arc, RwLock};

use beadie::{Bead, HotnessPolicy, JitBackend, ThresholdPolicy, TieredAdapter, TieredBound};

use crate::beadie_adapter::{ZyntaxCraneliftBackend, ZyntaxFunctionDef};
use crate::cranelift_backend::CraneliftBackend;
use crate::hir::{HirFunction, HirId, HirModule};
use crate::osr;
use crate::profiling::{ProfileConfig, ProfileData};
use crate::{CompilerError, CompilerResult};

#[cfg(feature = "llvm-backend")]
use crate::beadie_adapter::ZyntaxLlvmBackend;
#[cfg(feature = "llvm-backend")]
use crate::llvm_jit_backend::LLVMJitBackend;
#[cfg(feature = "llvm-backend")]
use inkwell::context::Context;

// ─────────────────────────────────────────────────────────────────────────────
// Public types (preserved from the legacy API)
// ─────────────────────────────────────────────────────────────────────────────

/// Optimization tier level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptimizationTier {
    Baseline,  // Tier 0 — fast compile, minimal opt
    Standard,  // Tier 1 — moderate opt
    Optimized, // Tier 2 — aggressive opt
}

impl OptimizationTier {
    pub fn cranelift_opt_level(&self) -> &'static str {
        match self {
            OptimizationTier::Baseline => "none",
            OptimizationTier::Standard => "speed",
            OptimizationTier::Optimized => "speed_and_size",
        }
    }

    pub fn next_tier(&self) -> Option<OptimizationTier> {
        match self {
            OptimizationTier::Baseline => Some(OptimizationTier::Standard),
            OptimizationTier::Standard => Some(OptimizationTier::Optimized),
            OptimizationTier::Optimized => None,
        }
    }

    fn index(self) -> usize {
        match self {
            OptimizationTier::Baseline => 0,
            OptimizationTier::Standard => 1,
            OptimizationTier::Optimized => 2,
        }
    }

    fn from_index(idx: usize) -> Option<OptimizationTier> {
        match idx {
            0 => Some(OptimizationTier::Baseline),
            1 => Some(OptimizationTier::Standard),
            2 => Some(OptimizationTier::Optimized),
            _ => None,
        }
    }
}

/// Backend choice for tier 2 (hot code).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier2Backend {
    Cranelift,
    #[cfg(feature = "llvm-backend")]
    LLVM,
}

/// Configuration for tiered compilation.
#[derive(Debug, Clone)]
pub struct TieredConfig {
    pub profile_config: ProfileConfig,
    /// Kept for API compat — beadie always uses background broker threads,
    /// so disabling it has no effect now. Setting to `false` would have
    /// required removing the broker entirely; instead we honor it by simply
    /// never crossing the promotion threshold (the tier 1/2 thresholds are
    /// effectively `u32::MAX`).
    pub enable_background_optimization: bool,
    /// Kept for API compat; not used by beadie's broker (it polls a channel).
    pub optimization_check_interval_ms: u64,
    /// Kept for API compat; beadie runs one worker thread per tier.
    pub max_parallel_optimizations: usize,
    pub tier2_backend: Tier2Backend,
    pub verbosity: u8,
    /// Caller-supplied content hash for the in-process LLVM dylib
    /// cache. When `Some`, the LLVM backend keys its cached `.so`
    /// (loaded once via `dlopen`) on this string XOR a runtime-symbol
    /// address fingerprint, and subsequent installs of an identical
    /// module reuse the existing function pointers — the bench
    /// harness uses this to skip the 270-330 ms macOS dlopen on
    /// iteration 2+ of the same kernel. When `None`, caching is off
    /// and every install pays the full pipeline cost.
    pub llvm_cache_key: Option<String>,
    /// Emit on-stack-replacement probes at tier-0 loop back-edges.
    ///
    /// A function entered once that runs a long loop cannot be promoted by
    /// call count — it never returns to be re-dispatched. With this on, its
    /// back-edges pick up a tier-1 helper as soon as one is installed and
    /// the frame finishes in the faster tier, which is what a cold-start
    /// workload needs: a worker or serverless invocation is often a single
    /// long call, and warming up first is not an option.
    ///
    /// Costs a load and a not-taken branch per back-edge while no helper
    /// exists, measured at under 1% across the bench kernels.
    pub enable_osr: bool,
    /// Route calls between compiled functions through reload cells so
    /// `reload_module` can replace a function under running code.
    pub enable_hot_reload: bool,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            profile_config: ProfileConfig::default(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 100,
            max_parallel_optimizations: 4,
            tier2_backend: Tier2Backend::Cranelift,
            verbosity: 0,
            llvm_cache_key: None,
            enable_osr: true,
            enable_hot_reload: false,
        }
    }
}

impl TieredConfig {
    pub fn development() -> Self {
        Self {
            profile_config: ProfileConfig::development(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 50,
            max_parallel_optimizations: 2,
            tier2_backend: Tier2Backend::Cranelift,
            verbosity: 2,
            llvm_cache_key: None,
            enable_osr: true,
            enable_hot_reload: false,
        }
    }

    pub fn production() -> Self {
        Self {
            profile_config: ProfileConfig::production(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 1000,
            max_parallel_optimizations: 8,
            tier2_backend: Tier2Backend::Cranelift,
            verbosity: 0,
            llvm_cache_key: None,
            enable_osr: true,
            enable_hot_reload: false,
        }
    }

    #[cfg(feature = "llvm-backend")]
    pub fn production_llvm() -> Self {
        Self {
            profile_config: ProfileConfig::production(),
            enable_background_optimization: true,
            optimization_check_interval_ms: 1000,
            max_parallel_optimizations: 8,
            tier2_backend: Tier2Backend::LLVM,
            verbosity: 0,
            llvm_cache_key: None,
            enable_osr: true,
            enable_hot_reload: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────────────

/// Per-function state held alongside its beadie bound bead.
struct FunctionEntry {
    bound: TieredBound,
    /// Pre-cloned HIR function, captured by promotion closures.
    function: Arc<HirFunction>,
    /// Shared module context needed to recompile effectful functions. A
    /// per-function promotion cannot resolve effects, handlers, globals, or
    /// callees from the function body alone.
    module: Arc<HirModule>,
    /// OSR registry id for this function. Embedded as a constant in
    /// tier-0 probe call sites so JIT'd code can find the bead.
    bead_id: u64,
}

/// Everything needed to restore the generation a reload replaced.
#[derive(Default)]
struct ReloadUndo {
    swapped: Vec<UndoSwap>,
}

/// Per-function undo record. Raw addresses are stored as `usize`; the
/// cells, beads and vtable globals they point at live for the process.
struct UndoSwap {
    id: HirId,
    name: String,
    /// Entry pointer the bead held before the swap (0 if none).
    old_entry: usize,
    old_body: Arc<HirFunction>,
    /// OSR resume points this reload published, to be unpublished.
    helper_sites: Vec<(u64, u64)>,
    /// Dispatch-table slots this reload patched: (slot address, value
    /// the slot held before).
    vtable_slots: Vec<(usize, usize)>,
}

/// Runtime symbol entry for FFI registration.
#[derive(Clone)]
struct RuntimeSymbol {
    name: String,
    /// `*const u8` cast to `usize` so the entry stays `Send`/`Sync`.
    ptr: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// TieredBackend
// ─────────────────────────────────────────────────────────────────────────────

pub struct TieredBackend {
    /// Beadie's tiered adapter: owns broker threads + per-bead state.
    /// Shared so a promotion request raised from running JIT'd code can
    /// submit a compile without reaching back into the backend.
    adapter: Arc<TieredAdapter>,

    /// Cranelift backend, locked behind a `Mutex` and shared with worker
    /// threads via `Arc`.
    cranelift: Arc<ZyntaxCraneliftBackend>,

    /// Optional LLVM backend for tier 2 hot code.
    #[cfg(feature = "llvm-backend")]
    llvm: Option<Arc<ZyntaxLlvmBackend>>,

    /// Owned LLVM context (must outlive the backend it powers).
    /// `Option` is used only so we can move it during `shutdown`.
    #[cfg(feature = "llvm-backend")]
    _llvm_context: Option<Box<Context>>,

    /// Per-function entries keyed by HIR function id.
    functions: HashMap<HirId, FunctionEntry>,
    /// The module the compiled code came from. A reload diffs the
    /// edited module against this and replaces it piecewise.
    current_module: Option<HirModule>,
    /// Undo record for the most recent applied reload, consumed by
    /// [`Self::rollback_last_reload`].
    last_undo: Option<ReloadUndo>,
    /// What a reload does with live state whose layout an edit changed.
    state_migration: crate::reload::StateMigration,

    /// Profile counters (for `get_statistics` only — promotion is driven by
    /// beadie's own counters).
    profile_data: ProfileData,

    /// Runtime FFI symbols registered post-construction.
    runtime_symbols: Arc<RwLock<Vec<RuntimeSymbol>>>,

    config: TieredConfig,
}

impl TieredBackend {
    /// Build the tiered backend.
    pub fn new(config: TieredConfig) -> CompilerResult<Self> {
        // Wire OSR runtime symbols so JIT'd back-edge code resolves them.
        let osr_syms = osr::osr_runtime_symbols();
        let cranelift_inner = CraneliftBackend::with_runtime_symbols(&osr_syms)?;
        let cranelift = Arc::new(ZyntaxCraneliftBackend::new(cranelift_inner));

        #[cfg(feature = "llvm-backend")]
        let (_llvm_context, llvm) = if matches!(config.tier2_backend, Tier2Backend::LLVM) {
            let context = Box::new(Context::create());
            // SAFETY: the `Box<Context>` is held alive for the lifetime of
            // `TieredBackend`. We hand a `'static` reference to the JIT
            // backend; the backend will never observe the context drop
            // before itself.
            let context_ref = unsafe { &*(context.as_ref() as *const Context) };
            let jit = LLVMJitBackend::new(context_ref)?;
            (Some(context), Some(Arc::new(ZyntaxLlvmBackend::new(jit))))
        } else {
            (None, None)
        };

        // With an LLVM tier above it, Cranelift's helpers are not resume
        // points worth having: tier 1 emits the same code as tier 0, and a
        // transfer into it consumes the loop's one chance to move up.
        #[cfg(feature = "llvm-backend")]
        if llvm.is_some() {
            cranelift.with_lock(|be| be.set_publish_osr_helpers(false));
        }

        if config.enable_hot_reload {
            cranelift.with_lock(|be| {
                be.set_reloadable_calls(true);
                // A helper carries no probes, so a frame that transfers
                // into one can never migrate again. Under hot reload the
                // migration worth keeping available is into *edited*
                // code, and the ladder's own helpers re-emit the code the
                // loop is already running.
                be.set_publish_osr_helpers(false);
            });
        }

        let adapter = Arc::new(TieredAdapter::new(make_policies(&config)));

        Ok(Self {
            adapter,
            cranelift,
            #[cfg(feature = "llvm-backend")]
            llvm,
            #[cfg(feature = "llvm-backend")]
            _llvm_context,
            functions: HashMap::new(),
            current_module: None,
            last_undo: None,
            state_migration: crate::reload::StateMigration::default(),
            profile_data: ProfileData::new(config.profile_config.clone()),
            runtime_symbols: Arc::new(RwLock::new(Vec::new())),
            config,
        })
    }

    /// Compile a HIR module — bulk-emits every function at tier 0 and
    /// registers each with the beadie adapter.
    pub fn compile_module(&mut self, module: HirModule) -> CompilerResult<()> {
        if self.config.verbosity >= 1 {
            eprintln!(
                "[TieredBackend] Compiling {} functions at Tier 0 (Baseline)",
                module.functions.len()
            );
        }

        // If extern declarations in this module reference symbols that
        // got registered after the JIT module was constructed (typically
        // from `load_plugin`), rebuild the JIT module with the accumulated
        // symbol set before compilation. Mirrors
        // `ZyntaxRuntime::compile_module` in zyntax_embed.
        // A rebuild replaces the JIT module and clears the backend's
        // function and global maps, since both index into the module it
        // discards. Anything already installed loses its entries, and
        // nothing re-declares them: a handler's `$optable$` global then
        // has no address, and the host reports the handler as missing.
        // So recompile what was installed before, after the rebuild.
        let rebuilt = self.cranelift.with_lock(|be| {
            if be.needs_rebuild_for_module(&module) {
                be.rebuild_with_accumulated_symbols().map(|()| true)
            } else {
                Ok(false)
            }
        })?;
        if rebuilt {
            if let Some(previous) = self.current_module.clone() {
                self.cranelift
                    .with_lock(|be| be.compile_module(&previous))?;
            }
        }

        // Bead ids have to exist before codegen, not after: a tier-0 probe
        // bakes its function's id into the address of the slot it loads, so
        // allocating them afterwards left every probe reading slot zero
        // while helpers published under the real ids — the transfer could
        // never happen.
        let bead_ids: HashMap<HirId, u64> = module
            .functions
            .keys()
            .map(|id| (*id, osr::next_bead_id()))
            .collect();
        self.cranelift
            .with_lock(|be| be.set_bead_ids(bead_ids.clone()));

        self.cranelift.with_lock(|be| be.compile_module(&module))?;

        self.current_module = Some(module.clone());

        // Hand the LLVM tier the whole module before anything promotes out
        // of it: a promotion recompiles one function, and that function's
        // callees have to come with it.
        #[cfg(feature = "llvm-backend")]
        if let Some(llvm) = &self.llvm {
            let shared = Arc::new(module.clone());
            llvm.with_lock(|be| be.set_module_context(Arc::clone(&shared)));
        }

        let module_context = Arc::new(module.clone());
        for (func_id, function) in module.functions.iter() {
            let bound = self.adapter.register(ptr::null_mut(), None);

            // Eagerly install the tier-0 code pointer so the bead reports
            // `Compiled(gen=0)` from the very first invocation.
            if let Some(p) = self.cranelift.with_lock(|be| be.get_function_ptr(*func_id)) {
                bound.bead().eager_install(p as *mut ());
            } else if osr::osr_trace_enabled() {
                eprintln!(
                    "[reload] no entry pointer for {:?} ({:?}) after module compile",
                    function.name.resolve_global().unwrap_or_default(),
                    func_id
                );
            }

            // Allocate a stable id and publish the bead in the OSR
            // registry so JIT'd probes can find it.
            let bead_id = bead_ids
                .get(func_id)
                .copied()
                .unwrap_or_else(osr::next_bead_id);
            osr::register_bead(bead_id, Arc::clone(bound.bead()));

            self.functions.insert(
                *func_id,
                FunctionEntry {
                    bound,
                    function: Arc::new(function.clone()),
                    module: Arc::clone(&module_context),
                    bead_id,
                },
            );
        }

        // Every bead now exists, so the handler can capture them.
        self.install_promotion_requester();

        Ok(())
    }

    /// Replace the parts of the running module the edited one changed.
    ///
    /// Functions are matched by name and compared by content
    /// fingerprint, so a fresh parse's disjoint `HirId`s and
    /// formatting-only edits both diff as unchanged. A changed function
    /// is recompiled under its existing id — its bead, reload cell and
    /// pointer-table entries all stay keyed as before — with the edited
    /// body's callee references remapped onto the running module's ids.
    /// Old code is retained; frames already inside it complete safely.
    pub fn reload_module(
        &mut self,
        new_module: &HirModule,
    ) -> CompilerResult<crate::reload::ReloadReport> {
        use std::collections::HashMap as Map;

        let old_module = self.current_module.clone().ok_or_else(|| {
            CompilerError::Backend("reload before any module was compiled".into())
        })?;

        let name_of = |f: &HirFunction| f.name.resolve_global();

        let mut old_by_name: Map<String, HirId> = Map::new();
        for (id, f) in &old_module.functions {
            if !f.is_external {
                if let Some(n) = name_of(f) {
                    old_by_name.insert(n, *id);
                }
            }
        }

        // Edited ids -> running ids, for every name present in both.
        let mut id_remap: Map<HirId, HirId> = Map::new();
        for (new_id, f) in &new_module.functions {
            if let Some(n) = name_of(f) {
                if let Some(old_id) = old_by_name.get(&n) {
                    id_remap.insert(*new_id, *old_id);
                }
            }
        }

        // Globals are matched by name and REUSED, never recompiled:
        // their addresses are live — handler frames on the stack hold
        // op-table pointers, module state lives in the data itself — so
        // the edited bodies are rewritten onto the running ids instead.
        // A global the edit introduces has no counterpart and is
        // compiled into the running module below.
        let mut old_globals_by_name: Map<String, HirId> = Map::new();
        for (id, g) in &old_module.globals {
            if let Some(n) = g.name.resolve_global() {
                old_globals_by_name.insert(n, *id);
            }
        }
        let mut global_remap: Map<HirId, HirId> = Map::new();
        let mut fresh_globals: Vec<HirId> = Vec::new();
        for (new_gid, g) in &new_module.globals {
            match g
                .name
                .resolve_global()
                .and_then(|n| old_globals_by_name.get(&n))
            {
                Some(old_gid) => {
                    global_remap.insert(*new_gid, *old_gid);
                }
                None => fresh_globals.push(*new_gid),
            }
        }

        // An effect's identity crosses generations as a number: a
        // `with` scope pushes its handler under the effect's id and a
        // perform looks the handler up by the same number. An edited
        // body carries the EDITED module's ids, so every one of them —
        // the typed `PerformEffect` field and the constant a `with`
        // scope passes to the push — is rewritten onto the running
        // program's, or a reloaded perform would miss handlers pushed
        // by code that did not reload (and vice versa).
        let mut old_effects_by_name: Map<String, HirId> = Map::new();
        for (id, e) in &old_module.effects {
            if let Some(n) = e.name.resolve_global() {
                old_effects_by_name.insert(n, *id);
            }
        }
        let mut effect_remap: Map<HirId, HirId> = Map::new();
        for (new_eid, e) in &new_module.effects {
            if let Some(old_eid) = e
                .name
                .resolve_global()
                .and_then(|n| old_effects_by_name.get(&n))
            {
                if old_eid != new_eid {
                    effect_remap.insert(*new_eid, *old_eid);
                }
            }
        }
        let effect_const_remap: Map<i64, i64> = effect_remap
            .iter()
            .map(|(n, o)| (n.as_u32() as i64, o.as_u32() as i64))
            .collect();

        // The module the reloaded bodies compile against: the running
        // program plus whatever the edit introduced. Codegen consults
        // it for an effect's operation order at a perform site, so it
        // has to be keyed by the ids the remapped bodies carry.
        let mut merged = old_module.clone();
        for (new_eid, e) in &new_module.effects {
            if !effect_remap.contains_key(new_eid) && !merged.effects.contains_key(new_eid) {
                merged.effects.insert(*new_eid, e.clone());
            }
        }
        for (new_hid, h) in &new_module.handlers {
            if !merged.handlers.values().any(|o| o.name == h.name) {
                let mut h = h.clone();
                h.effect_id = *effect_remap.get(&h.effect_id).unwrap_or(&h.effect_id);
                merged.handlers.insert(*new_hid, h);
            }
        }
        for gid in &fresh_globals {
            let mut g = new_module.globals[gid].clone();
            if let Some(crate::hir::HirConstant::VTable(vt)) = &mut g.initializer {
                for entry in &mut vt.methods {
                    if let Some(mapped) = id_remap.get(&entry.function_id) {
                        entry.function_id = *mapped;
                    }
                }
            }
            merged.globals.insert(*gid, g);
        }

        let mut report = crate::reload::ReloadReport::default();
        let mut seen_names: std::collections::HashSet<String> = Default::default();

        // The compile pass prepares these; nothing is applied unless
        // the whole edit set compiled.
        struct PreparedChange {
            old_id: HirId,
            name: String,
            body: HirFunction,
            bead_id: u64,
            entry_ptr: usize,
            pending_resume: Vec<(u64, *mut ())>,
        }
        struct PreparedAdd {
            new_id: HirId,
            name: String,
            body: HirFunction,
            bead_id: u64,
            entry_ptr: usize,
        }
        let mut changes: Vec<PreparedChange> = Vec::new();
        let mut adds: Vec<PreparedAdd> = Vec::new();
        let mut compile_failed: Vec<(String, String)> = Vec::new();
        // Test hook: treat the named function as a compile failure, to
        // exercise the all-or-nothing apply.
        let inject_fail = std::env::var("ZYNTAX_RELOAD_INJECT_FAIL").ok();

        // A stateful handler's state struct is shared between its ctor
        // (which allocates it) and its ops (which read it through
        // `self`). If an edit changes that layout, no piecewise reload
        // is sound: patched ops would read new offsets out of state old
        // ctors allocated, and vice versa. Decline every changed member
        // of such a handler — ops and ctor together — so every
        // generation in flight keeps a consistent view.
        let mut layout_declined: Map<String, String> = Map::new();
        {
            let new_ctors: Map<String, &HirFunction> = new_module
                .functions
                .values()
                .filter(|f| !f.is_external)
                .filter_map(|f| name_of(f).map(|n| (n, f)))
                .filter(|(n, _)| n.ends_with("$new"))
                .collect();
            for global in old_module.globals.values() {
                let Some(crate::hir::HirConstant::VTable(vt)) = &global.initializer else {
                    continue;
                };
                let Some(gname) = global.name.resolve_global() else {
                    continue;
                };
                let Some(handler) = gname.strip_prefix("$optable$") else {
                    continue;
                };
                let ctor_name = format!("{handler}$new");
                // A stateless handler has no ctor and no shared layout.
                let Some(&old_ctor_id) = old_by_name.get(&ctor_name) else {
                    continue;
                };
                let Some(new_ctor) = new_ctors.get(&ctor_name) else {
                    continue;
                };
                let old_ctor = &old_module.functions[&old_ctor_id];
                let old_layout = old_ctor.signature.returns.first().map(type_layout_key);
                let new_layout = new_ctor.signature.returns.first().map(type_layout_key);
                if old_layout == new_layout {
                    continue;
                }
                // With a migration policy the group reloads instead,
                // and every live region moves field-by-field into the
                // edited layout; without one the whole group keeps its
                // previous implementation so ctor and ops agree.
                if self.state_migration == crate::reload::StateMigration::ByFieldName {
                    if let Some(plan) = self.plan_state_migration(
                        handler,
                        &ctor_name,
                        old_ctor,
                        new_ctor,
                        &old_module,
                        new_module,
                    ) {
                        report.state_migrations.push(plan);
                        continue;
                    }
                }
                let reason = format!(
                    "handler {handler}: state layout changed; the running \
                     state cannot be read through the edited shape, so the \
                     handler keeps its previous implementation"
                );
                for entry in &vt.methods {
                    if let Some(op_name) = old_module
                        .functions
                        .get(&entry.function_id)
                        .and_then(name_of)
                    {
                        layout_declined.insert(op_name, reason.clone());
                    }
                }
                layout_declined.insert(ctor_name, reason);
            }
        }

        self.cranelift
            .with_lock(|be| be.set_defer_cell_publish(true));

        // A dispatch table the edit introduces holds the addresses of
        // that edit's own functions, which are not compiled yet. It is
        // emitted with empty slots here — so nothing it names dangles
        // through the finalizes that follow each compile — and filled
        // once the addresses exist, by the same atomic store that
        // patches a table already in use.
        if !fresh_globals.is_empty() {
            let merged_ref = &merged;
            let prepared = self.cranelift.with_lock(|be| {
                for gid in &fresh_globals {
                    let global = &merged_ref.globals[gid];
                    be.declare_global(*gid, global)?;
                    match &global.initializer {
                        Some(crate::hir::HirConstant::VTable(vt)) => {
                            be.define_empty_vtable(*gid, vt.methods.len())?
                        }
                        _ => be.define_global(*gid, global)?,
                    }
                }
                Ok::<_, CompilerError>(())
            });
            if let Err(e) = prepared {
                self.cranelift
                    .with_lock(|be| be.set_defer_cell_publish(false));
                let _ = self.cranelift.with_lock(|be| be.take_deferred_cells());
                report.failed.push((
                    "<globals>".to_string(),
                    format!("could not emit the edit's dispatch tables: {e}"),
                ));
                report.aborted = true;
                return Ok(report);
            }
        }

        for (new_id, new_fn) in &new_module.functions {
            if new_fn.is_external {
                continue;
            }
            let Some(name) = name_of(new_fn) else {
                continue;
            };
            seen_names.insert(name.clone());

            match old_by_name.get(&name) {
                Some(&old_id) => {
                    let old_fn = &old_module.functions[&old_id];
                    let fp_old = crate::reload::function_fingerprint(old_fn, &old_module);
                    let fp_new = crate::reload::function_fingerprint(new_fn, new_module);
                    if fp_old == fp_new {
                        report.unchanged += 1;
                        continue;
                    }
                    if let Some(reason) = layout_declined.get(&name) {
                        report.failed.push((name, reason.clone()));
                        continue;
                    }
                    if std::env::var_os("ZYNTAX_RELOAD_TRACE").is_some() {
                        let d_old = crate::hir_dump::dump_function(old_fn, &old_module);
                        let d_new = crate::hir_dump::dump_function(new_fn, new_module);
                        eprintln!(
                            "[reload] {name} differs: fp {fp_old:x} vs {fp_new:x}, {} vs {} bytes",
                            d_old.len(),
                            d_new.len()
                        );
                        if let Some(pos) =
                            d_old.bytes().zip(d_new.bytes()).position(|(a, b)| a != b)
                        {
                            let lo = pos.saturating_sub(40);
                            eprintln!(
                                "  first diff at byte {pos}:\n  -...{:?}\n  +...{:?}",
                                &d_old[lo..(pos + 20).min(d_old.len())],
                                &d_new[lo..(pos + 20).min(d_new.len())]
                            );
                        } else {
                            eprintln!("  (byte-identical dumps?!)");
                        }
                        if d_old.lines().count() != d_new.lines().count() {
                            eprintln!(
                                "  (line counts {} vs {})",
                                d_old.lines().count(),
                                d_new.lines().count()
                            );
                        }
                    }

                    let mut body = new_fn.clone();
                    remap_body(
                        &mut body,
                        &id_remap,
                        &global_remap,
                        &effect_remap,
                        &effect_const_remap,
                    );

                    if inject_fail.as_deref() == Some(name.as_str()) {
                        compile_failed.push((name, "injected compile failure (test hook)".into()));
                        continue;
                    }

                    let bead_id = self
                        .functions
                        .get(&old_id)
                        .map(|e| e.bead_id)
                        .unwrap_or_else(osr::next_bead_id);

                    // Resume points for loops already running the old
                    // code. Compiled from the edited body before the
                    // entry recompile, so the pointer table and reload
                    // cells end up holding the probe-carrying entry.
                    let mut pending_resume: Vec<(u64, *mut ())> = Vec::new();
                    if self.config.enable_osr {
                        let helpers = self.cranelift.with_lock(|be| {
                            be.set_compile_tier(1);
                            be.set_compile_bead_id(bead_id);
                            be.compile_function_in_module(old_id, &body, &merged)?;
                            be.finalize_definitions()?;
                            Ok::<_, CompilerError>(be.take_pending_osr_helpers())
                        });
                        match helpers {
                            Ok(pairs) => pending_resume = pairs,
                            Err(e) => report
                                .resume_fell_back
                                .push((name.clone(), format!("helper compile failed: {e}"))),
                        }
                    }

                    let compiled = self.cranelift.with_lock(|be| {
                        be.set_compile_tier(0);
                        be.set_compile_bead_id(bead_id);
                        be.compile_function_in_module(old_id, &body, &merged)?;
                        be.finalize_definitions()?;
                        Ok::<_, CompilerError>(be.get_function_ptr(old_id))
                    });
                    match compiled {
                        Ok(Some(entry_ptr)) => changes.push(PreparedChange {
                            old_id,
                            name,
                            body,
                            bead_id,
                            entry_ptr: entry_ptr as usize,
                            pending_resume,
                        }),
                        Ok(None) => compile_failed
                            .push((name, "recompile produced no entry pointer".into())),
                        Err(e) => compile_failed.push((name, e.to_string())),
                    }
                }
                None => {
                    // Introduced by the edit: compile fresh under its
                    // own id, registered like `compile_module` does if
                    // the whole set applies.
                    let mut body = new_fn.clone();
                    remap_body(
                        &mut body,
                        &id_remap,
                        &global_remap,
                        &effect_remap,
                        &effect_const_remap,
                    );
                    if inject_fail.as_deref() == Some(name.as_str()) {
                        compile_failed.push((name, "injected compile failure (test hook)".into()));
                        continue;
                    }
                    let bead_id = osr::next_bead_id();
                    let compiled = self.cranelift.with_lock(|be| {
                        be.set_compile_tier(0);
                        be.set_compile_bead_id(bead_id);
                        be.compile_function_in_module(*new_id, &body, &merged)?;
                        be.finalize_definitions()?;
                        Ok::<_, CompilerError>(be.get_function_ptr(*new_id))
                    });
                    match compiled {
                        Ok(Some(p)) => adds.push(PreparedAdd {
                            new_id: *new_id,
                            name,
                            body,
                            bead_id,
                            entry_ptr: p as usize,
                        }),
                        Ok(None) => {
                            compile_failed.push((name, "compile produced no entry pointer".into()))
                        }
                        Err(e) => compile_failed.push((name, e.to_string())),
                    }
                }
            }
        }

        self.cranelift
            .with_lock(|be| be.set_defer_cell_publish(false));
        let deferred_cells = self.cranelift.with_lock(|be| be.take_deferred_cells());

        if !compile_failed.is_empty() {
            // Abort: no bead was swapped, no resume point published, no
            // dispatch slot patched, and the deferred cell updates are
            // dropped — the running generation is exactly as it was.
            // (A cell update a concurrent tier promotion deferred into
            // this window is dropped with them; its bead still holds
            // the promoted code, and the next promotion or reload
            // republishes the cell.)
            report.failed.extend(compile_failed);
            report.aborted = true;
            return Ok(report);
        }

        // Apply: everything compiled, so the swaps, resume points,
        // dispatch patches and cell publications land together.
        let relevant: std::collections::HashSet<HirId> = changes
            .iter()
            .map(|c| c.old_id)
            .chain(adds.iter().map(|a| a.new_id))
            .collect();
        let mut undo = ReloadUndo::default();
        let mut updated_functions: Vec<(HirId, HirFunction)> = Vec::new();

        for change in changes {
            let PreparedChange {
                old_id,
                name,
                body,
                bead_id,
                entry_ptr,
                pending_resume,
            } = change;
            let old_fn = &old_module.functions[&old_id];

            let mut old_entry = 0usize;
            let mut old_body: Option<Arc<HirFunction>> = None;
            if let Some(fn_entry) = self.functions.get_mut(&old_id) {
                old_entry = fn_entry
                    .bound
                    .bead()
                    .compiled()
                    .map(|p| p as usize)
                    .unwrap_or(0);
                old_body = Some(Arc::clone(&fn_entry.function));
                fn_entry.bound.bead().swap_compiled(entry_ptr as *mut ());
                fn_entry.function = Arc::new(body.clone());
            }

            // A resume point is only sound where the old code's probe
            // writes the frame the edited body's helper reads: same
            // site, same live-ins. Anything else completes on the old
            // code and picks up the edit next call.
            let mut helper_sites: Vec<(u64, u64)> = Vec::new();
            if !pending_resume.is_empty() {
                let old_sites = site_layouts(old_fn);
                let new_sites = site_layouts(&body);
                for (site, code) in pending_resume {
                    match (old_sites.get(&site), new_sites.get(&site)) {
                        (Some(old_l), Some(new_l)) if old_l == new_l => {
                            osr::publish_helper(bead_id, site, code);
                            helper_sites.push((bead_id, site));
                        }
                        (Some(_), Some(_)) => report.resume_fell_back.push((
                            name.clone(),
                            format!("site {site:#x}: live-in layout changed"),
                        )),
                        (None, _) => report.resume_fell_back.push((
                            name.clone(),
                            format!("site {site:#x}: running code has no such loop"),
                        )),
                        (_, None) => {}
                    }
                }
                if !helper_sites.is_empty() {
                    report.resume_published.push(name.clone());
                }
            }

            // Dispatch tables hold this function's old entry wherever a
            // handler exposes it as an effect op; patch those slots so
            // scopes already entered reach the edit at their next
            // perform.
            let mut vtable_slots: Vec<(usize, usize)> = Vec::new();
            for (gid, global) in &old_module.globals {
                let Some(crate::hir::HirConstant::VTable(vt)) = &global.initializer else {
                    continue;
                };
                for (slot, entry) in vt.methods.iter().enumerate() {
                    if entry.function_id != old_id {
                        continue;
                    }
                    let addr = self.cranelift.with_lock(|be| be.global_data_addr(*gid));
                    if let Some((base, size)) = addr {
                        let offset = slot * std::mem::size_of::<usize>();
                        if offset + std::mem::size_of::<usize>() <= size {
                            // SAFETY: the vtable global is declared
                            // writable and sized to its slots; running
                            // threads read the slot with plain loads,
                            // so an atomic store publishes the new
                            // entry without tearing.
                            unsafe {
                                let slot_ptr =
                                    base.add(offset) as *const std::sync::atomic::AtomicUsize;
                                let prev = (*slot_ptr)
                                    .swap(entry_ptr, std::sync::atomic::Ordering::AcqRel);
                                vtable_slots.push((slot_ptr as usize, prev));
                            }
                        }
                    }
                }
            }
            if !vtable_slots.is_empty() {
                report.dispatch_patched.push(name.clone());
            }

            undo.swapped.push(UndoSwap {
                id: old_id,
                name: name.clone(),
                old_entry,
                old_body: old_body.unwrap_or_else(|| Arc::new(old_fn.clone())),
                helper_sites,
                vtable_slots,
            });
            updated_functions.push((old_id, body));
            report.reloaded.push(name);
        }

        for add in adds {
            let PreparedAdd {
                new_id,
                name,
                body,
                bead_id,
                entry_ptr,
            } = add;
            let bound = self.adapter.register(ptr::null_mut(), None);
            bound.bead().eager_install(entry_ptr as *mut ());
            osr::register_bead(bead_id, Arc::clone(bound.bead()));
            self.functions.insert(
                new_id,
                FunctionEntry {
                    bound,
                    function: Arc::new(body.clone()),
                    module: Arc::new(merged.clone()),
                    bead_id,
                },
            );
            updated_functions.push((new_id, body));
            report.added.push(name);
        }

        // Every function is compiled, so a table the edit introduced
        // can take their addresses.
        for gid in &fresh_globals {
            let Some(crate::hir::HirConstant::VTable(vt)) = &merged.globals[gid].initializer else {
                continue;
            };
            let Some((base, size)) = self.cranelift.with_lock(|be| be.global_data_addr(*gid))
            else {
                continue;
            };
            for (slot, entry) in vt.methods.iter().enumerate() {
                let Some(ptr) = self.get_function_pointer(entry.function_id) else {
                    continue;
                };
                let offset = slot * std::mem::size_of::<usize>();
                if offset + std::mem::size_of::<usize>() > size {
                    continue;
                }
                // SAFETY: the table was declared writable and defined
                // with exactly these slots; no thread can be reading it
                // yet — the code that references it is only reachable
                // once this reload applies.
                unsafe {
                    let slot_ptr = base.add(offset) as *const std::sync::atomic::AtomicUsize;
                    (*slot_ptr).store(ptr as usize, std::sync::atomic::Ordering::Release);
                }
            }
        }

        // Publish the deferred cell updates: the final value per id,
        // restricted to the functions this reload touched (finalization
        // also re-records every already-compiled function's pointer,
        // and publishing those snapshots could roll a concurrently
        // promoted cell backwards).
        {
            use std::collections::HashMap as Map;
            let mut last: Map<HirId, usize> = Map::new();
            for (id, ptr) in deferred_cells {
                if relevant.contains(&id) {
                    last.insert(id, ptr);
                }
            }
            self.cranelift.with_lock(|be| {
                for (id, ptr) in last {
                    be.publish_call_target(id, ptr);
                }
            });
        }

        for (name, _) in old_by_name.iter() {
            if !seen_names.contains(name) {
                report.removed_retained.push(name.clone());
            }
        }

        // The next reload diffs against what is now running: the
        // merged view, including the effects, handlers and globals the
        // edit introduced.
        for (id, body) in updated_functions {
            merged.functions.insert(id, body);
        }
        let module_context = Arc::new(merged.clone());
        for entry in self.functions.values_mut() {
            entry.module = Arc::clone(&module_context);
        }
        self.current_module = Some(merged);

        // The promotion requester captured each function's body when it
        // was installed; reinstall so a later promotion compiles the
        // edited bodies rather than the ones it captured.
        self.install_promotion_requester();

        // A reload that changed nothing keeps the previous undo record:
        // "roll back the last reload" means the last one that changed
        // the running module. Removals and additions count — their undo
        // has nothing to restore at this level (retained code is
        // untouched), but the embedder's metadata rollback pairs with
        // this record and must not pair with an older one.
        if !undo.swapped.is_empty()
            || !report.removed_retained.is_empty()
            || !report.added.is_empty()
        {
            self.last_undo = Some(undo);
        }

        Ok(report)
    }

    /// Restore the generation the most recent applied reload replaced:
    /// beads and reload cells swing back to the previous entry
    /// pointers, resume points it published are withdrawn, and
    /// dispatch-table slots it patched get their prior values back.
    /// Functions the reload *added* stay registered but become
    /// unreachable as their callers roll back. One-shot: consumes the
    /// undo record; a second call errors until another reload applies.
    pub fn rollback_last_reload(&mut self) -> CompilerResult<Vec<String>> {
        let undo = self
            .last_undo
            .take()
            .ok_or_else(|| CompilerError::Backend("no applied reload to roll back".into()))?;

        let mut restored = Vec::new();
        for swap in undo.swapped.into_iter().rev() {
            if let Some(fn_entry) = self.functions.get_mut(&swap.id) {
                if swap.old_entry != 0 {
                    fn_entry
                        .bound
                        .bead()
                        .swap_compiled(swap.old_entry as *mut ());
                }
                fn_entry.function = Arc::clone(&swap.old_body);
            }
            self.cranelift
                .with_lock(|be| be.publish_call_target(swap.id, swap.old_entry));
            for (bead_id, site) in swap.helper_sites {
                osr::publish_helper(bead_id, site, ptr::null_mut());
            }
            for (addr, prev) in swap.vtable_slots {
                // SAFETY: `addr` was recorded from the same writable
                // vtable global the reload patched; the global lives
                // for the process.
                unsafe {
                    (*(addr as *const std::sync::atomic::AtomicUsize))
                        .store(prev, std::sync::atomic::Ordering::Release);
                }
            }
            if let Some(module) = &mut self.current_module {
                module.functions.insert(swap.id, (*swap.old_body).clone());
            }
            restored.push(swap.name);
        }

        if let Some(module) = &self.current_module {
            let module_context = Arc::new(module.clone());
            for entry in self.functions.values_mut() {
                entry.module = Arc::clone(&module_context);
            }
        }

        self.install_promotion_requester();
        Ok(restored)
    }

    /// Choose what a reload does with live state whose layout an edit
    /// changed. Declining is the default; migrating moves the fields
    /// the two layouts share and lets the rest start from the edited
    /// constructor's initializers.
    pub fn set_state_migration(&mut self, policy: crate::reload::StateMigration) {
        self.state_migration = policy;
    }

    /// Work out how to move a handler's live state from the running
    /// layout into the edited one. `None` when the shapes cannot be
    /// related — the caller then declines as it would without a policy.
    fn plan_state_migration(
        &self,
        handler: &str,
        ctor_name: &str,
        old_ctor: &HirFunction,
        new_ctor: &HirFunction,
        old_module: &HirModule,
        new_module: &HirModule,
    ) -> Option<crate::reload::StateMigrationPlan> {
        let by_name = |m: &HirModule, want: &str| -> Option<crate::hir::HirEffectHandler> {
            m.handlers
                .values()
                .find(|h| h.name.resolve_global().as_deref() == Some(want))
                .cloned()
        };
        let old_h = by_name(old_module, handler)?;
        let new_h = by_name(new_module, handler)?;

        let old_ty = old_ctor.signature.returns.first()?;
        let new_ty = new_ctor.signature.returns.first()?;
        let (old_ext, new_ext) = self.cranelift.with_lock(|be| {
            (
                be.struct_field_extents(old_ty),
                be.struct_field_extents(new_ty),
            )
        });
        let (old_ext, new_ext) = (old_ext?, new_ext?);
        if old_ext.len() != old_h.state_fields.len() || new_ext.len() != new_h.state_fields.len() {
            return None;
        }

        let mut moves = Vec::new();
        let mut introduced = Vec::new();
        for (i, f) in new_h.state_fields.iter().enumerate() {
            let name = f.name.resolve_global()?;
            match old_h
                .state_fields
                .iter()
                .position(|o| o.name == f.name && o.ty == f.ty)
            {
                // Same name and same type: the value carries over.
                Some(j) => moves.push((old_ext[j].0, new_ext[i].0, new_ext[i].1.min(old_ext[j].1))),
                None => introduced.push(name),
            }
        }
        let dropped = old_h
            .state_fields
            .iter()
            .filter(|o| {
                !new_h
                    .state_fields
                    .iter()
                    .any(|n| n.name == o.name && n.ty == o.ty)
            })
            .filter_map(|o| o.name.resolve_global())
            .collect();

        Some(crate::reload::StateMigrationPlan {
            handler: handler.to_string(),
            effect_id: old_h.effect_id.as_u32() as u64,
            ctor: ctor_name.to_string(),
            moves,
            introduced,
            dropped,
        })
    }

    /// What a host-side handler push needs for the handler named
    /// `handler` in the running module: `(resolved name, effect id,
    /// op-table data address, async mask, stateful?)`. Mirrors the
    /// arguments the `with H { }` lowering computes for
    /// `__zyntax_effect_push_handler`. FQN-aware: an exact name wins,
    /// and an unqualified name matches a single `path::name` handler.
    pub fn handler_push_info(&self, handler: &str) -> Option<(String, u64, usize, u64, bool)> {
        self.try_handler_push_info(handler).ok()
    }

    /// [`Self::handler_push_info`] with the reason it failed. Five very
    /// different failures used to collapse into one `None`, and the
    /// caller reported all of them as ambiguity, which points at the
    /// wrong line for four of them.
    /// The effects `function` declares whose handlers carry state, as
    /// `(effect_id, effect_name)`.
    ///
    /// A perform resolves its handler op statically when nothing is in
    /// scope, and for a handler that keeps state that op reads an
    /// implicit `self` the handler stack has no frame to supply. A
    /// caller that can see this list ahead of the call can refuse it
    /// instead of letting it reach compiled code.
    ///
    /// Only what `function` itself declares counts. A function's
    /// declared effects are its contract with its caller; a callee's
    /// are the callee's business, and a caller routinely establishes a
    /// handler for them itself before calling. Walking into callees
    /// would refuse a function whose body opens a handler scope around
    /// the performs it makes, which is the ordinary way to use one.
    ///
    /// Empty for a function that declares no effects, or whose effects
    /// are all handled by handlers that keep no state, both of which
    /// are fine to call with nothing in scope.
    pub fn stateful_effects_of(&self, function: &str) -> Vec<(u64, String)> {
        let Some(module) = self.current_module.as_ref() else {
            return Vec::new();
        };
        let Some(func) = module
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some(function))
        else {
            return Vec::new();
        };

        let mut out: Vec<(u64, String)> = Vec::new();
        for effect_name in &func.signature.effects {
            let Some(name) = effect_name.resolve_global() else {
                continue;
            };
            let Some((eid, _)) = module
                .effects
                .iter()
                .find(|(_, e)| e.name.resolve_global().as_deref() == Some(name.as_str()))
            else {
                continue;
            };
            // Whether state is carried is a property of the effect, not
            // of one handler: every handler of such an effect takes the
            // leading state slot, so any of them answers.
            let stateful = module
                .handlers
                .values()
                .any(|h| h.effect_id == *eid && !h.state_fields.is_empty());
            let id = eid.as_u32() as u64;
            if stateful && !out.iter().any(|(existing, _)| *existing == id) {
                out.push((id, name));
            }
        }
        out
    }

    pub fn try_handler_push_info(
        &self,
        handler: &str,
    ) -> Result<(String, u64, usize, u64, bool), String> {
        let module = self
            .current_module
            .as_ref()
            .ok_or_else(|| "no module has been compiled yet".to_string())?;
        let suffix = format!("::{handler}");
        let mut matched: Option<(&crate::hir::HirEffectHandler, String)> = None;
        for h in module.handlers.values() {
            let Some(name) = h.name.resolve_global() else {
                continue;
            };
            if name == handler {
                matched = Some((h, name));
                break;
            }
            if name.ends_with(&suffix) {
                if let Some((_, first)) = &matched {
                    return Err(format!(
                        "`{handler}` is ambiguous: `{first}` and `{name}` both match; qualify it"
                    ));
                }
                matched = Some((h, name));
            }
        }
        let (h, resolved) =
            matched.ok_or_else(|| format!("no handler named `{handler}` in the loaded program"))?;
        let effect = module.effects.get(&h.effect_id).ok_or_else(|| {
            format!(
                "handler `{resolved}` names effect {:?}, which is not in the module",
                h.effect_id
            )
        })?;
        let mut async_mask = 0u64;
        for (idx, op) in effect.operations.iter().enumerate().take(64) {
            if h.implementations
                .iter()
                .any(|i| i.op_name == op.name && i.is_async)
            {
                async_mask |= 1u64 << idx;
            }
        }
        let table_name = format!("$optable${resolved}");
        let gid = module
            .globals
            .iter()
            .find(|(_, g)| g.name.resolve_global().as_deref() == Some(table_name.as_str()))
            .map(|(id, _)| *id)
            .ok_or_else(|| format!("handler `{resolved}` has no `{table_name}` global"))?;
        let (addr, _size) = self
            .cranelift
            .with_lock(|be| be.global_data_addr(gid))
            .ok_or_else(|| {
                format!(
                    "`{table_name}` ({gid:?}) is in the module but was never declared to the \
                     backend, so it has no address. A JIT rebuild clears the backend's global \
                     map; anything installed before it needs recompiling."
                )
            })?;
        Ok((
            resolved,
            h.effect_id.as_u32() as u64,
            addr as usize,
            async_mask,
            !h.state_fields.is_empty(),
        ))
    }

    /// Current native-code pointer for `func_id`, or `None` if unknown.
    pub fn get_function_pointer(&self, func_id: HirId) -> Option<*const u8> {
        self.functions
            .get(&func_id)
            .and_then(|e| e.bound.bead().compiled())
            .map(|p| p as *const u8)
    }

    /// Record an invocation. Drives tier promotion via beadie.
    pub fn record_call(&self, func_id: HirId) {
        // Sample at the configured rate for cheap profile stats. Beadie
        // counts independently.
        let count = self.profile_data.get_function_count(func_id);
        if self
            .config
            .profile_config
            .sample_rate
            .checked_mul(1)
            .map(|r| count % r != 0)
            .unwrap_or(false)
        {
            // sample_rate = 0 would div-by-zero; treat that as "never sample
            // beyond the first" by skipping. We still drive beadie below.
        } else {
            self.profile_data.record_function_call(func_id);
        }

        let entry = match self.functions.get(&func_id) {
            Some(e) => e,
            None => return,
        };

        // Build a closure beadie can call from any tier broker thread.
        let func_arc = Arc::clone(&entry.function);
        let module_arc = Arc::clone(&entry.module);
        let bead_id = entry.bead_id;
        let cranelift = Arc::clone(&self.cranelift);
        #[cfg(feature = "llvm-backend")]
        let llvm = self.llvm.as_ref().map(Arc::clone);
        let tier2_backend = self.config.tier2_backend;
        let verbosity = self.config.verbosity;

        let closure = move |tier_idx: usize, bead: &Arc<Bead>| -> *mut () {
            compile_at_tier(
                tier_idx,
                bead,
                func_id,
                bead_id,
                &func_arc,
                &module_arc,
                &cranelift,
                #[cfg(feature = "llvm-backend")]
                llvm.as_ref(),
                tier2_backend,
                verbosity,
            )
        };

        // We only care about side-effects (queueing a promotion); the return
        // value of `on_invoke` is the current code pointer, which we already
        // exposed via `get_function_pointer`.
        let _ = self.adapter.on_invoke(&entry.bound, closure);
    }

    /// Force-recompile `func_id` at `target_tier`, bypassing thresholds.
    /// Install the handler for a tier-0 function asking for the top tier
    /// because it holds a resumable loop.
    ///
    /// It jumps straight there rather than one tier per call: a frame that
    /// is still running cannot supply the extra invocations the ladder
    /// needs, and the intermediate tier produces the same code as the one
    /// it is already in. Called once every function is registered, since
    /// the handler captures their beads.
    fn install_promotion_requester(&self) {
        // Aim at whichever tier actually differs from the one the frame is
        // already in. Without LLVM the ladder emits the same code at every
        // tier, so there is nothing above Standard worth reaching.
        #[cfg(feature = "llvm-backend")]
        let target = match self.config.tier2_backend {
            Tier2Backend::LLVM => OptimizationTier::Optimized,
            Tier2Backend::Cranelift => OptimizationTier::Standard,
        };
        #[cfg(not(feature = "llvm-backend"))]
        let target = OptimizationTier::Standard;
        let tier_idx = target.index();
        let tier2_backend = self.config.tier2_backend;
        let verbosity = self.config.verbosity;
        let adapter = Arc::clone(&self.adapter);
        let cranelift = Arc::clone(&self.cranelift);
        #[cfg(feature = "llvm-backend")]
        let llvm = self.llvm.as_ref().map(Arc::clone);

        // bead id -> everything a compile needs, so the handler can run on
        // the thread that raised the request without reaching for `self`.
        let by_bead: HashMap<u64, (HirId, TieredBound, Arc<HirFunction>, Arc<HirModule>)> = self
            .functions
            .iter()
            .map(|(id, e)| {
                (
                    e.bead_id,
                    (
                        *id,
                        e.bound.clone(),
                        Arc::clone(&e.function),
                        Arc::clone(&e.module),
                    ),
                )
            })
            .collect();

        osr::set_promotion_requester(move |bead_id| {
            let Some((func_id, bound, func_arc, module_arc)) = by_bead.get(&bead_id) else {
                if osr::osr_trace_enabled() {
                    eprintln!("[osr] request for unknown bead={bead_id}");
                }
                return;
            };
            let func_arc = Arc::clone(func_arc);
            let module_arc = Arc::clone(module_arc);
            let cranelift = Arc::clone(&cranelift);
            #[cfg(feature = "llvm-backend")]
            let llvm = llvm.clone();
            let func_id = *func_id;
            // The compile itself runs on a broker thread, so raising the
            // request costs the running loop only the submission.
            let submitted = adapter.force_promote(bound, tier_idx, move |bead| {
                compile_at_tier(
                    tier_idx,
                    bead,
                    func_id,
                    bead_id,
                    &func_arc,
                    &module_arc,
                    &cranelift,
                    #[cfg(feature = "llvm-backend")]
                    llvm.as_ref(),
                    tier2_backend,
                    verbosity,
                )
            });
            if osr::osr_trace_enabled() {
                eprintln!("[osr] force_promote(bead={bead_id}, tier={tier_idx}) -> {submitted}");
            }
        });
    }

    pub fn optimize_function(
        &mut self,
        func_id: HirId,
        target_tier: OptimizationTier,
    ) -> CompilerResult<()> {
        let entry = self
            .functions
            .get(&func_id)
            .ok_or_else(|| CompilerError::Backend(format!("Function {:?} not found", func_id)))?;

        let func_arc = Arc::clone(&entry.function);
        let module_arc = Arc::clone(&entry.module);
        let bead_id = entry.bead_id;
        let cranelift = Arc::clone(&self.cranelift);
        #[cfg(feature = "llvm-backend")]
        let llvm = self.llvm.as_ref().map(Arc::clone);
        let tier2_backend = self.config.tier2_backend;
        let verbosity = self.config.verbosity;
        let tier_idx = target_tier.index();

        let promoted = self
            .adapter
            .force_promote(&entry.bound, tier_idx, move |bead| {
                compile_at_tier(
                    tier_idx,
                    bead,
                    func_id,
                    bead_id,
                    &func_arc,
                    &module_arc,
                    &cranelift,
                    #[cfg(feature = "llvm-backend")]
                    llvm.as_ref(),
                    tier2_backend,
                    verbosity,
                )
            });

        if !promoted && verbosity >= 1 {
            eprintln!(
                "[TieredBackend] force_promote({:?}, {:?}) rejected (already queued, blacklisted, or out of range)",
                func_id, target_tier
            );
        }

        Ok(())
    }

    /// Snapshot statistics for diagnostics.
    pub fn get_statistics(&self) -> TieredStatistics {
        let profile_stats = self.profile_data.get_statistics();
        let mut baseline_count = 0usize;
        let mut standard_count = 0usize;
        let mut optimized_count = 0usize;

        for entry in self.functions.values() {
            match entry.bound.current_tier() {
                Some(0) => baseline_count += 1,
                Some(1) => standard_count += 1,
                Some(_) => optimized_count += 1,
                None => {}
            }
        }

        TieredStatistics {
            profile_stats,
            baseline_functions: baseline_count,
            standard_functions: standard_count,
            optimized_functions: optimized_count,
            // Beadie does not surface queue depths; expose 0 instead of
            // lying or panicking. Background activity is observable via the
            // tier counts themselves.
            queued_for_optimization: 0,
            currently_optimizing: 0,
        }
    }

    /// Releases bead registrations on shutdown so a long-lived process
    /// reusing `TieredBackend` instances doesn't leak entries.
    pub fn shutdown(&mut self) {
        for entry in self.functions.values() {
            osr::unregister_bead(entry.bead_id);
        }
        self.functions.clear();
    }

    /// Register an FFI symbol (used when reloading modules to make ZRTL
    /// plugin pointers visible to fresh Cranelift compiles).
    ///
    /// The symbol is recorded in two places:
    ///   1. `TieredBackend.runtime_symbols` — bookkeeping for any later
    ///      JIT-module rebuild we drive from this layer.
    ///   2. The inner Cranelift backend's runtime-symbol list — so the
    ///      next `rebuild_with_accumulated_symbols` re-attaches it to
    ///      the live JIT module's symbol table.
    ///
    /// Note: this method does **not** rebuild the JIT module on its own.
    /// Plugin loaders should call [`Self::rebuild_with_accumulated_symbols`]
    /// once after batching all symbol registrations for a plugin (or
    /// directory of plugins) to push them into the live JIT module.
    pub fn register_runtime_symbol(&mut self, name: &str, ptr: *const u8) {
        self.runtime_symbols.write().unwrap().push(RuntimeSymbol {
            name: name.to_string(),
            ptr: ptr as usize,
        });
        self.cranelift
            .with_lock(|be| be.register_runtime_symbol(name, ptr));
        // The LLVM tier binds externals with `add_global_mapping`, and a
        // declaration it has no mapping for resolves to null — which a
        // promoted function then calls. Every symbol the ground tier can
        // reach must be visible to the tiers above it.
        #[cfg(feature = "llvm-backend")]
        if let Some(llvm) = &self.llvm {
            llvm.with_lock(|be| be.register_symbol(name, ptr));
        }
    }

    /// Rebuild the inner Cranelift JIT module with all accumulated runtime
    /// symbols. Call after a plugin (or batch of plugins) has had its
    /// symbols registered via [`Self::register_runtime_symbol`] so the
    /// next `compile_module` can resolve them at finalization.
    ///
    /// Safe to call before any function has been compiled at tier 0
    /// (the typical "load plugins, then compile module" flow). Calling
    /// it after tier-0 compiles would invalidate previously-issued code
    /// pointers — beads would still hold them, and `swap_compiled` from
    /// later tiers would fail. The current ZynML driver only loads
    /// plugins at startup, so the unsafe ordering doesn't arise.
    pub fn rebuild_with_accumulated_symbols(&mut self) -> CompilerResult<()> {
        self.cranelift
            .with_lock(|be| be.rebuild_with_accumulated_symbols())?;
        // The rebuild discarded the JIT module the installed code was
        // declared into, taking the backend's function and global maps
        // with it. Re-declare what is still supposed to be live, or the
        // next lookup of an already-installed global finds nothing.
        if let Some(previous) = self.current_module.clone() {
            self.cranelift
                .with_lock(|be| be.compile_module(&previous))?;
        }
        Ok(())
    }

    /// Forward plugin symbol signatures to the inner Cranelift backend.
    /// Required for auto-boxing: without these, the backend doesn't
    /// know plugin functions like `$IO$println_dynamic` expect a
    /// `DynamicBox` and emits raw-i64 calls that the callee mis-reads
    /// as fat-pointer bytes.
    pub fn register_symbol_signatures(&mut self, symbols: &[crate::zrtl::RuntimeSymbolInfo]) {
        self.cranelift
            .with_lock(|be| be.register_symbol_signatures(symbols));
        #[cfg(feature = "llvm-backend")]
        if let Some(llvm) = &self.llvm {
            llvm.with_lock(|be| be.register_symbol_signatures(symbols));
        }
    }

    /// Toggle emission of OSR back-edge probes in tier-0 code.
    ///
    /// Enable / disable OSR back-edge probes on the wrapped Cranelift
    /// backend. Each probe site loads the bead's arm byte and only calls
    /// into the runtime once a tier ≥ 1 compile has installed helpers.
    pub fn set_emit_osr_probes(&mut self, enabled: bool) {
        self.cranelift
            .with_lock(|be| be.set_emit_osr_probes(enabled));
    }
}

impl Drop for TieredBackend {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build the per-tier hotness policies from a `TieredConfig`.
///
/// Public so `zyntax_embed`'s interpreter-backed runtime can build the
/// same `TieredAdapter` policy stack used by the native `TieredBackend`.
pub fn make_policies(config: &TieredConfig) -> Vec<Box<dyn HotnessPolicy>> {
    let warm = clamp_to_u32(config.profile_config.warm_threshold);
    let hot = clamp_to_u32(config.profile_config.hot_threshold);

    // Tier 0 is always eager-installed by `compile_module`, so its policy
    // never fires. Use a tiny threshold so any code path that registers a
    // bead without eager-installing still gets a baseline compile quickly.
    let tier0 = ThresholdPolicy::new(1);

    // Tier 1 (Standard) — promote at warm threshold.
    let queue_ahead_1 = (warm / 5).max(1);
    let tier1 = ThresholdPolicy::new(warm).queue_ahead(queue_ahead_1);

    // Tier 2 (Optimized) — promote at hot threshold.
    let queue_ahead_2 = (hot / 10).max(10);
    let tier2 = ThresholdPolicy::new(hot).queue_ahead(queue_ahead_2);

    if config.enable_background_optimization {
        vec![Box::new(tier0), Box::new(tier1), Box::new(tier2)]
    } else {
        // Disable promotion by setting tier 1/2 thresholds out of reach.
        let unreachable = ThresholdPolicy::new(u32::MAX);
        vec![
            Box::new(tier0),
            Box::new(ThresholdPolicy::new(u32::MAX)),
            Box::new(unreachable),
        ]
    }
}

fn clamp_to_u32(v: u64) -> u32 {
    if v > u32::MAX as u64 {
        u32::MAX
    } else {
        v as u32
    }
}

/// Dispatch the correct JIT backend for a tier index.
///
/// - `tier_idx == 0` / `tier_idx == 1` → Cranelift (baseline / opt).
/// - `tier_idx == 2` → Cranelift or LLVM, based on `tier2_backend`.
///
/// Public so `zyntax_embed::InterpRuntime` can reuse the same per-tier
/// dispatch as the native `TieredBackend`. Returns `*mut ()` (the
/// compiled fn ptr) or `ptr::null_mut()` on failure.
#[allow(clippy::too_many_arguments)]
pub fn compile_at_tier(
    tier_idx: usize,
    bead: &Arc<Bead>,
    func_id: HirId,
    bead_id: u64,
    func_arc: &Arc<HirFunction>,
    module_arc: &Arc<HirModule>,
    cranelift: &Arc<ZyntaxCraneliftBackend>,
    #[cfg(feature = "llvm-backend")] llvm: Option<&Arc<ZyntaxLlvmBackend>>,
    tier2_backend: Tier2Backend,
    verbosity: u8,
) -> *mut () {
    let def = ZyntaxFunctionDef {
        id: func_id,
        function: (**func_arc).clone(),
        module: Arc::clone(module_arc),
        tier: tier_idx,
        bead_id,
    };

    if verbosity >= 1 || crate::osr::osr_trace_enabled() {
        eprintln!(
            "[TieredBackend] Recompiling {:?} at tier {} ({:?})",
            func_id,
            tier_idx,
            OptimizationTier::from_index(tier_idx)
        );
    }

    #[cfg(feature = "llvm-backend")]
    if tier_idx == 2 && matches!(tier2_backend, Tier2Backend::LLVM) {
        if let Some(llvm) = llvm {
            return match llvm.compile(bead, def) {
                Ok(p) => p,
                Err(e) => {
                    // A failure here means the function silently stays in
                    // the tier below, which looks identical from the outside
                    // to a tier that simply never fired. Say so.
                    log::warn!("[TieredBackend] LLVM compile failed: {e}");
                    if verbosity >= 1 || crate::osr::osr_trace_enabled() {
                        eprintln!("[TieredBackend] LLVM compile failed: {e}");
                    }
                    ptr::null_mut()
                }
            };
        }
    }

    // Cranelift handles tier 0/1, and tier 2 when the config doesn't pick
    // LLVM (or the LLVM feature is off).
    let _ = tier2_backend; // silence unused-variable when llvm-backend is off
    match cranelift.compile(bead, def) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("[TieredBackend] Cranelift compile failed: {e}");
            if verbosity >= 1 || crate::osr::osr_trace_enabled() {
                eprintln!("[TieredBackend] Cranelift compile failed: {e}");
            }
            ptr::null_mut()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TieredStatistics {
    pub profile_stats: crate::profiling::ProfileStatistics,
    pub baseline_functions: usize,
    pub standard_functions: usize,
    pub optimized_functions: usize,
    pub queued_for_optimization: usize,
    pub currently_optimizing: usize,
}

impl TieredStatistics {
    pub fn format(&self) -> String {
        format!(
            "Tiered Compilation: {} Baseline (T0), {} Standard (T1), {} Optimized (T2)\n\
             Queue: {} waiting, {} optimizing\n\
             {}",
            self.baseline_functions,
            self.standard_functions,
            self.optimized_functions,
            self.queued_for_optimization,
            self.currently_optimizing,
            self.profile_stats.format()
        )
    }
}

/// Per-site live-in layout of a function's loops: site key to
/// `(phi_count, live-in types)`. Two functions agree at a site exactly
/// when the frame one's probe writes is the frame the other's helper
/// reads.
fn site_layouts(
    func: &HirFunction,
) -> std::collections::HashMap<u64, (usize, Vec<crate::hir::HirType>)> {
    let mut map = std::collections::HashMap::new();
    for header in osr::find_loop_headers(func) {
        if let Ok(layout) = osr::osr_layout(func, header) {
            map.insert(
                layout.site_key(),
                (layout.phi_count, layout.live_in_types.clone()),
            );
        }
    }
    map
}

/// Canonical layout of a type: the structural shape two generations
/// must share to read each other's state. Struct and union names are
/// ignored — only field order, types and packing bear on layout.
fn type_layout_key(ty: &crate::hir::HirType) -> String {
    fn go(ty: &crate::hir::HirType, out: &mut String, depth: usize) {
        use crate::hir::HirType as T;
        if depth > 16 {
            out.push('…');
            return;
        }
        match ty {
            T::Ptr(inner) => {
                out.push('*');
                go(inner, out, depth + 1);
            }
            T::Ref { pointee, .. } => {
                out.push('&');
                go(pointee, out, depth + 1);
            }
            T::Array(inner, n) => {
                out.push_str(&format!("[{n}]"));
                go(inner, out, depth + 1);
            }
            T::Vector(inner, n) => {
                out.push_str(&format!("<{n}>"));
                go(inner, out, depth + 1);
            }
            T::Struct(s) => {
                out.push_str(if s.packed { "s!{" } else { "s{" });
                for f in &s.fields {
                    go(f, out, depth + 1);
                    out.push(',');
                }
                out.push('}');
            }
            T::Union(u) => {
                out.push_str("u{");
                go(&u.discriminant_type, out, depth + 1);
                out.push(';');
                for v in &u.variants {
                    go(&v.ty, out, depth + 1);
                    out.push(',');
                }
                out.push('}');
            }
            other => out.push_str(&format!("{other:?}")),
        }
    }
    let mut out = String::new();
    go(ty, &mut out, 0);
    out
}

/// Rewrite the callee ids an edited function carries onto the running
/// module's ids, matched by name beforehand. Only id-carrying callables
/// change; symbol and intrinsic calls are name-based already.
fn remap_body(
    func: &mut HirFunction,
    id_remap: &std::collections::HashMap<HirId, HirId>,
    global_remap: &std::collections::HashMap<HirId, HirId>,
    effect_remap: &std::collections::HashMap<HirId, HirId>,
    effect_const_remap: &std::collections::HashMap<i64, i64>,
) {
    use crate::hir::{HirConstant, HirValueKind};

    // A value naming a global names the running program's global.
    for value in func.values.values_mut() {
        if let HirValueKind::Global(gid) = &mut value.kind {
            if let Some(mapped) = global_remap.get(gid) {
                *gid = *mapped;
            }
        }
    }

    let mut push_effect_args: Vec<HirId> = Vec::new();
    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            match inst {
                crate::hir::HirInstruction::Call { callee, args, .. } => match callee {
                    crate::hir::HirCallable::Function(id)
                    | crate::hir::HirCallable::FuncRef(id) => {
                        if let Some(mapped) = id_remap.get(id) {
                            *id = *mapped;
                        }
                    }
                    crate::hir::HirCallable::Symbol(name) => {
                        // The effect id a `with` scope pushes under is
                        // a plain constant by the time it reaches here;
                        // only its position in this call names it as an
                        // effect.
                        if name == "__zyntax_effect_push_handler" {
                            if let Some(arg) = args.first() {
                                push_effect_args.push(*arg);
                            }
                        }
                    }
                    _ => {}
                },
                crate::hir::HirInstruction::CreateClosure { function, .. } => {
                    if let Some(mapped) = id_remap.get(function) {
                        *function = *mapped;
                    }
                }
                crate::hir::HirInstruction::PerformEffect { effect_id, .. } => {
                    if let Some(mapped) = effect_remap.get(effect_id) {
                        *effect_id = *mapped;
                    }
                }
                _ => {}
            }
        }
    }

    for arg in push_effect_args {
        if let Some(value) = func.values.get_mut(&arg) {
            if let HirValueKind::Constant(HirConstant::I64(n)) = &mut value.kind {
                if let Some(mapped) = effect_const_remap.get(n) {
                    *n = *mapped;
                }
            }
        }
    }
}
