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
    /// OSR registry id for this function. Embedded as a constant in
    /// tier-0 probe call sites so JIT'd code can find the bead.
    bead_id: u64,
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
            cranelift.with_lock(|be| be.set_reloadable_calls(true));
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
        self.cranelift.with_lock(|be| {
            if be.needs_rebuild_for_module(&module) {
                be.rebuild_with_accumulated_symbols()
            } else {
                Ok(())
            }
        })?;

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

        for (func_id, function) in module.functions.iter() {
            let bound = self.adapter.register(ptr::null_mut(), None);

            // Eagerly install the tier-0 code pointer so the bead reports
            // `Compiled(gen=0)` from the very first invocation.
            if let Some(p) = self.cranelift.with_lock(|be| be.get_function_ptr(*func_id)) {
                bound.bead().eager_install(p as *mut ());
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

        let mut report = crate::reload::ReloadReport::default();
        let mut updated_functions: Vec<(HirId, HirFunction)> = Vec::new();
        let mut seen_names: std::collections::HashSet<String> = Default::default();

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
                    remap_callees(&mut body, &id_remap);

                    let bead_id = self
                        .functions
                        .get(&old_id)
                        .map(|e| e.bead_id)
                        .unwrap_or_else(osr::next_bead_id);
                    let compiled = self.cranelift.with_lock(|be| {
                        be.set_compile_tier(0);
                        be.set_compile_bead_id(bead_id);
                        be.compile_function(old_id, &body)?;
                        be.finalize_definitions()?;
                        Ok::<_, CompilerError>(be.get_function_ptr(old_id))
                    });
                    match compiled {
                        Ok(Some(entry_ptr)) => {
                            if let Some(fn_entry) = self.functions.get_mut(&old_id) {
                                fn_entry.bound.bead().swap_compiled(entry_ptr as *mut ());
                                fn_entry.function = Arc::new(body.clone());
                            }
                            updated_functions.push((old_id, body));
                            report.reloaded.push(name);
                        }
                        Ok(None) => report
                            .failed
                            .push((name, "recompile produced no entry pointer".into())),
                        Err(e) => report.failed.push((name, e.to_string())),
                    }
                }
                None => {
                    // Introduced by the edit: compile fresh under its
                    // own id and register it like `compile_module` does.
                    let mut body = new_fn.clone();
                    remap_callees(&mut body, &id_remap);
                    let bead_id = osr::next_bead_id();
                    let compiled = self.cranelift.with_lock(|be| {
                        be.set_compile_tier(0);
                        be.set_compile_bead_id(bead_id);
                        be.compile_function(*new_id, &body)?;
                        be.finalize_definitions()?;
                        Ok::<_, CompilerError>(be.get_function_ptr(*new_id))
                    });
                    match compiled {
                        Ok(Some(p)) => {
                            let bound = self.adapter.register(ptr::null_mut(), None);
                            bound.bead().eager_install(p as *mut ());
                            osr::register_bead(bead_id, Arc::clone(bound.bead()));
                            self.functions.insert(
                                *new_id,
                                FunctionEntry {
                                    bound,
                                    function: Arc::new(body.clone()),
                                    bead_id,
                                },
                            );
                            updated_functions.push((*new_id, body));
                            report.added.push(name);
                        }
                        Ok(None) => report
                            .failed
                            .push((name, "compile produced no entry pointer".into())),
                        Err(e) => report.failed.push((name, e.to_string())),
                    }
                }
            }
        }

        for (name, _) in old_by_name.iter() {
            if !seen_names.contains(name) {
                report.removed_retained.push(name.clone());
            }
        }

        // The next reload diffs against what is now running.
        if let Some(module) = &mut self.current_module {
            for (id, body) in updated_functions {
                module.functions.insert(id, body);
            }
        }

        Ok(report)
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
        let by_bead: HashMap<u64, (HirId, TieredBound, Arc<HirFunction>)> = self
            .functions
            .iter()
            .map(|(id, e)| (e.bead_id, (*id, e.bound.clone(), Arc::clone(&e.function))))
            .collect();

        osr::set_promotion_requester(move |bead_id| {
            let Some((func_id, bound, func_arc)) = by_bead.get(&bead_id) else {
                if osr::osr_trace_enabled() {
                    eprintln!("[osr] request for unknown bead={bead_id}");
                }
                return;
            };
            let func_arc = Arc::clone(func_arc);
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
            .with_lock(|be| be.rebuild_with_accumulated_symbols())
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
    cranelift: &Arc<ZyntaxCraneliftBackend>,
    #[cfg(feature = "llvm-backend")] llvm: Option<&Arc<ZyntaxLlvmBackend>>,
    tier2_backend: Tier2Backend,
    verbosity: u8,
) -> *mut () {
    let def = ZyntaxFunctionDef {
        id: func_id,
        function: (**func_arc).clone(),
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

/// Rewrite the callee ids an edited function carries onto the running
/// module's ids, matched by name beforehand. Only id-carrying callables
/// change; symbol and intrinsic calls are name-based already.
fn remap_callees(func: &mut HirFunction, id_remap: &std::collections::HashMap<HirId, HirId>) {
    for block in func.blocks.values_mut() {
        for inst in &mut block.instructions {
            match inst {
                crate::hir::HirInstruction::Call { callee, .. } => match callee {
                    crate::hir::HirCallable::Function(id)
                    | crate::hir::HirCallable::FuncRef(id) => {
                        if let Some(mapped) = id_remap.get(id) {
                            *id = *mapped;
                        }
                    }
                    _ => {}
                },
                crate::hir::HirInstruction::CreateClosure { function, .. } => {
                    if let Some(mapped) = id_remap.get(function) {
                        *function = *mapped;
                    }
                }
                _ => {}
            }
        }
    }
}
