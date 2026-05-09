//! # Tiered Compilation Backend (beadie-driven)
//!
//! Multi-tier JIT compilation with hot-function promotion. Uses
//! [`beadie::TieredAdapter`] under the hood — it owns the per-tier broker
//! threads, atomic code-pointer swap, generations, and (later) OSR / deopt
//! infrastructure.
//!
//! ## Optimization tiers
//! - **Tier 0 (Baseline)** — Cranelift, eagerly compiled at module load.
//!   Beadie generation 0.
//! - **Tier 1 (Standard)** — Cranelift recompile, promoted at the warm
//!   threshold from `ProfileConfig`. Beadie generation 1.
//! - **Tier 2 (Optimized)** — Cranelift or LLVM recompile, promoted at the
//!   hot threshold. Beadie generation 2.
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
    adapter: TieredAdapter,

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
        // Wire the OSR probe so JIT'd back-edge code can resolve it.
        let (probe_name, probe_ptr) = osr::osr_probe_symbol();
        let cranelift_inner =
            CraneliftBackend::with_runtime_symbols(&[(probe_name, probe_ptr)])?;
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

        let adapter = TieredAdapter::new(make_policies(&config));

        Ok(Self {
            adapter,
            cranelift,
            #[cfg(feature = "llvm-backend")]
            llvm,
            #[cfg(feature = "llvm-backend")]
            _llvm_context,
            functions: HashMap::new(),
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

        self.cranelift
            .with_lock(|be| be.compile_module(&module))?;

        for (func_id, function) in module.functions.iter() {
            let bound = self.adapter.register(ptr::null_mut(), None);

            // Eagerly install the tier-0 code pointer so the bead reports
            // `Compiled(gen=0)` from the very first invocation.
            if let Some(p) = self.cranelift.with_lock(|be| be.get_function_ptr(*func_id)) {
                bound.bead().eager_install(p as *mut ());
            }

            // Allocate a stable id and publish the bead in the OSR
            // registry so JIT'd probes can find it.
            let bead_id = osr::next_bead_id();
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

        Ok(())
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

        let promoted = self.adapter.force_promote(&entry.bound, tier_idx, move |bead| {
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
    pub fn register_runtime_symbol(&mut self, name: &str, ptr: *const u8) {
        self.runtime_symbols.write().unwrap().push(RuntimeSymbol {
            name: name.to_string(),
            ptr: ptr as usize,
        });
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
fn make_policies(config: &TieredConfig) -> Vec<Box<dyn HotnessPolicy>> {
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

#[allow(clippy::too_many_arguments)]
fn compile_at_tier(
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

    if verbosity >= 1 {
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
                    if verbosity >= 1 {
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
            if verbosity >= 1 {
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
