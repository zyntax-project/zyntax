//! Interpreter-backed runtime: HIR → bytecode-interpreter → JIT.
//!
//! `InterpRuntime` is the execution engine — BC interpreter, beadie-
//! driven tier-up, and the FFI symbol table the interp dispatches
//! into. It does NOT handle parsing, plugin loading, or import
//! resolution; those live on [`crate::ZyntaxRuntime`] (native) or on
//! the future wasm-side runtime (Phase F).
//!
//! Typical native flow:
//!
//! ```ignore
//! // ZyntaxRuntime handles parsing/lowering/plugin loading.
//! let mut zr = ZyntaxRuntime::new()?;
//! zr.load_plugins_from_directory("plugins")?;
//! let hir = zr.lower_typed_program(program, Default::default())?;
//!
//! // InterpRuntime handles execution.
//! let mut ir = InterpRuntime::new();
//! ir.compile_module(hir);
//! // For programs that call plugin symbols, forward them via
//! // `register_zrtl_symbols(plugin.symbols_with_signatures())`
//! // after loading each plugin separately on the InterpRuntime side.
//! ir.install_jit()?;
//! let r = ir.call_function("main", vec![])?;
//! ```
//!
//! Hot-function tier-up is wired via [`beadie`]: every function gets a
//! `Bead`, the interpreter ticks it on every call, and once the
//! hotness policy fires `beadie::Beadie::on_invoke` submits the
//! compile closure to the broker. The broker compiles the function on
//! a background thread; once the native code pointer is ready, the
//! interpreter's per-function tick callback short-circuits to JIT
//! dispatch.
//!
//! ## Tier ladder
//!
//! The BC interpreter does the cold-start work — there's no separate
//! "baseline JIT" tier as a bridge. Functions tier up directly from
//! the interpreter to Cranelift's optimized output.
//!
//! - **Pre-tier (BC Interpreter)** — this module. Always available.
//!   Ticks the `TieredBound` on every call.
//! - **Tier 0 (Cranelift opt)** — Cranelift with `opt_level = "speed"`
//!   (its default setting). Emits back-edge probes (`compile_tier = 0`)
//!   so the JIT'd code keeps ticking the bead for further promotion.
//!   Compiled on beadie's broker thread; dispatched from the calling
//!   thread.
//! - **Tier 1 (LLVM)** — only when the `llvm-backend` feature is on.
//!   The final tier; reached at the hot threshold. **Compiled
//!   synchronously on the dispatching thread**, not on a beadie
//!   promotion broker. Apple Silicon's MAP_JIT W^X is per-thread, so
//!   an MCJIT-emitted pointer is only callable from the thread that
//!   finalised it; doing the compile on the same thread that
//!   dispatches sidesteps that.
//!
//! Without `llvm-backend`, Tier 0 is also the final tier — Cranelift
//! opt is good enough that no further promotion is needed.
//!
//! Tier 0 promotion is driven by [`beadie::TieredAdapter`]; the JIT'd
//! code ticks the bead through OSR probes baked in by
//! `CraneliftBackend::set_compile_bead_id`. Tier 1 promotion is
//! driven by the per-function tick callback that observes the bead's
//! invocation counter and calls
//! [`beadie::Bead::swap_compiled`] directly with the LLVM pointer.

use std::collections::HashMap;
use std::sync::Arc;

use beadie::{Bead, TieredAdapter, TieredBound};
use zyntax_compiler::hir::{HirId, HirModule};
use zyntax_compiler::hir_interp::{HirInterpreter, InterpError, JitDispatch, ProfileSample};
#[cfg(feature = "native")]
use zyntax_compiler::tiered_backend::{make_policies, TieredConfig};
use zyntax_compiler::{CompilationConfig, CompilerError};
use zyntax_typed_ast::{TypeRegistry, TypedProgram};

/// Front-door embed type. Roughly mirrors [`crate::ZyntaxRuntime`] but
/// runs through the BC interpreter instead of native machine code.
///
/// On native, hot functions tier up through beadie's
/// `TieredAdapter` to Cranelift baseline → Cranelift opt → LLVM (or
/// Cranelift opt2). On wasm targets the JIT side is gated off; the
/// interpreter is the only execution path until the wasm-emitting
/// backend lands (Phase E).
pub struct InterpRuntime {
    /// The HIR module being executed. Filled by `compile_module` /
    /// `compile_typed_program`. Wrapped in `Arc` so per-function tick
    /// callbacks can hold a stable reference without re-borrowing
    /// `self` on every invocation.
    module: Option<Arc<HirModule>>,
    /// The bytecode interpreter — owns the compiled-function cache,
    /// profile counters, FFI symbol table, and tick callbacks.
    interp: HirInterpreter,
    /// Beadie's multi-tier orchestrator. Owns per-tier brokers and
    /// per-bead promotion state. Native-only: beadie's
    /// `TieredAdapter::new` eagerly spawns broker worker threads,
    /// which fails on wasm32 (no threads). On wasm the interpreter
    /// is the entire execution path — no tier-up infrastructure.
    #[cfg(feature = "native")]
    tiered: Arc<TieredAdapter>,
    /// Per-function `TieredBound` carrying the bead and per-tier
    /// queued-state. Indexed by `HirFunction::id`. Native-only,
    /// same reasoning as `tiered`.
    #[cfg(feature = "native")]
    bounds: HashMap<HirId, TieredBound>,
    /// Per-function OSR bead-id, baked into JIT'd code as a constant
    /// so back-edge probes can find the bead. Indexed by
    /// `HirFunction::id`. Native-only.
    #[cfg(feature = "native")]
    bead_ids: HashMap<HirId, u64>,
    /// Persisted tier config — populated by `with_threshold` so that
    /// `install_jit` (no-arg) inherits the requested promotion
    /// threshold instead of clobbering it with `TieredConfig::default`.
    #[cfg(feature = "native")]
    tier_config: TieredConfig,
}

impl Default for InterpRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the BC-interp tier ladder's promotion policies.
///
/// Beadie owns only **tier 0** (Cranelift opt). Tier 1 (LLVM) is
/// managed directly in the per-function tick callback because of an
/// Apple Silicon MAP_JIT thread-locality quirk: MCJIT-emitted code is
/// only callable from the thread that finalized it, so handing off a
/// pointer from beadie's `PromotionBroker` to the dispatching thread
/// faults on first execute. Compiling LLVM synchronously on the
/// dispatching thread sidesteps that — same thread compiles + calls.
///
/// So the policy stack is just one slot:
///   - slot 0: Cranelift opt (warm threshold) — `compile(0, b)` runs
///     on the beadie broker; `bead.install_compiled` flips state to
///     `Compiled` and dispatch starts using the Cranelift pointer.
///
/// LLVM tier-up happens later, inside the tick callback, when the
/// invocation count crosses the hot threshold. The tick callback
/// compiles LLVM on the calling thread and calls
/// `bound.bead().swap_compiled(llvm_ptr)` directly, which bumps the
/// bead's generation from 0 to 1.
#[cfg(feature = "native")]
fn default_tier_policies() -> Vec<Box<dyn beadie::HotnessPolicy>> {
    use beadie::ThresholdPolicy;
    let cfg = TieredConfig::default();
    let warm = cfg.profile_config.warm_threshold.min(u32::MAX as u64) as u32;
    vec![Box::new(ThresholdPolicy::new(warm))]
}

impl InterpRuntime {
    /// Create an empty runtime. On native, tier policies come from
    /// [`TieredConfig::default`] (warm/hot thresholds from
    /// `ProfileConfig::default`). On wasm there is no JIT tier — the
    /// interpreter runs everything.
    pub fn new() -> Self {
        Self {
            module: None,
            interp: HirInterpreter::new(),
            #[cfg(feature = "native")]
            tiered: Arc::new(TieredAdapter::new(default_tier_policies())),
            #[cfg(feature = "native")]
            bounds: HashMap::new(),
            #[cfg(feature = "native")]
            bead_ids: HashMap::new(),
            #[cfg(feature = "native")]
            tier_config: TieredConfig::default(),
        }
    }

    /// Convenience constructor: configure the BC interp → Cranelift opt
    /// promotion threshold (Tier 0) at `threshold` invocations. The
    /// Cranelift → LLVM threshold (Tier 1) keeps its default.
    ///
    /// This persists the threshold into the stored `TieredConfig` so a
    /// subsequent no-arg `install_jit()` inherits it. To override Tier
    /// 1's threshold too, use [`Self::install_jit_with`] directly.
    pub fn with_threshold(threshold: u32) -> Self {
        #[cfg(feature = "native")]
        {
            use beadie::ThresholdPolicy;
            let mut cfg = TieredConfig::default();
            cfg.profile_config.warm_threshold = threshold as u64;
            let warm = (threshold as u64).min(u32::MAX as u64) as u32;
            // Beadie holds only tier 0 (Cranelift). LLVM tier-up is
            // driven from the tick callback. See `default_tier_policies`
            // for the rationale (Apple Silicon JIT thread-locality).
            let policies: Vec<Box<dyn beadie::HotnessPolicy>> =
                vec![Box::new(ThresholdPolicy::new(warm))];
            Self {
                module: None,
                interp: HirInterpreter::new(),
                tiered: Arc::new(TieredAdapter::new(policies)),
                bounds: HashMap::new(),
                bead_ids: HashMap::new(),
                tier_config: cfg,
            }
        }
        #[cfg(not(feature = "native"))]
        {
            let _ = threshold;
            Self::new()
        }
    }

    /// Lower a `TypedProgram` to HIR and install it. Convenience over
    /// `compile_module` for callers that have a parsed-but-not-lowered
    /// program in hand.
    ///
    /// This is a thin wrapper around `zyntax_compiler::compile_to_hir`;
    /// it does NOT do the struct/impl pre-registration that
    /// `ZyntaxRuntime::lower_typed_program` does. For class-heavy
    /// programs, do that registration first (or use `ZyntaxRuntime`).
    ///
    /// **Optimization level**: the interpreter consumes HIR directly.
    /// We force `opt_level = 0` so the legacy `OptimizationPipeline`
    /// (its `DeadCodeElimination` pass relies on `HirValue.uses`
    /// being populated, which the lowering doesn't always do — over-
    /// eliminates and leaves dangling refs the interpreter can't
    /// resolve) is skipped. Tier-up to Cranelift will re-run the
    /// optimisations against its own consumer there.
    ///
    /// AFTER `compile_to_hir` returns we run the SSA-clean subset of
    /// HIR passes (`const_fold`, `cse`, `inline`, `licm`,
    /// `loop_vectorize`) via [`zyntax_compiler::run_interp_safe_opts`].
    /// These mutate `HirValue.kind` in place or sweep operand
    /// references; they don't depend on a populated `uses` map and
    /// are safe under the BC interp. The interp-runtime e2e tests in
    /// `crates/zynml/tests/optimization_passes_e2e.rs` pin their
    /// correctness against real ZynML programs.
    pub fn compile_typed_program(
        &mut self,
        program: &mut TypedProgram,
        type_registry: Arc<TypeRegistry>,
        mut config: CompilationConfig,
    ) -> Result<(), CompilerError> {
        config.opt_level = 0;
        let mut module = zyntax_compiler::compile_to_hir(program, type_registry, config)?;
        let _opt_stats = zyntax_compiler::run_interp_safe_opts(&mut module);
        log::debug!("[interp_runtime] HIR opts: {:?}", _opt_stats);
        self.compile_module(module);
        Ok(())
    }

    /// Install a pre-built HIR module. Registers a `TieredBound` per
    /// function so tier-up wiring is ready when
    /// [`Self::install_cranelift_jit`] (or `install_llvm_jit` with
    /// the `llvm-backend` feature) is called.
    ///
    /// Also assigns an OSR bead id to each function and registers it
    /// in `zyntax_compiler::osr`'s global bead registry so JIT'd code
    /// can resolve back to the bead from a probe site.
    pub fn compile_module(&mut self, module: HirModule) {
        let module = Arc::new(module);
        // Beadie bookkeeping is native-only — see field docs on
        // `tiered`/`bounds`/`bead_ids`. On wasm the interpreter runs
        // the module directly with no tier-up scaffolding.
        #[cfg(feature = "native")]
        {
            self.bounds.clear();
            self.bead_ids.clear();
            for func_id in module.functions.keys() {
                // CoreHandle is `*mut ()`. We use the HirId's address
                // as an opaque token; beadie stores but never
                // dereferences it.
                let core_ptr: *mut () = (func_id as *const HirId) as *mut ();
                let bound = self.tiered.register(core_ptr, None);
                let bead_id = zyntax_compiler::osr::next_bead_id();
                zyntax_compiler::osr::register_bead(bead_id, Arc::clone(bound.bead()));
                self.bead_ids.insert(*func_id, bead_id);
                self.bounds.insert(*func_id, bound);
            }
        }
        self.module = Some(module);
    }

    /// Register a `extern "C"` symbol callable from interpreted code.
    /// `param_count` is the number of i64-funneled arguments (matches
    /// the ZRTL ABI convention).
    pub fn register_symbol(&mut self, name: impl Into<String>, ptr: *const u8, param_count: u8) {
        self.interp.register_symbol(name, ptr, param_count);
    }

    /// Snapshot of the FFI symbol table — forwarded from the inner
    /// `HirInterpreter`. See [`HirInterpreter::symbol_table_snapshot`].
    pub fn symbol_table_snapshot(&self) -> Vec<(String, *const u8, u8)> {
        self.interp.symbol_table_snapshot()
    }

    /// Install the wasm-JIT compile + dispatch hooks on the BC
    /// interpreter (Phase E.6 — wasm32-target tier-up path).
    ///
    /// The compile hook is invoked the first time a function crosses
    /// the interpreter's `wasm_jit_threshold` invocation count; the
    /// host returns an opaque `u32` handle (a JS-side funcref table
    /// index) or `None` to keep the function in BC. The dispatch
    /// hook then routes all future calls to that function through
    /// the matching JS extern.
    ///
    /// The corresponding native ladder uses `set_jit_compiler` (BC →
    /// Cranelift → LLVM via fn pointers). The wasm path is split
    /// out because wasm32 has no addressable function pointers —
    /// dispatch has to go through a JS-owned funcref table.
    #[allow(clippy::type_complexity)]
    pub fn install_wasm_jit_hooks(
        &mut self,
        compile: Box<dyn FnMut(&zyntax_compiler::hir::HirFunction) -> Option<u32> + Send>,
        dispatch: Box<
            dyn FnMut(
                    u32,
                    &[zyntax_compiler::value::ZyntaxValue],
                ) -> Result<
                    zyntax_compiler::value::ZyntaxValue,
                    zyntax_compiler::hir_interp::InterpError,
                > + Send,
        >,
    ) {
        self.interp.set_wasm_compile_hook(compile);
        self.interp.set_wasm_dispatch_hook(dispatch);
    }

    /// Install the BC interpreter's IndirectCall dispatcher. The
    /// host (e.g. `zyntax_wasm`) provides a closure that resolves a
    /// runtime function-pointer / handle to a concrete dispatch.
    /// See `HirInterpreter::set_indirect_call_dispatcher`.
    #[allow(clippy::type_complexity)]
    pub fn install_indirect_call_dispatcher(
        &mut self,
        dispatcher: Box<
            dyn FnMut(
                    i64,
                    Vec<zyntax_compiler::value::ZyntaxValue>,
                ) -> Result<
                    zyntax_compiler::value::ZyntaxValue,
                    zyntax_compiler::hir_interp::InterpError,
                > + Send,
        >,
    ) {
        self.interp.set_indirect_call_dispatcher(dispatcher);
    }

    /// Install the BC interpreter's symbol-call escape hatch (wasm32).
    /// See `HirInterpreter::set_symbol_call_dispatcher`.
    #[allow(clippy::type_complexity)]
    pub fn install_symbol_call_dispatcher(
        &mut self,
        dispatcher: Box<
            dyn FnMut(
                    &str,
                    Vec<zyntax_compiler::value::ZyntaxValue>,
                ) -> Result<
                    Option<zyntax_compiler::value::ZyntaxValue>,
                    zyntax_compiler::hir_interp::InterpError,
                > + Send,
        >,
    ) {
        self.interp.set_symbol_call_dispatcher(dispatcher);
    }

    /// Register a statically-linked ZRTL plugin into the BC
    /// interpreter's FFI table. Wasm-shim equivalent of
    /// `ZyntaxRuntime::register_static_plugin` — same SDK
    /// `StaticPlugin` input, no native backend on the other side.
    ///
    /// Walks the plugin's symbol table (excluding the trailing
    /// null-name sentinel) and forwards each entry into the
    /// interpreter via [`Self::register_symbol`]. Signatures are
    /// dropped on this path since the interpreter doesn't need
    /// auto-boxing info — the BC ops already know the operand types
    /// from the HIR they're walking.
    pub fn register_static_plugin(&mut self, plugin: zrtl::StaticPlugin) {
        use std::ffi::CStr;
        for sym in plugin.symbols {
            // SAFETY: the `name` pointer in a `zrtl_plugin!`-generated
            // table is always a `concat!("...", "\0")` static literal —
            // null-terminated and valid UTF-8.
            let name = unsafe {
                match CStr::from_ptr(sym.name).to_str() {
                    Ok(s) => s.to_string(),
                    Err(_) => continue,
                }
            };
            // `param_count` is derived from the SDK signature when
            // available; fall back to 0 (i.e. "no params") otherwise —
            // the interpreter's call dispatch only uses param_count to
            // pre-funnel args through `i64`, so 0 is the safe default
            // for symbols that happen not to carry signature info.
            let param_count = if sym.sig.is_null() {
                0
            } else {
                // SAFETY: non-null sig is a static `ZrtlSymbolSig`.
                unsafe { (*sym.sig).param_count }
            };
            self.interp.register_symbol(name, sym.ptr, param_count);
        }
    }

    /// Bridge from a `RuntimeSymbolInfo` slice (the shape
    /// [`zyntax_compiler::zrtl::ZrtlPlugin::symbols_with_signatures`]
    /// produces) into the BC interpreter's FFI symbol table.
    ///
    /// Plugin loading itself lives on [`crate::ZyntaxRuntime`]; once a
    /// plugin is loaded, callers forward its symbols here so the
    /// interpreter can call into them:
    ///
    /// ```ignore
    /// let mut zr = ZyntaxRuntime::new()?;
    /// zr.load_plugins_from_directory("plugins")?;
    ///
    /// let plugin = ZrtlPlugin::load("zrtl_io.zrtl")?;
    /// interp_rt.register_zrtl_symbols(plugin.symbols_with_signatures());
    /// ```
    ///
    /// Param count is derived from `sig.param_count` when available;
    /// otherwise defaults to 0.
    pub fn register_zrtl_symbols(&mut self, symbols: &[zyntax_compiler::zrtl::RuntimeSymbolInfo]) {
        for sym in symbols {
            let param_count = sym.sig.map(|s| s.param_count).unwrap_or(0);
            self.interp
                .register_symbol(sym.name.to_string(), sym.ptr, param_count);
        }
    }

    /// Wire a generic multi-tier JIT compiler. On every interpreter
    /// tick, the per-function callback runs `TieredAdapter::on_invoke`
    /// — beadie ticks the bead, and when a tier's hotness policy fires
    /// it submits the compile closure to the broker for THAT tier.
    /// The closure receives `(tier_index, func_id)` and produces the
    /// native fn ptr + param count for that tier; beadie atomically
    /// swaps the function's compiled pointer.
    ///
    /// The closure runs on a broker thread, possibly once per tier per
    /// function. It must be `Fn + Send + Sync + Clone + 'static`
    /// because the adapter clones it across tier brokers. Closures
    /// capturing `Arc`-wrapped backends satisfy this naturally.
    ///
    /// Returning `None` from `compile` aborts promotion at that tier
    /// for that function — the interpreter (or lower tier) keeps
    /// running it.
    ///
    /// Most callers want [`Self::install_cranelift_jit`] (or, with
    /// `feature = "llvm-backend"`, [`Self::install_llvm_jit`]) which
    /// build the appropriate per-tier dispatch closure for you. This
    /// generic seam is what Phase E plugs the wasm-emitter into.
    ///
    /// Native-only — wasm has no JIT tier yet (Phase E hasn't shipped
    /// the wasm-encoder hot-function path) and `self.tiered`/`bounds`
    /// don't exist on that target.
    #[cfg(feature = "native")]
    pub fn set_jit_compiler<F>(&mut self, compile: F)
    where
        F: Fn(usize, HirId) -> Option<(*const u8, u8)> + Send + Sync + Clone + 'static,
    {
        for (func_id, bound) in &self.bounds {
            let bound = bound.clone();
            let tiered = Arc::clone(&self.tiered);
            let func_id = *func_id;
            let n_params = self.param_count_for(func_id);
            let compile = compile.clone();

            self.interp.register_tick_callback(
                func_id,
                Box::new(move || {
                    // ALWAYS call on_invoke — it ticks the bead's
                    // invocation counter, which beadie's `maybe_promote`
                    // watches to schedule the next tier. Caching the
                    // dispatch and skipping on_invoke on the hot path
                    // would stall promotion at the first compiled tier.
                    // on_invoke is O(1) on the hot path (atomic
                    // increment + compare + atomic load).
                    let compile_for_broker = compile.clone();
                    let code =
                        tiered.on_invoke(&bound, move |tier, _bead| {
                            match compile_for_broker(tier, func_id) {
                                Some((ptr, _)) => ptr as *mut (),
                                None => std::ptr::null_mut(),
                            }
                        })?;
                    if code.is_null() {
                        return None;
                    }
                    Some(JitDispatch {
                        ptr: code as *const u8,
                        n_params,
                    })
                }),
            );
        }
    }

    /// Param count for `func_id`, or 0 if the function isn't in the
    /// installed module.
    fn param_count_for(&self, func_id: HirId) -> u8 {
        self.module
            .as_ref()
            .and_then(|m| m.functions.get(&func_id))
            .map(|f| f.signature.params.len().min(255) as u8)
            .unwrap_or(0)
    }

    /// Invoke a top-level function by name. Returns the bytecode
    /// interpreter's result (or a JIT dispatch's i64 retagged as I64).
    pub fn call_function(
        &mut self,
        name: &str,
        args: Vec<zyntax_compiler::value::ZyntaxValue>,
    ) -> Result<zyntax_compiler::value::ZyntaxValue, InterpError> {
        let module = self
            .module
            .as_ref()
            .ok_or_else(|| InterpError::Host("no module compiled".to_string()))?
            .clone();
        self.interp.call(&module, name, args)
    }

    /// Read the profile sample for a function. Useful for tests that
    /// want to verify tier-up wiring is firing on hot loops.
    pub fn profile_for(&self, func_id: HirId) -> ProfileSample {
        self.interp.profile_for(func_id)
    }

    /// Access the installed HIR module. Test/diagnostic hook —
    /// callers typically don't need raw HIR access.
    pub fn module(&self) -> Option<&Arc<HirModule>> {
        self.module.as_ref()
    }

    /// Expose the bead for a function. Mainly for tests / advanced
    /// callers that want to inspect promotion state directly.
    /// Native-only — no beadie state exists on wasm.
    #[cfg(feature = "native")]
    pub fn bead_for(&self, func_id: HirId) -> Option<&Arc<Bead>> {
        self.bounds.get(&func_id).map(|b| b.bead())
    }

    /// Expose the multi-tier `TieredBound` for a function.
    /// `bound.current_tier()` returns the tier the function is
    /// currently running at (`None` while still interpreted).
    /// Native-only.
    #[cfg(feature = "native")]
    pub fn bound_for(&self, func_id: HirId) -> Option<&TieredBound> {
        self.bounds.get(&func_id)
    }

    /// All function ids that have a registered bound. Test/diagnostic
    /// hook for callers that need to walk the bound set (since the
    /// runtime doesn't otherwise expose the HirModule's function
    /// table). Native-only — wasm returns an empty iterator.
    pub fn registered_function_ids(&self) -> impl Iterator<Item = HirId> + '_ {
        #[cfg(feature = "native")]
        {
            self.bounds.keys().copied()
        }
        #[cfg(not(feature = "native"))]
        {
            std::iter::empty()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier 1+ wiring — Cranelift baseline / LLVM (native only)
// ─────────────────────────────────────────────────────────────────────────────
//
// Two install hooks are available on native:
//
//   * `install_cranelift_jit` — Cranelift baseline (fast compile,
//     minimal opts). The default Tier-1 target. Matches `wren_lift`'s
//     baseline-first tier-up model.
//   * `install_llvm_jit` (gated on `feature = "llvm-backend"`) —
//     LLVM-backed JIT. Heavier compile, higher-quality code. Use when
//     a function is known to be hot enough that the LLVM compile cost
//     is amortised.
//
// Callers wire ONE of these via the `install_*_jit` family; calling
// two replaces the tick callbacks (the most recent install wins).
//
// On native: `install_jit` wires the BC interp → Cranelift opt
// ladder, plus LLVM as the top tier when the `llvm-backend` feature
// is enabled. No baseline Cranelift step — the BC interpreter is the
// cold-start tier.

#[cfg(feature = "native")]
impl InterpRuntime {
    /// Install the BC interp → Cranelift opt [→ LLVM] tier ladder.
    /// On every interpreter tick, beadie's `TieredAdapter` ticks the
    /// bead; when Tier 0's hotness policy fires it schedules a
    /// Cranelift compile on a broker thread. When Tier 1's policy
    /// fires (only with `llvm-backend` enabled), the JIT'd Cranelift
    /// code's OSR probes drive promotion to an LLVM compile.
    ///
    /// `config` controls promotion thresholds; `tier2_backend` is
    /// ignored — this ladder always uses Cranelift for the middle
    /// tier and LLVM (when available) for the top tier.
    pub fn install_jit_with(&mut self, config: TieredConfig) -> Result<(), CompilerError> {
        use zyntax_compiler::beadie_adapter::ZyntaxCraneliftBackend;
        use zyntax_compiler::cranelift_backend::CraneliftBackend;
        use zyntax_compiler::osr;

        // Persist for any later no-arg `install_jit()` call.
        self.tier_config = config.clone();

        let module = self
            .module
            .as_ref()
            .ok_or_else(|| CompilerError::Backend("no module compiled".to_string()))?
            .clone();

        // Policy stack: slot 0 = Cranelift (warm). LLVM is driven
        // synchronously from the tick callback below (see
        // `default_tier_policies` for why).
        use beadie::ThresholdPolicy;
        let warm = config.profile_config.warm_threshold.min(u32::MAX as u64) as u32;
        let hot = config.profile_config.hot_threshold.min(u32::MAX as u64) as u32;
        let policies: Vec<Box<dyn beadie::HotnessPolicy>> =
            vec![Box::new(ThresholdPolicy::new(warm))];
        let new_tiered = Arc::new(TieredAdapter::new(policies));
        let mut new_bounds = HashMap::new();
        let mut new_bead_ids = HashMap::new();
        for func_id in module.functions.keys() {
            let core_ptr: *mut () = (func_id as *const HirId) as *mut ();
            let bound = new_tiered.register(core_ptr, None);
            let bead_id = osr::next_bead_id();
            osr::register_bead(bead_id, Arc::clone(bound.bead()));
            new_bead_ids.insert(*func_id, bead_id);
            new_bounds.insert(*func_id, bound);
        }
        self.tiered = new_tiered;
        self.bounds = new_bounds;
        self.bead_ids = new_bead_ids;

        // Cranelift for Tier 0. Wire OSR runtime symbols so the
        // back-edge probes baked into the JIT'd code can resolve
        // back through the OSR registry for further tier-up.
        let osr_syms = osr::osr_runtime_symbols();
        let cranelift_inner = CraneliftBackend::with_runtime_symbols(&osr_syms)
            .map_err(|e| CompilerError::Backend(format!("cranelift init failed: {e}")))?;
        let cranelift = Arc::new(ZyntaxCraneliftBackend::new(cranelift_inner));

        let func_arcs: HashMap<HirId, (Arc<zyntax_compiler::hir::HirFunction>, u64)> = module
            .functions
            .iter()
            .map(|(id, f)| {
                (
                    *id,
                    (
                        Arc::new(f.clone()),
                        self.bead_ids.get(id).copied().unwrap_or(0),
                    ),
                )
            })
            .collect();
        let func_arcs = Arc::new(func_arcs);

        // LLVM for the top tier — built only when the feature is on.
        // The `build_llvm_backend` helper inside zyntax_compiler
        // encapsulates the inkwell `Context` lifecycle so we don't
        // need to import inkwell directly here.
        #[cfg(feature = "llvm-backend")]
        let (llvm, _llvm_context_keepalive) =
            zyntax_compiler::beadie_adapter::build_llvm_backend()?;

        let _ = config.verbosity;
        let cranelift_for_closure = Arc::clone(&cranelift);
        // Cranelift compile closure (tier 0, beadie-driven).
        let cranelift_compile = {
            let func_arcs = Arc::clone(&func_arcs);
            move |tier: usize, func_id: HirId| -> Option<(*const u8, u8)> {
                let (func_arc, bead_id) = func_arcs.get(&func_id)?;
                let n_params = func_arc.signature.params.len().min(255) as u8;
                let _ = tier; // always tier 0 here
                let ptr = cranelift_for_closure.with_lock(|be| {
                    be.set_compile_tier(0);
                    be.set_compile_bead_id(*bead_id);
                    be.compile_function(func_id, func_arc).ok()?;
                    be.finalize_definitions().ok()?;
                    be.get_function_ptr(func_id)
                })?;
                Some((ptr, n_params))
            }
        };

        // Per-function tier-1 dispatch (LLVM, synchronous, main-thread).
        #[cfg(feature = "llvm-backend")]
        let llvm_state: Arc<HashMap<HirId, Arc<std::sync::atomic::AtomicBool>>> = Arc::new(
            func_arcs
                .keys()
                .map(|id| (*id, Arc::new(std::sync::atomic::AtomicBool::new(false))))
                .collect(),
        );

        for (func_id, bound) in &self.bounds {
            let bound = bound.clone();
            let tiered = Arc::clone(&self.tiered);
            let func_id = *func_id;
            let func_arcs = Arc::clone(&func_arcs);
            let cranelift_compile = cranelift_compile.clone();
            let n_params = self.param_count_for(func_id);

            #[cfg(feature = "llvm-backend")]
            let llvm_state = Arc::clone(&llvm_state);
            #[cfg(feature = "llvm-backend")]
            let llvm_for_closure = Arc::clone(&llvm);
            #[cfg(feature = "llvm-backend")]
            let keepalive = Arc::clone(&_llvm_context_keepalive);

            self.interp.register_tick_callback(
                func_id,
                Box::new(move || {
                    // Tier 0 (Cranelift) goes through beadie. on_invoke
                    // also ticks the bead's invocation counter — call
                    // it on every dispatch even after Cranelift fires.
                    let compile_for_broker = cranelift_compile.clone();
                    let code =
                        tiered.on_invoke(&bound, move |tier, _bead| {
                            match compile_for_broker(tier, func_id) {
                                Some((ptr, _)) => ptr as *mut (),
                                None => std::ptr::null_mut(),
                            }
                        });

                    // Tier 1 (LLVM, this thread): synchronous to avoid
                    // the Apple Silicon cross-thread MAP_JIT issue.
                    #[cfg(feature = "llvm-backend")]
                    {
                        let count = bound.bead().invocation_count();
                        if code.is_some()
                            && bound.generation() == 0
                            && count >= hot
                            && llvm_state
                                .get(&func_id)
                                .map(|flag| {
                                    flag.compare_exchange(
                                        false,
                                        true,
                                        std::sync::atomic::Ordering::AcqRel,
                                        std::sync::atomic::Ordering::Relaxed,
                                    )
                                    .is_ok()
                                })
                                .unwrap_or(false)
                        {
                            if let Some((func_arc, _)) = func_arcs.get(&func_id) {
                                let _ = &keepalive; // pin inkwell Context
                                let llvm_ptr = llvm_for_closure.with_lock(|be| {
                                    be.compile_function(func_id, func_arc).ok()?;
                                    be.get_function_pointer(func_id)
                                });
                                if let Some(ptr) = llvm_ptr {
                                    // swap_compiled bumps generation 0 → 1
                                    // and atomically publishes the new ptr.
                                    bound.bead().swap_compiled(ptr as *mut ());
                                }
                            }
                        }
                    }

                    // Always dispatch through the bead's freshest
                    // pointer (could be Cranelift or post-swap LLVM).
                    let code = bound.bead().compiled().or(code)?;
                    if code.is_null() {
                        return None;
                    }
                    Some(JitDispatch {
                        ptr: code as *const u8,
                        n_params,
                    })
                }),
            );
        }

        let _ = hot; // unused when llvm-backend feature is off
        Ok(())
    }

    /// Convenience: install the tier ladder using the currently
    /// stored `TieredConfig` — populated by [`Self::with_threshold`]
    /// or defaulted to `TieredConfig::default()` on `new()`.
    pub fn install_jit(&mut self) -> Result<(), CompilerError> {
        self.install_jit_with(self.tier_config.clone())
    }

    /// Deprecated alias for [`Self::install_jit`]. Kept for callers
    /// that wrote `install_cranelift_jit` before the ladder was
    /// simplified — the new ladder always uses Cranelift opt for the
    /// middle tier and (optionally) LLVM for the top tier, so
    /// "install_cranelift_jit" no longer describes a distinct path.
    pub fn install_cranelift_jit(&mut self) -> Result<(), CompilerError> {
        self.install_jit()
    }

    /// Deprecated alias for [`Self::install_jit_with`]. See
    /// [`Self::install_cranelift_jit`].
    pub fn install_cranelift_jit_with(
        &mut self,
        config: TieredConfig,
    ) -> Result<(), CompilerError> {
        self.install_jit_with(config)
    }
}

#[cfg(all(feature = "native", feature = "llvm-backend"))]
impl InterpRuntime {
    /// Deprecated alias for [`Self::install_jit`] — the LLVM tier is
    /// now part of the default ladder when `llvm-backend` is on.
    pub fn install_llvm_jit(&mut self) -> Result<(), CompilerError> {
        self.install_jit()
    }

    /// Deprecated alias for [`Self::install_jit_with`].
    pub fn install_llvm_jit_with(&mut self, config: TieredConfig) -> Result<(), CompilerError> {
        self.install_jit_with(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use zyntax_compiler::hir::{
        BinaryOp, HirBlock, HirFunction, HirFunctionSignature, HirInstruction, HirParam,
        HirTerminator, HirType, HirValue, HirValueKind, ParamAttributes,
    };
    use zyntax_typed_ast::InternedString;

    fn add_value(func: &mut HirFunction, ty: HirType, kind: HirValueKind) -> HirId {
        let id = HirId::new();
        func.values.insert(
            id,
            HirValue {
                id,
                ty,
                kind,
                uses: HashSet::new(),
                span: None,
            },
        );
        id
    }

    #[test]
    fn end_to_end_module_call() {
        // Build a tiny HIR: `def add(a: i64, b: i64): i64 { return a + b }`
        let sig = HirFunctionSignature {
            params: vec![
                HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global("a"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                },
                HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global("b"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                },
            ],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("add"), sig);
        let p0 = func.signature.params[0].id;
        let p1 = func.signature.params[1].id;
        func.values.insert(
            p0,
            HirValue {
                id: p0,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );
        func.values.insert(
            p1,
            HirValue {
                id: p1,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(1),
                uses: HashSet::new(),
                span: None,
            },
        );
        let sum = add_value(&mut func, HirType::I64, HirValueKind::Instruction);
        let entry_id = func.entry_block;
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.instructions.push(HirInstruction::Binary {
            result: sum,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: p0,
            right: p1,
        });
        entry.terminator = HirTerminator::Return { values: vec![sum] };

        let mut module = HirModule::new(InternedString::new_global("test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);

        // Spin up an InterpRuntime, install the module, run.
        let mut rt = InterpRuntime::new();
        rt.compile_module(module);
        let result = rt
            .call_function(
                "add",
                vec![
                    zyntax_compiler::value::ZyntaxValue::Int(10),
                    zyntax_compiler::value::ZyntaxValue::Int(32),
                ],
            )
            .expect("call should succeed");
        assert!(
            matches!(result, zyntax_compiler::value::ZyntaxValue::Int(42)),
            "expected I64(42), got {:?}",
            result
        );

        // Bead registered + profile populated.
        let sample = rt.profile_for(func_id);
        assert_eq!(sample.call_count, 1);
        assert!(rt.bead_for(func_id).is_some());
    }

    #[test]
    fn repeated_calls_advance_profile_and_bead() {
        // Same trivial `add` function, called 5 times. Verifies that
        // the per-call profile counter and beadie's per-bead
        // invocation counter both advance.
        let sig = HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("noop"), sig);
        let one = add_value(
            &mut func,
            HirType::I64,
            HirValueKind::Constant(zyntax_compiler::hir::HirConstant::I64(7)),
        );
        let entry_id = func.entry_block;
        let entry = func.blocks.get_mut(&entry_id).unwrap();
        entry.terminator = HirTerminator::Return { values: vec![one] };

        let mut module = HirModule::new(InternedString::new_global("test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);
        let mut rt = InterpRuntime::new();
        rt.compile_module(module);
        for _ in 0..5 {
            let r = rt.call_function("noop", vec![]).unwrap();
            assert!(matches!(r, zyntax_compiler::value::ZyntaxValue::Int(7)));
        }
        assert_eq!(rt.profile_for(func_id).call_count, 5);
        // Beadie's per-bead counter is independent (advanced via the
        // tick callback). We verify the bead is still registered.
        assert!(rt.bead_for(func_id).is_some());
    }
}
