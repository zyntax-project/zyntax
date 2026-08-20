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
use zyntax_compiler::hir_interp::{
    jit_dispatch_supported, jit_float_mask, HirInterpreter, InterpError, JitDispatch, JitRet,
    ProfileSample,
};
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
    /// The functions a host can enter through, which dead-code
    /// analysis prunes from. Empty means nothing is pruned, which is
    /// the safe answer when nobody has said.
    entry_names: Vec<String>,
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
    /// Handles for in-flight eager LLVM tier-up compiles.
    ///
    /// The thread runs MCJIT codegen, which reads LLVM's process-global
    /// registries. Those are C++ statics destroyed by the exit handlers,
    /// so a thread still inside codegen when the process tears down
    /// faults on freed state. Keeping the handle makes that lifetime
    /// something an owner controls rather than a race against `exit`.
    #[cfg(all(feature = "native", feature = "llvm-backend"))]
    llvm_tierup: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(all(feature = "native", feature = "llvm-backend"))]
impl Drop for InterpRuntime {
    fn drop(&mut self) {
        // Wait out any in-flight MCJIT codegen. Not interruptible once
        // entered, so this waits rather than cancels; it is tens to a
        // few hundred milliseconds and only on the last runtime to go.
        for h in self.llvm_tierup.drain(..) {
            let _ = h.join();
        }
    }
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
    /// Name the functions a host can enter the program through.
    ///
    /// Dead-code analysis prunes from these, so a wrong answer prunes
    /// from the wrong root. They come from what a grammar declares or
    /// what a host configures; this layer has no entry point of its own
    /// to assume. With none given, nothing is pruned.
    pub fn set_entry_names(&mut self, names: Vec<String>) {
        self.entry_names = names;
    }

    /// What [`Self::set_entry_names`] was told, for passing to the
    /// reachability analysis.
    fn entry_name_refs(&self) -> Vec<&str> {
        self.entry_names.iter().map(String::as_str).collect()
    }

    /// Create an empty runtime. On native, tier policies come from
    /// [`TieredConfig::default`] (warm/hot thresholds from
    /// `ProfileConfig::default`). On wasm there is no JIT tier — the
    /// interpreter runs everything.
    pub fn new() -> Self {
        Self {
            entry_names: Vec::new(),
            module: None,
            interp: HirInterpreter::new(),
            #[cfg(all(feature = "native", feature = "llvm-backend"))]
            llvm_tierup: Vec::new(),
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
                entry_names: Vec::new(),
                module: None,
                interp: HirInterpreter::new(),
                #[cfg(all(feature = "native", feature = "llvm-backend"))]
                llvm_tierup: Vec::new(),
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
    /// Functions in the installed module that the bytecode interpreter
    /// cannot execute, with what each one uses.
    ///
    /// Ask this where the interpreter is the ONLY engine. Bytecode
    /// compilation is lazy, so without it a program that performs an
    /// effect installs cleanly and only fails when the path is reached.
    /// Where the interpreter is a tier with a JIT beneath it, a
    /// non-empty answer is not an error: the JIT runs those functions.
    pub fn unsupported_constructs(&self) -> Vec<(String, &'static str)> {
        self.module
            .as_ref()
            .map(|m| zyntax_compiler::hir_interp::unsupported_constructs(m))
            .unwrap_or_default()
    }

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
            // Prefer the typed registration path when a signature is
            // present — that wires the BC interp's `Op::CallSym`
            // marshalling through the platform float ABI so float
            // params don't bit-truncate. The untyped fallback stays
            // for symbols registered without a sig (legacy plugin
            // surface).
            match sym.sig {
                Some(sig) => self
                    .interp
                    .register_symbol_typed(sym.name.to_string(), sym.ptr, sig),
                None => self
                    .interp
                    .register_symbol(sym.name.to_string(), sym.ptr, 0),
            }
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
    /// `feature = "llvm-backend"`, `Self::install_llvm_jit`) which
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
                        // Generic seam: no signature info available
                        // here. Assume all-i64 — callers wiring
                        // f64-arg / float-return functions through this
                        // seam must use the typed install path
                        // (`install_jit_with`) instead.
                        float_mask: 0,
                        ret: JitRet::Int,
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
        // back through the OSR registry for further tier-up, plus every
        // FFI symbol the interpreter knows about.
        //
        // The JIT falls back to `dlsym` for anything it wasn't given, and
        // on Linux the executable's symbols are absent from `.dynsym`
        // unless it was linked `-rdynamic`, so helpers like
        // `zyntax_box_f64` resolve on macOS and fail here. Handing the
        // table over makes resolution explicit on both platforms.
        let interp_syms = self.symbol_table_snapshot();
        let mut syms: Vec<(&str, *const u8)> = interp_syms
            .iter()
            .map(|(name, ptr, _)| (name.as_str(), *ptr))
            .collect();
        syms.extend(osr::osr_runtime_symbols());
        #[allow(unused_mut)]
        let mut cranelift_inner = CraneliftBackend::with_runtime_symbols(&syms)
            .map_err(|e| CompilerError::Backend(format!("cranelift init failed: {e}")))?;
        // With an LLVM tier above it, Cranelift's helpers are not resume
        // points worth having: tier 1 emits the same code as tier 0, and a
        // transfer into a probe-free helper consumes the loop's one chance
        // to move up.
        #[cfg(feature = "llvm-backend")]
        cranelift_inner.set_publish_osr_helpers(false);
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
        // Flips to `true` once the background LLVM compile thread (spawned
        // below) finishes. The hot-path tick callback gates its tier-up
        // attempt on this so it never blocks waiting for the BG thread's
        // Mutex. Until the flag flips, functions stay on Cranelift.
        #[cfg(feature = "llvm-backend")]
        let llvm_ready: Arc<std::sync::atomic::AtomicBool> =
            Arc::new(std::sync::atomic::AtomicBool::new(false));

        let _ = config.verbosity;
        let cranelift_for_closure = Arc::clone(&cranelift);

        // Compile the WHOLE module up front into the tier-up
        // backend so cross-function calls (e.g. `main` →
        // `mandel_count`) resolve against the module-wide function
        // table. The previous per-function `compile_function`
        // shape used the legacy single-function entry that
        // declares against an empty module — any inter-function
        // Call inside the compiled body resolved to an unknown
        // symbol, baking in the wrong (null/zero) target. The
        // observable effect was the JIT'd mandelbrot main always
        // returning `Int(0)` (mandel_count never actually ran)
        // and naive-recursive fib doing the same for the same
        // reason. Compiling the module once at install time fixes
        // both with cross-function symbol resolution.
        // Tier-0 code carries a back-edge probe so a frame already
        // running it can transfer into tier-1 LLVM code instead of
        // finishing in the slower tier. The probe is a load of this
        // site's helper slot plus a not-taken branch.
        // Reachability DCE: restrict Cranelift body-compilation to functions
        // transitively callable from `main`. `import prelude` pulls in ~100
        // helpers; on the bench kernels (mandelbrot, nbody, naive-recursive
        // fib) none of them are reached, so this skips ~30-40 ms of Cranelift
        // codegen per install. Any unexpected call still works via the BC
        // interp's lazy-compile fallback.
        let reachable = zyntax_compiler::reachable_function_ids(&module, &self.entry_name_refs());
        let bead_ids_for_cranelift = self.bead_ids.clone();
        let enable_osr = config.enable_osr;
        cranelift.with_lock(|be| -> Result<(), CompilerError> {
            be.set_only_compile_reachable(Some(reachable));
            be.set_compile_tier(0);
            // Probe sites embed the slot address for their own function's
            // bead; site keys are only unique within a function, so sharing
            // one bead across the module would alias them.
            be.set_bead_ids(bead_ids_for_cranelift);
            be.set_emit_osr_probes(enable_osr);
            be.compile_module(&module)
                .map_err(|e| CompilerError::Backend(format!("tier-up compile: {e}")))?;
            be.finalize_definitions()
                .map_err(|e| CompilerError::Backend(format!("tier-up finalize: {e}")))?;
            if std::env::var("ZYNTAX_TRACE_TIER_UP").is_ok() {
                for (fid, f) in &module.functions {
                    let name = f.name.resolve_global().unwrap_or_default();
                    let ptr = be.get_function_ptr(*fid);
                    eprintln!("[TIER-UP-INSTALL] {name} {fid:?} -> {ptr:?}");
                }
            }
            Ok(())
        })?;

        // LLVM eager module compile (mirrors the Cranelift pre-compile
        // above). The per-function tier-up callback for LLVM previously
        // called `LLVMJitBackend::compile_function`, which builds a
        // temp single-function HirModule and fails as soon as the
        // function calls any other (Cranelift IR refs an HirId not
        // present in the temp module → "Function not found"). Pre-
        // compiling the whole module here gives LLVM the full cross-
        // function symbol table up front, so the tier-up swap is just
        // a pointer lookup — no per-function recompile race.
        //
        // Soft failure: the LLVM backend currently doesn't translate
        // every HIR type (Opaque types from `@reference class` lowering
        // or extern types like Tensor return "Type translation not yet
        // implemented"). If the eager compile errors out, fall back to
        // a Cranelift-only ladder for this module rather than poisoning
        // the whole install — the runtime still runs, the LLVM tier
        // just doesn't engage. Cached function pointers from a partial
        // earlier compile are cleared by the next compile_module call.
        // Only when the LLVM tier can actually fire. A caller that sets
        // `hot_threshold` beyond reach wants a Cranelift-only ladder, and
        // eagerly compiling a module nothing will ever install is work the
        // process then has to wait out on the way down.
        #[cfg(feature = "llvm-backend")]
        if hot != u32::MAX {
            // Feed every runtime FFI symbol from the BC interpreter
            // into the LLVM backend before compile_module fires. The
            // AOT-via-object path links a closed-world shared library
            // — every named call from the .so must resolve against a
            // trampoline stub that the linker emits. Symbols that
            // aren't pre-registered here would otherwise show up as
            // undefined-symbol link errors.
            //
            // Mirrors the wasm-JIT and Cranelift paths, which all
            // pull from the same `symbol_table_snapshot()`.
            let bc_symbols = self.interp.symbol_table_snapshot();
            // Also include OSR runtime symbols so back-edge probes
            // baked into the IR (when emitted) resolve correctly.
            let osr_syms = osr::osr_runtime_symbols();
            // Same reachable-from-main filter the Cranelift install
            // uses above. Without it LLVM lowers + O3s + verifies
            // every prelude helper (Option<T>, Result<T,E>, List /
            // Iterator / Tensor / etc.) and pays the codegen +
            // verifier cost for ~100 functions the kernel never
            // reaches. The single-kernel install measured 300-900 ms
            // before this gate landed; with it the LLVM compile sees
            // only the same subset Cranelift does.
            let llvm_reachable =
                zyntax_compiler::reachable_function_ids(&module, &self.entry_name_refs());
            let llvm_entry_names: std::collections::HashSet<String> =
                self.entry_names.iter().cloned().collect();
            let llvm_cache_key = config.llvm_cache_key.clone();
            let box_infos = crate::effect_runtime::box_runtime_symbol_infos();
            llvm.with_lock(|be| {
                for (name, ptr, _arity) in &bc_symbols {
                    be.register_symbol(name.clone(), *ptr);
                }
                for (name, ptr) in &osr_syms {
                    be.register_symbol((*name).to_string(), *ptr);
                }
                // Wire the `zyntax_box_*` signatures so LLVM emits
                // the correct return / param types. Without this the
                // backend defaults each `Call::Symbol` to
                // `void(args) → i32 0`, then panics the moment
                // an unboxed f64 / f32 / bool value is consumed.
                be.register_symbol_signatures(&box_infos);
                be.set_only_compile_reachable(Some(llvm_reachable));
                // An entry keeps the name it was written with, so a
                // host can still ask for it by that name; every other
                // local function is mangled to its id. Which names
                // those are is configured, not decided here.
                be.set_entry_names(llvm_entry_names);
                be.set_cache_key(llvm_cache_key);
            });

            // Background-thread compile. On the cold path
            // `compile_module` runs the long-pole pipeline (lower → opt
            // → link → dlopen, ~80-300 ms depending on platform);
            // running it on the install thread blocks BC interp warmup
            // for that entire window. Spawning a dedicated thread lets
            // the BC interp start executing immediately and the
            // per-function tick callback below polls `llvm_ready`
            // before attempting a tier-up swap. Until the flag flips,
            // the function stays on Cranelift — already fast enough
            // for the first few invocations. The cached-dylib path
            // returns in a few ms, so backgrounding is effectively a
            // no-op there but costs nothing.
            let llvm_for_bg = Arc::clone(&llvm);
            let llvm_ready_for_bg = Arc::clone(&llvm_ready);
            let bead_ids_for_bg = self.bead_ids.clone();
            let module_for_bg = module.clone();
            let trace = std::env::var("ZYNTAX_TRACE_TIER_UP").is_ok();
            let handle = std::thread::Builder::new()
                .name("zyntax-llvm-tierup".to_string())
                // LLVM's pass pipeline recurses deeply — SimplifyCFG and the
                // inliner in particular — and a spawned thread gets 2 MB by
                // default, which it overruns on a module of any size.
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    let r = llvm_for_bg.with_lock(|be| {
                        // Tier 1 additionally emits a resume point per loop
                        // header, so a frame already running tier-0 code can
                        // finish in LLVM rather than waiting for the next
                        // call.
                        be.set_compile_tier(1);
                        let r = be.compile_module(&module_for_bg);
                        if r.is_ok() {
                            // Slots are keyed by bead, and a whole-module
                            // install covers many, so each helper is
                            // published against its own function's bead.
                            for (func_id, site, code) in be.take_pending_osr_helpers() {
                                if let Some(bead_id) = bead_ids_for_bg.get(&func_id) {
                                    zyntax_compiler::osr::note_llvm_helper(code as usize);
                                    zyntax_compiler::osr::publish_helper(*bead_id, site, code);
                                }
                            }
                        }
                        r
                    });
                    match r {
                        Ok(()) => {
                            llvm_ready_for_bg.store(true, std::sync::atomic::Ordering::Release);
                            if trace {
                                llvm_for_bg.with_lock(|be| {
                                    for (fid, f) in &module_for_bg.functions {
                                        let name = f.name.resolve_global().unwrap_or_default();
                                        let ptr = be.get_function_pointer(*fid);
                                        eprintln!(
                                            "[TIER-UP-INSTALL-LLVM] {name} {fid:?} -> {ptr:?}"
                                        );
                                    }
                                });
                            }
                        }
                        Err(e) => {
                            // Soft failure: most often IR generation, which
                            // both installers share -- a HIR type the LLVM
                            // backend does not translate yet (Opaque from
                            // @reference-class lowering, extern types like
                            // Tensor), or a module the verifier rejects. Only
                            // if the object-file installer was selected can it
                            // additionally be `NoLinker` or a link/dlopen
                            // failure; the default MCJIT path needs no
                            // toolchain. The runtime still runs, and the LLVM
                            // tier just never engages because `llvm_ready`
                            // stays false. Surface the cause loudly: the bench
                            // harness captures stderr but doesn't init
                            // `env_logger`.
                            eprintln!("[LLVM-TIER-DISABLED] {e}");
                            log::warn!(
                                "LLVM tier-up disabled for this module: eager compile_module failed ({e}). \
                                 Cranelift tier-up will still drive the JIT path."
                            );
                        }
                    }
                })
                .expect("spawn zyntax-llvm-tierup thread");
            self.llvm_tierup.push(handle);
        }

        // JIT-prime the entry function so the first call dispatches
        // compiled code. beadie returns `None` from the first
        // `on_invoke` regardless of whether the pre-compiled function
        // pointer is ready — without priming, BC interp runs the
        // entry function once before the bead transitions to
        // `Compiled`. For short kernels that doesn't matter; for
        // rayzor-scale loops (nbody's 20×500K outer loop) the first
        // BC-interp run is hundreds of times slower than steady state
        // and either dominates wall-clock or trips the benchmark's
        // sustained-load assumptions.
        //
        // Mirrors rayzor's `TierPreset::Benchmark` which uses
        // `BailoutStrategy::Immediate` to leave the interpreter after
        // ~10 instructions and re-enter the call in compiled code.
        // We achieve the same end state by walking the entry pointer
        // forward before any user call lands.
        //
        // Was gated behind `ZYNTAX_ENABLE_JIT_PRIME=1` because
        // standalone JIT compilation of nbody's `main` hit a Cranelift
        // `unreachable` trap — that bug was fixed by commit 432d017
        // (SSA Unreachable → synthesised Return). The gate is now
        // stale and the priming is always-on.
        let entry_names = self.entry_name_refs();
        if let Some(main_id) = module.functions.iter().find_map(|(id, f)| {
            let name = f.name.resolve_global()?;
            entry_names.contains(&name.as_str()).then_some(*id)
        }) {
            if let Some(bound) = self.bounds.get(&main_id).cloned() {
                let cranelift_for_prime = Arc::clone(&cranelift);
                self.tiered.on_invoke(&bound, move |_tier, _bead| {
                    cranelift_for_prime
                        .with_lock(|be| be.get_function_ptr(main_id))
                        .map(|p| p as *mut ())
                        .unwrap_or(std::ptr::null_mut())
                });
                let deadline = web_time::Instant::now() + std::time::Duration::from_millis(500);
                while bound.bead().compiled().is_none() && web_time::Instant::now() < deadline {
                    std::hint::spin_loop();
                }
            }
        }

        // Cranelift dispatch closure (tier 0, beadie-driven). With
        // the module pre-compiled above the closure just looks up
        // the function pointer — no recompilation per tier-up.
        let cranelift_compile = {
            let func_arcs = Arc::clone(&func_arcs);
            move |tier: usize, func_id: HirId| -> Option<(*const u8, u8)> {
                let (func_arc, _bead_id) = func_arcs.get(&func_id)?;
                let n_params = func_arc.signature.params.len().min(255) as u8;
                let _ = tier; // always tier 0 here
                let ptr = cranelift_for_closure.with_lock(|be| be.get_function_ptr(func_id))?;
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

            // Pre-compute the FFI-bridge dispatch shape for this
            // function so the hot path doesn't re-derive it from the
            // signature on every invocation. If the signature falls
            // outside the bridge's supported matrix (>4-arg with
            // mixed f64/i64, or struct returns), `supported` is
            // false and the tick-callback returns None — that
            // function stays in BC interp instead of crashing at
            // the bridge.
            let (float_mask, ret_kind, jit_supported) =
                if let Some((func_arc, _)) = func_arcs.get(&func_id) {
                    let p = &func_arc.signature.params;
                    let ret = func_arc
                        .signature
                        .returns
                        .first()
                        .cloned()
                        .unwrap_or(zyntax_compiler::hir::HirType::Void);
                    let param_tys: Vec<_> = p.iter().map(|hp| hp.ty.clone()).collect();
                    let ret_kind = match ret {
                        zyntax_compiler::hir::HirType::F32 => JitRet::F32,
                        zyntax_compiler::hir::HirType::F64 => JitRet::F64,
                        _ => JitRet::Int,
                    };
                    (
                        jit_float_mask(&param_tys),
                        ret_kind,
                        jit_dispatch_supported(&param_tys, &ret),
                    )
                } else {
                    (0, JitRet::Int, false)
                };

            #[cfg(feature = "llvm-backend")]
            let llvm_state = Arc::clone(&llvm_state);
            #[cfg(feature = "llvm-backend")]
            let llvm_for_closure = Arc::clone(&llvm);
            #[cfg(feature = "llvm-backend")]
            let keepalive = Arc::clone(&_llvm_context_keepalive);
            #[cfg(feature = "llvm-backend")]
            let llvm_ready_for_closure = Arc::clone(&llvm_ready);

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
                    //
                    // Gate generation: tier 0's broker uses beadie's
                    // `install_compiled` (no generation bump), so right
                    // after the initial Cranelift install the bead
                    // reports `compiled() = Some(ptr)` AND `generation()
                    // = 0`. The LLVM upgrade uses `swap_compiled` which
                    // bumps to 1. So `generation() == 0` means "first
                    // tier landed, ready for LLVM upgrade"; `== 1`
                    // means LLVM has already fired and this gate must
                    // not re-fire. The `llvm_state` flag enforces
                    // once-per-function regardless.
                    #[cfg(feature = "llvm-backend")]
                    {
                        let count = bound.bead().invocation_count();
                        // The `llvm_ready` gate keeps the with_lock call
                        // below from blocking on the background compile
                        // thread's Mutex hold. Once the BG thread releases
                        // the Mutex and stores `true`, the next hot tick
                        // sees the flag, the CAS fires once, and the
                        // pointer lookup is a few-ns hash hit.
                        if llvm_ready_for_closure.load(std::sync::atomic::Ordering::Acquire)
                            && code.is_some()
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
                            let _ = &keepalive; // pin inkwell Context
                                                // LLVM was pre-compiled in the BG thread, so
                                                // this is just a pointer lookup — no per-
                                                // function recompile.
                            let llvm_ptr =
                                llvm_for_closure.with_lock(|be| be.get_function_pointer(func_id));
                            if let Some(ptr) = llvm_ptr {
                                // swap_compiled bumps generation 0 → 1
                                // and atomically publishes the new ptr.
                                bound.bead().swap_compiled(ptr as *mut ());
                            }
                        }
                    }

                    // Always dispatch through the bead's freshest
                    // pointer (could be Cranelift or post-swap LLVM).
                    let code = bound.bead().compiled().or(code)?;
                    if code.is_null() {
                        return None;
                    }
                    if !jit_supported {
                        // Signature outside the FFI-bridge matrix —
                        // never hand a dispatch back, the function
                        // keeps running in BC interp.
                        return None;
                    }
                    Some(JitDispatch {
                        ptr: code as *const u8,
                        n_params,
                        float_mask,
                        ret: ret_kind,
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
        HirTerminator, HirType, HirValue, HirValueKind, ParamAttributes, ParamOwnership,
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
                    ownership: ParamOwnership::default(),
                },
                HirParam {
                    id: HirId::new(),
                    name: InternedString::new_global("b"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                    ownership: ParamOwnership::default(),
                },
            ],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
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
            is_fiber: false,
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
