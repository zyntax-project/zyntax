//! The tiered runtime: a program starts interpreted and moves up.
//!
//! Alongside it live the tokens a host holds onto across a call, for
//! the fibers it steps and the effect handlers it installs.

use super::native_call::{call_dynamic_function, call_native_with_signature, dynamic_to_i64};
use super::promise::ZyntaxPromise;
use super::types::{NativeSignature, NativeType, RuntimeError, RuntimeEvent, RuntimeResult};
use super::{
    apply_krio_async_lowering, apply_krio_effect_lowering, apply_krio_fiber_lowering,
    capture_runtime_events_from_program, synthesize_handler_state, CompiledImportResolverCallback,
    ImportResolverCallback,
};
use crate::convert::FromZyntax;
use crate::error::ZyntaxError;
use crate::grammar::{GrammarError, LanguageGrammar};
use crate::value::ZyntaxValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use zyntax_compiler::{
    hir::{HirId, HirModule},
    lowering::AstLowering,
    tiered_backend::{OptimizationTier, TieredBackend, TieredConfig, TieredStatistics},
    zrtl::DynamicValue,
    CompilationConfig, CompilerError,
};

/// A multi-tier JIT runtime with automatic optimization
///
/// `TieredRuntime` provides adaptive compilation where frequently-called
/// functions are automatically optimized to higher tiers:
///
/// - **Tier 0 (Baseline)**: Fast compilation, minimal optimization (cold code)
/// - **Tier 1 (Standard)**: Moderate optimization (warm code)
/// - **Tier 2 (Optimized)**: Aggressive optimization (hot code)
///
/// ## How It Works
///
/// 1. All functions start at Tier 0 (baseline JIT with Cranelift)
/// 2. Execution counters track how often functions are called
/// 3. When a function crosses the "warm" threshold, it's recompiled at Tier 1
/// 4. When it crosses the "hot" threshold, it's recompiled at Tier 2
/// 5. Function pointers are atomically swapped after recompilation
///
/// ## Example
///
/// ```ignore
/// use zyntax_embed::{TieredRuntime, TieredConfig};
///
/// // Development: Fast startup, no background optimization
/// let mut runtime = TieredRuntime::development()?;
///
/// // Production: Full tiered optimization with background worker
/// let mut runtime = TieredRuntime::production()?;
///
/// // Production with LLVM for Tier 2 (requires llvm-backend feature)
/// let mut runtime = TieredRuntime::production_llvm()?;
/// ```
pub struct TieredRuntime {
    /// The tiered JIT backend
    backend: TieredBackend,
    /// Mapping from function names to HIR IDs
    function_ids: HashMap<String, HirId>,
    /// Function signatures for native calling
    function_signatures: HashMap<String, NativeSignature>,
    /// Tiered configuration
    config: TieredConfig,
    /// Registered language grammars (language name -> grammar)
    grammars: HashMap<String, Arc<LanguageGrammar>>,
    /// File extension to language mapping (e.g., ".zig" -> "zig")
    extension_map: HashMap<String, String>,
    /// Plugin signatures (symbol name -> ZRTL signature)
    /// Collected from loaded plugins for proper extern function type checking
    plugin_signatures: HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig>,
    /// Plugins whose symbols have been registered.
    ///
    /// Registration hands out raw pointers into the plugin's library, so the
    /// library has to outlive them. Dropping it closes the handle, and on a
    /// platform where that actually unmaps the image every registered
    /// pointer is left dangling.
    loaded_plugins: Vec<zyntax_compiler::zrtl::ZrtlPlugin>,
    /// Import resolver callbacks. Same role as `ZyntaxRuntime.import_resolvers`
    /// — consulted during `lower_typed_program` to pull in stdlib source
    /// (`prelude`, `tensor`, …) and any user-supplied module sources.
    import_resolvers: Vec<ImportResolverCallback>,
    /// Build-time parsed imports, consulted before source resolvers.
    compiled_import_resolvers: Vec<CompiledImportResolverCallback>,
    /// Modules a snapshot installed, keyed by the language that
    /// brought them, so a name means what it means inside the language
    /// asking rather than whichever language registered first.
    snapshot_modules: crate::import_chain::SnapshotModules,
    /// Captured runtime semantic events (render/stream).
    runtime_events: Vec<RuntimeEvent>,
    /// Optional callback invoked whenever a runtime event is captured.
    event_sink: Option<Arc<dyn Fn(&RuntimeEvent) + Send + Sync>>,
    /// Extern aliases threaded into typed-program lowering (e.g.
    /// `sleep` → `__zyntax_async_set_timeout`), the counterpart of the
    /// classic runtime's `config.builtins`. Grammar-driven loads carry
    /// their own map; this one serves the typed-program entry points.
    builtin_aliases: indexmap::IndexMap<String, String>,
    /// Host-driven fibers, keyed by token. See [`TieredRuntime::get_fiber`].
    host_fibers: HashMap<u64, HostFiber>,
    next_fiber_token: u64,
    /// Source-declared yield shape of each `fiber def`, with a
    /// generation that bumps when a reload changes the shape.
    fiber_shapes: HashMap<String, FiberShape>,
    /// Resolved handler names pinned by [`TieredRuntime::get_effect_handler`],
    /// keyed by token.
    handler_tokens: HashMap<u64, String>,
    next_handler_token: u64,
    /// Handler state regions the host allocated explicitly, keyed by
    /// instance handle. See [`TieredRuntime::new_handler_instance`].
    handler_instances: HashMap<u64, HandlerInstanceEntry>,
    next_handler_instance: u64,
    /// Undo record for the fiber-handle metadata the most recent
    /// applied reload changed, consumed by
    /// [`TieredRuntime::rollback_last_reload`] alongside the backend's
    /// code rollback.
    fiber_meta_undo: Option<FiberMetaUndo>,
    /// Built-in wrapper classes, extended via
    /// [`TieredRuntime::register_builtin_class`] before compilation —
    /// the same seam the classic runtime exposes.
    builtin_registry: Arc<std::sync::Mutex<zyntax_compiler::builtin_class::BuiltinRegistry>>,
}

/// What a reload did to fiber-handle metadata, for rollback.
#[derive(Default)]
struct FiberMetaUndo {
    /// Shape entries as they were before the reload replaced or
    /// removed them: `(function, prior entry — None if newly added)`.
    shapes: Vec<(String, Option<FiberShape>)>,
    /// Tokens the reload marked `machine_gone`.
    marked_gone: Vec<u64>,
}

/// Opaque host handle to a runtime-owned fiber. Stable across reloads
/// and OSR: the token names the machine, not its code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiberToken(u64);

/// Opaque host handle to a resolved EFFECT handler — a `handler H for
/// E { ... }` declaration, not any other sense of the word.
/// [`TieredRuntime::get_effect_handler`] resolves the (possibly
/// unqualified) name ONCE and pins the fully qualified result, so later
/// edits that introduce same-named handlers in other modules cannot
/// re-aim or ambiguate the host's binding — bare strings resolve per
/// call and can.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectHandlerToken(u64);

/// An installed handler frame, returned by
/// [`TieredRuntime::push_effect_handler`] and consumed by
/// [`TieredRuntime::pop_effect_handler`]. Deliberately not `Copy`: a
/// frame is ended exactly once.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct HandlerFrame(u64, Option<u64>);

/// The handler context in force at one point on one thread, taken by
/// [`TieredRuntime::capture_handler_context`] so a callback registered
/// now can run under it later.
///
/// Holds an install on each handler instance it names, so capturing is
/// enough to keep that state alive past the extent that pushed it.
/// Not `Send`: the frames point at state the compiled code reaches
/// through a thread-local stack.
#[derive(Debug)]
pub struct HandlerContext {
    frames: Vec<crate::effect_runtime::HandlerFrame>,
    instances: Vec<u64>,
}

/// An open [`TieredRuntime::enter_handler_context`] scope, closed by
/// [`TieredRuntime::leave_handler_context`]. Not `Copy`: a scope is
/// closed exactly once.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct HandlerContextScope(usize);

/// Opaque handle to ONE allocated instance of a stateful handler's
/// state.
///
/// Creating an instance and installing it are separate steps, so the
/// same state can back several installs: a machine can advance it
/// through [`TieredRuntime::bind_fiber_handler_instance`] while host
/// code reads it through [`TieredRuntime::push_handler_instance`].
/// Installs borrow the instance; the runtime owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandlerInstance(u64);

/// A host-allocated handler state and the dispatch shape it installs
/// with, plus what it takes to know when the region can go.
///
/// Freeing is owner-driven rather than scope-driven: one region can
/// back a bind and any number of pushes, so no single install may
/// release it. Each install holds a count, the owner asks for the drop,
/// and the last one out frees.
struct HandlerInstanceEntry {
    handler: String,
    effect_id: u64,
    /// The state region, as `usize` so the entry stays `Send`/`Sync`.
    /// Null for a stateless handler.
    state: usize,
    table: usize,
    async_mask: u64,
    /// Installs currently naming this region.
    installs: usize,
    /// The owner has released its handle; free once `installs` is zero.
    /// Set at creation for a region the runtime allocated implicitly on
    /// a caller's behalf, which nothing else can ever name.
    dropped_by_owner: bool,
}

/// One step of a host-driven machine.
#[derive(Debug, Clone, PartialEq)]
pub enum HostFiberStep {
    /// The machine yielded. The payload decodes with the shape the
    /// handle was created against — a machine in flight always yields
    /// that shape, because an incompatible edit leaves it completing
    /// on the code it started with.
    Yielded(ZyntaxValue),
    /// The machine completed. Later resumes return `Done` again.
    Done,
    /// The machine aborted (`Fiber.abort` semantics). The host signal
    /// is the state itself; a UI typically unmounts or remounts.
    Errored,
    /// An edit removed the machine's function. The fiber was NOT
    /// resumed: a machine whose source is gone should stop observing,
    /// and the host acts on the value — typically drop + remount.
    MachineGone,
}

/// What the runtime knows about a host-driven fiber.
#[derive(Debug, Clone)]
pub struct HostFiberInfo {
    pub function: String,
    /// Yield shape this handle decodes, captured at creation.
    pub yield_shape: String,
    /// The shape generation the handle was created against.
    pub shape_generation: u64,
    /// True when reloads changed the function's declared yield shape
    /// after this fiber was created. Payloads still decode with the
    /// creation shape — the running machine is of that generation —
    /// but the source has moved on; recreate the machine to adopt it.
    pub shape_stale: bool,
    /// True when an edit removed the machine's function.
    pub machine_gone: bool,
    pub done: bool,
}

/// How a yielded payload is decoded for the host. Captured at fiber
/// creation so a later edit never changes how this handle reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostYieldKind {
    Int,
    Bool,
    /// Anything that isn't a scalar the host channel can carry —
    /// surfaced as the raw payload.
    Opaque,
}

struct HostFiber {
    ptr: usize,
    /// Handler instances this machine holds an install of.
    bound_instances: Vec<u64>,
    function: String,
    yield_shape: String,
    yield_kind: HostYieldKind,
    shape_generation: u64,
    machine_gone: bool,
    done: bool,
}

#[derive(Clone)]
struct FiberShape {
    shape: String,
    kind: HostYieldKind,
    generation: u64,
}

fn yield_kind_of(ty: &zyntax_typed_ast::Type) -> HostYieldKind {
    use zyntax_typed_ast::{PrimitiveType as P, Type as T};
    match ty {
        T::Primitive(P::Bool) => HostYieldKind::Bool,
        T::Primitive(
            P::I8
            | P::I16
            | P::I32
            | P::I64
            | P::U8
            | P::U16
            | P::U32
            | P::U64
            | P::ISize
            | P::USize,
        ) => HostYieldKind::Int,
        _ => HostYieldKind::Opaque,
    }
}

/// Move one handler-state region into the edited layout: allocate with
/// the edited constructor and copy each field the two layouts share.
fn migrate_one(
    moves: &[(usize, usize, usize)],
    ctor: extern "C" fn() -> *mut u8,
    old_state: *mut u8,
) -> Option<*mut u8> {
    let fresh = ctor();
    if fresh.is_null() {
        return None;
    }
    for (from, to, size) in moves {
        // SAFETY: both regions are handler states of this handler, the
        // old one allocated by the previous constructor and the new one
        // just returned by the edited one, and every (offset, size)
        // came from the layout of the matching generation. The regions
        // are distinct allocations.
        unsafe {
            std::ptr::copy_nonoverlapping(old_state.add(*from), fresh.add(*to), *size);
        }
    }
    Some(fresh)
}

fn decode_yield(kind: HostYieldKind, payload: i64) -> ZyntaxValue {
    match kind {
        HostYieldKind::Int | HostYieldKind::Opaque => ZyntaxValue::Int(payload),
        HostYieldKind::Bool => ZyntaxValue::Bool(payload != 0),
    }
}

/// Resolve a possibly-unqualified `name` against module-scoped
/// candidates. An exact match wins; otherwise `name` matches a single
/// candidate spelled `path::name`. Zero or several matches error with
/// the candidates, so the caller can qualify.
fn resolve_scoped_name<'a>(
    name: &str,
    candidates: impl Iterator<Item = &'a str>,
) -> Result<String, Vec<String>> {
    let suffix = format!("::{name}");
    let mut matches: Vec<&str> = Vec::new();
    for c in candidates {
        if c == name {
            return Ok(c.to_string());
        }
        if c.ends_with(&suffix) {
            matches.push(c);
        }
    }
    match matches.as_slice() {
        [one] => Ok(one.to_string()),
        _ => Err(matches.into_iter().map(String::from).collect()),
    }
}

/// The `(name, shape, kind)` of every `fiber def` in a parsed program,
/// snapshotted before lowering consumes it.
fn collect_fiber_decls(
    program: &zyntax_typed_ast::TypedProgram,
) -> Vec<(String, String, HostYieldKind)> {
    use zyntax_typed_ast::TypedDeclaration;
    program
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) if f.is_fiber => {
                let name = f.name.resolve_global()?;
                Some((
                    name,
                    format!("{:?}", f.return_type),
                    yield_kind_of(&f.return_type),
                ))
            }
            _ => None,
        })
        .collect()
}

impl TieredRuntime {
    /// Create a tiered runtime with the given configuration
    pub fn new(config: TieredConfig) -> RuntimeResult<Self> {
        let mut backend = TieredBackend::new(config.clone())?;

        // Tier-0 loops carry a back-edge probe so a frame already running
        // them can transfer into tier-1 code. Applied at construction so
        // consumers reaching the backend through other entry points get the
        // same setting.
        backend.set_emit_osr_probes(config.enable_osr);

        // The boxing runtime the upper tier calls into. The ground tier
        // builds dynamic boxes inline and never needs these; a tier that
        // calls them by name has to be able to resolve them, and on Linux
        // the executable's own symbols are not in `.dynsym`.
        for (name, ptr, _) in zyntax_compiler::zrtl::box_runtime_symbols() {
            backend.register_runtime_symbol(name, ptr);
        }

        let mut runtime = Self {
            backend,
            function_ids: HashMap::new(),
            function_signatures: HashMap::new(),
            config,
            grammars: HashMap::new(),
            extension_map: HashMap::new(),
            plugin_signatures: HashMap::new(),
            loaded_plugins: Vec::new(),
            import_resolvers: Vec::new(),
            compiled_import_resolvers: Vec::new(),
            snapshot_modules: Default::default(),
            runtime_events: Vec::new(),
            event_sink: None,
            builtin_aliases: indexmap::IndexMap::new(),
            host_fibers: HashMap::new(),
            next_fiber_token: 1,
            fiber_shapes: HashMap::new(),
            handler_tokens: HashMap::new(),
            next_handler_token: 1,
            handler_instances: HashMap::new(),
            next_handler_instance: 1,
            fiber_meta_undo: None,
            builtin_registry: Arc::new(std::sync::Mutex::new(
                zyntax_compiler::builtin_class::BuiltinRegistry::with_defaults(),
            )),
        };

        // The effect and fiber runtime, exactly as the classic runtime
        // registers it — handler stacks, op dispatch, `krio_fiber_*`.
        crate::effect_runtime::register_effect_runtime_symbols(&mut runtime);
        crate::effect_runtime::register_fiber_runtime_symbols(&mut runtime);
        let _ = krio_adapter::fiber::install();
        runtime
            .backend
            .rebuild_with_accumulated_symbols()
            .map_err(|e| RuntimeError::Execution(format!("rebuild_jit: {e}")))?;

        Ok(runtime)
    }

    /// Register an import resolver callback. Same shape as
    /// [`ZyntaxRuntime::add_import_resolver`].
    pub fn add_import_resolver(&mut self, resolver: ImportResolverCallback) {
        self.import_resolvers.push(resolver);
    }

    /// Register a resolver for build-time parsed import artifacts.
    pub fn add_compiled_import_resolver(&mut self, resolver: CompiledImportResolverCallback) {
        self.compiled_import_resolvers.push(resolver);
    }

    /// Extern aliases for typed-program compiles, the counterpart of
    /// the classic runtime's `config_mut().builtins` — e.g. `sleep` →
    /// `__zyntax_async_set_timeout`. Applied by
    /// [`Self::compile_typed_program`] and
    /// [`Self::reload_typed_program`].
    pub fn builtin_aliases_mut(&mut self) -> &mut indexmap::IndexMap<String, String> {
        &mut self.builtin_aliases
    }

    /// Resolve a module name through the registered resolvers.
    pub fn resolve_import(&self, module_path: &str) -> Result<Option<String>, String> {
        crate::import_chain::resolve_import_with(&self.import_resolvers, module_path)
    }

    /// Create a runtime optimized for development
    ///
    /// - Fast compilation with minimal optimization
    /// - No background optimization worker
    /// - Good for rapid iteration and debugging
    pub fn development() -> RuntimeResult<Self> {
        Self::new(TieredConfig::development())
    }

    /// Create a runtime optimized for production
    ///
    /// - Full tiered optimization with Cranelift
    /// - Background optimization worker enabled
    /// - Automatic promotion of hot functions
    pub fn production() -> RuntimeResult<Self> {
        Self::new(TieredConfig::production())
    }

    /// Create a runtime with LLVM for maximum Tier 2 optimization
    ///
    /// - Uses LLVM MCJIT for hot-path optimization
    /// - Best performance for compute-intensive workloads
    /// - Requires the `llvm-backend` feature
    #[cfg(feature = "llvm-backend")]
    pub fn production_llvm() -> RuntimeResult<Self> {
        Self::new(TieredConfig::production_llvm())
    }

    /// Compile a HIR module into the tiered runtime
    pub fn compile_module(&mut self, mut module: HirModule) -> RuntimeResult<()> {
        // Run interp-safe HIR opts before backend installation. Without this,
        // user programs run through `TieredRuntime::compile_module` never get
        // CSE / LICM / inline / const_fold / aggregate_split — the bench-only
        // `run_interp_safe_opts` entry was the only place these fired,
        // leaving production code unoptimised. (Skippable via
        // `ZYNTAX_DISABLE_INTERP_OPTS=1`.)
        if std::env::var("ZYNTAX_DISABLE_INTERP_OPTS").is_err() {
            let _stats = zyntax_compiler::run_interp_safe_opts(&mut module);
        }
        zyntax_compiler::hir_dump::dump_module_to_dir(&module, "post-opt-tiered-compile_module");

        // Store function name -> ID mapping and signatures (resolve InternedString to actual string)
        for (id, func) in &module.functions {
            if let Some(name) = func.name.resolve_global() {
                self.function_ids.insert(name.clone(), *id);

                // Store the function signature for later use in call/call_async
                let native_sig = NativeSignature::from_hir_signature(&func.signature);
                self.function_signatures.insert(name, native_sig);
            }
        }

        self.backend.set_emit_osr_probes(self.config.enable_osr);

        // Compile the module (consumes it)
        self.backend.compile_module(module)?;

        Ok(())
    }

    /// Get a function pointer by name
    pub fn get_function_ptr(&self, name: &str) -> Option<*const u8> {
        self.function_ids
            .get(name)
            .and_then(|id| self.backend.get_function_pointer(*id))
    }

    /// Call a function by name with automatic type conversion
    ///
    /// This also records the call for profiling, which may trigger
    /// automatic optimization if the function becomes hot.
    pub fn call<T: FromZyntax>(&self, name: &str, args: &[ZyntaxValue]) -> RuntimeResult<T> {
        let result = self.call_raw(name, args)?;
        T::from_zyntax(result).map_err(RuntimeError::from)
    }

    /// Call a function and get the raw ZyntaxValue result
    pub fn call_raw(&self, name: &str, args: &[ZyntaxValue]) -> RuntimeResult<ZyntaxValue> {
        let func_id = self
            .function_ids
            .get(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;

        // A perform whose effect has no frame in scope resolves its
        // handler op statically, and a stateful op then reads an
        // implicit `self` that nothing supplied. Refusing the call is
        // the difference between an error the host can report and a
        // null dereference inside compiled code.
        for (effect_id, effect_name) in self.backend.stateful_effects_of(name) {
            if !crate::effect_runtime::has_handler_for(effect_id) {
                return Err(RuntimeError::Execution(format!(
                    "cannot call `{name}`: it uses effect `{effect_name}`, and no handler for \
                     `{effect_name}` is active on this thread. Handlers for `{effect_name}` keep \
                     their own state, so one has to be active around the call"
                )));
            }
        }

        // Record the call for profiling
        self.backend.record_call(*func_id);

        let ptr = self
            .backend
            .get_function_pointer(*func_id)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;

        // If we have a recorded HIR-derived signature, the function uses
        // the native scalar/pointer ABI rather than the DynamicValue ABI.
        // Dispatch through `call_native_with_signature` so scalar returns
        // (e.g. `def main(): i64`) are decoded correctly. Without this,
        // raw i64 returns get reinterpreted as a `DynamicValue` struct and
        // the subsequent dereference SIGSEGVs.
        if let Some(sig) = self.function_signatures.get(name) {
            // SAFETY: signature stored at load time matches the JIT
            // function's actual signature.
            return unsafe { call_native_with_signature(ptr, args, sig) };
        }

        // Convert arguments to DynamicValues
        let dynamic_args: Vec<DynamicValue> =
            args.iter().cloned().map(|v| v.into_dynamic()).collect();

        // Call the function using the variadic caller helper
        // SAFETY: We trust the caller has provided the correct function pointer
        // and matching arguments. from_dynamic is safe because call_dynamic_function
        // returns a valid DynamicValue.
        unsafe {
            let result = call_dynamic_function(ptr, &dynamic_args)?;
            ZyntaxValue::from_dynamic(result).map_err(RuntimeError::from)
        }
    }

    /// Call an async function, returning a Promise
    ///
    /// With the new Promise-based async ABI:
    /// - Calling `double(21)` returns a Promise struct `{state_machine_ptr, poll_fn_ptr}`
    /// - The Promise contains everything needed to poll for completion
    /// - No `_new`/`_poll` naming convention needed
    pub fn call_async(&self, name: &str, args: &[ZyntaxValue]) -> RuntimeResult<ZyntaxPromise> {
        // First, try the new Promise-returning API
        // The async function directly returns Promise<T> = {state_machine_ptr, poll_fn_ptr}
        if let Some(func_id) = self.function_ids.get(name) {
            self.backend.record_call(*func_id);
            let func_ptr = self
                .backend
                .get_function_pointer(*func_id)
                .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;

            // Look up the stored signature for this function
            let signature = self
                .function_signatures
                .get(name)
                .cloned()
                .unwrap_or_else(|| {
                    // Fallback: infer signature from args (legacy behavior)
                    let params: Vec<NativeType> = args
                        .iter()
                        .map(|arg| match arg {
                            ZyntaxValue::Int(_) => NativeType::I64,
                            ZyntaxValue::Float(_) => NativeType::F64,
                            ZyntaxValue::Bool(_) => NativeType::Bool,
                            ZyntaxValue::String(_) => NativeType::Ptr,
                            ZyntaxValue::Null => NativeType::Ptr,
                            ZyntaxValue::Pointer(_) => NativeType::Ptr,
                            _ => NativeType::Ptr, // All other types (Array, Struct, Map, etc.)
                        })
                        .collect();
                    NativeSignature {
                        params,
                        ret: NativeType::Ptr,
                    }
                });

            let dynamic_args: Vec<DynamicValue> =
                args.iter().cloned().map(|v| v.into_dynamic()).collect();

            // Call the function - it returns a Promise struct
            return Ok(unsafe {
                ZyntaxPromise::from_async_call(func_ptr, dynamic_args, &signature)
            });
        }

        // Fall back to legacy _new/_poll naming convention for backwards compatibility
        let new_name = format!("{}_new", name);
        let poll_name = format!("{}_poll", name);

        if let (Some(new_id), Some(poll_id)) = (
            self.function_ids.get(&new_name),
            self.function_ids.get(&poll_name),
        ) {
            self.backend.record_call(*new_id);
            self.backend.record_call(*poll_id);

            let new_ptr = self
                .backend
                .get_function_pointer(*new_id)
                .ok_or_else(|| RuntimeError::FunctionNotFound(new_name.clone()))?;
            let poll_ptr = self
                .backend
                .get_function_pointer(*poll_id)
                .ok_or_else(|| RuntimeError::FunctionNotFound(poll_name.clone()))?;

            let dynamic_args: Vec<DynamicValue> =
                args.iter().cloned().map(|v| v.into_dynamic()).collect();

            return Ok(ZyntaxPromise::with_poll_fn(new_ptr, poll_ptr, dynamic_args));
        }

        Err(RuntimeError::FunctionNotFound(format!(
            "Async function '{}' not found (tried both Promise-returning and legacy _new/_poll APIs)", name
        )))
    }

    /// Manually optimize a function to a specific tier
    ///
    /// Useful for pre-warming hot paths or testing.
    pub fn optimize_function(&mut self, name: &str, tier: OptimizationTier) -> RuntimeResult<()> {
        let func_id = self
            .function_ids
            .get(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;

        self.backend.optimize_function(*func_id, tier)?;
        Ok(())
    }

    /// Get optimization statistics
    pub fn statistics(&self) -> TieredStatistics {
        self.backend.get_statistics()
    }

    /// Get the current optimization tier for a function
    pub fn function_tier(&self, name: &str) -> Option<OptimizationTier> {
        // Implementation would query the backend's function_tiers map
        // For now, return None as this requires backend API access
        let _ = name;
        None
    }

    /// Get the tiered configuration
    pub fn config(&self) -> &TieredConfig {
        &self.config
    }

    /// Shutdown the runtime (stops background optimization). Frees any
    /// host-driven fibers still registered — their stacks and handler
    /// segments do not outlive the runtime that owns them.
    pub fn shutdown(&mut self) {
        for (_, hf) in self.host_fibers.drain() {
            let ptr = hf.ptr as *mut u8;
            crate::effect_runtime::__zyntax_effect_fiber_forget(ptr);
            // SAFETY: the registry owned the handle exclusively.
            unsafe { zyntax_compiler::zrtl::krio_fiber_free(ptr as *mut _) };
        }
        // Nothing can be installed any more, so every region the
        // runtime is holding goes, whether or not its owner got round
        // to dropping it.
        for (_, e) in self.handler_instances.drain() {
            // SAFETY: the fibers that could have named these are freed
            // above and the thread's handler stack cannot outlive the
            // runtime that owns the code its frames point into.
            unsafe { crate::effect_runtime::free_handler_state(e.state as *mut u8) };
        }
        self.backend.shutdown();
    }

    /// Load a ZRTL plugin from a file path
    ///
    /// This loads a native dynamic library (.zrtl, .so, .dylib, .dll) and
    /// registers all its exported symbols as external functions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// runtime.load_plugin("./my_runtime.zrtl")?;
    /// ```
    pub fn load_plugin<P: AsRef<std::path::Path>>(&mut self, path: P) -> RuntimeResult<()> {
        use zyntax_compiler::zrtl::ZrtlPlugin;

        let plugin = ZrtlPlugin::load(path).map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // Register all symbols from the plugin as runtime symbols
        // AND collect their signatures for type checking
        for symbol_info in plugin.symbols_with_signatures() {
            self.backend
                .register_runtime_symbol(symbol_info.name, symbol_info.ptr);

            // Store signature if available
            if let Some(sig) = symbol_info.sig {
                self.plugin_signatures
                    .insert(symbol_info.name.to_string(), sig);
            }
        }

        // Register symbol signatures for auto-boxing in the Cranelift
        // backend. Without this the backend doesn't know plugin functions
        // like `$IO$println_dynamic` expect a `DynamicBox`, so it passes
        // raw values through and the callee mis-reads them as fat-pointer
        // bytes. Mirrors `ZyntaxRuntime::load_plugin`.
        self.backend
            .register_symbol_signatures(plugin.symbols_with_signatures());

        // Push the new symbols into the live JIT module so finalization
        // can resolve them. Mirrors `ZyntaxRuntime::load_plugin`.
        self.backend
            .rebuild_with_accumulated_symbols()
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        self.loaded_plugins.push(plugin);
        Ok(())
    }

    /// Load all ZRTL plugins from a directory
    ///
    /// Loads all `.zrtl` files from the specified directory.
    ///
    /// # Returns
    ///
    /// The number of plugins loaded successfully.
    pub fn load_plugins_from_directory<P: AsRef<std::path::Path>>(
        &mut self,
        dir: P,
    ) -> RuntimeResult<usize> {
        use zyntax_compiler::zrtl::ZrtlRegistry;

        let mut registry = ZrtlRegistry::new();
        let count = registry
            .load_directory(&dir)
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // Register all collected symbols AND their signatures
        for symbol_info in registry.collect_symbols_with_signatures() {
            self.backend
                .register_runtime_symbol(symbol_info.name, symbol_info.ptr);

            // Store signature if available
            if let Some(sig) = symbol_info.sig {
                self.plugin_signatures
                    .insert(symbol_info.name.to_string(), sig);
            }
        }

        // Register signatures for auto-boxing in the Cranelift backend.
        let symbols_with_sigs = registry.collect_symbols_with_signatures();
        self.backend.register_symbol_signatures(&symbols_with_sigs);

        // Push the new symbols into the live JIT module so finalization
        // can resolve them. Mirrors `ZyntaxRuntime::load_plugins_from_directory`.
        self.backend
            .rebuild_with_accumulated_symbols()
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        Ok(count)
    }

    /// Get plugin signatures for all loaded plugins
    ///
    /// Returns a reference to the mapping of symbol names to ZRTL signatures.
    /// This can be used during parsing to create properly typed extern function declarations.
    ///
    /// # Returns
    ///
    /// A HashMap mapping symbol names (e.g., "$IO$println_dynamic") to their ZRTL signatures.
    pub fn plugin_signatures(&self) -> &HashMap<String, zyntax_compiler::zrtl::ZrtlSymbolSig> {
        &self.plugin_signatures
    }

    /// Register a runtime event sink callback.
    pub fn set_event_sink<F>(&mut self, sink: F)
    where
        F: Fn(&RuntimeEvent) + Send + Sync + 'static,
    {
        self.event_sink = Some(Arc::new(sink));
    }

    /// Clear the runtime event sink callback.
    pub fn clear_event_sink(&mut self) {
        self.event_sink = None;
    }

    /// View captured runtime events.
    pub fn runtime_events(&self) -> &[RuntimeEvent] {
        &self.runtime_events
    }

    /// Drain and return captured runtime events.
    pub fn drain_runtime_events(&mut self) -> Vec<RuntimeEvent> {
        std::mem::take(&mut self.runtime_events)
    }

    // ========================================================================
    // Multi-Language Grammar Registry
    // ========================================================================
    /// Install a language: its grammar, its standard library, and the
    /// type ids both were built against.
    ///
    /// This is the whole of what a language does to become usable in a
    /// runtime. Decoding the snapshot's modules reserves their
    /// build-time type ids, and doing it here, in the order the build
    /// wrote them, is why a language no longer has to remember to do
    /// it before parsing anything.
    ///
    /// Hold the snapshot for the process and install the same one into
    /// however many runtimes get built. Decoding happens once for the
    /// snapshot rather than once per runtime.
    ///
    /// Returns the grammar it registered, so a host that wants one of
    /// its own does not decode the same bytes twice.
    ///
    /// Plugins can load before or after this. A grammar's builtin
    /// names are recorded as the names they stand for and resolved
    /// when something asks, so nothing here depends on a plugin having
    /// arrived first.
    pub fn install_snapshot(
        &mut self,
        snapshot: Arc<crate::Snapshot>,
    ) -> RuntimeResult<LanguageGrammar> {
        let grammar =
            LanguageGrammar::from_compiled_bytes(snapshot.grammar_bytes()).map_err(|e| {
                RuntimeError::Execution(format!(
                    "snapshot for '{}' has an unreadable grammar: {e}",
                    snapshot.language()
                ))
            })?;
        let mut grammar = grammar;
        grammar.set_language(snapshot.language());
        self.register_grammar(snapshot.language(), grammar.clone());

        // Reserve the ids before anything can parse against them. The
        // build recorded what to reserve, so no module is decoded here
        // and one nobody imports is never decoded at all.
        snapshot.reserve_type_ids();

        for module in snapshot.module_names() {
            self.snapshot_modules.insert(
                (snapshot.language().to_string(), module.to_string()),
                Arc::clone(&snapshot),
            );
        }

        // A module that kept its source stays available to hosts that
        // would rather parse it than trust the artifact.
        let sources = snapshot;
        self.add_import_resolver(Box::new(move |module_name| {
            Ok(sources.module_source(module_name).map(str::to_string))
        }));

        Ok(grammar)
    }

    /// Register a language grammar with the runtime
    ///
    /// See `ZyntaxRuntime::register_grammar` for full documentation.
    pub fn register_grammar(&mut self, language: &str, grammar: LanguageGrammar) {
        // Note: Builtin aliases are resolved during parsing via ZynPEG's builtin resolution.
        // The @builtin section in the grammar maps DSL names (e.g., "image_load") to
        // runtime symbols (e.g., "$Image$load"). This resolution happens in the parser,
        // not here in the runtime.
        let grammar = Arc::new(grammar);

        // Register file extensions from grammar metadata
        for ext in grammar.file_extensions() {
            let ext_key = if ext.starts_with('.') {
                ext.clone()
            } else {
                format!(".{}", ext)
            };
            self.extension_map.insert(ext_key, language.to_string());
        }

        self.grammars.insert(language.to_string(), grammar);
    }

    /// Register a grammar from a .zyn file
    pub fn register_grammar_file<P: AsRef<Path>>(
        &mut self,
        language: &str,
        zyn_path: P,
    ) -> Result<(), GrammarError> {
        let grammar = LanguageGrammar::compile_zyn_file(zyn_path)?;
        self.register_grammar(language, grammar);
        Ok(())
    }

    /// Register a grammar from a pre-compiled .zpeg file
    pub fn register_grammar_zpeg<P: AsRef<Path>>(
        &mut self,
        language: &str,
        zpeg_path: P,
    ) -> Result<(), GrammarError> {
        let grammar = LanguageGrammar::load(zpeg_path)?;
        self.register_grammar(language, grammar);
        Ok(())
    }

    /// Get a registered grammar by language name
    pub fn get_grammar(&self, language: &str) -> Option<&Arc<LanguageGrammar>> {
        self.grammars.get(language)
    }

    /// Get the language name for a file extension
    pub fn language_for_extension(&self, extension: &str) -> Option<&str> {
        let ext_key = if extension.starts_with('.') {
            extension.to_string()
        } else {
            format!(".{}", extension)
        };
        self.extension_map.get(&ext_key).map(|s| s.as_str())
    }

    /// List all registered language names
    pub fn languages(&self) -> Vec<&str> {
        self.grammars.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a language grammar is registered
    pub fn has_language(&self, language: &str) -> bool {
        self.grammars.contains_key(language)
    }

    /// Load a module from source code using a registered language grammar
    ///
    /// See `ZyntaxRuntime::load_module` for full documentation.
    /// Register a host function with its call signature, mirroring
    /// [`ZyntaxRuntime::register_function_typed`]: the symbol reaches
    /// the JIT's resolution table and the signature reaches call-site
    /// lowering and the parser's extern declarations.
    pub fn register_function_typed(
        &mut self,
        name: &'static str,
        ptr: *const u8,
        sig: zyntax_compiler::zrtl::ZrtlSymbolSig,
    ) {
        self.backend.register_runtime_symbol(name, ptr);
        self.plugin_signatures.insert(name.to_string(), sig);
        let info = zyntax_compiler::zrtl::RuntimeSymbolInfo {
            name,
            ptr,
            sig: Some(sig),
        };
        self.backend.register_symbol_signatures(&[info]);
    }

    /// Publish host functions registered through
    /// [`Self::register_function_typed`] into the live tier-0 JIT module.
    ///
    /// Registering is deliberately batchable: embedders add every host
    /// symbol first, call this once, and only then compile a typed program
    /// that declares those externs. This mirrors
    /// [`ZyntaxRuntime::finalize_runtime_symbols`].
    pub fn finalize_runtime_symbols(&mut self) -> RuntimeResult<()> {
        self.backend
            .rebuild_with_accumulated_symbols()
            .map_err(|error| RuntimeError::Execution(error.to_string()))
    }

    /// Compile a pre-parsed typed program, mirroring
    /// [`ZyntaxRuntime::compile_typed_program`]. This is the path a
    /// `Grammar2`-based frontend takes; `load_module` covers the
    /// interpreter-grammar one.
    pub fn compile_typed_program(
        &mut self,
        mut program: zyntax_typed_ast::TypedProgram,
    ) -> RuntimeResult<Vec<String>> {
        capture_runtime_events_from_program(
            &mut program,
            &mut self.runtime_events,
            self.event_sink.as_ref(),
        );
        let fiber_decls = collect_fiber_decls(&program);
        let mut hir_module = self.lower_typed_program(program, self.builtin_aliases.clone())?;
        apply_krio_async_lowering(&mut hir_module)?;
        apply_krio_effect_lowering(&mut hir_module)?;
        apply_krio_fiber_lowering(&mut hir_module);

        let function_names: Vec<String> = hir_module
            .functions
            .values()
            .filter(|f| !f.is_external)
            .filter_map(|f| f.name.resolve_global())
            .collect();

        self.compile_module(hir_module)?;
        let _ = self.apply_fiber_decls(fiber_decls);
        Ok(function_names)
    }

    /// Reload a pre-parsed typed program against the running module —
    /// the typed-program twin of [`Self::reload_module_source`].
    pub fn reload_typed_program(
        &mut self,
        mut program: zyntax_typed_ast::TypedProgram,
    ) -> RuntimeResult<zyntax_compiler::reload::ReloadReport> {
        capture_runtime_events_from_program(
            &mut program,
            &mut self.runtime_events,
            self.event_sink.as_ref(),
        );
        let fiber_decls = collect_fiber_decls(&program);
        let mut hir_module = self.lower_typed_program(program, self.builtin_aliases.clone())?;
        apply_krio_async_lowering(&mut hir_module)?;
        apply_krio_effect_lowering(&mut hir_module)?;
        apply_krio_fiber_lowering(&mut hir_module);
        if std::env::var("ZYNTAX_DISABLE_INTERP_OPTS").is_err() {
            let _stats = zyntax_compiler::run_interp_safe_opts(&mut hir_module);
        }
        zyntax_compiler::hir_dump::dump_module_to_dir(&hir_module, "post-opt-typed-program");
        let report = self
            .backend
            .reload_module(&hir_module)
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // An aborted reload changed nothing, so the handles' view of
        // shapes and machines must not move either.
        if !report.aborted {
            self.apply_reload_fiber_meta(fiber_decls, &report);
            if !report.state_migrations.is_empty() {
                let plans = report.state_migrations.clone();
                self.apply_state_migrations(&plans);
            }
        }

        // Reload is an observable event: frameworks subscribe to
        // invalidate whatever the edit touched.
        let event = RuntimeEvent::Reload {
            reloaded: report.reloaded.clone(),
            added: report.added.clone(),
            dispatch_patched: report.dispatch_patched.clone(),
            failed: report.failed.clone(),
        };
        if let Some(sink) = &self.event_sink {
            sink(&event);
        }
        self.runtime_events.push(event);

        Ok(report)
    }

    /// Choose what a reload does with live handler state whose layout
    /// an edit changed: keep the previous implementation (the default),
    /// or move the fields the two layouts share into a region the
    /// edited constructor allocates.
    pub fn set_state_migration(&mut self, policy: zyntax_compiler::reload::StateMigration) {
        self.backend.set_state_migration(policy);
    }

    /// Move every live region of the handlers a reload planned
    /// migrations for. Returns how many regions moved.
    fn apply_state_migrations(
        &mut self,
        plans: &[zyntax_compiler::reload::StateMigrationPlan],
    ) -> usize {
        let mut total = 0;
        for plan in plans {
            let Some(&ctor_id) = self.function_ids.get(&plan.ctor) else {
                continue;
            };
            let Some(ctor_ptr) = self.backend.get_function_pointer(ctor_id) else {
                continue;
            };
            // SAFETY: a synthesized handler constructor is `(): *state`.
            let ctor: extern "C" fn() -> *mut u8 = unsafe { std::mem::transmute(ctor_ptr) };

            // A host-owned instance that is not installed anywhere is
            // named by no frame, so the live-frame walk below cannot
            // see it. Migrate the registry's own regions first, and
            // record the moves so a frame still holding one is updated
            // to the same replacement rather than a second copy.
            let mut moved: HashMap<usize, usize> = HashMap::new();
            for entry in self.handler_instances.values_mut() {
                if entry.effect_id != plan.effect_id || entry.state == 0 {
                    continue;
                }
                if let Some(fresh) = migrate_one(&plan.moves, ctor, entry.state as *mut u8) {
                    moved.insert(entry.state, fresh as usize);
                    entry.state = fresh as usize;
                    total += 1;
                }
            }

            total += crate::effect_runtime::migrate_handler_states(plan.effect_id, |old_state| {
                if let Some(already) = moved.get(&(old_state as usize)) {
                    return Some(*already as *mut u8);
                }
                migrate_one(&plan.moves, ctor, old_state)
            });
        }
        total
    }

    /// Restore the generation the most recent applied reload replaced.
    /// The embedder's escape hatch when an edit turns out wrong at
    /// runtime: beads, reload cells, resume points and dispatch tables
    /// all swing back; state is untouched, exactly as in a reload.
    /// Returns the restored function names, and emits the same
    /// observable event a reload does — rolling back is a code change
    /// too, and a subscribed framework must invalidate again.
    pub fn rollback_last_reload(&mut self) -> RuntimeResult<Vec<String>> {
        let restored = self
            .backend
            .rollback_last_reload()
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // The handles' view rolls back with the code: shape entries
        // (and their generations) return to their prior state, and
        // tokens this reload marked gone resume again.
        if let Some(undo) = self.fiber_meta_undo.take() {
            for (name, prior) in undo.shapes.into_iter().rev() {
                match prior {
                    Some(entry) => {
                        self.fiber_shapes.insert(name, entry);
                    }
                    None => {
                        self.fiber_shapes.remove(&name);
                    }
                }
            }
            for token in undo.marked_gone {
                if let Some(hf) = self.host_fibers.get_mut(&token) {
                    hf.machine_gone = false;
                }
            }
        }

        let event = RuntimeEvent::Reload {
            reloaded: restored.clone(),
            added: Vec::new(),
            dispatch_patched: Vec::new(),
            failed: Vec::new(),
        };
        if let Some(sink) = &self.event_sink {
            sink(&event);
        }
        self.runtime_events.push(event);

        Ok(restored)
    }

    // ── Host-driven fibers ──────────────────────────────────────────
    //
    // A downstream framework drives FSMs from outside the language:
    // it constructs a machine from a compiled `fiber def`, holds a
    // token, and steps the machine on its own schedule — with effect
    // handlers installed around each step when the machine observes
    // events. Tokens survive reloads and OSR; the edge cases an edit
    // creates (function deleted, yield shape changed) surface as
    // values and handle metadata, never as traps.

    /// Get a runtime-owned instance of the machine the `fiber def`
    /// named `function` declares, as a token the host drives.
    ///
    /// Each call hands back a FRESH paused instance — two tokens are
    /// two independent machines. The machine must be parameterless:
    /// host-driven FSMs take their inputs through effects, which is
    /// what [`Self::resume_fiber_within`] installs handlers for.
    pub fn get_fiber(&mut self, function: &str) -> RuntimeResult<FiberToken> {
        // FQN-aware: `machine` finds `app::machine` when unambiguous;
        // a qualified name is exact.
        let function = &resolve_scoped_name(function, self.fiber_shapes.keys().map(String::as_str))
            .map_err(|candidates| {
                if candidates.is_empty() {
                    RuntimeError::Execution(format!(
                        "`{function}` is not a fiber function in the loaded program"
                    ))
                } else {
                    RuntimeError::Execution(format!(
                        "`{function}` is ambiguous; qualify it: {candidates:?}"
                    ))
                }
            })?;
        let shape = self.fiber_shapes.get(function).ok_or_else(|| {
            RuntimeError::Execution(format!(
                "`{function}` is not a fiber function in the loaded program"
            ))
        })?;
        let id = *self
            .function_ids
            .get(function)
            .ok_or_else(|| RuntimeError::FunctionNotFound(function.to_string()))?;
        if self
            .function_signatures
            .get(function)
            .map(|s| !s.params.is_empty())
            .unwrap_or(false)
        {
            return Err(RuntimeError::Execution(format!(
                "`{function}` takes parameters; a host-constructed machine must be \
                 parameterless — feed it through effects instead"
            )));
        }
        let entry = self.backend.get_function_pointer(id).ok_or_else(|| {
            RuntimeError::Execution(format!("no compiled entry for `{function}`"))
        })?;
        // SAFETY: `entry` is the compiled trampoline of a parameterless
        // `fiber def`; the installed fiber backend interprets the
        // closure pointer as exactly that.
        let ptr = unsafe { zyntax_compiler::zrtl::krio_fiber_new(entry as *mut u8, 0) };
        if ptr.is_null() {
            return Err(RuntimeError::Execution(format!(
                "fiber construction failed for `{function}`"
            )));
        }
        let token = self.next_fiber_token;
        self.next_fiber_token += 1;
        self.host_fibers.insert(
            token,
            HostFiber {
                ptr: ptr as usize,
                bound_instances: Vec::new(),
                function: function.to_string(),
                yield_shape: shape.shape.clone(),
                yield_kind: shape.kind,
                shape_generation: shape.generation,
                machine_gone: false,
                done: false,
            },
        );
        Ok(FiberToken(token))
    }

    /// Drive the machine one step: run to its next yield or completion.
    pub fn resume_fiber(&mut self, token: FiberToken) -> RuntimeResult<HostFiberStep> {
        self.resume_fiber_within(token, &[])
    }

    /// Drive the machine one step with handler scopes installed around
    /// it — the host equivalent of wrapping the resume in `with H { }`
    /// blocks, leftmost outermost. The machine's own handler segment
    /// layers on top, so its interior scopes keep precedence.
    ///
    /// Names resolve per call (FQN-aware). For a binding that cannot
    /// drift across edits, resolve once with [`Self::get_effect_handler`] and
    /// drive with [`Self::resume_fiber_handled`].
    pub fn resume_fiber_within(
        &mut self,
        token: FiberToken,
        handlers: &[&str],
    ) -> RuntimeResult<HostFiberStep> {
        let (ptr, kind) = {
            let hf = self.host_fibers.get(&token.0).ok_or_else(|| {
                RuntimeError::Execution("unknown or dropped fiber token".to_string())
            })?;
            if hf.machine_gone {
                return Ok(HostFiberStep::MachineGone);
            }
            if hf.done {
                return Ok(HostFiberStep::Done);
            }
            (hf.ptr as *mut u8, hf.yield_kind)
        };

        let mut frames: Vec<u64> = Vec::with_capacity(handlers.len());
        for h in handlers {
            match self.push_named_handler(h) {
                Ok(frame) => frames.push(frame),
                Err(e) => {
                    // A partial install must not leak: unwind the
                    // frames already pushed before surfacing the error.
                    for frame in frames.into_iter().rev() {
                        crate::effect_runtime::__zyntax_effect_pop_handler(frame);
                    }
                    return Err(e);
                }
            }
        }
        let baseline = crate::effect_runtime::__zyntax_effect_fiber_enter(ptr);
        // SAFETY: `ptr` is a live handle owned by this registry; the
        // enter/leave bracket mirrors generated `FiberResume` lowering.
        let raw = unsafe { zyntax_compiler::zrtl::krio_fiber_resume(ptr as *mut _) };
        crate::effect_runtime::__zyntax_effect_fiber_leave(ptr, baseline);
        for frame in frames.into_iter().rev() {
            crate::effect_runtime::__zyntax_effect_pop_handler(frame);
        }

        use zyntax_compiler::fiber_backend::{
            unpack_fiber_step, FIBER_STEP_DONE, FIBER_STEP_YIELDED,
        };
        let (tag, payload) = unpack_fiber_step(raw);
        let step = if tag == FIBER_STEP_YIELDED {
            HostFiberStep::Yielded(decode_yield(kind, payload))
        } else if tag == FIBER_STEP_DONE {
            HostFiberStep::Done
        } else {
            HostFiberStep::Errored
        };
        if !matches!(step, HostFiberStep::Yielded(_)) {
            if let Some(hf) = self.host_fibers.get_mut(&token.0) {
                hf.done = true;
            }
        }
        Ok(step)
    }

    /// Call a compiled function with an explicit native signature.
    ///
    /// The signature is what makes the call safe: it names the argument
    /// and return representation, so the pointer is invoked through a
    /// matching ABI rather than an inferred one. [`Self::call_raw`]
    /// infers, and cannot for every function.
    ///
    /// Mirrors `ZyntaxRuntime::call_function`.
    pub fn call_function(
        &self,
        name: &str,
        args: &[ZyntaxValue],
        signature: &NativeSignature,
    ) -> RuntimeResult<ZyntaxValue> {
        if args.len() != signature.params.len() {
            return Err(RuntimeError::Execution(format!(
                "Function '{name}' expects {} arguments, got {}",
                signature.params.len(),
                args.len()
            )));
        }
        let ptr = self
            .function_pointer(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;
        // SAFETY: `ptr` is a compiled entry point owned by this runtime,
        // and `signature` is the caller's statement of its ABI — the same
        // contract the classic runtime's `call_function` carries.
        unsafe { call_native_with_signature(ptr, args, signature) }
    }

    /// Publish `name` as a symbol later modules can link against.
    ///
    /// Registered as a runtime symbol rather than into one tier's export
    /// table: every tier that can call the symbol has to be able to
    /// resolve it, and a function exported while cold may be running
    /// from a higher tier by the time something links to it.
    ///
    /// Mirrors `ZyntaxRuntime::export_function`.
    pub fn export_function(&mut self, name: &str) -> RuntimeResult<()> {
        let ptr = self
            .function_pointer(name)
            .ok_or_else(|| RuntimeError::FunctionNotFound(name.to_string()))?;
        self.backend.register_runtime_symbol(name, ptr);
        Ok(())
    }

    /// Install `handler` for a dynamic extent the HOST controls, and
    /// return the frame that ends it.
    ///
    /// The fiber entry points scope a handler around a machine step.
    /// This scopes one around whatever the host does next — a render, a
    /// query, any plain call — which is what an embedder needs when the
    /// code that performs is not a fiber.
    ///
    /// Every frame must be handed to [`Self::pop_effect_handler`], in
    /// reverse order of installation. Leaving one installed leaks it
    /// into unrelated work on the same thread.
    ///
    /// Each call allocates its own state, so successive extents do not
    /// share one and neither does a machine bound to the same handler.
    /// Use [`Self::push_handler_instance`] when they should.
    pub fn push_effect_handler(
        &mut self,
        handler: EffectHandlerToken,
    ) -> RuntimeResult<HandlerFrame> {
        // Allocate through an instance the runtime owns and immediately
        // disowns: nothing else can ever name this region, so it dies
        // with the frame rather than leaking.
        let instance = self.new_handler_instance(handler)?;
        let frame = self.push_handler_instance(instance)?;
        if let Some(e) = self.handler_instances.get_mut(&instance.0) {
            e.dropped_by_owner = true;
        }
        Ok(frame)
    }

    /// End the extent a [`Self::push_effect_handler`] frame opened, and
    /// release the install this frame held on its handler state.
    pub fn pop_effect_handler(&mut self, frame: HandlerFrame) {
        crate::effect_runtime::__zyntax_effect_pop_handler(frame.0);
        if let Some(id) = frame.1 {
            self.release_handler_install(id);
        }
    }

    /// Record the handler context in force right now, to reinstate
    /// around a callback that runs later.
    ///
    /// A host that stores a zero-argument function and calls it on its
    /// own schedule — a reactive flush, an input event — needs the
    /// handlers that were installed where that function was written.
    /// By the time it runs, the extent that installed them has usually
    /// closed and nothing is in scope, so a perform would find no
    /// handler.
    ///
    /// The context claims an install on every handler instance it
    /// names, so the state stays alive for as long as the context does
    /// even after the original extent ends. Hand it back to
    /// [`Self::release_handler_context`] to give those claims up.
    ///
    /// Captures the calling thread's stack, and
    /// [`Self::enter_handler_context`] must run on that same thread:
    /// the stack is thread-local and the state pointers are not `Send`.
    pub fn capture_handler_context(&mut self) -> HandlerContext {
        let frames = crate::effect_runtime::capture_handler_frames();
        // Claim an install per instance whose state a frame names, so
        // the region outlives the extent that pushed it. A frame whose
        // state the runtime did not allocate (a `with` block's, whose
        // lifetime the compiled code owns) has no instance to claim.
        let mut instances = Vec::new();
        for frame in &frames {
            let state = frame.handler_state as usize;
            if let Some((id, entry)) = self
                .handler_instances
                .iter_mut()
                .find(|(_, e)| e.state == state)
            {
                entry.installs += 1;
                instances.push(*id);
            }
        }
        HandlerContext { frames, instances }
    }

    /// Reinstate a captured context, returning the scope to close.
    ///
    /// Layers on top of whatever is installed rather than replacing it,
    /// so a callback that captures a context of its own nests the way a
    /// resumed fiber does.
    ///
    /// Takes `&self` and returns before the body runs, so the caller
    /// holds no borrow of the runtime across the callback. That matters
    /// because the body is compiled code that may call host externs
    /// which re-enter the runtime.
    pub fn enter_handler_context(&self, context: &HandlerContext) -> HandlerContextScope {
        HandlerContextScope(crate::effect_runtime::enter_handler_frames(&context.frames))
    }

    /// Close a scope [`Self::enter_handler_context`] opened, restoring
    /// the stack the caller had. A body that left a frame open does not
    /// strand it on the caller's stack.
    pub fn leave_handler_context(&self, scope: HandlerContextScope) {
        crate::effect_runtime::leave_handler_frames(scope.0);
    }

    /// Give up the installs a context claimed. The handler state it
    /// named is freed once nothing else names it and its owner has let
    /// it go, on the same terms as any other install.
    pub fn release_handler_context(&mut self, context: HandlerContext) {
        for id in context.instances {
            self.release_handler_install(id);
        }
    }

    /// Allocate ONE instance of a handler's state and hand back a
    /// handle to it.
    ///
    /// This is the seam that lets a single state back more than one
    /// install. [`Self::push_effect_handler`] and
    /// [`Self::bind_fiber_handler`] each allocate their own, so a
    /// machine and the host reading after it see different storage;
    /// create an instance here instead and install THAT in both places
    /// with [`Self::bind_fiber_handler_instance`] and
    /// [`Self::push_handler_instance`].
    ///
    /// A stateless handler has no state to share, but still gets an
    /// instance so callers need not special-case it.
    ///
    /// The runtime owns the instance and every install borrows it, so
    /// dropping a fiber never invalidates one. Handler state is not
    /// reclaimed today — the same is true of the state a `with H { }`
    /// scope allocates — so an instance's region lives as long as the
    /// runtime; [`Self::drop_handler_instance`] releases the handle,
    /// not the memory.
    pub fn new_handler_instance(
        &mut self,
        handler: EffectHandlerToken,
    ) -> RuntimeResult<HandlerInstance> {
        let name = self
            .handler_tokens
            .get(&handler.0)
            .cloned()
            .ok_or_else(|| RuntimeError::Execution("unknown handler token".to_string()))?;
        let (resolved, effect_id, table, async_mask, stateful) = self.handler_shape(&name)?;
        let state = self.alloc_handler_state(&resolved, stateful)?;
        let id = self.next_handler_instance;
        self.next_handler_instance += 1;
        self.handler_instances.insert(
            id,
            HandlerInstanceEntry {
                handler: resolved,
                effect_id,
                state: state as usize,
                table: table as usize,
                async_mask,
                installs: 0,
                dropped_by_owner: false,
            },
        );
        Ok(HandlerInstance(id))
    }

    /// Install `instance` for a host-controlled extent, the way
    /// [`Self::push_effect_handler`] does but against state the caller
    /// already owns. Pair with [`Self::pop_effect_handler`].
    pub fn push_handler_instance(
        &mut self,
        instance: HandlerInstance,
    ) -> RuntimeResult<HandlerFrame> {
        let e = self.handler_instance_entry(instance)?;
        let (effect_id, state, table, async_mask) = (
            e.effect_id,
            e.state as *mut u8,
            e.table as *mut u8,
            e.async_mask,
        );
        let frame = crate::effect_runtime::__zyntax_effect_push_handler(
            effect_id, state, table, async_mask,
        );
        if let Some(e) = self.handler_instances.get_mut(&instance.0) {
            e.installs += 1;
        }
        Ok(HandlerFrame(frame, Some(instance.0)))
    }

    /// Bind `instance` to a machine for its lifetime, the way
    /// [`Self::bind_fiber_handler`] does but against state the caller
    /// already owns, so the same region is readable from a pushed
    /// frame afterwards.
    pub fn bind_fiber_handler_instance(
        &mut self,
        token: FiberToken,
        instance: HandlerInstance,
    ) -> RuntimeResult<()> {
        let ptr = {
            let hf = self.host_fibers.get(&token.0).ok_or_else(|| {
                RuntimeError::Execution("unknown or dropped fiber token".to_string())
            })?;
            hf.ptr as *mut u8
        };
        let e = self.handler_instance_entry(instance)?;
        let (effect_id, state, table, async_mask) = (
            e.effect_id,
            e.state as *mut u8,
            e.table as *mut u8,
            e.async_mask,
        );
        crate::effect_runtime::fiber_bind_handler(ptr, effect_id, state, table, async_mask);
        if let Some(e) = self.handler_instances.get_mut(&instance.0) {
            e.installs += 1;
        }
        if let Some(hf) = self.host_fibers.get_mut(&token.0) {
            hf.bound_instances.push(instance.0);
        }
        Ok(())
    }

    /// The handler an instance is an instance OF, fully qualified.
    pub fn handler_instance_name(&self, instance: HandlerInstance) -> Option<&str> {
        self.handler_instances
            .get(&instance.0)
            .map(|e| e.handler.as_str())
    }

    /// Give up ownership of the instance. The region is released once
    /// nothing is installed against it: a frame still open or a machine
    /// still bound keeps it alive until that install ends, so this is
    /// safe to call while either is outstanding.
    pub fn drop_handler_instance(&mut self, instance: HandlerInstance) {
        if let Some(e) = self.handler_instances.get_mut(&instance.0) {
            e.dropped_by_owner = true;
        }
        self.reap_handler_instance(instance.0);
    }

    /// Drop one install's claim, freeing the region if it was the last
    /// and the owner is done with it.
    fn release_handler_install(&mut self, id: u64) {
        if let Some(e) = self.handler_instances.get_mut(&id) {
            e.installs = e.installs.saturating_sub(1);
        }
        self.reap_handler_instance(id);
    }

    /// Free and forget an instance once no install names it and its
    /// owner has let it go.
    fn reap_handler_instance(&mut self, id: u64) {
        let ready = self
            .handler_instances
            .get(&id)
            .map(|e| e.dropped_by_owner && e.installs == 0)
            .unwrap_or(false);
        if !ready {
            return;
        }
        if let Some(e) = self.handler_instances.remove(&id) {
            // SAFETY: no frame names this region — every install has
            // been released — and it came from the handler's
            // constructor, so libc `free` is the matching release.
            unsafe { crate::effect_runtime::free_handler_state(e.state as *mut u8) };
        }
    }

    fn handler_instance_entry(
        &self,
        instance: HandlerInstance,
    ) -> RuntimeResult<&HandlerInstanceEntry> {
        self.handler_instances
            .get(&instance.0)
            .ok_or_else(|| RuntimeError::Execution("unknown handler instance".to_string()))
    }

    /// Resolve a handler name ONCE — FQN-aware, ambiguity is an error —
    /// and pin the result as a token. The token stays aimed at exactly
    /// that handler no matter what names later edits introduce; use it
    /// with [`Self::resume_fiber_handled`] and
    /// [`Self::bind_fiber_handler`].
    pub fn get_effect_handler(&mut self, name: &str) -> RuntimeResult<EffectHandlerToken> {
        let (resolved, _, _, _, _) = self
            .backend
            .try_handler_push_info(name)
            .map_err(RuntimeError::Execution)?;
        let token = self.next_handler_token;
        self.next_handler_token += 1;
        self.handler_tokens.insert(token, resolved);
        Ok(EffectHandlerToken(token))
    }

    /// The fully qualified name a handler token is pinned to.
    pub fn effect_handler_name(&self, token: EffectHandlerToken) -> Option<&str> {
        self.handler_tokens.get(&token.0).map(String::as_str)
    }

    /// [`Self::resume_fiber_within`] with pinned handler tokens instead
    /// of per-call name resolution.
    pub fn resume_fiber_handled(
        &mut self,
        token: FiberToken,
        handlers: &[EffectHandlerToken],
    ) -> RuntimeResult<HostFiberStep> {
        let names: Vec<String> = handlers
            .iter()
            .map(|h| {
                self.handler_tokens
                    .get(&h.0)
                    .cloned()
                    .ok_or_else(|| RuntimeError::Execution("unknown handler token".to_string()))
            })
            .collect::<RuntimeResult<_>>()?;
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        self.resume_fiber_within(token, &name_refs)
    }

    /// Free the machine and forget its token — what a UI does when the
    /// component unmounts, or when an edit made the machine stale and
    /// it chooses to remount. Frees the fiber's stack and its saved
    /// handler segment; the token is dead afterwards.
    pub fn drop_fiber(&mut self, token: FiberToken) -> RuntimeResult<()> {
        let hf = self
            .host_fibers
            .remove(&token.0)
            .ok_or_else(|| RuntimeError::Execution("unknown or dropped fiber token".to_string()))?;
        let ptr = hf.ptr as *mut u8;
        crate::effect_runtime::__zyntax_effect_fiber_forget(ptr);
        // SAFETY: the registry owned this handle exclusively; nothing
        // holds it after removal.
        unsafe { zyntax_compiler::zrtl::krio_fiber_free(ptr as *mut _) };
        // The machine's segment is gone, so its binds are no longer
        // installed anywhere.
        for id in hf.bound_instances {
            self.release_handler_install(id);
        }
        Ok(())
    }

    /// What the runtime knows about a host-driven fiber, including the
    /// staleness signals a reload leaves on the handle.
    pub fn fiber_info(&self, token: FiberToken) -> Option<HostFiberInfo> {
        let hf = self.host_fibers.get(&token.0)?;
        let shape_stale = self
            .fiber_shapes
            .get(&hf.function)
            .map(|s| s.generation != hf.shape_generation)
            .unwrap_or(false);
        Some(HostFiberInfo {
            function: hf.function.clone(),
            yield_shape: hf.yield_shape.clone(),
            shape_generation: hf.shape_generation,
            shape_stale,
            machine_gone: hf.machine_gone,
            done: hf.done,
        })
    }

    /// Everything about a handler that does NOT depend on which
    /// instance of its state you mean: `(resolved name, effect id,
    /// op table, async mask, stateful?)`.
    fn handler_shape(&self, handler: &str) -> RuntimeResult<(String, u64, *mut u8, u64, bool)> {
        let (resolved, effect_id, table_addr, async_mask, stateful) = self
            .backend
            .try_handler_push_info(handler)
            .map_err(RuntimeError::Execution)?;
        Ok((
            resolved,
            effect_id,
            table_addr as *mut u8,
            async_mask,
            stateful,
        ))
    }

    /// Run a stateful handler's synthesized constructor to allocate one
    /// state region. Null for a stateless handler, which has none.
    fn alloc_handler_state(&self, resolved: &str, stateful: bool) -> RuntimeResult<*mut u8> {
        if !stateful {
            return Ok(std::ptr::null_mut());
        }
        let ctor = format!("{resolved}$new");
        let id = *self.function_ids.get(&ctor).ok_or_else(|| {
            RuntimeError::Execution(format!("stateful handler `{resolved}` has no constructor"))
        })?;
        let p = self
            .backend
            .get_function_pointer(id)
            .ok_or_else(|| RuntimeError::Execution(format!("no compiled entry for `{ctor}`")))?;
        // SAFETY: `H$new` is synthesized as `(): *state`.
        let f: extern "C" fn() -> *mut u8 = unsafe { std::mem::transmute(p) };
        Ok(f())
    }

    /// Resolve the named handler into the frame a push (or bind)
    /// installs: `(effect_id, state, op_table, async_mask)`, with
    /// fresh handler state allocated when it is stateful.
    fn named_handler_frame(&self, handler: &str) -> RuntimeResult<(u64, *mut u8, *mut u8, u64)> {
        let (resolved, effect_id, table, async_mask, stateful) = self.handler_shape(handler)?;
        let state = self.alloc_handler_state(&resolved, stateful)?;
        Ok((effect_id, state, table, async_mask))
    }

    /// Push a `with H`-equivalent frame for the named handler,
    /// allocating fresh handler state when it is stateful. Returns the
    /// frame id for the matching pop.
    fn push_named_handler(&self, handler: &str) -> RuntimeResult<u64> {
        let (effect_id, state, table, async_mask) = self.named_handler_frame(handler)?;
        Ok(crate::effect_runtime::__zyntax_effect_push_handler(
            effect_id, state, table, async_mask,
        ))
    }

    /// Bind the named handler to the machine persistently: the frame —
    /// including its handler state, allocated ONCE here — joins the
    /// fiber's saved handler segment, so every resume installs it and
    /// state carries across steps. This is the durable event-source
    /// binding; [`Self::resume_fiber_within`] is the per-step
    /// alternative, whose stateful handlers start fresh each call.
    ///
    /// Bound handlers layer beneath the machine's own `with` scopes,
    /// which keep precedence; among bound handlers, the earliest bound
    /// wins. Unbinding is dropping the fiber.
    ///
    /// A durable binding deserves a durable name: prefer resolving the
    /// handler once with [`Self::get_effect_handler`] and binding the token.
    ///
    /// This allocates the machine's state, so nothing outside the
    /// machine can read it. To share one region with host code, create
    /// it with [`Self::new_handler_instance`] and bind that instead.
    pub fn bind_fiber_handler(
        &mut self,
        token: FiberToken,
        handler: EffectHandlerToken,
    ) -> RuntimeResult<()> {
        let name = self
            .handler_tokens
            .get(&handler.0)
            .cloned()
            .ok_or_else(|| RuntimeError::Execution("unknown handler token".to_string()))?;
        self.bind_fiber_handler_named(token, &name)
    }

    /// [`Self::bind_fiber_handler`] by (FQN-aware) name, resolved at
    /// this call.
    pub fn bind_fiber_handler_named(
        &mut self,
        token: FiberToken,
        handler: &str,
    ) -> RuntimeResult<()> {
        let ptr = {
            let hf = self.host_fibers.get(&token.0).ok_or_else(|| {
                RuntimeError::Execution("unknown or dropped fiber token".to_string())
            })?;
            hf.ptr as *mut u8
        };
        let (resolved, effect_id, table, async_mask, stateful) = self.handler_shape(handler)?;
        let state = self.alloc_handler_state(&resolved, stateful)?;
        crate::effect_runtime::fiber_bind_handler(ptr, effect_id, state, table, async_mask);

        // Register the region so dropping the machine releases it. The
        // runtime owns and immediately disowns it: only this bind can
        // ever name it.
        let id = self.next_handler_instance;
        self.next_handler_instance += 1;
        self.handler_instances.insert(
            id,
            HandlerInstanceEntry {
                handler: resolved,
                effect_id,
                state: state as usize,
                table: table as usize,
                async_mask,
                installs: 1,
                dropped_by_owner: true,
            },
        );
        if let Some(hf) = self.host_fibers.get_mut(&token.0) {
            hf.bound_instances.push(id);
        }
        Ok(())
    }

    /// Register an additional built-in wrapper class, joining the
    /// compiler defaults in the registry each compilation snapshots —
    /// the same seam [`ZyntaxRuntime::register_builtin_class`]
    /// exposes. Call before the compilation that should see it.
    pub fn register_builtin_class(
        &self,
        class: Arc<dyn zyntax_compiler::builtin_class::BuiltinClass + Send + Sync>,
    ) {
        if let Ok(mut reg) = self.builtin_registry.lock() {
            reg.register(class);
        }
    }

    /// Snapshot the built-in registry for a lowering run.
    fn snapshot_builtin_registry(&self) -> Arc<zyntax_compiler::builtin_class::BuiltinRegistry> {
        let mut snapshot = zyntax_compiler::builtin_class::BuiltinRegistry::new();
        if let Ok(reg) = self.builtin_registry.lock() {
            for class in reg.classes() {
                snapshot.register(class.clone());
            }
        }
        Arc::new(snapshot)
    }

    /// Fold a parsed program's `fiber def` yield shapes into the shape
    /// registry, bumping the generation of any function whose shape
    /// changed — the signal [`Self::fiber_info`] exposes as staleness.
    /// Returns each touched entry's prior state, for rollback.
    fn apply_fiber_decls(
        &mut self,
        decls: Vec<(String, String, HostYieldKind)>,
    ) -> Vec<(String, Option<FiberShape>)> {
        let mut prior = Vec::new();
        for (name, shape, kind) in decls {
            match self.fiber_shapes.get_mut(&name) {
                Some(existing) if existing.shape != shape => {
                    prior.push((name.clone(), Some(existing.clone())));
                    existing.generation += 1;
                    existing.shape = shape;
                    existing.kind = kind;
                }
                Some(_) => {}
                None => {
                    prior.push((name.clone(), None));
                    self.fiber_shapes.insert(
                        name,
                        FiberShape {
                            shape,
                            kind,
                            generation: 0,
                        },
                    );
                }
            }
        }
        prior
    }

    /// Retire the machines of removed functions: their declarations
    /// leave the shape registry (so no new machine of a deleted
    /// function can be constructed) and their live tokens answer
    /// `MachineGone` from now on instead of resuming. Returns what was
    /// removed and which tokens were marked, for rollback.
    fn retire_removed_machines(
        &mut self,
        removed: &[String],
    ) -> (Vec<(String, Option<FiberShape>)>, Vec<u64>) {
        let mut prior_shapes = Vec::new();
        let mut marked = Vec::new();
        for name in removed {
            if let Some(entry) = self.fiber_shapes.remove(name) {
                prior_shapes.push((name.clone(), Some(entry)));
            }
            for (token, hf) in self.host_fibers.iter_mut() {
                if &hf.function == name && !hf.machine_gone {
                    hf.machine_gone = true;
                    marked.push(*token);
                }
            }
        }
        (prior_shapes, marked)
    }

    /// Record a non-aborted reload's fiber-metadata changes so a
    /// rollback can restore the handles' view along with the code. The
    /// undo record is (re)set exactly when the backend sets its own —
    /// a reload that changed the module — so the two halves of a
    /// rollback always describe the same reload.
    fn apply_reload_fiber_meta(
        &mut self,
        decls: Vec<(String, String, HostYieldKind)>,
        report: &zyntax_compiler::reload::ReloadReport,
    ) {
        let mut undo = FiberMetaUndo {
            shapes: self.apply_fiber_decls(decls),
            marked_gone: Vec::new(),
        };
        let (removed_shapes, marked) = self.retire_removed_machines(&report.removed_retained);
        undo.shapes.extend(removed_shapes);
        undo.marked_gone = marked;
        if !report.reloaded.is_empty()
            || !report.added.is_empty()
            || !report.removed_retained.is_empty()
        {
            self.fiber_meta_undo = Some(undo);
        }
    }

    /// Native entry pointer for `name`, or `None` if unknown. The
    /// pointer stays valid for the life of the runtime; a reload swaps
    /// what new calls dispatch to, not what this pointer points at.
    pub fn function_pointer(&self, name: &str) -> Option<*const u8> {
        let id = self.function_ids.get(name)?;
        self.backend.get_function_pointer(*id)
    }

    /// Reload edited source against the running module.
    ///
    /// Parses and lowers exactly as [`Self::load_module`] does, then
    /// hands the edited module to the backend, which swaps only the
    /// functions whose content changed. State is untouched; the next
    /// call to a reloaded function runs the new code.
    pub fn reload_module_source(
        &mut self,
        language: &str,
        source: &str,
    ) -> RuntimeResult<zyntax_compiler::reload::ReloadReport> {
        let grammar = self.grammars.get(language).cloned().ok_or_else(|| {
            RuntimeError::Execution(format!(
                "Unknown language '{}'. Registered languages: {:?}",
                language,
                self.languages()
            ))
        })?;

        let mut typed_program = grammar
            .parse_with_signatures(source, &format!("<{language}>"), &self.plugin_signatures)
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;
        capture_runtime_events_from_program(
            &mut typed_program,
            &mut self.runtime_events,
            self.event_sink.as_ref(),
        );

        let builtins: indexmap::IndexMap<String, String> = grammar
            .builtins()
            .functions
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let fiber_decls = collect_fiber_decls(&typed_program);
        let mut hir_module = self.lower_typed_program(typed_program, builtins)?;
        apply_krio_async_lowering(&mut hir_module)?;
        apply_krio_effect_lowering(&mut hir_module)?;
        apply_krio_fiber_lowering(&mut hir_module);

        // The running module went through the same optimization pipeline
        // on load; diffing an optimized function against an unoptimized
        // lowering of identical source would report every function
        // changed.
        if std::env::var("ZYNTAX_DISABLE_INTERP_OPTS").is_err() {
            let _stats = zyntax_compiler::run_interp_safe_opts(&mut hir_module);
        }

        let report = self
            .backend
            .reload_module(&hir_module)
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;

        // An aborted reload changed nothing, so the handles' view of
        // shapes and machines must not move either.
        if !report.aborted {
            self.apply_reload_fiber_meta(fiber_decls, &report);
            if !report.state_migrations.is_empty() {
                let plans = report.state_migrations.clone();
                self.apply_state_migrations(&plans);
            }
        }

        // Reload is an observable event: frameworks subscribe to
        // invalidate whatever the edit touched.
        let event = RuntimeEvent::Reload {
            reloaded: report.reloaded.clone(),
            added: report.added.clone(),
            dispatch_patched: report.dispatch_patched.clone(),
            failed: report.failed.clone(),
        };
        if let Some(sink) = &self.event_sink {
            sink(&event);
        }
        self.runtime_events.push(event);

        Ok(report)
    }

    pub fn load_module(&mut self, language: &str, source: &str) -> RuntimeResult<Vec<String>> {
        let grammar = self.grammars.get(language).cloned().ok_or_else(|| {
            RuntimeError::Execution(format!(
                "Unknown language '{}'. Registered languages: {:?}",
                language,
                self.languages()
            ))
        })?;

        // Parse source to TypedAST with plugin signatures for proper extern declarations
        let mut typed_program = grammar
            .parse_with_signatures(source, &format!("<{language}>"), &self.plugin_signatures)
            .map_err(|e| RuntimeError::Execution(e.to_string()))?;
        capture_runtime_events_from_program(
            &mut typed_program,
            &mut self.runtime_events,
            self.event_sink.as_ref(),
        );

        // Lower to HIR with grammar builtins
        let builtins: indexmap::IndexMap<String, String> = grammar
            .builtins()
            .functions
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let fiber_decls = collect_fiber_decls(&typed_program);
        let mut hir_module = self.lower_typed_program(typed_program, builtins)?;
        apply_krio_async_lowering(&mut hir_module)?;
        apply_krio_effect_lowering(&mut hir_module)?;
        apply_krio_fiber_lowering(&mut hir_module);

        // Collect function names before compilation. `f.name.to_string()`
        // returns the debug repr of InternedString (e.g.
        // "InternedString(SymbolU32 { value: 4 })") — `resolve_global()`
        // returns the actual symbol text. Mirrors `ZyntaxRuntime::load_module`.
        let function_names: Vec<String> = hir_module
            .functions
            .values()
            .filter(|f| !f.is_external)
            .filter_map(|f| f.name.resolve_global())
            .collect();

        // Compile the module
        self.compile_module(hir_module)?;
        let _ = self.apply_fiber_decls(fiber_decls);

        Ok(function_names)
    }

    /// Load a module from a file, auto-detecting the language from extension
    pub fn load_module_file<P: AsRef<Path>>(&mut self, path: P) -> RuntimeResult<Vec<String>> {
        let path = path.as_ref();

        let extension = path.extension().and_then(|e| e.to_str()).ok_or_else(|| {
            RuntimeError::Execution(format!("File '{}' has no extension", path.display()))
        })?;

        let language = self
            .language_for_extension(extension)
            .ok_or_else(|| {
                RuntimeError::Execution(format!(
                    "No grammar registered for extension '.{}'",
                    extension
                ))
            })?
            .to_string();

        let source = std::fs::read_to_string(path).map_err(|e| {
            RuntimeError::Execution(format!("Failed to read '{}': {}", path.display(), e))
        })?;

        self.load_module(&language, &source)
    }

    /// Lower a TypedProgram to HirModule
    fn lower_typed_program(
        &self,
        mut program: zyntax_typed_ast::TypedProgram,
        builtins: indexmap::IndexMap<String, String>,
    ) -> RuntimeResult<HirModule> {
        use zyntax_compiler::lowering::{LoweringConfig, LoweringContext};
        use zyntax_typed_ast::{
            type_registry::*, AstArena, InternedString, TypeRegistry, TypedDeclaration,
        };

        // Stateful handlers need their state struct, ctor and implicit
        // `self` synthesized before the registry snapshot, exactly as
        // in the classic runtime's lowering above.
        synthesize_handler_state(&mut program);

        // Rebuild type registry from declarations
        for decl_node in &program.declarations {
            if let TypedDeclaration::Class(class) = &decl_node.node {
                let type_id = if let zyntax_typed_ast::Type::Named { id, .. } = &decl_node.ty {
                    *id
                } else {
                    TypeId::next()
                };

                let field_defs: Vec<FieldDef> = class
                    .fields
                    .iter()
                    .map(|f| FieldDef {
                        name: f.name,
                        ty: f.ty.clone(),
                        visibility: f.visibility,
                        mutability: f.mutability,
                        is_static: f.is_static,
                        span: f.span,
                        getter: None,
                        setter: None,
                        is_synthetic: false,
                    })
                    .collect();

                // Strict V1 reference-class lowering: propagate `@reference`
                // annotation in the rebuild path that precedes
                // `lower_typed_program`.
                let is_reference = class.annotations.iter().any(|ann| {
                    ann.name
                        .resolve_global()
                        .as_deref()
                        .map(|n| n == "reference")
                        .unwrap_or(false)
                });
                let mut metadata: zyntax_typed_ast::type_registry::TypeMetadata =
                    Default::default();
                metadata.is_reference = is_reference;

                let type_def = TypeDefinition {
                    id: type_id,
                    module: None,
                    name: class.name,
                    kind: TypeKind::Struct {
                        fields: field_defs.clone(),
                        is_tuple: false,
                    },
                    type_params: vec![],
                    constraints: vec![],
                    fields: field_defs,
                    methods: vec![],
                    constructors: vec![],
                    metadata,
                    span: class.span,
                };
                program.type_registry.register_type(type_def);
            }
        }

        let mut type_registry = program.type_registry.clone();

        // Process imports FIRST so stdlib `extern def`s (e.g. tensor.zynml's
        // `arange`, `Tensor::sum`) get parsed and merged into the program
        // before lowering. Without this, calls like `Tensor::arange(...)`
        // lower to a call against an unresolved function and segfault at
        // runtime. Mirrors `ZyntaxRuntime::lower_typed_program`.
        crate::import_chain::process_imports_for_traits(
            &self.grammars,
            &self.plugin_signatures,
            &self.import_resolvers,
            &self.compiled_import_resolvers,
            &self.snapshot_modules,
            &mut program,
            &mut type_registry,
        )?;

        // Now process extern declarations from the merged program
        // to ensure all opaque types are registered.
        crate::import_chain::process_extern_declarations_mut(&program, &mut type_registry)?;

        // Resolve all `Type::Unresolved` in the TypedAST before lowering
        // (e.g. extern types coming from imports become `Type::Extern`).
        crate::import_chain::resolve_unresolved_types(&mut program, &type_registry);

        // Sync the program's type registry with the locally-merged one
        // before passing it on to register_impl_blocks et al.
        program.type_registry = type_registry;

        // Register impl blocks before lowering
        zyntax_compiler::register_impl_blocks(&mut program).map_err(|e| {
            RuntimeError::Execution(format!("Failed to register impl blocks: {:?}", e))
        })?;

        // Generate automatic trait implementations for abstract types
        zyntax_compiler::generate_abstract_trait_impls(&mut program).map_err(|e| {
            RuntimeError::Execution(format!("Failed to generate abstract trait impls: {:?}", e))
        })?;

        // Register the generated impl blocks
        zyntax_compiler::register_impl_blocks(&mut program).map_err(|e| {
            RuntimeError::Execution(format!("Failed to register generated impl blocks: {:?}", e))
        })?;

        let arena = AstArena::new();
        // The module a program lowers under is the file it came from.
        // Naming every program `main` put them all in one module, which
        // is the wrong answer for anything that qualifies a name by the
        // module holding it.
        let module_name = program
            .source_files
            .first()
            .map(|file| crate::grammar::module_name_of(&file.name))
            .or_else(|| program.type_registry.current_module())
            .unwrap_or_else(|| InternedString::new_global("module"));
        // Use the type registry from the parsed program (now contains registered structs)
        let type_registry = std::sync::Arc::new(program.type_registry.clone());

        // Fiber<T>'s `Fiber.abort(err)` static method maps to the
        // `krio_fiber_abort_with` runtime stub (Wren-style abort
        // from inside the fiber body). See the matching entry in
        // `lower_typed_program`'s public entry point.
        let mut builtins = builtins;
        builtins
            .entry("Fiber$abort".to_string())
            .or_insert_with(|| "krio_fiber_abort_with".to_string());

        // Create LoweringConfig with builtins for extern call resolution
        let lowering_config = LoweringConfig {
            builtins,
            use_krio_async: cfg!(feature = "krio-async-backend"),
            ..LoweringConfig::default()
        };

        // Run pattern engine
        {
            let mut engine = pattern_engine::PatternEngine::new(pattern_engine::EngineConfig {
                target: pattern_engine::LoweringTarget::Cpu,
                max_iterations: 64,
                trace: cfg!(debug_assertions),
                verify_after: false,
            });
            engine.register_pass(normalization_pass::Pass);
            engine.register_pass(algebraic_effects_pass::Pass);
            engine.finalize().map_err(|e| {
                RuntimeError::Execution(format!("Pattern engine finalize error: {}", e))
            })?;
            let _result = engine.run(&mut program, &type_registry);
        }

        let mut lowering_ctx = LoweringContext::new(
            module_name,
            type_registry.clone(),
            std::sync::Arc::new(std::sync::Mutex::new(arena)),
            lowering_config,
        );
        lowering_ctx.set_builtin_registry(self.snapshot_builtin_registry());

        let mut hir_module = lowering_ctx
            .lower_program(&mut program)
            .map_err(|e| RuntimeError::Execution(format!("Lowering error: {:?}", e)))?;

        // Display lowering diagnostics (type inference warnings, etc.)
        lowering_ctx.display_diagnostics(&program);

        zyntax_compiler::monomorphize_module(&mut hir_module)
            .map_err(|e| RuntimeError::Execution(format!("Monomorphization error: {:?}", e)))?;

        Ok(hir_module)
    }

    /// List all loaded function names
    pub fn functions(&self) -> Vec<&str> {
        self.function_ids.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a function is defined
    pub fn has_function(&self, name: &str) -> bool {
        self.function_ids.contains_key(name)
    }
}

impl Drop for TieredRuntime {
    fn drop(&mut self) {
        self.shutdown();
    }
}
