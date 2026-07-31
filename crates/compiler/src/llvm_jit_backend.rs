//! # LLVM AOT-via-Object JIT Backend
//!
//! This backend compiles HIR → LLVM IR, lowers to a position-
//! independent object file, links it via the system linker into a
//! shared object, and `dlopen`s the result. Function pointers are
//! extracted with `dlsym`.
//!
//! ## Why not MCJIT?
//! The previous incarnation used `module.create_jit_execution_engine(...)`
//! (LLVM MCJIT). MCJIT runs into MAP_JIT cross-thread invalidation
//! issues on Apple Silicon, and on x86_64 its TargetMachine selection
//! is conservative compared to a host-tuned AOT compile.
//!
//! ## Pipeline
//!
//! 1. **Lower**: HIR → LLVM IR (via `LLVMBackend`).
//! 2. **Verify + optimise**: pre-pass `module.verify()`, then
//!    `default<O3>` (or matching level), then post-pass `verify()`.
//! 3. **Emit**: `TargetMachine::write_to_file(FileType::Object)` to a
//!    tempfile. Reloc mode `PIC`, code model `Default`.
//! 4. **Trampolines**: synthesise an assembly file defining each
//!    runtime symbol as a real function whose body loads the host
//!    pointer and tail-jumps. Replaces MCJIT's `add_global_mapping`.
//! 5. **Link**: shell out to `cc`/`clang`/`gcc` with `-shared` (Linux)
//!    or `-dynamiclib` (macOS). See [`crate::llvm_link::link_to_dylib`].
//! 6. **Load**: `dlopen(RTLD_NOW | RTLD_GLOBAL)` and `dlsym` each
//!    function symbol into `function_pointers`.
//!
//! ## vs AOT LLVM Backend
//! - AOT (`llvm_backend.rs`): compile entire program to standalone exe.
//! - JIT (this file): same lowering but loaded back into the running
//!   process via dlopen. Suitable for tiered compilation.
//!
//! Both share the HIR → LLVM IR translation logic.

use crate::hir::{HirFunction, HirId, HirModule};
use crate::llvm_backend::LLVMBackend;
use crate::{CompilerError, CompilerResult};
use indexmap::IndexMap;
use inkwell::{
    context::Context,
    passes::PassBuilderOptions,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    OptimizationLevel,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

/// Process-global cache of fully-loaded JIT dylibs, keyed by a content
/// hash supplied by the caller. The library is leaked into the cache
/// so its mapped code pages stay alive for the rest of the process —
/// dropping it would munmap the pages and dangle every function
/// pointer we ever resolved through it. The cache survives across
/// fresh `LLVMJitBackend` instances within the same process, which is
/// exactly the bench-harness pattern (each kernel iteration creates
/// a fresh runtime + fresh backend but reuses the same source).
///
/// The map's value is the dlsym'd `function_name → host_address` table
/// that the cached library exports. When a cache hit lands, the
/// caller rebuilds its `HirId → host_address` map against the current
/// HirModule's `get_function_symbols` output without touching the
/// linker or dlopen — saving the dominant ~370 ms install cost on
/// macOS aarch64 and ~70 ms on Linux x86_64.
static DYLIB_CACHE: LazyLock<Mutex<HashMap<String, &'static HashMap<String, usize>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Monotonic counter for tempfile naming uniqueness within a process.
static AOT_TMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// LLVM AOT-via-object JIT backend.
///
/// Compiles HIR modules to native code by writing a PIC object,
/// linking it into a shared library through the system linker, and
/// `dlopen`-ing the result. Function pointers are read back via
/// `dlsym` and indexed by `HirId`.
pub struct LLVMJitBackend<'ctx> {
    /// LLVM context reference.
    context: &'ctx Context,

    /// Loaded shared object. Holding this `Library` keeps the mapped
    /// code pages alive — dropping it would munmap them and any held
    /// function pointer would dangle. Stored after a successful
    /// `compile_module` so the runtime can keep dispatching.
    loaded_lib: Option<libloading::Library>,

    /// Function pointers cache (stored as usize for thread safety).
    pub(crate) function_pointers: IndexMap<HirId, usize>,

    /// Optimization level.
    opt_level: OptimizationLevel,

    /// Runtime symbols available to JIT-compiled code. Each entry is a
    /// host-side function pointer that will get baked into a
    /// trampoline stub during link.
    pub(crate) runtime_symbols: IndexMap<String, usize>,

    /// Symbol signatures for auto-boxing (symbol name → signature).
    symbol_signatures: Vec<crate::zrtl::RuntimeSymbolInfo>,

    /// Optional reachable-from-`main` function filter passed through
    /// to `LLVMBackend::set_only_compile_reachable` on every
    /// `compile_module`. Set by the runtime via
    /// `set_only_compile_reachable`; mirrors the Cranelift JIT path
    /// that the runtime already gates on `reachable_function_ids`.
    only_compile_reachable: Option<std::collections::HashSet<HirId>>,

    /// Optional content-hash cache key supplied by the caller. When
    /// set, `compile_module` checks `DYLIB_CACHE` for a previously
    /// loaded dylib with the same effective key (caller key + a
    /// runtime-address fingerprint) and short-circuits the install
    /// pipeline on a hit. See [`Self::set_cache_key`].
    cache_key: Option<String>,
}

impl<'ctx> LLVMJitBackend<'ctx> {
    /// Create a new LLVM JIT backend with aggressive optimisation.
    pub fn new(context: &'ctx Context) -> CompilerResult<Self> {
        Self::with_opt_level(context, OptimizationLevel::Aggressive)
    }

    /// Create with custom optimisation level.
    pub fn with_opt_level(
        context: &'ctx Context,
        opt_level: OptimizationLevel,
    ) -> CompilerResult<Self> {
        // Initialise LLVM targets.
        Target::initialize_native(&InitializationConfig::default()).map_err(|e| {
            CompilerError::Backend(format!("Failed to initialise LLVM target: {}", e))
        })?;

        Ok(Self {
            context,
            loaded_lib: None,
            function_pointers: IndexMap::new(),
            opt_level,
            runtime_symbols: IndexMap::new(),
            symbol_signatures: Vec::new(),
            only_compile_reachable: None,
            cache_key: None,
        })
    }

    /// Set a content-hash cache key for subsequent `compile_module`
    /// calls. When supplied, the install path consults the
    /// process-global `DYLIB_CACHE`: a cache hit reuses the already-
    /// loaded dylib's `dlsym`'d address table and skips the LLVM IR
    /// emit + opt + linker + dlopen chain (saves ~370 ms on macOS
    /// aarch64, ~70 ms on Linux x86_64). The effective key combines
    /// `key` with a runtime-address fingerprint so cache entries
    /// stale across process restarts (ASLR moves runtime symbols)
    /// invalidate automatically.
    ///
    /// Pass `None` to disable caching for this backend.
    pub fn set_cache_key(&mut self, key: Option<String>) {
        self.cache_key = key;
    }

    /// Register symbol signatures for auto-boxing support.
    pub fn register_symbol_signatures(&mut self, symbols: &[crate::zrtl::RuntimeSymbolInfo]) {
        self.symbol_signatures.extend(symbols.iter().cloned());
    }

    /// Register a runtime symbol callable from JIT-compiled code.
    ///
    /// Call before `compile_module`. Each registered symbol becomes
    /// a trampoline stub in the linked dylib, so the .so's calls to
    /// `<name>` land on a thin shim that jumps to `ptr`.
    /// Limit subsequent `compile_module` calls to the supplied
    /// reachable-from-entry subset. Mirrors
    /// `CraneliftBackend::set_only_compile_reachable` so the LLVM AOT
    /// install pays the same scope as the Cranelift install. Without
    /// this the LLVM backend lowers + verifies + O3s every prelude
    /// helper (~100 functions on a typical bench), even when the
    /// kernel reaches only a handful.
    pub fn set_only_compile_reachable(
        &mut self,
        allowed: Option<std::collections::HashSet<HirId>>,
    ) {
        self.only_compile_reachable = allowed;
    }

    pub fn register_symbol(&mut self, name: impl Into<String>, ptr: *const u8) {
        self.runtime_symbols.insert(name.into(), ptr as usize);
    }

    /// Register multiple runtime symbols at once.
    pub fn register_symbols(&mut self, symbols: &[(&str, *const u8)]) {
        for (name, ptr) in symbols {
            self.runtime_symbols
                .insert((*name).to_string(), *ptr as usize);
        }
    }

    /// Build `PassBuilderOptions` for LLVM optimisation passes.
    ///
    /// Loop vectorisation (LV) and SLP vectorisation stay on — those
    /// are the headline x86_64 wins for the numerical kernels.
    ///
    /// `set_loop_unrolling(true)` is deliberately NOT set: under
    /// `default<O3>`, forcing it overrides LLVM's own size/trip-count
    /// heuristic — on kernels with very large trip counts (e.g.
    /// inlined_call's 100M-iter loop with an inlinable callee) the
    /// forced unroller tries to fully unroll and the pipeline hangs.
    /// Without this override, `default<O3>` still runs the standard
    /// unroller with its built-in trip-count cap.
    ///
    /// `set_loop_interleaving(true)` is left off because it composes
    /// with the unroller. `merge_functions(true)` is on because it's
    /// cheap and shrinks dispatch tables for monomorphised generics.
    fn create_pass_options() -> PassBuilderOptions {
        let opts = PassBuilderOptions::create();
        opts.set_loop_vectorization(true);
        opts.set_loop_slp_vectorization(true);
        opts.set_merge_functions(true);
        opts
    }

    /// Mangle a function-symbol name for export through the system
    /// linker / dlsym pipeline.
    ///
    /// The MCJIT-era naming scheme used `format!("func_{:?}", id)`
    /// which produces `func_HirId(42)` — parens are linker-hostile
    /// and the `,`/`<`/`>` chars from generic Debug formats would
    /// also break ELF/Mach-O symbol parsers. This helper sanitises to
    /// `[A-Za-z0-9_$]` (all of which are accepted by the system
    /// linker on both ELF and Mach-O — `$` is widely supported, and
    /// we already use it for handler-op names like `Console$info`).
    pub fn mangle_function_name(name: &str) -> String {
        let mut s = String::with_capacity(name.len());
        for c in name.chars() {
            match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '$' => s.push(c),
                '(' => s.push_str("_LP_"),
                ')' => s.push_str("_RP_"),
                '<' => s.push_str("_LT_"),
                '>' => s.push_str("_GT_"),
                ',' => s.push_str("_C_"),
                ' ' => s.push_str("_S_"),
                '-' => s.push_str("_D_"),
                ':' => s.push_str("_K_"),
                '.' => s.push_str("_P_"),
                _ => {
                    s.push_str(&format!("_X{:x}_", c as u32));
                }
            }
        }
        s
    }

    /// Compute the exported name a given function will carry in the
    /// dylib. Mirrors `llvm_backend.rs:281`: external functions and
    /// the `main` entry keep their original name; everything else
    /// gets the mangled `func_<HirId>` form.
    ///
    /// We mangle the raw `HirId` Debug string (which contains
    /// `HirId(<n>)` parens) so the dlsym side sees a valid symbol.
    /// The LLVM IR emit side currently uses the raw Debug form; we
    /// patch the IR-side naming to use the same mangled form in
    /// [`Self::compile_module_to_ir`].
    fn exported_name_for(func: &HirFunction, id: HirId) -> String {
        let actual_name = func
            .name
            .resolve_global()
            .unwrap_or_else(|| format!("{:?}", func.name));
        if func.is_external || actual_name == "main" {
            actual_name
        } else {
            Self::mangle_function_name(&format!("func_{:?}", id))
        }
    }

    /// Patch the names of every internal (non-extern, non-main)
    /// function in an already-lowered LLVM module to use the
    /// `[A-Za-z0-9_$]` mangling. The HIR-lowering pass at
    /// `llvm_backend.rs:285` emits `format!("func_{:?}", id)` which
    /// contains parens. We rename in place so the symbols come out
    /// of the linker with a name the loader will accept.
    fn rename_mangled_functions(
        backend: &LLVMBackend<'ctx>,
        hir_module: &HirModule,
    ) -> CompilerResult<()> {
        for (id, func) in &hir_module.functions {
            // Skip externs and `main` — they keep their actual name.
            let actual_name = func
                .name
                .resolve_global()
                .unwrap_or_else(|| format!("{:?}", func.name));
            if func.is_external || actual_name == "main" {
                continue;
            }
            let raw = format!("func_{:?}", id);
            let mangled = Self::mangle_function_name(&raw);
            if raw == mangled {
                continue;
            }
            if let Some(fv) = backend.module().get_function(&raw) {
                fv.as_global_value().set_name(&mangled);
            }
        }
        Ok(())
    }

    /// Step (1)–(2) of the pipeline: lower HIR → LLVM IR, verify,
    /// host-tune the target machine, run optimisation passes. The
    /// resulting `LLVMBackend` owns a `Module` ready for object-file
    /// emission via [`Self::compile_to_object_file`].
    fn compile_module_to_ir(
        &self,
        hir_module: &HirModule,
    ) -> CompilerResult<(LLVMBackend<'ctx>, TargetMachine)> {
        // Step 1: Lower HIR → LLVM IR.
        let mut backend = LLVMBackend::new(self.context, "zyntax_jit");
        backend.register_symbol_signatures(&self.symbol_signatures);
        backend.set_only_compile_reachable(self.only_compile_reachable.clone());
        // On-host JIT: target == host, so the target's VNNI capability is
        // the host's (the TargetMachine below is likewise built from
        // `get_host_cpu_features()`). A cross/portable-AOT compile would
        // instead derive this from the AOT target's declared features so
        // no host-only `vpdpbusd` is baked into the object.
        #[cfg(target_arch = "x86_64")]
        backend.set_x86_target_vnni(
            std::arch::is_x86_feature_detected!("avxvnni")
                || std::arch::is_x86_feature_detected!("avx512vnni"),
        );
        let _llvm_ir = backend.compile_module(hir_module)?;

        // Patch internal function names to a linker-safe mangling.
        Self::rename_mangled_functions(&backend, hir_module)?;

        // Step 2: Build a host-tuned TargetMachine. PIC reloc mode is
        // required for `-shared` / `-dynamiclib`; default code model
        // is correct for both x86_64 and aarch64.
        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple).map_err(|e| {
            CompilerError::Backend(format!("Failed to resolve target from triple: {}", e))
        })?;
        let target_machine = target
            .create_target_machine(
                &target_triple,
                TargetMachine::get_host_cpu_name()
                    .to_str()
                    .unwrap_or("generic"),
                TargetMachine::get_host_cpu_features()
                    .to_str()
                    .unwrap_or(""),
                self.opt_level,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| CompilerError::Backend("Failed to create host target machine".into()))?;
        backend.module().set_triple(&target_triple);
        backend
            .module()
            .set_data_layout(&target_machine.get_target_data().get_data_layout());

        if std::env::var("ZYNTAX_DUMP_LLVM_IR").is_ok() {
            let ir = backend.module().print_to_string().to_string();
            static DUMP_COUNTER: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            let n = DUMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let fname = format!("zyntax_llvm_ir_preopt_{n:04}.ll");
            let path = std::env::temp_dir().join(fname);
            if std::fs::write(&path, &ir).is_ok() {
                eprintln!(
                    "[LLVM-IR] pre-opt IR written to {} ({} bytes)",
                    path.display(),
                    ir.len()
                );
            }
        }

        // Pre-pass verification — catches HIR→LLVM lowering bugs
        // before the optimiser amplifies them.
        if let Err(msg) = backend.module().verify() {
            return Err(CompilerError::Backend(format!(
                "LLVM module verification failed (pre-opt): {}",
                msg.to_string()
            )));
        }

        // Step 3: Run the optimisation pipeline.
        if self.opt_level != OptimizationLevel::None {
            let passes = match self.opt_level {
                OptimizationLevel::None => "default<O0>",
                OptimizationLevel::Less => "default<O1>",
                OptimizationLevel::Default => "default<O2>",
                OptimizationLevel::Aggressive => "default<O3>",
            };
            let pass_options = Self::create_pass_options();
            backend
                .module()
                .run_passes(passes, &target_machine, pass_options)
                .map_err(|e| {
                    CompilerError::Backend(format!("LLVM optimisation passes failed: {}", e))
                })?;

            if let Err(msg) = backend.module().verify() {
                return Err(CompilerError::Backend(format!(
                    "LLVM module verification failed (post-opt): {}",
                    msg.to_string()
                )));
            }

            if std::env::var("ZYNTAX_DUMP_LLVM_IR").is_ok() {
                let ir = backend.module().print_to_string().to_string();
                static POSTOPT_COUNTER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let n = POSTOPT_COUNTER.fetch_add(1, Ordering::Relaxed);
                let fname = format!("zyntax_llvm_ir_postopt_{n:04}.ll");
                let path = std::env::temp_dir().join(fname);
                if std::fs::write(&path, &ir).is_ok() {
                    eprintln!(
                        "[LLVM-IR] post-opt IR written to {} ({} bytes)",
                        path.display(),
                        ir.len()
                    );
                }
            }
        }

        Ok((backend, target_machine))
    }

    /// Write the LLVM module to a position-independent object file.
    ///
    /// Mirrors rayzor's `compile_to_object_file` (rayzor
    /// llvm_jit_backend.rs:1191). Uses the host-tuned TargetMachine
    /// from `Self::compile_module_to_ir` for codegen options. The
    /// resulting `.o` is ready for the system linker.
    pub fn compile_to_object_file(
        &self,
        backend: &LLVMBackend<'ctx>,
        target_machine: &TargetMachine,
        output_path: &Path,
    ) -> CompilerResult<()> {
        target_machine
            .write_to_file(backend.module(), FileType::Object, output_path)
            .map_err(|e| CompilerError::Backend(format!("Failed to emit object file: {}", e)))
    }

    /// Enumerate every emitted-with-body function in the module as
    /// `(HirId, exported-symbol-name)`. Externs are excluded — they
    /// are call targets, not export targets.
    ///
    /// The returned names already use the mangled form, so callers
    /// can feed them straight to dlsym.
    pub fn get_function_symbols(&self, hir_module: &HirModule) -> BTreeMap<HirId, String> {
        let mut map = BTreeMap::new();
        for (id, function) in &hir_module.functions {
            if function.is_external {
                continue;
            }
            if let Some(allowed) = &self.only_compile_reachable {
                if !allowed.contains(id) {
                    continue;
                }
            }
            map.insert(*id, Self::exported_name_for(function, *id));
        }
        map
    }

    /// Build a unique tempfile path with the supplied extension.
    fn build_temp_path(extension: &str) -> std::path::PathBuf {
        let counter = AOT_TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut p = std::env::temp_dir();
        p.push(format!("zyntax_llvm_{nanos}_{pid}_{counter}.{extension}"));
        p
    }

    /// Compile a full HIR module via AOT-to-object + system linker +
    /// dlopen. On success, `function_pointers` is populated and the
    /// loaded library is pinned on the backend.
    ///
    /// Errors:
    /// - HIR → LLVM IR lowering / verification / optimisation failure
    ///   surface as `CompilerError::Backend`.
    /// - No system linker found / link errors / dlopen errors surface
    ///   as `CompilerError::Backend` too. The runtime's calling code
    ///   treats any error as a soft-fail and stays on Cranelift.
    pub fn compile_module(&mut self, hir_module: &HirModule) -> CompilerResult<()> {
        use crate::llvm_link::{link_to_dylib, load_dylib, Triple};

        // Reset state from any previous compile.
        self.function_pointers.clear();
        self.loaded_lib = None;

        // Cache check. The effective key is `caller_key ⊕ runtime-
        // address fingerprint`: the fingerprint catches stale entries
        // across process restarts (ASLR moves library-resident symbols
        // like `println_dynamic`, the BC interp's tick callbacks, etc.
        // — the cached dylib's trampolines hardcode those addresses so
        // we have to invalidate when they change). When the caller
        // didn't supply a key, caching is off entirely.
        let cache_id = self.cache_key.as_ref().map(|k| {
            let mut h = fnv1a_64(k.as_bytes());
            // Domain separator + sorted (name, addr) so the fingerprint
            // is independent of `runtime_symbols` insertion order.
            h = fnv1a_64_update(h, b"\0syms\0");
            let mut sorted: Vec<(&String, &usize)> = self.runtime_symbols.iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(b.0));
            for (name, addr) in sorted {
                h = fnv1a_64_update(h, name.as_bytes());
                h = fnv1a_64_update(h, &(*addr as u64).to_le_bytes());
            }
            format!("{h:016x}")
        });
        if let Some(id) = cache_id.as_ref() {
            if let Some(name_to_ptr) = DYLIB_CACHE.lock().unwrap().get(id).copied() {
                // Cache hit. Re-resolve current HirModule's function
                // symbols against the cached `name → addr` table.
                let func_syms = self.get_function_symbols(hir_module);
                for (hir_id, name) in &func_syms {
                    if let Some(ptr) = name_to_ptr.get(name) {
                        self.function_pointers.insert(*hir_id, *ptr);
                    }
                }
                // `loaded_lib` stays None; the library lives in the
                // cache for the rest of the process.
                return Ok(());
            }
        }

        // (1)–(3) Lower + optimise + build target machine.
        let (backend, target_machine) = self.compile_module_to_ir(hir_module)?;

        // (4) Write PIC object file to a tempfile.
        let triple = Triple::host();
        let obj_path = Self::build_temp_path("o");
        let dylib_path = Self::build_temp_path(triple.dylib_ext());
        self.compile_to_object_file(&backend, &target_machine, &obj_path)?;
        // Module + TargetMachine are no longer needed; drop them so
        // the LLVM resources are freed before the slow linker stage.
        drop(backend);
        drop(target_machine);

        // (5) Synthesise trampolines + link to dylib. The runtime
        // symbols list comes from `self.runtime_symbols` — populated
        // by `register_symbol[s]` before this call.
        let symbols: Vec<(String, usize)> = self
            .runtime_symbols
            .iter()
            .map(|(n, p)| (n.clone(), *p))
            .collect();
        let link_result = link_to_dylib(&obj_path, &dylib_path, &symbols);
        if let Err(e) = link_result {
            let _ = std::fs::remove_file(&obj_path);
            return Err(CompilerError::Backend(format!("link: {e}")));
        }

        // The .o is no longer needed once the .dylib exists.
        let _ = std::fs::remove_file(&obj_path);

        // (6) dlopen + dlsym every exported function symbol.
        let func_syms = self.get_function_symbols(hir_module);
        let names: Vec<String> = func_syms.values().cloned().collect();
        let load_result = load_dylib(&dylib_path, &names);
        let (lib, name_to_ptr) = match load_result {
            Ok(v) => v,
            Err(e) => {
                let _ = std::fs::remove_file(&dylib_path);
                return Err(CompilerError::Backend(format!("load: {e}")));
            }
        };

        // (7) Populate function_pointers from the dlsym results.
        for (hir_id, name) in &func_syms {
            if let Some(ptr) = name_to_ptr.get(name) {
                self.function_pointers.insert(*hir_id, *ptr);
            }
        }

        // (8) The mapping is now mmap'd; the inode can go. POSIX
        // dlopen pins the inode + mapping past unlink so this is
        // safe on macOS and Linux.
        let _ = std::fs::remove_file(&dylib_path);

        // Populate the process-global cache if a key was supplied.
        // Leaks the library and the `name → addr` map into 'static
        // storage — both have to stay alive for the lifetime of every
        // future cache hit, and a leak is the simplest way to express
        // "alive for the rest of the process". For runs without a
        // cache key, no leak occurs (the library is moved into
        // `self.loaded_lib` as before).
        if let Some(id) = cache_id {
            let leaked_map: &'static HashMap<String, usize> =
                Box::leak(Box::new(name_to_ptr.clone()));
            Box::leak(Box::new(lib));
            DYLIB_CACHE.lock().unwrap().insert(id, leaked_map);
        } else {
            // Pin the library — dropping it would unmap the code pages
            // and any held function pointer would dangle.
            self.loaded_lib = Some(lib);
        }

        Ok(())
    }

    /// Compile a single function. Kept as a thin wrapper around
    /// `compile_module` for source compatibility with the older
    /// MCJIT-era API. The single-function shape is incorrect in
    /// general (any call to another function in the original module
    /// dangles) — callers should prefer `compile_module`.
    pub fn compile_function(&mut self, id: HirId, function: &HirFunction) -> CompilerResult<()> {
        use std::collections::HashSet;

        let temp_module = HirModule {
            id: HirId::new(),
            name: function.name,
            functions: [(id, function.clone())].iter().cloned().collect(),
            globals: IndexMap::new(),
            types: IndexMap::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            version: 0,
            dependencies: HashSet::new(),
            effects: IndexMap::new(),
            handlers: IndexMap::new(),
        };
        self.compile_module(&temp_module)?;
        Ok(())
    }

    /// Get a function pointer for a HirId, if it was emitted and
    /// dlsym'd successfully.
    pub fn get_function_pointer(&self, func_id: HirId) -> Option<*const u8> {
        self.function_pointers
            .get(&func_id)
            .map(|&addr| addr as *const u8)
    }

    /// Get optimisation level.
    pub fn optimization_level(&self) -> OptimizationLevel {
        self.opt_level
    }
}

/// 64-bit FNV-1a — fast, allocation-free, good enough for cache keys
/// where we just need collision resistance, not cryptographic strength.
fn fnv1a_64(input: &[u8]) -> u64 {
    fnv1a_64_update(0xcbf29ce484222325, input)
}

fn fnv1a_64_update(mut h: u64, input: &[u8]) -> u64 {
    for &b in input {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Function signature type for JIT-compiled functions.
pub type JitFunctionPointer = unsafe extern "C" fn() -> ();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llvm_jit_backend_creation() {
        let context = Context::create();
        let backend = LLVMJitBackend::new(&context);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_llvm_jit_backend_opt_levels() {
        let context = Context::create();
        for opt_level in &[
            OptimizationLevel::None,
            OptimizationLevel::Less,
            OptimizationLevel::Default,
            OptimizationLevel::Aggressive,
        ] {
            let backend = LLVMJitBackend::with_opt_level(&context, *opt_level);
            assert!(backend.is_ok());
            assert_eq!(backend.unwrap().optimization_level(), *opt_level);
        }
    }

    #[test]
    fn test_mangle_function_name_clean_pass_through() {
        assert_eq!(LLVMJitBackend::mangle_function_name("foo"), "foo");
        assert_eq!(LLVMJitBackend::mangle_function_name("a_b_c"), "a_b_c");
        // `$` is preserved (used by handler-op names).
        assert_eq!(
            LLVMJitBackend::mangle_function_name("Console$info"),
            "Console$info"
        );
    }

    #[test]
    fn test_mangle_function_name_parens_and_specials() {
        // The MCJIT-era format produced "func_HirId(42)" — parens are
        // linker-hostile.
        let mangled = LLVMJitBackend::mangle_function_name("func_HirId(42)");
        assert!(mangled.contains("_LP_"));
        assert!(mangled.contains("_RP_"));
        assert!(!mangled.contains('('));
        assert!(!mangled.contains(')'));
    }

    /// End-to-end AOT-via-object pipeline test: build a tiny HIR
    /// module containing `def add(a, b) { return a + b }`, run it
    /// through the full lower→opt→object→link→dlopen pipeline, then
    /// call the resulting function pointer and assert correctness.
    ///
    /// This is the single canonical proof that the AOT tier works
    /// when LLVM lowering doesn't trip on the existing prelude opaque-
    /// type issues that affect the bench kernels. Soft-falls if the
    /// host has no system linker (CI containers).
    #[test]
    fn test_aot_end_to_end_add() {
        use crate::hir::{
            BinaryOp, HirBlock, HirFunction, HirFunctionSignature, HirInstruction, HirParam,
            HirTerminator, HirType, HirValue, HirValueKind, ParamAttributes,
        };
        use std::collections::HashSet;
        use zyntax_typed_ast::InternedString;

        // Skip if there's no linker available — CI cells without a
        // toolchain still need to pass.
        if crate::llvm_link::find_linker().is_err() {
            eprintln!("test_aot_end_to_end_add: no linker available, skipping");
            return;
        }

        // Build `def add(a: i64, b: i64): i64 { return a + b }`.
        let p0_id = HirId::new();
        let p1_id = HirId::new();
        let sig = HirFunctionSignature {
            params: vec![
                HirParam {
                    id: p0_id,
                    name: InternedString::new_global("a"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                },
                HirParam {
                    id: p1_id,
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
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        };
        let mut func = HirFunction::new(InternedString::new_global("aot_add"), sig);
        func.values.insert(
            p0_id,
            HirValue {
                id: p0_id,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(0),
                uses: HashSet::new(),
                span: None,
            },
        );
        func.values.insert(
            p1_id,
            HirValue {
                id: p1_id,
                ty: HirType::I64,
                kind: HirValueKind::Parameter(1),
                uses: HashSet::new(),
                span: None,
            },
        );
        let sum_id = HirId::new();
        func.values.insert(
            sum_id,
            HirValue {
                id: sum_id,
                ty: HirType::I64,
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );
        let entry_id = func.entry_block;
        let entry: &mut HirBlock = func.blocks.get_mut(&entry_id).unwrap();
        entry.instructions.push(HirInstruction::Binary {
            result: sum_id,
            op: BinaryOp::Add,
            ty: HirType::I64,
            left: p0_id,
            right: p1_id,
        });
        entry.terminator = HirTerminator::Return {
            values: vec![sum_id],
        };

        let mut module = HirModule::new(InternedString::new_global("aot_test"));
        let func_id = func.id;
        module.functions.insert(func_id, func);

        // Drive the AOT pipeline.
        let context = Context::create();
        let mut backend = LLVMJitBackend::new(&context).expect("backend init");
        backend
            .compile_module(&module)
            .expect("AOT compile_module should succeed for trivial add");

        let ptr = backend
            .get_function_pointer(func_id)
            .expect("dlsym should produce a function pointer");
        let f: unsafe extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        let result = unsafe { f(7, 35) };
        assert_eq!(result, 42, "AOT-linked add(7, 35) should return 42");
    }
}
