//! # LLVM JIT Backend using MCJIT
//!
//! This backend uses LLVM's Modern JIT (MCJIT) for dynamic compilation with maximum optimization.
//! It's designed to be used as Tier 2 in the tiered compilation system for hot functions.
//!
//! ## Features
//! - On-demand compilation with full LLVM optimizations
//! - Function pointer retrieval for direct calls
//! - Memory-efficient: only compiles functions that are actually hot
//! - Uses LLVMBackend for HIR → LLVM IR translation
//!
//! ## vs AOT LLVM Backend
//! - AOT (`llvm_backend.rs`): Compiles entire program to object files/executables
//! - JIT (this file): Compiles individual functions to memory at runtime
//!
//! Both use the same HIR → LLVM IR translation logic.
//!
//! ## MCJIT Architecture Note
//! MCJIT requires the module to be fully populated before the execution engine is created.
//! We compile HIR → LLVM IR first, then create the execution engine from the populated module.

use crate::hir::{HirFunction, HirId, HirModule};
use crate::llvm_backend::LLVMBackend;
use crate::{CompilerError, CompilerResult};
use indexmap::IndexMap;
use inkwell::{
    context::Context,
    execution_engine::ExecutionEngine,
    passes::PassBuilderOptions,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
    OptimizationLevel,
};

/// LLVM JIT backend using MCJIT
///
/// This backend compiles HIR modules to native code using LLVM's MCJIT.
/// The execution engine is created lazily after the module is compiled.
pub struct LLVMJitBackend<'ctx> {
    /// LLVM context reference
    context: &'ctx Context,

    /// Execution engine (MCJIT) - created after module compilation
    execution_engine: Option<ExecutionEngine<'ctx>>,

    /// Function pointers cache (stored as usize for thread safety)
    function_pointers: IndexMap<HirId, usize>,

    /// Optimization level
    opt_level: OptimizationLevel,

    /// Runtime symbols to register with the execution engine
    /// Maps function name to function pointer address
    runtime_symbols: IndexMap<String, usize>,

    /// Symbol signatures for auto-boxing (symbol name → signature)
    symbol_signatures: Vec<crate::zrtl::RuntimeSymbolInfo>,
}

impl<'ctx> LLVMJitBackend<'ctx> {
    /// Create a new LLVM JIT backend with aggressive optimization
    pub fn new(context: &'ctx Context) -> CompilerResult<Self> {
        Self::with_opt_level(context, OptimizationLevel::Aggressive)
    }

    /// Create with custom optimization level
    pub fn with_opt_level(
        context: &'ctx Context,
        opt_level: OptimizationLevel,
    ) -> CompilerResult<Self> {
        // Initialize LLVM targets
        Target::initialize_native(&InitializationConfig::default()).map_err(|e| {
            CompilerError::Backend(format!("Failed to initialize LLVM target: {}", e))
        })?;

        // Link in MCJIT
        ExecutionEngine::link_in_mc_jit();

        Ok(Self {
            context,
            execution_engine: None,
            function_pointers: IndexMap::new(),
            opt_level,
            runtime_symbols: IndexMap::new(),
            symbol_signatures: Vec::new(),
        })
    }

    /// Register symbol signatures for auto-boxing support
    pub fn register_symbol_signatures(&mut self, symbols: &[crate::zrtl::RuntimeSymbolInfo]) {
        self.symbol_signatures.extend(symbols.iter().cloned());
    }

    /// Register a runtime symbol that will be available to JIT-compiled code
    ///
    /// Call this before `compile_module` to make external functions available.
    pub fn register_symbol(&mut self, name: impl Into<String>, ptr: *const u8) {
        self.runtime_symbols.insert(name.into(), ptr as usize);
    }

    /// Register multiple runtime symbols at once
    pub fn register_symbols(&mut self, symbols: &[(&str, *const u8)]) {
        for (name, ptr) in symbols {
            self.runtime_symbols
                .insert((*name).to_string(), *ptr as usize);
        }
    }

    /// Build `PassBuilderOptions` for LLVM optimization passes.
    ///
    /// Loop vectorization (LV) and SLP vectorization stay on — those
    /// are the headline x86_64 wins for the numerical kernels, and
    /// rayzor's tuning at compiler/src/codegen/llvm_jit_backend.rs:349
    /// keeps them on as well. Both passes have their own trip-count
    /// and cost gates, so they don't blow up on huge loops.
    ///
    /// `set_loop_unrolling(true)` is deliberately NOT set. Under the
    /// `default<O3>` pipeline, forcing unrolling on overrides LLVM's
    /// own size/trip-count heuristic — on ZynML kernels with very
    /// large trip counts (e.g. inlined_call's 100_000_000-iter loop
    /// with an inlinable callee) the forced unroller tries to fully
    /// unroll and the JIT pipeline hangs indefinitely. Without this
    /// override, `default<O3>` still runs the standard unroller with
    /// its built-in trip-count cap.
    ///
    /// `set_loop_interleaving(true)` is also left off because it
    /// composes with the unroller, and `merge_functions(true)` is on
    /// because it's cheap and shrinks dispatch tables for
    /// monomorphised generic code.
    fn create_pass_options() -> PassBuilderOptions {
        let opts = PassBuilderOptions::create();
        opts.set_loop_vectorization(true);
        opts.set_loop_slp_vectorization(true);
        opts.set_merge_functions(true);
        opts
    }

    /// Compile a full HIR module
    ///
    /// Translates all functions to LLVM IR and JIT compiles them.
    /// Function pointers become available via get_function_pointer().
    ///
    /// IMPORTANT: The execution engine is created AFTER compilation to ensure
    /// MCJIT sees all functions in the module.
    pub fn compile_module(&mut self, hir_module: &HirModule) -> CompilerResult<()> {
        // Step 1: Create backend and compile HIR → LLVM IR
        let mut backend = LLVMBackend::new(self.context, "zyntax_jit");

        // Register symbol signatures for auto-boxing
        backend.register_symbol_signatures(&self.symbol_signatures);

        let _llvm_ir = backend.compile_module(hir_module)?;

        // Step 2: Tune the module + MCJIT for the host CPU. Without this,
        // MCJIT's internal TargetMachine uses generic x86_64 defaults
        // (no FMA, no AVX2, conservative instruction scheduling). Setting
        // the triple + data layout + host CPU/features and running an
        // explicit pass pipeline before engine creation matches rayzor's
        // JIT tuning at compiler/src/codegen/llvm_jit_backend.rs:216-357.
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
                RelocMode::Default,
                CodeModel::Default,
            )
            .ok_or_else(|| CompilerError::Backend("Failed to create host target machine".into()))?;
        backend.module().set_triple(&target_triple);
        backend
            .module()
            .set_data_layout(&target_machine.get_target_data().get_data_layout());

        // Step 3: Run an explicit optimization pipeline. While LLVM's
        // `default<O3>` enables most of these by default, explicit flags
        // ensure they fire on every host — the implicit defaults are more
        // conservative on x86_64 Linux than on aarch64 macOS (per
        // rayzor llvm_jit_backend.rs:346-348). Skipped at OptLevel::None
        // so unoptimised builds stay fast.
        // Pre-pass verification: catch HIR-to-LLVM-IR lowering bugs
        // before they get amplified by O3 transforms. `Module::verify`
        // returns a multi-line diagnostic if any function is malformed
        // (wrong block terminators, type mismatches across calls,
        // undefined values flowing into a phi, etc.). The runtime hang
        // we hit on inlined_call's 100M FP loop made it clear we need
        // this gate — a buggy IR may silently optimise into an
        // infinite loop instead of crashing.
        if let Err(msg) = backend.module().verify() {
            return Err(CompilerError::Backend(format!(
                "LLVM module verification failed (pre-opt): {}",
                msg.to_string()
            )));
        }

        // Optional IR dump (pre-opt) for debugging codegen bugs.
        if std::env::var("ZYNTAX_DUMP_LLVM_IR").is_ok() {
            let ir = backend.module().print_to_string().to_string();
            let path = std::env::temp_dir().join("zyntax_llvm_ir_preopt.ll");
            if std::fs::write(&path, &ir).is_ok() {
                eprintln!(
                    "[LLVM-IR] pre-opt IR written to {} ({} bytes)",
                    path.display(),
                    ir.len()
                );
            }
        }

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
                    CompilerError::Backend(format!("LLVM optimization passes failed: {}", e))
                })?;

            // Post-pass verification: if the optimiser produced an
            // invalid module (rare but real — happens when our IR has
            // subtle UB the optimiser exploits), surface it here
            // instead of handing broken code to MCJIT.
            if let Err(msg) = backend.module().verify() {
                return Err(CompilerError::Backend(format!(
                    "LLVM module verification failed (post-opt): {}",
                    msg.to_string()
                )));
            }

            // Optional post-opt IR dump.
            if std::env::var("ZYNTAX_DUMP_LLVM_IR").is_ok() {
                let ir = backend.module().print_to_string().to_string();
                let path = std::env::temp_dir().join("zyntax_llvm_ir_postopt.ll");
                if std::fs::write(&path, &ir).is_ok() {
                    eprintln!(
                        "[LLVM-IR] post-opt IR written to {} ({} bytes)",
                        path.display(),
                        ir.len()
                    );
                }
            }
        }

        // Step 4: Collect external function declarations from the module BEFORE consuming it
        // We need the function values for add_global_mapping
        let mut external_functions: Vec<(String, inkwell::values::FunctionValue<'ctx>)> =
            Vec::new();
        for (_, function) in &hir_module.functions {
            if function.is_external {
                if let Some(name) = function.name.resolve_global() {
                    if let Some(llvm_func) = backend.module().get_function(&name) {
                        external_functions.push((name, llvm_func));
                    }
                }
            }
        }

        // Step 5: Consume the module and create execution engine
        let module = backend.into_module();
        let execution_engine = module
            .create_jit_execution_engine(self.opt_level)
            .map_err(|e| {
                CompilerError::Backend(format!("Failed to create JIT execution engine: {}", e))
            })?;

        // Step 4: Register runtime symbols with the execution engine using add_global_mapping
        for (name, llvm_func) in &external_functions {
            if let Some(addr) = self.runtime_symbols.get(name) {
                execution_engine.add_global_mapping(llvm_func, *addr);
                log::debug!(
                    "Registered runtime symbol '{}' at address 0x{:x}",
                    name,
                    addr
                );
            }
        }

        // Step 5: Extract function pointers from the execution engine
        for (id, function) in &hir_module.functions {
            // Skip external functions - they don't have compiled code in this module
            if function.is_external {
                continue;
            }

            // Must match naming logic in llvm_backend.rs:
            // - Main function uses actual name for entry point
            // - Other functions use mangled name with HirId
            let actual_name = function
                .name
                .resolve_global()
                .unwrap_or_else(|| format!("{:?}", function.name));
            let fn_name = if actual_name == "main" {
                actual_name
            } else {
                format!("func_{:?}", id)
            };

            // Get function address from JIT execution engine
            let fn_ptr = execution_engine
                .get_function_address(&fn_name)
                .map_err(|e| {
                    CompilerError::Backend(format!(
                        "Failed to get function address for '{}': {:?}",
                        fn_name, e
                    ))
                })?;

            // Cache the pointer (stored as usize for thread safety)
            self.function_pointers.insert(*id, fn_ptr as usize);
        }

        // Store execution engine (keeps JIT code alive)
        self.execution_engine = Some(execution_engine);

        Ok(())
    }

    /// Compile a single function
    ///
    /// Note: For JIT use, prefer compile_module() which handles forward references correctly.
    /// Single-function compilation may fail if the function references other functions.
    pub fn compile_function(&mut self, id: HirId, function: &HirFunction) -> CompilerResult<()> {
        // Create a temporary module with just this function
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

        // Compile it
        self.compile_module(&temp_module)?;

        Ok(())
    }

    /// Get a function pointer
    pub fn get_function_pointer(&self, func_id: HirId) -> Option<*const u8> {
        self.function_pointers
            .get(&func_id)
            .map(|&addr| addr as *const u8)
    }

    /// Get optimization level
    pub fn optimization_level(&self) -> OptimizationLevel {
        self.opt_level
    }
}

/// Function signature type for JIT-compiled functions
/// This is a type alias for function pointers returned by the JIT
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

        // Test all optimization levels
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
}
