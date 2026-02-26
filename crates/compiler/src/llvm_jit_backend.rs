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
    targets::{InitializationConfig, Target},
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

        // Step 2: Collect external function declarations from the module BEFORE consuming it
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

        // Step 3: Consume the module and create execution engine
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
