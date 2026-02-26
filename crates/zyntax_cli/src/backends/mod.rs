//! Backend compilation (Cranelift and LLVM)

pub mod cranelift_jit;
pub mod llvm_aot;

use log::info;
use std::path::PathBuf;
use zyntax_compiler::hir::HirModule;
use zyntax_compiler::zpack::ZPack;
use zyntax_typed_ast::{EntryPointResolver, ModuleArchitecture};

pub use cranelift_jit::compile_jit;
pub use llvm_aot::{compile_and_run_llvm, compile_llvm};

/// Backend type (compiler infrastructure)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Cranelift - Fast compilation, good for development
    Cranelift,
    /// LLVM - Maximum optimization, good for production
    Llvm,
}

impl Backend {
    pub fn from_str(s: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match s.to_lowercase().as_str() {
            "cranelift" => Ok(Backend::Cranelift),
            "llvm" => Ok(Backend::Llvm),
            _ => Err(format!("Unknown backend: {}. Use: cranelift, llvm", s).into()),
        }
    }
}

/// Compile HIR module with the specified backend
///
/// - `jit`: If true, JIT compile and run immediately
/// - If false, compile to native executable (AOT)
/// - `entry_point`: Custom entry point (e.g., "main", "MyClass.run", "com.example.Main.start")
///   Falls back to "main" if not specified.
/// - `resolver_arch`: Module architecture for entry point resolution
/// - `packs`: Loaded ZPack archives containing runtime symbols (JIT mode only)
/// - `static_libs`: Static libraries to link (AOT mode only - users provide these directly)
pub fn compile(
    module: HirModule,
    backend: Backend,
    output: Option<PathBuf>,
    opt_level: u8,
    jit: bool,
    entry_point: Option<String>,
    resolver_arch: &ModuleArchitecture,
    packs: &[ZPack],
    static_libs: &[PathBuf],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create entry point resolver based on module architecture
    let entry_resolver = EntryPointResolver::new(resolver_arch.clone());
    let entry = entry_point
        .as_deref()
        .unwrap_or(entry_resolver.default_entry());
    let candidates = entry_resolver.resolve_candidates(entry);

    // Collect runtime symbols from all packs (for JIT mode)
    let mut pack_symbols: Vec<(&'static str, *const u8)> = Vec::new();
    let mut pack_symbols_with_sigs: Vec<zyntax_compiler::zrtl::RuntimeSymbolInfo> = Vec::new();
    for pack in packs.iter() {
        pack_symbols.extend(pack.runtime_symbols());
        pack_symbols_with_sigs.extend_from_slice(pack.runtime_symbols_with_signatures());
    }

    if verbose && !pack_symbols.is_empty() {
        info!("Loaded {} symbols from zpacks", pack_symbols.len());
    }

    match (backend, jit) {
        // JIT execution - uses dynamic runtime symbols from ZPack
        (Backend::Cranelift, true) => compile_jit(
            module,
            opt_level,
            true,
            &candidates,
            &pack_symbols,
            &pack_symbols_with_sigs,
            verbose,
        ),
        (Backend::Llvm, true) => {
            compile_and_run_llvm(module, opt_level, Some(entry), &pack_symbols, verbose)?;
            Ok(())
        }
        // AOT compilation - users link static libraries directly
        (Backend::Cranelift, false) => compile_jit(
            module,
            opt_level,
            false,
            &candidates,
            &pack_symbols,
            &pack_symbols_with_sigs,
            verbose,
        ),
        (Backend::Llvm, false) => {
            // Static libraries are provided directly by the user via --lib flag
            // ZPack is for JIT only - AOT users link their runtime libraries directly
            compile_llvm(
                module,
                output,
                opt_level,
                Some(entry),
                &pack_symbols,
                static_libs,
                verbose,
            )
        }
    }
}

/// Compile and run HIR module for REPL, returning the result value
pub fn compile_and_run_repl(
    module: HirModule,
    backend: &Backend,
    opt_level: u8,
    verbose: bool,
) -> Result<i64, Box<dyn std::error::Error>> {
    // REPL uses default entry point
    match backend {
        Backend::Cranelift => cranelift_jit::compile_and_run_repl(module, opt_level, verbose),
        Backend::Llvm => compile_and_run_llvm(module, opt_level, None, &[], verbose),
    }
}
