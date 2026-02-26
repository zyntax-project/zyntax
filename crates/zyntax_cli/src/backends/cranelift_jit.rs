//! Cranelift JIT backend compilation
//!
//! The Cranelift backend is frontend-agnostic. Runtime symbols are loaded
//! exclusively from ZPack archives, not compiled into the CLI.

use colored::Colorize;
use log::info;
use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::HirModule;

/// Compile HIR module with Cranelift JIT backend
///
/// - `entry_candidates`: Pre-resolved entry point candidates from EntryPointResolver
///   These are in order of preference (most likely first).
/// - `pack_symbols`: Runtime symbols from loaded ZPack archives
/// - `pack_symbols_with_sigs`: Runtime symbols with signature information (for auto-boxing)
pub fn compile_jit(
    module: HirModule,
    _opt_level: u8,
    run: bool,
    entry_candidates: &[String],
    pack_symbols: &[(&'static str, *const u8)],
    pack_symbols_with_sigs: &[zyntax_compiler::zrtl::RuntimeSymbolInfo],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Runtime symbols come exclusively from ZPack archives
    // This keeps the compiler frontend-agnostic
    let runtime_symbols: Vec<(&'static str, *const u8)> = pack_symbols.to_vec();

    if runtime_symbols.is_empty() && verbose {
        info!("No runtime symbols loaded - external calls may fail");
        info!("For JIT with runtime support, use: --pack <runtime.zpack>");
    }

    if verbose {
        println!(
            "{} Loaded {} runtime symbols from zpacks",
            "info:".blue(),
            runtime_symbols.len()
        );
        for (name, _) in &runtime_symbols {
            println!("  - {}", name);
        }
    }

    let mut backend = CraneliftBackend::with_runtime_symbols(&runtime_symbols)
        .map_err(|e| format!("Failed to initialize backend: {}", e))?;

    // Register symbol signatures for auto-boxing support
    backend.register_symbol_signatures(pack_symbols_with_sigs);

    if verbose {
        println!("{} Compiling functions...", "info:".blue());
    }

    backend
        .compile_module(&module)
        .map_err(|e| format!("Compilation failed: {}", e))?;

    // Finalize definitions to make function pointers available
    backend
        .finalize_definitions()
        .map_err(|e| format!("Failed to finalize: {}", e))?;

    if run {
        if verbose {
            println!("{} Looking for entry point...", "info:".green().bold());
        }

        execute_entry(&backend, &module, entry_candidates, verbose)?;
    } else {
        println!("{} Compilation successful", "success:".green().bold());
    }

    Ok(())
}

/// Find and execute the entry point function
///
/// Takes pre-resolved candidates from EntryPointResolver.
/// Candidates are in order of preference (most likely first).
fn execute_entry(
    backend: &CraneliftBackend,
    module: &HirModule,
    candidates: &[String],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if verbose {
        println!(
            "{} Looking for entry point candidates: {:?}",
            "debug:".cyan(),
            candidates
        );
    }

    // Find entry function by name - try all candidates in order
    let (entry_id, matched_name) = module
        .functions
        .iter()
        .find_map(|(id, func)| {
            func.name.resolve_global().and_then(|name| {
                candidates
                    .iter()
                    .find(|c| c.as_str() == name)
                    .map(|matched| (*id, matched.clone()))
            })
        })
        .ok_or_else(|| {
            // List available functions in error message for debugging
            let available: Vec<String> = module
                .functions
                .values()
                .filter_map(|f| f.name.resolve_global())
                .collect();
            format!(
                "No entry point found. Tried: {:?}. Available functions: {:?}",
                candidates, available
            )
        })?;

    // Get the function for return type info
    let entry_fn = module
        .functions
        .get(&entry_id)
        .ok_or_else(|| format!("Entry function '{}' not found", matched_name))?;

    // Get function pointer
    let fn_ptr = backend
        .get_function_ptr(entry_id)
        .ok_or_else(|| format!("Failed to get '{}' function pointer", matched_name))?;

    if verbose {
        println!(
            "{} Executing {}() at {:?}...",
            "info:".cyan(),
            matched_name,
            fn_ptr
        );
    }

    // Determine return type and call function
    let _result = unsafe {
        if entry_fn.signature.returns.is_empty() {
            // Void return
            let f: fn() = std::mem::transmute(fn_ptr);
            f();
            println!("{} {}() completed", "result:".green().bold(), matched_name);
            0
        } else {
            match &entry_fn.signature.returns[0] {
                zyntax_compiler::hir::HirType::I32 => {
                    let f: fn() -> i32 = std::mem::transmute(fn_ptr);
                    let ret = f();
                    println!(
                        "{} {}() returned: {}",
                        "result:".green().bold(),
                        matched_name,
                        ret
                    );
                    ret as i64
                }
                zyntax_compiler::hir::HirType::I64 => {
                    let f: fn() -> i64 = std::mem::transmute(fn_ptr);
                    let ret = f();
                    println!(
                        "{} {}() returned: {}",
                        "result:".green().bold(),
                        matched_name,
                        ret
                    );
                    ret
                }
                zyntax_compiler::hir::HirType::F32 => {
                    let f: fn() -> f32 = std::mem::transmute(fn_ptr);
                    let ret = f();
                    println!(
                        "{} {}() returned: {}",
                        "result:".green().bold(),
                        matched_name,
                        ret
                    );
                    0
                }
                zyntax_compiler::hir::HirType::F64 => {
                    let f: fn() -> f64 = std::mem::transmute(fn_ptr);
                    let ret = f();
                    println!(
                        "{} {}() returned: {}",
                        "result:".green().bold(),
                        matched_name,
                        ret
                    );
                    0
                }
                zyntax_compiler::hir::HirType::Void => {
                    let f: fn() = std::mem::transmute(fn_ptr);
                    f();
                    println!("{} {}() completed", "result:".green().bold(), matched_name);
                    0
                }
                _ => {
                    return Err(format!(
                        "Unsupported {} return type: {:?}",
                        matched_name, entry_fn.signature.returns[0]
                    )
                    .into());
                }
            }
        }
    };

    if verbose {
        println!(
            "{} Execution completed with code: {}",
            "info:".green(),
            _result
        );
    }

    Ok(())
}

/// Compile and run for REPL mode - returns the result value directly
///
/// Note: REPL mode runs without runtime symbols. For full functionality,
/// users should use `zyntax compile --jit --pack <runtime.zpack>` instead.
pub fn compile_and_run_repl(
    module: HirModule,
    _opt_level: u8,
    verbose: bool,
) -> Result<i64, Box<dyn std::error::Error>> {
    // REPL runs without runtime symbols - only basic expressions work
    // For full runtime support, use compile --jit --pack <runtime.zpack>
    let runtime_symbols: Vec<(&'static str, *const u8)> = Vec::new();

    if verbose {
        info!("REPL mode: No runtime symbols (use --pack for full runtime)");
    }

    let mut backend = CraneliftBackend::with_runtime_symbols(&runtime_symbols)
        .map_err(|e| format!("Failed to initialize backend: {}", e))?;

    backend
        .compile_module(&module)
        .map_err(|e| format!("Compilation failed: {}", e))?;

    backend
        .finalize_definitions()
        .map_err(|e| format!("Failed to finalize: {}", e))?;

    // Find and execute main function, returning the result
    execute_main_repl(&backend, &module, verbose)
}

/// Execute main function for REPL and return the result value
fn execute_main_repl(
    backend: &CraneliftBackend,
    module: &HirModule,
    verbose: bool,
) -> Result<i64, Box<dyn std::error::Error>> {
    // Find main function by name
    let (main_id, main_fn) = module
        .functions
        .iter()
        .find(|(_, func)| {
            func.name
                .resolve_global()
                .map(|name| name == "main")
                .unwrap_or(false)
        })
        .ok_or("No 'main' function found in module")?;

    let main_id = *main_id;

    let fn_ptr = backend
        .get_function_ptr(main_id)
        .ok_or("Failed to get main function pointer")?;

    if verbose {
        println!("{} Executing at {:?}...", "info:".cyan(), fn_ptr);
    }

    // Execute and return value
    let result = unsafe {
        if main_fn.signature.returns.is_empty() {
            let f: fn() = std::mem::transmute(fn_ptr);
            f();
            0
        } else {
            match &main_fn.signature.returns[0] {
                zyntax_compiler::hir::HirType::I32 => {
                    let f: fn() -> i32 = std::mem::transmute(fn_ptr);
                    f() as i64
                }
                zyntax_compiler::hir::HirType::I64 => {
                    let f: fn() -> i64 = std::mem::transmute(fn_ptr);
                    f()
                }
                zyntax_compiler::hir::HirType::F32 => {
                    let f: fn() -> f32 = std::mem::transmute(fn_ptr);
                    f() as i64
                }
                zyntax_compiler::hir::HirType::F64 => {
                    let f: fn() -> f64 = std::mem::transmute(fn_ptr);
                    f() as i64
                }
                zyntax_compiler::hir::HirType::Void => {
                    let f: fn() = std::mem::transmute(fn_ptr);
                    f();
                    0
                }
                _ => {
                    return Err(format!(
                        "Unsupported return type: {:?}",
                        main_fn.signature.returns[0]
                    )
                    .into());
                }
            }
        }
    };

    Ok(result)
}
