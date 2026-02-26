//! LLVM AOT backend compilation
//!
//! Compiles HIR modules to native executables using LLVM's optimizing compiler.
//! This backend produces highly optimized machine code suitable for production use.

use colored::Colorize;
use log::{debug, error, info};
use std::path::{Path, PathBuf};
use zyntax_compiler::hir::HirModule;

/// Standard library search paths for different platforms
#[cfg(all(feature = "llvm-backend", target_os = "macos"))]
const LIBRARY_SEARCH_PATHS: &[&str] = &[
    "/usr/local/lib",
    "/opt/homebrew/lib",
    "/usr/lib",
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib",
];

#[cfg(all(feature = "llvm-backend", target_os = "linux"))]
const LIBRARY_SEARCH_PATHS: &[&str] = &[
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/lib/x86_64-linux-gnu",
    "/lib/aarch64-linux-gnu",
];

#[cfg(all(feature = "llvm-backend", target_os = "windows"))]
const LIBRARY_SEARCH_PATHS: &[&str] = &["C:\\Windows\\System32", "C:\\Program Files\\Common Files"];

#[cfg(all(
    feature = "llvm-backend",
    not(any(target_os = "macos", target_os = "linux", target_os = "windows"))
))]
const LIBRARY_SEARCH_PATHS: &[&str] = &["/usr/local/lib", "/usr/lib"];

/// Resolve a library path, searching standard locations if needed
///
/// If the path exists as-is, returns it unchanged.
/// If not, searches standard library paths for:
/// - The exact name
/// - lib{name}.a (Unix static library convention)
/// - {name}.lib (Windows static library convention)
#[cfg(feature = "llvm-backend")]
fn resolve_library_path(lib: &Path, verbose: bool) -> Option<PathBuf> {
    // If it's already an absolute path or exists, use it directly
    if lib.is_absolute() || lib.exists() {
        return Some(lib.to_path_buf());
    }

    let lib_name = lib.file_name()?.to_str()?;

    // Generate possible library file names
    let candidates: Vec<String> = if lib_name.starts_with("lib") && lib_name.ends_with(".a") {
        // Already in lib*.a format
        vec![lib_name.to_string()]
    } else if lib_name.ends_with(".a") || lib_name.ends_with(".lib") {
        // Has extension but no lib prefix
        vec![lib_name.to_string(), format!("lib{}", lib_name)]
    } else {
        // Bare name - try common conventions
        vec![
            format!("lib{}.a", lib_name), // Unix: libfoo.a
            format!("{}.lib", lib_name),  // Windows: foo.lib
            format!("{}.a", lib_name),    // Alternative: foo.a
            lib_name.to_string(),         // Exact name
        ]
    };

    if verbose {
        info!("Searching for library '{}' in standard paths...", lib_name);
    }

    // Search each path for each candidate
    for search_path in LIBRARY_SEARCH_PATHS {
        let search_dir = Path::new(search_path);
        if !search_dir.exists() {
            continue;
        }

        for candidate in &candidates {
            let full_path = search_dir.join(candidate);
            if full_path.exists() {
                if verbose {
                    info!("Found library: {:?}", full_path);
                }
                return Some(full_path);
            }
        }
    }

    if verbose {
        info!("Library '{}' not found in standard paths", lib_name);
    }

    None
}

/// Compile HIR module with LLVM AOT backend
#[cfg(feature = "llvm-backend")]
pub fn compile_llvm(
    module: HirModule,
    output: Option<PathBuf>,
    opt_level: u8,
    _entry_point: Option<&str>,
    _pack_symbols: &[(&'static str, *const u8)],
    static_libs: &[PathBuf],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use inkwell::context::Context;
    use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    };
    use inkwell::OptimizationLevel;
    use zyntax_compiler::llvm_backend::LLVMBackend;

    let output_path = output.unwrap_or_else(|| PathBuf::from("a.out"));

    if verbose {
        info!("Initializing LLVM backend...");
    }

    // Initialize LLVM targets
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| format!("Failed to initialize LLVM target: {}", e))?;

    // Create LLVM context and backend
    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "zyntax_aot");

    if verbose {
        info!("Compiling HIR to LLVM IR...");
    }

    // Compile HIR module to LLVM IR
    let llvm_ir = backend
        .compile_module(&module)
        .map_err(|e| format!("Compilation failed: {}", e))?;

    if verbose {
        info!("Generated {} bytes of LLVM IR", llvm_ir.len());
        debug!("LLVM IR:\n{}", llvm_ir);
        // Also write LLVM IR to file for debugging
        let ir_path = output_path.with_extension("ll");
        std::fs::write(&ir_path, &llvm_ir).unwrap_or_else(|e| {
            error!("Failed to write LLVM IR: {}", e);
        });
        info!("Wrote LLVM IR to {:?}", ir_path);
    }

    // Get target triple for the host machine
    let triple = TargetMachine::get_default_triple();
    let target =
        Target::from_triple(&triple).map_err(|e| format!("Failed to get target: {}", e))?;

    // Map optimization level
    let llvm_opt = match opt_level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        _ => OptimizationLevel::Aggressive,
    };

    if verbose {
        info!(
            "Target: {}, Optimization: {:?}",
            triple.as_str().to_str().unwrap_or("unknown"),
            llvm_opt
        );
    }

    // Create target machine
    let target_machine = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            llvm_opt,
            RelocMode::Default,
            CodeModel::Default,
        )
        .ok_or("Failed to create target machine")?;

    // Write object file
    let obj_path = output_path.with_extension("o");

    if verbose {
        info!("Writing object file to {:?}...", obj_path);
    }

    target_machine
        .write_to_file(backend.module(), FileType::Object, &obj_path)
        .map_err(|e| format!("Failed to write object file: {}", e))?;

    // Link to create executable
    if verbose {
        info!("Linking executable...");
    }

    // Build linker command
    let mut linker = std::process::Command::new("cc");
    linker.arg(&obj_path);
    linker.arg("-o");
    linker.arg(&output_path);

    // Resolve and add static libraries
    // Libraries can be specified as:
    // - Full path: /path/to/libfoo.a
    // - Library name: foo (searches for libfoo.a in standard paths)
    // - Partial name: libfoo.a (searches in standard paths)
    let mut resolved_count = 0;
    for lib_path in static_libs {
        if let Some(resolved) = resolve_library_path(lib_path, verbose) {
            if verbose {
                info!("Linking static library: {:?}", resolved);
            }
            linker.arg(&resolved);
            resolved_count += 1;
        } else {
            // Library not found - pass to linker anyway, let it report the error
            // This allows using -l style library names that the linker can resolve
            let lib_str = lib_path.to_string_lossy();
            if !lib_str.starts_with('-') {
                // Pass as -l flag for the linker to search
                let lib_name = lib_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.strip_prefix("lib").unwrap_or(s))
                    .unwrap_or(&lib_str);
                if verbose {
                    info!(
                        "Library '{}' not found, passing -l{} to linker",
                        lib_path.display(),
                        lib_name
                    );
                }
                linker.arg(format!("-l{}", lib_name));
            } else {
                linker.arg(lib_path);
            }
        }
    }

    // If no static libs provided, note that external symbols may fail
    if static_libs.is_empty() {
        if verbose {
            info!("No static libraries specified - external symbols may cause linker errors");
            info!("For AOT with runtime support, use: --lib <library>");
        }
    } else if verbose {
        info!(
            "Resolved {} of {} libraries",
            resolved_count,
            static_libs.len()
        );
    }

    let status = linker
        .status()
        .map_err(|e| format!("Failed to run linker: {}", e))?;

    if !status.success() {
        return Err(format!("Linker failed with status: {}", status).into());
    }

    // Clean up object file
    let _ = std::fs::remove_file(&obj_path);

    println!(
        "{} Successfully compiled to {}",
        "success:".green().bold(),
        output_path.display()
    );

    Ok(())
}

/// Compile HIR module with LLVM AOT backend (stub when feature not enabled)
#[cfg(not(feature = "llvm-backend"))]
pub fn compile_llvm(
    _module: HirModule,
    output: Option<PathBuf>,
    _opt_level: u8,
    _entry_point: Option<&str>,
    _pack_symbols: &[(&'static str, *const u8)],
    _static_libs: &[PathBuf],
    _verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = output.unwrap_or_else(|| PathBuf::from("a.out"));

    error!("LLVM backend not enabled");
    error!("Rebuild with: cargo build --release --features llvm-backend");
    debug!("Output would be: {}", output_path.display());

    Err("LLVM backend not enabled. Rebuild with --features llvm-backend".into())
}

/// Compile and run with LLVM JIT backend
#[cfg(feature = "llvm-backend")]
pub fn compile_and_run_llvm(
    module: HirModule,
    opt_level: u8,
    entry_point: Option<&str>,
    pack_symbols: &[(&'static str, *const u8)],
    verbose: bool,
) -> Result<i64, Box<dyn std::error::Error>> {
    use inkwell::context::Context;
    use inkwell::OptimizationLevel;
    use zyntax_compiler::llvm_jit_backend::LLVMJitBackend;

    if verbose {
        info!("Initializing LLVM JIT backend...");
    }

    // Runtime symbols come exclusively from ZPack archives
    // This keeps the compiler frontend-agnostic - no built-in runtimes
    let runtime_symbols: Vec<(&'static str, *const u8)> = pack_symbols.to_vec();

    if runtime_symbols.is_empty() {
        info!("No runtime symbols loaded - external calls will fail");
        info!("For JIT with runtime support, use: --pack <runtime.zpack>");
    }

    if verbose {
        info!(
            "Loaded {} runtime symbols from zpacks",
            runtime_symbols.len()
        );
        for (name, _) in &runtime_symbols {
            debug!("  - {}", name);
        }
    }

    // Create LLVM context
    let context = Context::create();

    // Map optimization level
    let llvm_opt = match opt_level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        _ => OptimizationLevel::Aggressive,
    };

    // Create JIT backend
    let mut backend = LLVMJitBackend::with_opt_level(&context, llvm_opt)
        .map_err(|e| format!("Failed to create JIT backend: {}", e))?;

    // Register runtime symbols with the LLVM JIT backend
    for (name, ptr) in &runtime_symbols {
        backend.register_symbol(*name, *ptr);
    }

    if verbose {
        info!("Compiling module with LLVM JIT...");
    }

    // Compile module
    backend
        .compile_module(&module)
        .map_err(|e| format!("JIT compilation failed: {}", e))?;

    // Find main function
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

    // Get function pointer
    let fn_ptr = backend
        .get_function_pointer(main_id)
        .ok_or("Failed to get main function pointer")?;

    debug!("Got main function pointer: {:p}", fn_ptr);
    debug!("About to execute main()...");

    // Execute main function
    // IMPORTANT: Use extern "C" calling convention to match LLVM's default
    let result = unsafe {
        if main_fn.signature.returns.is_empty() {
            let f: extern "C" fn() = std::mem::transmute(fn_ptr);
            f();
            0
        } else {
            match &main_fn.signature.returns[0] {
                zyntax_compiler::hir::HirType::I32 => {
                    let f: extern "C" fn() -> i32 = std::mem::transmute(fn_ptr);
                    f() as i64
                }
                zyntax_compiler::hir::HirType::I64 => {
                    let f: extern "C" fn() -> i64 = std::mem::transmute(fn_ptr);
                    f()
                }
                zyntax_compiler::hir::HirType::F32 => {
                    let f: extern "C" fn() -> f32 = std::mem::transmute(fn_ptr);
                    f() as i64
                }
                zyntax_compiler::hir::HirType::F64 => {
                    let f: extern "C" fn() -> f64 = std::mem::transmute(fn_ptr);
                    f() as i64
                }
                zyntax_compiler::hir::HirType::Void => {
                    let f: extern "C" fn() = std::mem::transmute(fn_ptr);
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

    println!("{} main() returned: {}", "result:".green().bold(), result);

    Ok(result)
}

/// Compile and run with LLVM JIT backend (stub when feature not enabled)
#[cfg(not(feature = "llvm-backend"))]
pub fn compile_and_run_llvm(
    _module: HirModule,
    _opt_level: u8,
    _entry_point: Option<&str>,
    _pack_symbols: &[(&'static str, *const u8)],
    _verbose: bool,
) -> Result<i64, Box<dyn std::error::Error>> {
    error!("LLVM JIT backend not enabled");
    error!("Rebuild with: cargo build --release --features llvm-backend");

    Err("LLVM backend not enabled. Rebuild with --features llvm-backend".into())
}
