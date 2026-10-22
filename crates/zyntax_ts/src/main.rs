use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use zyntax_embed::{TieredConfig, TieredRuntime};
use zyntax_ts::{ErrorKind, Indexing, Options, ProjectError, Soundness};

#[cfg(feature = "allocation-audit")]
use zyntax_compiler::allocation_audit::{self, Phase};
#[cfg(feature = "allocation-audit")]
#[global_allocator]
static ALLOCATOR: allocation_audit::CountingAllocator = allocation_audit::CountingAllocator;

fn main() -> ExitCode {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(run)
        .expect("program thread")
        .join()
        .unwrap_or(ExitCode::from(4))
}

fn run() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();
    if command == "--version" {
        println!("zynode {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    let mut options = Options::default();
    let mut json = false;
    let mut llvm = false;
    let mut config = TieredConfig::default();
    let mut file = None;
    let mut cache_dir = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--soundness" => match args.next().as_deref() {
                Some("typescript") => options.soundness = Soundness::TypeScript,
                Some("checked") => options.soundness = Soundness::Checked,
                _ => return usage(),
            },
            "--indexing" => match args.next().as_deref() {
                Some("checked") => options.indexing = Indexing::Checked,
                Some("proven") => options.indexing = Indexing::Proven,
                _ => return usage(),
            },
            "--diagnostics=json" => json = true,
            "--cache-dir" => match args.next() {
                Some(path) => cache_dir = Some(PathBuf::from(path)),
                None => return usage(),
            },
            "--llvm" => llvm = true,
            "--no-osr" => config.enable_osr = false,
            "--baseline-threshold" => match args.next().and_then(|s| s.parse::<u32>().ok()) {
                Some(n) if n > 0 => config.baseline_threshold = n,
                _ => return usage(),
            },
            _ if !arg.starts_with('-') && file.is_none() => file = Some(PathBuf::from(arg)),
            _ => return usage(),
        }
    }
    if !matches!(command.as_str(), "check" | "run") {
        return usage();
    }
    let Some(file) = file else {
        return usage();
    };
    if llvm && command == "run" {
        #[cfg(feature = "llvm-backend")]
        {
            config.tier2_backend = zyntax_compiler::tiered_backend::Tier2Backend::LLVM;
        }
        #[cfg(not(feature = "llvm-backend"))]
        {
            eprintln!("zynode was built without llvm-backend");
            return ExitCode::from(2);
        }
    }
    // `ZYNTAX_TRACE_LOWER_PHASES=1` times frontend and runtime phases on stderr.
    let trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
    let mut phase = std::time::Instant::now();
    let mut lap = |name: &str| {
        if trace {
            eprintln!(
                "[ZYNODE] {name:<10} {:8.2} ms",
                phase.elapsed().as_secs_f64() * 1000.0
            );
        }
        phase = std::time::Instant::now();
    };
    #[cfg(feature = "allocation-audit")]
    allocation_audit::set_phase(Phase::Runtime);
    let mut runtime = if command == "run" {
        match TieredRuntime::new(config) {
            Ok(mut runtime) => {
                runtime.enter_only_through_entry_points();
                Some(runtime)
            }
            Err(error) => {
                eprintln!("zynode: {error}");
                return ExitCode::from(4);
            }
        }
    } else {
        None
    };
    lap("runtime");
    #[cfg(feature = "allocation-audit")]
    allocation_audit::set_phase(Phase::Parse);
    let using_cache = cache_dir.is_some();
    let mut preparation = None;
    let loaded = match cache_dir {
        Some(directory) => {
            if let Some(runtime) = runtime.as_mut() {
                zyntax_ts::load_project_for_runtime_background(&file, options, directory, runtime)
                    .map(|(project, _, worker)| {
                        preparation = worker;
                        project
                    })
            } else {
                zyntax_ts::load_project_cached(&file, options, directory)
                    .map(|(project, _)| project)
            }
        }
        None => zyntax_ts::load_project(&file, options),
    };
    let project = match loaded {
        Ok(project) => project,
        Err(ProjectError::Native(error)) => {
            eprintln!("zynode: {error}");
            return ExitCode::from(4);
        }
        Err(ProjectError::Io { file, error }) => {
            eprintln!("{}: {error}", file.display());
            return ExitCode::from(2);
        }
        Err(ProjectError::Diagnostic {
            file,
            source_code,
            error: e,
        }) => {
            let filename = file.to_string_lossy();
            if json {
                eprintln!(
                    "{}",
                    serde_json::json!({"kind": format!("{:?}", e.kind), "message": e.message, "file": filename, "start": e.start, "end": e.end})
                );
            } else {
                eprint!(
                    "{}",
                    e.render(
                        &filename,
                        &source_code,
                        zyntax_typed_ast::diagnostics::colors_enabled()
                    )
                );
            }
            return ExitCode::from(match e.kind {
                ErrorKind::Unsupported => 3,
                ErrorKind::Soundness => 5,
                _ => 1,
            });
        }
    };
    lap("project");
    if command == "check" {
        #[cfg(feature = "allocation-audit")]
        allocation_audit::report();
        return ExitCode::SUCCESS;
    }
    let mut runtime = runtime.expect("run command has a runtime");
    #[cfg(feature = "allocation-audit")]
    allocation_audit::set_phase(Phase::Register);
    let registered = if using_cache {
        Ok(())
    } else {
        zyntax_ts::register_runtime(&mut runtime)
    };
    lap("register");
    #[cfg(feature = "allocation-audit")]
    allocation_audit::set_phase(Phase::Lower);
    let result = registered.and_then(|()| {
        let (program, resolver) = project.into_parts();
        runtime.add_compiled_import_resolver(resolver);
        runtime.enter_only_through_entry_points();
        runtime.compile_typed_program(program)
    });
    lap("compile+run");
    let code = match result {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zynode: {e}");
            ExitCode::from(4)
        }
    };
    drop(preparation);
    runtime.stop();
    // Compiling workers can still hold code; process exit owns their storage.
    std::mem::forget(runtime);
    let _ = std::io::stdout().flush();
    lap("shutdown");
    #[cfg(feature = "allocation-audit")]
    allocation_audit::report();
    code
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: zynode <check|run> [--soundness typescript|checked] [--indexing checked|proven] [--cache-dir DIR] [--diagnostics=json] [--llvm] [--baseline-threshold N] [--no-osr] <file.ts>"
    );
    ExitCode::from(2)
}
