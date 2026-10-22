use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use zyntax_embed::{TieredConfig, TieredRuntime};
use zyntax_ts::{ErrorKind, Options, ProjectError, Soundness};

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
    if command == "check" {
        return ExitCode::SUCCESS;
    }
    let mut runtime = runtime.expect("run command has a runtime");
    let registered = if using_cache {
        Ok(())
    } else {
        zyntax_ts::register_runtime(&mut runtime)
    };
    let result = registered.and_then(|()| {
        let (program, resolver) = project.into_parts();
        runtime.add_compiled_import_resolver(resolver);
        runtime.enter_only_through_entry_points();
        runtime.compile_typed_program(program)
    });
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
    code
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: zynode <check|run> [--soundness typescript|checked] [--cache-dir DIR] [--diagnostics=json] [--llvm] [--baseline-threshold N] [--no-osr] <file.ts>"
    );
    ExitCode::from(2)
}
