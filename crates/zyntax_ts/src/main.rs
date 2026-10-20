use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use zyntax_embed::{TieredConfig, TieredRuntime};
use zyntax_ts::{ErrorKind, Options, Soundness};

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
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--soundness" => match args.next().as_deref() {
                Some("typescript") => options.soundness = Soundness::TypeScript,
                Some("checked") => options.soundness = Soundness::Checked,
                _ => return usage(),
            },
            "--diagnostics=json" => json = true,
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
    let source = match std::fs::read_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", file.display());
            return ExitCode::from(2);
        }
    };
    let filename = file.to_string_lossy();
    let program = match zyntax_ts::parse_program(&source, &filename, options) {
        Ok(program) => program,
        Err(e) => {
            if json {
                eprintln!(
                    "{}",
                    serde_json::json!({"kind": format!("{:?}", e.kind), "message": e.message, "start": e.start, "end": e.end})
                );
            } else {
                eprint!(
                    "{}",
                    e.render(
                        &filename,
                        &source,
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
    if llvm {
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
    let mut runtime = match TieredRuntime::new(config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("zynode: {e}");
            return ExitCode::from(4);
        }
    };
    let result = zyntax_ts::register_runtime(&mut runtime)
        .and_then(|()| {
            runtime.enter_only_through_entry_points();
            runtime.compile_typed_program(program)
        })
        .and_then(|_| runtime.call_raw(zyntax_ts::ENTRY, &[]));
    let code = match result {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zynode: {e}");
            ExitCode::from(4)
        }
    };
    runtime.stop();
    // Compiling workers can still hold code; process exit owns their storage.
    std::mem::forget(runtime);
    let _ = std::io::stdout().flush();
    code
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: zynode <check|run> [--soundness typescript|checked] [--diagnostics=json] [--llvm] [--baseline-threshold N] [--no-osr] <file.ts>"
    );
    ExitCode::from(2)
}
