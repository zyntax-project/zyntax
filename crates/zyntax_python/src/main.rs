//! `zypy`: compile a Python file and run it.
//!
//! The conformance suite spawns this and compares its stdout with
//! CPython's for the same file, so what this prints is the program's
//! output and nothing else; diagnostics go to stderr.

use std::path::PathBuf;
use std::process::ExitCode;
use zyntax_embed::{TieredConfig, TieredRuntime};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    // ZYPY_LLVM=1 selects LLVM tier-up; use it for checked benchmark runs.
    let llvm = cfg!(feature = "llvm-backend") && std::env::var_os("ZYPY_LLVM").is_some();
    if command.as_deref() == Some("backend") {
        println!("{}", if llvm { "llvm" } else { "cranelift" });
        return ExitCode::SUCCESS;
    }
    let path = match (command.as_deref(), args.next()) {
        (Some("run"), Some(p)) => PathBuf::from(p),
        _ => {
            eprintln!("usage: zypy run <file.py> [args...]");
            return ExitCode::from(2);
        }
    };
    // What the program sees as `sys.argv`: its own path, then the rest.
    let mut argv = vec![path.display().to_string()];
    argv.extend(args);
    zyntax_python::set_args(argv);
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("zypy: cannot read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    // A module the program imports is a file beside it, `a.b` at
    // `a/b.py`.
    let root = path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let resolve = |module: &str| {
        let mut file = root.clone();
        for part in module.split('.') {
            file.push(part);
        }
        file.set_extension("py");
        std::fs::read_to_string(&file).ok()
    };
    // `ZYNTAX_TRACE_LOWER_PHASES=1` times each step of a run on stderr.
    let trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
    let mut phase = std::time::Instant::now();
    let mut lap = |what: &str| {
        if trace {
            eprintln!(
                "[ZYPY] {what:<10} {:8.2} ms",
                phase.elapsed().as_secs_f64() * 1000.0
            );
        }
        phase = std::time::Instant::now();
    };
    let file = path.display().to_string();
    // `ZYPY_PROFILE_STARTUP=N` parses and compiles the program N more
    // times before the run, so a sampling profiler sees the start-up
    // path rather than the program; a measurement switch only.
    if let Some(n) = std::env::var("ZYPY_PROFILE_STARTUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        for _ in 0..n {
            let Ok(program) = zyntax_python::parse_program_with(&source, &file, &resolve) else {
                break;
            };
            let Ok(mut rt) = TieredRuntime::new(TieredConfig::default()) else {
                break;
            };
            if zyntax_python::register_runtime(&mut rt).is_err()
                || rt.compile_typed_program(program).is_err()
            {
                break;
            }
        }
    }
    let program = match zyntax_python::parse_program_with(&source, &file, &resolve) {
        Ok(p) => p,
        Err(e) => {
            // Shown against the file it happened in.
            let (name, text) = match e.module().and_then(|m| resolve(m).map(|t| (m, t))) {
                Some((module, text)) => (module.to_string(), text),
                None => (file.clone(), source.clone()),
            };
            // Colour follows the terminal, NO_COLOR and CLICOLOR_FORCE.
            let colors = zyntax_typed_ast::diagnostics::colors_enabled();
            eprint!("{}", e.render(&name, &text, colors));
            return ExitCode::from(3);
        }
    };
    lap("parse");
    // Only the LLVM build changes the default.
    #[allow(unused_mut)]
    let mut config = TieredConfig::default();
    #[cfg(feature = "llvm-backend")]
    if llvm {
        config.tier2_backend = zyntax_compiler::tiered_backend::Tier2Backend::LLVM;
    }
    let mut rt = match TieredRuntime::new(config) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("zypy: runtime: {e}");
            return ExitCode::from(4);
        }
    };
    lap("runtime");
    if let Err(e) = zyntax_python::register_runtime(&mut rt) {
        eprintln!("zypy: runtime: {e}");
        return ExitCode::from(4);
    }
    lap("register");
    if let Err(e) = rt.compile_typed_program(program) {
        eprintln!("zypy: compile: {e}");
        return ExitCode::from(3);
    }
    lap("compile");
    let outcome = rt.call_raw(zyntax_python::ENTRY, &[]);
    lap("run");
    let code = match outcome {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("zypy: {e}");
            1
        }
    };
    // The runtime's own threads are told to stop, not waited for: what
    // it holds is the process's and goes with it.
    rt.stop();
    lap("shutdown");
    exit_now(rt, code)
}

/// End the process at once, the runtime left where it stands: no
/// destructor runs, so a compile thread still in LLVM is killed rather
/// than racing the statics an orderly exit would tear down under it.
#[cfg(unix)]
fn exit_now(rt: TieredRuntime, code: i32) -> ExitCode {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    std::mem::forget(rt);
    // SAFETY: `_exit` ends the process without returning; nothing after
    // it runs, and the streams were flushed above.
    unsafe { libc::_exit(code) }
}

#[cfg(not(unix))]
fn exit_now(rt: TieredRuntime, code: i32) -> ExitCode {
    drop(rt);
    ExitCode::from(code as u8)
}
