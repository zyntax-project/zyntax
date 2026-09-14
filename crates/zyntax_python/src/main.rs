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
    let path = match (args.next().as_deref(), args.next()) {
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
    let mut rt = match TieredRuntime::new(TieredConfig::default()) {
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
    match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zypy: {e}");
            ExitCode::from(1)
        }
    }
}
