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
    let program = match zyntax_python::parse_program_with(&source, &resolve) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("zypy: {e}");
            return ExitCode::from(3);
        }
    };
    let mut rt = match TieredRuntime::new(TieredConfig::default()) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("zypy: runtime: {e}");
            return ExitCode::from(4);
        }
    };
    if let Err(e) = zyntax_python::register_runtime(&mut rt) {
        eprintln!("zypy: runtime: {e}");
        return ExitCode::from(4);
    }
    if let Err(e) = rt.compile_typed_program(program) {
        eprintln!("zypy: compile: {e}");
        return ExitCode::from(3);
    }
    match rt.call_raw(zyntax_python::ENTRY, &[]) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zypy: {e}");
            ExitCode::from(1)
        }
    }
}
