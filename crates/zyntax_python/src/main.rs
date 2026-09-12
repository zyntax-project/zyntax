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
            eprintln!("usage: zypy run <file.py>");
            return ExitCode::from(2);
        }
    };
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("zypy: cannot read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };
    let program = match zyntax_python::parse_program(&source) {
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
    // The IO plugin, for `print`. Looked for beside the workspace's
    // plugin build, or where ZYPY_PLUGINS says.
    let plugins = std::env::var("ZYPY_PLUGINS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../plugins/target/zrtl")
        });
    if let Err(e) = rt.load_plugin(plugins.join("zrtl_io.zrtl")) {
        eprintln!(
            "zypy: cannot load the IO plugin from {}: {e}",
            plugins.display()
        );
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
