//! `zylua`: compile a Lua file and run it.
//!
//! The conformance suite spawns this and compares its stdout with
//! Lua's for the same file, so what this prints is the program's
//! output and nothing else; diagnostics go to stderr.

use std::path::PathBuf;
use std::process::ExitCode;
use zyntax_embed::{TieredConfig, TieredRuntime};

/// The stack the program runs on: Lua recursion is bounded by its own
/// limit, not the main thread's.
const STACK_BYTES: usize = 1 << 30;

fn main() -> ExitCode {
    let run = std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(run)
        .expect("the program's thread");
    match run.join() {
        Ok(code) => code,
        Err(_) => ExitCode::from(134),
    }
}

const USAGE: &str = "usage: zylua [run] [-e stat] [file.lua [args...]]";

fn run() -> ExitCode {
    let mut args = std::env::args().skip(1).peekable();
    // ZYLUA_LLVM=1 selects LLVM tier-up; use it for checked benchmark runs.
    let llvm = cfg!(feature = "llvm-backend") && std::env::var_os("ZYLUA_LLVM").is_some();
    if args.peek().map(String::as_str) == Some("backend") {
        println!("{}", if llvm { "llvm" } else { "cranelift" });
        return ExitCode::SUCCESS;
    }
    if args.peek().map(String::as_str) == Some("run") {
        args.next();
    }
    // `-e stat` runs `stat` ahead of the chunk, as `lua -e` does; the
    // statement joins the chunk's first line so nothing below moves.
    // Without a file, the statements are the whole program.
    let mut prelude = String::new();
    let mut next = args.next();
    while next.as_deref() == Some("-e") {
        match args.next() {
            Some(stat) => {
                prelude.push_str(&stat);
                prelude.push_str("; ");
            }
            None => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        }
        next = args.next();
    }
    if next.as_deref() == Some("--") {
        next = args.next();
    }
    let path = match next {
        Some(p) => Some(PathBuf::from(p)),
        None if !prelude.is_empty() => None,
        None => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    // What the program sees as `arg`: the interpreter at -1, its own
    // path at 0, then the rest.
    let interpreter = std::env::args()
        .next()
        .unwrap_or_else(|| "zylua".to_string());
    let mut argv = vec![interpreter];
    if let Some(path) = &path {
        argv.push(path.display().to_string());
    }
    argv.extend(args);
    zyntax_lua::set_args(argv);
    let (source, file) = match &path {
        Some(path) => match std::fs::read(path) {
            Ok(bytes) => (
                prelude + &zyntax_lua::source_text(&bytes),
                path.display().to_string(),
            ),
            Err(e) => {
                eprintln!("zylua: cannot read {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        None => (prelude, "=(command line)".to_string()),
    };
    // `ZYNTAX_TRACE_LOWER_PHASES=1` times each step of a run on stderr.
    let trace = std::env::var_os("ZYNTAX_TRACE_LOWER_PHASES").is_some();
    let mut phase = std::time::Instant::now();
    let mut lap = |what: &str| {
        if trace {
            eprintln!(
                "[ZYLUA] {what:<10} {:8.2} ms",
                phase.elapsed().as_secs_f64() * 1000.0
            );
        }
        phase = std::time::Instant::now();
    };
    let program = match zyntax_lua::parse_program(&source, &file) {
        Ok(p) => p,
        Err(e) => {
            // Colour follows the terminal, NO_COLOR and CLICOLOR_FORCE.
            let colors = zyntax_typed_ast::diagnostics::colors_enabled();
            eprint!("{}", e.render(&file, &source, colors));
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
            eprintln!("zylua: runtime: {e}");
            return ExitCode::from(4);
        }
    };
    lap("runtime");
    if let Err(e) = zyntax_lua::register_runtime(&mut rt) {
        eprintln!("zylua: runtime: {e}");
        return ExitCode::from(4);
    }
    zyntax_lua::set_runtime(&mut rt);
    lap("register");
    if let Err(e) = rt.compile_typed_program(program) {
        eprintln!("zylua: compile: {e}");
        return ExitCode::from(3);
    }
    lap("compile");
    let outcome = rt.call_raw(zyntax_lua::ENTRY, &[]);
    lap("run");
    match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("zylua: {e}");
            ExitCode::from(1)
        }
    }
}
