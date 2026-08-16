//! How much of compiling a program is the program?
//!
//! A snapshot ships the standard library already parsed. If compiling a
//! one-line program costs about what compiling a real one costs, then
//! the work is not the program: it is the library being handled again
//! for every compile.

use std::time::Instant;

fn main() {
    let trivial = "fn main() -> Int {\n    return 1\n}\n";
    let real = std::fs::read_to_string("crates/zynml/examples/hello.zynml")
        .unwrap_or_else(|_| trivial.to_string());

    eprintln!(
        "{:<9} {:>8} {:>8} {:>8} {:>8}  {}",
        "source", "setup", "parse", "lower", "declared", "bytes"
    );
    // Three rounds, alternating, because the first compile in a process
    // pays for warm-up that has nothing to do with either program.
    for round in 0..3 {
        for (name, source) in [("real", real.as_str()), ("trivial", trivial)] {
            phases(&format!("{name}{round}"), source);
        }
    }
}

/// Split one compile the way the bench harness does, so the parse and
/// lower halves can be read apart.
fn phases(name: &str, source: &str) {
    let t = Instant::now();
    let zynml = zynml::ZynML::new().expect("runtime");
    let setup = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let program = zynml
        .grammar2()
        .expect("compiled parser")
        .parse_with_filename(source, "<probe>")
        .expect("parse");
    let parse = t.elapsed().as_secs_f64() * 1000.0;
    let declared = program.declarations.len();

    let builtins = zynml
        .runtime()
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let t = Instant::now();
    let module = zynml
        .runtime()
        .lower_typed_program(program, builtins)
        .expect("lower");
    let lower = t.elapsed().as_secs_f64() * 1000.0;

    // What codegen keeps, against what lowering produced.
    let reachable = zyntax_compiler::reachable_function_ids(&module, &["main"]);

    eprintln!(
        "{name:<9} {setup:>7.2}ms {parse:>7.2}ms {lower:>7.2}ms {declared:>8}  \
         {} lowered, {} reachable ({} bytes)",
        module.functions.len(),
        reachable.len(),
        source.len(),
    );

    if std::env::var_os("SHOW_NAMES").is_some() {
        let mut kept: Vec<String> = Vec::new();
        let mut dropped: Vec<String> = Vec::new();
        for (id, f) in &module.functions {
            let n = f.name.resolve_global().unwrap_or_default();
            if reachable.contains(id) {
                kept.push(n);
            } else {
                dropped.push(n);
            }
        }
        kept.sort();
        dropped.sort();
        eprintln!("  reachable: {kept:?}");
        eprintln!("  dropped ({}): {:?}", dropped.len(), &dropped);
    }
}
