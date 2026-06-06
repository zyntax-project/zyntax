//! Diagnostic: does method inlining actually fire on the bench
//! kernels? Reports the `InlineStats` for each.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::inline;
use zyntax_embed::ZyntaxRuntime;

fn report(label: &str, source: &str) {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(source, "<probe>")
        .expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut module = rt.lower_typed_program(program, builtins).expect("lower");
    let before: Vec<_> = module
        .functions
        .values()
        .filter_map(|f| f.name.resolve_global())
        .collect();
    let stats = inline::run_module(&mut module);
    eprintln!("=== {label} ===");
    eprintln!("functions: {before:?}");
    eprintln!("inline stats: {stats:?}");
}

#[test]
#[ignore = "diagnostic only"]
fn inline_check_all_benches() {
    report(
        "mandelbrot",
        include_str!("../examples/bench_mandelbrot.zynml"),
    );
    report("nbody", include_str!("../examples/bench_nbody.zynml"));
    report("fib", include_str!("../examples/bench_fib.zynml"));
}
