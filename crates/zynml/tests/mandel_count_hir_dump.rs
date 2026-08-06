use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir_dump::dump_module;
use zyntax_embed::ZyntaxRuntime;

#[test]
#[ignore = "diagnostic only"]
fn dump_mandel_count_hir() {
    let src = include_str!("../benchmarks/bench_mandelbrot.zynml");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<dump>").expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt.lower_typed_program(program, builtins).expect("lower");
    eprintln!("{}", dump_module(&module));
}
