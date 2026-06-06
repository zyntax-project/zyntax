use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

#[ignore = "kernel probe — slow; run with cargo test -- --ignored"]
#[test]
fn fib_kernel_runs() {
    let src = include_str!("../examples/bench_fib.zynml");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<probe>").expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt.lower_typed_program(program, builtins).expect("lower");
    let mut rt = ZyntaxRuntime::new().expect("rt");
    rt.compile_module(&module).expect("compile");
    let r = rt.call_function_raw("main", vec![]).expect("call main");
    eprintln!("fib result = {r:?}");
}
