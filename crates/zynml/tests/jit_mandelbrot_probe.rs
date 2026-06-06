//! Diagnostic probe: drive mandelbrot through the tiered runtime
//! (BC interp → Cranelift JIT) and report every call's result so
//! we can see when the wrong answer appears.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::profiling::ProfileConfig;
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::ZyntaxRuntime;

#[ignore = "kernel probe — slow; run with cargo test -- --ignored"]
#[test]
#[ignore = "diagnostic only — reports tier-up behavior call by call"]
fn mandelbrot_tiered_call_sequence() {
    let src = include_str!("../examples/bench_mandelbrot.zynml");
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
    let mut cfg = TieredConfig::default();
    cfg.profile_config = ProfileConfig {
        warm_threshold: 1,
        hot_threshold: 5,
        ..ProfileConfig::default()
    };
    rt.install_interp_jit_with(cfg).expect("install_jit");
    for i in 0..20 {
        let r = rt.call_function_raw("main", vec![]).expect("call");
        eprintln!("call {i:2} = {r:?}");
    }
}
