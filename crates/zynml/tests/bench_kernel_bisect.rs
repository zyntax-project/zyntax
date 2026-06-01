//! Bisection probe for which opt pass breaks the bench kernels.
//! Each test applies a SINGLE opt pass to one of the kernels then
//! executes through the BC interp under a hard 5s wall-clock budget
//! (cargo test default timeout is per-process, so we add our own
//! per-test guard via a thread join with timeout).

use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::{
    alloca_promote, cfg_simplify, const_fold, cse, drop_insert, hir::HirModule, inline, licm,
    load_cse, loop_vectorize, reduction_vectorize,
};
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

const MANDEL: &str = include_str!("../examples/bench_mandelbrot.zynml");
const NBODY: &str = include_str!("../examples/bench_nbody.zynml");

fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(source, "<bisect>")
        .expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rt.lower_typed_program(program, builtins).expect("lower")
}

fn run_with_timeout(module: HirModule, secs: u64) -> Option<ZyntaxValue> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut rt = ZyntaxRuntime::new().expect("rt");
        rt.compile_module(&module).expect("install");
        let r = rt.call_function_raw("main", vec![]).expect("call");
        let _ = tx.send(r);
    });
    rx.recv_timeout(Duration::from_secs(secs)).ok()
}

fn bisect(source: &str, label: &str) {
    // Baseline — no opts.
    let baseline = run_with_timeout(lower(source), 10)
        .expect("baseline run-noopt must finish — if this hangs, the kernel itself is broken");
    eprintln!("[{label}] noopt baseline = {baseline:?}");

    // Per-pass: apply ONE pass, then run with a tight budget.
    let passes: &[(&str, fn(&mut HirModule))] = &[
        ("alloca_promote", |m| {
            let _ = alloca_promote::run_module(m);
        }),
        ("const_fold", |m| {
            let _ = const_fold::fold_module(m);
        }),
        ("cse", |m| {
            let _ = cse::eliminate_module(m);
        }),
        ("load_cse", |m| {
            let _ = load_cse::run_module(m);
        }),
        ("inline", |m| {
            let _ = inline::run_module(m);
        }),
        ("licm", |m| {
            let _ = licm::run_module(m);
        }),
        ("loop_vectorize", |m| {
            let _ = loop_vectorize::run_module(m);
        }),
        ("reduction_vectorize", |m| {
            let _ = reduction_vectorize::run_module(m);
        }),
        ("cfg_simplify", |m| {
            let _ = cfg_simplify::run_module(m);
        }),
        ("drop_insert", |m| {
            let _ = drop_insert::run_module(m);
        }),
    ];

    for (name, apply) in passes {
        let mut m = lower(source);
        apply(&mut m);
        match run_with_timeout(m, 5) {
            Some(r) => eprintln!("[{label}] {name} OK = {r:?}"),
            None => panic!("[{label}] {name} HUNG (>5s)"),
        }
    }
}

#[test]
fn mandelbrot_bisect() {
    bisect(MANDEL, "mandelbrot");
}

#[test]
fn nbody_bisect() {
    bisect(NBODY, "nbody");
}
