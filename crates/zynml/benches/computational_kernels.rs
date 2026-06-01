//! Criterion benches for two heavy-arithmetic ZynML kernels:
//! `bench_mandelbrot.zynml` and `bench_nbody.zynml`.
//!
//! Each kernel is measured along three axes so we can see where the
//! HIR optimization pipeline pays off:
//!
//!   1. `lower-only`     — cost of parse + lowering, no HIR opts.
//!                          This is the floor; useful when something
//!                          regresses to tell whether the cost is in
//!                          the front-end or the optimiser.
//!   2. `lower+opt`       — parse + lowering + `run_interp_safe_opts`.
//!                          Difference from `lower-only` shows the
//!                          fixed-cost of the optimisation pipeline.
//!   3. `run`             — full pipeline (lower + opt) PLUS executing
//!                          `main()` through the BC interpreter. The
//!                          difference between `run-opt-off` and
//!                          `run-opt-on` is the runtime payoff of the
//!                          HIR passes.
//!
//! Both kernels stay small enough that the BC interpreter returns in
//! tens of ms — Criterion picks a sample count that keeps each bench
//! under a few seconds. The values they return are pinned by the
//! per-pass bisect harness at
//! `crates/zynml/tests/bench_kernel_bisect.rs` so a future
//! optimization regression that silently changes semantics fails
//! loudly there.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::{run_interp_safe_opts, HirModule};
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

const MANDELBROT_SRC: &str = include_str!("../examples/bench_mandelbrot.zynml");
const NBODY_SRC: &str = include_str!("../examples/bench_nbody.zynml");

/// Lower a ZynML source the same way the runtime does, but stop at
/// the raw `HirModule` so the harness can apply (or skip) opts.
fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(source, "<bench>")
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

/// Install `module` into a fresh runtime and call `main()`.
fn run(module: HirModule) -> ZyntaxValue {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    rt.compile_module(&module).expect("install");
    rt.call_function_raw("main", vec![]).expect("call main")
}

// ─── Benchmarks ──────────────────────────────────────────────────

/// Bench harness for one kernel. Adds six benchmarks under a group
/// named `kernel`: `lower-only`, `lower+opt`, `run-noopt`,
/// `run-opt`, plus a baseline `run-noopt-twice` to catch
/// measurement noise.
fn bench_kernel(c: &mut Criterion, kernel: &str, source: &str) {
    let mut g = c.benchmark_group(kernel);
    // The BC interp isn't free; keep per-bench cost bounded.
    g.sample_size(20);
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));

    g.bench_function("lower-only", |b| {
        b.iter(|| {
            let m = lower(black_box(source));
            black_box(m);
        });
    });

    g.bench_function("lower+opt", |b| {
        b.iter(|| {
            let mut m = lower(black_box(source));
            let _ = run_interp_safe_opts(&mut m);
            black_box(m);
        });
    });

    g.bench_function("run-noopt", |b| {
        b.iter(|| {
            let m = lower(black_box(source));
            let r = run(m);
            black_box(r);
        });
    });

    g.bench_function("run-opt", |b| {
        b.iter(|| {
            let mut m = lower(black_box(source));
            let _ = run_interp_safe_opts(&mut m);
            let r = run(m);
            black_box(r);
        });
    });

    g.finish();
}

fn bench_mandelbrot(c: &mut Criterion) {
    bench_kernel(c, "mandelbrot", MANDELBROT_SRC);
}

fn bench_nbody(c: &mut Criterion) {
    bench_kernel(c, "nbody", NBODY_SRC);
}

criterion_group!(kernels, bench_mandelbrot, bench_nbody);
criterion_main!(kernels);
