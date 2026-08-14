//! Parses and lowers a kernel in a loop, without executing it, so a
//! profiler samples the front half of the pipeline rather than the
//! kernel's own run time.
//!
//! Execution dominates a bench iteration by one to two orders of
//! magnitude, which buries parse, type checking and lowering in a
//! sampling profile taken over the whole run.

use zynml::ZynML;

fn main() {
    let mut args = std::env::args().skip(1);
    let kernel = args.next().unwrap_or_else(|| "bench_nbody".to_string());
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(20);

    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("benchmarks/{kernel}.zynml"));
    let source = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));

    let mut setup_total = 0.0;
    let mut parse_total = 0.0;
    let mut lower_total = 0.0;
    let mut decls = 0usize;

    for _ in 0..iters {
        let t = std::time::Instant::now();
        let zynml = ZynML::new().expect("runtime");
        let this_setup = t.elapsed().as_secs_f64() * 1000.0;
        if setup_total == 0.0 {
            // The first construction carries the process-wide
            // initialisation the rest reuse, so it is reported apart
            // from the average rather than folded into it.
            eprintln!("  first runtime construction: {this_setup:.2} ms");
        }
        setup_total += this_setup;
        let t = std::time::Instant::now();
        let program = zynml
            .grammar2()
            .expect("grammar")
            .parse_with_filename(&source, "<profile>")
            .expect("parse");
        parse_total += t.elapsed().as_secs_f64() * 1000.0;

        let builtins = zynml
            .runtime()
            .config()
            .builtins
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let t = std::time::Instant::now();
        let module = zynml
            .runtime()
            .lower_typed_program(program, builtins)
            .expect("lower");
        lower_total += t.elapsed().as_secs_f64() * 1000.0;
        decls += module.functions.len();
    }

    eprintln!(
        "{kernel}: {iters} iterations, {} bytes\n  runtime_new = {:.2} ms each\n  parse = {:.2} ms each\n  lower = {:.2} ms each\n  (fns={})",
        source.len(),
        setup_total / iters as f64,
        parse_total / iters as f64,
        lower_total / iters as f64,
        decls / iters,
    );
}
