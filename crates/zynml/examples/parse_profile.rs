//! Parses the prelude in a loop so a profiler has something to sample.
//!
//! The prelude is the largest single parse any compile does, and the
//! one every program pays, so it is the shape worth sampling.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE};

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let t = std::time::Instant::now();
    let mut decls = 0usize;
    for _ in 0..iters {
        let program = grammar
            .parse_with_filename(ZYNML_STDLIB_PRELUDE, "prelude.zynml")
            .expect("parse prelude");
        decls += program.declarations.len();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "{iters} parses of {} bytes: {ms:.1} ms total, {:.2} ms each, {:.0} KB/s (decls={decls})",
        ZYNML_STDLIB_PRELUDE.len(),
        ms / iters as f64,
        (ZYNML_STDLIB_PRELUDE.len() as f64 * iters as f64) / (ms / 1000.0) / 1024.0,
    );
}
