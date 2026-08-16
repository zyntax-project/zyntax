//! Times decoding the standard library out of the snapshot.
//!
//! Installing a language decodes what it ships, and that is the
//! largest thing left in a cold start. This is the shape a profiler
//! should be pointed at.

use std::time::Instant;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let bytes = zynml::snapshot_bytes();
    eprintln!("snapshot is {} KB", bytes.len() / 1024);

    // Load parses the container and leaves the modules encoded.
    let t = Instant::now();
    for _ in 0..iters {
        let snapshot = zyntax_embed::Snapshot::load(bytes).expect("load");
        std::hint::black_box(snapshot.module_names().count());
    }
    let load = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Decoding one module is what installing used to do for all of them.
    let t = Instant::now();
    let mut decls = 0usize;
    for _ in 0..iters {
        let snapshot = zyntax_embed::Snapshot::load(bytes).expect("load");
        let prelude = snapshot
            .module("prelude")
            .expect("decode")
            .expect("present");
        decls += prelude.program().declarations.len();
    }
    let with_prelude = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // The grammar is the other half of what installing a language reads.
    let snapshot = zyntax_embed::Snapshot::load(bytes).expect("load");
    let grammar_bytes = snapshot.grammar_bytes().to_vec();
    let t = Instant::now();
    for _ in 0..iters {
        let grammar =
            zyntax_embed::LanguageGrammar::from_compiled_bytes(&grammar_bytes).expect("grammar");
        std::hint::black_box(grammar.name().len());
    }
    eprintln!(
        "grammar decode     {:.2} ms  ({} KB)",
        t.elapsed().as_secs_f64() * 1000.0 / iters as f64,
        grammar_bytes.len() / 1024
    );

    eprintln!(
        "load only          {load:.2} ms\n\
         load + prelude     {with_prelude:.2} ms\n\
         prelude decode     {:.2} ms  ({} declarations)",
        with_prelude - load,
        decls / iters
    );
}
