//! What the optimized HIR hands OSR at each of nbody's loop headers.
//!
//! Diagnostic: prints every header, its live-in count, and each live-in's
//! type with whether it fits an i64 slot — the two things `osr_layout`
//! rejects on.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::osr;
use zyntax_embed::ZyntaxRuntime;

#[test]
#[ignore = "diagnostic only"]
fn audit_nbody_osr_layouts() {
    let src = std::fs::read_to_string("benchmarks/bench_nbody.zynml")
        .or_else(|_| std::fs::read_to_string("crates/zynml/benchmarks/bench_nbody.zynml"))
        .expect("nbody source");

    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(&src, "<audit>").expect("parse");
    let mut rt = ZyntaxRuntime::new().expect("rt");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt.lower_typed_program(program, builtins).expect("lower");

    for func in module.functions.values() {
        let headers = osr::find_loop_headers(func);
        if headers.is_empty() {
            continue;
        }
        let name = func.name.resolve_global().unwrap_or_default();
        println!("\n=== {name} ({} blocks) ===", func.blocks.len());
        // A header is outermost when no other header dominates it.
        let dominated_by: Vec<(u64, bool)> = headers
            .iter()
            .map(|h| {
                let idx = osr::block_index_of(func, *h).unwrap_or(u64::MAX);
                let nested = headers
                    .iter()
                    .any(|other| other != h && osr::blocks_dominated_by(func, *other).contains(h));
                (idx, nested)
            })
            .collect();
        for header in headers {
            let idx = osr::block_index_of(func, header).unwrap_or(u64::MAX);
            let nested = dominated_by
                .iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, n)| *n)
                .unwrap_or(false);
            let depth = if nested { "inner" } else { "OUTERMOST" };
            match osr::osr_layout(func, header) {
                Ok(layout) => {
                    println!(
                        "  header {idx} [{depth}]: OK — {} live-ins ({} phi), frame {} bytes",
                        layout.live_ins.len(),
                        layout.phi_count,
                        layout.frame.size
                    );
                }
                Err(reason) => {
                    println!("  header {idx} [{depth}]: {reason:?}");
                    // Distinguish the coarse guard (a block in the region is
                    // also entered from outside) from the conflict it exists
                    // to prevent (a value the frame supplies is redefined by
                    // a phi inside the region, so the seeded value would be
                    // shadowed).
                    let region = osr::blocks_reachable_from(func, header);
                    println!(
                        "      region via cached successors: {} blocks (function has {})",
                        region.len(),
                        func.blocks.len()
                    );
                    let dominated = osr::blocks_dominated_by(func, header);
                    let mut external_entries = Vec::new();
                    let mut shadowed = Vec::new();
                    for b in &region {
                        let Some(blk) = func.blocks.get(b) else {
                            continue;
                        };
                        if *b != header
                            && blk
                                .predecessors
                                .iter()
                                .any(|p| !region.contains(p) && *p != header)
                        {
                            external_entries.push(*b);
                        }
                        if !dominated.contains(b) {
                            for phi in &blk.phis {
                                shadowed.push((*b, phi.result));
                            }
                        }
                    }
                    println!(
                        "      blocks entered from outside: {}, phis outside the dominated set: {}",
                        external_entries.len(),
                        shadowed.len()
                    );
                    // Show what the live-ins actually are so the rejection
                    // reason can be tied to a concrete type.
                    if let Some(block) = func.blocks.get(&header) {
                        for phi in &block.phis {
                            let fits = osr::type_fits_i64(&phi.ty);
                            println!(
                                "      phi {:?}: {:?}{}",
                                phi.result,
                                phi.ty,
                                if fits { "" } else { "   <-- does not fit" }
                            );
                        }
                    }
                }
            }
        }
    }
}
