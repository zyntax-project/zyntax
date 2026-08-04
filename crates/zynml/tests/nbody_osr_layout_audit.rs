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
    let src = std::fs::read_to_string("examples/bench_nbody.zynml")
        .or_else(|_| std::fs::read_to_string("crates/zynml/examples/bench_nbody.zynml"))
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
        for header in headers {
            let idx = osr::block_index_of(func, header).unwrap_or(u64::MAX);
            match osr::osr_layout(func, header) {
                Ok(layout) => {
                    println!(
                        "  header {idx}: OK — {} live-ins ({} phi)",
                        layout.live_ins.len(),
                        layout.phi_count
                    );
                }
                Err(reason) => {
                    println!("  header {idx}: {reason:?}");
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
