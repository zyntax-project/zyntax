//! Which benchmark loops the safety gate would actually let us spread.
//!
//! A dispatch is worth only what it can be applied to, so this reports
//! coverage over the real kernels rather than invented ones.
//!
//! **What it showed, and why the dispatch is not built yet.** The gate
//! accepts elementwise compute loops (`axpy`, `vadd`) and setup fills.
//! It does NOT accept the matrix multiply, whose three accepted loops
//! are all fills: `Tensor$zeros`, `Tensor$fill`, and one in `main`. Its
//! compute loop addresses memory as `row * cols + col`, and the gate
//! only accepts a subscript that is the induction variable itself.
//!
//! That is the wrong half of the coverage. Measured on this machine at
//! ten cores, spreading an elementwise kernel buys 1.1x to 2.4x because
//! it is bandwidth-bound, while the matrix multiply buys 4.3x at 512
//! and 6.0x at 1024. So the gate covers the loops where threading does
//! not pay and misses the one where it does. Accepting affine
//! subscripts is what makes a dispatch worth building.

use std::path::Path;
use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::parallel_safe;
use zyntax_embed::ZyntaxRuntime;

#[test]
fn report_parallel_coverage() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "zynml"))
        .collect();
    files.sort();

    println!(
        "\n  {:<30}{:>12}{:>12}{:>10}",
        "kernel", "independent", "carried", "opaque"
    );
    println!("  {}", "-".repeat(64));
    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        let Ok(grammar) = Grammar2::from_source(ZYNML_GRAMMAR) else {
            continue;
        };
        let Ok(program) = grammar.parse_with_filename(&src, "<cov>") else {
            continue;
        };
        let Ok(mut rt) = ZyntaxRuntime::new() else {
            continue;
        };
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
        let Ok(mut module) = rt.lower_typed_program(program, builtins) else {
            continue;
        };
        zyntax_compiler::run_interp_safe_opts(&mut module);
        let s = parallel_safe::analyze_module(&module);
        println!(
            "  {:<30}{:>12}{:>12}{:>10}",
            f.file_stem().unwrap().to_string_lossy(),
            s.independent,
            s.carried_dependency,
            s.opaque_body
        );
        // Which function each independent loop is in, since a setup
        // fill running once is not worth spreading and a compute loop
        // is.
        for func in module.functions.values() {
            let (found, _) = parallel_safe::analyze(func);
            for p in &found {
                println!(
                    "        in {}: reads {} writes {}",
                    func.name.resolve_global().unwrap_or_default(),
                    p.reads.len(),
                    p.writes.len()
                );
            }
        }
    }
}
