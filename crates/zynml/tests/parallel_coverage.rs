//! Which benchmark loops the safety gate would actually let us spread.
//!
//! A dispatch is worth only what it can be applied to, so this reports
//! coverage over the real kernels rather than invented ones.
//!
//! Two columns, not one. A loop can be independent as written, or
//! independent provided something the function cannot see for itself
//! holds, and the second is only worth as much as the dispatch site's
//! willingness to establish what it names. Reporting them together
//! would make the gate look wider than it is.
//!
//! **What it shows.** Elementwise kernels and setup fills are
//! independent outright. The matrix multiplies are independent against
//! named obligations: a row loop writing `i * cols + j` is a band per
//! row, but nothing inside the function says the output's column count
//! and the counter's limit are the same number, nor that the operands
//! are different storage.
//!
//! The prefill kernel is the one to read. It is a whole transformer
//! layer rather than one shape in isolation, and its two matrix
//! multiplies qualify at both levels, which is where spreading work
//! actually pays: measured on this machine at ten cores, an elementwise
//! kernel buys 1.1x to 2.4x because it is bandwidth-bound, while a
//! matrix multiply buys 4.3x at 512 and 6.0x at 1024.
//!
//! Note what `gemm` asks for against what the tensor matmul asks for.
//! Written with its dimensions as parameters, one value serves as both
//! the output row's width and the inner counter's limit, so there is
//! nothing to establish; written against a struct carrying its own
//! shape, the two arrive as separate fields and have to be matched at
//! the dispatch. Passing dimensions is the more analysable program, and
//! the difference is visible in the columns rather than in an opinion.

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
        "\n  {:<28}{:>12}{:>12}{:>10}{:>8}",
        "kernel", "independent", "conditional", "carried", "opaque"
    );
    println!("  {}", "-".repeat(70));
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
            "  {:<28}{:>12}{:>12}{:>10}{:>8}",
            f.file_stem().unwrap().to_string_lossy(),
            s.independent,
            s.conditional,
            s.carried_dependency,
            s.opaque_body
        );
        // Which function each loop is in, since a setup fill running
        // once is not worth spreading and a compute loop is, and what
        // each conditional one is waiting on.
        for func in module.functions.values() {
            let (found, _) = parallel_safe::analyze(func);
            for p in &found {
                let name = func.name.resolve_global().unwrap_or_default();
                let asks = if p.is_unconditional() {
                    String::new()
                } else {
                    let disjoint = p
                        .obligations
                        .iter()
                        .filter(|o| matches!(o, parallel_safe::Obligation::Disjoint(_, _)))
                        .count();
                    let counts = p.obligations.len() - disjoint;
                    format!("  needs {disjoint} apart, {counts} equal")
                };
                println!(
                    "        in {}: reads {} writes {}{}",
                    name,
                    p.reads.len(),
                    p.writes.len(),
                    asks
                );
            }
        }
    }
}
