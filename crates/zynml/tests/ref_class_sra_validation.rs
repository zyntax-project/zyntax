//! End-to-end validation of strict V1 `@reference` class lowering.
//!
//! Two checks:
//!   1. The program returns 30 (10 + 20) — the heap-allocated `Point`
//!      with malloc + GEP-Store + GEP-Load behaves identically to a
//!      value-typed struct at the language level.
//!   2. `scalar_replace_alloc::run_module` collapses the malloc back
//!      to SSA registers (`mallocs_eliminated >= 1`) — confirming the
//!      emitted shape (single-block malloc + tracked GEPs + Stores +
//!      Loads, no escape) matches what the pass recognises.

use indexmap::IndexMap;
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::scalar_replace_alloc;
use zyntax_embed::ZyntaxRuntime;

const PROGRAM: &str = r#"
@reference
struct Point {
    x: i64,
    y: i64
}

def main(): i64 {
    let p = Point { x: 10, y: 20 }
    return p.x + p.y
}
"#;

fn parse(source: &str) -> zyntax_typed_ast::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar compiles");
    grammar
        .parse_with_filename(source, "<ref_class_sra>")
        .expect("source parses")
}

#[test]
fn ref_class_program_returns_thirty() {
    let program = parse(PROGRAM);
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program).expect("compile");
    let r = rt.call_function_raw("main", vec![]).expect("call main()");
    assert_eq!(
        r.as_i64(),
        Some(30),
        "@reference Point{{x:10, y:20}}.x + .y = {:?}",
        r
    );
}

#[test]
fn ref_class_scalar_replace_eliminates_malloc() {
    // Lower a FRESH program to HIR without running the optimisation
    // pipeline, then call `scalar_replace_alloc::run_module` directly
    // so we observe the pass's effect on the raw lowering — not on a
    // module that the runtime's opt sweep has already processed.
    let program = parse(PROGRAM);
    let rt = ZyntaxRuntime::new().expect("runtime");
    let mut module = rt
        .lower_typed_program(program, IndexMap::new())
        .expect("lower");

    let stats = scalar_replace_alloc::run_module(&mut module);
    assert!(
        stats.mallocs_eliminated >= 1,
        "scalar_replace_alloc should eliminate the non-escaping Point malloc; \
         got stats={:?}",
        stats
    );
}
