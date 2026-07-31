//! Loop-carried locals, `let` initialisers, and array indexing.
//!
//! Reproductions for the gaps reported against the loop substrate, written
//! directly in ZynML so they can be fixed and verified here rather than
//! through a downstream DSL.
//!
//! Each test asserts the *intended* behaviour, so a failure is the gap and
//! a pass means it is closed.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

/// Compile `src` and call `main`, returning the integer it produced.
fn run_main(src: &str) -> i64 {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    let program = grammar
        .parse_with_filename(src, "<loop_substrate>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
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
    let module = rt
        .lower_typed_program(program, builtins)
        .expect("program should lower");
    rt.compile_module(&module).expect("program should compile");
    match rt
        .call_function_raw("main", vec![])
        .expect("main should execute")
    {
        ZyntaxValue::Int(v) => v,
        ZyntaxValue::I32(v) => v as i64,
        other => panic!("expected an integer result, got {other:?}"),
    }
}

/// Gap 1, `mut` spelling. A counter written in the loop body must be
/// visible to the next iteration and after the loop.
#[test]
fn a_mutable_local_carries_across_iterations() {
    let got = run_main(
        r#"
        def main(): i64 {
            let mut i: i64 = 0
            while i < 5 {
                i = i + 1
            }
            return i
        }
        "#,
    );
    assert_eq!(got, 5, "counter should carry across iterations");
}

/// Gap 1, non-`mut` spelling — the form the report used. Kept separate so
/// the two spellings can be told apart: if this fails while the `mut` one
/// passes, the gap is about the binding form, not about loops.
#[test]
fn a_plain_local_carries_across_iterations() {
    let got = run_main(
        r#"
        def main(): i64 {
            let i: i64 = 0
            while i < 5 {
                i = i + 1
            }
            return i
        }
        "#,
    );
    assert_eq!(got, 5, "counter should carry across iterations");
}

/// Gap 2. A local read straight from its initialiser must see the
/// initialiser's value, with no intervening store.
#[test]
fn a_local_reads_correctly_from_its_initialiser() {
    let got = run_main(
        r#"
        def main(): i64 {
            let i: i64 = 3
            if i == 3 {
                return 1
            }
            return 0
        }
        "#,
    );
    assert_eq!(
        got, 1,
        "the initialiser should be visible to the first read"
    );
}

/// The control from the report: the same read after an explicit store is
/// expected to work today, so a failure here would mean something broader
/// than gap 2.
#[test]
fn a_local_reads_correctly_after_a_store() {
    let got = run_main(
        r#"
        def main(): i64 {
            let mut i: i64 = 0
            i = i + 3
            if i == 3 {
                return 1
            }
            return 0
        }
        "#,
    );
    assert_eq!(got, 1);
}

/// The report's examples are untyped. If inference is what breaks the
/// loop-carried value, these fail while their annotated twins above pass.
#[test]
fn an_untyped_local_carries_across_iterations() {
    let got = run_main(
        r#"
        def main(): i64 {
            let mut i = 0
            while i < 5 {
                i = i + 1
            }
            return i
        }
        "#,
    );
    assert_eq!(got, 5, "an inferred counter should carry across iterations");
}

#[test]
fn an_untyped_local_reads_correctly_from_its_initialiser() {
    let got = run_main(
        r#"
        def main(): i64 {
            let i = 3
            if i == 3 {
                return 1
            }
            return 0
        }
        "#,
    );
    assert_eq!(
        got, 1,
        "an inferred initialiser should be visible to the first read"
    );
}
