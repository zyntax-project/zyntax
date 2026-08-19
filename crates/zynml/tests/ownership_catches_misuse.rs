//! What the ownership chain actually catches.
//!
//! Three pieces have to agree for any of this to fire: a parameter says
//! it consumes (`own`), a pass turns that into a `Move` at the call, and
//! the borrow check reads the move. Before they did, `run_borrow_check`
//! reported nothing at all on a program that released a buffer twice.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::{analysis, borrow_check, move_insert, BorrowError};
use zyntax_embed::ZyntaxRuntime;

/// Borrow-check one program, with the move insertion the check depends on.
fn errors_in(src: &str) -> Vec<BorrowError> {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<own>").expect("parse");
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
    let mut module = rt.lower_typed_program(program, builtins).expect("lower");

    move_insert::run_module(&mut module);
    let mut runner = analysis::AnalysisRunner::new(module.clone());
    let a = runner.run_all().expect("analysis");
    borrow_check::run_borrow_check(&module, Some(&a))
        .expect("borrow check")
        .errors
}

fn use_after_move_count(errors: &[BorrowError]) -> usize {
    errors
        .iter()
        .filter(|e| matches!(e, BorrowError::UseAfterMove { .. }))
        .count()
}

/// Releasing the same buffer twice. The second release is a use of
/// something already moved.
#[test]
fn a_double_release_is_caught() {
    let errors = errors_in(
        r#"
import prelude
import simd
def main(): i64 {
    let x: Ptr<f32> = alloc_f32(8)
    free(x)
    free(x)
    return 0
}
"#,
    );
    assert!(
        use_after_move_count(&errors) >= 1,
        "releasing twice should be reported, got {errors:?}"
    );
}

/// Reading through a buffer after releasing it. This is why the check
/// has to treat computing an address as a use of the pointer.
#[test]
fn a_use_after_release_is_caught() {
    let errors = errors_in(
        r#"
import prelude
import simd
def main(): i64 {
    let x: Ptr<f32> = alloc_f32(8)
    x[0] = 1.0
    free(x)
    x[1] = 2.0
    return 0
}
"#,
    );
    assert!(
        use_after_move_count(&errors) >= 1,
        "an access after release should be reported, got {errors:?}"
    );
}

/// The case that must stay quiet: allocate, use, release once. A guard
/// that fires here would make the language unusable.
#[test]
fn a_correct_program_is_left_alone() {
    let errors = errors_in(
        r#"
import prelude
import simd
def scale(p: Ptr<f32>, k: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { p[i] = p[i] * k  i = i + 1 }
    return n
}
def main(): i64 {
    let x: Ptr<f32> = alloc_f32(8)
    let mut i: i64 = 0
    while i < 8 { x[i] = 1.0  i = i + 1 }
    let r: i64 = scale(x, 2.0, 8)
    free(x)
    return r
}
"#,
    );
    assert_eq!(
        use_after_move_count(&errors),
        0,
        "a correct program must not be flagged, got {errors:?}"
    );
}

/// Borrowing is not moving: passing a buffer to a function that only
/// reads it leaves the caller free to release it afterwards.
#[test]
fn passing_to_a_borrowing_function_is_not_a_move() {
    let errors = errors_in(
        r#"
import prelude
import simd
def total(p: Ptr<f32>, n: i64): f32 {
    let mut s: f32 = 0.0
    let mut i: i64 = 0
    while i < n { s = s + p[i]  i = i + 1 }
    return s
}
def main(): i64 {
    let x: Ptr<f32> = alloc_f32(8)
    let a: f32 = total(x, 8)
    let b: f32 = total(x, 8)
    free(x)
    return 0
}
"#,
    );
    assert_eq!(
        use_after_move_count(&errors),
        0,
        "two reads then a release is correct, got {errors:?}"
    );
}
