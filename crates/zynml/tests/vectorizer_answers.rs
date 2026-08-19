//! A widened loop computes what the scalar one computed.
//!
//! A vectorizer that gets the shape wrong does not usually crash. It
//! writes one lane's answer into four, or leaves a vector where a
//! scalar was asked for, and the program carries on. So these check the
//! value, which is the only thing that distinguishes a loop that was
//! widened correctly from one that was widened at all.
//!
//! Both shapes below reached the wrong answer through a whole language
//! stack that had no other complaint about them.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

const PRE: &str = "import prelude\nimport simd\n";

/// Compile and run `main`, on a thread so a kernel that fails to
/// terminate fails the test rather than the suite.
fn answer(body: &str) -> ZyntaxValue {
    let src = format!("{PRE}{body}");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(&src, "<vec>").expect("parse");
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
    let module = rt.lower_typed_program(program, builtins).expect("lower");

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let got = (|| {
            let mut rt = ZyntaxRuntime::new().map_err(|e| format!("{e:?}"))?;
            rt.compile_module(&module).map_err(|e| format!("{e:?}"))?;
            rt.call_function_raw("main", vec![])
                .map_err(|e| format!("{e:?}"))
        })();
        let _ = tx.send(got);
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("the kernel should finish")
        .expect("the kernel should run")
}

/// A buffer written from the counter holds what the counter was.
///
/// Widening a loop retypes an instruction's result and leaves its
/// operands alone. For a store of `a[i] = f(i)` that leaves the counter
/// a scalar feeding a vector: four elements written from one value, or
/// a type the backend refuses outright. Making it right needs the
/// counter itself as a vector holding `i, i+1, i+2, i+3`, which nothing
/// builds, so the loop has to stay scalar.
#[test]
fn a_store_computed_from_the_counter_is_not_widened() {
    let got = answer(
        r#"
def main(): i64 {
    let p: Ptr<f32> = alloc_f32(64)
    let mut i: i64 = 0
    while i < 64 { p[i] = (i as f32)  i = i + 1 }
    let mut wrong: i64 = 0
    let mut j: i64 = 0
    while j < 64 {
        if p[j] != (j as f32) { wrong = wrong + 1 }
        j = j + 1
    }
    free(p)
    return wrong
}"#,
    );
    assert_eq!(
        got,
        ZyntaxValue::Int(0),
        "every element should hold its own index"
    );
}

/// The same through arithmetic rather than a cast, since the rule is
/// about reading the counter and not about one instruction.
#[test]
fn arithmetic_on_the_counter_is_not_widened_either() {
    let got = answer(
        r#"
def main(): i64 {
    let p: Ptr<f32> = alloc_f32(64)
    let mut i: i64 = 0
    while i < 64 { p[i] = ((i * 3 + 1) as f32)  i = i + 1 }
    let mut total: f32 = 0.0
    let mut j: i64 = 0
    while j < 64 { total = total + p[j]  j = j + 1 }
    free(p)
    return (total as i64)
}"#,
    );
    // 3 * (0 + .. + 63) + 64 = 3 * 2016 + 64
    assert_eq!(got, ZyntaxValue::Int(6112));
}

/// A sum over a buffer is the sum, not one lane's share of it.
///
/// Widening the accumulator makes the header phi a vector of partial
/// sums. The horizontal reduce that adds the lanes feeds the scalar
/// tail, so the tail's phi is the only value that is both whole and
/// final; anything reading the accumulator after the loop has to read
/// that one.
#[test]
fn a_sum_after_a_widened_loop_is_the_whole_sum() {
    let got = answer(
        r#"
def main(): i64 {
    let p: Ptr<f32> = alloc_f32(64)
    let mut i: i64 = 0
    while i < 64 { p[i] = 1.0  i = i + 1 }
    let mut total: f32 = 0.0
    let mut j: i64 = 0
    while j < 64 { total = total + p[j]  j = j + 1 }
    free(p)
    return (total as i64)
}"#,
    );
    assert_eq!(
        got,
        ZyntaxValue::Int(64),
        "64 ones sum to 64, not to one lane's 16"
    );
}

/// A trip count that is not a multiple of the lane width leaves work
/// for the tail, so the tail's contribution has to reach the answer
/// too.
#[test]
fn the_tail_of_a_sum_reaches_the_answer() {
    let got = answer(
        r#"
def main(): i64 {
    let n: i64 = 70
    let p: Ptr<f32> = alloc_f32(n)
    let mut i: i64 = 0
    while i < n { p[i] = 1.0  i = i + 1 }
    let mut total: f32 = 0.0
    let mut j: i64 = 0
    while j < n { total = total + p[j]  j = j + 1 }
    free(p)
    return (total as i64)
}"#,
    );
    assert_eq!(
        got,
        ZyntaxValue::Int(70),
        "68 from the vector body, 2 more from the tail"
    );
}

/// A running maximum keeps the largest element, which is the shape the
/// attention softmax takes before it exponentiates.
#[test]
fn a_running_maximum_survives_the_loop() {
    let got = answer(
        r#"
def main(): i64 {
    let p: Ptr<f32> = alloc_f32(64)
    let mut i: i64 = 0
    while i < 64 { p[i] = ((i % 5) as f32)  i = i + 1 }
    let mut top: f32 = -1000000.0
    let mut j: i64 = 0
    while j < 64 {
        let v: f32 = p[j]
        if v > top { top = v }
        j = j + 1
    }
    free(p)
    return (top as i64)
}"#,
    );
    assert_eq!(got, ZyntaxValue::Int(4));
}
