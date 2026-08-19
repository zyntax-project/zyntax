//! Which loops the compiler is willing to call independent.
//!
//! Spreading a loop across cores is safe exactly when no iteration can
//! observe another's writes. The evidence is the same the vectorizer
//! uses: every address comes from the induction variable, over a base
//! that does not change. What matters most here is what is REFUSED, so
//! these lean on the negative cases.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::parallel_safe;
use zyntax_embed::ZyntaxRuntime;

/// How many loops in `func` were found independent.
fn independent_loops(src: &str, func: &str) -> usize {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<par>").expect("parse");
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
    let f = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(func))
        .unwrap_or_else(|| panic!("{func} should be lowered"));
    parallel_safe::analyze(f).0.len()
}

const PRE: &str = "import prelude\nimport simd\n";

/// Elementwise over three buffers. Iteration `i` reads and writes only
/// at `i`, so the order iterations run in cannot matter.
#[test]
fn an_elementwise_loop_is_independent() {
    let n = independent_loops(
        &format!(
            "{PRE}
def vadd(o: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ o[i] = a[i] + b[i]  i = i + 1 }}
    return n
}}"
        ),
        "vadd",
    );
    assert_eq!(n, 1, "an elementwise loop should be independent");
}

/// A running sum. The accumulator is carried around the back edge, so
/// two cores would read and write the same value. Splitting it needs
/// per-core partials, which is a different shape, so it must be refused
/// rather than quietly split.
#[test]
fn a_reduction_is_refused() {
    let n = independent_loops(
        &format!(
            "{PRE}
def total(p: Ptr<f32>, n: i64): f32 {{
    let mut s: f32 = 0.0
    let mut i: i64 = 0
    while i < n {{ s = s + p[i]  i = i + 1 }}
    return s
}}"
        ),
        "total",
    );
    assert_eq!(
        n, 0,
        "an accumulator carried between iterations is not independent"
    );
}

/// Reading a neighbour. Iteration `i` reads what iteration `i+1` writes,
/// so the answer depends on the order they run in.
#[test]
fn a_neighbour_read_is_refused() {
    let n = independent_loops(
        &format!(
            "{PRE}
def shift(o: Ptr<f32>, a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ o[i] = a[i + 1]  i = i + 1 }}
    return n
}}"
        ),
        "shift",
    );
    assert_eq!(
        n, 0,
        "an address that is not the induction variable is not independent"
    );
}

/// A call whose effects are invisible here could touch anything.
#[test]
fn a_call_in_the_body_is_refused() {
    let n = independent_loops(
        &format!(
            "{PRE}
def side(x: f32): f32 {{ return x * 2.0 }}
def apply(o: Ptr<f32>, a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ o[i] = side(a[i])  i = i + 1 }}
    return n
}}"
        ),
        "apply",
    );
    assert_eq!(n, 0, "a call with unknown effects is not independent");
}

/// A loop that only reads has no race, but also nothing to gain, so it
/// is not offered.
#[test]
fn a_read_only_loop_is_not_offered() {
    let n = independent_loops(
        &format!(
            "{PRE}
def touch(a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    let mut k: i64 = 0
    while i < n {{ k = k + 1  i = i + 1 }}
    return k
}}"
        ),
        "touch",
    );
    assert_eq!(n, 0);
}

/// A scaled update writes what it also reads, but only ever at the same
/// index, so iterations still do not interfere.
#[test]
fn a_scaled_update_is_independent() {
    let n = independent_loops(
        &format!(
            "{PRE}
def axpy(y: Ptr<f32>, x: Ptr<f32>, a: f32, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ y[i] = a * x[i] + y[i]  i = i + 1 }}
    return n
}}"
        ),
        "axpy",
    );
    assert_eq!(
        n, 1,
        "reading and writing the same index is still independent"
    );
}
