//! Which loops the compiler is willing to call independent.
//!
//! Spreading a loop across cores is safe exactly when no iteration can
//! observe another's writes. The evidence is a band: where every
//! address the body touches sits relative to the counter, and how wide
//! a run of them one iteration owns.
//!
//! There are three answers, not two, and keeping them apart is the
//! point of these. A loop can be independent as written; it can be
//! independent provided something the function cannot see for itself
//! holds, in which case what it needs is named; or it can be refused.
//! Reading the second as the first is how a data race ships, so the
//! assertions below say which one they mean.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::parallel_safe;
use zyntax_embed::ZyntaxRuntime;

/// Every loop in `func` the analysis was willing to offer.
fn loops_of(src: &str, func: &str) -> Vec<parallel_safe::ParallelLoop> {
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
    let mut f = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(func))
        .unwrap_or_else(|| panic!("{func} should be lowered"))
        .clone();
    // SSA construction gives a loop header a phi for every variable
    // written anywhere inside it, so a nested loop leaves the outer
    // header holding the inner one's counter for no reader. That is
    // bookkeeping, not state carried between iterations, and dropping
    // it is the first thing any pipeline does.
    zyntax_compiler::phi_prune::run_function(&mut f);
    parallel_safe::analyze(&f).0
}

/// Loops independent as written, needing nothing of the caller.
fn independent_loops(src: &str, func: &str) -> usize {
    loops_of(src, func)
        .iter()
        .filter(|l| l.is_unconditional())
        .count()
}

/// Loops independent only if what they name holds, with what each one
/// asks for.
fn conditional_loops(src: &str, func: &str) -> Vec<Vec<parallel_safe::Obligation>> {
    loops_of(src, func)
        .into_iter()
        .filter(|l| !l.is_unconditional())
        .map(|l| l.obligations)
        .collect()
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

/// Reading a neighbour. Through one buffer, iteration `i` reads what
/// iteration `i+1` writes, and the answer depends on the order they run
/// in. Through two, nothing conflicts.
///
/// Which of those it is, the body cannot say: `o` and `a` are separate
/// parameters, and nothing stops a caller passing one buffer twice. So
/// the loop is not independent as written, and what it needs is that
/// the two are apart.
#[test]
fn a_neighbour_read_needs_the_buffers_held_apart() {
    let src = format!(
        "{PRE}
def shift(o: Ptr<f32>, a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ o[i] = a[i + 1]  i = i + 1 }}
    return n
}}"
    );
    assert_eq!(
        independent_loops(&src, "shift"),
        0,
        "reading one past the counter is not independent on its own"
    );
    let conditional = conditional_loops(&src, "shift");
    assert_eq!(
        conditional.len(),
        1,
        "the loop should still be offered, conditionally"
    );
    assert!(
        conditional[0]
            .iter()
            .any(|o| matches!(o, parallel_safe::Obligation::Disjoint(_, _))),
        "what it needs is the two buffers apart, but it asked for {:?}",
        conditional[0]
    );
}

/// The same shape through a single buffer cannot be rescued by anything
/// the caller might promise: `a[i]` and `a[i + 1]` are the same storage
/// by construction, one band apart.
#[test]
fn a_neighbour_read_through_one_buffer_is_refused() {
    let src = format!(
        "{PRE}
def slide(a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ a[i] = a[i + 1]  i = i + 1 }}
    return n
}}"
    );
    assert_eq!(independent_loops(&src, "slide"), 0);
    assert!(
        conditional_loops(&src, "slide").is_empty(),
        "no promise makes one buffer two, so this must be refused outright"
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

/// A row per iteration. The write is `i * cols + j` with `j` walking
/// `0..cols`, so iteration `i` owns exactly the run of `cols` elements
/// starting at `i * cols` and no two iterations meet. Both the width of
/// the band and the counter's limit are the same value, so there is
/// nothing left to ask of the caller.
#[test]
fn a_row_per_iteration_is_independent() {
    let src = format!(
        "{PRE}
def rowfill(o: Ptr<f32>, rows: i64, cols: i64): i64 {{
    let mut i: i64 = 0
    while i < rows {{
        let mut j: i64 = 0
        while j < cols {{
            o[i * cols + j] = 1.0
            j = j + 1
        }}
        i = i + 1
    }}
    return 0
}}"
    );
    assert_eq!(
        independent_loops(&src, "rowfill"),
        2,
        "the row loop and the column loop should both be independent, \
         and neither should need anything of the caller: got {:?}",
        conditional_loops(&src, "rowfill")
    );
}

/// The same shape with the band's width and the counter's limit named
/// separately. Nothing in the body says the two agree, and if the
/// counter runs further than the band is wide the rows overlap, so the
/// loop is offered only against that fact.
#[test]
fn a_row_whose_width_is_named_twice_asks_for_them_to_agree() {
    let src = format!(
        "{PRE}
def rowfill2(o: Ptr<f32>, rows: i64, stride: i64, count: i64): i64 {{
    let mut i: i64 = 0
    while i < rows {{
        let mut j: i64 = 0
        while j < count {{
            o[i * stride + j] = 1.0
            j = j + 1
        }}
        i = i + 1
    }}
    return 0
}}"
    );
    let conditional = conditional_loops(&src, "rowfill2");
    assert!(
        conditional.iter().any(|obs| obs
            .iter()
            .any(|o| matches!(o, parallel_safe::Obligation::SameCount(_, _)))),
        "the row loop should ask that the stride and the count agree, \
         but the conditional loops asked for {conditional:?}"
    );
}

/// A strided write with nothing walking inside the stride. Iteration
/// `i` touches `i * stride` alone, which is one address per iteration
/// only while `stride` is at least one; a stride of zero sends every
/// iteration to the same place. Nothing here rules that out, so it is
/// refused rather than offered against a fact no one stated.
#[test]
fn a_bare_strided_write_is_refused() {
    let src = format!(
        "{PRE}
def poke(o: Ptr<f32>, n: i64, stride: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{
        o[i * stride] = 1.0
        i = i + 1
    }}
    return 0
}}"
    );
    assert_eq!(independent_loops(&src, "poke"), 0);
    assert!(
        conditional_loops(&src, "poke").is_empty(),
        "a stride that may be zero is not a band"
    );
}

/// Reading down a column while writing along a row. The read walks the
/// other matrix in a pattern the counter does not describe, which is
/// fine on its own, but only while the two are not one buffer.
#[test]
fn a_column_read_beside_a_row_write_needs_the_buffers_apart() {
    let src = format!(
        "{PRE}
def gather(o: Ptr<f32>, a: Ptr<f32>, rows: i64, cols: i64): i64 {{
    let mut i: i64 = 0
    while i < rows {{
        let mut j: i64 = 0
        while j < cols {{
            o[i * cols + j] = a[j * cols + i]
            j = j + 1
        }}
        i = i + 1
    }}
    return 0
}}"
    );
    assert_eq!(
        independent_loops(&src, "gather"),
        0,
        "nothing here says the two matrices are different storage"
    );
    let conditional = conditional_loops(&src, "gather");
    assert!(
        conditional.iter().any(|obs| obs
            .iter()
            .any(|o| matches!(o, parallel_safe::Obligation::Disjoint(_, _)))),
        "it should ask for the buffers apart, but asked for {conditional:?}"
    );
}

/// A vector store covers four elements from where it starts. When the
/// counter moves four at a time, iteration `i` finishes exactly where
/// `i + 1` begins and the two never meet.
#[test]
fn a_vector_store_stepping_by_its_width_is_independent() {
    let src = format!(
        "{PRE}
def vcopy(o: Ptr<f32>, a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i + 4 <= n {{
        vstore_f32x4(o + i, vload_f32x4(a + i))
        i = i + 4
    }}
    return n
}}"
    );
    assert_eq!(
        independent_loops(&src, "vcopy"),
        1,
        "four lanes stepping four at a time do not overlap: {:?}",
        conditional_loops(&src, "vcopy")
    );
}

/// The same store with the counter moving one at a time. Iteration `i`
/// writes four elements from `i` and iteration `i + 1` writes four from
/// `i + 1`, so three of them are written twice. The band each iteration
/// starts in is still its own; what is not is where the access ends.
#[test]
fn a_vector_store_stepping_by_less_than_its_width_is_refused() {
    let src = format!(
        "{PRE}
def vsmear(o: Ptr<f32>, a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i + 4 <= n {{
        vstore_f32x4(o + i, vload_f32x4(a + i))
        i = i + 1
    }}
    return n
}}"
    );
    assert_eq!(
        independent_loops(&src, "vsmear"),
        0,
        "overlapping vector writes are not independent"
    );
    assert!(
        conditional_loops(&src, "vsmear").is_empty(),
        "one buffer overlapping itself is not something a caller can promise away"
    );
}
