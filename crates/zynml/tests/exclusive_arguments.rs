//! `mut` means nothing else names it, and the compiler holds callers to
//! that.
//!
//! The claim is what lets a loop writing through one parameter and
//! reading through another be spread across cores without asking the
//! caller anything: two parameters where one is exclusive are two
//! buffers. That reasoning is only worth as much as the enforcement
//! underneath it, so these check the enforcement first and the
//! conclusion second.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::parallel_safe;
use zyntax_embed::ZyntaxRuntime;

const PRE: &str = "import prelude\nimport simd\n";

/// Lower `src`, returning the error text where it was refused.
fn lower(src: &str) -> Result<zyntax_compiler::HirModule, String> {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<excl>").expect("parse");
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
    rt.lower_typed_program(program, builtins)
        .map_err(|e| format!("{e:?}"))
}

const SCALE: &str = r#"
def scale(mut out: Ptr<f32>, a: Ptr<f32>, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { out[i] = a[i] * 2.0  i = i + 1 }
    return n
}
"#;

/// Handing one buffer to an exclusive parameter and to another
/// parameter of the same call breaks the claim where it is made, so the
/// call is refused.
#[test]
fn passing_one_buffer_twice_is_refused() {
    let src = format!(
        "{PRE}{SCALE}
def main(): i64 {{
    let p: Ptr<f32> = alloc_f32(64)
    let r: i64 = scale(p, p, 64)
    free(p)
    return r
}}"
    );
    let err = lower(&src).expect_err("passing `p` twice should be refused");
    assert!(
        err.contains("exclusive"),
        "the message should say what was violated, but was: {err}"
    );
}

/// Reaching the same buffer by a different name is the same violation,
/// so an element address of it is caught too.
#[test]
fn reaching_the_same_buffer_through_an_element_is_refused() {
    let src = format!(
        "{PRE}{SCALE}
def main(): i64 {{
    let p: Ptr<f32> = alloc_f32(64)
    let r: i64 = scale(p, p + 4, 32)
    free(p)
    return r
}}"
    );
    let err = lower(&src).expect_err("an element of `p` is still `p`");
    assert!(err.contains("exclusive"), "was: {err}");
}

/// Two buffers are two buffers.
#[test]
fn two_separate_buffers_are_accepted() {
    let src = format!(
        "{PRE}{SCALE}
def main(): i64 {{
    let p: Ptr<f32> = alloc_f32(64)
    let q: Ptr<f32> = alloc_f32(64)
    let r: i64 = scale(p, q, 64)
    free(p)
    free(q)
    return r
}}"
    );
    lower(&src).expect("distinct buffers should be fine");
}

/// And a parameter that claims nothing constrains nothing, so the same
/// call without `mut` is allowed however the caller aliases it.
#[test]
fn a_shared_parameter_places_no_demand_on_the_caller() {
    let src = format!(
        "{PRE}
def touch(out: Ptr<f32>, a: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ out[i] = a[i] * 2.0  i = i + 1 }}
    return n
}}
def main(): i64 {{
    let p: Ptr<f32> = alloc_f32(64)
    let r: i64 = touch(p, p, 64)
    free(p)
    return r
}}"
    );
    lower(&src).expect("nothing was claimed, so nothing is violated");
}

/// Which loops in `func` need nothing established, and which are
/// offered only against something.
fn split(src: &str, func: &str) -> (usize, usize) {
    let module = lower(src).expect("should lower");
    let mut f = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(func))
        .unwrap_or_else(|| panic!("{func} should be lowered"))
        .clone();
    zyntax_compiler::phi_prune::run_function(&mut f);
    let (found, _) = parallel_safe::analyze(&f);
    let free = found.iter().filter(|l| l.is_unconditional()).count();
    (free, found.len() - free)
}

/// The conclusion the enforcement pays for.
///
/// The same kernel is conditionally independent when its destination is
/// an ordinary parameter, because the body cannot rule out the caller
/// having passed one buffer twice, and independent outright when the
/// destination is declared exclusive, because the caller was held to it.
#[test]
fn declaring_the_destination_exclusive_settles_the_obligation() {
    let shared = format!(
        "{PRE}
def gemm(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, m: i64, k: i64, n: i64): i64 {{
    let mut i: i64 = 0
    while i < m {{
        let mut j: i64 = 0
        while j < n {{
            let mut acc: f32 = 0.0
            let mut p: i64 = 0
            while p < k {{ acc = acc + a[i * k + p] * b[p * n + j]  p = p + 1 }}
            out[i * n + j] = acc
            j = j + 1
        }}
        i = i + 1
    }}
    return 0
}}"
    );
    let exclusive = shared.replace("def gemm(out:", "def gemm(mut out:");

    assert_eq!(
        split(&shared, "gemm"),
        (0, 2),
        "with a shared destination both loops should be offered only \
         against the buffers being apart"
    );
    assert_eq!(
        split(&exclusive, "gemm"),
        (2, 0),
        "with an exclusive destination there is nothing left to ask"
    );
}
