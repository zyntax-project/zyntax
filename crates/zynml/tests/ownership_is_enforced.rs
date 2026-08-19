//! Ownership is enforced for ZynML, not merely available.
//!
//! These go through the front door: whatever a program does wrong has to
//! be rejected by `load_source`, not by a checker a test remembered to
//! call.

use zynml::ZynML;

fn load(src: &str) -> Result<Vec<String>, String> {
    let mut rt = ZynML::new().map_err(|e| e.to_string())?;
    rt.load_source(src).map_err(|e| format!("{e:?}"))
}

/// Releasing the same buffer twice does not compile.
#[test]
fn a_double_release_does_not_compile() {
    let r = load(
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
    assert!(r.is_err(), "releasing twice should be rejected, got {r:?}");
}

/// Reading through a buffer after releasing it does not compile.
#[test]
fn a_use_after_release_does_not_compile() {
    let r = load(
        r#"
import prelude
import simd
def main(): i64 {
    let x: Ptr<f32> = alloc_f32(8)
    free(x)
    x[0] = 1.0
    return 0
}
"#,
    );
    assert!(
        r.is_err(),
        "an access after release should be rejected, got {r:?}"
    );
}

/// The program that does it correctly still compiles and still runs.
#[test]
fn a_correct_program_still_compiles_and_runs() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
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
    let mut i: i64 = 0
    while i < 8 { x[i] = 1.5  i = i + 1 }
    let s: f32 = total(x, 8)
    free(x)
    return (s as i64)
}
"#,
    )
    .expect("a correct program must still compile");
    let v: i64 = rt.call_with_result("main").expect("and still run");
    assert_eq!(v, 12, "8 * 1.5");
}
