//! Scalar loops that become vector ones without being asked.
//!
//! A kernel language should not require the intrinsics to be written out
//! by hand for the ordinary elementwise shape. `out[i] = a[i] + b[i]`
//! carries all the information needed to do it four lanes at a time, and
//! the vectorizer recognises exactly that.
//!
//! What these check is both halves: that the vector form is reached, and
//! that it still computes the same answer, tail included.

use std::path::Path;
use std::sync::Mutex;
use zynml::{ZynML, ZynMLConfig};

/// The dump directory is named by an environment variable, which the
/// whole process shares, so these run one at a time.
static DUMPING: Mutex<()> = Mutex::new(());

fn build(src: &str, dump: &str) -> (f64, usize) {
    let _serialised = DUMPING.lock().unwrap_or_else(|e| e.into_inner());
    // Built by joining rather than formatting: `canonicalize` returns a
    // verbatim `\\?\C:\...` path on Windows, which takes no forward
    // slash as a separator, so pasting one in makes the name invalid.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
        .join("target")
        .join(format!("hirdump_{dump}"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    // Process-global, and these tests share a process, so each one
    // points it at its own directory immediately before compiling.
    std::env::set_var("ZYNTAX_DUMP_HIR_DIR", &dir);

    let plugins = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugins/target/zrtl");
    let cfg = ZynMLConfig {
        plugins_dir: plugins.to_string_lossy().to_string(),
        ..ZynMLConfig::default()
    };
    let mut z = ZynML::with_config(cfg).expect("runtime");
    z.load_source(src).expect("should compile");
    let value = z.call_with_result::<f64>("main").expect("should run");

    // The dump names carry a counter that advances per compile in a
    // process, so the file is whatever landed in this test's directory
    // rather than a fixed name.
    let hir: String = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "hir"))
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    assert!(
        !hir.is_empty(),
        "no HIR dump at {}; the vector-op count below would be meaningless",
        dir.display()
    );
    let vector_ops = hir.matches("vload").count() + hir.matches("vstore").count();
    (value, vector_ops)
}

/// Buffers handed in, which a kernel called from outside has.
///
/// 8 * (1.5 + 2.5) = 32 over the whole buffer, and the loop reaches the
/// vector form rather than stepping an element at a time.
#[test]
fn a_loop_over_buffer_parameters_vectorizes() {
    let (value, vector_ops) = build(
        r#"
import prelude
import simd
def vadd(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { out[i] = a[i] + b[i] i = i + 1 }
    return n
}
def fill(p: Ptr<f32>, v: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { p[i] = v i = i + 1 }
    return n
}
def total(p: Ptr<f32>, n: i64): f32 {
    let mut s: f32 = 0.0
    let mut i: i64 = 0
    while i < n { s = s + p[i] i = i + 1 }
    return s
}
def main(): f64 {
    let a: Ptr<f32> = alloc_f32(8)
    let b: Ptr<f32> = alloc_f32(8)
    let c: Ptr<f32> = alloc_f32(8)
    let f1: i64 = fill(a, 1.5, 8)
    let f2: i64 = fill(b, 2.5, 8)
    let r: i64 = vadd(c, a, b, 8)
    return (total(c, 8) as f64)
}
"#,
        "params",
    );
    assert_eq!(value, 32.0);
    assert!(vector_ops > 0, "the loop should reach the vector form");
}

/// Buffers the function allocated itself.
///
/// The same loop, and it used to stay scalar: a local live across the
/// loop carries a phi that reads as a definition inside it, so the buffer
/// did not look loop-invariant. 8 * (1.5 + 2.5) = 32 again.
#[test]
fn a_loop_over_local_buffers_vectorizes() {
    let (value, vector_ops) = build(
        r#"
import prelude
import simd
def work(): f32 {
    let a: Ptr<f32> = alloc_f32(8)
    let b: Ptr<f32> = alloc_f32(8)
    let c: Ptr<f32> = alloc_f32(8)
    let mut i: i64 = 0
    while i < 8 { a[i] = 1.5 i = i + 1 }
    let mut j: i64 = 0
    while j < 8 { b[j] = 2.5 j = j + 1 }
    let mut k: i64 = 0
    while k < 8 { c[k] = a[k] + b[k] k = k + 1 }
    let mut s: f32 = 0.0
    let mut m: i64 = 0
    while m < 8 { s = s + c[m] m = m + 1 }
    return s
}
def main(): f64 { return (work() as f64) }
"#,
        "locals",
    );
    assert_eq!(value, 32.0);
    assert!(vector_ops > 0, "the loop should reach the vector form");
}

/// A length that is not a multiple of the lane count, so the scalar
/// epilogue the vectorizer leaves behind has to carry the remainder:
/// 7 * (1.0 + 2.0) = 21.
#[test]
fn the_tail_the_vectorizer_leaves_is_correct() {
    let (value, _) = build(
        r#"
import prelude
import simd
def vadd(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { out[i] = a[i] + b[i] i = i + 1 }
    return n
}
def fill(p: Ptr<f32>, v: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { p[i] = v i = i + 1 }
    return n
}
def total(p: Ptr<f32>, n: i64): f32 {
    let mut s: f32 = 0.0
    let mut i: i64 = 0
    while i < n { s = s + p[i] i = i + 1 }
    return s
}
def main(): f64 {
    let a: Ptr<f32> = alloc_f32(7)
    let b: Ptr<f32> = alloc_f32(7)
    let c: Ptr<f32> = alloc_f32(7)
    let f1: i64 = fill(a, 1.0, 7)
    let f2: i64 = fill(b, 2.0, 7)
    let r: i64 = vadd(c, a, b, 7)
    return (total(c, 7) as f64)
}
"#,
        "tail",
    );
    assert_eq!(value, 21.0);
}
