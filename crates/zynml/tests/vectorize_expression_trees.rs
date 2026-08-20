//! Loops whose body is more than one operation over two arrays.
//!
//! The loop matcher used to accept exactly `c[i] = a[i] op b[i]`: one
//! binary over two element reads. That rejected most real kernels. A
//! scaled update carries a loop-invariant scalar, which is not a read
//! at all, and a dot product carries a multiply feeding an add.
//!
//! It now matches an expression tree whose leaves are element reads and
//! loop-invariant values, the latter broadcast once ahead of the loop.
//!
//! FMA contraction had to move after the vectorizers for any of it to
//! land. A multiply feeding an add is the shape both want, and whichever
//! ran first took it: fusing to a call first left the matcher something
//! it could not widen. Fusing afterwards costs nothing, because the pass
//! contracts vector-typed operands just as readily.

use std::path::Path;
use std::sync::Mutex;
use zynml::{ZynML, ZynMLConfig};

/// The dump directory is named by a process-global variable.
static DUMPING: Mutex<()> = Mutex::new(());

/// Run `main`, and report how many vector memory ops landed inside one
/// named function.
///
/// Scoping to the function matters: these programs carry helper loops
/// that vectorize on their own, so a module-wide count would report
/// success even when the kernel under test stayed scalar.
fn build(src: &str, dump: &str, func: &str) -> (f64, usize, String) {
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
    std::env::set_var("ZYNTAX_DUMP_HIR_DIR", &dir);

    let plugins = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugins/target/zrtl");
    let cfg = ZynMLConfig {
        plugins_dir: plugins.to_string_lossy().to_string(),
        ..ZynMLConfig::default()
    };
    let mut z = ZynML::with_config(cfg).expect("runtime");
    z.load_source(src).expect("should compile");
    let value = z.call_with_result::<f64>("main").expect("should run");

    let hir: String = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "hir"))
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect();
    assert!(
        !hir.is_empty(),
        "no HIR dump at {}; a count here would be meaningless",
        dir.display()
    );

    let marker = format!("function @{func}");
    let body = hir
        .split(&marker)
        .nth(1)
        .unwrap_or_else(|| panic!("{marker} absent from the dump"))
        .split("\n}")
        .next()
        .unwrap()
        .to_string();
    let n = body.matches("vload").count() + body.matches("vstore").count();
    (value, n, body)
}

const HELPERS: &str = r#"
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
"#;

/// `y[i] = alpha * x[i] + y[i]`, the BLAS-1 kernel. `alpha` is the same
/// on every iteration and in every lane, so it is broadcast once ahead
/// of the loop rather than per iteration.
#[test]
fn a_scaled_update_vectorizes() {
    let src = format!(
        r#"
import prelude
import simd
def axpy(y: Ptr<f32>, x: Ptr<f32>, alpha: f32, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ y[i] = alpha * x[i] + y[i] i = i + 1 }}
    return n
}}
{HELPERS}
def main(): f64 {{
    let x: Ptr<f32> = alloc_f32(8)
    let y: Ptr<f32> = alloc_f32(8)
    let f1: i64 = fill(x, 1.5, 8)
    let f2: i64 = fill(y, 0.5, 8)
    let r: i64 = axpy(y, x, 2.0, 8)
    return (total(y, 8) as f64)
}}
"#
    );
    let (value, vector_ops, body) = build(&src, "axpy", "axpy");
    assert_eq!(value, 28.0, "8 * (2.0 * 1.5 + 0.5)");
    assert!(
        vector_ops >= 3,
        "two reads and a write should widen, saw {vector_ops}"
    );
    assert!(
        body.contains("vector_splat"),
        "alpha should be broadcast, not reloaded per lane"
    );
}

/// The broadcast belongs outside the loop. Emitting it per iteration
/// would be work the loop cannot remove.
#[test]
fn the_broadcast_is_hoisted_out_of_the_loop() {
    let src = format!(
        r#"
import prelude
import simd
def axpy(y: Ptr<f32>, x: Ptr<f32>, alpha: f32, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ y[i] = alpha * x[i] + y[i] i = i + 1 }}
    return n
}}
{HELPERS}
def main(): f64 {{
    let x: Ptr<f32> = alloc_f32(8)
    let y: Ptr<f32> = alloc_f32(8)
    let f1: i64 = fill(x, 1.0, 8)
    let f2: i64 = fill(y, 0.0, 8)
    let r: i64 = axpy(y, x, 3.0, 8)
    return (total(y, 8) as f64)
}}
"#
    );
    let (value, _, body) = build(&src, "axpy_hoist", "axpy");
    assert_eq!(value, 24.0);
    // The splat must sit before the block that carries the vector loads.
    let splat = body.find("vector_splat").expect("a broadcast");
    let vload = body.find("vload").expect("a vector load");
    assert!(
        splat < vload,
        "the broadcast should precede the loop body, not sit inside it"
    );
}

/// A length that is not a whole number of vectors, so the scalar tail
/// the vectorizer leaves has to carry the remainder of a scaled update.
#[test]
fn the_tail_of_a_scaled_update_is_correct() {
    let src = format!(
        r#"
import prelude
import simd
def axpy(y: Ptr<f32>, x: Ptr<f32>, alpha: f32, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ y[i] = alpha * x[i] + y[i] i = i + 1 }}
    return n
}}
{HELPERS}
def main(): f64 {{
    let x: Ptr<f32> = alloc_f32(7)
    let y: Ptr<f32> = alloc_f32(7)
    let f1: i64 = fill(x, 2.0, 7)
    let f2: i64 = fill(y, 1.0, 7)
    let r: i64 = axpy(y, x, 3.0, 7)
    return (total(y, 7) as f64)
}}
"#
    );
    let (value, _, _) = build(&src, "axpy_tail", "axpy");
    // 7 * (3.0 * 2.0 + 1.0) = 49
    assert_eq!(value, 49.0, "the scalar epilogue must carry the remainder");
}

/// `acc = acc + a[i] * b[i]`. This is the reduction matcher's shape, and
/// FMA contraction used to consume it before that matcher ran, which is
/// why moving the contraction is what made this one widen.
#[test]
fn a_dot_product_reduction_vectorizes() {
    let src = format!(
        r#"
import prelude
import simd
def dot(a: Ptr<f32>, b: Ptr<f32>, n: i64): f32 {{
    let mut acc: f32 = 0.0
    let mut i: i64 = 0
    while i < n {{ acc = acc + a[i] * b[i] i = i + 1 }}
    return acc
}}
{HELPERS}
def main(): f64 {{
    let a: Ptr<f32> = alloc_f32(8)
    let b: Ptr<f32> = alloc_f32(8)
    let f1: i64 = fill(a, 1.5, 8)
    let f2: i64 = fill(b, 2.0, 8)
    return (dot(a, b, 8) as f64)
}}
"#
    );
    let (value, vector_ops, _) = build(&src, "dot", "dot");
    assert_eq!(value, 24.0, "8 * 1.5 * 2.0");
    assert!(
        vector_ops > 0,
        "the reduction should widen, saw {vector_ops}"
    );
}

/// A deeper tree than either of the above, with no invariant leaf, to
/// check the matcher recurses rather than handling one special case.
#[test]
fn a_three_read_expression_vectorizes() {
    let src = format!(
        r#"
import prelude
import simd
def blend(o: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {{
    let mut i: i64 = 0
    while i < n {{ o[i] = a[i] * b[i] + a[i] i = i + 1 }}
    return n
}}
{HELPERS}
def main(): f64 {{
    let a: Ptr<f32> = alloc_f32(8)
    let b: Ptr<f32> = alloc_f32(8)
    let o: Ptr<f32> = alloc_f32(8)
    let f1: i64 = fill(a, 2.0, 8)
    let f2: i64 = fill(b, 3.0, 8)
    let r: i64 = blend(o, a, b, 8)
    return (total(o, 8) as f64)
}}
"#
    );
    let (value, vector_ops, _) = build(&src, "blend", "blend");
    // 8 * (2.0 * 3.0 + 2.0) = 64
    assert_eq!(value, 64.0);
    assert!(
        vector_ops > 0,
        "a two-op tree should widen, saw {vector_ops}"
    );
}
