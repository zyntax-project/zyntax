//! Lane count follows the element and the target, not a literal.
//!
//! The vectorizers wrote 4 lanes for every element type. That is right
//! for `f32` in a 128-bit register and wrong for everything else: `f64`
//! fits two, and a wider register fits proportionally more.
//!
//! The default stays at the width every backend accepts, because one
//! module is compiled by several and the Cranelift backend holds
//! 128-bit vectors only. `ZYNTAX_VECTOR_BITS` raises it for measuring
//! the LLVM tier.

use std::path::Path;
use std::sync::Mutex;
use zynml::{ZynML, ZynMLConfig};

static DUMPING: Mutex<()> = Mutex::new(());

/// Run `main`, and report the vector types appearing inside one function.
fn build(src: &str, dump: &str, func: &str) -> (f64, String) {
    let _serialised = DUMPING.lock().unwrap_or_else(|e| e.into_inner());
    let dir = format!(
        "{}/target/hirdump_{dump}",
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap()
            .display()
    );
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
        "no HIR dump; an assertion here would be meaningless"
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
    (value, body)
}

/// `f32` fills a 128-bit register four at a time.
#[test]
fn a_float_kernel_uses_four_lanes() {
    let (v, body) = build(
        r#"
import prelude
import simd
def vadd(o: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { o[i] = a[i] + b[i]  i = i + 1 }
    return n
}
def fill(p: Ptr<f32>, v: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { p[i] = v  i = i + 1 }
    return n
}
def total(p: Ptr<f32>, n: i64): f32 {
    let mut s: f32 = 0.0
    let mut i: i64 = 0
    while i < n { s = s + p[i]  i = i + 1 }
    return s
}
def main(): f64 {
    let a: Ptr<f32> = alloc_f32(8)
    let b: Ptr<f32> = alloc_f32(8)
    let o: Ptr<f32> = alloc_f32(8)
    let f1: i64 = fill(a, 1.5, 8)
    let f2: i64 = fill(b, 2.5, 8)
    let r: i64 = vadd(o, a, b, 8)
    return (total(o, 8) as f64)
}
"#,
        "lanes_f32",
        "vadd",
    );
    assert_eq!(v, 32.0, "8 * (1.5 + 2.5)");
    assert!(
        body.contains("<4 x f32>"),
        "f32 should widen four at a time, body was:\n{body}"
    );
}

/// `f64` is twice the width, so half as many fit. Writing 4 here, as the
/// old code did, asked for a 256-bit vector that no backend in the
/// ladder accepts.
#[test]
fn a_double_kernel_uses_two_lanes() {
    let (v, body) = build(
        r#"
import prelude
import simd
def vadd(o: Ptr<f64>, a: Ptr<f64>, b: Ptr<f64>, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { o[i] = a[i] + b[i]  i = i + 1 }
    return n
}
def fill(p: Ptr<f64>, v: f64, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { p[i] = v  i = i + 1 }
    return n
}
def total(p: Ptr<f64>, n: i64): f64 {
    let mut s: f64 = 0.0
    let mut i: i64 = 0
    while i < n { s = s + p[i]  i = i + 1 }
    return s
}
def main(): f64 {
    let a: Ptr<f64> = alloc_f64(8)
    let b: Ptr<f64> = alloc_f64(8)
    let o: Ptr<f64> = alloc_f64(8)
    let f1: i64 = fill(a, 1.5, 8)
    let f2: i64 = fill(b, 2.5, 8)
    let r: i64 = vadd(o, a, b, 8)
    return total(o, 8)
}
"#,
        "lanes_f64",
        "vadd",
    );
    assert_eq!(v, 32.0, "8 * (1.5 + 2.5)");
    assert!(
        body.contains("<2 x f64>"),
        "f64 should widen two at a time, body was:\n{body}"
    );
    assert!(
        !body.contains("<4 x f64>"),
        "four f64 lanes is 256 bits, which no backend in the ladder accepts"
    );
}
