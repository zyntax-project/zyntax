//! Tensors whose length is not a whole number of vectors.
//!
//! Sweeping a buffer four lanes at a time only reaches a multiple of
//! four, so every earlier tensor test had to pick one. A tail loop reads
//! and writes the remainder one element at a time, which needs scalar
//! indexing through a `Ptr<T>`: `p[i]` for the read and `p[i] = v` for
//! the write.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

const TENSOR: &str = r#"
import prelude
import simd

struct Tensor {
    data: Ptr<f32>,
    len: i64
}

// Vectors while four remain, then one element at a time.
def filled(n: i64, v: f32): Tensor {
    let buf: Ptr<f32> = alloc_f32(n)
    let fill: f32x4 = f32x4::splat(v)
    let mut off: i64 = 0
    while off + 4 <= n {
        vstore_f32x4(buf + off, fill)
        off = off + 4
    }
    while off < n {
        buf[off] = v
        off = off + 1
    }
    return Tensor { data: buf, len: n }
}

def sum(t: Tensor): f32 {
    let mut acc: f32x4 = f32x4::splat(0.0)
    let mut off: i64 = 0
    while off + 4 <= t.len {
        acc = acc + vload_f32x4(t.data + off)
        off = off + 4
    }
    let mut total: f32 = acc.sum()
    while off < t.len {
        total = total + t.data[off]
        off = off + 1
    }
    return total
}

def add(a: Tensor, b: Tensor): Tensor {
    let out: Ptr<f32> = alloc_f32(a.len)
    let mut off: i64 = 0
    while off + 4 <= a.len {
        vstore_f32x4(out + off, vload_f32x4(a.data + off) + vload_f32x4(b.data + off))
        off = off + 4
    }
    while off < a.len {
        out[off] = a.data[off] + b.data[off]
        off = off + 1
    }
    return Tensor { data: out, len: a.len }
}
"#;

fn run(kernel: &str) -> f64 {
    let plugins = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugins/target/zrtl");
    let cfg = ZynMLConfig {
        plugins_dir: plugins.to_string_lossy().to_string(),
        ..ZynMLConfig::default()
    };
    let mut z = ZynML::with_config(cfg).expect("runtime");
    z.load_source(&format!("{TENSOR}\n{kernel}"))
        .expect("should compile");
    z.call_with_result::<f64>("main").expect("should run")
}

/// A length with a remainder in every position: 7 = one vector plus
/// three, so the tail runs and the vector body runs once. 7 * 2.0 = 14.
#[test]
fn a_length_that_is_not_a_whole_number_of_vectors() {
    assert_eq!(
        run(r#"
        def main(): f64 { return (sum(filled(7, 2.0)) as f64) }
        "#),
        14.0
    );
}

/// Shorter than a single vector, so only the tail runs. 3 * 5.0 = 15.
#[test]
fn a_length_below_one_vector() {
    assert_eq!(
        run(r#"
        def main(): f64 { return (sum(filled(3, 5.0)) as f64) }
        "#),
        15.0
    );
}

/// The exact multiple still works, so the tail is skipped cleanly.
#[test]
fn an_exact_multiple_still_works() {
    assert_eq!(
        run(r#"
        def main(): f64 { return (sum(filled(8, 1.5)) as f64) }
        "#),
        12.0
    );
}

/// Element-wise add across a ragged length: (1.5 + 3.0) * 5 = 22.5.
///
#[test]
fn elementwise_add_over_a_ragged_length() {
    assert_eq!(
        run(r#"
        def main(): f64 {
            let a: Tensor = filled(5, 1.5)
            let b: Tensor = filled(5, 3.0)
            let c: Tensor = add(a, b)
            return (sum(c) as f64)
        }
        "#),
        22.5
    );
}
