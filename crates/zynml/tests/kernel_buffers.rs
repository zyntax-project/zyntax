//! Kernels over buffers passed in, which is the shape a kernel language
//! is for.
//!
//! `def saxpy(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64)` is the
//! signature an ML or HPC kernel has: the caller owns the memory and the
//! kernel writes through it. That needs indexing to work on a parameter,
//! not only on a buffer allocated in the same function, and it needs an
//! argument to arrive at the type its parameter declares.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

fn run(src: &str) -> f64 {
    let plugins = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../plugins/target/zrtl");
    let cfg = ZynMLConfig {
        plugins_dir: plugins.to_string_lossy().to_string(),
        ..ZynMLConfig::default()
    };
    let mut z = ZynML::with_config(cfg).expect("runtime");
    z.load_source(src).expect("should compile");
    z.call_with_result::<f64>("main").expect("should run")
}

/// Reading and writing through buffer parameters, scalar: for 8
/// elements, 1.5 * 2.0 + 2.5 = 5.5, and lane 3 carries it.
#[test]
fn a_kernel_writes_through_its_output_buffer() {
    assert_eq!(
        run(r#"
import prelude
import simd

def saxpy(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, k: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n {
        out[i] = a[i] * k + b[i]
        i = i + 1
    }
    return n
}

def fill(p: Ptr<f32>, v: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n {
        p[i] = v
        i = i + 1
    }
    return n
}

def main(): f64 {
    let a: Ptr<f32> = alloc_f32(8)
    let b: Ptr<f32> = alloc_f32(8)
    let out: Ptr<f32> = alloc_f32(8)
    let f1: i64 = fill(a, 1.5, 8)
    let f2: i64 = fill(b, 2.5, 8)
    let r: i64 = saxpy(out, a, b, 2.0, 8)
    return (out[3] as f64)
}
"#),
        5.5
    );
}

/// The same kernel with whole-vector access, which is what makes it a
/// kernel rather than a loop: 4 lanes at a time through parameters.
#[test]
fn a_kernel_sweeps_its_buffers_by_vector() {
    assert_eq!(
        run(r#"
import prelude
import simd

def vadd(out: Ptr<f32>, a: Ptr<f32>, b: Ptr<f32>, n: i64): i64 {
    let mut i: i64 = 0
    while i + 4 <= n {
        vstore_f32x4(out + i, vload_f32x4(a + i) + vload_f32x4(b + i))
        i = i + 4
    }
    while i < n {
        out[i] = a[i] + b[i]
        i = i + 1
    }
    return n
}

def fill(p: Ptr<f32>, v: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n {
        p[i] = v
        i = i + 1
    }
    return n
}

def main(): f64 {
    let a: Ptr<f32> = alloc_f32(7)
    let b: Ptr<f32> = alloc_f32(7)
    let out: Ptr<f32> = alloc_f32(7)
    let f1: i64 = fill(a, 1.5, 7)
    let f2: i64 = fill(b, 3.0, 7)
    let r: i64 = vadd(out, a, b, 7)
    // Element 6 is in the scalar tail, element 0 in the vector body.
    return ((out[0] + out[6]) as f64)
}
"#),
        9.0
    );
}
