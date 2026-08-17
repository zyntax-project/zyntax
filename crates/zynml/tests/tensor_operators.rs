//! Operator syntax for a tensor written in ZynML.
//!
//! `tensor_add(a, b)` is a kernel; `a + b` is a tensor library. The
//! operators are ordinary trait impls whose bodies are the same SIMD
//! loops the free functions in `tensor_in_zynml.rs` run, so the
//! arithmetic stays inline HIR while the surface reads the way a
//! numerical API should.
//!
//! These go through the ZynML front door rather than `ZyntaxRuntime`
//! directly, because that is the path supplying entry names, and
//! lowering seeds what it emits from those.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

/// A tensor whose free-function form is already covered; here the same
/// loops live in `impl` blocks instead.
const TENSOR: &str = r#"
import prelude
import simd

struct Tensor {
    data: Ptr<f32>,
    len: i64
}

def filled(n: i64, v: f32): Tensor {
    let buf: Ptr<f32> = alloc_f32(n)
    let fill: f32x4 = f32x4::splat(v)
    let mut off: i64 = 0
    while off < n {
        vstore_f32x4(buf + off, fill)
        off = off + 4
    }
    return Tensor { data: buf, len: n }
}

impl Tensor {
    def sum(self): f32 {
        let mut acc: f32x4 = f32x4::splat(0.0)
        let mut off: i64 = 0
        while off < self.len {
            acc = acc + vload_f32x4(self.data + off)
            off = off + 4
        }
        return acc.sum()
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor
    def add(self, rhs: Tensor): Tensor {
        let out: Ptr<f32> = alloc_f32(self.len)
        let mut off: i64 = 0
        while off < self.len {
            vstore_f32x4(out + off, vload_f32x4(self.data + off) + vload_f32x4(rhs.data + off))
            off = off + 4
        }
        return Tensor { data: out, len: self.len }
    }
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

/// A reduction written as a method rather than a free function.
#[test]
fn a_method_can_reduce_a_buffer() {
    assert_eq!(
        run(r#"
        def main(): f64 {
            let a: Tensor = filled(8, 1.5)
            return (a.sum() as f64)
        }
        "#),
        12.0
    );
}

/// `a + b` dispatching to a ZynML `Add` impl: (1.5 + 3.0) * 8 = 36.
#[test]
fn addition_reads_as_addition() {
    assert_eq!(
        run(r#"
        def main(): f64 {
            let a: Tensor = filled(8, 1.5)
            let b: Tensor = filled(8, 3.0)
            let c: Tensor = a + b
            return (c.sum() as f64)
        }
        "#),
        36.0
    );
}
