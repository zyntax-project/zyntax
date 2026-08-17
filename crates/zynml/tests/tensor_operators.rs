//! Operator syntax for a tensor written in ZynML.
//!
//! `tensor_add(a, b)` is a kernel; `a + b` is a tensor library. The
//! operators want to be ordinary trait impls whose bodies are the same
//! SIMD loops the free functions in `tensor_in_zynml.rs` already run, so
//! the arithmetic stays inline HIR while the surface reads the way a
//! numerical API should.
//!
//! It does not work yet, and the reason is narrower than it looks. A
//! SIMD loop is fine in a free function: every case in
//! `tensor_in_zynml.rs` sweeps a buffer with `vload_f32x4` and
//! `vstore_f32x4` and passes on both the interpreter and the JIT. Move
//! the identical loop inside an `impl` and it breaks, whether the method
//! is an operator or inherent, while an `impl` method with a loop-free
//! body is fine. So it is the combination of a method body and a vector
//! loop, not the operator machinery and not the pointer field.
//!
//! Ignored because they fail. Remove the attributes with the fix.

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
///
/// The loop is the one `tensor_sum` runs today. Inside an `impl` the
/// enclosing function never reaches the module, and the call reports
/// `main` as missing.
#[test]
#[ignore = "a vector loop inside an impl method stops the caller lowering"]
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
///
/// Same loop again, this time in an operator. Segfaults rather than
/// failing to lower, so the two are not one symptom.
#[test]
#[ignore = "a vector loop inside an operator impl segfaults"]
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
