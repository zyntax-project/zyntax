//! A tensor built through its own type name.
//!
//! `Tensor::filled(n, v)` is how a numerical API names a constructor, and
//! it reads that way only if the result can be used where it is made:
//! passed onward, chained into a method, returned. Resolving `Type::member`
//! means finding the function that member names, so it depends on the
//! declarations having been collected first.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

const TENSOR: &str = r#"
import prelude
import simd

struct Tensor {
    data: Ptr<f32>,
    len: i64
}

impl Tensor {
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

    def zeros(n: i64): Tensor {
        return Tensor::filled(n, 0.0)
    }

    def sum(self): f32 {
        let mut acc: f32x4 = f32x4::splat(0.0)
        let mut off: i64 = 0
        while off + 4 <= self.len {
            acc = acc + vload_f32x4(self.data + off)
            off = off + 4
        }
        let mut total: f32 = acc.sum()
        while off < self.len {
            total = total + self.data[off]
            off = off + 1
        }
        return total
    }

    def release(self) {
        free(self.data)
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor
    def add(self, rhs: Tensor): Tensor {
        let out: Ptr<f32> = alloc_f32(self.len)
        let mut off: i64 = 0
        while off + 4 <= self.len {
            vstore_f32x4(out + off, vload_f32x4(self.data + off) + vload_f32x4(rhs.data + off))
            off = off + 4
        }
        while off < self.len {
            out[off] = self.data[off] + rhs.data[off]
            off = off + 1
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

/// Bound to a name, which is the case that already worked.
#[test]
fn a_constructor_result_can_be_bound() {
    assert_eq!(
        run("def main(): f64 { let t: Tensor = Tensor::filled(7, 2.0) return (t.sum() as f64) }"),
        14.0
    );
}

/// Chained straight into a method, with nothing holding it in between.
#[test]
fn a_constructor_result_can_be_used_where_it_is_made() {
    assert_eq!(
        run("def main(): f64 { return (Tensor::filled(7, 2.0).sum() as f64) }"),
        14.0
    );
}

/// One constructor calling another by the same syntax.
#[test]
fn a_constructor_can_call_a_constructor() {
    assert_eq!(
        run("def main(): f64 { return (Tensor::zeros(8).sum() as f64) }"),
        0.0
    );
}

/// Constructor results as operands, at a ragged length:
/// (1.5 + 3.0) * 5 = 22.5.
#[test]
fn constructor_results_are_operands() {
    assert_eq!(
        run(r#"
        def main(): f64 {
            let c: Tensor = Tensor::filled(5, 1.5) + Tensor::filled(5, 3.0)
            let s: f32 = c.sum()
            c.release()
            return (s as f64)
        }
        "#),
        22.5
    );
}
