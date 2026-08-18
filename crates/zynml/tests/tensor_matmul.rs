//! A two-dimensional tensor, and the kernel that needs one.
//!
//! A tensor without a shape is a buffer. Shape is what makes `at(i, j)`
//! mean something, and matmul is the kernel that exists because of it.
//! Written in ZynML the whole way down: the struct, the accessors, and
//! the three loops.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig};

const TENSOR: &str = r#"
import prelude
import simd

struct Tensor {
    data: Ptr<f32>,
    rows: i64,
    cols: i64
}

impl Tensor {
    def zeros(rows: i64, cols: i64): Tensor {
        let n: i64 = rows * cols
        let buf: Ptr<f32> = alloc_f32(n)
        let mut i: i64 = 0
        while i < n {
            buf[i] = 0.0
            i = i + 1
        }
        return Tensor { data: buf, rows: rows, cols: cols }
    }

    def at(self, r: i64, c: i64): f32 {
        return self.data[r * self.cols + c]
    }

    // Writing through the buffer the tensor holds, which is what an
    // accessor on a shaped tensor has to be able to do.
    def put(self, r: i64, c: i64, v: f32): i64 {
        self.data[r * self.cols + c] = v
        return 0
    }

    def size(self): i64 {
        return self.rows * self.cols
    }

    def release(self) {
        free(self.data)
    }
}

def matmul(a: Tensor, b: Tensor): Tensor {
    let out: Tensor = Tensor::zeros(a.rows, b.cols)
    let mut i: i64 = 0
    while i < a.rows {
        let mut j: i64 = 0
        while j < b.cols {
            let mut acc: f32 = 0.0
            let mut k: i64 = 0
            while k < a.cols {
                acc = acc + a.at(i, k) * b.at(k, j)
                k = k + 1
            }
            let w: i64 = out.put(i, j, acc)
            j = j + 1
        }
        i = i + 1
    }
    return out
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

/// Shape is carried and the accessors agree on it.
#[test]
fn a_shaped_tensor_addresses_by_row_and_column() {
    assert_eq!(
        run(r#"
        def main(): f64 {
            let m: Tensor = Tensor::zeros(2, 3)
            let w: i64 = m.put(1, 2, 7.5)
            let v: f32 = m.at(1, 2)
            m.release()
            return (v as f64)
        }
        "#),
        7.5
    );
}

/// The element written at one position is not visible at another.
#[test]
fn positions_are_distinct() {
    assert_eq!(
        run(r#"
        def main(): f64 {
            let m: Tensor = Tensor::zeros(2, 3)
            let w1: i64 = m.put(0, 0, 1.0)
            let w2: i64 = m.put(1, 2, 4.0)
            let s: f32 = m.at(0, 0) * 10.0 + m.at(1, 2) + m.at(0, 1)
            m.release()
            return (s as f64)
        }
        "#),
        14.0
    );
}

/// [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]].
#[test]
fn matmul_computes_the_product() {
    assert_eq!(
        run(r#"
        def main(): f64 {
            let a: Tensor = Tensor::zeros(2, 2)
            let b: Tensor = Tensor::zeros(2, 2)
            let a1: i64 = a.put(0, 0, 1.0)
            let a2: i64 = a.put(0, 1, 2.0)
            let a3: i64 = a.put(1, 0, 3.0)
            let a4: i64 = a.put(1, 1, 4.0)
            let b1: i64 = b.put(0, 0, 5.0)
            let b2: i64 = b.put(0, 1, 6.0)
            let b3: i64 = b.put(1, 0, 7.0)
            let b4: i64 = b.put(1, 1, 8.0)
            let c: Tensor = matmul(a, b)
            // Every element, weighted so a wrong one anywhere shows.
            let s: f32 = c.at(0,0) * 1000.0 + c.at(0,1) * 100.0
                       + c.at(1,0) * 10.0 + c.at(1,1)
            return (s as f64)
        }
        "#),
        19.0 * 1000.0 + 22.0 * 100.0 + 43.0 * 10.0 + 50.0
    );
}

/// A non-square product, so the shapes cannot be standing in for each
/// other: (2x3) x (3x2) = (2x2), every element 1*1*3 = 3.
#[test]
fn matmul_handles_rectangular_shapes() {
    assert_eq!(
        run(r#"
        def fill(t: Tensor, v: f32): i64 {
            let mut i: i64 = 0
            while i < t.rows {
                let mut j: i64 = 0
                while j < t.cols {
                    let w: i64 = t.put(i, j, v)
                    j = j + 1
                }
                i = i + 1
            }
            return 0
        }
        def main(): f64 {
            let a: Tensor = Tensor::zeros(2, 3)
            let b: Tensor = Tensor::zeros(3, 2)
            let f1: i64 = fill(a, 1.0)
            let f2: i64 = fill(b, 1.0)
            let c: Tensor = matmul(a, b)
            return ((c.at(0,0) + c.at(0,1) + c.at(1,0) + c.at(1,1) + (c.rows * c.cols) as f32) as f64)
        }
        "#),
        16.0
    );
}
