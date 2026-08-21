//! `a @ b` on the stdlib tensor.
//!
//! The operator has a trait, the grammar maps `@` to it, and the plugin
//! has carried a real matrix multiply since Accelerate was reached. What
//! the three did not have was a symbol in common: `$Tensor$matmul` named
//! the scalar dot product, so the trait impl was commented out and `@`
//! resolved to nothing.
//!
//! These check the operator end to end rather than the plugin function,
//! which its own tests already cover. A binding that returns a scalar
//! cannot satisfy them: the result is read back as a matrix.

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

/// The product of two square matrices, read back element by element.
///
/// Each element is weighted by a different power of ten, so a single
/// wrong entry cannot be absorbed by the others.
#[test]
fn a_square_product_is_a_matrix() {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    // weighted 1, 10, 100, 1000 -> 19 + 220 + 4300 + 50000
    assert_eq!(
        run(r#"
import prelude
import tensor

def main(): f64 {
    let a: Tensor = Tensor::zeros_2d(2, 2)
    a.set(0, 1.0)  a.set(1, 2.0)  a.set(2, 3.0)  a.set(3, 4.0)
    let b: Tensor = Tensor::zeros_2d(2, 2)
    b.set(0, 5.0)  b.set(1, 6.0)  b.set(2, 7.0)  b.set(3, 8.0)
    let c: Tensor = a @ b
    let s: f32 = c.get(0) * 1.0
        + c.get(1) * 10.0
        + c.get(2) * 100.0
        + c.get(3) * 1000.0
    return (s as f64)
}
"#),
        19.0 + 220.0 + 4300.0 + 50000.0
    );
}

/// The operands are not interchangeable, so an implementation that
/// happened to be symmetric would not pass.
#[test]
fn the_operands_do_not_commute() {
    assert_eq!(
        run(r#"
import prelude
import tensor

def main(): f64 {
    let a: Tensor = Tensor::zeros_2d(2, 2)
    a.set(0, 1.0)  a.set(1, 2.0)  a.set(2, 3.0)  a.set(3, 4.0)
    let b: Tensor = Tensor::zeros_2d(2, 2)
    b.set(0, 5.0)  b.set(1, 6.0)  b.set(2, 7.0)  b.set(3, 8.0)
    let ab: Tensor = a @ b
    let ba: Tensor = b @ a
    // ab[0][0] = 19, ba[0][0] = 23
    let d: f32 = ab.get(0) - ba.get(0)
    return (d as f64)
}
"#),
        -4.0
    );
}

/// A rectangular product, so the two shapes cannot stand in for each
/// other and a result carrying the wrong one is visible.
#[test]
fn a_rectangular_product_keeps_both_shapes() {
    // [[1,2,3],[4,5,6]] (2x3) @ [[1,2],[3,4],[5,6]] (3x2) = [[22,28],[49,64]]
    assert_eq!(
        run(r#"
import prelude
import tensor

def main(): f64 {
    let a: Tensor = Tensor::zeros_2d(2, 3)
    a.set(0, 1.0)  a.set(1, 2.0)  a.set(2, 3.0)
    a.set(3, 4.0)  a.set(4, 5.0)  a.set(5, 6.0)
    let b: Tensor = Tensor::zeros_2d(3, 2)
    b.set(0, 1.0)  b.set(1, 2.0)
    b.set(2, 3.0)  b.set(3, 4.0)
    b.set(4, 5.0)  b.set(5, 6.0)
    let c: Tensor = a @ b
    // 2x2 result, weighted so no element can be absorbed by another
    let s: f32 = c.get(0) * 1.0
        + c.get(1) * 10.0
        + c.get(2) * 100.0
        + c.get(3) * 1000.0
    return (s as f64)
}
"#),
        22.0 + 280.0 + 4900.0 + 64000.0
    );
}
