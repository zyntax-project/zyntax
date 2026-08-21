//! A list literal is laid out for the parameter it is passed into.
//!
//! An integer literal types itself `i32`, and a list takes its element
//! type from its first element, so `[3, 2]` is a list of `i32` however
//! it is used. A `List<i64>` parameter needs those eight bytes apart.
//! Lowering already had the mechanism -- the declared element type wins
//! over what the literal infers about itself -- but it was only ever set
//! for a `let`. At a call it was not, so the data went in four bytes
//! apart and came back out eight at a time, and `[3, 2]` arrived as one
//! word of `0x2_00000003`.
//!
//! A one-element list hid it: the packed high half is zero, so `[6]`
//! reads back as 6 and only two or more elements go wrong. That is why
//! `reshape([6])` worked and `reshape([3, 2])` returned null.
//!
//! Checked through the tensor surface, because that is where the shape
//! is read back out again. A synthetic `def f(xs: List<i64>)` reaching
//! for `xs[1]` runs into separate, older faults that mask this one.

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

/// An instance method taking a shape of more than one dimension.
///
/// `reshape` returned null here: the two dimensions arrived packed into
/// one word, so the element count it computed did not match the tensor
/// it was handed.
#[test]
fn an_instance_method_reads_a_multi_element_shape() {
    assert_eq!(
        run(r#"
import prelude
import tensor

def main(): f64 {
    let t: Tensor = Tensor::arange(1.0, 7.0, 1.0)
    let r: Tensor = t.reshape([3, 2])
    // Both dimensions, and an element from the far end, so a shape that
    // happened to be right over data that was not cannot pass.
    return (r.shape(0) * 1000 + r.shape(1) * 100) as f64 + (r.get(5) as f64)
}
"#),
        3000.0 + 200.0 + 6.0
    );
}

/// A static method taking a shape and a value.
///
/// `full` returned null for the same reason `reshape` did. `zeros` did
/// not, which is why it is not the one checked here: it already laid
/// its shape out correctly and would pass either way.
#[test]
fn a_static_method_reads_a_multi_element_shape() {
    assert_eq!(
        run(r#"
import prelude
import tensor

def main(): f64 {
    let t: Tensor = Tensor::full([4, 3], 7.0)
    let d0: i64 = t.shape(0)
    let d1: i64 = t.shape(1)
    return (d0 * 1000 + d1 * 100) as f64 + (t.get(11) as f64)
}
"#),
        4000.0 + 300.0 + 7.0
    );
}

/// Three dimensions, so a fault that happened to survive two would not.
#[test]
fn three_dimensions_survive_the_call() {
    assert_eq!(
        run(r#"
import prelude
import tensor

def main(): f64 {
    let t: Tensor = Tensor::full([2, 3, 4], 1.0)
    let d0: i64 = t.shape(0)
    let d1: i64 = t.shape(1)
    let d2: i64 = t.shape(2)
    return (d0 * 10000 + d1 * 100 + d2) as f64
}
"#),
        20304.0
    );
}

/// A one-element shape, which read back correctly even while this was
/// broken, so it has to keep doing so.
#[test]
fn a_single_element_shape_still_agrees() {
    assert_eq!(
        run(r#"
import prelude
import tensor

def main(): f64 {
    let t: Tensor = Tensor::arange(1.0, 7.0, 1.0)
    let r: Tensor = t.reshape([6])
    return (r.shape(0) as f64) + (r.get(5) as f64)
}
"#),
        6.0 + 6.0
    );
}
