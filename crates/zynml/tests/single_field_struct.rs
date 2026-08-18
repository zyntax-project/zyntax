//! A struct with one field behaves like any other struct.
//!
//! A struct wrapping a single scalar is carried as that scalar rather
//! than by address, which is the right thing to do with it: no memory,
//! no copy, the value in a register. Setting and reading its field then
//! has to mean the value itself, because there is nowhere to address.
//! Read as an address, the scalar is dereferenced as though it were one.
//!
//! The shape matters for a numerical API, where a newtype over one
//! number is how a unit, an index space or a handle gets a type of its
//! own.

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

/// Build one and read its field back.
#[test]
fn a_single_field_struct_round_trips() {
    assert_eq!(
        run(r#"
import prelude
struct U { a: i64 }
def main(): f64 {
    let x: U = U { a: 7 }
    return (x.a as f64)
}
"#),
        7.0
    );
}

/// Through a method, so the value crosses a call boundary as well.
#[test]
fn a_single_field_struct_reads_through_a_method() {
    assert_eq!(
        run(r#"
import prelude
struct U { a: i64 }
impl U { def get(self): i64 { return self.a } }
def main(): f64 {
    let x: U = U { a: 7 }
    return (x.get() as f64)
}
"#),
        7.0
    );
}

/// A float field, which travels in a different register class.
#[test]
fn a_single_float_field_round_trips() {
    assert_eq!(
        run(r#"
import prelude
struct U { a: f32 }
impl U { def get(self): f32 { return self.a } }
def main(): f64 {
    let x: U = U { a: 2.5 }
    return (x.get() as f64)
}
"#),
        2.5
    );
}

/// Two of them, so one cannot be standing in for the other.
#[test]
fn two_single_field_structs_stay_distinct() {
    assert_eq!(
        run(r#"
import prelude
struct U { a: i64 }
def make(n: i64): U { return U { a: n } }
def main(): f64 {
    let x: U = make(3)
    let y: U = make(4)
    return ((x.a * 10 + y.a) as f64)
}
"#),
        34.0
    );
}
