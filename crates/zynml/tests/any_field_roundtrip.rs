//! A field declared `Any` reads back as its value, not as its box.
//!
//! Storing into an `Any` field boxes the value, so reading one has to
//! unbox it. The unbox is emitted where an annotated binding states the
//! concrete type it wants, which made this look like a missing coercion
//! at the field read. It was not: the coercion at the binding fired, but
//! the field's own type never reached it. A binding with no annotation
//! carries the name its initializer resolved to rather than the registry
//! entry, and the field-type lookup only understood the latter, so the
//! read typed as `Unknown` and the `Any -> f64` unbox classified as an
//! ordinary conversion and became a no-op. The `i64` in the struct then
//! flowed into an `fadd` as though it were the stored `f64`.

use zynml::ZynML;

fn run(src: &str) -> i64 {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(src).expect("load");
    rt.call_with_result("main").expect("call")
}

/// The float case, which is where a returned box address is most
/// obvious: it arrives as an enormous integer rather than the sum.
#[test]
fn a_float_survives_an_any_field() {
    let out = run(r#"
        struct Bag { payload: Any }
        def main(): i64 {
            let mut sum: f64 = 0.0
            let mut i: i64 = 0
            while i < 1000 {
                let bag = Bag { payload: 1.5 }
                let v: f64 = bag.payload
                sum = sum + v
                i = i + 1
            }
            return sum as i64
        }
    "#);
    assert_eq!(out, 1500, "1000 * 1.5, not a box address");
}

/// An integer payload takes a different getter, so it is worth its own
/// case: here a box address would still be an integer and could pass
/// unnoticed.
#[test]
fn an_integer_survives_an_any_field() {
    let out = run(r#"
        struct Bag { payload: Any }
        def main(): i64 {
            let bag = Bag { payload: 42 }
            let v: i64 = bag.payload
            return v
        }
    "#);
    assert_eq!(out, 42);
}

/// The binding annotated on the struct rather than inferred, so the
/// object type reaches the field lookup by the other spelling.
#[test]
fn an_annotated_binding_reaches_the_same_field() {
    let out = run(r#"
        struct Bag { payload: Any }
        def main(): i64 {
            let bag: Bag = Bag { payload: 7.5 }
            let v: f64 = bag.payload
            return v as i64
        }
    "#);
    assert_eq!(out, 7);
}

/// A field that is not `Any` is unaffected: it was never boxed, so
/// nothing may be unboxed out of it.
#[test]
fn a_concrete_field_is_left_alone() {
    let out = run(r#"
        struct Point { x: f64, y: f64 }
        def main(): i64 {
            let p = Point { x: 3.5, y: 4.5 }
            let a: f64 = p.x
            let b: f64 = p.y
            return (a + b) as i64
        }
    "#);
    assert_eq!(out, 8);
}

/// Both a boxed and a concrete field on the one struct, so the unbox
/// has to be chosen per field rather than per struct.
#[test]
fn a_boxed_and_a_concrete_field_sit_together() {
    let out = run(r#"
        struct Mixed { tag: i64, payload: Any }
        def main(): i64 {
            let m = Mixed { tag: 5, payload: 2.5 }
            let t: i64 = m.tag
            let p: f64 = m.payload
            return t + (p as i64)
        }
    "#);
    assert_eq!(out, 7);
}
