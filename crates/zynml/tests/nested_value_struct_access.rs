//! Reading a field of a value struct that is itself a field of another
//! value struct.
//!
//! `l.from.x` returns a value with a neighbouring field in its high
//! bits, while `l.to.x` at the same depth is correct, so the defect is
//! in how a nested aggregate field is materialised rather than in
//! offsets. `ExtractValue` in the Cranelift backend loads on its last
//! index unconditionally; a field that is itself an aggregate needs a
//! different treatment, and the surrounding SSA has to agree on what
//! that is (returning the address alone makes it worse, so the
//! representation contract is what needs settling).
//!
//! This blocks nested WRITES on value structs too: a write rebuilds the
//! aggregate from a read, so it inherits the read's wrong value.
//! Handler state is unaffected because it is a `@reference` struct and
//! goes through the pointer path.

use zynml::ZynML;

const SRC: &str = r#"
struct Point { x: i64, y: i64 }
struct Line { from: Point, to: Point }

def from_x(): i64 {
    let l = Line { from: Point { x: 1, y: 2 }, to: Point { x: 3, y: 4 } }
    return l.from.x
}
def from_y(): i64 {
    let l = Line { from: Point { x: 1, y: 2 }, to: Point { x: 3, y: 4 } }
    return l.from.y
}
def to_x(): i64 {
    let l = Line { from: Point { x: 1, y: 2 }, to: Point { x: 3, y: 4 } }
    return l.to.x
}
def to_y(): i64 {
    let l = Line { from: Point { x: 1, y: 2 }, to: Point { x: 3, y: 4 } }
    return l.to.y
}
"#;

#[test]
#[ignore = "known defect: a nested value-struct field read returns a neighbouring field in its high bits"]
fn a_nested_value_struct_read_is_correct() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(SRC).expect("load");
    for (f, want) in [("from_x", 1i64), ("from_y", 2), ("to_x", 3), ("to_y", 4)] {
        let got: i64 = rt.call_with_result(f).expect("call");
        assert_eq!(got, want, "{f} read the wrong value");
    }
}

/// The half that does work today, kept un-ignored so a fix to the
/// above cannot silently break it.
#[test]
fn the_second_nested_field_reads_correctly() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(SRC).expect("load");
    let x: i64 = rt.call_with_result("to_x").expect("call");
    let y: i64 = rt.call_with_result("to_y").expect("call");
    assert_eq!((x, y), (3, 4));
}
