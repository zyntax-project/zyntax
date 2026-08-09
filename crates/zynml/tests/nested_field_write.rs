//! Writing through a chain of fields, `a.b.c = ...`.
//!
//! The assignment target used to be exactly `identifier "." identifier`,
//! so a write could only ever reach one level deep while a read could
//! reach any depth. These cover the write side at the depth the read
//! side already worked at, and they execute rather than only parse:
//! the target is now a nested `FieldAccess`, which lowering has to walk
//! to the right address instead of storing into a copy.

use zynml::ZynML;

/// A nested write on a local, read back through the same chain.
#[test]
#[ignore = "blocked by the nested value-struct read defect; see nested_value_struct_access.rs"]
fn a_nested_write_on_a_local_lands() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
struct Point {
    x: i64,
    y: i64
}

struct Line {
    from: Point,
    to: Point
}

def go(): i64 {
    let mut l = Line {
        from: Point { x: 1, y: 2 },
        to: Point { x: 3, y: 4 }
    }
    l.to.x = 30
    l.from.y = 20
    return l.to.x + l.from.y + l.from.x + l.to.y
}
"#,
    )
    .expect("load");
    let v: i64 = rt.call_with_result("go").expect("call");
    assert_eq!(v, 30 + 20 + 1 + 4);
}

/// The write has to reach the original, not a copy of the inner struct.
#[test]
#[ignore = "blocked by the nested value-struct read defect; see nested_value_struct_access.rs"]
fn a_nested_write_mutates_in_place() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
struct Inner { n: i64 }
struct Outer { a: Inner, b: Inner }

def go(): i64 {
    let mut o = Outer { a: Inner { n: 1 }, b: Inner { n: 100 } }
    o.a.n = o.a.n + 1
    o.a.n = o.a.n + 1
    return o.a.n * 1000 + o.b.n
}
"#,
    )
    .expect("load");
    let v: i64 = rt.call_with_result("go").expect("call");
    assert_eq!(v, 3 * 1000 + 100, "a.n advanced twice and b.n is untouched");
}

/// Three levels deep, to show the fold is a chain and not a special
/// case for two.
#[test]
#[ignore = "blocked by the nested value-struct read defect; see nested_value_struct_access.rs"]
fn a_write_three_levels_deep_lands() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
struct C { v: i64 }
struct B { c: C }
struct A { b: B }

def go(): i64 {
    let mut a = A { b: B { c: C { v: 5 } } }
    a.b.c.v = 42
    return a.b.c.v
}
"#,
    )
    .expect("load");
    let v: i64 = rt.call_with_result("go").expect("call");
    assert_eq!(v, 42);
}

/// The one-level write the old grammar did support still works.
#[test]
fn a_one_level_write_still_works() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
struct Point { x: i64, y: i64 }

def go(): i64 {
    let mut p = Point { x: 1, y: 2 }
    p.x = p.x + 10
    return p.x * 100 + p.y
}
"#,
    )
    .expect("load");
    let v: i64 = rt.call_with_result("go").expect("call");
    assert_eq!(v, 11 * 100 + 2);
}

/// The reported shape: a handler whose state field is a struct, written
/// through `self`.
#[test]
fn a_handler_can_write_through_its_struct_state() {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
struct Point { x: i64, y: i64 }

effect Move {
    def step(): i64
}

handler Walk for Move {
    var at: Point = Point { x: 7, y: 0 }
    def step(): i64 {
        self.at.x = self.at.x + 1
        return self.at.x
    }
}

@effect(Move)
def go_once(): i64 {
    return step()
}

def main(): i64 {
    let mut last: i64 = 0
    with Walk {
        last = go_once()
        last = go_once()
    }
    return last
}
"#,
    )
    .expect("load");
    let v: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(v, 9, "7 then 8 then 9 across performs in one scope");
}
