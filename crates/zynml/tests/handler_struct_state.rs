//! Aggregates and handler state.
//!
//! A struct literal is fine inside a handler op body, and the state it
//! computes survives fiber resumes and reads from a pushed frame. What
//! does not parse is a struct literal as a handler FIELD initializer —
//! `var at: Point = Point { x: 0, y: 0 }` — even though the identical
//! literal parses in a `let`. Field defaults appear to accept scalar
//! literals only.
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

const SRC: &str = r#"
struct Point:
    x: i64
    y: i64

effect Tracker {
    def move_by(dx: i64, dy: i64)
    def get_x(): i64
    def get_y(): i64
}

handler Trail for Tracker {
    var x: i64 = 0
    var y: i64 = 0
    def move_by(dx: i64, dy: i64) {
        let probe = Point { x: 1, y: 2 }
        self.x = self.x + dx + probe.x - 1
        self.y = self.y + dy
    }
    def get_x(): i64 { return self.x }
    def get_y(): i64 { return self.y }
}

@effect(Tracker)
fiber def walker(): i64 {
    let mut steps: i64 = 0
    while steps < 100 {
        steps = steps + 1
        move_by(3, 4)
        yield steps
    }
    return steps
}

@effect(Tracker)
def read_x(): i64 { return get_x() }

@effect(Tracker)
def read_y(): i64 { return get_y() }
"#;

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    let mut rt = TieredRuntime::new(config).expect("rt");
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(SRC, "struct_state.zyn").expect("parse"))
        .expect("compile");
    rt
}

#[test]
fn a_struct_literal_works_inside_a_handler_op() {
    let mut rt = runtime();
    let walker = rt.get_fiber("walker").expect("fiber");
    let trail = rt.get_effect_handler("Trail").expect("handler");
    let inst = rt.new_handler_instance(trail).expect("instance");
    rt.bind_fiber_handler_instance(walker, inst).expect("bind");

    for _ in 0..3 {
        rt.resume_fiber(walker).expect("resume");
    }

    let frame = rt.push_handler_instance(inst).expect("push");
    let x = rt.call_raw("read_x", &[]).expect("read x");
    let y = rt.call_raw("read_y", &[]).expect("read y");
    rt.pop_effect_handler(frame);

    assert_eq!(x, ZyntaxValue::Int(9), "three moves of dx=3");
    assert_eq!(y, ZyntaxValue::Int(12), "three moves of dy=4");
    rt.drop_fiber(walker).expect("drop");
}

/// A struct literal as a handler field's default.
///
/// Fails at the initializer with a parse error, where the same literal
/// in a `let` inside an op body parses. Ignored because it documents a
/// grammar gap rather than a regression.
#[test]
#[ignore = "handler field initializers take scalar literals, not struct literals"]
fn a_handler_field_can_default_to_a_struct_literal() {
    const WITH_FIELD: &str = r#"
struct Point:
    x: i64
    y: i64

effect Tracker {
    def get_x(): i64
}

handler Trail for Tracker {
    var at: Point = Point { x: 7, y: 0 }
    def get_x(): i64 { return self.at.x }
}

@effect(Tracker)
def read_x(): i64 { return get_x() }
"#;
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    let parsed = g.parse_with_filename(WITH_FIELD, "field_default.zyn");
    assert!(
        parsed.is_ok(),
        "a struct literal should be usable as a field default: {:?}",
        parsed.err()
    );
}
