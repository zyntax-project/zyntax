//! `Null<T>`: the absent value of an optional, written with the type it
//! stands in for.
//!
//! A generic handler field has no context to infer from, so `var
//! content: T = Null<T>` says "no value yet" where the alternative was a
//! meaningless `= 0`. `Null<T>` is the same thing as `?T`: it resolves
//! to `Type::Optional`, the variant the backend lowers as a tagged union
//! (None = 0, Some = 1). Resolving it through the type registry instead
//! would mint a nominal placeholder and a value stored through it would
//! lose its real type.

use zynml::ZynML;
use zynml::{Grammar2, ZYNML_GRAMMAR};

fn parses(src: &str) -> bool {
    Grammar2::from_source(ZYNML_GRAMMAR)
        .expect("grammar")
        .parse_with_filename(src, "n.zyn")
        .is_ok()
}

fn run(src: &str) -> Result<i64, String> {
    let mut rt = ZynML::new().map_err(|e| e.to_string())?;
    rt.load_source(src).map_err(|e| e.to_string())?;
    rt.call_with_result::<i64>("main")
        .map_err(|e| e.to_string())
}

/// The shape this was added for.
#[test]
fn a_generic_handler_field_can_start_empty() {
    assert!(parses(
        "effect Signal<T> {\n    def get(): T\n    def set(val: T)\n}\n\n\
         handler MintedSignal<T> for Signal<T> {\n    var content: T = Null<T>\n    \
         def get(): T { return self.content }\n    def set(val: T) { self.content = val }\n}\n"
    ));
}

/// `Null<T>` parses in both positions, over every type expression the
/// language has, including the nested generics that close on `>>`.
#[test]
fn null_accepts_any_inner_type_expression() {
    for inner in [
        "i64",
        "T",
        "List<i64>",
        "List<List<i64>>",
        "Point",
        "(i64, i64)",
        "[i64; 4]",
        "?i64",
        "(i64) => i64",
        "Fiber<i64>",
        "Null<i64>",
    ] {
        assert!(
            parses(&format!(
                "def main(): i64 {{\n    let x = Null<{inner}>\n    return 1\n}}\n"
            )),
            "`Null<{inner}>` should parse as an expression"
        );
        assert!(
            parses(&format!("def f(x: Null<{inner}>): i64 {{ return 1 }}\n")),
            "`Null<{inner}>` should parse as a type"
        );
    }
}

/// `null` is a keyword, not a name, but only when it stands alone.
#[test]
fn null_does_not_eat_identifiers_that_start_with_it() {
    assert!(parses(
        "def main(): i64 {\n    let nullable = 3\n    return nullable\n}\n"
    ));
    assert!(parses("def nullish(): i64 { return 1 }\n"));
    assert!(parses(
        "def main(): i64 {\n    let x = null\n    return 1\n}\n"
    ));
}

/// A scalar optional round-trips through `Null<T>` exactly as `?T` does.
#[test]
fn a_scalar_optional_round_trips() {
    assert_eq!(
        run(
            "def main(): i64 {\n    let v: Null<i64> = Some(41)\n    match v {\n        \
             case Some(n) { return n + 1 }\n        case None() { return -1 }\n    }\n}\n"
        ),
        Ok(42)
    );
    assert_eq!(
        run(
            "def main(): i64 {\n    let v: Null<i64> = Null<i64>\n    match v {\n        \
             case Some(n) { return n }\n        case None() { return 99 }\n    }\n}\n"
        ),
        Ok(99)
    );
}

/// An optional over an aggregate. Both spellings, since they are one
/// type: the binding in `case Some(v)` has to learn that `v` is a
/// `Point`, or the first field access on it drops the whole function.
#[test]
fn an_optional_over_a_struct_works() {
    const WITH_QUESTION: &str = "struct Point { x: i64, y: i64 }\ndef main(): i64 {\n    \
        let p: ?Point = Some(Point { x: 3, y: 4 })\n    match p {\n        \
        case Some(v) { return v.x + v.y }\n        case None() { return -1 }\n    }\n}\n";
    let with_null = WITH_QUESTION.replace("?Point", "Null<Point>");
    assert_eq!(run(WITH_QUESTION), Ok(7));
    assert_eq!(run(&with_null), Ok(7));
}

/// The halves that do work, kept un-ignored so a fix to the above
/// cannot quietly break them.
#[test]
fn structs_and_scalar_optionals_each_work_alone() {
    assert_eq!(
        run("struct Point { x: i64, y: i64 }\ndef main(): i64 {\n    \
             let p = Point { x: 3, y: 4 }\n    return p.x + p.y\n}\n"),
        Ok(7)
    );
    assert_eq!(
        run(
            "def main(): i64 {\n    let v: ?i64 = Some(41)\n    match v {\n        \
             case Some(n) { return n + 1 }\n        case None() { return -1 }\n    }\n}\n"
        ),
        Ok(42)
    );
}

/// The resolution fixes any payload whose type the scrutinee carries.
#[test]
fn payload_bindings_learn_their_type() {
    assert_eq!(
        run("struct Point { x: i64, y: i64 }\ndef main(): i64 {\n    \
             let p: Null<Point> = Some(Point { x: 5, y: 6 })\n    \
             match p {\n        case Some(v) { return v.y }\n        \
             case None() { return -1 }\n    }\n}\n"),
        Ok(6)
    );
}

/// A list payload does not work, for a reason that has nothing to do
/// with optionals: annotating a list at all is broken. `let xs = [..]`
/// indexes correctly, `let xs: List<i64> = [..]` returns two elements
/// packed into one word, the same width signature as an integer literal
/// stored into a wider slot. Verified pre-existing by stashing the
/// optional work and re-running.
#[test]
#[ignore = "pre-existing: `let xs: List<i64> = [..]` mis-reads; unrelated to optionals"]
fn an_annotated_list_indexes_correctly() {
    assert_eq!(
        run(
            "def main(): i64 {\n    let xs: List<i64> = [10, 20, 30]\n    \
             return xs[0] + xs[2]\n}\n"
        ),
        Ok(40)
    );
    assert_eq!(
        run(
            "def main(): i64 {\n    let xs: Null<List<i64>> = Some([10, 20, 30])\n    \
             match xs {\n        case Some(v) { return v[0] + v[2] }\n        \
             case None() { return -1 }\n    }\n}\n"
        ),
        Ok(40)
    );
}

/// The unannotated form, kept un-ignored as the control.
#[test]
fn an_unannotated_list_indexes_correctly() {
    assert_eq!(
        run("def main(): i64 {\n    let xs = [10, 20, 30]\n    return xs[0] + xs[2]\n}\n"),
        Ok(40)
    );
}

/// The empty case still takes the None arm once the payload type is
/// known, so resolving the type did not turn every optional into a Some.
#[test]
fn an_empty_optional_over_a_struct_takes_the_none_arm() {
    assert_eq!(
        run("struct Point { x: i64, y: i64 }\ndef main(): i64 {\n    \
             let p: Null<Point> = Null<Point>\n    match p {\n        \
             case Some(v) { return v.x }\n        case None() { return 99 }\n    }\n}\n"),
        Ok(99)
    );
}
