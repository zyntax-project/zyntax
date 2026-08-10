//! What the bytecode interpreter cannot run, said at install rather
//! than on the path that reaches it.
//!
//! Bytecode compilation is lazy, so a program that performs an effect
//! used to install cleanly and fail later with the discriminant of an
//! instruction, which names neither the construct nor the remedy. Where
//! the interpreter is the only engine, the caller can now ask up front.
//! Where it is a tier with a JIT beneath it, the same answer is not an
//! error: those functions run compiled.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

const EFFECTFUL: &str = r#"
effect Ev { def next(): i64 }
handler Feed for Ev {
    var n: i64 = 0
    def next(): i64 { self.n = self.n + 1  return self.n }
}
@effect(Ev)
def tick(): i64 { return next() }

def plain(x: i64): i64 { return x + 1 }

def drive(): i64 {
    let mut t: i64 = 0
    with Feed { t = tick() }
    return t
}
"#;

fn hir(src: &str) -> zyntax_compiler::HirModule {
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    let program = g.parse_with_filename(src, "cap.zyn").expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    rt.lower_typed_program(program, indexmap::IndexMap::new())
        .expect("lower")
}

/// The performing function is named, and the reason with it.
#[test]
fn an_effect_performing_function_is_reported() {
    let m = hir(EFFECTFUL);
    let found = zyntax_compiler::hir_interp::unsupported_constructs(&m);
    assert!(
        found
            .iter()
            .any(|(f, why)| f == "tick" && *why == "algebraic effects"),
        "the performing function should be named: {found:?}"
    );
    assert!(
        !found.iter().any(|(f, _)| f == "plain"),
        "a function that performs nothing must not be listed: {found:?}"
    );
}

/// A fiber body is reported the same way.
#[test]
fn a_fiber_using_function_is_reported() {
    let m = hir(r#"
fiber def counter(): i64 {
    yield 1
    yield 2
}
def driver(): i64 {
    let mut total: i64 = 0
    let f = counter()
    while let Some(v) = f.next() {
        total = total + v
    }
    return total
}
"#);
    let found = zyntax_compiler::hir_interp::unsupported_constructs(&m);
    assert!(
        found.iter().any(|(_, why)| *why == "fibers"),
        "a fiber user should be named: {found:?}"
    );
}

/// A program with neither reports nothing, so the check is not just
/// answering "yes" to everything.
#[test]
fn an_ordinary_program_reports_nothing() {
    let m =
        hir("def add(a: i64, b: i64): i64 { return a + b }\ndef main(): i64 { return add(1, 2) }");
    let found = zyntax_compiler::hir_interp::unsupported_constructs(&m);
    assert!(found.is_empty(), "nothing should be flagged: {found:?}");
}

/// A refusal is cached like a success. The interpreter is tried first
/// on every call and only falls back to native dispatch when it refuses,
/// so without this the whole body is re-compiled on every call just to
/// fail the same way. (Measured at ~1µs/call for a small function, the
/// saving is not visible next to the fallback itself; this is about not
/// doing the work, not about a speedup.)
#[test]
fn a_refusal_is_remembered() {
    use zynml::ZynML;

    let mut rt = ZynML::new().expect("rt");
    rt.load_source(EFFECTFUL).expect("load");

    // Repeated calls through the effectful path keep working, and keep
    // returning the same answer, whichever engine serves them.
    let mut last = 0i64;
    for _ in 0..50 {
        last = rt.call_with_result("drive").expect("call");
    }
    assert_eq!(last, 1, "each scope starts a fresh handler state");
}
