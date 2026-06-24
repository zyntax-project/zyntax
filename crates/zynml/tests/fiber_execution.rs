//! End-to-end execution tests for fibers from ZynML source.
//!
//! The minimal worked example: a `fiber def` yields a few values;
//! the caller iterates with `while let Some(x) = f.next() { ... }`
//! and accumulates into a sum.
//!
//! This is the smallest "fibers actually run" test we can write.
//! If it returns the right number, every layer of the chain is
//! correct: parser, typed AST, call-site lowering, BuiltinClass
//! dispatch, `apply_krio_fiber_lowering`, runtime FFI marshalling
//! into `krio_fiber_*`, and the krio-fiber backend itself.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

fn run_returning_int(src: &str) -> Result<i64, String> {
    let mut rt = ZyntaxRuntime::new().map_err(|e| format!("rt: {e:?}"))?;
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).map_err(|e| format!("grammar: {e:?}"))?;
    let program = grammar
        .parse_with_filename(src, "<fiber_execution>")
        .map_err(|e| format!("parse: {e:?}"))?;
    rt.compile_typed_program(program)
        .map_err(|e| format!("compile: {e:?}"))?;
    // Use the JIT path (`call_raw`) — same path the `zynml run` CLI
    // takes. The BC interpreter path (`call_function_raw`) doesn't
    // yet implement every HIR instruction the krio-fiber lowering
    // produces (Discriminant variants for union ops).
    let result = rt
        .call_raw("main", &[])
        .map_err(|e| format!("call: {e:?}"))?;
    result
        .as_i64()
        .ok_or_else(|| format!("expected i64, got {result:?}"))
}

/// The minimum-viable fiber run: a fiber that yields three integers,
/// consumed via `while let Some(x) = f.next()`. Returns the sum.
///
/// 1 + 2 + 3 = 6. Exercises the full chain end-to-end: parser, typed
/// AST, call-site lowering, BuiltinClass dispatch, krio fiber
/// lowering, the runtime FFI marshalling into `krio_fiber_*`, the
/// match-on-Option pattern binding, and the while-let CFG desugar.
#[test]
fn fiber_yields_then_iterator_consume() {
    let src = r#"
        fiber def yields_one_two_three(): i64 {
            yield 1
            yield 2
            yield 3
        }

        def main(): i64 {
            let f = yields_one_two_three()
            let mut sum: i64 = 0
            while let Some(x) = f.next() {
                sum = sum + x
            }
            return sum
        }
    "#;
    let result = run_returning_int(src).expect("fiber program should execute");
    assert_eq!(result, 6, "1 + 2 + 3 should sum to 6");
}

/// Cooperative cancel via `Fiber::cancel()` — the consumer asks the
/// fiber to stop after two yields. The third+ values never reach the
/// loop body because the cancel-aware krio resume returns Done at
/// the next resume boundary.
///
/// 1 + 2 = 3 (cancel fires after the second iteration, third
/// `f.next()` yields `None`). First piece of the Wren-style
/// abort surface: a no-error-payload cancel signal from the caller.
#[test]
fn fiber_cancel_stops_iteration() {
    let src = r#"
        fiber def counter(): i64 {
            yield 1
            yield 2
            yield 3
            yield 4
            yield 5
        }

        def main(): i64 {
            let f = counter()
            let mut sum: i64 = 0
            let mut count: i64 = 0
            while let Some(x) = f.next() {
                sum = sum + x
                count = count + 1
                if count >= 2 {
                    f.cancel()
                }
            }
            return sum
        }
    "#;
    let result = run_returning_int(src).expect("fiber program should execute");
    assert_eq!(result, 3, "1 + 2 should sum to 3 (cancel stops the rest)");
}
