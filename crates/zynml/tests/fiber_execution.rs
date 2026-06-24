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

/// `Fiber.abort(err)` from inside the fiber body — Wren-style.
/// The body yields a value, then aborts with an error payload;
/// the caller's `f.next()` returns `None` (the iteration ends)
/// instead of seeing the post-abort yields. Implemented as a
/// deferred abort: the body suspends at the abort point, the
/// resume side's `encode_step` swaps `Yielded` for `Errored`.
#[test]
fn fiber_abort_stops_iteration() {
    let src = r#"
        fiber def maybe_error(): i64 {
            yield 10
            Fiber.abort(99)
            yield 20
            yield 30
        }

        def main(): i64 {
            let f = maybe_error()
            let mut sum: i64 = 0
            while let Some(x) = f.next() {
                sum = sum + x
            }
            return sum
        }
    "#;
    let result = run_returning_int(src).expect("fiber program should execute");
    assert_eq!(
        result, 10,
        "yield 10 fires; Fiber.abort(99) stops the loop before yield 20 / yield 30"
    );
}

/// `f.error()` returns `Option<FiberError<String>>` reflecting
/// whether the fiber aborted. Test encodes the variant into the
/// return value:
///   * 100 = `Some(_)` (fiber aborted)
///   * 200 = `None` (fiber completed)
/// Avoids printing the FiberError struct directly — Display
/// dispatch on pattern-bound struct payloads needs type-resolver
/// work that's pending (see project memory).
#[test]
fn fiber_error_detects_abort() {
    let src = r#"
        fiber def aborts(): i64 {
            yield 1
            Fiber.abort("oops")
            yield 99
        }

        def main(): i64 {
            let f = aborts()
            while let Some(x) = f.next() {
            }
            match f.error() {
                case Some(_) {
                    return 100
                }
                case None() {
                    return 200
                }
            }
        }
    "#;
    let result = run_returning_int(src).expect("fiber program should execute");
    assert_eq!(
        result, 100,
        "f.error() should be Some after Fiber.abort fired"
    );
}

/// Companion test: a fiber that completes normally has
/// `f.error() == None`. Verifies the slot is empty when no abort
/// fired (and that the krio-side `take_error` doesn't leak state
/// across fiber instances by always returning `Some`).
#[test]
fn fiber_error_none_on_normal_completion() {
    let src = r#"
        fiber def completes(): i64 {
            yield 1
            yield 2
        }

        def main(): i64 {
            let f = completes()
            while let Some(x) = f.next() {
            }
            match f.error() {
                case Some(_) {
                    return 100
                }
                case None() {
                    return 200
                }
            }
        }
    "#;
    let result = run_returning_int(src).expect("fiber program should execute");
    assert_eq!(
        result, 200,
        "f.error() should be None after normal completion"
    );
}
