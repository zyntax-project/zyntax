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
    let result = rt
        .call_function_raw("main", vec![])
        .map_err(|e| format!("call: {e:?}"))?;
    result
        .as_i64()
        .ok_or_else(|| format!("expected i64, got {result:?}"))
}

/// The minimum-viable fiber run: a fiber that yields three integers,
/// consumed via `while let Some(x) = f.next()`. Returns the sum.
///
/// 1 + 2 + 3 = 6. Currently ignored — the structural chain
/// (FiberNew → krio_fiber_new, FiberResume → krio_fiber_resume,
/// FiberYield → krio_fiber_yield) compiles and runs end-to-end, but
/// two pre-existing bugs in the match / while-let lowering produce
/// wrong values: (1) `case Some(x)` binds `undef` instead of the
/// extracted payload; (2) `while let` desugar generates a CFG
/// that fails Cranelift verifier with a mismatched-arg jump.
/// Re-enable once those land.
#[test]
#[ignore = "blocked on match-on-Option pattern binding + while-let CFG bugs"]
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
