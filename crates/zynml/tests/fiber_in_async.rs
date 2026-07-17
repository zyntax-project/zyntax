//! Composition C — driving a fiber from inside an async body.
//!
//! Coverage:
//! * fiber-in-async WITHOUT await — works.
//! * plain `await` inside a `while` loop — works (regression test for
//!   the save-before-suspend fix in krio_adapter's await lowering; the
//!   bridge call is now emitted AFTER the slot saves, so the native
//!   synchronous-resolve path doesn't clobber loop-carried state).
//! * fiber-drive + `await` in the same loop — still open (the fiber
//!   handle must survive the await suspension AND the fiber-step
//!   decode; that's the cooperative fiber-in-async work in Phase 5).

#![cfg(feature = "krio-async-backend")]

use std::time::Instant;
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

/// Async body iterates a fiber to completion — no await. The fiber
/// handle lives across the while-let loop's state boundary, which is
/// exactly what the legacy captures analysis choked on.
#[test]
fn async_drives_fiber_no_await() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            fiber def gen(): i64 {
                yield 1
                yield 2
                yield 3
            }

            async def main(): i64 {
                let f = gen()
                let mut sum: i64 = 0
                while let Some(x) = f.next() {
                    sum = sum + x
                }
                return sum
            }
            "#,
            "<fiber_in_async>",
        )
        .expect("parse");

    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program)
        .expect("fiber-in-async should compile on the krio path");

    let promise = rt.call_async("main", &[]).expect("call_async");
    let result = promise.await_raw().expect("promise resolves");
    assert_eq!(result.as_i64(), Some(6), "1+2+3 = 6");
}

/// Await INSIDE the fiber-driving loop — the full composition. Two fixes
/// make it work: (1) captures-lift SSA reconstruction so the fiber handle
/// is saved/restored across the await (`repair_ssa_for_reloads`), and
/// (2) a synchronous-completion latch in the scheduler so a task that
/// runs to completion inside one recursive `poll()` (the native bridge
/// resolves inline) is seen as Ready instead of being re-polled — the
/// re-poll used to re-enter the state machine and resume the fiber the
/// loop-exit path had already freed.
#[test]
fn async_drives_fiber_with_await_in_loop() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            fiber def gen(): i64 {
                yield 10
                yield 20
                yield 30
            }

            async def main(): i64 {
                let f = gen()
                let mut sum: i64 = 0
                while let Some(x) = f.next() {
                    await sleep(20)
                    sum = sum + x
                }
                return sum
            }
            "#,
            "<fiber_in_async_await>",
        )
        .expect("parse");

    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program)
        .expect("await-in-fiber-loop should compile on the krio path");

    let promise = rt.call_async("main", &[]).expect("call_async");
    let start = Instant::now();
    let result = promise.await_raw().expect("promise resolves");
    let elapsed = start.elapsed();

    assert_eq!(result.as_i64(), Some(60), "10+20+30 = 60");
    // Three awaits of 20ms each, cooperatively parked.
    assert!(
        elapsed.as_millis() >= 50,
        "expected ≥ 50ms from 3× parked sleeps; got {:?}",
        elapsed
    );
}

/// `await` inside a plain counter-driven while loop, NO fiber.
/// Regression test for the save-before-suspend bug: the krio await
/// lowering used to emit the yielding bridge call
/// (`__zyntax_async_set_timeout`) BEFORE the `AsyncSaveSlot`s. On the
/// native runtime that bridge resolves synchronously and recursively,
/// so the trailing saves clobbered the loop-carried state back to its
/// pre-suspension values every iteration and the loop spun forever.
/// Emitting the saves before the bridge fixes it. Sums 0+1+2 = 3.
#[test]
fn async_await_in_plain_loop() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            async def main(): i64 {
                let mut i: i64 = 0
                let mut sum: i64 = 0
                while i < 3 {
                    await sleep(20)
                    sum = sum + i
                    i = i + 1
                }
                return sum
            }
            "#,
            "<iso_await_loop>",
        )
        .expect("parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program).expect("compile");
    let promise = rt.call_async("main", &[]).expect("call_async");
    let result = promise.await_raw().expect("resolves");
    assert_eq!(result.as_i64(), Some(3), "0+1+2 = 3");
}
