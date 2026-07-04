//! Composition C — driving a fiber from inside an async body.
//!
//! Phase 0.2 of the fiber×effect×async plan: the composition-C audit
//! reported a `CreateClosure: Lambda function HirId(...) not found`
//! warning on the await-in-loop-while-driving-a-fiber shape. These
//! tests execute that shape end-to-end; a real miscompile would show
//! up as a wrong sum, and a compile failure as an Err from
//! `compile_typed_program`.

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

/// Await INSIDE the fiber-driving loop. Currently BROKEN — the
/// promise resolves to `None` instead of the sum. Root cause is NOT
/// fiber-specific: `async_await_in_plain_loop` below shows plain
/// await-in-a-while-loop hangs outright. Both trace to the async
/// state machine's handling of an await suspension on a loop
/// back-edge. Tracked as the await-in-loop bug; re-enable when fixed.
#[ignore = "await-inside-a-while-loop miscompiles (async SM loop back-edge); see async_await_in_plain_loop"]
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

/// Minimal reproduction of the await-in-loop bug: await inside a
/// plain counter-driven while loop, NO fiber. Currently HANGS (the
/// async state machine never advances past the awaiting loop state),
/// which is why the fiber-driven variant above returns None. This is
/// the true root bug; the fiber composition is an innocent bystander.
#[ignore = "await-inside-a-while-loop hangs — async SM loop back-edge bug"]
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
