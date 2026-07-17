//! Cooperative executor: two async tasks each doing `await sleep` overlap
//! instead of running one-after-the-other. The native `set_timeout` now
//! records a timer and returns (parks) rather than blocking inline, and
//! `drive_tasks` drains all tasks' timers in deadline order — so two tasks
//! that each sleep ~50ms finish in ~50ms wall-clock, not ~100ms.

#![cfg(feature = "krio-async-backend")]

use std::time::Instant;
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{drive_tasks, ZyntaxRuntime, ZyntaxValue};

fn compile(rt: &mut ZyntaxRuntime, src: &str) {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<coop>").expect("parse");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program).expect("compile");
}

#[test]
fn two_tasks_overlap_their_sleeps() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def taskA(): i64 {
            await sleep(50)
            return 1
        }
        async def taskB(): i64 {
            await sleep(50)
            return 2
        }
        "#,
    );

    let a = rt.call_async("taskA", &[]).expect("call taskA");
    let b = rt.call_async("taskB", &[]).expect("call taskB");

    let start = Instant::now();
    let results = drive_tasks(&[a, b]);
    let elapsed = start.elapsed();

    assert_eq!(
        results[0].as_ref().ok().and_then(ZyntaxValue::as_i64),
        Some(1),
        "taskA -> 1"
    );
    assert_eq!(
        results[1].as_ref().ok().and_then(ZyntaxValue::as_i64),
        Some(2),
        "taskB -> 2"
    );
    // Concurrency proof: both slept ~50ms, but overlapped — total well under
    // the ~100ms a sequential (blocking) executor would take.
    assert!(
        elapsed.as_millis() >= 45,
        "both tasks actually waited out their timers; got {elapsed:?}"
    );
    assert!(
        elapsed.as_millis() < 90,
        "tasks must overlap (~50ms total), not run sequentially (~100ms); got {elapsed:?}"
    );
}

/// Multiple awaits per task, staggered deadlines — resolution order
/// interleaves the two tasks (A@10, B@15, A@20) rather than draining A
/// fully first. Correctness check: both still produce their values.
#[test]
fn interleaved_multi_await_tasks_complete() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def taskA(): i64 {
            await sleep(10)
            await sleep(10)
            return 11
        }
        async def taskB(): i64 {
            await sleep(15)
            return 22
        }
        "#,
    );

    let a = rt.call_async("taskA", &[]).expect("call taskA");
    let b = rt.call_async("taskB", &[]).expect("call taskB");
    let results = drive_tasks(&[a, b]);

    assert_eq!(
        results[0].as_ref().ok().and_then(ZyntaxValue::as_i64),
        Some(11)
    );
    assert_eq!(
        results[1].as_ref().ok().and_then(ZyntaxValue::as_i64),
        Some(22)
    );
}
