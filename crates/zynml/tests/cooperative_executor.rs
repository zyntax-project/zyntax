//! Cooperative executor: two async tasks each doing `await sleep` overlap
//! instead of running one-after-the-other. The native `set_timeout` now
//! records a timer and returns (parks) rather than blocking inline, and
//! `drive_tasks` drains all tasks' timers in deadline order — so two tasks
//! that each sleep ~50ms finish in ~50ms wall-clock, not ~100ms.

#![cfg(feature = "krio-async-backend")]

use std::time::{Duration, Instant};
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::host_futures::has_pending_timers;
use zyntax_embed::{drive_tasks, PromiseRace, ZyntaxRuntime, ZyntaxValue};

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

/// Promise.race resolves with the first task and cancels the losers,
/// tearing down their parked timers so the executor stops driving them.
#[test]
fn race_resolves_first_and_cancels_loser() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def fast(): i64 {
            await sleep(20)
            return 1
        }
        async def slow(): i64 {
            await sleep(500)
            return 2
        }
        "#,
    );

    let a = rt.call_async("fast", &[]).expect("call fast");
    let b = rt.call_async("slow", &[]).expect("call slow");
    let mut race = PromiseRace::new(vec![a, b]);

    let start = Instant::now();
    let (idx, val) = race.await_first().expect("race resolves");
    let elapsed = start.elapsed();

    assert_eq!(idx, 0, "fast wins");
    assert_eq!(val.as_i64(), Some(1));
    // Returned when `fast` finished (~20ms), not after `slow`'s 500ms.
    assert!(
        elapsed.as_millis() < 300,
        "race returns on the first winner, not the slow loser; got {elapsed:?}"
    );
    // The loser was cancelled and its 500ms timer deregistered.
    assert!(
        !has_pending_timers(),
        "the cancelled loser's timer must be torn down"
    );
}

/// await_with_timeout cancels a task that overruns and deregisters its
/// parked timer, returning at ~the timeout rather than the full sleep.
#[test]
fn timeout_cancels_overrunning_task() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def slow(): i64 {
            await sleep(500)
            return 9
        }
        "#,
    );

    let p = rt.call_async("slow", &[]).expect("call slow");
    let start = Instant::now();
    let r = p.await_with_timeout(Duration::from_millis(50));
    let elapsed = start.elapsed();

    assert!(r.is_err(), "should time out before the 500ms sleep");
    assert!(
        elapsed.as_millis() >= 45 && elapsed.as_millis() < 300,
        "returns at ~the timeout, not after the full sleep; got {elapsed:?}"
    );
    assert!(
        !has_pending_timers(),
        "the timed-out task's timer must be deregistered"
    );
}

/// Cancelling (here via timeout) a task that owns a fiber frees that
/// fiber — a cancelled task never runs its normal scope-exit FiberDrop,
/// so the executor tears it down. Verifies the fiber is gone from the
/// per-task registry and nothing double-frees / crashes.
#[test]
fn timeout_frees_a_fiber_owning_tasks_fiber() {
    use zyntax_embed::host_futures::task_fiber_count;

    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
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
                await sleep(500)
                sum = sum + x
            }
            return sum
        }
        "#,
    );

    let p = rt.call_async("main", &[]).expect("call main");
    // Times out after the fiber is created + first step, well before the
    // 500ms sleep — so the task is cancelled mid-drive, holding the fiber.
    let r = p.await_with_timeout(Duration::from_millis(50));
    assert!(r.is_err(), "should time out mid-drive");

    // The single task drove as index 0; its fiber was freed on cancel.
    assert_eq!(
        task_fiber_count(0),
        0,
        "the cancelled task's fiber must be freed, not leaked"
    );
    assert!(!has_pending_timers(), "timed-out task's timer deregistered");
}
