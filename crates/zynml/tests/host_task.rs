//! A host's own scheduler drives async tasks through `HostTask`: each step
//! advances a task without sleeping and says what it waits on next.

#![cfg(feature = "krio-async-backend")]

use std::time::{Duration, Instant};
use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::host_futures::{has_pending_timers, parked_count};
use zyntax_embed::{HostTask, HostTaskStep, ZyntaxRuntime};

fn compile(rt: &mut ZyntaxRuntime, src: &str) {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(src, "<host_task>")
        .expect("parse");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program).expect("compile");
}

/// Step `tasks` as a host would: sleep only when every one waits on a
/// timer, and only until the earliest. Returns each task's result word.
fn run(tasks: &mut [HostTask]) -> Vec<i64> {
    let mut results = vec![None; tasks.len()];
    while results.iter().any(Option::is_none) {
        let mut earliest: Option<Instant> = None;
        for (task, result) in tasks.iter_mut().zip(results.iter_mut()) {
            if result.is_some() {
                continue;
            }
            match task.step() {
                HostTaskStep::Ready(value) => *result = Some(value),
                HostTaskStep::Timer(deadline) => {
                    earliest = Some(earliest.map_or(deadline, |e| e.min(deadline)));
                }
                HostTaskStep::Parked => panic!("nothing here parks but a timer"),
                HostTaskStep::Yield => earliest = Some(Instant::now()),
            }
        }
        if let Some(deadline) = earliest {
            std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        }
    }
    results.into_iter().map(Option::unwrap).collect()
}

#[test]
fn a_step_reports_the_timer_and_never_sleeps() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def later(): i64 {
            await sleep(200)
            return 7
        }
        "#,
    );
    let mut task = HostTask::new(rt.call_async("later", &[]).expect("call"));
    let start = Instant::now();
    let step = task.step();
    assert!(start.elapsed() < Duration::from_millis(100), "{step:?}");
    let HostTaskStep::Timer(deadline) = step else {
        panic!("waits on its timer: {step:?}");
    };
    assert!(deadline > start + Duration::from_millis(150));
    // Stepped again before the timer is due, it still waits on it.
    assert_eq!(task.step(), HostTaskStep::Timer(deadline));
    std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
    assert_eq!(task.step(), HostTaskStep::Ready(7));
    assert_eq!(
        task.step(),
        HostTaskStep::Ready(7),
        "a finished task stays finished"
    );
}

#[test]
fn tasks_stepped_together_overlap_their_sleeps() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def a(): i64 {
            await sleep(40)
            await sleep(40)
            return 1
        }
        async def b(): i64 {
            await sleep(60)
            return 2
        }
        "#,
    );
    let mut tasks = [
        HostTask::new(rt.call_async("a", &[]).expect("a")),
        HostTask::new(rt.call_async("b", &[]).expect("b")),
    ];
    let start = Instant::now();
    assert_eq!(run(&mut tasks), [1, 2]);
    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(75), "{elapsed:?}");
    assert!(
        elapsed < Duration::from_millis(140),
        "overlapped: {elapsed:?}"
    );
}

#[test]
fn an_awaited_function_finishes_under_its_caller() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def inner(): i64 {
            await sleep(10)
            return 20
        }
        async def outer(): i64 {
            let x = await inner()
            await sleep(10)
            return x + 1
        }
        "#,
    );
    let mut tasks = [HostTask::new(rt.call_async("outer", &[]).expect("call"))];
    assert_eq!(run(&mut tasks), [21]);
}

#[test]
fn a_float_result_is_its_bits() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def half(): f64 {
            await sleep(5)
            return 2.5
        }
        "#,
    );
    let mut tasks = [HostTask::new(rt.call_async("half", &[]).expect("call"))];
    assert_eq!(f64::from_bits(run(&mut tasks)[0] as u64), 2.5);
}

#[test]
fn dropping_an_unfinished_task_tears_its_parking_down() {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    compile(
        &mut rt,
        r#"
        async def never(): i64 {
            await sleep(60000)
            return 1
        }
        "#,
    );
    let before = parked_count();
    let mut task = HostTask::new(rt.call_async("never", &[]).expect("call"));
    assert!(matches!(task.step(), HostTaskStep::Timer(_)));
    assert_eq!(parked_count(), before + 1);
    drop(task);
    assert_eq!(parked_count(), before);
    assert!(!has_pending_timers());
}
