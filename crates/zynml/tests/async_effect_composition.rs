//! An `async def` that performs a RESUMABLE algebraic effect now composes:
//! the perform becomes a real suspension/resume state in the async state
//! machine, the handler's `k(v)` resumes the continuation, and an `await`
//! reached after the resume is driven cooperatively to completion.
//!
//! Regression guard for the async + resumable-perform composition fix. The
//! two-pass krio lowering previously double-lowered such a function (the
//! async pass produced a promise entry whose `is_async` flag was cleared but
//! whose signature kept `@effect`; the resumable-effect pass then re-lowered
//! it and clobbered the SM with an empty-yield layout), so the handler's
//! resume landed on a non-existent state and spun. Now the async pass owns
//! these fns end to end (perform suspension + `Resume<T>` upgrade) and the
//! effect pass skips them (it gates on a raw `PerformEffect` in the body).

#![cfg(feature = "krio-async-backend")]

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{drive_tasks, ZyntaxRuntime, ZyntaxValue};

/// Compile `src`, drive async `run` to completion, return its i64 result.
fn run_program(src: &str) -> Option<i64> {
    let mut rt = ZyntaxRuntime::new().expect("rt");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(src, "<async_effect_composition>")
        .expect("parse");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program).expect("compile");
    let p = rt.call_async("run", &[]).expect("call run");
    drive_tasks(&[p])[0]
        .as_ref()
        .ok()
        .and_then(ZyntaxValue::as_i64)
}

const HEADER: &str = r#"
    effect E { def op(): i64 }
    handler H for E { def op(k: Resume<i64>): i64 { return k(1) } }
"#;

/// `run` opens `with H` and awaits `work`, which performs `op` (resumed to 1).
fn program(work_body: &str) -> String {
    format!(
        r#"{HEADER}
        @effect(E)
        async def work(): i64 {{ {work_body} }}
        async def run(): i64 {{
            let mut v: i64 = 0
            with H {{ v = await work() }}
            return v
        }}"#
    )
}

#[test]
fn async_fn_performs_resumable_effect() {
    // perform only (no await): handler resumes k(1) -> x = 1 -> 101.
    assert_eq!(
        run_program(&program("let x = op()\n return x + 100")),
        Some(101),
        "async perform-only: handler resume must reach the continuation"
    );
}

#[test]
fn perform_then_await_resumes_and_completes() {
    // perform, then await AFTER the resume: the continuation parks on the
    // timer and is driven cooperatively to Ready (the parking-handler path).
    assert_eq!(
        run_program(&program("let x = op()\n await sleep(10)\n return x + 100")),
        Some(101),
        "await after a resumable resume must complete cooperatively"
    );
}

#[test]
fn await_then_perform_carries_value() {
    // await first, then perform: the resumed value (1) must survive across
    // the earlier suspension (previously returned 100 — value was lost).
    assert_eq!(
        run_program(&program("await sleep(10)\n let x = op()\n return x + 100")),
        Some(101),
        "resumed value must survive an await that precedes the perform"
    );
}

#[test]
fn perform_result_survives_conditional_await() {
    // perform, then a CONDITIONAL await, then use the perform result at the
    // branch merge. The result is produced at the perform's resume block, so
    // the merge phi must reload it on the await path; a naive dominance check
    // (suspend block doesn't dominate the resume region) previously pruned
    // that phi → Cranelift SSA verification failure.
    assert_eq!(
        run_program(&program(
            "let x = op()\n if x > 0 {\n await sleep(10)\n }\n return x + 100"
        )),
        Some(101),
        "perform result must survive a conditional await used at the merge"
    );
}

#[test]
fn await_result_survives_conditional_await() {
    // Same shape with a pure-async value (no effect): an await result used
    // after a later conditional await. Regression guard for the general
    // captures-lift SSA reconstruction, not just the effect composition.
    let src = r#"
        async def work(): i64 {
            let y = await sleep(5)
            if y >= 0 {
                await sleep(10)
            }
            return y + 100
        }
        async def run(): i64 { return await work() }
    "#;
    assert_eq!(
        run_program(src),
        Some(100),
        "await result must survive a later conditional await used at the merge"
    );
}

#[test]
fn async_handler_op_no_await_resumes() {
    // An async handler op with no await runs `k(7)` inline; the perform site
    // drives its promise and returns the resumed value.
    assert_eq!(
        run_program(
            r#"
            effect E { def op(): i64 }
            handler H for E {
                async def op(k: Resume<i64>): i64 { return k(7) }
            }
            @effect(E)
            async def work(): i64 { return op() }
            async def run(): i64 {
                var v: i64 = 0
                with H { v = await work() }
                return v
            }
            "#
        ),
        Some(7),
        "async handler op (no await) must drive to k(7)"
    );
}

#[test]
fn async_handler_await_parks_and_balances() {
    // The handler's await actually PARKS (the ~30ms elapses; it doesn't spin)
    // and the handler-stack segments return to baseline afterwards (a
    // with-block the handler opens across its await stays isolated).
    use std::time::Instant;
    let src = r#"
        effect E { def op(): i64 }
        handler H for E {
            async def op(k: Resume<i64>): i64 { await sleep(30) return k(7) }
        }
        @effect(E)
        async def work(): i64 { return op() }
        async def run(): i64 { var v: i64 = 0 with H { v = await work() } return v }
    "#;
    let t = Instant::now();
    assert_eq!(run_program(src), Some(7));
    assert!(
        t.elapsed().as_millis() >= 25,
        "handler must actually park on its timer (~30ms), not spin"
    );
    assert_eq!(
        zyntax_embed::host_futures::handler_stack_depth(),
        0,
        "handler stack must return to baseline"
    );
    assert_eq!(
        zyntax_embed::host_futures::task_handler_segment_count(),
        0,
        "no handler-stack segment may leak after the drive"
    );
}

#[test]
fn async_handler_multi_shot_simple_performer() {
    // The handler resumes twice; a performer with no droppable scope state
    // re-runs its post-perform code per resume. k(1)->101, k(2)->102, sum=203.
    // (Multi-shot where the performer owns droppable state — e.g. a fiber —
    // is a pre-existing limitation shared by sync and async handlers alike.)
    assert_eq!(
        run_program(
            r#"
            effect E { def op(): i64 }
            handler H for E {
                async def op(k: Resume<i64>): i64 { let a = k(1) let b = k(2) return a + b }
            }
            @effect(E)
            async def work(): i64 { let x = op() return x + 100 }
            async def run(): i64 { var v: i64 = 0 with H { v = await work() } return v }
            "#
        ),
        Some(203),
        "multi-shot async handler re-runs the performer per resume"
    );
}

#[test]
fn async_handler_does_not_corrupt_a_concurrent_task() {
    // An async-handler task runs alongside a normal task. The self-contained
    // drive blocks the handler task for its own await but must not corrupt the
    // other task (which completes independently).
    let mut rt = ZyntaxRuntime::new().expect("rt");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            effect E { def op(): i64 }
            handler H for E {
                async def op(k: Resume<i64>): i64 { await sleep(10) return k(7) }
            }
            @effect(E)
            async def work(): i64 { return op() }
            async def taskA(): i64 { var v: i64 = 0 with H { v = await work() } return v }
            async def taskB(): i64 { await sleep(5) return 99 }
            "#,
            "<concurrent>",
        )
        .expect("parse");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program).expect("compile");
    let a = rt.call_async("taskA", &[]).expect("taskA");
    let b = rt.call_async("taskB", &[]).expect("taskB");
    let r = drive_tasks(&[a, b]);
    assert_eq!(r[0].as_ref().ok().and_then(ZyntaxValue::as_i64), Some(7));
    assert_eq!(r[1].as_ref().ok().and_then(ZyntaxValue::as_i64), Some(99));
}

#[test]
fn async_handler_op_awaits_then_resumes() {
    // The async handler op AWAITS, then resumes the performer. The perform
    // site drives the handler's own await chain (self-contained) to k(7).
    assert_eq!(
        run_program(
            r#"
            effect E { def op(): i64 }
            handler H for E {
                async def op(k: Resume<i64>): i64 {
                    await sleep(10)
                    return k(7)
                }
            }
            @effect(E)
            async def work(): i64 { return op() }
            async def run(): i64 {
                var v: i64 = 0
                with H { v = await work() }
                return v
            }
            "#
        ),
        Some(7),
        "async handler op must await, then resume the performer to k(7)"
    );
}
