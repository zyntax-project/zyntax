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
fn fiber_handle_survives_a_resumable_perform() {
    // A fiber created before a resumable perform and iterated after it: the
    // fiber handle is a value live across the perform. It must be saved BEFORE
    // the suspend (emit_save_load placed the capture save AFTER the perform,
    // stranding it on the resume side — the handle came back NULL and
    // `f.next()` dereferenced it). k(1) -> x=1; sum 10+20=30; 1+30=31.
    assert_eq!(
        run_program(
            r#"
            effect E { def op(): i64 }
            handler H for E { def op(k: Resume<i64>): i64 { return k(1) } }
            fiber def gen() { yield 10 yield 20 }
            @effect(E)
            async def work(): i64 {
                let f = gen()
                let x = op()
                var s: i64 = 0
                while let Some(y) = f.next() { s = s + y }
                return x + s
            }
            async def run(): i64 { var v: i64 = 0 with H { v = await work() } return v }
            "#
        ),
        Some(31),
        "a fiber handle live across a resumable perform must survive the suspend"
    );
}

#[test]
fn call_result_survives_a_resumable_perform() {
    // A CALL result (not a rematerialisable constant) live across a resumable
    // perform. `z = helper(3)` needs a real capture slot; it is saved before
    // the perform and reloaded after. The inliner then inlines `helper`,
    // rewriting `call_result -> return_value` across the function — but its
    // hand-rolled operand substitution skipped `AsyncSaveSlot`, so the save
    // kept pointing at the orphaned call-result id. Codegen drops a save of an
    // unmapped value, so z reloaded 0 (returned 1 instead of 1 + 6 = 7). The
    // fix routes operand substitution through the canonical `replace_uses`.
    assert_eq!(
        run_program(
            r#"
            effect E { def op(): i64 }
            handler H for E { def op(k: Resume<i64>): i64 { return k(1) } }
            def helper(a: i64): i64 { return a + a }
            @effect(E)
            async def work(): i64 {
                let z = helper(3)
                let x = op()
                return x + z
            }
            async def run(): i64 { var v: i64 = 0 with H { v = await work() } return v }
            "#
        ),
        Some(7),
        "a call result live across a resumable perform must survive inlining"
    );
}

#[test]
fn perform_resume_await_then_fiber_next() {
    // The full composition in one flow: perform -> resume -> await -> fiber.next().
    // The performer's `await` resolves inside the perform's resume-drive loop,
    // which polls the performer to Ready (running the fiber loop to completion
    // and freeing the fiber at scope exit). The resume loop then re-polled the
    // SAME finished SM, re-entering the post-await region and re-resuming the
    // freed fiber -> null `_trampoline_state` -> non-unwinding abort. The drive
    // now records the SM completion by pointer; the resume loop takes it and
    // returns instead of re-polling. k(1) -> x=1; 10+20=30; 1+30=31.
    assert_eq!(
        run_program(
            r#"
            effect E { def op(): i64 }
            handler H for E { def op(k: Resume<i64>): i64 { return k(1) } }
            fiber def gen() { yield 10 yield 20 }
            @effect(E)
            async def work(): i64 {
                let f = gen()
                let x = op()
                await sleep(10)
                var s: i64 = 0
                while let Some(y) = f.next() { s = s + y }
                return x + s
            }
            async def run(): i64 { var v: i64 = 0 with H { v = await work() } return v }
            "#
        ),
        Some(31),
        "perform -> resume -> await -> fiber.next() must not re-poll the finished SM"
    );
}

#[test]
fn mixed_async_and_sync_handlers_for_one_op() {
    // One operation handled by a sync handler (H1) and an async handler (H2),
    // selected at runtime by which `with` block is in scope. The perform site
    // picks the drive-vs-call convention at runtime (async_mask + finish_op).
    const SRC: &str = r#"
        effect E { def op(): i64 }
        handler H1 for E { def op(k: Resume<i64>): i64 { return k(1) } }
        handler H2 for E { async def op(k: Resume<i64>): i64 { await sleep(10) return k(2) } }
        @effect(E)
        async def work(): i64 { let x = op() return x + 100 }
        async def runSync(): i64 { var v: i64 = 0 with H1 { v = await work() } return v }
        async def runAsync(): i64 { var v: i64 = 0 with H2 { v = await work() } return v }
    "#;
    let run_entry = |entry: &str| -> Option<i64> {
        let mut rt = ZyntaxRuntime::new().expect("rt");
        let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
        let program = grammar.parse_with_filename(SRC, "<mixed>").expect("parse");
        rt.config_mut().builtins.insert(
            "sleep".to_string(),
            "__zyntax_async_set_timeout".to_string(),
        );
        rt.compile_typed_program(program).expect("compile");
        let p = rt.call_async(entry, &[]).expect("call");
        drive_tasks(&[p])[0]
            .as_ref()
            .ok()
            .and_then(ZyntaxValue::as_i64)
    };
    assert_eq!(
        run_entry("runSync"),
        Some(101),
        "sync handler: k(1) -> 1 -> 101"
    );
    assert_eq!(
        run_entry("runAsync"),
        Some(102),
        "async handler: await, then k(2) -> 2 -> 102"
    );
}

#[test]
fn async_handler_tasks_interleave() {
    // Two tasks each perform an effect whose async handler awaits ~50ms.
    // They must OVERLAP (both handlers park, the executor drives both timers
    // together) — well under the ~100ms a blocking drive would take.
    use std::time::Instant;
    let mut rt = ZyntaxRuntime::new().expect("rt");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            effect E { def op(): i64 }
            handler H for E {
                async def op(k: Resume<i64>): i64 { await sleep(50) return k(7) }
            }
            @effect(E)
            async def work(): i64 { return op() }
            async def taskA(): i64 { var v: i64 = 0 with H { v = await work() } return v }
            async def taskB(): i64 { var v: i64 = 0 with H { v = await work() } return v }
            "#,
            "<interleave>",
        )
        .expect("parse");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program).expect("compile");
    let a = rt.call_async("taskA", &[]).expect("taskA");
    let b = rt.call_async("taskB", &[]).expect("taskB");
    let t = Instant::now();
    let r = drive_tasks(&[a, b]);
    let ms = t.elapsed().as_millis();
    assert_eq!(r[0].as_ref().ok().and_then(ZyntaxValue::as_i64), Some(7));
    assert_eq!(r[1].as_ref().ok().and_then(ZyntaxValue::as_i64), Some(7));
    assert!(
        ms >= 45,
        "both handlers actually waited their ~50ms timer; got {ms}ms"
    );
    assert!(
        ms < 90,
        "the two async-handler tasks must INTERLEAVE (~50ms), not run \
         sequentially (~100ms); got {ms}ms"
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
