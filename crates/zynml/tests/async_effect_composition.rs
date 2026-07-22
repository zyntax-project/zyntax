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
