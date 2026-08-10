//! The async state machine is released when its task is finished.
//!
//! The entry function `malloc`s the slot array and nothing used to free
//! it, so every spawn leaked. It is released when the promise drops,
//! but only once the task is finished and no async table still names
//! the region: a parked timer, a latched completion, or a
//! handler/performer pairing all mean something can still poll it.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

fn rss_kb() -> i64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(0)
}

/// Many completed tasks must not accumulate. The body has enough locals
/// across its await to make the slot array worth measuring.
#[test]
fn completed_tasks_release_their_state_machines() {
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    let program = g
        .parse_with_filename(
            r#"
async def work(): i64 {
    let a: i64 = 1
    let b: i64 = 2
    let c: i64 = 3
    let d: i64 = 4
    let e: i64 = 5
    let f: i64 = 6
    let g2: i64 = 7
    let h: i64 = 8
    await sleep(0)
    return a + b + c + d + e + f + g2 + h
}
"#,
            "sm_leak.zyn",
        )
        .expect("parse");

    let mut rt = ZyntaxRuntime::new().expect("rt");
    rt.config_mut().builtins.insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt.compile_typed_program(program).expect("compile");

    for _ in 0..1_000 {
        let p = rt.call_async("work", &[]).expect("spawn");
        assert_eq!(p.await_raw().expect("resolve").as_i64(), Some(36));
    }
    let base = rss_kb();
    for _ in 0..50_000 {
        let p = rt.call_async("work", &[]).expect("spawn");
        assert_eq!(p.await_raw().expect("resolve").as_i64(), Some(36));
    }
    let grew = rss_kb() - base;
    eprintln!("RSS delta over 50k completed tasks: {grew}kB");
    assert!(
        grew < 2_000,
        "50k completed tasks grew RSS by {grew}kB; state machines are not being released"
    );
}
