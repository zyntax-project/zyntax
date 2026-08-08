//! Hot reload against async tasks. An await frame is a state machine
//! like a fiber, and its poll pointer is captured when the task is
//! spawned — so a task suspended across a reload completes on the code
//! it started with (edits renumber the machine's states; transferring
//! mid-flight is never sound), while a task spawned after the reload
//! runs the edited code from its first poll.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime};

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("runtime should start");
    rt.builtin_aliases_mut().insert(
        "sleep".to_string(),
        "__zyntax_async_set_timeout".to_string(),
    );
    rt
}

fn parse(src: &str) -> zyntax_embed::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    grammar
        .parse_with_filename(src, "<hot_reload_async>")
        .expect("source should parse")
}

fn source(ret: i64) -> String {
    format!(
        r#"
async def work(): i64 {{
    await sleep(30)
    return {ret}
}}
"#
    )
}

/// The task suspended on its timer when the edit landed must resolve
/// with the value its generation returns; a task spawned afterwards
/// resolves with the edited value.
#[test]
fn a_suspended_task_completes_on_its_generation() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1)))
        .expect("v1 should compile");

    // Spawn: runs to the first await and parks on the timer. Nothing
    // drives the task while we reload, so the interleaving is fixed.
    let suspended = rt.call_async("work", &[]).expect("call_async");

    let report = rt
        .reload_typed_program(parse(&source(2)))
        .expect("reload should succeed");
    assert!(
        !report.reloaded.is_empty() || !report.added.is_empty(),
        "the edit must have produced new code: {report:?}"
    );

    let old = suspended.await_raw().expect("suspended task resolves");
    assert_eq!(
        old.as_i64(),
        Some(1),
        "a task in flight completes on the code it started with"
    );

    let fresh = rt
        .call_async("work", &[])
        .expect("call_async")
        .await_raw()
        .expect("fresh task resolves");
    assert_eq!(
        fresh.as_i64(),
        Some(2),
        "a task spawned after the edit runs the edited code"
    );
}
