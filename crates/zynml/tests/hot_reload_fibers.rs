//! Persistent fibers across hot reload.
//!
//! A fiber is the FSM substrate: its loop-carried variables are the
//! machine's state and each yield is a state boundary. These tests pin
//! the two halves of the reload story — a fiber suspended across the
//! edit keeps its state and continues on the edited transition, and a
//! fiber created after the edit starts on the edited code even though
//! its creator never recompiled.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

fn source(step: i64) -> String {
    // `ticker` is the machine: `acc` is its state, each yield a
    // transition, and it runs until the cap. `drive` pumps it to
    // completion and returns how many transitions that took — few
    // enough only if the edited step ran mid-flight.
    format!(
        r#"
fiber def ticker(): i64 {{
    let mut acc: i64 = 0
    while acc < 3000000 {{
        acc = acc + {step}
        yield acc
    }}
    return acc
}}

def drive(): i64 {{
    let f = ticker()
    let mut count: i64 = 0
    while let Some(x) = f.next() {{
        count = count + 1
    }}
    return count
}}

def first_tick(): i64 {{
    let f = ticker()
    match f.next() {{
        case Some(v) {{ return v }}
        case None() {{ return -1 }}
    }}
}}
"#
    )
}

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    TieredRuntime::new(config).expect("runtime should start")
}

fn parse(src: &str) -> zyntax_embed::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    grammar
        .parse_with_filename(src, "<hot_reload_fibers>")
        .expect("source should parse")
}

/// One fiber lives across the whole run, pumped to completion from a
/// call the host entered once. The edit lands while the fiber is
/// suspended between yields; from then on each transition advances by
/// the edited step, so the cap is reached in far fewer transitions
/// than the original step could ever manage.
#[test]
fn a_suspended_fiber_keeps_its_state_and_picks_up_the_edit() {
    const CAP: i64 = 3_000_000;

    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1)))
        .expect("v1 should compile");

    let entry = rt
        .function_pointer("drive")
        .expect("drive should have an entry pointer");
    // SAFETY: `drive` was compiled with signature () -> i64, and the
    // code stays mapped for the life of the runtime.
    let drive: extern "C" fn() -> i64 = unsafe { std::mem::transmute(entry) };
    let worker = std::thread::spawn(move || drive());

    std::thread::sleep(std::time::Duration::from_millis(150));
    let report = rt
        .reload_typed_program(parse(&source(1000)))
        .expect("reload should succeed");
    let count = worker.join().expect("drive should complete");

    assert!(
        report.reloaded.contains(&"ticker".to_string()),
        "{report:?}"
    );
    assert!(
        !report.reloaded.contains(&"drive".to_string()),
        "the driver did not change: {report:?}"
    );
    assert!(
        count < CAP,
        "the fiber took every transition at the old step; it never migrated (count = {count})"
    );
    assert!(
        count > CAP / 1000,
        "the fiber cannot have taken fewer transitions than the edited step allows (count = {count})"
    );
}

/// The creator (`first_tick`) is untouched by the edit, so it never
/// recompiles — yet the fiber it creates afterwards must start on the
/// edited body. The address it hands `fiber ticker()` has to be read
/// at creation time, not baked when the creator was compiled.
#[test]
fn a_fiber_created_after_the_edit_runs_the_edited_code_from_the_start() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1)))
        .expect("v1 should compile");

    let tick = |rt: &TieredRuntime| -> i64 {
        match rt.call_raw("first_tick", &[]).expect("first_tick") {
            ZyntaxValue::Int(v) => v,
            other => panic!("expected an int, got {other:?}"),
        }
    };
    assert_eq!(tick(&rt), 1);

    let report = rt
        .reload_typed_program(parse(&source(7)))
        .expect("reload should succeed");
    assert!(
        report.reloaded.contains(&"ticker".to_string()),
        "{report:?}"
    );
    assert!(
        !report.reloaded.contains(&"first_tick".to_string()),
        "the creator did not change and must not reload: {report:?}"
    );

    assert_eq!(
        tick(&rt),
        7,
        "a fiber created after the edit starts on the edited step"
    );
}
