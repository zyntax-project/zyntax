//! Hot reload across the effect × fiber composition.
//!
//! An effectful fiber is the observing FSM: each transition performs an
//! effect to read an event, folds it into loop-carried state, and
//! yields. The handler is the machine's event source, installed with
//! `with` around the pump loop. A reload must preserve all three legs
//! at once — the fiber's state, its handler-stack segment, and the
//! dispatch path — while the edited transition takes over.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

/// `machine` observes events via `next_event()` (handler-provided 3),
/// folds them into `state` with a reloadable transition, and yields
/// until the cap. `drive` pumps to completion inside `with Feed` and
/// returns the transition count.
fn source(weight: i64) -> String {
    format!(
        r#"
effect Event {{
    def next_event(): i64
}}

handler Feed for Event {{
    def next_event(): i64 {{ return 3 }}
}}

@effect(Event)
fiber def machine(): i64 {{
    let mut state: i64 = 0
    while state < 3000000 {{
        let e = next_event()
        state = state + e * {weight}
        yield state
    }}
    return state
}}

def drive(): i64 {{
    let mut count: i64 = 0
    with Feed {{
        let f = machine()
        while let Some(x) = f.next() {{
            count = count + 1
        }}
    }}
    return count
}}

def first_step(): i64 {{
    let mut out: i64 = 0
    with Feed {{
        let f = probe()
        while let Some(v) = f.next() {{
            out = v
        }}
    }}
    return out
}}

@effect(Event)
fiber def probe(): i64 {{
    yield next_event()
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
        .parse_with_filename(src, "<hot_reload_effect_fibers>")
        .expect("source should parse")
}

/// Baseline: the composition runs at all under the tiered runtime —
/// events dispatch through the fiber's handler segment and fold into
/// state. One transition: state = 0 + 3 * 1.
#[test]
fn an_effectful_fiber_observes_events_under_the_tiered_runtime() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1)))
        .expect("v1 should compile");
    let v = rt.call_raw("first_step", &[]).expect("first_step");
    assert_eq!(v, ZyntaxValue::Int(3));
}

/// The machine performs effects, and reloading an effect-bearing
/// function is beyond the call/loop machinery until op-table patching
/// lands: the reload must decline it — reported, not crashed — and the
/// running machine completes untouched on the code it started with.
/// Flip this into a migration test when phase 3 reaches effects.
#[test]
fn an_observing_machine_declines_the_edit_safely_for_now() {
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

    std::thread::sleep(std::time::Duration::from_millis(100));
    let report = rt
        .reload_typed_program(parse(&source(100)))
        .expect("reload must not crash a running effectful program");
    let count = worker.join().expect("drive should complete");

    assert!(
        report.failed.iter().any(|(n, _)| n == "machine"),
        "the effectful machine must be declined with a reason: {report:?}"
    );
    assert_eq!(
        count,
        CAP / 3,
        "the running machine completes on the code it started with"
    );
}

/// Editing the HANDLER is the phase-3 boundary: the reload recompiles
/// it, but effect op tables still hold the pointers baked at module
/// compile, so dispatch keeps reaching the old implementation until
/// op-table patching lands. This pins today's contract — flip the
/// dispatch assertion when phase 3 does.
#[test]
fn a_handler_edit_reloads_but_dispatch_keeps_the_old_implementation_for_now() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1)))
        .expect("v1 should compile");
    assert_eq!(
        rt.call_raw("first_step", &[]).expect("first_step"),
        ZyntaxValue::Int(3)
    );

    let edited = source(1).replace("return 3", "return 5");
    let report = rt
        .reload_typed_program(parse(&edited))
        .expect("reload should succeed");
    assert!(
        report
            .reloaded
            .iter()
            .any(|n| n.contains("next_event") || n.contains("Feed")),
        "the handler implementation must diff as changed: {report:?}"
    );

    let v = rt.call_raw("first_step", &[]).expect("first_step");
    assert_eq!(
        v,
        ZyntaxValue::Int(3),
        "op tables are not patched yet; dispatch still reaches the old body"
    );
}

/// The FSM shape that observes an event and folds it in a match arm,
/// all inside a `with` scope, still loses the arm's assignment across
/// the merge: the read that resolves the merge runs before the match's
/// edges are fully wired, so the collapse fires on incomplete
/// predecessors and no join phi forms. Un-ignore when the
/// translation-order gap is closed.
#[test]
#[ignore = "match-arm assignment inside `with` reads its merge before the match's edges exist"]
fn a_match_arm_assignment_inside_with_reaches_the_merge() {
    let mut rt = runtime();
    let src = r#"
effect Event { def next_event(): i64 }
handler Feed for Event { def next_event(): i64 { return 3 } }

@effect(Event)
fiber def probe(): i64 {
    yield next_event()
}

def observe(): i64 {
    let mut out: i64 = 0
    with Feed {
        let f = probe()
        match f.next() {
            case Some(v) { out = v }
            case None() { }
        }
    }
    return out
}
"#;
    rt.compile_typed_program(parse(src))
        .expect("should compile");
    let v = rt.call_raw("observe", &[]).expect("observe should exist");
    assert_eq!(v, ZyntaxValue::Int(3));
}
