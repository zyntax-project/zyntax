//! Hot reload across the effect × fiber composition.
//!
//! An effectful fiber is the observing FSM: each transition performs an
//! effect to read an event, folds it into loop-carried state, and
//! yields. The handler is the machine's event source, installed with
//! `with` around the pump loop. A reload must preserve all three legs
//! at once — the fiber's state, its handler-stack segment, and the
//! dispatch path — while the edited transition takes over.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{OptimizationTier, TieredConfig, TieredRuntime, ZyntaxValue};

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

/// A hot effect-bearing function must be recompiled with the complete module
/// context. Compiling its body in isolation loses the effect and handler
/// tables; the promoter used to panic after emitting a trap for the missing
/// handler and then trying to append a result instruction to that block.
#[test]
fn an_effectful_function_promotes_with_its_module_context() {
    let mut rt = runtime();
    let src = r#"
effect Event { def next_event(): i64 }
handler Feed for Event { def next_event(): i64 { return 3 } }

@effect(Event)
def observe(): i64 {
    let mut out: i64 = 0
    with Feed { out = next_event() }
    return out
}
"#;
    rt.compile_typed_program(parse(src)).expect("compile");
    assert_eq!(
        rt.call_raw("observe", &[]).expect("baseline call"),
        ZyntaxValue::Int(3)
    );
    let baseline = rt.function_pointer("observe").expect("baseline pointer");

    rt.optimize_function("observe", OptimizationTier::Standard)
        .expect("request promotion");
    let mut promoted = false;
    for _ in 0..100 {
        if rt.function_pointer("observe") != Some(baseline) {
            promoted = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(promoted, "tier-1 promotion must install a new entry");
    assert_eq!(
        rt.call_raw("observe", &[]).expect("promoted call"),
        ZyntaxValue::Int(3)
    );
}

/// An effect-performing function reloads like any other: the edited
/// transition is live for machines constructed afterwards, while the
/// machine already running completes on the generation it started
/// with. The performing body carries the edited module's ids for the
/// effect it performs and the dispatch table it reads; both are
/// rewritten onto the running program's, or the reloaded perform
/// would miss the handler `drive` pushed before the edit.
#[test]
fn an_observing_machine_reloads_and_the_running_one_finishes_its_generation() {
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
        report.reloaded.contains(&"machine".to_string()),
        "the effect-performing machine must reload: {report:?}"
    );
    assert!(report.failed.is_empty(), "{report:?}");
    assert_eq!(
        count,
        CAP / 3,
        "the machine already running completes on its own generation"
    );

    // The edit is live for the next machine: each event now folds in
    // 3 * 100, so the same cap takes a hundredth of the transitions.
    let after: i64 = {
        let entry = rt
            .function_pointer("drive")
            .expect("drive should have an entry pointer");
        // SAFETY: as above — `drive` is `() -> i64` and stays mapped.
        let drive: extern "C" fn() -> i64 = unsafe { std::mem::transmute(entry) };
        drive()
    };
    assert_eq!(
        after,
        CAP / 300,
        "a machine constructed after the edit runs the edited transition"
    );
}

/// Editing the handler retargets dispatch in place: the reload
/// recompiles the implementation and patches the handler's
/// dispatch-table slot, so the next perform — even from a scope
/// entered long before the edit — reaches the new body. Handler state
/// and the running program are untouched.
#[test]
fn a_handler_edit_retargets_dispatch_at_the_next_perform() {
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
            .dispatch_patched
            .iter()
            .any(|n| n.contains("next_event")),
        "the handler's dispatch slot must be patched: {report:?}"
    );

    let v = rt.call_raw("first_step", &[]).expect("first_step");
    assert_eq!(
        v,
        ZyntaxValue::Int(5),
        "a fresh perform must reach the edited handler"
    );
}

/// The event source is edited under a machine that is mid-run: the
/// machine itself never reloads, its state and handler segment are
/// untouched, and from its next perform the events it observes carry
/// the edited value. The transition count lands strictly between the
/// all-old and all-new extremes only if dispatch retargeted mid-flight.
#[test]
fn a_running_machine_observes_the_edited_event_source() {
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
    let edited = source(1).replace("return 3", "return 30");
    let report = rt
        .reload_typed_program(parse(&edited))
        .expect("reload should succeed");
    let count = worker.join().expect("drive should complete");

    assert!(
        report
            .dispatch_patched
            .iter()
            .any(|n| n.contains("next_event")),
        "{report:?}"
    );
    assert!(
        count < CAP / 3,
        "every event carried the old value; dispatch never retargeted (count = {count})"
    );
    assert!(
        count > CAP / 30,
        "the machine cannot outpace the edited events (count = {count})"
    );
}

/// The FSM shape that observes an event and folds it in a match arm,
/// all inside a `with` scope: the arm's assignment must reach the
/// merge. The scheduler can translate the merge's reader before every
/// arm is wired, so predecessor knowledge has to combine the CFG's
/// up-front edges with the terminators desugaring wires later — losing
/// either side collapses the merge without its phi.
#[test]
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

/// Reload is itself an observable event: one `RuntimeEvent::Reload`
/// per applied reload, carrying the per-function outcomes — the
/// boundary a UI framework subscribes to for invalidation.
#[test]
fn a_reload_is_observable_as_a_runtime_event() {
    use std::sync::{Arc, Mutex};

    let mut rt = runtime();
    let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    rt.set_event_sink(move |event| {
        if let zyntax_embed::RuntimeEvent::Reload {
            reloaded,
            dispatch_patched,
            ..
        } = event
        {
            sink.lock().unwrap().push(format!(
                "reloaded={reloaded:?} patched={dispatch_patched:?}"
            ));
        }
    });

    rt.compile_typed_program(parse(&source(1)))
        .expect("v1 should compile");
    let edited = source(1).replace("return 3", "return 9");
    rt.reload_typed_program(parse(&edited))
        .expect("reload should succeed");

    let events = seen.lock().unwrap();
    assert_eq!(events.len(), 1, "one reload, one event: {events:?}");
    assert!(
        events[0].contains("next_event"),
        "the event names what changed: {events:?}"
    );
}
