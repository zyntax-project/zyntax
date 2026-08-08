//! The reload generation ladder runs both ways: a reload that compiled
//! only partially never applies at all, and an applied reload can be
//! rolled back — beads, reload cells and dispatch tables swing back to
//! the previous generation, with state untouched either way.

use zynml::{Grammar2, ZynML, ZynMLConfig, ZynMLRuntimeProfile, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

fn runtime() -> ZynML {
    let config = ZynMLConfig {
        runtime_profile: ZynMLRuntimeProfile::TieredDevelopment,
        tier_overrides: zynml::TierOverrides {
            enable_hot_reload: Some(true),
            ..Default::default()
        },
        ..Default::default()
    };
    ZynML::with_config(config).expect("runtime should start")
}

// Both functions sit past the inliner's cap so `main`'s calls stay
// real calls through the reload cells.
fn source(f_step: i64, g_step: i64) -> String {
    let mut f_filler = String::new();
    for _ in 0..120 {
        f_filler.push_str(&format!("    acc = acc + {f_step}\n"));
    }
    let mut g_filler = String::new();
    for _ in 0..120 {
        g_filler.push_str(&format!("    acc = acc + {g_step}\n"));
    }
    format!(
        "def f(n: i64): i64 {{\n    if n < 2 {{\n        return n\n    }}\n    \
         let mut acc: i64 = 0\n{f_filler}    return acc + f(n - 1) + f(n - 2)\n}}\n\n\
         def g(n: i64): i64 {{\n    if n < 2 {{\n        return n\n    }}\n    \
         let mut acc: i64 = 0\n{g_filler}    return acc + g(n - 1) + g(n - 2)\n}}\n\n\
         def main(): i64 {{\n    return f(2) * 1000 + g(2)\n}}\n"
    )
}

/// An applied reload can be undone: the previous generation's code is
/// live again at the next call, and a reload after the rollback diffs
/// against the restored code.
#[test]
fn a_rollback_restores_the_previous_generation() {
    let mut rt = runtime();
    rt.load_source(&source(1, 1)).expect("v1 should compile");
    let v1: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v1, 121_121);

    let report = rt
        .reload_source(&source(2, 1))
        .expect("reload should succeed");
    assert_eq!(report.reloaded, vec!["f".to_string()], "{report:?}");
    let v2: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v2, 241_121);

    let restored = rt.rollback_last_reload().expect("rollback should succeed");
    assert_eq!(restored, vec!["f".to_string()]);
    let back: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(back, 121_121, "the previous generation must be live again");

    // The rollback also restored the diff baseline: reloading the
    // edit again is a real change, not a no-op.
    let again = rt
        .reload_source(&source(2, 1))
        .expect("reload should succeed");
    assert_eq!(again.reloaded, vec!["f".to_string()], "{again:?}");
    let v2_again: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v2_again, 241_121);
}

/// Rollback is one-shot and only meaningful after an applied reload.
#[test]
fn a_rollback_without_an_applied_reload_errors() {
    let mut rt = runtime();
    rt.load_source(&source(1, 1)).expect("v1 should compile");
    assert!(rt.rollback_last_reload().is_err());

    let report = rt
        .reload_source(&source(2, 1))
        .expect("reload should succeed");
    assert_eq!(report.reloaded, vec!["f".to_string()]);
    rt.rollback_last_reload().expect("first rollback succeeds");
    assert!(
        rt.rollback_last_reload().is_err(),
        "the undo record is consumed"
    );
}

/// A compile failure anywhere in the edit set aborts the whole reload:
/// functions that DID compile are not applied, so the running program
/// never sees a half-edited generation. (The failure is injected via a
/// test hook — real compile failures on validated HIR are rare.)
#[test]
fn a_partial_compile_failure_applies_nothing() {
    let mut rt = runtime();
    rt.load_source(&source(1, 1)).expect("v1 should compile");

    std::env::set_var("ZYNTAX_RELOAD_INJECT_FAIL", "g");
    let report = rt.reload_source(&source(2, 2));
    std::env::remove_var("ZYNTAX_RELOAD_INJECT_FAIL");
    let report = report.expect("an aborted reload reports, not errors");

    assert!(report.aborted, "{report:?}");
    assert!(report.failed.iter().any(|(n, _)| n == "g"), "{report:?}");
    assert!(
        report.reloaded.is_empty(),
        "nothing may apply from an aborted set: {report:?}"
    );

    let v: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v, 121_121, "f compiled but must not have been applied");

    // The aborted reload left no undo record.
    assert!(rt.rollback_last_reload().is_err());

    // A clean retry of the same edit applies fully.
    let retry = rt
        .reload_source(&source(2, 2))
        .expect("reload should succeed");
    assert!(!retry.aborted, "{retry:?}");
    let v: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v, 241_241);
}

/// Rolling back a handler edit restores the dispatch-table slots: a
/// scope entered after the rollback performs into the original
/// implementation again.
#[test]
fn a_rollback_restores_handler_dispatch() {
    fn parse(src: &str) -> zyntax_embed::TypedProgram {
        let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
        grammar
            .parse_with_filename(src, "<hot_reload_rollback>")
            .expect("source should parse")
    }
    let src_v1 = r#"
effect Event { def next_event(): i64 }
handler Feed for Event { def next_event(): i64 { return 3 } }

@effect(Event)
fiber def probe(): i64 {
    yield next_event()
}

def first_step(): i64 {
    let mut out: i64 = 0
    with Feed {
        let f = probe()
        while let Some(v) = f.next() {
            out = v
        }
    }
    return out
}
"#;

    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("runtime should start");

    rt.compile_typed_program(parse(src_v1))
        .expect("v1 should compile");
    assert_eq!(
        rt.call_raw("first_step", &[]).expect("first_step"),
        ZyntaxValue::Int(3)
    );

    let edited = src_v1.replace("return 3", "return 7");
    let report = rt
        .reload_typed_program(parse(&edited))
        .expect("reload should succeed");
    assert!(
        report
            .dispatch_patched
            .iter()
            .any(|n| n.contains("next_event")),
        "{report:?}"
    );
    assert_eq!(
        rt.call_raw("first_step", &[]).expect("first_step"),
        ZyntaxValue::Int(7)
    );

    let restored = rt.rollback_last_reload().expect("rollback should succeed");
    assert!(
        restored.iter().any(|n| n.contains("next_event")),
        "{restored:?}"
    );
    assert_eq!(
        rt.call_raw("first_step", &[]).expect("first_step"),
        ZyntaxValue::Int(3),
        "dispatch must reach the original implementation again"
    );
}
