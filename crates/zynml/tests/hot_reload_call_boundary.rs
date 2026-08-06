//! Call-boundary hot reload: an edited function's next call runs the
//! new code, through callers that were not recompiled.
//!
//! The caller/callee split is the load-bearing part: `main` is
//! unchanged across the edit, so it only observes the new `f` if its
//! call goes through the reload cell rather than a direct reference
//! baked at compile time.

use zynml::{ZynML, ZynMLConfig, ZynMLRuntimeProfile};

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

// `f` is built past the inliner's size cap so the call in `main`
// stays a real call — through the reload cell, which is what these
// versions test. An inlinable callee is covered separately below:
// inlining copies the callee into the caller, so the caller
// legitimately reloads with it.
fn source(step: i64, extra_blank_lines: bool) -> String {
    // Self-recursive and past the recursive-inline instruction cap, so
    // the inliner refuses it and `main`'s call to `f` stays a real
    // call through the reload cell.
    let mut filler = String::new();
    for _ in 0..120 {
        filler.push_str(&format!("    acc = acc + {step}\n"));
    }
    let gap = if extra_blank_lines { "\n\n" } else { "" };
    format!(
        "def f(n: i64): i64 {{\n    if n < 2 {{\n        return n\n    }}\n    \
         let mut acc: i64 = 0\n{gap}{filler}    return acc + f(n - 1) + f(n - 2)\n}}\n\n\
         def main(): i64 {{\n    return f(2)\n}}\n"
    )
}

#[test]
fn an_edited_function_is_live_at_the_next_call_without_recompiling_callers() {
    let mut rt = runtime();
    rt.load_source(&source(1, false))
        .expect("v1 should compile");
    let before: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(before, 121);

    let report = rt
        .reload_source(&source(2, false))
        .expect("reload should succeed");
    assert_eq!(report.reloaded, vec!["f".to_string()], "{report:?}");
    assert!(report.failed.is_empty(), "{report:?}");
    assert!(
        !report.reloaded.contains(&"main".to_string()),
        "main did not change and must not reload: {report:?}"
    );

    let after: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(after, 241, "unchanged main must call the reloaded f");
}

#[test]
fn a_formatting_only_edit_reloads_nothing() {
    let mut rt = runtime();
    rt.load_source(&source(1, false))
        .expect("v1 should compile");

    let report = rt
        .reload_source(&source(1, true))
        .expect("reload should succeed");
    assert!(report.is_noop(), "{report:?}");
    assert!(report.unchanged >= 2, "{report:?}");

    let v: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v, 121);
}

#[test]
fn reloading_the_same_source_is_a_noop() {
    let mut rt = runtime();
    rt.load_source(&source(1, false))
        .expect("v1 should compile");
    let report = rt
        .reload_source(&source(1, false))
        .expect("reload should succeed");
    assert!(report.is_noop(), "{report:?}");
}

#[test]
fn an_added_function_becomes_callable_and_a_removed_one_is_retained() {
    let mut rt = runtime();
    rt.load_source(&source(2, false))
        .expect("v2 should compile");

    let mut with_extra = source(2, false);
    with_extra.push_str("\ndef g(): i64 {\n    return 40\n}\n");
    let with_extra = with_extra.replace("return f(2)", "return f(2) + g()");

    let report = rt
        .reload_source(&with_extra)
        .expect("reload should succeed");
    assert!(report.added.contains(&"g".to_string()), "{report:?}");
    assert!(
        report.reloaded.contains(&"main".to_string()),
        "main's body changed: {report:?}"
    );
    let v: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v, 281);

    // Remove g again; it must be reported retained, and main reverts.
    let report = rt
        .reload_source(&source(2, false))
        .expect("reload should succeed");
    assert!(
        report.removed_retained.contains(&"g".to_string()),
        "{report:?}"
    );
    let v: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(v, 241);
}

/// When the optimizer inlined the callee into its caller, editing the
/// callee changes the caller's optimized body too — both reload, and
/// the caller's next call returns the edited result. The diff running
/// on optimized functions is what makes this sound: staleness through
/// inlining is indistinguishable from a direct edit.
#[test]
fn an_inlined_callee_reloads_its_caller_too() {
    let mut rt = runtime();
    rt.load_source(
        "def tiny(): i64 {\n    return 1\n}\n\ndef main(): i64 {\n    return tiny()\n}\n",
    )
    .expect("v1 should compile");
    let before: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(before, 1);

    let report = rt
        .reload_source(
            "def tiny(): i64 {\n    return 2\n}\n\ndef main(): i64 {\n    return tiny()\n}\n",
        )
        .expect("reload should succeed");
    assert!(report.reloaded.contains(&"tiny".to_string()), "{report:?}");
    assert!(
        report.reloaded.contains(&"main".to_string()),
        "main embeds an inlined copy of tiny and must reload with it: {report:?}"
    );

    let after: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(after, 2);
}
