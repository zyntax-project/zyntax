//! Hot reload against stateful handlers: layout is the compatibility
//! contract. The state struct is shared between the handler's ctor
//! (which allocates it) and its ops (which read it through `self`), so
//! an edit that changes its shape cannot reload piecewise — and an
//! edit that keeps the shape reloads freely.

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

const V1: &str = r#"
effect Counter {
    def next(): i64
}

handler Seq for Counter {
    var n: i64 = 0
    def next(): i64 {
        self.n = self.n + 1
        return self.n
    }
}

@effect(Counter)
def tick(): i64 {
    return next()
}

def main(): i64 {
    let mut total: i64 = 0
    with Seq {
        total = tick()
        total = total + tick()
        total = total + tick()
    }
    return total
}
"#;

/// Adding a field to the handler's state changes the layout every
/// generation shares: patched ops would read new offsets out of state
/// old ctors allocated. The reload must decline the whole handler
/// group — ops and ctor together — and the program keeps running the
/// consistent old handler.
#[test]
fn a_state_layout_change_declines_the_handler_group() {
    let mut rt = runtime();
    rt.load_source(V1).expect("v1 should compile");
    let before: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(before, 6, "1+2+3 across performs in one scope");

    let edited = V1.replace(
        "    var n: i64 = 0\n    def next(): i64 {\n        self.n = self.n + 1\n        return self.n\n    }",
        "    var n: i64 = 0\n    var bump: i64 = 100\n    def next(): i64 {\n        self.n = self.n + 1\n        return self.n + self.bump\n    }",
    );
    assert_ne!(edited, V1, "the edit must have applied");

    let report = rt.reload_source(&edited).expect("reload should report");
    let declined_op = report
        .failed
        .iter()
        .find(|(n, _)| n.contains("Seq$next"))
        .map(|(_, r)| r.clone());
    let declined_ctor = report
        .failed
        .iter()
        .find(|(n, _)| n.contains("Seq$new"))
        .map(|(_, r)| r.clone());
    assert!(
        declined_op.as_deref().is_some_and(|r| r.contains("layout")),
        "the op must be declined for layout: {report:?}"
    );
    assert!(
        declined_ctor
            .as_deref()
            .is_some_and(|r| r.contains("layout")),
        "the ctor must be declined with it: {report:?}"
    );
    assert!(
        !report.dispatch_patched.iter().any(|n| n.contains("Seq$")),
        "no dispatch slot of the declined handler may be patched: {report:?}"
    );

    let after: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(after, 6, "the old handler keeps running, consistently");
}

/// An edit that keeps the state shape — here a new initial value —
/// reloads the ctor like any function, and the next scope starts from
/// the edited initializer.
#[test]
fn a_state_value_only_edit_reloads_the_ctor() {
    let mut rt = runtime();
    rt.load_source(V1).expect("v1 should compile");
    let before: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(before, 6);

    let edited = V1.replace("var n: i64 = 0", "var n: i64 = 10");
    let report = rt.reload_source(&edited).expect("reload should succeed");
    assert!(
        report.reloaded.iter().any(|n| n.contains("Seq$new")),
        "same layout, so the ctor reloads: {report:?}"
    );
    assert!(
        !report.failed.iter().any(|(n, _)| n.contains("Seq$")),
        "nothing in the handler group may be declined: {report:?}"
    );

    let after: i64 = rt.call_with_result("main").expect("main should run");
    assert_eq!(
        after, 36,
        "a fresh scope counts from the edited base: 11+12+13"
    );
}
