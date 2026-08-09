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

/// The migration escape hatch. With the policy set, a state-layout
/// change no longer declines: the fields the two layouts share move
/// into a region the edited constructor allocates, fields the edit
/// introduces start from its initializers, and the handler group
/// reloads. The live scope keeps counting from where it was.
#[test]
fn a_migration_policy_moves_live_state_into_the_edited_layout() {
    use zyntax_embed::StateMigration;

    // The machine yields between performs so a reload can land while
    // the scope — and its state — is live.
    let src = r#"
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
fiber def ticker(): i64 {
    yield next()
    yield next()
    yield next()
    return 0
}
"#;
    let edited = r#"
effect Counter {
    def next(): i64
}

handler Seq for Counter {
    var bump: i64 = 100
    var n: i64 = 0
    def next(): i64 {
        self.n = self.n + self.bump
        return self.n
    }
}

@effect(Counter)
fiber def ticker(): i64 {
    yield next()
    yield next()
    yield next()
    return 0
}
"#;

    fn parse(src: &str) -> zyntax_embed::TypedProgram {
        let grammar = zynml::Grammar2::from_source(zynml::ZYNML_GRAMMAR).expect("grammar");
        grammar
            .parse_with_filename(src, "<hot_reload_handler_state>")
            .expect("source should parse")
    }

    let mut config = zyntax_embed::TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = zyntax_embed::TieredRuntime::new(config).expect("runtime");
    rt.set_state_migration(StateMigration::ByFieldName);

    rt.compile_typed_program(parse(src)).expect("v1 compiles");

    // A machine with the handler bound: its state region is live and
    // reachable from the fiber's handler segment across steps.
    let machine = rt.get_fiber("ticker").expect("get");
    let seq = rt.get_effect_handler("Seq").expect("resolve");
    rt.bind_fiber_handler(machine, seq).expect("bind");
    assert_eq!(
        rt.resume_fiber(machine).expect("step"),
        zyntax_embed::HostFiberStep::Yielded(zyntax_embed::ZyntaxValue::Int(1))
    );
    assert_eq!(
        rt.resume_fiber(machine).expect("step"),
        zyntax_embed::HostFiberStep::Yielded(zyntax_embed::ZyntaxValue::Int(2))
    );

    let report = rt
        .reload_typed_program(parse(edited))
        .expect("reload should succeed");
    let plan = report
        .state_migrations
        .iter()
        .find(|p| p.handler.contains("Seq"))
        .unwrap_or_else(|| panic!("a migration must be planned: {report:?}"));
    assert_eq!(
        plan.introduced,
        vec!["bump".to_string()],
        "the added field starts from the edited initializer: {plan:?}"
    );
    assert!(plan.dropped.is_empty(), "{plan:?}");
    assert!(
        !report.failed.iter().any(|(n, _)| n.contains("Seq")),
        "a migrated group must not be declined: {report:?}"
    );

    // n carried over (2), bump came from the new initializer (100).
    assert_eq!(
        rt.resume_fiber(machine).expect("step"),
        zyntax_embed::HostFiberStep::Yielded(zyntax_embed::ZyntaxValue::Int(102)),
        "the live state moved into the edited layout"
    );
    rt.drop_fiber(machine).expect("drop");
}
