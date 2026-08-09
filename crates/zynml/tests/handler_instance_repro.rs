//! Sharing one handler state between a machine and the host.
//!
//! `push_effect_handler` and `bind_fiber_handler` each allocate their
//! own state, so a machine advancing state its handler owns is
//! invisible to host code performing through the same token. Creating
//! the instance separately and installing THAT in both places is what
//! makes the two halves refer to one region.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{NativeSignature, NativeType, TieredConfig, TieredRuntime, ZyntaxValue};

const SRC: &str = r#"
effect Counter {
    def bump(): i64
    def get(): i64
}

handler Seq for Counter {
    var n: i64 = 0
    def bump(): i64 { self.n = self.n + 1  return self.n }
    def get(): i64 { return self.n }
}

@effect(Counter)
fiber def machine(): i64 {
    let mut total: i64 = 0
    while total < 100 {
        total = total + bump()
        yield total
    }
    return total
}

@effect(Counter)
def read(): i64 { return get() }

@effect(Counter)
def tick(): i64 { return bump() }
"#;

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(SRC, "repro.zyn").expect("parse"))
        .expect("compile");
    rt
}

/// The reported case: a machine advances its handler's state, and host
/// code reads that same state afterwards. One instance, two installs.
#[test]
fn a_bound_handlers_state_is_visible_to_a_pushed_frame() {
    let mut rt = runtime();
    let machine = rt.get_fiber("machine").expect("fiber");
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");
    rt.bind_fiber_handler_instance(machine, counter)
        .expect("bind");

    for _ in 0..3 {
        rt.resume_fiber(machine).expect("resume");
    }

    let sig = NativeSignature::new(&[], NativeType::I64);
    let frame = rt.push_handler_instance(counter).expect("push");
    let seen = rt.call_function("read", &[], &sig).expect("read");
    rt.pop_effect_handler(frame);

    assert_eq!(
        seen,
        ZyntaxValue::Int(3),
        "the pushed frame should see the state the machine advanced"
    );

    // And it is one region, not a snapshot: the machine keeps counting
    // from where the host observed it.
    rt.resume_fiber(machine).expect("resume");
    let frame = rt.push_handler_instance(counter).expect("push");
    let seen = rt.call_function("read", &[], &sig).expect("read");
    rt.pop_effect_handler(frame);
    assert_eq!(seen, ZyntaxValue::Int(4));

    rt.drop_fiber(machine).expect("drop");
    rt.drop_handler_instance(counter);
}

/// The allocate-and-install shorthands are unchanged: each still gets
/// its own state, which is what they document.
#[test]
fn the_shorthands_still_allocate_their_own_state() {
    let mut rt = runtime();
    let machine = rt.get_fiber("machine").expect("fiber");
    let seq = rt.get_effect_handler("Seq").expect("handler");
    rt.bind_fiber_handler(machine, seq).expect("bind");

    for _ in 0..3 {
        rt.resume_fiber(machine).expect("resume");
    }

    let sig = NativeSignature::new(&[], NativeType::I64);
    let frame = rt.push_effect_handler(seq).expect("push");
    let seen = rt.call_function("read", &[], &sig).expect("read");
    rt.pop_effect_handler(frame);

    assert_eq!(
        seen,
        ZyntaxValue::Int(0),
        "a fresh push constructs its own instance"
    );
    rt.drop_fiber(machine).expect("drop");
}

/// Host code can own an instance no machine is bound to, and drive it
/// entirely from pushed extents.
#[test]
fn an_instance_needs_no_fiber() {
    let mut rt = runtime();
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");
    assert_eq!(rt.handler_instance_name(counter), Some("Seq"));

    let sig = NativeSignature::new(&[], NativeType::I64);
    for expect in 1..=3 {
        let frame = rt.push_handler_instance(counter).expect("push");
        let bumped = rt.call_function("tick", &[], &sig).expect("tick");
        rt.pop_effect_handler(frame);
        assert_eq!(
            bumped,
            ZyntaxValue::Int(expect),
            "each extent installs the same state, so the count carries"
        );
    }

    let frame = rt.push_handler_instance(counter).expect("push");
    let seen = rt.call_function("read", &[], &sig).expect("read");
    rt.pop_effect_handler(frame);
    assert_eq!(seen, ZyntaxValue::Int(3));
    rt.drop_handler_instance(counter);
}

/// Open question 2 from the report: an extent that contains a resume of
/// a machine bound to the same instance installs that state twice on
/// one thread. Both frames name one region, so the ops compose.
#[test]
fn an_instance_installed_twice_on_one_thread_stays_coherent() {
    let mut rt = runtime();
    let machine = rt.get_fiber("machine").expect("fiber");
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");
    rt.bind_fiber_handler_instance(machine, counter)
        .expect("bind");

    let sig = NativeSignature::new(&[], NativeType::I64);
    // Push the instance, then resume the machine bound to it from
    // inside that extent.
    let frame = rt.push_handler_instance(counter).expect("push");
    rt.resume_fiber(machine).expect("resume");
    rt.resume_fiber(machine).expect("resume");
    let seen = rt.call_function("read", &[], &sig).expect("read");
    rt.pop_effect_handler(frame);

    assert_eq!(
        seen,
        ZyntaxValue::Int(2),
        "the outer frame reads what the nested resumes advanced"
    );
    rt.drop_fiber(machine).expect("drop");
    rt.drop_handler_instance(counter);
}

/// Open question 3 from the report: an instance the host holds but has
/// not installed anywhere is named by no live frame, so the walk over
/// handler frames cannot reach it. The migration has to visit the
/// instance registry too, or a layout edit would leave that state
/// behind while every installed region moved.
#[test]
fn a_reload_migrates_an_instance_that_is_not_installed() {
    use zyntax_embed::StateMigration;

    let edited = SRC
        .replace(
            "var n: i64 = 0",
            "var bump_by: i64 = 100\n    var n: i64 = 0",
        )
        .replace("self.n = self.n + 1", "self.n = self.n + self.bump_by");

    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    rt.set_state_migration(StateMigration::ByFieldName);
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(SRC, "repro.zyn").expect("parse"))
        .expect("compile");

    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");
    let sig = NativeSignature::new(&[], NativeType::I64);

    // Advance it, then leave it installed nowhere across the reload.
    for _ in 0..3 {
        let frame = rt.push_handler_instance(counter).expect("push");
        rt.call_function("tick", &[], &sig).expect("tick");
        rt.pop_effect_handler(frame);
    }

    let report = rt
        .reload_typed_program(g.parse_with_filename(&edited, "repro.zyn").expect("parse"))
        .expect("reload");
    assert!(
        report
            .state_migrations
            .iter()
            .any(|p| p.handler.contains("Seq")),
        "a migration must be planned: {report:?}"
    );

    let frame = rt.push_handler_instance(counter).expect("push");
    let seen = rt.call_function("read", &[], &sig).expect("read");
    rt.pop_effect_handler(frame);
    assert_eq!(
        seen,
        ZyntaxValue::Int(3),
        "n carried across the layout change even though nothing held the state"
    );
    rt.drop_handler_instance(counter);
}
