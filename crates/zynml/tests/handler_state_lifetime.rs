//! When a handler's state region is released.
//!
//! `with H { }` frees its state on the scope-exit edge, which is sound
//! because that state is never shared. Host-installed state cannot work
//! that way: one region can back a bind and any number of pushes, so no
//! single install may release it. The owner asks for the drop, each
//! install holds a count, and the last one out frees.
//!
//! These check the observable half of that, which is that a region stays
//! readable for exactly as long as something is installed against it,
//! and that releasing twice or out of order does not fault.

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
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(SRC, "lifetime.zyn").expect("parse"))
        .expect("compile");
    rt
}

fn sig() -> NativeSignature {
    NativeSignature::new(&[], NativeType::I64)
}

/// The owner may let go while a frame is still open. The region has to
/// outlive the install, not the handle.
#[test]
fn an_instance_survives_its_owner_while_a_frame_is_open() {
    let mut rt = runtime();
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");

    let frame = rt.push_handler_instance(counter).expect("push");
    rt.call_function("tick", &[], &sig()).expect("tick");

    // Owner is done, but the frame still names the region.
    rt.drop_handler_instance(counter);

    let seen = rt.call_function("read", &[], &sig()).expect("read");
    assert_eq!(
        seen,
        ZyntaxValue::Int(1),
        "the open frame still reads live state after the owner let go"
    );
    rt.pop_effect_handler(frame);
}

/// Same, with a machine holding the install instead of a frame.
#[test]
fn an_instance_survives_its_owner_while_a_machine_is_bound() {
    let mut rt = runtime();
    let machine = rt.get_fiber("machine").expect("fiber");
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");
    rt.bind_fiber_handler_instance(machine, counter)
        .expect("bind");

    rt.resume_fiber(machine).expect("resume");
    rt.drop_handler_instance(counter);

    // The machine keeps advancing the region its bind still names.
    rt.resume_fiber(machine).expect("resume");
    rt.resume_fiber(machine).expect("resume");
    rt.drop_fiber(machine).expect("drop");
}

/// Releases in either order, and a region shared by two installs that
/// end one at a time.
#[test]
fn releases_are_order_independent() {
    // Owner first, then the installs.
    let mut rt = runtime();
    let machine = rt.get_fiber("machine").expect("fiber");
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");
    rt.bind_fiber_handler_instance(machine, counter)
        .expect("bind");
    let frame = rt.push_handler_instance(counter).expect("push");
    rt.drop_handler_instance(counter);
    rt.pop_effect_handler(frame);
    rt.drop_fiber(machine).expect("drop");

    // Installs first, then the owner.
    let mut rt = runtime();
    let machine = rt.get_fiber("machine").expect("fiber");
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let counter = rt.new_handler_instance(seq).expect("instance");
    rt.bind_fiber_handler_instance(machine, counter)
        .expect("bind");
    let frame = rt.push_handler_instance(counter).expect("push");
    rt.pop_effect_handler(frame);
    rt.drop_fiber(machine).expect("drop");
    rt.drop_handler_instance(counter);

    // Dropping a handle twice is not a second release.
    rt.drop_handler_instance(counter);
}

/// The shorthands allocate a region nothing else can name, so it goes
/// with the install rather than lingering. Repeating them many times
/// must not accumulate.
#[test]
fn the_shorthands_release_what_they_allocate() {
    let mut rt = runtime();
    let seq = rt.get_effect_handler("Seq").expect("handler");

    for _ in 0..200 {
        let frame = rt.push_effect_handler(seq).expect("push");
        let seen = rt.call_function("read", &[], &sig()).expect("read");
        assert_eq!(seen, ZyntaxValue::Int(0), "each push gets its own state");
        rt.pop_effect_handler(frame);
    }

    for _ in 0..200 {
        let machine = rt.get_fiber("machine").expect("fiber");
        rt.bind_fiber_handler(machine, seq).expect("bind");
        rt.resume_fiber(machine).expect("resume");
        rt.drop_fiber(machine).expect("drop");
    }
}

/// Shutting the runtime down reclaims whatever is still registered,
/// including a machine that was never dropped and an instance whose
/// owner never let go.
#[test]
fn shutdown_reclaims_outstanding_state() {
    let mut rt = runtime();
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let _kept = rt.new_handler_instance(seq).expect("instance");
    let machine = rt.get_fiber("machine").expect("fiber");
    rt.bind_fiber_handler(machine, seq).expect("bind");
    rt.resume_fiber(machine).expect("resume");
    drop(rt);
}

/// The release is real, not merely crash-free: 200k push/pop cycles
/// over a 64-byte state must not grow the process. Before host-installed
/// state was owned, this leaked ~12.5 MB, so the bound below fails on a
/// regression rather than merely tolerating one.
#[test]
fn repeated_installs_do_not_grow_the_process() {
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

    const BIG: &str = r#"
effect E { def bump(): i64 }
handler Big for E {
    var a: i64 = 0
    var b: i64 = 0
    var c: i64 = 0
    var d: i64 = 0
    var e: i64 = 0
    var f: i64 = 0
    var g: i64 = 0
    var h: i64 = 0
    def bump(): i64 { self.a = self.a + 1  return self.a }
}
@effect(E)
def tick(): i64 { return bump() }
"#;

    let mut config = TieredConfig::development();
    config.profile_config.warm_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(BIG, "big.zyn").expect("parse"))
        .expect("compile");
    let big = rt.get_effect_handler("Big").expect("handler");

    // Warm up so the measurement excludes first-call JIT and allocator
    // growth that has nothing to do with handler state.
    for _ in 0..2_000 {
        let frame = rt.push_effect_handler(big).expect("push");
        rt.call_function("tick", &[], &sig()).ok();
        rt.pop_effect_handler(frame);
    }
    let base = rss_kb();
    for _ in 0..200_000 {
        let frame = rt.push_effect_handler(big).expect("push");
        rt.call_function("tick", &[], &sig()).ok();
        rt.pop_effect_handler(frame);
    }
    let grew = rss_kb() - base;
    assert!(
        grew < 2_000,
        "200k installs grew RSS by {grew}kB; the state is not being released"
    );
}

/// A layout migration moves the region an instance owns. The registry
/// has to follow it, or the eventual release would free the pointer the
/// migration replaced rather than the one in use.
#[test]
fn an_instance_survives_a_layout_migration() {
    use zyntax_embed::StateMigration;

    const V1: &str = r#"
effect E { def bump(): i64  def get(): i64 }
handler S for E {
    var n: i64 = 0
    def bump(): i64 { self.n = self.n + 1  return self.n }
    def get(): i64 { return self.n }
}
@effect(E)
def tick(): i64 { return bump() }
@effect(E)
def peek(): i64 { return get() }
"#;
    let v2 = V1.replace("var n: i64 = 0", "var pad: i64 = 7\n    var n: i64 = 0");

    let mut config = TieredConfig::development();
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    rt.set_state_migration(StateMigration::ByFieldName);
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(V1, "mig.zyn").expect("parse"))
        .expect("compile");

    let e = rt.get_effect_handler("S").expect("handler");
    let inst = rt.new_handler_instance(e).expect("instance");
    for _ in 0..3 {
        let frame = rt.push_handler_instance(inst).expect("push");
        rt.call_function("tick", &[], &sig()).expect("tick");
        rt.pop_effect_handler(frame);
    }

    rt.reload_typed_program(g.parse_with_filename(&v2, "mig.zyn").expect("parse"))
        .expect("reload");

    let frame = rt.push_handler_instance(inst).expect("push");
    let seen = rt.call_function("peek", &[], &sig()).expect("peek");
    rt.pop_effect_handler(frame);
    assert_eq!(
        seen,
        ZyntaxValue::Int(3),
        "the instance follows its state into the edited layout"
    );

    // Releasing must target the migrated region, not the replaced one.
    rt.drop_handler_instance(inst);
}
