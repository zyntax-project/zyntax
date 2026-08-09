//! Handlers of one effect are independent of the order they are
//! declared in.
//!
//! A stateful handler's operations take an implicit `self` that a
//! stateless one's do not, so a perform site has to decide per handler
//! whether to pass a state pointer. Deciding it once for the effect —
//! from whichever handler happened to be declared first — makes the
//! calling convention of every scope depend on source order.

use zynml::ZynML;

const STATELESS_FIRST: &str = r#"
effect E {
    def op(): i64
}

handler Plain for E {
    def op(): i64 { return 1 }
}

handler Stateful for E {
    var n: i64 = 0
    def op(): i64 {
        self.n = self.n + 10
        return self.n
    }
}

@effect(E)
def use_it(): i64 {
    return op()
}

def stateful_scope(): i64 {
    let mut a: i64 = 0
    with Stateful {
        a = use_it()
    }
    return a
}

def stateless_scope(): i64 {
    let mut a: i64 = 0
    with Plain {
        a = use_it()
    }
    return a
}
"#;

/// The same program with the two handlers swapped.
fn stateful_first() -> String {
    let plain = "handler Plain for E {\n    def op(): i64 { return 1 }\n}\n\n";
    let stateful = "handler Stateful for E {\n    var n: i64 = 0\n    def op(): i64 {\n        \
                    self.n = self.n + 10\n        return self.n\n    }\n}\n\n";
    assert!(STATELESS_FIRST.contains(plain) && STATELESS_FIRST.contains(stateful));
    STATELESS_FIRST
        .replace(plain, "")
        .replace(stateful, &format!("{stateful}{plain}"))
}

fn check(source: &str, order: &str) {
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(source)
        .unwrap_or_else(|e| panic!("{order}: load failed: {e}"));

    let stateful: i64 = rt
        .call_with_result("stateful_scope")
        .unwrap_or_else(|e| panic!("{order}: stateful scope failed: {e}"));
    assert_eq!(stateful, 10, "{order}: the stateful handler counts from 0");

    let stateless: i64 = rt
        .call_with_result("stateless_scope")
        .unwrap_or_else(|e| panic!("{order}: stateless scope failed: {e}"));
    assert_eq!(stateless, 1, "{order}: the stateless handler returns 1");
}

#[test]
fn a_stateful_handler_works_when_declared_after_a_stateless_one() {
    check(STATELESS_FIRST, "stateless declared first");
}

#[test]
fn a_stateful_handler_works_when_declared_before_a_stateless_one() {
    check(&stateful_first(), "stateful declared first");
}

/// The same shape a host-driven FSM uses: several handlers of one
/// effect, the stateful one declared last, each installed around a
/// step. The machine must observe whichever handler the host names.
#[test]
fn a_machine_observes_either_handler_regardless_of_order() {
    use zyntax_embed::{HostFiberStep, TieredConfig, TieredRuntime, ZyntaxValue};

    let src = r#"
effect E {
    def op(): i64
}

handler Plain for E {
    def op(): i64 { return 1 }
}

handler Stateful for E {
    var n: i64 = 0
    def op(): i64 {
        self.n = self.n + 10
        return self.n
    }
}

@effect(E)
fiber def machine(): i64 {
    yield op()
    yield op()
    return 0
}
"#;
    let grammar = zynml::Grammar2::from_source(zynml::ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(src, "<handler_declaration_order>")
        .expect("parse");

    let mut config = TieredConfig::development();
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("runtime");
    rt.compile_typed_program(program).expect("compile");

    // The stateless handler, declared first.
    let a = rt.get_fiber("machine").expect("get");
    assert_eq!(
        rt.resume_fiber_within(a, &["Plain"]).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(1))
    );
    rt.drop_fiber(a).expect("drop");

    // The stateful one, declared last, bound so its state persists.
    let b = rt.get_fiber("machine").expect("get");
    let stateful = rt.get_effect_handler("Stateful").expect("resolve");
    rt.bind_fiber_handler(b, stateful).expect("bind");
    assert_eq!(
        rt.resume_fiber(b).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(10))
    );
    assert_eq!(
        rt.resume_fiber(b).expect("step"),
        HostFiberStep::Yielded(ZyntaxValue::Int(20)),
        "the state the later-declared handler carries survives the step"
    );
    rt.drop_fiber(b).expect("drop");
}
