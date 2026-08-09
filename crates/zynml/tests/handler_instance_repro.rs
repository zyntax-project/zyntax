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
"#;

//! A handler's state cannot be shared between a fiber and the host.
//!
//! `bind_fiber_handler` allocates the handler's state and joins it to
//! the fiber's saved segment; `push_effect_handler` allocates a fresh
//! one per call. Both go through `named_handler_frame`, which calls
//! `H$new` itself, so the caller never sees the state pointer and
//! cannot hand the same one to a second install.
//!
//! The effect is that a machine can advance state its own handler owns
//! while ordinary host-driven code reading through the same token sees
//! a zeroed instance.

#[test]
#[ignore = "known gap: a bound handler's state is not reachable from a pushed frame"]
fn a_bound_handlers_state_is_not_visible_to_a_pushed_frame() {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    let mut rt = TieredRuntime::new(config).expect("rt");
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(SRC, "repro.zyn").expect("parse"))
        .expect("compile");

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
        ZyntaxValue::Int(3),
        "the pushed frame should see the state the machine advanced"
    );
    rt.drop_fiber(machine).expect("drop");
}
