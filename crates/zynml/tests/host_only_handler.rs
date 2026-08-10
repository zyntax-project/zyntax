//! A handler and a machine that the PROGRAM never uses, installed only
//! by the host.
//!
//! Cranelift codegen is pruned to what is reachable from `main`. That
//! describes what the program calls, not what an embedder calls:
//! `push_effect_handler` installs any handler and `get_fiber`
//! constructs any machine, and neither appears as a call site. Pruned,
//! a handler's ops are never defined, so its op table is filled with
//! zero slots and the host cannot install it — which is what the
//! downstream report saw as an unresolvable handler.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{NativeSignature, NativeType, TieredConfig, TieredRuntime, ZyntaxValue};

/// `main` exists and touches none of the effect machinery, so the
/// reachability filter is active and has every reason to prune it.
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

def main(): i64 { return 0 }
"#;

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    rt.compile_typed_program(g.parse_with_filename(SRC, "host_only.zyn").expect("parse"))
        .expect("compile");
    rt
}

#[test]
fn a_handler_the_program_never_uses_is_installable_by_the_host() {
    let mut rt = runtime();
    let seq = rt
        .get_effect_handler("Seq")
        .expect("a handler only the host installs must still resolve");
    let inst = rt.new_handler_instance(seq).expect("instance");

    let sig = NativeSignature::new(&[], NativeType::I64);
    let frame = rt.push_handler_instance(inst).expect("push");
    let seen = rt.call_function("read", &[], &sig).expect("read");
    rt.pop_effect_handler(frame);
    assert_eq!(
        seen,
        ZyntaxValue::Int(0),
        "the op table must hold real ops, not zero slots"
    );
    rt.drop_handler_instance(inst);
}

#[test]
fn a_machine_the_program_never_constructs_is_drivable_by_the_host() {
    let mut rt = runtime();
    let seq = rt.get_effect_handler("Seq").expect("handler");
    let inst = rt.new_handler_instance(seq).expect("instance");
    let machine = rt
        .get_fiber("machine")
        .expect("a machine only the host constructs must still have an entry");
    rt.bind_fiber_handler_instance(machine, inst).expect("bind");

    // The machine accumulates `total + bump()`, and the handler counts
    // 1, 2, 3, so the running total is 1, 3, 6.
    for expect in [1, 3, 6] {
        assert_eq!(
            rt.resume_fiber(machine).expect("step"),
            zyntax_embed::HostFiberStep::Yielded(ZyntaxValue::Int(expect)),
            "the machine body must be compiled, and its performs must reach the handler"
        );
    }
    rt.drop_fiber(machine).expect("drop");
    rt.drop_handler_instance(inst);
}

/// The pruning itself. `ZyntaxRuntime::compile_module` and the
/// interpreter's tier-up both compile only what is reachable from
/// `main`, so this asserts against `reachable_function_ids` directly:
/// the end-to-end tests above run on a path that does not prune, and so
/// cannot tell whether the roots are right.
#[test]
fn host_enterable_functions_are_reachability_roots() {
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    let program = g.parse_with_filename(SRC, "roots.zyn").expect("parse");
    let rt = zyntax_embed::ZyntaxRuntime::new().expect("rt");
    let module = rt
        .lower_typed_program(program, indexmap::IndexMap::new())
        .expect("lower");

    let reachable = zyntax_compiler::reachable_function_ids(&module, &["main"]);
    let named = |want: &str| {
        module.functions.iter().any(|(id, f)| {
            f.name.resolve_global().as_deref() == Some(want) && reachable.contains(id)
        })
    };

    // `main` returns 0 and touches none of this, so without host roots
    // every one of these is pruned.
    assert!(
        named("Seq$bump"),
        "a handler op the host may install must be compiled"
    );
    assert!(named("Seq$get"), "every op of the handler, not just one");
    assert!(
        named("Seq$new"),
        "the handler's state constructor is only called by an installer"
    );
    assert!(
        named("machine"),
        "a machine body is entered through krio_fiber_new, never a call"
    );
}
