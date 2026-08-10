//! A handler's op-table global must still be addressable after the JIT
//! module is rebuilt.
//!
//! `rebuild_with_accumulated_symbols` replaces the `JITModule` and
//! clears `function_map`, `global_map` and `compiled_functions`, because
//! those index into the module it just threw away. Nothing re-declares
//! the globals of a module that was already installed, so an
//! `$optable$H` declared before a rebuild is absent from `global_map`
//! afterwards and `global_data_addr` returns `None`. The host reports
//! that as "no unambiguous handler named ...", which names the wrong
//! cause.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime};

const SRC: &str = r#"
effect Ev { def next(): i64 }
handler Feed for Ev {
    var n: i64 = 0
    def next(): i64 { self.n = self.n + 1  return self.n }
}
@effect(Ev)
def tick(): i64 { return next() }
"#;

fn parse(src: &str) -> zyntax_embed::TypedProgram {
    let g = Grammar2::from_source(ZYNML_GRAMMAR).expect("g");
    g.parse_with_filename(src, "rebuild.zyn").expect("parse")
}

#[test]
fn a_handler_resolves_after_a_jit_rebuild() {
    let mut config = TieredConfig::development();
    config.profile_config.warm_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    rt.compile_typed_program(parse(SRC)).expect("compile");

    // Resolves before any rebuild.
    rt.get_effect_handler("Feed")
        .expect("handler before rebuild");

    // Anything that registers new symbols rebuilds the JIT module.
    rt.finalize_runtime_symbols().expect("rebuild");

    rt.get_effect_handler("Feed")
        .expect("handler must still resolve after the JIT module is rebuilt");
}

/// The shape the downstream report hit: a second program installed
/// after the first brings new extern declarations, which forces the
/// rebuild. The first program's handler must survive it.
#[test]
fn a_handler_resolves_after_a_later_module_forces_a_rebuild() {
    let mut config = TieredConfig::development();
    config.profile_config.warm_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    rt.compile_typed_program(parse(SRC)).expect("compile");
    rt.get_effect_handler("Feed").expect("handler before");

    rt.finalize_runtime_symbols().expect("rebuild");
    rt.compile_typed_program(parse(SRC)).expect("recompile");

    rt.get_effect_handler("Feed")
        .expect("handler must survive a later install");
}

/// The failures are told apart now. Reporting a missing global as
/// ambiguity is what made the original bug hard to localise.
#[test]
fn resolution_failures_name_their_own_cause() {
    let mut config = TieredConfig::development();
    config.profile_config.warm_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("rt");
    rt.compile_typed_program(parse(SRC)).expect("compile");

    let err = rt
        .get_effect_handler("NoSuchHandler")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("no handler named"),
        "a missing name should say so, not report ambiguity: {err}"
    );
}
