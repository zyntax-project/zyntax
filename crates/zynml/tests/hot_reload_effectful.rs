//! Reloading code that performs effects.
//!
//! An effect's identity crosses generations as a number: a `with` scope
//! pushes its handler under the effect's id and a perform looks the
//! handler up by the same number. A freshly parsed edit numbers
//! everything differently, so a reloaded body has to be rewritten onto
//! the running program's ids — for the effect it performs, the dispatch
//! table it reads, and the constant a `with` scope pushes under.
//! Whichever side of that pair reloads, the other must still find it.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

fn runtime() -> TieredRuntime {
    let mut config = TieredConfig::development();
    config.enable_osr = true;
    config.enable_hot_reload = true;
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    TieredRuntime::new(config).expect("runtime should start")
}

fn parse(src: &str) -> zyntax_embed::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    grammar
        .parse_with_filename(src, "<hot_reload_effectful>")
        .expect("source should parse")
}

/// `read` performs, `main` scopes the handler. Editing one leaves the
/// other's ids untouched, so each direction tests a different half of
/// the remap.
fn source(perform_scale: i64, scope_bonus: i64) -> String {
    format!(
        r#"
effect Env {{
    def get(): i64
}}

handler Fixed for Env {{
    def get(): i64 {{ return 7 }}
}}

@effect(Env)
def read(): i64 {{
    return get() * {perform_scale}
}}

def main(): i64 {{
    let mut total: i64 = 0
    with Fixed {{
        total = read() + {scope_bonus}
    }}
    return total
}}
"#
    )
}

/// The performing function reloads while the `with` scope that pushed
/// the handler does not: the edited perform must look the handler up
/// under the running program's effect id, not the edited module's.
#[test]
fn a_reloaded_perform_finds_a_handler_the_unchanged_scope_pushes() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1, 0)))
        .expect("v1 should compile");
    assert_eq!(rt.call_raw("main", &[]).expect("main"), ZyntaxValue::Int(7));

    let report = rt
        .reload_typed_program(parse(&source(10, 0)))
        .expect("reload should succeed");
    assert!(
        report.reloaded.contains(&"read".to_string()),
        "the performing function must reload: {report:?}"
    );
    assert!(report.failed.is_empty(), "{report:?}");

    assert_eq!(
        rt.call_raw("main", &[]).expect("main"),
        ZyntaxValue::Int(70),
        "the reloaded perform reaches the handler the unchanged scope pushed"
    );
}

/// The mirror: the `with` scope reloads while the performing function
/// does not. The edited scope pushes under the effect id the unchanged
/// perform will look up.
#[test]
fn a_reloaded_scope_pushes_where_the_unchanged_perform_looks() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1, 0)))
        .expect("v1 should compile");
    assert_eq!(rt.call_raw("main", &[]).expect("main"), ZyntaxValue::Int(7));

    let report = rt
        .reload_typed_program(parse(&source(1, 100)))
        .expect("reload should succeed");
    assert!(
        report.reloaded.contains(&"main".to_string()),
        "the scoping function must reload: {report:?}"
    );
    assert!(
        !report.reloaded.contains(&"read".to_string()),
        "the perform did not change and must not reload: {report:?}"
    );

    assert_eq!(
        rt.call_raw("main", &[]).expect("main"),
        ZyntaxValue::Int(107),
        "the unchanged perform still finds the handler the edited scope pushed"
    );
}

const MULTI_OP: &str = r#"
effect Console {
    def first(): i64
    def second(): i64
    def third(): i64
}

handler Wired for Console {
    def first(): i64 { return 1 }
    def second(): i64 { return 20 }
    def third(): i64 { return 300 }
}

@effect(Console)
def pick(): i64 {
    return third()
}

def main(): i64 {
    let mut out: i64 = 0
    with Wired {
        out = pick()
    }
    return out
}
"#;

/// A perform names its operation by position in the effect's
/// declaration order, which codegen reads off the module. A reload
/// compiles against the running program, so the third operation stays
/// the third — compiling against an empty stand-in would silently
/// dispatch every perform to the first slot.
#[test]
fn a_reloaded_perform_keeps_its_operation_index() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(MULTI_OP))
        .expect("v1 should compile");
    assert_eq!(
        rt.call_raw("main", &[]).expect("main"),
        ZyntaxValue::Int(300)
    );

    let edited = MULTI_OP.replace("return third()", "return third() + 5");
    let report = rt
        .reload_typed_program(parse(&edited))
        .expect("reload should succeed");
    assert!(report.reloaded.contains(&"pick".to_string()), "{report:?}");

    assert_eq!(
        rt.call_raw("main", &[]).expect("main"),
        ZyntaxValue::Int(305),
        "the reloaded perform still dispatches to the third operation"
    );
}

/// An edit that introduces a whole handler brings a dispatch table with
/// it. The table's contents are the addresses of functions the same
/// edit introduces, so its slot is reserved before the bodies compile
/// and filled after.
#[test]
fn an_edit_can_introduce_a_handler_and_its_dispatch_table() {
    let mut rt = runtime();
    rt.compile_typed_program(parse(&source(1, 0)))
        .expect("v1 should compile");
    assert_eq!(rt.call_raw("main", &[]).expect("main"), ZyntaxValue::Int(7));

    let edited = r#"
effect Env {
    def get(): i64
}

handler Fixed for Env {
    def get(): i64 { return 7 }
}

handler Doubled for Env {
    def get(): i64 { return 14 }
}

@effect(Env)
def read(): i64 {
    return get()
}

def main(): i64 {
    let mut total: i64 = 0
    with Doubled {
        total = read()
    }
    return total
}
"#;
    let report = rt
        .reload_typed_program(parse(edited))
        .expect("reload should succeed");
    assert!(
        report.added.iter().any(|n| n.contains("Doubled")),
        "the new handler's op implementation must be added: {report:?}"
    );
    assert!(report.failed.is_empty(), "{report:?}");

    assert_eq!(
        rt.call_raw("main", &[]).expect("main"),
        ZyntaxValue::Int(14),
        "the scope dispatches through the newly introduced table"
    );
}

/// Handler state survives a reload of the code around it: the state
/// struct is untouched, and a reloaded op reads the same region its
/// unchanged constructor allocates.
#[test]
fn a_reloaded_op_reads_the_state_its_ctor_allocates() {
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
    let mut rt = runtime();
    rt.compile_typed_program(parse(src)).expect("v1 compiles");
    assert_eq!(
        rt.call_raw("main", &[]).expect("main"),
        ZyntaxValue::Int(6),
        "1 + 2 + 3"
    );

    // Same state layout, edited op body: n now advances by two.
    let edited = src.replace("self.n = self.n + 1", "self.n = self.n + 2");
    let report = rt
        .reload_typed_program(parse(&edited))
        .expect("reload should succeed");
    assert!(
        report.reloaded.iter().any(|n| n.contains("Seq$next")),
        "{report:?}"
    );
    assert!(
        !report.failed.iter().any(|(n, _)| n.contains("Seq")),
        "a same-layout edit must not decline: {report:?}"
    );

    assert_eq!(
        rt.call_raw("main", &[]).expect("main"),
        ZyntaxValue::Int(12),
        "2 + 4 + 6 through the same state region"
    );
}
