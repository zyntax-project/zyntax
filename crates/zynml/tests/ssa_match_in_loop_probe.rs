//! The match-and-continue FSM driver shape: an arm-local assignment
//! merges with an empty arm and re-enters the loop. The merge needs
//! its own phi; without one, one arm's definition leaks across the
//! join and the function fails Cranelift verification.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};

#[test]
fn match_and_continue_compiles_and_runs() {
    let mut config = TieredConfig::development();
    config.profile_config.warm_threshold = u32::MAX as u64;
    config.profile_config.hot_threshold = u32::MAX as u64;
    let mut rt = TieredRuntime::new(config).expect("runtime should start");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    let src = r#"
fiber def ticker(): i64 {
    yield 10
    yield 20
    yield 30
}

def drive(n: i64): i64 {
    let f = ticker()
    let mut last: i64 = 0
    let mut i: i64 = 0
    while i < n {
        match f.next() {
            case Some(v) { last = v }
            case None() { }
        }
        i = i + 1
    }
    return last
}
"#;
    let program = grammar
        .parse_with_filename(src, "<match_in_loop>")
        .expect("source should parse");
    rt.compile_typed_program(program).expect("should compile");
    assert!(
        rt.function_pointer("drive").is_some(),
        "drive must compile — a verifier rejection lands here as a silent skip"
    );

    // Five pumps over a three-yield fiber: the last two are None and
    // must leave `last` at the third yield's value.
    let v = rt
        .call_raw("drive", &[ZyntaxValue::Int(5)])
        .expect("drive should run");
    assert_eq!(v, ZyntaxValue::Int(30));

    // Stop mid-stream: the empty None arm never runs, `last` is the
    // second yield.
    let v = rt
        .call_raw("drive", &[ZyntaxValue::Int(2)])
        .expect("drive should run");
    assert_eq!(v, ZyntaxValue::Int(20));
}
