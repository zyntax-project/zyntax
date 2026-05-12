//! `ZyntaxRuntime::call_function` delegates to the BC interpreter —
//! the single execution path. Beadie's tier-up loop lives inside the
//! interp; there is no parallel dispatch machinery on the runtime
//! side. This test confirms execution still works correctly through
//! that delegation for representative function shapes.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::{NativeSignature, NativeType, TypedProgram, ZyntaxRuntime, ZyntaxValue};

fn parse(source: &str) -> TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    grammar
        .parse_with_filename(source, "<test>")
        .expect("source should parse")
}

#[test]
fn call_function_delegates_to_interp_no_args() {
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    let program = parse(
        r#"
        def answer(): i64 {
            return 42
        }
        "#,
    );
    rt.compile_typed_program(program).expect("compile");

    let sig = NativeSignature {
        params: vec![],
        ret: NativeType::I64,
    };
    let r = rt
        .call_function("answer", &[], &sig)
        .expect("call should succeed");
    assert_eq!(r, ZyntaxValue::Int(42));
}

#[test]
fn call_function_delegates_to_interp_with_args() {
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    let program = parse(
        r#"
        def add(a: i64, b: i64): i64 {
            return a + b
        }
        "#,
    );
    rt.compile_typed_program(program).expect("compile");

    let sig = NativeSignature {
        params: vec![NativeType::I64, NativeType::I64],
        ret: NativeType::I64,
    };
    let r = rt
        .call_function("add", &[ZyntaxValue::Int(10), ZyntaxValue::Int(32)], &sig)
        .expect("call should succeed");
    assert_eq!(r, ZyntaxValue::Int(42));
}
