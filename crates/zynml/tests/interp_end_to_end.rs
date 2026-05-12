//! End-to-end test of the TypedAST → HIR → bytecode-interpreter path.
//!
//! Proves the integration seam closes: a real ZynML source string is
//! parsed via Grammar2, lowered to HIR via ZyntaxRuntime's pipeline,
//! and executed through the BC interpreter — all via the single
//! `ZyntaxRuntime` front door.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;
use zyntax_embed::ZyntaxValue;

fn compile_and_install(source: &str) -> ZyntaxRuntime {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<test>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program)
        .expect("typed-program → HIR → install should succeed");
    rt
}

#[test]
fn interp_runs_trivial_return() {
    let rt = compile_and_install(
        r#"
        def answer(): i64 {
            return 42
        }
        "#,
    );

    let result = rt
        .call_function_raw("answer", vec![])
        .expect("call should succeed");
    assert_eq!(result.as_i64(), Some(42), "got {:?}", result);
}

#[test]
fn interp_runs_arithmetic_function() {
    let rt = compile_and_install(
        r#"
        def add(a: i64, b: i64): i64 {
            return a + b
        }
        "#,
    );

    let result = rt
        .call_function_raw("add", vec![ZyntaxValue::Int(10), ZyntaxValue::Int(32)])
        .expect("call should succeed");
    assert_eq!(result.as_i64(), Some(42), "got {:?}", result);
}

#[test]
fn interp_runs_multi_function_call() {
    let rt = compile_and_install(
        r#"
        def double(n: i64): i64 {
            return n * 2
        }

        def quadruple(n: i64): i64 {
            return double(double(n))
        }
        "#,
    );

    let r = rt
        .call_function_raw("quadruple", vec![ZyntaxValue::Int(5)])
        .unwrap();
    assert_eq!(r.as_i64(), Some(20), "got {:?}", r);
}

#[test]
fn interp_runs_while_loop_with_mut_local() {
    let rt = compile_and_install(
        r#"
        def sum_to(n: i64): i64 {
            let mut i = 0
            let mut total = 0
            while i < n {
                total = total + i
                i = i + 1
            }
            return total
        }
        "#,
    );

    let r = rt
        .call_function_raw("sum_to", vec![ZyntaxValue::Int(10)])
        .unwrap();
    assert_eq!(r.as_i64(), Some(45), "got {:?}", r);
}

#[test]
fn interp_runs_recursive_function() {
    let rt = compile_and_install(
        r#"
        def factorial(n: i64): i64 {
            if n <= 1 {
                return 1
            }
            return n * factorial(n - 1)
        }
        "#,
    );

    let r = rt
        .call_function_raw("factorial", vec![ZyntaxValue::Int(5)])
        .unwrap();
    assert_eq!(r.as_i64(), Some(120), "got {:?}", r);

    let r = rt
        .call_function_raw("factorial", vec![ZyntaxValue::Int(10)])
        .unwrap();
    assert_eq!(r.as_i64(), Some(3628800), "got {:?}", r);
}

#[test]
fn interp_runs_branching_function() {
    let rt = compile_and_install(
        r#"
        def max(a: i64, b: i64): i64 {
            if a > b {
                return a
            } else {
                return b
            }
        }
        "#,
    );

    let r1 = rt
        .call_function_raw("max", vec![ZyntaxValue::Int(10), ZyntaxValue::Int(20)])
        .unwrap();
    assert_eq!(r1.as_i64(), Some(20), "got {:?}", r1);

    let r2 = rt
        .call_function_raw("max", vec![ZyntaxValue::Int(99), ZyntaxValue::Int(7)])
        .unwrap();
    assert_eq!(r2.as_i64(), Some(99), "got {:?}", r2);
}
