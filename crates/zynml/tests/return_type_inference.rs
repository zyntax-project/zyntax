//! Return types for functions that don't state one.
//!
//! `def f(x) { ... }` has no return annotation. That is distinct from
//! stating `Unit` — it means "work it out" — and the type comes from what
//! the body returns. These pin the three outcomes: void when nothing is
//! returned, the body's type when the returns agree, and a runtime-settled
//! type when they don't.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_compiler::hir::HirType;
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

/// Lower `src` and return the HIR signature return types of `name`.
fn returns_of(src: &str, name: &str) -> Vec<HirType> {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    let program = grammar
        .parse_with_filename(src, "<return_type_inference>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt
        .lower_typed_program(program, builtins)
        .expect("program should lower");

    module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("`{name}` should be present in the lowered module"))
        .signature
        .returns
        .clone()
}

/// Compile `src` and call `main`.
fn run_main(src: &str) -> ZyntaxValue {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    let program = grammar
        .parse_with_filename(src, "<return_type_inference>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt
        .lower_typed_program(program, builtins)
        .expect("program should lower");
    rt.compile_module(&module).expect("program should compile");
    rt.call_function_raw("main", vec![])
        .expect("main should execute")
}

/// The case the `Unit` spelling could not express: a function that states
/// no return type but returns a float. It used to be indistinguishable
/// from a void function, and the fallback gave every such function an
/// integer signature.
#[test]
fn an_unannotated_float_return_is_a_float() {
    let returns = returns_of(
        r#"
        def half() {
            return 0.5
        }
        def main(): i64 {
            return 0
        }
        "#,
        "half",
    );
    assert_eq!(returns, vec![HirType::F64]);
}

#[test]
fn an_unannotated_integer_return_is_an_integer() {
    let returns = returns_of(
        r#"
        def three() {
            return 3
        }
        def main(): i64 {
            return 0
        }
        "#,
        "three",
    );
    assert_eq!(returns, vec![HirType::I64]);
}

#[test]
fn an_unannotated_bool_return_is_a_bool() {
    let returns = returns_of(
        r#"
        def yes() {
            return true
        }
        def main(): i64 {
            return 0
        }
        "#,
        "yes",
    );
    assert_eq!(returns, vec![HirType::Bool]);
}

/// A body that returns nothing is void — the change must not turn every
/// unannotated function into a value-returning one.
#[test]
fn a_body_that_returns_nothing_is_void() {
    let returns = returns_of(
        r#"
        def nothing(x: i64) {
            let y: i64 = x + 1
        }
        def main(): i64 {
            return 0
        }
        "#,
        "nothing",
    );
    assert!(
        returns.is_empty(),
        "a function that returns no value should have no return type, got {returns:?}"
    );
}

/// The return is nested inside an `if`, so the old top-level-only scan
/// missed it and compiled the function as void — silently dropping the
/// value on the one path that produced it.
#[test]
fn a_return_nested_in_control_flow_is_seen() {
    let returns = returns_of(
        r#"
        def maybe(x: i64) {
            if x > 0 {
                return 1.5
            }
            return 0.0
        }
        def main(): i64 {
            return 0
        }
        "#,
        "maybe",
    );
    assert_eq!(returns, vec![HirType::F64]);
}

#[test]
fn a_return_nested_in_a_loop_is_seen() {
    let returns = returns_of(
        r#"
        def scan(n: i64) {
            let mut i: i64 = 0
            while i < n {
                if i == 3 {
                    return 2.5
                }
                i = i + 1
            }
        }
        def main(): i64 {
            return 0
        }
        "#,
        "scan",
    );
    assert_eq!(returns, vec![HirType::F64]);
}

/// Paths that disagree can't be given a static type, so the function keeps
/// the runtime-settled machine word it had before any of this inferred
/// anything.
#[test]
fn disagreeing_return_paths_fall_back_to_a_machine_word() {
    let returns = returns_of(
        r#"
        def mixed(x: i64) {
            if x > 0 {
                return 1
            }
            return 2.5
        }
        def main(): i64 {
            return 0
        }
        "#,
        "mixed",
    );
    assert_eq!(returns, vec![HirType::I64]);
}

/// A stated return type always wins — inference never overrides what the
/// author wrote.
#[test]
fn a_stated_return_type_is_not_second_guessed() {
    let returns = returns_of(
        r#"
        def widened(): f32 {
            return 1.0
        }
        def main(): i64 {
            return 0
        }
        "#,
        "widened",
    );
    assert_eq!(returns, vec![HirType::F32]);
}

/// End to end: the inferred float survives compilation and marshalling,
/// rather than being reinterpreted through an integer return slot.
#[test]
fn an_inferred_float_return_survives_execution() {
    let got = run_main(
        r#"
        def half() {
            return 0.5
        }
        def main(): f64 {
            return half()
        }
        "#,
    );
    match got {
        ZyntaxValue::Float(v) => {
            assert!((v - 0.5).abs() < 1e-12, "expected 0.5, got {v}")
        }
        other => panic!("expected a float result, got {other:?}"),
    }
}

/// A lambda never writes a return type either, and the closure ABI's I64
/// default silently reinterpreted float-valued ones.
#[test]
fn a_lambda_body_settles_its_return_type() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar should compile");
    let src = r#"
        def main(): i64 {
            let half = def(): 0.5
            return 0
        }
        "#;
    let program = grammar
        .parse_with_filename(src, "<return_type_inference>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.add_import_resolver(Box::new(|m| match m {
        "prelude" => Ok(Some(ZYNML_STDLIB_PRELUDE.to_string())),
        "simd" => Ok(Some(ZYNML_STDLIB_SIMD.to_string())),
        _ => Ok(None),
    }));
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt
        .lower_typed_program(program, builtins)
        .expect("program should lower");

    let lambda = module
        .functions
        .values()
        .find(|f| {
            f.name
                .resolve_global()
                .is_some_and(|n| n.starts_with("__lambda"))
        })
        .expect("the lambda should be lowered into its own function");

    assert_eq!(lambda.signature.returns, vec![HirType::F64]);
}
