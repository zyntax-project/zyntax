//! Parse-shape test for the `?` propagation operator.
//!
//! Verifies the grammar lifts `expr?` into `TypedExpression::Try`
//! through the postfix folding path. No SSA / runtime work
//! exercised here — execution-level integration ships separately
//! once the prelude's Option / Result methods are callable from
//! sample programs.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedExpression, TypedStatement};

fn parse(src: &str) -> zyntax_typed_ast::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    grammar
        .parse_with_filename(src, "<question_mark_op_parse>")
        .expect("source should parse")
}

/// Walk the program and find the first Try expression by checking
/// every Let-binding initializer. Returns the inner expression's
/// kind for assertions.
fn first_try_kind(program: &zyntax_typed_ast::TypedProgram) -> Option<TypedExpression> {
    for decl in &program.declarations {
        if let TypedDeclaration::Function(f) = &decl.node {
            if let Some(body) = &f.body {
                for stmt in &body.statements {
                    if let TypedStatement::Let(let_stmt) = &stmt.node {
                        if let Some(init) = &let_stmt.initializer {
                            if let TypedExpression::Try(inner) = &init.node {
                                return Some(inner.node.clone());
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[test]
fn question_mark_on_call_lifts_to_try() {
    let program = parse(
        r#"
        def caller(): ?i32 {
            let x = lookup()?
            return x
        }
        "#,
    );
    let inner = first_try_kind(&program).expect("expected a Try expression inside caller");
    assert!(
        matches!(inner, TypedExpression::Call(_)),
        "expected Try-wrapping a Call, got {inner:?}"
    );
}

#[test]
fn question_mark_on_method_call_lifts_to_try() {
    let program = parse(
        r#"
        def caller(): ?i32 {
            let x = obj.method()?
            return x
        }
        "#,
    );
    let inner = first_try_kind(&program).expect("expected a Try expression inside caller");
    assert!(
        matches!(inner, TypedExpression::MethodCall(_)),
        "expected Try-wrapping a MethodCall, got {inner:?}"
    );
}

#[test]
fn question_mark_on_variable_lifts_to_try() {
    let program = parse(
        r#"
        def caller(): ?i32 {
            let x = maybe?
            return x
        }
        "#,
    );
    let inner = first_try_kind(&program).expect("expected a Try expression inside caller");
    assert!(
        matches!(inner, TypedExpression::Variable(_)),
        "expected Try-wrapping a Variable, got {inner:?}"
    );
}

#[test]
fn no_question_mark_means_no_try() {
    let program = parse(
        r#"
        def caller(): i32 {
            let x = lookup()
            return x
        }
        "#,
    );
    assert!(
        first_try_kind(&program).is_none(),
        "regular calls must not produce Try expressions"
    );
}
