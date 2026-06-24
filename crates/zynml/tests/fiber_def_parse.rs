//! Parse-shape test for `fiber def`.
//!
//! Verifies the grammar accepts `fiber def NAME(...)` and that the
//! resulting `TypedDeclaration::Function` carries `is_fiber: true`
//! all the way through to the typed AST. No HIR / runtime work
//! exercised — that lives in follow-on commits.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_typed_ast::typed_ast::TypedDeclaration;

fn parse(src: &str) -> zyntax_typed_ast::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    grammar
        .parse_with_filename(src, "<fiber_def_parse>")
        .expect("source should parse")
}

fn find_function<'a>(
    program: &'a zyntax_typed_ast::TypedProgram,
    name: &str,
) -> &'a zyntax_typed_ast::TypedFunction {
    for decl in &program.declarations {
        if let TypedDeclaration::Function(f) = &decl.node {
            if let Some(s) = f.name.resolve_global() {
                if s == name {
                    return f;
                }
            }
        }
    }
    panic!("function `{name}` not found in parsed program");
}

#[test]
fn fiber_def_block_with_return_type_sets_is_fiber() {
    let program = parse(
        r#"
        fiber def gen(): i32 {
            yield 1
            yield 2
        }
        "#,
    );
    let func = find_function(&program, "gen");
    assert!(func.is_fiber, "expected is_fiber=true for `fiber def gen`");
    assert!(
        !func.is_async,
        "fiber and async should be mutually exclusive"
    );
}

#[test]
fn fiber_def_simple_block_sets_is_fiber() {
    let program = parse(
        r#"
        fiber def nop() {
        }
        "#,
    );
    let func = find_function(&program, "nop");
    assert!(func.is_fiber);
}

#[test]
fn regular_def_does_not_set_is_fiber() {
    let program = parse(
        r#"
        def plain(): i32 {
            return 0
        }
        "#,
    );
    let func = find_function(&program, "plain");
    assert!(!func.is_fiber, "regular `def` must not set is_fiber");
    assert!(!func.is_async);
}

#[test]
fn async_def_does_not_set_is_fiber() {
    let program = parse(
        r#"
        async def fetch(): i32 {
            return 0
        }
        "#,
    );
    let func = find_function(&program, "fetch");
    assert!(!func.is_fiber, "async def must not collide with fiber");
    assert!(func.is_async, "async def must set is_async");
}
