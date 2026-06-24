//! Dot-notation for static type method calls.
//!
//! ZynML's grammar uses `.` for both instance method calls
//! (`tensor.shape()`) and static type-qualified calls
//! (`Tensor.arange(...)`). The disambiguation lives in
//! `import_chain::resolve_in_expr`: when a `MethodCall`'s receiver is
//! a `Variable(name)` and `name` starts with an uppercase letter
//! (ZynML's convention for type names), the AST is rewritten into
//! `Call(Path([TypeName, method]))`, which the downstream Path-based
//! resolution mangles into `$TypeName$method`.
//!
//! These tests verify the AST rewrite happens, not the runtime
//! correctness (the latter still depends on type inference + stdlib
//! integration that's only partly wired for the dot-syntax path).

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

#[test]
fn dot_static_call_compiles_via_runtime() {
    // The dot-static rewrite lives in
    // `zyntax_embed::import_chain::resolve_in_expr`, which is part
    // of the runtime's `process_imports_for_traits` step (NOT the
    // bare `compile_to_hir` path). This test goes through
    // `ZyntaxRuntime::compile_typed_program` so the rewrite fires.
    let mut rt = ZyntaxRuntime::new().expect("rt");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            def main(): i64 {
                let _a = Tensor.arange(1.0, 5.0, 1.0)
                return 0
            }
            "#,
            "<dot_static_call>",
        )
        .expect("parse");
    rt.compile_typed_program(program)
        .expect("compile should succeed — dot-static rewrite makes `Tensor.arange` reachable");
}

#[test]
fn lowercase_dot_call_treated_as_instance() {
    // Lowercase receiver — keeps the existing instance-method
    // dispatch path. The rewrite heuristic only kicks in on
    // uppercase-named receivers (ZynML's convention for type
    // names).
    let mut rt = ZyntaxRuntime::new().expect("rt");
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            def consume(x: i64): i64 {
                return x
            }
            def main(): i64 {
                let y = 42
                return consume(y)
            }
            "#,
            "<dot_static_lowercase>",
        )
        .expect("parse");
    rt.compile_typed_program(program).expect("compile");
}
