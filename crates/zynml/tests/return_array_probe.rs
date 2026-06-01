//! Regression tests for two related ZynML compiler issues uncovered
//! while implementing escape-driven allocation analysis:
//!
//!   1. **Cross-call `xs[i]` against a `List<T>` return value.** The
//!      SSA Index handler in `crates/compiler/src/ssa.rs` was only
//!      detecting the List<T> struct shape from local context
//!      (`Ptr(Struct{i64, i64, i64})`). Across a function-call
//!      boundary, the typed AST gives the receiver `Type::Any` and
//!      the HIR sees `Struct(name=List, fields=[])` (List is
//!      pre-registered with 0 fields in the type registry). The
//!      Index handler now also matches `Struct(name="List" |
//!      "Array")` regardless of field count, so the indirection
//!      through the List's data-pointer field works end-to-end.
//!
//!   2. **Alloca → Malloc promotion for escaping allocations.** With
//!      the front-end uniformly emitting Alloca, a function that
//!      returns a `List<T>` would otherwise hand the caller a
//!      pointer into the callee's stack frame. The `alloca_promote`
//!      pass detects the escape (the alloca's result flows into a
//!      Return) and rewrites the instruction to a
//!      `Call(Intrinsic::Malloc)` so the buffer survives the call
//!      return.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

/// Helper: parse, lower, optionally apply a pass, install, call
/// `main`. Returns the result formatted with `{:?}`.
fn run_with(source: &str, transform: impl FnOnce(&mut zyntax_compiler::HirModule)) -> String {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(source, "<probe>")
        .expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut module = rt.lower_typed_program(program, builtins).expect("lower");
    transform(&mut module);
    let mut rt = ZyntaxRuntime::new().expect("rt");
    rt.compile_module(&module).expect("compile");
    let r = rt.call_function_raw("main", vec![]).expect("call main");
    format!("{r:?}")
}

#[test]
fn callee_returns_scalar_through_array_index() {
    // Baseline: indexing into a local List<T> inside the callee.
    // Returns the scalar, so nothing escapes — should work even
    // without alloca_promote.
    let src = r#"
def make(): i64 {
    let xs = [10, 20, 30]
    return xs[1]
}
def main(): i64 {
    return make()
}
"#;
    assert_eq!(run_with(src, |_| {}), "Int(20)");
}

#[test]
fn caller_indexes_into_returned_list_with_promotion() {
    // The case both fixes are designed to address. With
    // alloca_promote applied, the escaping `xs` Alloca in `make` is
    // rewritten to Malloc; with the SSA Index handler's cross-call
    // List detection, `xs[1]` in `main` correctly chases the
    // List<T>.data pointer instead of GEP'ing straight into the
    // struct value (which used to return the `len` field, `Int(3)`).
    let src = r#"
def make(): Array<i64> {
    let xs = [10, 20, 30]
    return xs
}
def main(): i64 {
    let xs = make()
    return xs[1]
}
"#;
    assert_eq!(
        run_with(src, |m| {
            let stats = zyntax_compiler::alloca_promote::run_module(m);
            assert!(
                stats.promoted >= 1,
                "expected at least one escaping Alloca to promote; got {stats:?}"
            );
        }),
        "Int(20)"
    );
}
