//! Regression tests for two related ZynML lowering bugs:
//!
//!   1. `let r = 0.5 + 0.25; return r` used to truncate to `Int(0)`
//!      because the SSA Binary lowering emitted
//!      `Binary { op: FAdd, ty: I64 }` — the op was correct (FAdd
//!      from the operand types) but `inst.ty` came from the
//!      typed-AST expression type which the inferer left at
//!      `Type::Int`. The BC interpreter at
//!      `crates/compiler/src/hir_interp.rs` dispatches integer vs
//!      float by looking at `inst.ty`, so the FAdd silently fell
//!      back to `IAdd` over the operand bit-patterns and produced
//!      `Int(0)`.
//!
//!   2. `let total = a + b + c; if total < thresh` (where the
//!      `a, b, c` are floats) used to pick integer `Lt` because
//!      `convert_binary_op` keys off the typed-AST left-operand
//!      type, and the typed AST inferred `total: Type::Int` even
//!      though the lowered SSA value was `f64`.
//!
//! Both are fixed in `crates/compiler/src/ssa.rs` Binary handler:
//! the op picker now double-checks against the lowered HIR-value
//! type, and the `inst.ty` tag is patched to match the operand
//! width when a float op gets an integer-typed result.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_embed::ZyntaxRuntime;

fn run(src: &str) -> String {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<probe>").expect("parse");
    let rt = ZyntaxRuntime::new().expect("rt");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let module = rt.lower_typed_program(program, builtins).expect("lower");
    let mut rt = ZyntaxRuntime::new().expect("rt");
    rt.compile_module(&module).expect("compile");
    let r = rt.call_function_raw("main", vec![]).expect("call main");
    format!("{:?}", r)
}

#[test]
fn float_let_arithmetic_returns_float() {
    // Untyped let binding — typed AST often leaves `expr.ty` as
    // `Type::Int` for this, so the fix kicks in to patch the
    // Binary instruction's `ty` tag from the operand HIR types.
    let src = "def main(): f64 {\n    let r = 0.5 + 0.25\n    return r\n}\n";
    assert_eq!(run(src), "Float(0.75)");
}

#[test]
fn float_let_with_annotation_returns_float() {
    let src = "def main(): f64 {\n    let r: f64 = 0.5 + 0.25\n    return r\n}\n";
    assert_eq!(run(src), "Float(0.75)");
}

#[test]
fn float_chain_then_compare_picks_float_cmp() {
    // The comparison's operand-type fix path: `total = a + b + c`
    // has f64 HIR operands, but typed AST infers `total: Type::Int`,
    // so without the operand-HIR-type double-check the comparison
    // would pick integer `Lt` and the BC interp would compare the
    // f64 bit-patterns as integers (a few orders of magnitude away
    // from any sensible threshold).
    let src = r#"
def main(): i64 {
    let a = 0.5
    let b = 0.25
    let c = 0.125
    let total = a + b + c
    if total < 1.0 { return 1 }
    return 0
}
"#;
    assert_eq!(run(src), "Int(1)");
}
