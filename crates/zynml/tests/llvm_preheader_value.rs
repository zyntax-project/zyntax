//! A value hoisted into a loop preheader must survive to the LLVM tier.
//!
//! The LLVM backend emitted blocks in map order, which held only by
//! accident: the order passes happened to create blocks usually matched
//! the order the CFG needs. Hoisting a loop-invariant broadcast into a
//! preheader that was inserted after the loop body broke it, and broke
//! it silently. The read resolved before the definition was built and
//! became a zero vector, so a scaled kernel computed `0 * x + y` and
//! returned zeros with no diagnostic anywhere. Cranelift and both
//! interpreters were correct throughout, which is why only a run with
//! the LLVM tier enabled could catch it.

#![cfg(feature = "llvm-backend")]

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_embed::ZyntaxRuntime;

const AXPY: &str = r#"
import prelude
import simd
def axpy(y: Ptr<f32>, x: Ptr<f32>, alpha: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { y[i] = alpha * x[i] + y[i]  i = i + 1 }
    return n
}
def fill(p: Ptr<f32>, v: f32, n: i64): i64 {
    let mut i: i64 = 0
    while i < n { p[i] = v  i = i + 1 }
    return n
}
def main(): i64 {
    let n: i64 = 1024
    let x: Ptr<f32> = alloc_f32(n)
    let y: Ptr<f32> = alloc_f32(n)
    let f1: i64 = fill(x, 1.0, n)
    let f2: i64 = fill(y, 0.0, n)
    let w: i64 = axpy(y, x, 2.0, n)
    let out: f32 = y[0]
    return (out as i64)
}
"#;

/// The whole pipeline, then LLVM IR for inspection.
fn llvm_ir_for(src: &str) -> String {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar.parse_with_filename(src, "<pre>").expect("parse");
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
    let mut module = rt.lower_typed_program(program, builtins).expect("lower");
    // The vectorizer lives here, and the hoisted broadcast with it.
    zyntax_compiler::run_interp_safe_opts(&mut module);

    zyntax_compiler::compile_module_to_llvm_ir(&module, "preheader").expect("LLVM compile")
}

/// The scale reaching the fused multiply-add must be the value that was
/// broadcast, not a zero vector.
#[test]
fn a_hoisted_broadcast_reaches_the_vector_operation() {
    let ir = llvm_ir_for(AXPY);
    assert!(
        ir.contains("<4 x float>"),
        "the kernel should have widened before this can mean anything"
    );
    for line in ir.lines() {
        let is_vector_fma = line.contains("fma.v4f32") || line.contains("fmul <4 x float>");
        assert!(
            !(is_vector_fma && line.contains("zeroinitializer")),
            "the broadcast was read before it was built and became a zero \
             vector:\n  {line}"
        );
    }
}
