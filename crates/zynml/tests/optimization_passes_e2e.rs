//! End-to-end tests that verify the new HIR optimization passes
//! (const_fold, cse, licm, inline, loop_vectorize) preserve
//! semantics when run against real ZynML programs.
//!
//! Strategy: compile a ZynML source, apply each pass directly to
//! the lowered module, then execute the result through the BC
//! interpreter. Pinning correctness rather than the exact stats
//! makes these tests robust to internal HirId/order shuffles while
//! still confirming the passes don't introduce dangling SSA
//! references.

use std::sync::Arc;
use zyntax_compiler::{const_fold, cse, hir::HirModule, inline, licm, loop_vectorize, ssa::SsaForm};

use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};
use zynml::{Grammar2, ZYNML_GRAMMAR};

/// Compile a ZynML source through the same path as the runtime, but
/// hand back the raw `HirModule` so we can apply optimization passes
/// before installation.
fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar");
    let program = grammar
        .parse_with_filename(source, "<opt-test>")
        .expect("source should parse");
    let rt = ZyntaxRuntime::new().expect("runtime");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rt.lower_typed_program(program, builtins)
        .expect("lowering should succeed")
}

/// Install a `HirModule` into a fresh runtime and call `main()` (or
/// any specific function), returning whatever value the BC interp
/// produces.
fn run(module: HirModule, fn_name: &str, args: Vec<ZyntaxValue>) -> ZyntaxValue {
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_module(&module)
        .expect("module install should succeed");
    rt.call_function_raw(fn_name, args)
        .expect("call should succeed")
}

// ─── const_fold ───────────────────────────────────────────────────

#[test]
fn const_fold_arithmetic_preserves_result() {
    // `7 * 6` would normally be a Binary instruction at runtime;
    // const_fold collapses it to a Constant(42) before the interp
    // ever sees it.
    let mut module = lower(
        r#"
        def answer(): i64 {
            return 7 * 6
        }
        "#,
    );
    let stats = const_fold::fold_module(&mut module);
    assert!(stats.folded >= 1, "expected at least one fold, got {stats:?}");
    let r = run(module, "answer", vec![]);
    assert_eq!(r.as_i64(), Some(42), "{r:?}");
}

#[test]
fn const_fold_chained_arithmetic_propagates() {
    // (2 + 3) * (10 - 4) — both inner ops fold, then the multiply
    // folds, leaving a single Constant(30).
    let mut module = lower(
        r#"
        def thirty(): i64 {
            return (2 + 3) * (10 - 4)
        }
        "#,
    );
    let stats = const_fold::fold_module(&mut module);
    assert!(
        stats.folded >= 1,
        "expected at least one fold for chained arithmetic, got {stats:?}"
    );
    let r = run(module, "thirty", vec![]);
    assert_eq!(r.as_i64(), Some(30), "{r:?}");
}

// ─── cse ──────────────────────────────────────────────────────────

#[test]
fn cse_does_not_change_observable_behaviour() {
    // `a * b` appears twice; CSE collapses the second into the
    // first. Verify the return value is correct either way.
    let mut module = lower(
        r#"
        def both(a: i64, b: i64): i64 {
            let x = a * b
            let y = a * b
            return x + y
        }
        "#,
    );
    let _ = cse::eliminate_module(&mut module);
    let r = run(module, "both", vec![ZyntaxValue::Int(3), ZyntaxValue::Int(4)]);
    assert_eq!(r.as_i64(), Some(24), "3*4 + 3*4 = 24, got {r:?}");
}

// ─── inline ───────────────────────────────────────────────────────

#[test]
fn inline_leaf_function_preserves_result() {
    // `square(x)` is a single-block callee — leaf inlining target.
    let mut module = lower(
        r#"
        def square(x: i64): i64 {
            return x * x
        }
        def thirteen(): i64 {
            return square(3) + 4
        }
        "#,
    );
    let stats = inline::run_module(&mut module);
    // It MAY skip if the lowered shape uses cleanup code we don't
    // yet recognise — pin correctness regardless.
    let r = run(module, "thirteen", vec![]);
    assert_eq!(r.as_i64(), Some(13), "{r:?}; stats={stats:?}");
}

// ─── licm + const_fold composition ────────────────────────────────

#[test]
fn licm_then_const_fold_keeps_loop_semantics() {
    // The loop body computes `c + c` for an invariant c — LICM
    // should hoist it, and we still expect the correct sum back.
    let mut module = lower(
        r#"
        def sum_to_n(n: i64): i64 {
            let mut total: i64 = 0
            let mut i: i64 = 0
            let c: i64 = 7
            while i < n {
                total = total + c + c
                i = i + 1
            }
            return total
        }
        "#,
    );
    let _ = licm::run_module(&mut module);
    let _ = const_fold::fold_module(&mut module);
    let r = run(module, "sum_to_n", vec![ZyntaxValue::Int(5)]);
    // total = 5 iterations × (7+7) = 5 × 14 = 70
    assert_eq!(r.as_i64(), Some(70), "{r:?}");
}

// ─── stack the full pipeline ──────────────────────────────────────

#[test]
fn full_pipeline_compose_without_regression() {
    // Run every pass on a small program that exercises constants,
    // a redundant subexpression, and a callee — verify the
    // composed result is still correct.
    let mut module = lower(
        r#"
        def square(x: i64): i64 {
            return x * x
        }
        def main(a: i64, b: i64): i64 {
            let p = (3 + 4) * (10 - 4)   // const-foldable to 42
            let q = a * b
            let r = a * b               // CSE-able
            let s = square(a)           // inline-able
            return p + q + r + s
        }
        "#,
    );
    let cf = const_fold::fold_module(&mut module);
    let cse_s = cse::eliminate_module(&mut module);
    let il = inline::run_module(&mut module);
    let _ = licm::run_module(&mut module);
    let _ = loop_vectorize::run_module(&mut module);
    let cf2 = const_fold::fold_module(&mut module); // re-fold after inline
    let _ = cse::eliminate_module(&mut module);

    // For a=3, b=2:
    //   p = 42
    //   q + r = 6 + 6 = 12 (or after CSE: 6 picked twice, same total)
    //   s = 9
    //   total = 42 + 12 + 9 = 63
    let r = run(
        module,
        "main",
        vec![ZyntaxValue::Int(3), ZyntaxValue::Int(2)],
    );
    assert_eq!(
        r.as_i64(),
        Some(63),
        "got {r:?} (const_fold={cf:?}, cse={cse_s:?}, inline={il:?}, const_fold2={cf2:?})"
    );
}
