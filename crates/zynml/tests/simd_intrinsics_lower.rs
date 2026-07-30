//! ML roadmap Phase 0.2 — explicit SIMD intrinsics.
//!
//! The SIMD ops are methods on the first-class vector types
//! (`acc.dot_u8i8(a, b)`, `a.fma(b, c)`, `v.sum()`). This proves each
//! lowers to its dedicated HIR vector instruction (with the right dot
//! encoding flags), so the fast hardware instruction is optimizer-visible
//! and reached without a runtime call.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::hir::{HirCallable, HirFunction, HirInstruction, Intrinsic};
use zyntax_compiler::HirModule;
use zyntax_embed::ZyntaxRuntime;

fn lower(source: &str) -> HirModule {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<simd_intrinsics_lower>")
        .expect("source should parse");
    let rt = ZyntaxRuntime::new().expect("runtime");
    let builtins = rt
        .config()
        .builtins
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    rt.lower_typed_program(program, builtins)
        .expect("typed program should lower to HIR")
}

fn func<'m>(module: &'m HirModule, name: &str) -> &'m HirFunction {
    module
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(name))
        .unwrap_or_else(|| panic!("function `{name}` should be lowered"))
}

fn instructions(f: &HirFunction) -> impl Iterator<Item = &HirInstruction> {
    f.blocks.values().flat_map(|b| b.instructions.iter())
}

/// The three `dot_*` intrinsics lower to `VectorDot` with the encoding
/// flags their suffix names — the signed/unsigned + i7 bits that drive
/// `vpdpbusd` / `sdot` / wasm-dot selection in the backends.
#[test]
fn dot_intrinsics_lower_to_vector_dot_with_correct_flags() {
    let module = lower(
        r#"
        def k_u8i8(acc: i32x4, a: i8x16, b: i8x16): i32 {
            let r: i32x4 = acc.dot_u8i8(a, b)
            return r.sum()
        }
        def k_i8i8(acc: i32x4, a: i8x16, b: i8x16): i32 {
            let r: i32x4 = acc.dot_i8i8(a, b)
            return r.sum()
        }
        def k_u8i7(acc: i32x4, a: i8x16, b: i8x16): i32 {
            let r: i32x4 = acc.dot_u8i7(a, b)
            return r.sum()
        }
        "#,
    );

    for (name, exp_unsigned, exp_i7) in [
        ("k_u8i8", true, false),
        ("k_i8i8", false, false),
        ("k_u8i7", true, true),
    ] {
        let f = func(&module, name);
        let dot = instructions(f)
            .find_map(|inst| match inst {
                HirInstruction::VectorDot {
                    rhs_unsigned,
                    rhs_i7,
                    ..
                } => Some((*rhs_unsigned, *rhs_i7)),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{name} should contain a VectorDot"));
        assert_eq!(
            dot,
            (exp_unsigned, exp_i7),
            "{name}: VectorDot (rhs_unsigned, rhs_i7) mismatch"
        );

        // The dot method must be intercepted, never a real function call.
        assert!(
            !instructions(f).any(|inst| matches!(
                inst,
                HirInstruction::Call {
                    callee: HirCallable::Function(_) | HirCallable::Symbol(_),
                    ..
                }
            )),
            "{name}: dot/sum must be intercepted, not lowered as a call"
        );

        // And the horizontal reduce lowered too.
        assert!(
            instructions(f)
                .any(|inst| matches!(inst, HirInstruction::VectorHorizontalReduce { .. })),
            "{name}: hreduce should lower to VectorHorizontalReduce"
        );
    }
}

/// `fma(a, b, c)` lowers to a direct `Intrinsic::Fma` call (via the
/// intrinsic alias map), not a call to the prelude stub.
#[test]
fn fma_intrinsic_lowers_to_intrinsic_fma() {
    let module = lower(
        r#"
        def k(a: f32x4, b: f32x4, c: f32x4): f32x4 {
            return a.fma(b, c)
        }
        "#,
    );
    let f = func(&module, "k");
    assert!(
        instructions(f).any(|inst| matches!(
            inst,
            HirInstruction::Call {
                callee: HirCallable::Intrinsic(Intrinsic::Fma),
                ..
            }
        )),
        "fma should lower to an Intrinsic::Fma call"
    );
}
