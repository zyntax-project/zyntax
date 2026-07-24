//! End-to-end tests for first-class SIMD vector values.
//!
//! A real ZynML source string constructs `f32x4` / `i32x4` values with
//! the `Type::splat` / `Type::new` constructors, performs element-wise
//! arithmetic, horizontal reductions, unary lane math and lane access,
//! and is executed through the default BC interpreter — proving every
//! surface operation lowers to inline vector HIR (never an FFI call).

use std::time::{Duration, Instant};

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::ZyntaxRuntime;

fn compile_and_install(source: &str) -> ZyntaxRuntime {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<simd_first_class>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program)
        .expect("typed-program → HIR → install should succeed");
    rt
}

/// Same front door as `compile_and_install`, but installs the tier-up
/// ladder with a warm threshold of 1 so the first calls promote the
/// function from the BC interpreter to the Cranelift JIT.
fn compile_with_jit(source: &str) -> ZyntaxRuntime {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<simd_first_class_jit>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program).expect("compile");
    let mut cfg = TieredConfig::default();
    cfg.profile_config.warm_threshold = 1;
    rt.install_interp_jit_with(cfg).expect("install interp JIT");
    rt
}

fn poll_until(deadline: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cond()
}

/// Construct two vectors, multiply element-wise, reduce to a scalar:
/// dot([1,2,3,4], [5,6,7,8]) = 5 + 12 + 21 + 32 = 70.
#[test]
fn f32x4_dot_product() {
    let rt = compile_and_install(
        r#"
        def dot4(): f32 {
            let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, 4.0)
            let b: f32x4 = f32x4::new(5.0, 6.0, 7.0, 8.0)
            let p: f32x4 = a * b
            return p.sum()
        }
        "#,
    );

    let result = rt
        .call_function_raw("dot4", vec![])
        .expect("call should succeed");
    assert_eq!(result.as_float(), Some(70.0), "got {:?}", result);
}

/// Broadcast a scalar across all lanes, then read one lane back.
#[test]
fn f32x4_splat_then_lane() {
    let rt = compile_and_install(
        r#"
        def splat_lane(): f32 {
            let v: f32x4 = f32x4::splat(2.5)
            return v[3]
        }
        "#,
    );

    let result = rt
        .call_function_raw("splat_lane", vec![])
        .expect("call should succeed");
    assert_eq!(result.as_float(), Some(2.5), "got {:?}", result);
}

/// Integer lanes: element-wise add then horizontal sum.
/// ([1,2,3,4] + [10,20,30,40]).sum() = 11+22+33+44 = 110.
#[test]
fn i32x4_add_reduce() {
    let rt = compile_and_install(
        r#"
        def add_reduce(): i32 {
            let a: i32x4 = i32x4::new(1, 2, 3, 4)
            let b: i32x4 = i32x4::new(10, 20, 30, 40)
            let s: i32x4 = a + b
            return s.sum()
        }
        "#,
    );

    let result = rt
        .call_function_raw("add_reduce", vec![])
        .expect("call should succeed");
    assert_eq!(result.as_i64(), Some(110), "got {:?}", result);
}

/// Scalar broadcast in a binary op: v * 2.0 scales every lane, then
/// lane 1 is read back: (3.0 * 2.0) = 6.0.
#[test]
fn f32x4_scalar_broadcast_mul() {
    let rt = compile_and_install(
        r#"
        def scale(): f32 {
            let v: f32x4 = f32x4::new(1.0, 3.0, 5.0, 7.0)
            let scaled: f32x4 = v * 2.0
            return scaled[1]
        }
        "#,
    );

    let result = rt
        .call_function_raw("scale", vec![])
        .expect("call should succeed");
    assert_eq!(result.as_float(), Some(6.0), "got {:?}", result);
}

/// Fused multiply-add on vectors: `a * b + c`, read one lane.
/// The `fma_contract` pass fuses the vector FMul+FAdd into a single
/// `Intrinsic::Fma`, lowered inline per backend. lane1 = 2*6 + 20 = 32.
#[test]
fn f32x4_fma_lane() {
    let rt = compile_and_install(
        r#"
        def fma4(): f32 {
            let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, 4.0)
            let b: f32x4 = f32x4::new(5.0, 6.0, 7.0, 8.0)
            let c: f32x4 = f32x4::new(10.0, 20.0, 30.0, 40.0)
            let r: f32x4 = a * b + c
            return r[1]
        }
        "#,
    );

    let result = rt.call_function_raw("fma4", vec![]).expect("call");
    assert_eq!(result.as_float(), Some(32.0), "got {:?}", result);
}

/// The same vector FMA promoted to the Cranelift JIT (Cranelift's `fma`
/// lowers an f32x4 triple to a hardware fused multiply-add).
#[test]
fn f32x4_fma_tiers_up_to_cranelift() {
    let rt = compile_with_jit(
        r#"
        def fma4(): f32 {
            let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, 4.0)
            let b: f32x4 = f32x4::new(5.0, 6.0, 7.0, 8.0)
            let c: f32x4 = f32x4::new(10.0, 20.0, 30.0, 40.0)
            let r: f32x4 = a * b + c
            return r[1]
        }
        "#,
    );

    for _ in 0..4 {
        let result = rt.call_function_raw("fma4", vec![]).expect("call");
        assert_eq!(result.as_float(), Some(32.0), "interp got {:?}", result);
    }
    let func_ids = rt.interp_registered_function_ids();
    let tiered_up = poll_until(Duration::from_millis(2000), || {
        func_ids.iter().any(|fid| rt.interp_function_compiled(*fid))
    });
    assert!(tiered_up, "fma4 never tiered up to Cranelift");
    let result = rt.call_function_raw("fma4", vec![]).expect("call");
    assert_eq!(result.as_float(), Some(32.0), "post-JIT got {:?}", result);
}

/// Element-wise unary math: sqrt of each lane, then read lane 2.
/// sqrt(9.0) = 3.0.
#[test]
fn f32x4_sqrt_lane() {
    let rt = compile_and_install(
        r#"
        def sqrt_lane(): f32 {
            let v: f32x4 = f32x4::new(1.0, 4.0, 9.0, 16.0)
            let r: f32x4 = v.sqrt()
            return r[2]
        }
        "#,
    );

    let result = rt
        .call_function_raw("sqrt_lane", vec![])
        .expect("call should succeed");
    assert_eq!(result.as_float(), Some(3.0), "got {:?}", result);
}

/// An integer dot product promoted to the Cranelift JIT. Proves the
/// first-class vector ops lower to native vector instructions (not just
/// the scalarised interpreter path) and stay correct across the tier-up.
/// ([1,2,3,4] · [10,20,30,40]) = 10+40+90+160 = 300.
#[test]
fn i32x4_dot_product_tiers_up_to_cranelift() {
    let rt = compile_with_jit(
        r#"
        def dot4(): i32 {
            let a: i32x4 = i32x4::new(1, 2, 3, 4)
            let b: i32x4 = i32x4::new(10, 20, 30, 40)
            let p: i32x4 = a * b
            return p.sum()
        }
        "#,
    );

    // Drive calls so the warm counter crosses the promotion threshold,
    // then wait for the background Cranelift compile to land.
    for _ in 0..4 {
        let result = rt.call_function_raw("dot4", vec![]).expect("call");
        assert_eq!(result.as_i64(), Some(300), "interp got {:?}", result);
    }

    let func_ids = rt.interp_registered_function_ids();
    assert!(!func_ids.is_empty(), "no functions registered");
    let tiered_up = poll_until(Duration::from_millis(2000), || {
        func_ids.iter().any(|fid| rt.interp_function_compiled(*fid))
    });
    assert!(
        tiered_up,
        "dot4 never tiered up to Cranelift — cannot claim native-backend coverage"
    );

    // Post-tier-up calls now dispatch the Cranelift code; value holds.
    for _ in 0..4 {
        let result = rt.call_function_raw("dot4", vec![]).expect("call");
        assert_eq!(result.as_i64(), Some(300), "post-JIT got {:?}", result);
    }
}

/// f32 dot product promoted to the Cranelift JIT. Guards the float
/// return path specifically: the reduced `f32` scalar must be read from
/// the float register after tier-up, not the integer register.
/// ([1,2,3,4] · [5,6,7,8]) = 5+12+21+32 = 70.
#[test]
fn f32x4_dot_product_tiers_up_to_cranelift() {
    let rt = compile_with_jit(
        r#"
        def dot4f(): f32 {
            let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, 4.0)
            let b: f32x4 = f32x4::new(5.0, 6.0, 7.0, 8.0)
            let p: f32x4 = a * b
            return p.sum()
        }
        "#,
    );

    for _ in 0..4 {
        let result = rt.call_function_raw("dot4f", vec![]).expect("call");
        assert_eq!(result.as_float(), Some(70.0), "interp got {:?}", result);
    }

    let func_ids = rt.interp_registered_function_ids();
    let tiered_up = poll_until(Duration::from_millis(2000), || {
        func_ids.iter().any(|fid| rt.interp_function_compiled(*fid))
    });
    assert!(tiered_up, "dot4f never tiered up to Cranelift");

    // Post-tier-up: the f32 return must survive the float-register read.
    for _ in 0..4 {
        let result = rt.call_function_raw("dot4f", vec![]).expect("call");
        assert_eq!(result.as_float(), Some(70.0), "post-JIT got {:?}", result);
    }
}
