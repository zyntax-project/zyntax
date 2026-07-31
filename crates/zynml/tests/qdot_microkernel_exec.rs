//! ML roadmap Phase 0 exit gate — a quantized int8 dot-product
//! microkernel written entirely in ZynML.
//!
//! The kernel owns a typed buffer, sweeps it with whole-vector loads,
//! accumulates with the quantized dot, and reduces to a scalar — using
//! only the language surface (no hand-built HIR, no FFI). These tests
//! check it computes the right answer, including the unsigned×signed
//! case that distinguishes `dot_u8i8` from a plain signed dot.

use zynml::{Grammar2, ZYNML_GRAMMAR, ZYNML_STDLIB_PRELUDE, ZYNML_STDLIB_SIMD};
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

/// `a` is treated as unsigned bytes, `b` as signed bytes — the
/// quantized-inference convention (unsigned activations, signed weights).
const QDOT_SRC: &str = r#"
import prelude

def qdot(a: Ptr<i8>, b: Ptr<i8>, n_bytes: i64): i32 {
    let mut acc: i32x4 = i32x4::splat(0)
    let mut off: i64 = 0
    while off < n_bytes {
        let av: i8x16 = vload_i8x16(a + off)
        let bv: i8x16 = vload_i8x16(b + off)
        acc = acc.dot_u8i8(av, bv)
        off = off + 16
    }
    return acc.sum()
}
"#;

/// Compile the kernel and run it over the two buffers.
fn run_qdot(a: &[u8], b: &[i8]) -> i64 {
    assert_eq!(a.len(), b.len(), "buffers must match");
    assert_eq!(a.len() % 16, 0, "length must be a whole number of vectors");

    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(QDOT_SRC, "<qdot>")
        .expect("kernel should parse");

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
        .expect("kernel should lower to HIR");
    rt.compile_module(&module).expect("kernel should compile");

    let result = rt
        .call_function_raw(
            "qdot",
            vec![
                ZyntaxValue::Pointer(a.as_ptr() as *mut u8),
                ZyntaxValue::Pointer(b.as_ptr() as *mut u8),
                ZyntaxValue::Int(a.len() as i64),
            ],
        )
        .expect("kernel should execute");

    match result {
        ZyntaxValue::Int(v) => v,
        ZyntaxValue::I32(v) => v as i64,
        other => panic!("expected an integer result, got {other:?}"),
    }
}

/// Reference: unsigned `a` lanes times signed `b` lanes, widened.
fn reference(a: &[u8], b: &[i8]) -> i64 {
    a.iter().zip(b).map(|(x, y)| *x as i64 * *y as i64).sum()
}

#[test]
fn qdot_matches_reference_on_small_values() {
    let a: Vec<u8> = (0..64).map(|i| (i % 7) as u8).collect();
    let b: Vec<i8> = (0..64).map(|i| (i % 5) as i8).collect();
    assert_eq!(run_qdot(&a, &b), reference(&a, &b));
}

/// The case that separates unsigned×signed from signed×signed: `a` lanes
/// above 127 must widen as unsigned, and negative `b` lanes must stay
/// negative. A plain signed dot would get both wrong.
#[test]
fn qdot_handles_unsigned_a_above_127_and_negative_b() {
    let a: Vec<u8> = (0..32).map(|i| 200u8.wrapping_add(i as u8)).collect();
    let b: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { -3 } else { 5 }).collect();
    let got = run_qdot(&a, &b);
    let want = reference(&a, &b);
    assert_eq!(
        got, want,
        "unsigned a (>127) times signed b must widen correctly"
    );
    // Guard the test itself: this data must actually exercise the
    // distinction (a signed reading of `a` would differ).
    let signed_reading: i64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| *x as i8 as i64 * *y as i64)
        .sum();
    assert_ne!(
        want, signed_reading,
        "test data should distinguish unsigned from signed lanes"
    );
}

#[test]
fn qdot_is_zero_for_zero_weights() {
    let a: Vec<u8> = (0..48).map(|i| (i * 3 % 251) as u8).collect();
    let b: Vec<i8> = vec![0; 48];
    assert_eq!(run_qdot(&a, &b), 0);
}
