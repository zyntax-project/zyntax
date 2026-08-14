#![cfg(feature = "cranelift-backend")]

//! # ML test harness
//!
//! The correctness + codegen verification half of the ML roadmap's
//! two-harness architecture (`docs/ML_ROADMAP.md`, "Harness
//! architecture"). Companion to the `ml_bench` throughput runner
//! (`crates/zynml/examples/ml_bench.rs`).
//!
//! Every ML kernel gets two checks:
//!   1. **Numerical correctness** — compile + finalize + call, compare
//!      against a Rust reference within tolerance.
//!   2. **Codegen verification** — compile with `set_capture_ir(true)`,
//!      assert the fast instruction is emitted per host (`vpdpbusd` on
//!      AVX-VNNI x86, `sdot` on aarch64, `i32x4.dot` on wasm, `fma`, or
//!      the documented widening fallback).
//!
//! Only **Phase 0** is real here. Phases 1–4 are `#[ignore]` stubs, one
//! per planned kernel from the roadmap's per-phase "Test harness"
//! bullets — they compile (they are just ignored) and reference no
//! unbuilt language features.

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::{
    BinaryOp, CallingConvention, HirConstant, HirFunction, HirFunctionSignature, HirId,
    HirInstruction, HirParam, HirTerminator, HirType, HirValueKind, ParamAttributes,
};
use zyntax_typed_ast::InternedString;

// =========================================================================
// Shared helpers
// =========================================================================

/// Codegen-assert helper: compile a hand-built `HirFunction` through the
/// Cranelift backend with IR capture on, returning `(clif, disasm)`.
/// Modeled on the compiler's `test_vector_dot_i8x16_vpdpbusd`.
fn capture_cranelift_ir(func: &HirFunction) -> (String, Option<String>) {
    let mut backend = CraneliftBackend::new().expect("backend");
    backend.set_capture_ir(true);
    backend
        .compile_function(func.id, func)
        .expect("compile_function");
    backend.take_captured_ir().expect("captured IR")
}

/// Assert the captured native disassembly contains `mnemonic`, but only
/// when `host_supported` is true — so the same test is a no-op on hosts
/// that do not have the feature (e.g. the `vpdpbusd` assert is guarded
/// off on this aarch64 machine, and would assert on VNNI x86 / under
/// Intel SDE). Parameterizes the mnemonic + the host-feature gate.
fn assert_disasm_has(disasm: &Option<String>, mnemonic: &str, host_supported: bool) {
    if !host_supported {
        return;
    }
    let d = disasm.clone().unwrap_or_default();
    assert!(
        d.contains(mnemonic),
        "expected `{mnemonic}` in the disassembly:\n{d}"
    );
}

/// Numerical-correctness helper: compile + finalize a hand-built
/// function and hand back the backend (kept alive so the code stays
/// mapped) plus the raw function pointer.
fn finalize_fn(func: &HirFunction) -> (CraneliftBackend, *const u8) {
    let mut backend = CraneliftBackend::new().expect("backend");
    backend
        .compile_function(func.id, func)
        .expect("compile_function");
    backend.finalize_definitions().expect("finalize");
    let ptr = backend.get_function_ptr(func.id).expect("fn ptr");
    (backend, ptr)
}

/// True when the running x86_64 host has AVX-VNNI (the `vpdpbusd`
/// instruction). Always false off x86_64, so the codegen assert on
/// other arches is a no-op.
#[allow(unreachable_code)]
fn host_has_avxvnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        return std::arch::is_x86_feature_detected!("avxvnni")
            || std::arch::is_x86_feature_detected!("avx512vnni");
    }
    false
}

/// Build `fn dot16(a: *i8, b: *i8) -> {i32 | i32x4}` — a single i8x16
/// `VectorDot`. `rhs_unsigned` selects the u8×i8 (b zero-extended) form.
/// When `reduce`, the i32x4 result is horizontally summed to a scalar
/// i32 (the correctness shape); otherwise the raw i32x4 is returned (the
/// codegen shape, matching `test_vector_dot_i8x16_vpdpbusd`).
fn build_dot16(rhs_unsigned: bool, reduce: bool) -> HirFunction {
    let ptr_i8 = || HirType::Ptr(Box::new(HirType::I8));
    let i8x16 = HirType::Vector(Box::new(HirType::I8), 16);
    let i32x4 = HirType::Vector(Box::new(HirType::I32), 4);

    let ret = if reduce { HirType::I32 } else { i32x4.clone() };
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: InternedString::new_global("a"),
                ty: ptr_i8(),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: InternedString::new_global("b"),
                ty: ptr_i8(),
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![ret],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(InternedString::new_global("dot16"), sig);
    func.calling_convention = CallingConvention::C;

    let a_ptr = func.create_value(ptr_i8(), HirValueKind::Parameter(0));
    let b_ptr = func.create_value(ptr_i8(), HirValueKind::Parameter(1));
    let zero = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let av = func.create_value(i8x16.clone(), HirValueKind::Instruction);
    let bv = func.create_value(i8x16.clone(), HirValueKind::Instruction);
    let acc = func.create_value(i32x4.clone(), HirValueKind::Instruction);
    let dot = func.create_value(i32x4.clone(), HirValueKind::Instruction);

    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::VectorLoad {
        result: av,
        ty: i8x16.clone(),
        ptr: a_ptr,
        align: 1,
    });
    block.add_instruction(HirInstruction::VectorLoad {
        result: bv,
        ty: i8x16.clone(),
        ptr: b_ptr,
        align: 1,
    });
    block.add_instruction(HirInstruction::VectorSplat {
        result: acc,
        ty: i32x4.clone(),
        scalar: zero,
    });
    block.add_instruction(HirInstruction::VectorDot {
        result: dot,
        acc,
        a: av,
        b: bv,
        rhs_i7: false,
        rhs_unsigned,
    });

    if reduce {
        let r = func.create_value(HirType::I32, HirValueKind::Instruction);
        let block = func.blocks.get_mut(&entry).unwrap();
        block.add_instruction(HirInstruction::VectorHorizontalReduce {
            result: r,
            ty: HirType::I32,
            vector: dot,
            op: BinaryOp::Add,
        });
        block.set_terminator(HirTerminator::Return { values: vec![r] });
    } else {
        let block = func.blocks.get_mut(&entry).unwrap();
        block.set_terminator(HirTerminator::Return { values: vec![dot] });
    }

    func
}

// =========================================================================
// Phase 0 — REAL (must pass)
// =========================================================================

/// Phase 0 codegen: the u8×i8 `VectorDot { rhs_unsigned }` lowers to the
/// fused quantized dot. The CLIF is always the widening-dot shape
/// (arch-independent); on an AVX-VNNI x86 host the native disasm folds
/// to a single `vpdpbusd`. The disasm assert is guarded on
/// `is_x86_feature_detected!`, so it is a no-op on this aarch64 machine
/// (and on non-VNNI x86); it fires on VNNI x86 / under Intel SDE.
#[test]
fn phase0_qdot_u8i8_codegen_vpdpbusd() {
    let func = build_dot16(/* rhs_unsigned */ true, /* reduce */ false);
    let (clif, disasm) = capture_cranelift_ir(&func);

    // Host-independent: the u8×i8 dot must lower to the widening tree
    // (a sign-widened, b zero-widened) that the per-arch ISLE rules fold.
    assert!(
        clif.contains("swiden_low") && clif.contains("uwiden_low"),
        "u8xi8 dot CLIF is not the sign×unsigned widening shape:\n{clif}"
    );

    // Host-gated: the fused `vpdpbusd` on AVX-VNNI x86. No-op elsewhere.
    assert_disasm_has(&disasm, "vpdpbusd", host_has_avxvnni());
}

/// Phase 0 correctness: the quantized u8×i8 dot equals a Rust reference,
/// including a `b > 127` lane so the unsigned (b zero-extended) vs signed
/// interpretation is actually distinguished — a signed-widening bug would
/// read those bytes as negative and mismatch.
#[test]
fn phase0_qdot_u8i8_correctness() {
    let func = build_dot16(/* rhs_unsigned */ true, /* reduce */ true);
    let (_backend, ptr) = finalize_fn(&func);
    let f = unsafe {
        std::mem::transmute::<*const u8, unsafe extern "C" fn(*const i8, *const i8) -> i32>(ptr)
    };

    // a: signed i8, includes negatives. b: unsigned u8, includes bytes
    // > 127 (200, 255, 130) whose bit pattern is negative as i8.
    let a: [i8; 16] = [1, -2, 3, -4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -16];
    let b_u8: [u8; 16] = [1, 2, 200, 4, 255, 6, 7, 8, 9, 10, 130, 12, 13, 14, 15, 16];
    let b: [i8; 16] = b_u8.map(|x| x as i8);

    let got = unsafe { f(a.as_ptr(), b.as_ptr()) };

    // Reference: a sign-extended, b zero-extended (unsigned).
    let expect: i32 = (0..16).map(|i| (a[i] as i32) * (b_u8[i] as i32)).sum();

    assert_eq!(
        got, expect,
        "u8xi8 dot mismatch (b>127 unsigned case): got {got}, expected {expect}"
    );

    // Sanity: a plain signed×signed dot over the same bytes would differ,
    // proving the test actually exercises the unsigned path.
    let signed_ref: i32 = (0..16).map(|i| (a[i] as i32) * (b[i] as i32)).sum();
    assert_ne!(
        expect, signed_ref,
        "test buffers do not distinguish signed vs unsigned — pick a byte > 127"
    );
}

// =========================================================================
// Phase 1 — dense Tensor<T>  (ignored stubs)
// =========================================================================

#[test]
#[ignore = "pending Phase 1: elementwise add f32 correctness vs Rust reference (incl. broadcasting)"]
fn phase1_elementwise_add_f32_correctness() {}

#[test]
#[ignore = "pending Phase 1: elementwise loop lowers to VectorLoad/VectorStore/binop, no $Tensor$ FFI"]
fn phase1_elementwise_add_f32_codegen() {}

#[test]
#[ignore = "pending Phase 1: numerically stable softmax vs reference (large-magnitude inputs)"]
fn phase1_softmax_f32_stability() {}

#[test]
#[ignore = "pending Phase 1: RMSNorm/LayerNorm vs reference within tolerance"]
fn phase1_rmsnorm_f32_correctness() {}

#[test]
#[ignore = "pending Phase 1: reduce (sum/mean/max) lowers to a horizontal reduce"]
fn phase1_reduce_codegen() {}

// =========================================================================
// Phase 2 — dense matmul (GEMM)  (ignored stubs)
// =========================================================================

#[test]
#[ignore = "pending Phase 2: f32 GEMM vs reference within tolerance across an M/N/K sweep"]
fn phase2_gemm_f32_correctness() {}

#[test]
#[ignore = "pending Phase 2: GEMM inner loop lowers to fma; zero FFI calls in the hot loop"]
fn phase2_gemm_f32_codegen_fma() {}

#[test]
#[ignore = "pending Phase 2: fused (bias+GELU) epilogue == matmul then bias then activation"]
fn phase2_gemm_f32_fused_equivalence() {}

// =========================================================================
// Phase 3 — quantized  (ignored stubs)
// =========================================================================

#[test]
#[ignore = "pending Phase 3: int8 GEMM vs f32 GEMM within quantization error"]
fn phase3_qgemm_int8_vs_f32() {}

#[test]
#[ignore = "pending Phase 3: int8 GEMM inner lowers to vpdpbusd (x86 VNNI) / sdot (aarch64) / i32x4.dot (wasm); execute on NUC Alder Lake + under Intel SDE"]
fn phase3_qgemm_int8_codegen_vpdpbusd() {}

#[test]
#[ignore = "pending Phase 3: quantize/dequantize round-trip + per-group scale application"]
fn phase3_quant_dequant_roundtrip() {}

// =========================================================================
// Phase 4 — attention + runnable model  (ignored stubs)
// =========================================================================

#[test]
#[ignore = "pending Phase 4: attention (QK^T·scale·softmax·V, causal mask) vs reference"]
fn phase4_attention_correctness() {}

#[test]
#[ignore = "pending Phase 4: attention uses the fused/quantized paths; no FFI in the hot loop"]
fn phase4_attention_codegen() {}

#[test]
#[ignore = "pending Phase 4: tiny quantized model end-to-end logits vs captured known-good"]
fn phase4_transformer_block_e2e() {}
