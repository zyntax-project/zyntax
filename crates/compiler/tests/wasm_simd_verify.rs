//! Verify the wasm backend emits real v128 SIMD — by printing the emitted
//! module to WAT and asserting on it. The fused `VectorDot` must become the
//! relaxed dot-accumulate opcode, the splats `*.splat`, and the reduce
//! `i32x4.extract_lane` — not a scalar fallback.

#![cfg(feature = "wasm-jit")]

use zyntax_compiler::hir::{
    BinaryOp, HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirInstruction,
    HirTerminator, HirType, HirValueKind, Intrinsic,
};
use zyntax_compiler::wasm_backend::WasmBackend;
use zyntax_typed_ast::InternedString;

fn sig_f32x4() -> HirFunctionSignature {
    HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::Vector(Box::new(HirType::F32), 4)],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    }
}

/// A float-lane vector fused multiply-add must lower to the relaxed
/// madd opcode, not a scalar fallback or an out-of-line import.
#[test]
fn wasm_vector_fma_emits_relaxed_madd() {
    let mut func = HirFunction::new(InternedString::new_global("wfma"), sig_f32x4());
    let f32x4 = HirType::Vector(Box::new(HirType::F32), 4);

    let c1 = func.create_value(HirType::F32, HirValueKind::Constant(HirConstant::F32(2.0)));
    let c2 = func.create_value(HirType::F32, HirValueKind::Constant(HirConstant::F32(3.0)));
    let c3 = func.create_value(HirType::F32, HirValueKind::Constant(HirConstant::F32(4.0)));
    let a = func.create_value(f32x4.clone(), HirValueKind::Instruction);
    let b = func.create_value(f32x4.clone(), HirValueKind::Instruction);
    let c = func.create_value(f32x4.clone(), HirValueKind::Instruction);
    let r = func.create_value(f32x4.clone(), HirValueKind::Instruction);

    let entry = func.entry_block;
    let blk = func.blocks.get_mut(&entry).unwrap();
    for (result, scalar) in [(a, c1), (b, c2), (c, c3)] {
        blk.instructions.push(HirInstruction::VectorSplat {
            result,
            ty: f32x4.clone(),
            scalar,
        });
    }
    blk.instructions.push(HirInstruction::Call {
        result: Some(r),
        callee: HirCallable::Intrinsic(Intrinsic::Fma),
        args: vec![a, b, c],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    blk.terminator = HirTerminator::Return { values: vec![r] };

    let module = WasmBackend::new()
        .compile_function(&func)
        .expect("wasm compile");
    let wat = wasmprinter::print_bytes(&module.bytes).expect("print wat");
    eprintln!("\n===== VectorFma WAT =====\n{wat}");

    assert!(
        wat.contains("f32x4.relaxed_madd"),
        "wasm missing the fused madd opcode:\n{wat}"
    );
}

fn sig_i32() -> HirFunctionSignature {
    HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    }
}

#[test]
fn wasm_vector_dot_emits_v128_relaxed_dot() {
    let mut func = HirFunction::new(InternedString::new_global("wdot"), sig_i32());
    let i8x16 = HirType::Vector(Box::new(HirType::I8), 16);
    let i32x4 = HirType::Vector(Box::new(HirType::I32), 4);

    let c2 = func.create_value(HirType::I8, HirValueKind::Constant(HirConstant::I8(2)));
    let c3 = func.create_value(HirType::I8, HirValueKind::Constant(HirConstant::I8(3)));
    let c0 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let a = func.create_value(i8x16.clone(), HirValueKind::Instruction);
    let b = func.create_value(i8x16.clone(), HirValueKind::Instruction);
    let acc = func.create_value(i32x4.clone(), HirValueKind::Instruction);
    let d = func.create_value(i32x4.clone(), HirValueKind::Instruction);
    let r = func.create_value(HirType::I32, HirValueKind::Instruction);

    let entry = func.entry_block;
    let blk = func.blocks.get_mut(&entry).unwrap();
    blk.instructions.push(HirInstruction::VectorSplat {
        result: a,
        ty: i8x16.clone(),
        scalar: c2,
    });
    blk.instructions.push(HirInstruction::VectorSplat {
        result: b,
        ty: i8x16.clone(),
        scalar: c3,
    });
    blk.instructions.push(HirInstruction::VectorSplat {
        result: acc,
        ty: i32x4.clone(),
        scalar: c0,
    });
    blk.instructions.push(HirInstruction::VectorDot {
        result: d,
        acc,
        a,
        b,
        rhs_i7: true,
        rhs_unsigned: false,
    });
    blk.instructions
        .push(HirInstruction::VectorHorizontalReduce {
            result: r,
            ty: HirType::I32,
            vector: d,
            op: BinaryOp::Add,
        });
    blk.terminator = HirTerminator::Return { values: vec![r] };

    let module = WasmBackend::new()
        .compile_function(&func)
        .expect("wasm compile");
    let wat = wasmprinter::print_bytes(&module.bytes).expect("print wat");
    eprintln!("\n===== VectorDot WAT =====\n{wat}");

    assert!(
        wat.contains("i32x4.relaxed_dot_i8x16_i7x16_add_s"),
        "wasm missing the fused dot opcode:\n{wat}"
    );
    assert!(
        wat.contains("i8x16.splat") && wat.contains("i32x4.splat"),
        "wasm missing the splats:\n{wat}"
    );
    assert!(
        wat.contains("i32x4.extract_lane"),
        "wasm missing the reduce extract lanes:\n{wat}"
    );
    assert!(wat.contains("v128"), "wasm has no v128 locals:\n{wat}");
}
