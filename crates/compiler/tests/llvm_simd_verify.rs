//! Verify the LLVM backend emits the real dot-product / reduce intrinsics for
//! the SIMD ops — by dumping and asserting on the generated LLVM IR. No
//! "by construction" claims: the IR must contain `@llvm.aarch64.neon.sdot`
//! (which LLVM's AArch64 selector lowers to `SDOT`) and
//! `@llvm.vector.reduce.add` (→ `addv`).

#![cfg(feature = "llvm-backend")]

use inkwell::context::Context;
use zyntax_compiler::hir::{
    BinaryOp, HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule,
    HirTerminator, HirType, HirValueKind,
};
use zyntax_compiler::llvm_backend::LLVMBackend;
use zyntax_typed_ast::InternedString;

fn i32_ret_sig() -> HirFunctionSignature {
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

/// `fn wdot() -> i32 { dot(splat(0), splat(2):i8x16, splat(3):i8x16) |> reduce_add }`
#[test]
fn llvm_vector_dot_emits_sdot_intrinsic() {
    let mut func = HirFunction::new(InternedString::new_global("wdot"), i32_ret_sig());
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
        rhs_i7: false,
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

    let mut module = HirModule::new(InternedString::new_global("m"));
    module.functions.insert(func.id, func);

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "simd_verify");
    let ir = backend.compile_module(&module).expect("LLVM compile");
    eprintln!("\n===== VectorDot LLVM IR =====\n{ir}");

    // The fused widening dot lowers to the AArch64 dot-product intrinsic
    // (→ SDOT), and the horizontal reduce to the vector reduce intrinsic
    // (→ addv) — NOT a scalar mul/add chain.
    #[cfg(target_arch = "aarch64")]
    assert!(
        ir.contains("llvm.aarch64.neon.sdot"),
        "LLVM IR missing the sdot intrinsic:\n{ir}"
    );
    assert!(
        ir.contains("llvm.vector.reduce.add"),
        "LLVM IR missing the vector reduce intrinsic:\n{ir}"
    );

    // Final proof: emit native assembly via a host TargetMachine and confirm
    // the intrinsic actually became an `sdot` (+ `addv`) machine instruction —
    // not just a promise in the IR.
    use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
    };
    use inkwell::OptimizationLevel;
    Target::initialize_native(&InitializationConfig::default()).expect("init native target");
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).expect("target from triple");
    let tm = target
        .create_target_machine(
            &triple,
            TargetMachine::get_host_cpu_name().to_str().unwrap(),
            TargetMachine::get_host_cpu_features().to_str().unwrap(),
            OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .expect("target machine");
    let m = backend.module();
    m.set_triple(&triple);
    m.set_data_layout(&tm.get_target_data().get_data_layout());
    let asm = tm
        .write_to_memory_buffer(m, FileType::Assembly)
        .map(|buf| String::from_utf8_lossy(buf.as_slice()).to_string())
        .expect("emit native asm");
    eprintln!("\n===== VectorDot native asm =====\n{asm}");
    #[cfg(target_arch = "aarch64")]
    {
        assert!(asm.contains("sdot"), "native asm missing `sdot`:\n{asm}");
        assert!(asm.contains("addv"), "native asm missing `addv`:\n{asm}");
    }
}
