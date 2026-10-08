#![cfg(feature = "cranelift-backend")]
mod common;
use zyntax_compiler::{
    hir::*,
    hir_interp::{value_to_f64, HirInterpreter},
    value::ZyntaxValue,
};
use zyntax_typed_ast::InternedString;

fn fixture(d: HirConstant, ty: HirType, add: bool) -> (HirModule, HirId) {
    let mut sig = common::counted_loop().0.signature;
    sig.params.clear();
    sig.returns = vec![ty.clone()];
    let mut f = HirFunction::new(InternedString::new_global("divide"), sig);
    let x = f.create_value(ty.clone(), HirValueKind::Parameter(0));
    f.signature.params.push(HirParam {
        id: x,
        name: InternedString::new_global("x"),
        ty: ty.clone(),
        attributes: Default::default(),
        ownership: Default::default(),
    });
    let divisor = f.create_value(ty.clone(), HirValueKind::Constant(d));
    let quotient = f.create_value(ty.clone(), HirValueKind::Instruction);
    f.blocks
        .get_mut(&f.entry_block)
        .unwrap()
        .instructions
        .push(HirInstruction::Binary {
            result: quotient,
            op: BinaryOp::FDiv,
            ty: ty.clone(),
            left: x,
            right: divisor,
        });
    let result = if add {
        let y = f.create_value(
            ty.clone(),
            HirValueKind::Constant(HirConstant::F64(-f64::from_bits(2))),
        );
        let sum = f.create_value(ty.clone(), HirValueKind::Instruction);
        f.blocks
            .get_mut(&f.entry_block)
            .unwrap()
            .instructions
            .push(HirInstruction::Binary {
                result: sum,
                op: BinaryOp::FAdd,
                ty,
                left: quotient,
                right: y,
            });
        sum
    } else {
        quotient
    };
    f.blocks.get_mut(&f.entry_block).unwrap().terminator = HirTerminator::Return {
        values: vec![result],
    };
    let id = f.id;
    let mut m = HirModule::new(InternedString::new_global("divide"));
    m.functions.insert(id, f);
    (m, id)
}
fn equal(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
fn check64(m: HirModule, id: HirId, samples: &[(f64, f64)]) {
    let mut interp = HirInterpreter::new();
    let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    clif.compile_module(&m).unwrap();
    clif.finalize_definitions().unwrap();
    // SAFETY: the fixture has a C (f64) -> f64 signature.
    let call: extern "C" fn(f64) -> f64 =
        unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
    #[cfg(feature = "llvm-backend")]
    let context = inkwell::context::Context::create();
    #[cfg(feature = "llvm-backend")]
    let mut llvm = zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
    #[cfg(feature = "llvm-backend")]
    llvm.compile_module(&m).unwrap();
    for &(x, expected) in samples {
        equal(
            value_to_f64(
                &interp
                    .call(&m, "divide", vec![ZyntaxValue::Float(x)])
                    .unwrap(),
            )
            .unwrap(),
            expected,
        );
        equal(call(x), expected);
        #[cfg(feature = "llvm-backend")]
        {
            let call: extern "C" fn(f64) -> f64 =
                unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
            equal(call(x), expected);
        }
    }
}
#[test]
fn f64_division_retains_ieee_results_in_every_tier() {
    let mut bits = vec![
        0,
        1,
        2,
        3,
        1u64 << 63,
        (1u64 << 63) | 1,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0xffef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x7ff0_0000_0000_0001,
    ];
    let mut seed = 0x4d59_5df4_d0f3_3173u64;
    for _ in 0..256 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        bits.push(seed);
    }
    for d in [
        2.0,
        -2.0,
        0.5,
        -0.5,
        1.0,
        -1.0,
        8.0,
        f64::MIN_POSITIVE,
        2.0f64.powi(1022),
        2.0f64.powi(1023),
        f64::from_bits(1),
        3.0,
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NAN,
    ] {
        let (m, id) = fixture(HirConstant::F64(d), HirType::F64, false);
        let samples: Vec<_> = bits
            .iter()
            .map(|b| {
                let x = f64::from_bits(*b);
                (x, x / d)
            })
            .collect();
        check64(m, id, &samples);
    }
}
#[test]
fn f32_division_retains_ieee_results_in_every_tier() {
    for d in [
        2.0f32,
        -2.0,
        0.5,
        f32::MIN_POSITIVE,
        2.0f32.powi(126),
        2.0f32.powi(127),
        f32::from_bits(1),
        3.0,
        0.0,
        f32::INFINITY,
    ] {
        let (m, id) = fixture(HirConstant::F32(d), HirType::F32, false);
        let mut interp = HirInterpreter::new();
        let mut clif = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
        clif.compile_module(&m).unwrap();
        clif.finalize_definitions().unwrap();
        let call: extern "C" fn(f32) -> f32 =
            unsafe { std::mem::transmute(clif.get_function_ptr(id).unwrap()) };
        #[cfg(feature = "llvm-backend")]
        let context = inkwell::context::Context::create();
        #[cfg(feature = "llvm-backend")]
        let mut llvm = zyntax_compiler::llvm_jit_backend::LLVMJitBackend::new(&context).unwrap();
        #[cfg(feature = "llvm-backend")]
        llvm.compile_module(&m).unwrap();
        let mut bits = vec![
            0,
            1,
            2,
            3,
            1u32 << 31,
            (1u32 << 31) | 1,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0001,
            0x7f80_0001,
        ];
        let mut seed = 0x4d59_5df4u32;
        for _ in 0..256 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            bits.push(seed);
        }
        for b in bits {
            let x = f32::from_bits(b);
            let expected = x / d;
            equal(
                value_to_f64(
                    &interp
                        .call(&m, "divide", vec![ZyntaxValue::F32(x)])
                        .unwrap(),
                )
                .unwrap(),
                expected as f64,
            );
            equal(call(x) as f64, expected as f64);
            #[cfg(feature = "llvm-backend")]
            {
                let call: extern "C" fn(f32) -> f32 =
                    unsafe { std::mem::transmute(llvm.get_function_pointer(id).unwrap()) };
                equal(call(x) as f64, expected as f64);
            }
        }
    }
}
#[test]
fn lowering_keeps_division_and_addition_rounding_separate() {
    let (mut m, id) = fixture(HirConstant::F64(2.0), HirType::F64, true);
    zyntax_compiler::run_interp_safe_opts(&mut m);
    zyntax_compiler::run_interp_safe_opts(&mut m);
    assert!(m.functions[&id]
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .any(|i| matches!(
            i,
            HirInstruction::Binary {
                op: BinaryOp::FDiv,
                ..
            }
        )));
    assert!(!m.functions[&id]
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .any(|i| matches!(
            i,
            HirInstruction::Call {
                callee: HirCallable::Intrinsic(Intrinsic::Fma),
                ..
            }
        )));
    check64(m, id, &[(f64::from_bits(3), 0.0)]);
}
