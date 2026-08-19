//! LLVM backend: a `float + int` binary must reconcile the int operand
//! (int -> sitofp) before `fadd`, else `into_float_value()` panics / the IR is
//! invalid. Mirrors the Cranelift `reconcile_binary_operands` coverage.
#![cfg(feature = "llvm-backend")]

use inkwell::context::Context;
use zyntax_compiler::hir::{
    BinaryOp, HirFunction, HirFunctionSignature, HirId, HirInstruction, HirModule, HirParam,
    HirTerminator, HirType, HirValueKind, ParamAttributes,
};
use zyntax_compiler::llvm_backend::LLVMBackend;
use zyntax_typed_ast::InternedString;

#[test]
fn llvm_float_plus_int_emits_sitofp() {
    // fn f(x: f64, i: i64) -> f64 { x + i }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: InternedString::new_global("x"),
                ty: HirType::F64,
                attributes: ParamAttributes::default(),
                ownership: Default::default(),
            },
            HirParam {
                id: HirId::new(),
                name: InternedString::new_global("i"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
                ownership: Default::default(),
            },
        ],
        returns: vec![HirType::F64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };
    let mut func = HirFunction::new(InternedString::new_global("f"), sig);
    let x = func.create_value(HirType::F64, HirValueKind::Parameter(0));
    let i = func.create_value(HirType::I64, HirValueKind::Parameter(1));
    let result = func.create_value(HirType::F64, HirValueKind::Instruction);
    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::Binary {
        op: BinaryOp::FAdd,
        result,
        ty: HirType::F64,
        left: x,
        right: i,
    });
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let mut module = HirModule::new(InternedString::new_global("m"));
    module.functions.insert(func.id, func);

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "float_int");
    let ir = backend
        .compile_module(&module)
        .expect("float + int should compile on LLVM");
    eprintln!("=== LLVM IR ===\n{ir}");
    assert!(
        ir.contains("sitofp"),
        "int operand must be widened to float (sitofp) before fadd:\n{ir}"
    );
    assert!(ir.contains("fadd"), "must emit fadd:\n{ir}");
}
