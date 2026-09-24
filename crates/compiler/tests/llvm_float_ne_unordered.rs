//! LLVM backend: float `!=` is an unordered compare, so `x != x` holds
//! for NaN, as Cranelift's `FloatCC::NotEqual` does.
#![cfg(feature = "llvm-backend")]

use inkwell::context::Context;
use zyntax_compiler::hir::{
    BinaryOp, HirFunction, HirFunctionSignature, HirId, HirInstruction, HirModule, HirParam,
    HirTerminator, HirType, HirValueKind, ParamAttributes,
};
use zyntax_compiler::llvm_backend::LLVMBackend;
use zyntax_typed_ast::InternedString;

/// `fn f(x: f64) -> bool { x <op> x }` compiled to LLVM IR.
fn self_compare_ir(op: BinaryOp) -> String {
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: InternedString::new_global("x"),
            ty: HirType::F64,
            attributes: ParamAttributes::default(),
            ownership: Default::default(),
        }],
        returns: vec![HirType::Bool],
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
    let result = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let entry = func.entry_block;
    let block = func.blocks.get_mut(&entry).unwrap();
    block.add_instruction(HirInstruction::Binary {
        op,
        result,
        ty: HirType::Bool,
        left: x,
        right: x,
    });
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let mut module = HirModule::new(InternedString::new_global("m"));
    module.functions.insert(func.id, func);

    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "float_ne");
    backend
        .compile_module(&module)
        .expect("float != should compile on LLVM")
}

#[test]
fn llvm_float_ne_is_unordered() {
    for op in [BinaryOp::Ne, BinaryOp::FNe] {
        let ir = self_compare_ir(op);
        assert!(
            ir.contains("fcmp une"),
            "{op:?} on floats must be `fcmp une`:\n{ir}"
        );
        assert!(
            !ir.contains("fcmp one"),
            "{op:?} on floats must not be ordered:\n{ir}"
        );
    }
}
