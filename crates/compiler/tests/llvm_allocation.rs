#![cfg(feature = "llvm-backend")]

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use zyntax_compiler::hir::{
    HirCallable, HirConstant, HirFunction, HirFunctionSignature, HirInstruction, HirModule,
    HirParam, HirTerminator, HirType, HirValueKind, Intrinsic,
};
use zyntax_compiler::llvm_backend::LLVMBackend;
use zyntax_typed_ast::InternedString;

fn function(name: &str, param_ty: HirType) -> HirFunction {
    let mut f = HirFunction::new(
        InternedString::new_global(name),
        HirFunctionSignature {
            params: vec![],
            returns: vec![HirType::I64],
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            is_fiber: false,
            effects: vec![],
            is_pure: false,
        },
    );
    let param = f.create_value(param_ty.clone(), HirValueKind::Parameter(0));
    f.signature.params.push(HirParam {
        id: param,
        name: InternedString::new_global("value"),
        ty: param_ty,
        attributes: Default::default(),
        ownership: Default::default(),
    });
    f
}

#[test]
fn llvm_eliminates_an_unobserved_pool_allocation() {
    let mut f = function("scalar", HirType::I64);
    let value = f.signature.params[0].id;
    let size = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(8)));
    let ptr_ty = HirType::Ptr(Box::new(HirType::I64));
    let ptr = f.create_value(ptr_ty.clone(), HirValueKind::Instruction);
    let loaded = f.create_value(HirType::I64, HirValueKind::Instruction);
    let block = f.blocks.get_mut(&f.entry_block).unwrap();
    block.instructions = vec![
        HirInstruction::Call {
            result: Some(ptr),
            callee: HirCallable::Intrinsic(Intrinsic::Malloc),
            args: vec![size],
            type_args: vec![],
            const_args: vec![],
            is_tail: false,
        },
        HirInstruction::Store {
            value,
            ptr,
            align: 8,
            volatile: false,
        },
        HirInstruction::Load {
            result: loaded,
            ty: HirType::I64,
            ptr,
            align: 8,
            volatile: false,
        },
    ];
    block.terminator = HirTerminator::Return {
        values: vec![loaded],
    };
    let mut module = HirModule::new(InternedString::new_global("alloc"));
    module.functions.insert(f.id, f);
    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "alloc");
    let before = backend.compile_module(&module).unwrap();
    backend.module().verify().unwrap();
    assert!(before.contains("allocsize(0)"), "{before}");
    assert!(
        before.contains("allockind(\"alloc,uninitialized\")"),
        "{before}"
    );
    assert!(
        before.contains("declare noalias ptr @zyntax_alloc"),
        "{before}"
    );
    assert!(!before.contains("nofree"), "{before}");

    Target::initialize_native(&InitializationConfig::default()).unwrap();
    let triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&triple).unwrap();
    let machine = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .unwrap();
    backend
        .module()
        .run_passes("default<O3>", &machine, PassBuilderOptions::create())
        .unwrap();
    backend.module().verify().unwrap();
    let after = backend.module().print_to_string().to_string();
    assert!(
        !after
            .lines()
            .any(|l| l.contains("call") && l.contains("@zyntax_alloc")),
        "{after}"
    );
    assert!(after.contains("ret i64 %param_0"), "{after}");
}

#[test]
fn releasing_an_argument_is_not_declared_nofree() {
    let mut f = function("release", HirType::Ptr(Box::new(HirType::I64)));
    let ptr = f.signature.params[0].id;
    let zero = f.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(0)));
    let block = f.blocks.get_mut(&f.entry_block).unwrap();
    block.instructions.push(HirInstruction::Call {
        result: None,
        callee: HirCallable::Intrinsic(Intrinsic::Free),
        args: vec![ptr],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    });
    block.terminator = HirTerminator::Return { values: vec![zero] };
    let mut module = HirModule::new(InternedString::new_global("release"));
    module.functions.insert(f.id, f);
    let context = Context::create();
    let mut backend = LLVMBackend::new(&context, "release");
    let ir = backend.compile_module(&module).unwrap();
    backend.module().verify().unwrap();
    assert!(ir.contains("call void @zyntax_free"), "{ir}");
    assert!(!ir.contains("nofree"), "{ir}");
}
