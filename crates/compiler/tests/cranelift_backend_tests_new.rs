//! Additional Cranelift Backend Tests
//! Testing new features: indirect calls, multi-level extract/insert, math intrinsics, etc.

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::{arena::AstArena, InternedString};

fn create_test_string(s: &str) -> InternedString {
    let mut arena = AstArena::new();
    arena.intern_string(s)
}

/// Test indirect function calls (function pointers)
#[test]
fn test_indirect_function_call() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    // First compile a simple add function
    let add_func = create_simple_add_function();
    let add_func_id = add_func.id;
    backend
        .compile_function(add_func_id, &add_func)
        .expect("Failed to compile add function");

    // Now create a function that makes an indirect call
    let indirect_caller = create_indirect_call_function();
    backend
        .compile_function(indirect_caller.id, &indirect_caller)
        .expect("Failed to compile indirect call function");

    println!("✅ Successfully compiled indirect function call");
}

fn create_simple_add_function() -> HirFunction {
    let name = create_test_string("add");

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("a"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("b"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    let entry_block_id = func.entry_block;

    let param_a = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let param_b = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let add_inst = HirInstruction::Binary {
        op: BinaryOp::Add,
        result,
        ty: HirType::I32,
        left: param_a,
        right: param_b,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(add_inst);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

fn create_indirect_call_function() -> HirFunction {
    let name = create_test_string("call_indirect");

    let func_ty = HirType::Function(Box::new(HirFunctionType {
        params: vec![HirType::I32, HirType::I32],
        returns: vec![HirType::I32],
        lifetime_params: vec![],
        is_variadic: false,
    }));

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("func_ptr"),
                ty: func_ty.clone(),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("a"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("b"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    let entry_block_id = func.entry_block;

    let param_func_ptr = func.create_value(func_ty, HirValueKind::Parameter(0));
    let param_a = func.create_value(HirType::I32, HirValueKind::Parameter(1));
    let param_b = func.create_value(HirType::I32, HirValueKind::Parameter(2));

    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let indirect_call = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Indirect(param_func_ptr),
        args: vec![param_a, param_b],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(indirect_call);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Test multi-level ExtractValue
#[test]
fn test_multi_level_extract_value() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_multi_level_extract_function();

    backend
        .compile_function(func.id, &func)
        .expect("Failed to compile multi-level ExtractValue");

    println!("✅ Successfully compiled multi-level ExtractValue");
}

fn create_multi_level_extract_function() -> HirFunction {
    let name = create_test_string("extract_nested_field");

    let inner_ty = HirType::Struct(HirStructType {
        name: Some(create_test_string("Inner")),
        fields: vec![HirType::I32, HirType::I32],
        packed: false,
    });

    let outer_ty = HirType::Struct(HirStructType {
        name: Some(create_test_string("Outer")),
        fields: vec![inner_ty.clone(), HirType::I32],
        packed: false,
    });

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("s"),
            ty: HirType::Ptr(Box::new(outer_ty.clone())),
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    let entry_block_id = func.entry_block;

    let param_s = func.create_value(HirType::Ptr(Box::new(outer_ty)), HirValueKind::Parameter(0));

    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract_inst = HirInstruction::ExtractValue {
        result,
        ty: HirType::I32,
        aggregate: param_s,
        indices: vec![0, 1], // outer.inner.y
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract_inst);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Test multi-level InsertValue
#[test]
fn test_multi_level_insert_value() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_multi_level_insert_function();

    backend
        .compile_function(func.id, &func)
        .expect("Failed to compile multi-level InsertValue");

    println!("✅ Successfully compiled multi-level InsertValue");
}

fn create_multi_level_insert_function() -> HirFunction {
    let name = create_test_string("insert_nested_field");

    let inner_ty = HirType::Struct(HirStructType {
        name: Some(create_test_string("Inner")),
        fields: vec![HirType::I32, HirType::I32],
        packed: false,
    });

    let outer_ty = HirType::Struct(HirStructType {
        name: Some(create_test_string("Outer")),
        fields: vec![inner_ty.clone(), HirType::I32],
        packed: false,
    });

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("s"),
                ty: HirType::Ptr(Box::new(outer_ty.clone())),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("val"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(outer_ty.clone()))],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    let entry_block_id = func.entry_block;

    let param_s = func.create_value(
        HirType::Ptr(Box::new(outer_ty.clone())),
        HirValueKind::Parameter(0),
    );
    let param_val = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    let result = func.create_value(HirType::Ptr(Box::new(outer_ty)), HirValueKind::Instruction);
    let insert_inst = HirInstruction::InsertValue {
        result,
        ty: HirType::I32,
        aggregate: param_s,
        value: param_val,
        indices: vec![0, 1], // outer.inner.y
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(insert_inst);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Test math intrinsics
#[test]
fn test_math_intrinsics() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let sin_func = create_math_intrinsic_function("test_sin", Intrinsic::Sin);
    backend
        .compile_function(sin_func.id, &sin_func)
        .expect("Failed to compile sin");

    let cos_func = create_math_intrinsic_function("test_cos", Intrinsic::Cos);
    backend
        .compile_function(cos_func.id, &cos_func)
        .expect("Failed to compile cos");

    let log_func = create_math_intrinsic_function("test_log", Intrinsic::Log);
    backend
        .compile_function(log_func.id, &log_func)
        .expect("Failed to compile log");

    let exp_func = create_math_intrinsic_function("test_exp", Intrinsic::Exp);
    backend
        .compile_function(exp_func.id, &exp_func)
        .expect("Failed to compile exp");

    println!("✅ Successfully compiled all math intrinsics");
}

fn create_math_intrinsic_function(name: &str, intrinsic: Intrinsic) -> HirFunction {
    let name = create_test_string(name);

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("x"),
            ty: HirType::F64,
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::F64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    let entry_block_id = func.entry_block;

    let param_x = func.create_value(HirType::F64, HirValueKind::Parameter(0));

    let result = func.create_value(HirType::F64, HirValueKind::Instruction);
    let intrinsic_call = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(intrinsic),
        args: vec![param_x],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(intrinsic_call);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Test pow intrinsic
#[test]
fn test_pow_intrinsic() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let pow_func = create_pow_intrinsic_function();

    backend
        .compile_function(pow_func.id, &pow_func)
        .expect("Failed to compile pow");

    println!("✅ Successfully compiled pow intrinsic");
}

fn create_pow_intrinsic_function() -> HirFunction {
    let name = create_test_string("test_pow");

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("base"),
                ty: HirType::F64,
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("exp"),
                ty: HirType::F64,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::F64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    let entry_block_id = func.entry_block;

    let param_base = func.create_value(HirType::F64, HirValueKind::Parameter(0));
    let param_exp = func.create_value(HirType::F64, HirValueKind::Parameter(1));

    let result = func.create_value(HirType::F64, HirValueKind::Instruction);
    let pow_call = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(Intrinsic::Pow),
        args: vec![param_base, param_exp],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(pow_call);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Test bit manipulation intrinsics
#[test]
fn test_bit_manipulation_intrinsics() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let ctlz_func = create_bit_intrinsic_function("test_ctlz", Intrinsic::Ctlz);
    backend
        .compile_function(ctlz_func.id, &ctlz_func)
        .expect("Failed to compile ctlz");

    let cttz_func = create_bit_intrinsic_function("test_cttz", Intrinsic::Cttz);
    backend
        .compile_function(cttz_func.id, &cttz_func)
        .expect("Failed to compile cttz");

    let bswap_func = create_bit_intrinsic_function("test_bswap", Intrinsic::Bswap);
    backend
        .compile_function(bswap_func.id, &bswap_func)
        .expect("Failed to compile bswap");

    println!("✅ Successfully compiled all bit manipulation intrinsics");
}

fn create_bit_intrinsic_function(name: &str, intrinsic: Intrinsic) -> HirFunction {
    let name = create_test_string(name);

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("x"),
            ty: HirType::I32,
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    let entry_block_id = func.entry_block;

    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));

    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let intrinsic_call = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(intrinsic),
        args: vec![param_x],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(intrinsic_call);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}
