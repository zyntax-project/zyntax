//! Integration tests for data structures (structs and arrays)
//! These tests actually execute the compiled code to verify correctness

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::{arena::AstArena, InternedString};

fn create_test_string(s: &str) -> InternedString {
    let mut arena = AstArena::new();
    arena.intern_string(s)
}

fn compile_and_get_ptr(func: HirFunction) -> (CraneliftBackend, Option<*const u8>) {
    let func_id = func.id;
    let mut backend = CraneliftBackend::new().expect("Failed to create Cranelift backend");

    backend
        .compile_function(func_id, &func)
        .expect("Failed to compile function");

    backend
        .finalize_definitions()
        .expect("Failed to finalize definitions");

    let func_ptr = backend.get_function_ptr(func_id);
    (backend, func_ptr)
}

/// Test struct initialization and field access
#[test]
fn test_struct_roundtrip() {
    // Create function that:
    // 1. Allocates a Point struct on the stack
    // 2. Initializes x = 10, y = 20
    // 3. Reads y and returns it

    let name = create_test_string("struct_roundtrip");
    let point_ty = HirStructType {
        name: Some(create_test_string("Point")),
        fields: vec![HirType::I32, HirType::I32], // x, y
        packed: false,
    };

    let sig = HirFunctionSignature {
        params: vec![],
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

    // Alloca Point
    let struct_ptr = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(point_ty.clone()))),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: struct_ptr,
        ty: HirType::Struct(point_ty.clone()),
        count: None,
        align: 4,
    };

    // Insert x = 10
    let val_10 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(10)));
    let ptr_1 = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(point_ty.clone()))),
        HirValueKind::Instruction,
    );
    let insert_x = HirInstruction::InsertValue {
        result: ptr_1,
        ty: HirType::I32,
        aggregate: struct_ptr,
        value: val_10,
        indices: vec![0],
    };

    // Insert y = 20
    let val_20 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(20)));
    let ptr_2 = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(point_ty.clone()))),
        HirValueKind::Instruction,
    );
    let insert_y = HirInstruction::InsertValue {
        result: ptr_2,
        ty: HirType::I32,
        aggregate: ptr_1,
        value: val_20,
        indices: vec![1],
    };

    // Extract y
    let y_result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract_y = HirInstruction::ExtractValue {
        result: y_result,
        ty: HirType::I32,
        aggregate: ptr_2,
        indices: vec![1],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(insert_x);
    block.add_instruction(insert_y);
    block.add_instruction(extract_y);
    block.set_terminator(HirTerminator::Return {
        values: vec![y_result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    let func_typed: extern "C" fn() -> i32 = unsafe { std::mem::transmute(func_ptr) };

    assert_eq!(func_typed(), 20, "Should return y value (20)");
}

/// Test nested struct access
#[test]
fn test_nested_struct_execution() {
    // Nested struct: Outer { inner: Inner { a, b }, c }
    // Initialize and read inner.b

    let name = create_test_string("nested_struct_test");

    let inner_ty = HirStructType {
        name: Some(create_test_string("Inner")),
        fields: vec![HirType::I32, HirType::I32],
        packed: false,
    };

    let outer_ty = HirStructType {
        name: Some(create_test_string("Outer")),
        fields: vec![HirType::Struct(inner_ty), HirType::I32],
        packed: false,
    };

    let sig = HirFunctionSignature {
        params: vec![],
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

    // Alloca Outer
    let struct_ptr = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(outer_ty.clone()))),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: struct_ptr,
        ty: HirType::Struct(outer_ty.clone()),
        count: None,
        align: 4,
    };

    // Insert outer.inner.a = 5
    let val_5 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(5)));
    let ptr_1 = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(outer_ty.clone()))),
        HirValueKind::Instruction,
    );
    let insert_a = HirInstruction::InsertValue {
        result: ptr_1,
        ty: HirType::I32,
        aggregate: struct_ptr,
        value: val_5,
        indices: vec![0, 0], // inner.a
    };

    // Insert outer.inner.b = 42
    let val_42 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(42)));
    let ptr_2 = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(outer_ty.clone()))),
        HirValueKind::Instruction,
    );
    let insert_b = HirInstruction::InsertValue {
        result: ptr_2,
        ty: HirType::I32,
        aggregate: ptr_1,
        value: val_42,
        indices: vec![0, 1], // inner.b
    };

    // Insert outer.c = 100
    let val_100 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(100)));
    let ptr_3 = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(outer_ty.clone()))),
        HirValueKind::Instruction,
    );
    let insert_c = HirInstruction::InsertValue {
        result: ptr_3,
        ty: HirType::I32,
        aggregate: ptr_2,
        value: val_100,
        indices: vec![1], // c
    };

    // Extract outer.inner.b
    let b_result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract_b = HirInstruction::ExtractValue {
        result: b_result,
        ty: HirType::I32,
        aggregate: ptr_3,
        indices: vec![0, 1],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(insert_a);
    block.add_instruction(insert_b);
    block.add_instruction(insert_c);
    block.add_instruction(extract_b);
    block.set_terminator(HirTerminator::Return {
        values: vec![b_result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    let func_typed: extern "C" fn() -> i32 = unsafe { std::mem::transmute(func_ptr) };

    assert_eq!(func_typed(), 42, "Should return inner.b value (42)");
}

/// Test array initialization and access
#[test]
fn test_array_roundtrip() {
    // Create array [i32; 4], initialize elements, read element 2

    let name = create_test_string("array_roundtrip");
    let array_ty = HirType::Array(Box::new(HirType::I32), 4);

    let sig = HirFunctionSignature {
        params: vec![],
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

    // Alloca array
    let array_ptr = func.create_value(
        HirType::Ptr(Box::new(array_ty.clone())),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: array_ptr,
        ty: array_ty.clone(),
        count: None,
        align: 4,
    };

    // Initialize array elements
    let mut current_ptr = array_ptr;
    for i in 0..4 {
        let value = func.create_value(
            HirType::I32,
            HirValueKind::Constant(HirConstant::I32((i * 10) as i32)),
        );
        let new_ptr = func.create_value(
            HirType::Ptr(Box::new(array_ty.clone())),
            HirValueKind::Instruction,
        );
        let insert = HirInstruction::InsertValue {
            result: new_ptr,
            ty: HirType::I32,
            aggregate: current_ptr,
            value,
            indices: vec![i as u32],
        };

        let block = func.blocks.get_mut(&entry_block_id).unwrap();
        block.add_instruction(insert);

        current_ptr = new_ptr;
    }

    // Extract element 2 (value should be 20)
    let elem_result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: elem_result,
        ty: HirType::I32,
        aggregate: current_ptr,
        indices: vec![2],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.instructions.insert(0, alloca); // Add alloca at beginning
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return {
        values: vec![elem_result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    let func_typed: extern "C" fn() -> i32 = unsafe { std::mem::transmute(func_ptr) };

    assert_eq!(func_typed(), 20, "Should return arr[2] value (20)");
}

/// Test 2D array access
#[test]
fn test_2d_array_execution() {
    // Create [[i32; 3]; 2], initialize and read [1][2]

    let name = create_test_string("test_2d_array");
    let inner_array_ty = HirType::Array(Box::new(HirType::I32), 3);
    let outer_array_ty = HirType::Array(Box::new(inner_array_ty), 2);

    let sig = HirFunctionSignature {
        params: vec![],
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

    // Alloca 2D array
    let array_ptr = func.create_value(
        HirType::Ptr(Box::new(outer_array_ty.clone())),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: array_ptr,
        ty: outer_array_ty.clone(),
        count: None,
        align: 4,
    };

    // Insert arr[1][2] = 99
    let val_99 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(99)));
    let ptr_1 = func.create_value(
        HirType::Ptr(Box::new(outer_array_ty.clone())),
        HirValueKind::Instruction,
    );
    let insert = HirInstruction::InsertValue {
        result: ptr_1,
        ty: HirType::I32,
        aggregate: array_ptr,
        value: val_99,
        indices: vec![1, 2],
    };

    // Extract arr[1][2]
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result,
        ty: HirType::I32,
        aggregate: ptr_1,
        indices: vec![1, 2],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(insert);
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    let func_typed: extern "C" fn() -> i32 = unsafe { std::mem::transmute(func_ptr) };

    assert_eq!(func_typed(), 99, "Should return arr[1][2] value (99)");
}

/// Test array of structs
#[test]
fn test_array_of_structs_execution() {
    // Create array of Points, initialize points[1].y = 77, read it back

    let name = create_test_string("array_of_structs");

    let point_ty = HirStructType {
        name: Some(create_test_string("Point")),
        fields: vec![HirType::I32, HirType::I32],
        packed: false,
    };

    let array_ty = HirType::Array(Box::new(HirType::Struct(point_ty)), 3);

    let sig = HirFunctionSignature {
        params: vec![],
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

    // Alloca array
    let array_ptr = func.create_value(
        HirType::Ptr(Box::new(array_ty.clone())),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: array_ptr,
        ty: array_ty.clone(),
        count: None,
        align: 4,
    };

    // Insert points[1].y = 77
    let val_77 = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(77)));
    let ptr_1 = func.create_value(
        HirType::Ptr(Box::new(array_ty.clone())),
        HirValueKind::Instruction,
    );
    let insert = HirInstruction::InsertValue {
        result: ptr_1,
        ty: HirType::I32,
        aggregate: array_ptr,
        value: val_77,
        indices: vec![1, 1], // array index 1, struct field 1
    };

    // Extract points[1].y
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result,
        ty: HirType::I32,
        aggregate: ptr_1,
        indices: vec![1, 1],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(insert);
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    let func_typed: extern "C" fn() -> i32 = unsafe { std::mem::transmute(func_ptr) };

    assert_eq!(func_typed(), 77, "Should return points[1].y value (77)");
}

/// Test struct with different field sizes
#[test]
fn test_mixed_size_struct() {
    // Struct with i8, i32, i64 - tests alignment

    let name = create_test_string("mixed_size_struct");

    let struct_ty = HirStructType {
        name: Some(create_test_string("Mixed")),
        fields: vec![HirType::I8, HirType::I32, HirType::I64],
        packed: false,
    };

    let sig = HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::I64],
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

    // Alloca struct
    let struct_ptr = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(struct_ty.clone()))),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: struct_ptr,
        ty: HirType::Struct(struct_ty.clone()),
        count: None,
        align: 8,
    };

    // Insert field 2 (i64) = 123456789
    let val = func.create_value(
        HirType::I64,
        HirValueKind::Constant(HirConstant::I64(123456789)),
    );
    let ptr_1 = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(struct_ty.clone()))),
        HirValueKind::Instruction,
    );
    let insert = HirInstruction::InsertValue {
        result: ptr_1,
        ty: HirType::I64,
        aggregate: struct_ptr,
        value: val,
        indices: vec![2],
    };

    // Extract field 2
    let result = func.create_value(HirType::I64, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result,
        ty: HirType::I64,
        aggregate: ptr_1,
        indices: vec![2],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(insert);
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    let func_typed: extern "C" fn() -> i64 = unsafe { std::mem::transmute(func_ptr) };

    assert_eq!(func_typed(), 123456789, "Should return i64 field value");
}
