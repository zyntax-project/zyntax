#![cfg(feature = "cranelift-backend")]

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::{
    CallingConvention, HirConstant, HirFunction, HirFunctionSignature, HirId, HirInstruction,
    HirParam, HirTerminator, HirType, HirValueKind, ParamAttributes,
};
use zyntax_typed_ast::{arena::AstArena, InternedString};

/// Helper to create an interned string for tests
fn create_test_string(s: &str) -> InternedString {
    let mut arena = AstArena::new();
    arena.intern_string(s)
}

/// Compile a HIR function and return the backend + function pointer
fn compile_and_get_ptr(func: HirFunction) -> (CraneliftBackend, Option<*const u8>) {
    let func_id = func.id;
    let mut backend = CraneliftBackend::new().expect("Failed to create Cranelift backend");
    backend
        .compile_function(func_id, &func)
        .expect("Failed to compile function");

    // Manually finalize and get function pointer
    backend
        .finalize_definitions()
        .expect("Failed to finalize definitions");

    let func_ptr = backend.get_function_ptr(func_id);
    (backend, func_ptr)
}

/// Test that Alloca + Store + Load works correctly
/// fn test_memory_roundtrip(x: i32) -> i32 {
///     let ptr = alloca i32
///     store x, ptr
///     let result = load ptr
///     return result
/// }
#[test]
fn test_memory_roundtrip_execution() {
    let name = create_test_string("test_memory_roundtrip");

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("x"),
            ty: HirType::I32,
            attributes: ParamAttributes::default(),
            ownership: Default::default(),
        }],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    // Match the platform `extern "C"` ABI — `HirFunction::new` defaults
    // to `Fast`, which diverges from MS x64 on Windows.
    func.calling_convention = CallingConvention::C;
    let entry_block_id = func.entry_block;
    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));

    // Alloca i32
    let ptr = func.create_value(
        HirType::Ptr(Box::new(HirType::I32)),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: ptr,
        ty: HirType::I32,
        count: None,
        align: 4,
    };

    // Store x to ptr
    let store = HirInstruction::Store {
        value: param_x,
        ptr,
        align: 4,
        volatile: false,
    };

    // Load from ptr
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let load = HirInstruction::Load {
        result,
        ty: HirType::I32,
        ptr,
        align: 4,
        volatile: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(store);
    block.add_instruction(load);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    // Execute the function
    let func_typed: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Test with various values
    assert_eq!(func_typed(42), 42);
    assert_eq!(func_typed(0), 0);
    assert_eq!(func_typed(-100), -100);
    assert_eq!(func_typed(i32::MAX), i32::MAX);
    assert_eq!(func_typed(i32::MIN), i32::MIN);
}

/// Test that we can store and load i64 values
/// fn test_memory_i64(x: i64) -> i64 {
///     let ptr = alloca i64
///     store x, ptr
///     let result = load ptr
///     return result
/// }
#[test]
fn test_memory_i64_execution() {
    let name = create_test_string("test_memory_i64");

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("x"),
            ty: HirType::I64,
            attributes: ParamAttributes::default(),
            ownership: Default::default(),
        }],
        returns: vec![HirType::I64],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    // Match the platform `extern "C"` ABI — `HirFunction::new` defaults
    // to `Fast`, which diverges from MS x64 on Windows.
    func.calling_convention = CallingConvention::C;
    let entry_block_id = func.entry_block;
    let param_x = func.create_value(HirType::I64, HirValueKind::Parameter(0));

    // Alloca i64
    let ptr = func.create_value(
        HirType::Ptr(Box::new(HirType::I64)),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: ptr,
        ty: HirType::I64,
        count: None,
        align: 8,
    };

    // Store x to ptr
    let store = HirInstruction::Store {
        value: param_x,
        ptr,
        align: 8,
        volatile: false,
    };

    // Load from ptr
    let result = func.create_value(HirType::I64, HirValueKind::Instruction);
    let load = HirInstruction::Load {
        result,
        ty: HirType::I64,
        ptr,
        align: 8,
        volatile: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(store);
    block.add_instruction(load);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    // Execute the function
    let func_typed: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(func_ptr) };

    // Test with various values
    assert_eq!(func_typed(12345678901234), 12345678901234);
    assert_eq!(func_typed(0), 0);
    assert_eq!(func_typed(-99999999999), -99999999999);
    assert_eq!(func_typed(i64::MAX), i64::MAX);
    assert_eq!(func_typed(i64::MIN), i64::MIN);
}

/// Test that we can store and load float values
/// fn test_memory_f64(x: f64) -> f64 {
///     let ptr = alloca f64
///     store x, ptr
///     let result = load ptr
///     return result
/// }
#[test]
fn test_memory_f64_execution() {
    let name = create_test_string("test_memory_f64");

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("x"),
            ty: HirType::F64,
            attributes: ParamAttributes::default(),
            ownership: Default::default(),
        }],
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

    let mut func = HirFunction::new(name, sig);
    // Match the platform `extern "C"` ABI — `HirFunction::new` defaults
    // to `Fast`, which diverges from MS x64 on Windows.
    func.calling_convention = CallingConvention::C;
    let entry_block_id = func.entry_block;
    let param_x = func.create_value(HirType::F64, HirValueKind::Parameter(0));

    // Alloca f64
    let ptr = func.create_value(
        HirType::Ptr(Box::new(HirType::F64)),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: ptr,
        ty: HirType::F64,
        count: None,
        align: 8,
    };

    // Store x to ptr
    let store = HirInstruction::Store {
        value: param_x,
        ptr,
        align: 8,
        volatile: false,
    };

    // Load from ptr
    let result = func.create_value(HirType::F64, HirValueKind::Instruction);
    let load = HirInstruction::Load {
        result,
        ty: HirType::F64,
        ptr,
        align: 8,
        volatile: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(store);
    block.add_instruction(load);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    // Execute the function
    let func_typed: extern "C" fn(f64) -> f64 = unsafe { std::mem::transmute(func_ptr) };

    // Test with various values
    assert_eq!(func_typed(std::f64::consts::PI), std::f64::consts::PI);
    assert_eq!(func_typed(0.0), 0.0);
    assert_eq!(func_typed(-123.456), -123.456);
    assert_eq!(func_typed(f64::MAX), f64::MAX);
    assert_eq!(func_typed(f64::MIN), f64::MIN);
}

/// Test multiple allocations
/// fn test_two_variables(a: i32, b: i32) -> i32 {
///     let ptr_a = alloca i32
///     let ptr_b = alloca i32
///     store a, ptr_a
///     store b, ptr_b
///     let val_a = load ptr_a
///     let val_b = load ptr_b
///     return val_a + val_b
/// }
#[test]
fn test_multiple_allocations_execution() {
    let name = create_test_string("test_two_variables");

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("a"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
                ownership: Default::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("b"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
                ownership: Default::default(),
            },
        ],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    // Match the platform `extern "C"` ABI — `HirFunction::new` defaults
    // to `Fast`, which diverges from MS x64 on Windows.
    func.calling_convention = CallingConvention::C;
    let entry_block_id = func.entry_block;
    let param_a = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let param_b = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    // Alloca for a
    let ptr_a = func.create_value(
        HirType::Ptr(Box::new(HirType::I32)),
        HirValueKind::Instruction,
    );
    let alloca_a = HirInstruction::Alloca {
        result: ptr_a,
        ty: HirType::I32,
        count: None,
        align: 4,
    };

    // Alloca for b
    let ptr_b = func.create_value(
        HirType::Ptr(Box::new(HirType::I32)),
        HirValueKind::Instruction,
    );
    let alloca_b = HirInstruction::Alloca {
        result: ptr_b,
        ty: HirType::I32,
        count: None,
        align: 4,
    };

    // Store a to ptr_a
    let store_a = HirInstruction::Store {
        value: param_a,
        ptr: ptr_a,
        align: 4,
        volatile: false,
    };

    // Store b to ptr_b
    let store_b = HirInstruction::Store {
        value: param_b,
        ptr: ptr_b,
        align: 4,
        volatile: false,
    };

    // Load from ptr_a
    let val_a = func.create_value(HirType::I32, HirValueKind::Instruction);
    let load_a = HirInstruction::Load {
        result: val_a,
        ty: HirType::I32,
        ptr: ptr_a,
        align: 4,
        volatile: false,
    };

    // Load from ptr_b
    let val_b = func.create_value(HirType::I32, HirValueKind::Instruction);
    let load_b = HirInstruction::Load {
        result: val_b,
        ty: HirType::I32,
        ptr: ptr_b,
        align: 4,
        volatile: false,
    };

    // Add val_a + val_b
    let sum = func.create_value(HirType::I32, HirValueKind::Instruction);
    let add = HirInstruction::Binary {
        result: sum,
        op: zyntax_compiler::hir::BinaryOp::Add,
        ty: HirType::I32,
        left: val_a,
        right: val_b,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca_a);
    block.add_instruction(alloca_b);
    block.add_instruction(store_a);
    block.add_instruction(store_b);
    block.add_instruction(load_a);
    block.add_instruction(load_b);
    block.add_instruction(add);
    block.set_terminator(HirTerminator::Return { values: vec![sum] });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    // Execute the function
    let func_typed: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Test with various values
    assert_eq!(func_typed(10, 20), 30);
    assert_eq!(func_typed(5, 7), 12);
    assert_eq!(func_typed(-10, 10), 0);
    assert_eq!(func_typed(100, -50), 50);
}

/// Test updating a value in memory
/// fn test_increment(x: i32) -> i32 {
///     let ptr = alloca i32
///     store x, ptr
///     let old_val = load ptr
///     let new_val = old_val + 1
///     store new_val, ptr
///     let result = load ptr
///     return result
/// }
#[test]
fn test_memory_update_execution() {
    let name = create_test_string("test_increment");

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("x"),
            ty: HirType::I32,
            attributes: ParamAttributes::default(),
            ownership: Default::default(),
        }],
        returns: vec![HirType::I32],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        is_fiber: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(name, sig);
    // Match the platform `extern "C"` ABI — `HirFunction::new` defaults
    // to `Fast`, which diverges from MS x64 on Windows.
    func.calling_convention = CallingConvention::C;
    let entry_block_id = func.entry_block;
    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));

    // Alloca i32
    let ptr = func.create_value(
        HirType::Ptr(Box::new(HirType::I32)),
        HirValueKind::Instruction,
    );
    let alloca = HirInstruction::Alloca {
        result: ptr,
        ty: HirType::I32,
        count: None,
        align: 4,
    };

    // Store x to ptr
    let store1 = HirInstruction::Store {
        value: param_x,
        ptr,
        align: 4,
        volatile: false,
    };

    // Load from ptr
    let old_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let load1 = HirInstruction::Load {
        result: old_val,
        ty: HirType::I32,
        ptr,
        align: 4,
        volatile: false,
    };

    // Add 1 to old_val
    let one = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(1)));
    let new_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let add = HirInstruction::Binary {
        result: new_val,
        op: zyntax_compiler::hir::BinaryOp::Add,
        ty: HirType::I32,
        left: old_val,
        right: one,
    };

    // Store new_val to ptr
    let store2 = HirInstruction::Store {
        value: new_val,
        ptr,
        align: 4,
        volatile: false,
    };

    // Load from ptr again
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let load2 = HirInstruction::Load {
        result,
        ty: HirType::I32,
        ptr,
        align: 4,
        volatile: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.add_instruction(store1);
    block.add_instruction(load1);
    block.add_instruction(add);
    block.add_instruction(store2);
    block.add_instruction(load2);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer");

    // Execute the function
    let func_typed: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Test with various values - should return x + 1
    assert_eq!(func_typed(0), 1);
    assert_eq!(func_typed(41), 42);
    assert_eq!(func_typed(-1), 0);
    assert_eq!(func_typed(100), 101);
}
