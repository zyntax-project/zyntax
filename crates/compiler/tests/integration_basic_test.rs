//! Integration Tests - Phase 1: Basic Arithmetic
//! These tests validate the ENTIRE compilation pipeline by:
//! 1. Manually building HIR functions
//! 2. Compiling through Cranelift JIT backend
//! 3. Getting function pointers from JIT
//! 4. ACTUALLY EXECUTING the compiled code
//! 5. Verifying results with assertions

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::{arena::AstArena, InternedString};

fn create_test_string(s: &str) -> InternedString {
    let mut arena = AstArena::new();
    arena.intern_string(s)
}

/// Test 1: Simple addition - fn add(a: i32, b: i32) -> i32 { a + b }
#[test]
fn test_simple_addition_execution() {
    let func = create_binary_op_function("add", BinaryOp::Add);

    // Compile through Cranelift
    let (backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for add");

    // Execute the compiled function
    let add_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(add_fn(5, 3), 8, "5 + 3 should equal 8");
    assert_eq!(add_fn(10, -5), 5, "10 + (-5) should equal 5");
    assert_eq!(add_fn(0, 0), 0, "0 + 0 should equal 0");
    assert_eq!(add_fn(-10, -20), -30, "-10 + (-20) should equal -30");
    assert_eq!(add_fn(100, 200), 300, "100 + 200 should equal 300");

    println!("✅ Addition executed correctly via JIT");
}

/// Test 2: Simple subtraction - fn sub(a: i32, b: i32) -> i32 { a - b }
#[test]
fn test_simple_subtraction_execution() {
    let func = create_binary_op_function("sub", BinaryOp::Sub);

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for sub");

    let sub_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(sub_fn(10, 3), 7, "10 - 3 should equal 7");
    assert_eq!(sub_fn(5, 10), -5, "5 - 10 should equal -5");
    assert_eq!(sub_fn(0, 0), 0, "0 - 0 should equal 0");
    assert_eq!(sub_fn(-10, 5), -15, "-10 - 5 should equal -15");
    assert_eq!(sub_fn(100, 100), 0, "100 - 100 should equal 0");

    println!("✅ Subtraction executed correctly via JIT");
}

/// Test 3: Simple multiplication - fn mul(a: i32, b: i32) -> i32 { a * b }
#[test]
fn test_simple_multiplication_execution() {
    let func = create_binary_op_function("mul", BinaryOp::Mul);

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for mul");

    let mul_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(mul_fn(5, 3), 15, "5 * 3 should equal 15");
    assert_eq!(mul_fn(10, -2), -20, "10 * (-2) should equal -20");
    assert_eq!(mul_fn(0, 100), 0, "0 * 100 should equal 0");
    assert_eq!(mul_fn(-5, -5), 25, "-5 * (-5) should equal 25");
    assert_eq!(mul_fn(7, 7), 49, "7 * 7 should equal 49");

    println!("✅ Multiplication executed correctly via JIT");
}

/// Test 4: Simple division - fn div(a: i32, b: i32) -> i32 { a / b }
#[test]
fn test_simple_division_execution() {
    let func = create_binary_op_function("div", BinaryOp::Div);

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for div");

    let div_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(div_fn(10, 2), 5, "10 / 2 should equal 5");
    assert_eq!(div_fn(15, 3), 5, "15 / 3 should equal 5");
    assert_eq!(div_fn(7, 2), 3, "7 / 2 should equal 3 (integer division)");
    assert_eq!(div_fn(-10, 2), -5, "-10 / 2 should equal -5");
    assert_eq!(div_fn(100, 10), 10, "100 / 10 should equal 10");

    println!("✅ Division executed correctly via JIT");
}

/// Test 5: Float addition - fn fadd(a: f64, b: f64) -> f64 { a + b }
#[test]
fn test_float_addition_execution() {
    let func = create_float_binary_op_function("fadd", BinaryOp::FAdd);

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for fadd");

    let fadd_fn: extern "C" fn(f64, f64) -> f64 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert!(
        (fadd_fn(5.5, 3.2) - 8.7).abs() < 1e-10,
        "5.5 + 3.2 should equal 8.7"
    );
    assert!(
        (fadd_fn(10.0, -5.0) - 5.0).abs() < 1e-10,
        "10.0 + (-5.0) should equal 5.0"
    );
    assert!(
        (fadd_fn(0.0, 0.0) - 0.0).abs() < 1e-10,
        "0.0 + 0.0 should equal 0.0"
    );
    assert!(
        (fadd_fn(-10.5, -20.5) - (-31.0)).abs() < 1e-10,
        "-10.5 + (-20.5) should equal -31.0"
    );

    println!("✅ Float addition executed correctly via JIT");
}

/// Test 6: Comparison operations - fn eq(a: i32, b: i32) -> i32 { a == b }
#[test]
fn test_comparison_execution() {
    let func = create_binary_op_function("eq", BinaryOp::Eq);

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for eq");

    let eq_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results (Cranelift returns 1 for true, 0 for false)
    assert_eq!(eq_fn(5, 5), 1, "5 == 5 should be true");
    assert_eq!(eq_fn(5, 3), 0, "5 == 3 should be false");
    assert_eq!(eq_fn(0, 0), 1, "0 == 0 should be true");
    assert_eq!(eq_fn(-10, -10), 1, "-10 == -10 should be true");
    assert_eq!(eq_fn(100, 200), 0, "100 == 200 should be false");

    println!("✅ Comparison executed correctly via JIT");
}

// =============================================================================
// Helper Functions for Building HIR
// =============================================================================

/// Compiles a function and returns the backend with the function pointer
fn compile_and_get_ptr(func: HirFunction) -> (CraneliftBackend, Option<*const u8>) {
    use cranelift_module::Module;

    let func_id = func.id;
    let mut backend = CraneliftBackend::new().expect("Failed to create Cranelift backend");
    backend
        .compile_function(func_id, &func)
        .expect("Failed to compile function");

    // Manually finalize and get function pointer
    // This is what compile_module does internally
    backend
        .finalize_definitions()
        .expect("Failed to finalize definitions");

    let func_ptr = backend.get_function_ptr(func_id);
    (backend, func_ptr)
}

/// Creates a binary operation function: fn op(a: i32, b: i32) -> i32 { a op b }
fn create_binary_op_function(name: &str, op: BinaryOp) -> HirFunction {
    let name = create_test_string(name);

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

    // Create parameter values
    let param_a = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let param_b = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    // Create result value
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);

    // Create binary operation instruction
    let bin_inst = HirInstruction::Binary {
        op,
        result,
        ty: HirType::I32,
        left: param_a,
        right: param_b,
    };

    // Add instruction to entry block
    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(bin_inst);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Creates a float binary operation function: fn op(a: f64, b: f64) -> f64 { a op b }
fn create_float_binary_op_function(name: &str, op: BinaryOp) -> HirFunction {
    let name = create_test_string(name);

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("a"),
                ty: HirType::F64,
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("b"),
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

    let param_a = func.create_value(HirType::F64, HirValueKind::Parameter(0));
    let param_b = func.create_value(HirType::F64, HirValueKind::Parameter(1));

    let result = func.create_value(HirType::F64, HirValueKind::Instruction);

    let bin_inst = HirInstruction::Binary {
        op,
        result,
        ty: HirType::F64,
        left: param_a,
        right: param_b,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(bin_inst);
    block.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}
