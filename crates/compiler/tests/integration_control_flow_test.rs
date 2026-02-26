//! Integration Tests - Phase 2: Control Flow
//! These tests validate control flow constructs with actual JIT execution:
//! 1. If/else conditionals
//! 2. While loops
//! 3. Phi nodes (SSA value merging)
//! 4. Early returns
//! 5. Nested control flow

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::{arena::AstArena, InternedString};

fn create_test_string(s: &str) -> InternedString {
    let mut arena = AstArena::new();
    arena.intern_string(s)
}

/// Compiles a function and returns the backend with the function pointer
fn compile_and_get_ptr(func: HirFunction) -> (CraneliftBackend, Option<*const u8>) {
    use cranelift_module::Module;

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

/// Test 1: Simple if/else - fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }
#[test]
fn test_if_else_execution() {
    let func = create_max_function();

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for max");

    let max_fn: extern "C" fn(i32, i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(max_fn(10, 5), 10, "max(10, 5) should equal 10");
    assert_eq!(max_fn(5, 10), 10, "max(5, 10) should equal 10");
    assert_eq!(max_fn(7, 7), 7, "max(7, 7) should equal 7");
    assert_eq!(max_fn(-5, -10), -5, "max(-5, -10) should equal -5");
    assert_eq!(max_fn(0, 0), 0, "max(0, 0) should equal 0");

    println!("✅ If/else control flow executed correctly via JIT");
}

/// Test 2: Absolute value - fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }
#[test]
fn test_abs_execution() {
    let func = create_abs_function();

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for abs");

    let abs_fn: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(abs_fn(10), 10, "abs(10) should equal 10");
    assert_eq!(abs_fn(-10), 10, "abs(-10) should equal 10");
    assert_eq!(abs_fn(0), 0, "abs(0) should equal 0");
    assert_eq!(abs_fn(-100), 100, "abs(-100) should equal 100");
    assert_eq!(abs_fn(42), 42, "abs(42) should equal 42");

    println!("✅ Absolute value with if/else executed correctly via JIT");
}

/// Test 3: Countdown loop - fn countdown(n: i32) -> i32 { while n > 0 { n = n - 1 } return n }
#[test]
fn test_countdown_loop_execution() {
    let func = create_countdown_function();

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for countdown");

    let countdown_fn: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(countdown_fn(0), 0, "countdown(0) should equal 0");
    assert_eq!(countdown_fn(1), 0, "countdown(1) should equal 0");
    assert_eq!(countdown_fn(5), 0, "countdown(5) should equal 0");
    assert_eq!(countdown_fn(10), 0, "countdown(10) should equal 0");
    assert_eq!(
        countdown_fn(-5),
        -5,
        "countdown(-5) should equal -5 (no loop)"
    );

    println!("✅ While loop executed correctly via JIT");
}

/// Test 4: Factorial - fn factorial(n: i32) -> i32 with loop and accumulator
#[test]
fn test_factorial_loop_execution() {
    let func = create_factorial_function();

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for factorial");

    let factorial_fn: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(factorial_fn(0), 1, "factorial(0) should equal 1");
    assert_eq!(factorial_fn(1), 1, "factorial(1) should equal 1");
    assert_eq!(factorial_fn(5), 120, "factorial(5) should equal 120");
    assert_eq!(factorial_fn(6), 720, "factorial(6) should equal 720");
    assert_eq!(factorial_fn(7), 5040, "factorial(7) should equal 5040");

    println!("✅ Factorial loop with accumulator executed correctly via JIT");
}

/// Test 5: Phi nodes - fn sign(x: i32) -> i32 { if x > 0 { 1 } else if x < 0 { -1 } else { 0 } }
#[test]
fn test_phi_nodes_execution() {
    let func = create_sign_function();

    let (_backend, func_ptr) = compile_and_get_ptr(func);
    let func_ptr = func_ptr.expect("Failed to get function pointer for sign");

    let sign_fn: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Verify actual execution results
    assert_eq!(sign_fn(10), 1, "sign(10) should equal 1");
    assert_eq!(sign_fn(-10), -1, "sign(-10) should equal -1");
    assert_eq!(sign_fn(0), 0, "sign(0) should equal 0");
    assert_eq!(sign_fn(100), 1, "sign(100) should equal 1");
    assert_eq!(sign_fn(-100), -1, "sign(-100) should equal -1");

    println!("✅ Phi nodes (SSA merging) executed correctly via JIT");
}

// =============================================================================
// Helper Functions for Building HIR with Control Flow
// =============================================================================

/// Creates max function: fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }
fn create_max_function() -> HirFunction {
    let name = create_test_string("max");

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

    // Create blocks
    let entry_block = func.entry_block;
    let then_block = func.create_block();
    let else_block = func.create_block();
    let merge_block = func.create_block();

    // Create parameter values
    let param_a = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let param_b = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    // Entry block: compare a > b
    let cmp_result = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let cmp_inst = HirInstruction::Binary {
        op: BinaryOp::Gt,
        result: cmp_result,
        ty: HirType::I32,
        left: param_a,
        right: param_b,
    };

    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.add_instruction(cmp_inst);
    entry.set_terminator(HirTerminator::CondBranch {
        condition: cmp_result,
        true_target: then_block,
        false_target: else_block,
    });

    // Then block: return a
    let then = func.blocks.get_mut(&then_block).unwrap();
    then.set_terminator(HirTerminator::Branch {
        target: merge_block,
    });

    // Else block: return b
    let else_blk = func.blocks.get_mut(&else_block).unwrap();
    else_blk.set_terminator(HirTerminator::Branch {
        target: merge_block,
    });

    // Merge block: phi node to select between a and b
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let phi = HirPhi {
        result,
        ty: HirType::I32,
        incoming: vec![(then_block, param_a), (else_block, param_b)],
    };

    let merge = func.blocks.get_mut(&merge_block).unwrap();
    merge.add_phi(phi);
    merge.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Creates abs function: fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }
fn create_abs_function() -> HirFunction {
    let name = create_test_string("abs");

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

    let entry_block = func.entry_block;
    let then_block = func.create_block();
    let else_block = func.create_block();
    let merge_block = func.create_block();

    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));

    // Entry: compare x < 0
    let zero = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let cmp_result = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let cmp_inst = HirInstruction::Binary {
        op: BinaryOp::Lt,
        result: cmp_result,
        ty: HirType::I32,
        left: param_x,
        right: zero,
    };

    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.add_instruction(cmp_inst);
    entry.set_terminator(HirTerminator::CondBranch {
        condition: cmp_result,
        true_target: then_block,
        false_target: else_block,
    });

    // Then block: negate x
    let neg_result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let neg_inst = HirInstruction::Unary {
        op: UnaryOp::Neg,
        result: neg_result,
        ty: HirType::I32,
        operand: param_x,
    };

    let then = func.blocks.get_mut(&then_block).unwrap();
    then.add_instruction(neg_inst);
    then.set_terminator(HirTerminator::Branch {
        target: merge_block,
    });

    // Else block: use x as is
    let else_blk = func.blocks.get_mut(&else_block).unwrap();
    else_blk.set_terminator(HirTerminator::Branch {
        target: merge_block,
    });

    // Merge block: phi node
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let phi = HirPhi {
        result,
        ty: HirType::I32,
        incoming: vec![(then_block, neg_result), (else_block, param_x)],
    };

    let merge = func.blocks.get_mut(&merge_block).unwrap();
    merge.add_phi(phi);
    merge.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}

/// Creates countdown: fn countdown(n: i32) -> i32 { while n > 0 { n = n - 1 } return n }
fn create_countdown_function() -> HirFunction {
    let name = create_test_string("countdown");

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("n"),
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

    let entry_block = func.entry_block;
    let loop_header = func.create_block();
    let loop_body = func.create_block();
    let exit_block = func.create_block();

    let param_n = func.create_value(HirType::I32, HirValueKind::Parameter(0));

    // Entry: jump to loop header
    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.set_terminator(HirTerminator::Branch {
        target: loop_header,
    });

    // Loop header: phi node for n, check n > 0
    let n_phi = func.create_value(HirType::I32, HirValueKind::Instruction);
    let zero = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let cmp_result = func.create_value(HirType::Bool, HirValueKind::Instruction);

    // Loop body: n = n - 1
    let one = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(1)));
    let n_minus_1 = func.create_value(HirType::I32, HirValueKind::Instruction);
    let sub_inst = HirInstruction::Binary {
        op: BinaryOp::Sub,
        result: n_minus_1,
        ty: HirType::I32,
        left: n_phi,
        right: one,
    };

    let body = func.blocks.get_mut(&loop_body).unwrap();
    body.add_instruction(sub_inst);
    body.set_terminator(HirTerminator::Branch {
        target: loop_header,
    });

    // Now set up loop header with phi node
    let phi = HirPhi {
        result: n_phi,
        ty: HirType::I32,
        incoming: vec![(entry_block, param_n), (loop_body, n_minus_1)],
    };

    let cmp_inst = HirInstruction::Binary {
        op: BinaryOp::Gt,
        result: cmp_result,
        ty: HirType::I32,
        left: n_phi,
        right: zero,
    };

    let header = func.blocks.get_mut(&loop_header).unwrap();
    header.add_phi(phi);
    header.add_instruction(cmp_inst);
    header.set_terminator(HirTerminator::CondBranch {
        condition: cmp_result,
        true_target: loop_body,
        false_target: exit_block,
    });

    // Exit block: return n
    let exit = func.blocks.get_mut(&exit_block).unwrap();
    exit.set_terminator(HirTerminator::Return {
        values: vec![n_phi],
    });

    func
}

/// Creates factorial: fn factorial(n: i32) -> i32
fn create_factorial_function() -> HirFunction {
    let name = create_test_string("factorial");

    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("n"),
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

    let entry_block = func.entry_block;
    let loop_header = func.create_block();
    let loop_body = func.create_block();
    let exit_block = func.create_block();

    let param_n = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let one = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(1)));

    // Entry: jump to loop header
    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.set_terminator(HirTerminator::Branch {
        target: loop_header,
    });

    // Phi nodes for counter (i) and accumulator (acc)
    let i_phi = func.create_value(HirType::I32, HirValueKind::Instruction);
    let acc_phi = func.create_value(HirType::I32, HirValueKind::Instruction);

    // Loop body: acc = acc * i, i = i + 1
    let new_acc = func.create_value(HirType::I32, HirValueKind::Instruction);
    let mul_inst = HirInstruction::Binary {
        op: BinaryOp::Mul,
        result: new_acc,
        ty: HirType::I32,
        left: acc_phi,
        right: i_phi,
    };

    let new_i = func.create_value(HirType::I32, HirValueKind::Instruction);
    let add_inst = HirInstruction::Binary {
        op: BinaryOp::Add,
        result: new_i,
        ty: HirType::I32,
        left: i_phi,
        right: one,
    };

    let body = func.blocks.get_mut(&loop_body).unwrap();
    body.add_instruction(mul_inst);
    body.add_instruction(add_inst);
    body.set_terminator(HirTerminator::Branch {
        target: loop_header,
    });

    // Loop header: check i <= n
    let cmp_result = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let cmp_inst = HirInstruction::Binary {
        op: BinaryOp::Le,
        result: cmp_result,
        ty: HirType::I32,
        left: i_phi,
        right: param_n,
    };

    let i_phi_node = HirPhi {
        result: i_phi,
        ty: HirType::I32,
        incoming: vec![(entry_block, one), (loop_body, new_i)],
    };

    let acc_phi_node = HirPhi {
        result: acc_phi,
        ty: HirType::I32,
        incoming: vec![(entry_block, one), (loop_body, new_acc)],
    };

    let header = func.blocks.get_mut(&loop_header).unwrap();
    header.add_phi(i_phi_node);
    header.add_phi(acc_phi_node);
    header.add_instruction(cmp_inst);
    header.set_terminator(HirTerminator::CondBranch {
        condition: cmp_result,
        true_target: loop_body,
        false_target: exit_block,
    });

    // Exit block: return accumulator
    let exit = func.blocks.get_mut(&exit_block).unwrap();
    exit.set_terminator(HirTerminator::Return {
        values: vec![acc_phi],
    });

    func
}

/// Creates sign: fn sign(x: i32) -> i32 { if x > 0 { 1 } else if x < 0 { -1 } else { 0 } }
fn create_sign_function() -> HirFunction {
    let name = create_test_string("sign");

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

    let entry_block = func.entry_block;
    let positive_block = func.create_block();
    let check_negative_block = func.create_block();
    let negative_block = func.create_block();
    let zero_block = func.create_block();
    let merge_block = func.create_block();

    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let zero = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let one = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(1)));
    let minus_one = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(-1)));

    // Entry: check x > 0
    let cmp_positive = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let cmp_pos_inst = HirInstruction::Binary {
        op: BinaryOp::Gt,
        result: cmp_positive,
        ty: HirType::I32,
        left: param_x,
        right: zero,
    };

    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.add_instruction(cmp_pos_inst);
    entry.set_terminator(HirTerminator::CondBranch {
        condition: cmp_positive,
        true_target: positive_block,
        false_target: check_negative_block,
    });

    // Positive block: return 1
    let positive = func.blocks.get_mut(&positive_block).unwrap();
    positive.set_terminator(HirTerminator::Branch {
        target: merge_block,
    });

    // Check negative: x < 0
    let cmp_negative = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let cmp_neg_inst = HirInstruction::Binary {
        op: BinaryOp::Lt,
        result: cmp_negative,
        ty: HirType::I32,
        left: param_x,
        right: zero,
    };

    let check_neg = func.blocks.get_mut(&check_negative_block).unwrap();
    check_neg.add_instruction(cmp_neg_inst);
    check_neg.set_terminator(HirTerminator::CondBranch {
        condition: cmp_negative,
        true_target: negative_block,
        false_target: zero_block,
    });

    // Negative block: return -1
    let negative = func.blocks.get_mut(&negative_block).unwrap();
    negative.set_terminator(HirTerminator::Branch {
        target: merge_block,
    });

    // Zero block: return 0
    let zero_blk = func.blocks.get_mut(&zero_block).unwrap();
    zero_blk.set_terminator(HirTerminator::Branch {
        target: merge_block,
    });

    // Merge block: phi node with 3 predecessors
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let phi = HirPhi {
        result,
        ty: HirType::I32,
        incoming: vec![
            (positive_block, one),
            (negative_block, minus_one),
            (zero_block, zero),
        ],
    };

    let merge = func.blocks.get_mut(&merge_block).unwrap();
    merge.add_phi(phi);
    merge.set_terminator(HirTerminator::Return {
        values: vec![result],
    });

    func
}
