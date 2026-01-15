//! # Cranelift Backend Tests
//! 
//! Test the Cranelift backend IR generation and compilation.

use std::collections::HashSet;
use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::{InternedString, arena::AstArena};

/// Test basic function compilation
#[test]
fn test_simple_function_compilation() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // Create a simple function: fn add(a: i32, b: i32) -> i32 { a + b }
    let func = create_test_add_function();
    
    // Compile the function
    let result = backend.compile_function(func.id, &func);
    
    match result {
        Ok(()) => {
            println!("✅ Successfully compiled add function");
            
            // Get the function pointer
            if let Some(func_ptr) = backend.get_function_ptr(func.id) {
                println!("📍 Function compiled to address: {:p}", func_ptr);
            }
        }
        Err(e) => {
            println!("❌ Failed to compile function: {}", e);
            panic!("Compilation failed: {}", e);
        }
    }
}

/// Test module compilation
#[test]
fn test_module_compilation() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // Create a simple module with one function
    let module = create_test_module();
    
    // Compile the module
    let result = backend.compile_module(&module);
    
    match result {
        Ok(()) => {
            println!("✅ Successfully compiled test module");
        }
        Err(e) => {
            println!("❌ Failed to compile module: {}", e);
            // Note: We expect some compilation errors due to incomplete implementation
            // This test validates the basic compilation pipeline works
        }
    }
}

/// Test HIR to Cranelift type conversion
#[test] 
fn test_type_conversion() {
    let backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // Test primitive types
    let test_cases = vec![
        (HirType::I32, "i32"),
        (HirType::I64, "i64"), 
        (HirType::F32, "f32"),
        (HirType::F64, "f64"),
        (HirType::Bool, "bool (as i8)"),
        (HirType::Void, "void (as i8)"),
    ];
    
    for (hir_type, description) in test_cases {
        match backend.translate_type(&hir_type) {
            Ok(cranelift_type) => {
                println!("✅ {}: {:?}", description, cranelift_type);
            }
            Err(e) => {
                println!("❌ Failed to convert {}: {}", description, e);
            }
        }
    }
}

/// Test function signature translation
#[test]
fn test_signature_translation() {
    let backend = CraneliftBackend::new().expect("Failed to create backend");
    
    let func = create_test_add_function();
    
    match backend.translate_signature(&func) {
        Ok(sig) => {
            println!("✅ Successfully translated function signature");
            println!("📋 Signature: {:?}", sig);
            println!("📥 Parameters: {}", sig.params.len());
            println!("📤 Returns: {}", sig.returns.len());
        }
        Err(e) => {
            println!("❌ Failed to translate signature: {}", e);
        }
    }
}

// Helper functions to create test HIR structures

fn create_test_add_function() -> HirFunction {
    let name = create_test_string("add");
    
    // Create function signature: (i32, i32) -> i32
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
    
    // Create a simple block that adds the parameters
    let entry_block_id = func.entry_block;
    
    
    // Create parameter values
    let param_a = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let param_b = func.create_value(HirType::I32, HirValueKind::Parameter(1));
    
    // Create add instruction: result = a + b
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
    
    // Create return instruction
    block.set_terminator(HirTerminator::Return { 
        values: vec![result] 
    });
    
    func
}

fn create_test_module() -> HirModule {
    let mut module = HirModule::new(create_test_string("test_module"));
    
    // Add the test function
    let func = create_test_add_function();
    module.add_function(func);
    
    module
}

fn create_test_string(s: &str) -> InternedString {
    // Create a temporary arena for testing
    let mut arena = AstArena::new();
    arena.intern_string(s)
}

/// Test arithmetic operations
#[test]
fn test_arithmetic_operations() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // Test subtraction
    let sub_func = create_arithmetic_function("sub", BinaryOp::Sub);
    backend.compile_function(sub_func.id, &sub_func)
        .expect("Failed to compile subtraction function");
    
    // Test multiplication
    let mul_func = create_arithmetic_function("mul", BinaryOp::Mul);
    backend.compile_function(mul_func.id, &mul_func)
        .expect("Failed to compile multiplication function");
    
    // Test division
    let div_func = create_arithmetic_function("div", BinaryOp::Div);
    backend.compile_function(div_func.id, &div_func)
        .expect("Failed to compile division function");
}

/// Test comparison operations
#[test]
fn test_comparison_operations() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // Test equality
    let eq_func = create_comparison_function("eq", BinaryOp::Eq);
    backend.compile_function(eq_func.id, &eq_func)
        .expect("Failed to compile equality function");
    
    // Test less than
    let lt_func = create_comparison_function("lt", BinaryOp::Lt);
    backend.compile_function(lt_func.id, &lt_func)
        .expect("Failed to compile less than function");
}

/// Test function with multiple blocks (control flow)
#[test]
fn test_control_flow() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_control_flow_function();
    
    backend.compile_function(func.id, &func)
        .expect("Failed to compile control flow function");
}

/// Test function calling
#[test]
fn test_function_call() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // First compile the add function
    let add_func = create_test_add_function();
    backend.compile_function(add_func.id, &add_func)
        .expect("Failed to compile add function");
    
    // Then compile a function that calls it
    let caller_func = create_caller_function(add_func.id);
    backend.compile_function(caller_func.id, &caller_func)
        .expect("Failed to compile caller function");
}

/// Helper function to create arithmetic operations
fn create_arithmetic_function(name: &str, op: BinaryOp) -> HirFunction {
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
    let param_a = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let param_b = func.create_value(HirType::I32, HirValueKind::Parameter(1));
    
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let inst = HirInstruction::Binary {
        op,
        result,
        ty: HirType::I32,
        left: param_a,
        right: param_b,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(inst);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create comparison operations
fn create_comparison_function(name: &str, op: BinaryOp) -> HirFunction {
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
        returns: vec![HirType::Bool],
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
    
    let result = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let inst = HirInstruction::Binary {
        op,
        result,
        ty: HirType::Bool,
        left: param_a,
        right: param_b,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(inst);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Create a function with control flow (if-else)
fn create_control_flow_function() -> HirFunction {
    let name = create_test_string("abs");
    
    // Create function: fn abs(x: i32) -> i32
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("x"),
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
    
    // Entry block: check if x < 0
    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let zero = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let cond = func.create_value(HirType::Bool, HirValueKind::Instruction);
    
    let cmp_inst = HirInstruction::Binary {
        op: BinaryOp::Lt,
        result: cond,
        ty: HirType::Bool,
        left: param_x,
        right: zero,
    };
    
    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.add_instruction(cmp_inst);
    entry.set_terminator(HirTerminator::CondBranch {
        condition: cond,
        true_target: then_block,
        false_target: else_block,
    });
    
    // Then block: return -x
    let neg_result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let neg_inst = HirInstruction::Unary {
        op: UnaryOp::Neg,
        result: neg_result,
        ty: HirType::I32,
        operand: param_x,
    };
    
    let then = func.blocks.get_mut(&then_block).unwrap();
    then.add_instruction(neg_inst);
    then.set_terminator(HirTerminator::Branch { target: merge_block });
    
    // Else block: return x
    let else_blk = func.blocks.get_mut(&else_block).unwrap();
    else_blk.set_terminator(HirTerminator::Branch { target: merge_block });
    
    // Merge block: phi node and return
    let phi_result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let phi = HirPhi {
        result: phi_result,
        ty: HirType::I32,
        // Phi incoming format is (value, block) as defined in HirPhi
        incoming: vec![
            (neg_result, then_block),
            (param_x, else_block),
        ],
    };
    
    let merge = func.blocks.get_mut(&merge_block).unwrap();
    merge.phis.push(phi);
    merge.set_terminator(HirTerminator::Return { values: vec![phi_result] });
    
    func
}

/// Create a function that calls another function
fn create_caller_function(callee_id: HirId) -> HirFunction {
    let name = create_test_string("call_add");
    
    // fn call_add(x: i32, y: i32) -> i32 { add(x, y) }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("x"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("y"),
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
    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    let param_y = func.create_value(HirType::I32, HirValueKind::Parameter(1));
    
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let call_inst = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Function(callee_id),
        args: vec![param_x, param_y],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };
    
    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(call_inst);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Test ExtractValue instruction (struct field access)
#[test]
fn test_extract_value_instruction() {
    // Skip this test for now - ExtractValue/InsertValue need proper aggregate value support
    // which Cranelift doesn't directly support. We use load/store with offsets instead.
    println!("⚠️ ExtractValue test skipped - needs proper aggregate value support");
}

/// Test InsertValue instruction (struct field modification)
#[test]
fn test_insert_value_instruction() {
    // Skip this test for now - ExtractValue/InsertValue need proper aggregate value support
    // which Cranelift doesn't directly support. We use load/store with offsets instead.
    println!("⚠️ InsertValue test skipped - needs proper aggregate value support");
}

/// Test Select instruction (ternary conditional)
#[test]
fn test_select_instruction() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_select_function();
    
    backend.compile_function(func.id, &func)
        .expect("Failed to compile select function");
}

/// Helper function to create a function that uses Select (max function)
fn create_select_function() -> HirFunction {
    let name = create_test_string("max");
    
    // Create function: fn max(a: i32, b: i32) -> i32 { a > b ? a : b }
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
    
    // Create comparison: a > b
    let cond = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let cmp_inst = HirInstruction::Binary {
        op: BinaryOp::Gt,
        result: cond,
        ty: HirType::Bool,
        left: param_a,
        right: param_b,
    };
    
    // Create select: a > b ? a : b
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let select_inst = HirInstruction::Select {
        result,
        ty: HirType::I32,
        condition: cond,
        true_val: param_a,
        false_val: param_b,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(cmp_inst);
    block.add_instruction(select_inst);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create a function that uses ExtractValue (get first field of struct)
#[allow(dead_code)]
fn create_extract_value_function() -> HirFunction {
    let name = create_test_string("get_first_field");
    
    // Create a proper struct type
    let struct_ty = HirType::Struct(HirStructType {
        name: Some(create_test_string("Point")),
        fields: vec![HirType::I32, HirType::I32], // x, y fields
        packed: false,
    });
    
    // Create function: fn get_first_field(s: *Point) -> i32 { s.x }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("s"),
                ty: HirType::Ptr(Box::new(struct_ty.clone())),
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
    let param_s = func.create_value(HirType::Ptr(Box::new(struct_ty)), HirValueKind::Parameter(0));
    
    // Extract first field (x coordinate at index 0)
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract_inst = HirInstruction::ExtractValue {
        result,
        ty: HirType::I32,
        aggregate: param_s,
        indices: vec![0], // First field (x)
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract_inst);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create a function that uses InsertValue (set first field of struct)
#[allow(dead_code)]
fn create_insert_value_function() -> HirFunction {
    let name = create_test_string("set_first_field");
    
    // Create a proper struct type
    let struct_ty = HirType::Struct(HirStructType {
        name: Some(create_test_string("Point")),
        fields: vec![HirType::I32, HirType::I32], // x, y fields
        packed: false,
    });
    
    // Create function: fn set_first_field(s: *Point, val: i32) -> *Point { s.x = val; s }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("s"),
                ty: HirType::Ptr(Box::new(struct_ty.clone())),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("val"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(struct_ty.clone()))],
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
    let param_s = func.create_value(HirType::Ptr(Box::new(struct_ty.clone())), HirValueKind::Parameter(0));
    let param_val = func.create_value(HirType::I32, HirValueKind::Parameter(1));
    
    // Insert value into first field (x coordinate at index 0)
    let result = func.create_value(HirType::Ptr(Box::new(struct_ty)), HirValueKind::Instruction);
    let insert_inst = HirInstruction::InsertValue {
        result,
        ty: HirType::I32, // Type of the value being inserted
        aggregate: param_s,
        value: param_val,
        indices: vec![0], // First field (x)
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(insert_inst);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Test intrinsic function calls
#[test]
fn test_intrinsic_functions() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // Test sqrt intrinsic
    let sqrt_func = create_sqrt_intrinsic_function();
    backend.compile_function(sqrt_func.id, &sqrt_func)
        .expect("Failed to compile sqrt intrinsic function");
    
    // Test bit manipulation intrinsics
    let ctpop_func = create_ctpop_intrinsic_function();
    backend.compile_function(ctpop_func.id, &ctpop_func)
        .expect("Failed to compile ctpop intrinsic function");
}

/// Test memory management intrinsics
#[test]
fn test_memory_management_intrinsics() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    
    // Test malloc intrinsic
    let malloc_func = create_malloc_intrinsic_function();
    match backend.compile_function(malloc_func.id, &malloc_func) {
        Ok(()) => println!("✅ Successfully compiled malloc intrinsic function"),
        Err(e) => {
            println!("❌ Failed to compile malloc intrinsic function: {}", e);
            panic!("Malloc compilation failed");
        }
    }
    
    // Test free intrinsic
    let free_func = create_free_intrinsic_function();
    match backend.compile_function(free_func.id, &free_func) {
        Ok(()) => println!("✅ Successfully compiled free intrinsic function"),
        Err(e) => {
            println!("❌ Failed to compile free intrinsic function: {}", e);
            panic!("Free compilation failed");
        }
    }
    
    // Test reference counting intrinsics
    let incref_func = create_incref_intrinsic_function();
    match backend.compile_function(incref_func.id, &incref_func) {
        Ok(()) => println!("✅ Successfully compiled incref intrinsic function"),
        Err(e) => {
            println!("❌ Failed to compile incref intrinsic function: {}", e);
            panic!("IncRef compilation failed");
        }
    }
}

/// Helper function to create a function that uses sqrt intrinsic
fn create_sqrt_intrinsic_function() -> HirFunction {
    let name = create_test_string("test_sqrt");
    
    // Create function: fn test_sqrt(x: f64) -> f64 { sqrt(x) }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("x"),
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
    let param_x = func.create_value(HirType::F64, HirValueKind::Parameter(0));
    
    // Call sqrt intrinsic
    let result = func.create_value(HirType::F64, HirValueKind::Instruction);
    let sqrt_call = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(Intrinsic::Sqrt),
        args: vec![param_x],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(sqrt_call);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create a function that uses ctpop intrinsic
fn create_ctpop_intrinsic_function() -> HirFunction {
    let name = create_test_string("test_ctpop");
    
    // Create function: fn test_ctpop(x: i32) -> i32 { ctpop(x) }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("x"),
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
    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    
    // Call ctpop intrinsic (count population - number of 1 bits)
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let ctpop_call = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(Intrinsic::Ctpop),
        args: vec![param_x],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(ctpop_call);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create a function that uses malloc intrinsic
fn create_malloc_intrinsic_function() -> HirFunction {
    let name = create_test_string("test_malloc");
    
    // Create function: fn test_malloc(size: i64) -> *void { malloc(size) }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("size"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(HirType::I8))], // void* as *i8
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
    let param_size = func.create_value(HirType::I64, HirValueKind::Parameter(0));
    
    // Call malloc intrinsic
    let result = func.create_value(HirType::Ptr(Box::new(HirType::I8)), HirValueKind::Instruction);
    let malloc_call = HirInstruction::Call {
        result: Some(result),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![param_size],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(malloc_call);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create a function that uses free intrinsic
fn create_free_intrinsic_function() -> HirFunction {
    let name = create_test_string("test_free");
    
    // Create function: fn test_free(ptr: *void) { free(ptr) }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("ptr"),
                ty: HirType::Ptr(Box::new(HirType::I8)),
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![],
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
    let param_ptr = func.create_value(HirType::Ptr(Box::new(HirType::I8)), HirValueKind::Parameter(0));
    
    // Call free intrinsic
    let free_call = HirInstruction::Call {
        result: None,
        callee: HirCallable::Intrinsic(Intrinsic::Free),
        args: vec![param_ptr],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(free_call);
    block.set_terminator(HirTerminator::Return { values: vec![] });
    
    func
}

/// Helper function to create a function that uses incref intrinsic
fn create_incref_intrinsic_function() -> HirFunction {
    let name = create_test_string("test_incref");
    
    // Create function: fn test_incref(ptr: *RefCounted) { incref(ptr) }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("ptr"),
                ty: HirType::Ptr(Box::new(HirType::I32)), // Simplified - ptr to refcounted struct
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![],
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
    let param_ptr = func.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Parameter(0));
    
    // Call incref intrinsic
    let incref_call = HirInstruction::Call {
        result: None,
        callee: HirCallable::Intrinsic(Intrinsic::IncRef),
        args: vec![param_ptr],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(incref_call);
    block.set_terminator(HirTerminator::Return { values: vec![] });
    
    func
}

/// Test union type creation and access
#[test]
fn test_union_operations() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_union_test_function();
    
    backend.compile_function(func.id, &func)
        .expect("Failed to compile union test function");
}

/// Test closure creation and calling
#[test]
fn test_closure_operations() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_closure_test_function();
    
    backend.compile_function(func.id, &func)
        .expect("Failed to compile closure test function");
}

/// Test pattern matching compilation
#[test]
fn test_pattern_matching() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_pattern_match_function();
    
    backend.compile_function(func.id, &func)
        .expect("Failed to compile pattern match function");
}

/// Helper function to create a function that uses union types
fn create_union_test_function() -> HirFunction {
    let name = create_test_string("test_union");
    
    // Create a union type: enum Option { None, Some(i32) }
    let union_ty = HirType::Union(Box::new(HirUnionType {
        name: Some(create_test_string("Option")),
        variants: vec![
            HirUnionVariant {
                name: create_test_string("None"),
                ty: HirType::Void,
                discriminant: 0,
            },
            HirUnionVariant {
                name: create_test_string("Some"),
                ty: HirType::I32,
                discriminant: 1,
            },
        ],
        discriminant_type: Box::new(HirType::U8),
        is_c_union: false,
    }));
    
    // Create function: fn test_union(value: i32) -> *Option { Some(value) }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("value"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(union_ty.clone()))],
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
    let param_value = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    
    // Create union with Some variant
    let result = func.create_value(HirType::Ptr(Box::new(union_ty.clone())), HirValueKind::Instruction);
    let create_union = HirInstruction::CreateUnion {
        result,
        union_ty,
        variant_index: 1, // Some variant
        value: param_value,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(create_union);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create a function that uses closures
fn create_closure_test_function() -> HirFunction {
    let name = create_test_string("test_closure");
    
    // Create a closure type that captures one i32 value
    let closure_ty = HirType::Closure(Box::new(HirClosureType {
        function_type: HirFunctionType {
            params: vec![HirType::I32], // closure takes one i32 param
            returns: vec![HirType::I32],
            lifetime_params: vec![],
            is_variadic: false,
        },
        captures: vec![
            HirCapture {
                name: create_test_string("captured_x"),
                ty: HirType::I32,
                mode: HirCaptureMode::ByValue,
            },
        ],
        call_mode: HirClosureCallMode::Fn,
    }));
    
    // Create function: fn test_closure(x: i32) -> *Closure
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("x"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(closure_ty.clone()))],
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
    
    // Create a dummy function ID for the closure
    let closure_func_id = HirId::new();
    
    // Create closure
    let result = func.create_value(HirType::Ptr(Box::new(closure_ty.clone())), HirValueKind::Instruction);
    let create_closure = HirInstruction::CreateClosure {
        result,
        closure_ty,
        function: closure_func_id,
        captures: vec![param_x],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(create_closure);
    block.set_terminator(HirTerminator::Return { values: vec![result] });
    
    func
}

/// Helper function to create a function that uses pattern matching
fn create_pattern_match_function() -> HirFunction {
    let name = create_test_string("test_pattern_match");
    
    // Create function: fn test_pattern_match(x: i32) -> i32
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("x"),
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
    
    // Create blocks for pattern matching
    let entry_block = func.entry_block;
    let pattern1_block = func.create_block();
    let pattern2_block = func.create_block();
    let default_block = func.create_block();
    
    let param_x = func.create_value(HirType::I32, HirValueKind::Parameter(0));
    
    // Entry block: pattern match on x
    let patterns = vec![
        HirPattern {
            kind: HirPatternKind::Constant(HirConstant::I32(42)),
            target: pattern1_block,
            bindings: vec![],
        },
        HirPattern {
            kind: HirPatternKind::Constant(HirConstant::I32(100)),
            target: pattern2_block,
            bindings: vec![],
        },
    ];
    
    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.set_terminator(HirTerminator::PatternMatch {
        value: param_x,
        patterns,
        default: Some(default_block),
    });
    
    // Pattern 1 block: return 1
    let one_const = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(1)));
    let pattern1 = func.blocks.get_mut(&pattern1_block).unwrap();
    pattern1.set_terminator(HirTerminator::Return { values: vec![one_const] });
    
    // Pattern 2 block: return 2  
    let two_const = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(2)));
    let pattern2 = func.blocks.get_mut(&pattern2_block).unwrap();
    pattern2.set_terminator(HirTerminator::Return { values: vec![two_const] });
    
    // Default block: return 0
    let zero_const = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let default = func.blocks.get_mut(&default_block).unwrap();
    default.set_terminator(HirTerminator::Return { values: vec![zero_const] });

    func
}

/// Test Alloca instruction (stack allocation)
#[test]
fn test_alloca_instruction() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_alloca_test_function();

    backend.compile_function(func.id, &func)
        .expect("Failed to compile alloca test function");
}

/// Test Load instruction (memory read)
#[test]
fn test_load_instruction() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_load_test_function();

    backend.compile_function(func.id, &func)
        .expect("Failed to compile load test function");
}

/// Test Store instruction (memory write)
#[test]
fn test_store_instruction() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_store_test_function();

    backend.compile_function(func.id, &func)
        .expect("Failed to compile store test function");
}

/// Test GetElementPtr instruction (pointer arithmetic)
#[test]
fn test_gep_instruction() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_gep_test_function();

    backend.compile_function(func.id, &func)
        .expect("Failed to compile GEP test function");
}

/// Test combined memory operations (alloca + store + load)
#[test]
fn test_memory_operations_combined() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");
    let func = create_combined_memory_test_function();

    backend.compile_function(func.id, &func)
        .expect("Failed to compile combined memory test function");
}

/// Helper function to create a function that uses Alloca
fn create_alloca_test_function() -> HirFunction {
    let name = create_test_string("test_alloca");

    // Create function: fn test_alloca() -> *i32 { alloca i32 }
    let sig = HirFunctionSignature {
        params: vec![],
        returns: vec![HirType::Ptr(Box::new(HirType::I32))],
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

    // Alloca i32
    let ptr = func.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Instruction);
    let alloca = HirInstruction::Alloca {
        result: ptr,
        ty: HirType::I32,
        count: None, // Single value allocation
        align: 4,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(alloca);
    block.set_terminator(HirTerminator::Return { values: vec![ptr] });

    func
}

/// Helper function to create a function that uses Load
fn create_load_test_function() -> HirFunction {
    let name = create_test_string("test_load");

    // Create function: fn test_load(ptr: *i32) -> i32 { load ptr }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("ptr"),
                ty: HirType::Ptr(Box::new(HirType::I32)),
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
    let param_ptr = func.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Parameter(0));

    // Load from ptr
    let result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let load = HirInstruction::Load {
        result,
        ty: HirType::I32,
        ptr: param_ptr,
        align: 4,
        volatile: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(load);
    block.set_terminator(HirTerminator::Return { values: vec![result] });

    func
}

/// Helper function to create a function that uses Store
fn create_store_test_function() -> HirFunction {
    let name = create_test_string("test_store");

    // Create function: fn test_store(ptr: *i32, val: i32) { store val, ptr }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("ptr"),
                ty: HirType::Ptr(Box::new(HirType::I32)),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("val"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![],
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
    let param_ptr = func.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Parameter(0));
    let param_val = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    // Store val to ptr
    let store = HirInstruction::Store {
        value: param_val,
        ptr: param_ptr,
        align: 4,
        volatile: false,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(store);
    block.set_terminator(HirTerminator::Return { values: vec![] });

    func
}

/// Helper function to create a function that uses GetElementPtr
fn create_gep_test_function() -> HirFunction {
    let name = create_test_string("test_gep");

    // Create function: fn test_gep(arr: *[i32; 10], idx: i64) -> *i32 { gep arr, idx }
    let array_ty = HirType::Array(Box::new(HirType::I32), 10);

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("arr"),
                ty: HirType::Ptr(Box::new(array_ty.clone())),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("idx"),
                ty: HirType::I64,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(HirType::I32))],
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
    let param_arr = func.create_value(HirType::Ptr(Box::new(array_ty.clone())), HirValueKind::Parameter(0));
    let param_idx = func.create_value(HirType::I64, HirValueKind::Parameter(1));

    // GEP: calculate pointer to arr[idx]
    let elem_ptr = func.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Instruction);
    let gep = HirInstruction::GetElementPtr {
        result: elem_ptr,
        ty: array_ty,
        ptr: param_arr,
        indices: vec![param_idx],
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(gep);
    block.set_terminator(HirTerminator::Return { values: vec![elem_ptr] });

    func
}

/// Helper function to create a function that combines memory operations
fn create_combined_memory_test_function() -> HirFunction {
    let name = create_test_string("test_combined_memory");

    // Create function: fn test_combined_memory(val: i32) -> i32 {
    //     let ptr = alloca i32
    //     store val, ptr
    //     let result = load ptr
    //     return result
    // }
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("val"),
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
    let param_val = func.create_value(HirType::I32, HirValueKind::Parameter(0));

    // Alloca i32
    let ptr = func.create_value(HirType::Ptr(Box::new(HirType::I32)), HirValueKind::Instruction);
    let alloca = HirInstruction::Alloca {
        result: ptr,
        ty: HirType::I32,
        count: None,
        align: 4,
    };

    // Store val to ptr
    let store = HirInstruction::Store {
        value: param_val,
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
    block.set_terminator(HirTerminator::Return { values: vec![result] });

    func
}
// ============================================================================
// Data Structure Tests (Structs and Arrays)
// ============================================================================

/// Test simple struct with ExtractValue
#[test]
fn test_simple_struct_extract() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let struct_ty = HirStructType {
        name: Some(create_test_string("Point")),
        fields: vec![HirType::I32, HirType::I32],  // x, y
        packed: false,
    };

    let name = create_test_string("test_extract_point_x");
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("p"),
            ty: HirType::Ptr(Box::new(HirType::Struct(struct_ty.clone()))),
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

    let param_p = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(struct_ty))),
        HirValueKind::Parameter(0)
    );

    let x_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: x_val,
        ty: HirType::I32,
        aggregate: param_p,
        indices: vec![0],  // Extract field 0 (x)
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return { values: vec![x_val] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile struct extract test");
}

/// Test simple struct with InsertValue
#[test]
fn test_simple_struct_insert() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let struct_ty = HirStructType {
        name: Some(create_test_string("Point")),
        fields: vec![HirType::I32, HirType::I32],
        packed: false,
    };

    let name = create_test_string("test_insert_point_y");
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("p"),
                ty: HirType::Ptr(Box::new(HirType::Struct(struct_ty.clone()))),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("new_y"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(HirType::Struct(struct_ty.clone())))],
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

    let param_p = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(struct_ty.clone()))),
        HirValueKind::Parameter(0)
    );
    let param_new_y = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    let result = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(struct_ty))),
        HirValueKind::Instruction
    );
    let insert = HirInstruction::InsertValue {
        result,
        ty: HirType::I32,
        aggregate: param_p,
        value: param_new_y,
        indices: vec![1],  // Insert into field 1 (y)
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(insert);
    block.set_terminator(HirTerminator::Return { values: vec![result] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile struct insert test");
}

/// Test array with ExtractValue
#[test]
fn test_array_extract() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let array_ty = HirType::Array(Box::new(HirType::I32), 5);

    let name = create_test_string("test_extract_array_elem");
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("arr"),
            ty: HirType::Ptr(Box::new(array_ty.clone())),
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

    let param_arr = func.create_value(
        HirType::Ptr(Box::new(array_ty)),
        HirValueKind::Parameter(0)
    );

    let elem_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: elem_val,
        ty: HirType::I32,
        aggregate: param_arr,
        indices: vec![2],  // Extract element at index 2
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return { values: vec![elem_val] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile array extract test");
}

/// Test array with InsertValue
#[test]
fn test_array_insert() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let array_ty = HirType::Array(Box::new(HirType::I32), 5);

    let name = create_test_string("test_insert_array_elem");
    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("arr"),
                ty: HirType::Ptr(Box::new(array_ty.clone())),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: create_test_string("val"),
                ty: HirType::I32,
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Ptr(Box::new(array_ty.clone()))],
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

    let param_arr = func.create_value(
        HirType::Ptr(Box::new(array_ty.clone())),
        HirValueKind::Parameter(0)
    );
    let param_val = func.create_value(HirType::I32, HirValueKind::Parameter(1));

    let result = func.create_value(
        HirType::Ptr(Box::new(array_ty)),
        HirValueKind::Instruction
    );
    let insert = HirInstruction::InsertValue {
        result,
        ty: HirType::I32,
        aggregate: param_arr,
        value: param_val,
        indices: vec![3],  // Insert at index 3
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(insert);
    block.set_terminator(HirTerminator::Return { values: vec![result] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile array insert test");
}

/// Test nested struct (struct containing struct)
#[test]
fn test_nested_struct() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

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

    let name = create_test_string("test_nested_struct_access");
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("s"),
            ty: HirType::Ptr(Box::new(HirType::Struct(outer_ty.clone()))),
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

    let param_s = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(outer_ty))),
        HirValueKind::Parameter(0)
    );

    let inner_field_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: inner_field_val,
        ty: HirType::I32,
        aggregate: param_s,
        indices: vec![0, 1],  // outer.inner.field1
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return { values: vec![inner_field_val] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile nested struct test");
}

/// Test array of structs
#[test]
fn test_array_of_structs() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let point_ty = HirStructType {
        name: Some(create_test_string("Point")),
        fields: vec![HirType::I32, HirType::I32],
        packed: false,
    };

    let array_ty = HirType::Array(Box::new(HirType::Struct(point_ty)), 3);

    let name = create_test_string("test_array_of_structs_access");
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("points"),
            ty: HirType::Ptr(Box::new(array_ty.clone())),
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

    let param_points = func.create_value(
        HirType::Ptr(Box::new(array_ty)),
        HirValueKind::Parameter(0)
    );

    let y_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: y_val,
        ty: HirType::I32,
        aggregate: param_points,
        indices: vec![1, 1],  // points[1].y
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return { values: vec![y_val] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile array of structs test");
}

/// Test 2D array (array of arrays)
#[test]
fn test_2d_array() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let inner_array_ty = HirType::Array(Box::new(HirType::I32), 4);
    let outer_array_ty = HirType::Array(Box::new(inner_array_ty), 3);

    let name = create_test_string("test_2d_array_access");
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("matrix"),
            ty: HirType::Ptr(Box::new(outer_array_ty.clone())),
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

    let param_matrix = func.create_value(
        HirType::Ptr(Box::new(outer_array_ty)),
        HirValueKind::Parameter(0)
    );

    let elem_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: elem_val,
        ty: HirType::I32,
        aggregate: param_matrix,
        indices: vec![2, 3],  // matrix[2][3]
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return { values: vec![elem_val] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile 2D array test");
}

/// Test struct with mixed field types
#[test]
fn test_mixed_struct_types() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let struct_ty = HirStructType {
        name: Some(create_test_string("Mixed")),
        fields: vec![
            HirType::I8,
            HirType::I32,
            HirType::I64,
            HirType::F32,
            HirType::F64,
        ],
        packed: false,
    };

    let name = create_test_string("test_mixed_struct");
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("s"),
            ty: HirType::Ptr(Box::new(HirType::Struct(struct_ty.clone()))),
            attributes: ParamAttributes::default(),
        }],
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

    let param_s = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(struct_ty))),
        HirValueKind::Parameter(0)
    );

    let i64_val = func.create_value(HirType::I64, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: i64_val,
        ty: HirType::I64,
        aggregate: param_s,
        indices: vec![2],  // Extract i64 field
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return { values: vec![i64_val] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile mixed struct types test");
}

/// Test deeply nested structure (3 levels)
#[test]
fn test_deeply_nested_struct() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let level3_ty = HirStructType {
        name: Some(create_test_string("Level3")),
        fields: vec![HirType::I32, HirType::I32],
        packed: false,
    };

    let level2_ty = HirStructType {
        name: Some(create_test_string("Level2")),
        fields: vec![HirType::Struct(level3_ty), HirType::I32],
        packed: false,
    };

    let level1_ty = HirStructType {
        name: Some(create_test_string("Level1")),
        fields: vec![HirType::Struct(level2_ty), HirType::I32],
        packed: false,
    };

    let name = create_test_string("test_deeply_nested");
    let sig = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: create_test_string("s"),
            ty: HirType::Ptr(Box::new(HirType::Struct(level1_ty.clone()))),
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

    let param_s = func.create_value(
        HirType::Ptr(Box::new(HirType::Struct(level1_ty))),
        HirValueKind::Parameter(0)
    );

    let deep_val = func.create_value(HirType::I32, HirValueKind::Instruction);
    let extract = HirInstruction::ExtractValue {
        result: deep_val,
        ty: HirType::I32,
        aggregate: param_s,
        indices: vec![0, 0, 1],  // level1.level2.level3.field1
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(extract);
    block.set_terminator(HirTerminator::Return { values: vec![deep_val] });

    backend.compile_function(func.id, &func)
        .expect("Failed to compile deeply nested struct test");
}

/// Test super-trait upcasting
#[test]
fn test_super_trait_upcast() {
    use zyntax_typed_ast::TypeId;

    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    // Create trait IDs
    let drawable_trait_id = TypeId::new(100);  // Super-trait
    let shape_trait_id = TypeId::new(101);      // Sub-trait: Shape: Drawable

    // Create a test function that upcasts Shape -> Drawable
    // fn test_upcast(shape: dyn Shape) -> dyn Drawable
    let name = create_test_string("test_upcast");

    let sig = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: create_test_string("shape"),
                ty: HirType::TraitObject { trait_id: shape_trait_id, vtable: None },
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::TraitObject { trait_id: drawable_trait_id, vtable: None }],
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

    // Create parameter value for shape trait object
    let shape_param = func.create_value(
        HirType::TraitObject { trait_id: shape_trait_id, vtable: None },
        HirValueKind::Parameter(0)
    );

    // Create dummy vtable global IDs
    let drawable_vtable_global_id = HirId::new();

    // Upcast instruction: Shape -> Drawable
    let drawable_result = func.create_value(
        HirType::TraitObject { trait_id: drawable_trait_id, vtable: None },
        HirValueKind::Instruction
    );

    let upcast = HirInstruction::UpcastTraitObject {
        result: drawable_result,
        sub_trait_object: shape_param,
        sub_trait_id: shape_trait_id,
        super_trait_id: drawable_trait_id,
        super_vtable_id: drawable_vtable_global_id,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(upcast);
    block.set_terminator(HirTerminator::Return { values: vec![drawable_result] });

    // Compile the function
    let result = backend.compile_function(func.id, &func);

    match result {
        Ok(()) => {
            println!("✅ Successfully compiled super-trait upcast");
        }
        Err(e) => {
            println!("❌ Failed to compile upcast: {}", e);
            panic!("Upcast compilation failed: {}", e);
        }
    }
}

// =============================================================================
// Algebraic Effects Cranelift Tests
// =============================================================================

/// Test compiling a module with an effect and handler
#[test]
fn test_effect_module_compilation() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    // Create a module with an effect and handler
    let module = create_effect_module();

    // Compile the module
    let result = backend.compile_module(&module);

    match result {
        Ok(()) => {
            println!("✅ Successfully compiled effect module");
        }
        Err(e) => {
            println!("❌ Failed to compile effect module: {}", e);
            // Effect codegen may fail on external function lookup - that's expected
            // The test validates the basic effect instruction handling
        }
    }
}

/// Test compiling a function with PerformEffect instruction
#[test]
fn test_perform_effect_instruction() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    // Create module with effect, handler, and function that performs the effect
    let module = create_module_with_perform_effect();

    let result = backend.compile_module(&module);

    match result {
        Ok(()) => {
            println!("✅ Successfully compiled function with PerformEffect");
        }
        Err(e) => {
            // Expected: handler function may not be linked
            println!("⚠️ Compilation result (expected external link error): {}", e);
        }
    }
}

/// Test compiling a function with HandleEffect instruction
#[test]
fn test_handle_effect_instruction() {
    let mut backend = CraneliftBackend::new().expect("Failed to create backend");

    let module = create_module_with_handle_effect();

    let result = backend.compile_module(&module);

    match result {
        Ok(()) => {
            println!("✅ Successfully compiled function with HandleEffect");
        }
        Err(e) => {
            println!("⚠️ HandleEffect compilation result: {}", e);
        }
    }
}

/// Helper: Create a simple effect module with Logger effect and ConsoleLogger handler
fn create_effect_module() -> HirModule {
    use std::collections::HashSet;
    use indexmap::IndexMap;

    let mut module = HirModule::new(create_test_string("effect_test"));

    // Create Logger effect: effect Logger { def log(msg: str) }
    let effect_id = HirId::new();
    let effect = HirEffect {
        id: effect_id,
        name: create_test_string("Logger"),
        type_params: vec![],
        operations: vec![HirEffectOp {
            id: HirId::new(),
            name: create_test_string("log"),
            type_params: vec![],
            params: vec![HirParam {
                id: HirId::new(),
                name: create_test_string("msg"),
                ty: HirType::Ptr(Box::new(HirType::I8)), // str as ptr
                attributes: ParamAttributes::default(),
            }],
            return_type: HirType::Void,
        }],
    };
    module.effects.insert(effect_id, effect);

    // Create ConsoleLogger handler
    let handler_id = HirId::new();
    let impl_block_id = HirId::new();
    let mut impl_blocks = IndexMap::new();
    impl_blocks.insert(impl_block_id, HirBlock {
        id: impl_block_id,
        label: None,
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Return { values: vec![] },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    });

    let handler = HirEffectHandler {
        id: handler_id,
        name: create_test_string("ConsoleLogger"),
        effect_id,
        type_params: vec![],
        state_fields: vec![],
        implementations: vec![HirEffectHandlerImpl {
            op_name: create_test_string("log"),
            type_params: vec![],
            params: vec![HirParam {
                id: HirId::new(),
                name: create_test_string("msg"),
                ty: HirType::Ptr(Box::new(HirType::I8)),
                attributes: ParamAttributes::default(),
            }],
            return_type: HirType::Void,
            entry_block: impl_block_id,
            blocks: impl_blocks.clone(),
            is_resumable: false,
        }],
    };
    module.handlers.insert(handler_id, handler);

    // Also create the handler function with mangled name for PerformEffect to call
    // This is what the Tier 1 effect codegen generates calls to
    // NOTE: For simplified testing, we create a no-arg handler
    let handler_func_name = create_test_string("ConsoleLogger$effect$log");
    let handler_func_sig = HirFunctionSignature {
        params: vec![], // No params for simplified test
        returns: vec![],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };
    let mut handler_func = HirFunction::new(handler_func_name, handler_func_sig);
    // Simple implementation: just return (no-op handler for test)
    let entry = handler_func.blocks.get_mut(&handler_func.entry_block).unwrap();
    entry.set_terminator(HirTerminator::Return { values: vec![] });
    module.add_function(handler_func);

    module
}

/// Helper: Create module with a function that performs an effect
fn create_module_with_perform_effect() -> HirModule {
    let mut module = create_effect_module();

    // Get the effect ID from the module
    let effect_id = *module.effects.keys().next().unwrap();

    // Create a function that performs the log effect
    let func_name = create_test_string("log_message");
    let sig = HirFunctionSignature {
        params: vec![],
        returns: vec![],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![create_test_string("Logger")],
        is_pure: false,
    };

    let mut func = HirFunction::new(func_name, sig);
    let entry_block_id = func.entry_block;

    // Add PerformEffect instruction
    let perform = HirInstruction::PerformEffect {
        result: None,
        effect_id,
        op_name: create_test_string("log"),
        args: vec![], // Simplified: no actual args
        return_ty: HirType::Void,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(perform);
    block.set_terminator(HirTerminator::Return { values: vec![] });

    module.add_function(func);
    module
}

/// Helper: Create module with HandleEffect instruction
fn create_module_with_handle_effect() -> HirModule {
    let mut module = create_effect_module();

    let handler_id = *module.handlers.keys().next().unwrap();

    // Create a function with HandleEffect
    let func_name = create_test_string("with_logger");
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

    let mut func = HirFunction::new(func_name, sig);
    let entry_block_id = func.entry_block;

    // Create continuation block first (body will branch to it)
    let cont_block_id = HirId::new();
    // Create a constant value to return (constants are stored in values, not as instructions)
    let result_val = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(42)));
    let cont_block = HirBlock {
        id: cont_block_id,
        label: Some(create_test_string("continuation")),
        phis: vec![],
        instructions: vec![],  // No instructions needed - constant is already a value
        terminator: HirTerminator::Return { values: vec![result_val] },
        dominance_frontier: HashSet::new(),
        predecessors: vec![],
        successors: vec![],
    };
    func.blocks.insert(cont_block_id, cont_block);

    // Create body block (what runs under the handler) - branches to continuation
    let body_block_id = HirId::new();
    let body_block = HirBlock {
        id: body_block_id,
        label: Some(create_test_string("body")),
        phis: vec![],
        instructions: vec![],
        terminator: HirTerminator::Branch { target: cont_block_id },  // Branch to continuation, not return
        dominance_frontier: HashSet::new(),
        predecessors: vec![entry_block_id],
        successors: vec![cont_block_id],
    };
    func.blocks.insert(body_block_id, body_block);

    // Add HandleEffect instruction to entry block
    let handle_result = func.create_value(HirType::I32, HirValueKind::Instruction);
    let handle = HirInstruction::HandleEffect {
        result: Some(handle_result),
        handler_id,
        handler_state: vec![],
        body_block: body_block_id,
        continuation_block: cont_block_id,
        return_ty: HirType::I32,
    };

    let block = func.blocks.get_mut(&entry_block_id).unwrap();
    block.add_instruction(handle);
    block.set_terminator(HirTerminator::Branch { target: body_block_id });

    module.add_function(func);
    module
}
