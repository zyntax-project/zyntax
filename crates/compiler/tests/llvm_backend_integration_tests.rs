// Integration tests for LLVM backend
//
// These tests verify that the LLVM backend can compile real HIR programs
// and produce valid LLVM IR.

#[cfg(feature = "llvm-backend")]
mod llvm_tests {
    use inkwell::context::Context;
    use inkwell::OptimizationLevel;
    use std::collections::{HashMap, HashSet};
    use zyntax_compiler::hir::*;
    use zyntax_compiler::llvm_backend::LLVMBackend;
    use zyntax_typed_ast::{InternedString, TypeId};

    /// Helper to create a placeholder InternedString for testing
    /// Uses transmute with value 1 (InternedString wraps NonZeroU32, so can't be zero)
    fn make_name(_s: &str) -> InternedString {
        unsafe { std::mem::transmute(1u32) }
    }

    /// Helper to create a test block with all required fields
    fn make_block(
        id: HirId,
        instructions: Vec<HirInstruction>,
        terminator: HirTerminator,
    ) -> HirBlock {
        HirBlock {
            id,
            label: None,
            phis: vec![],
            instructions,
            terminator,
            dominance_frontier: HashSet::new(),
            predecessors: vec![],
            successors: vec![],
        }
    }

    /// Helper to create a simple HIR module for testing
    fn create_test_module() -> HirModule {
        HirModule {
            id: HirId::new(),
            name: make_name("test_module"),
            functions: HashMap::new(),
            globals: HashMap::new(),
            types: HashMap::new(),
            imports: vec![],
            exports: vec![],
            version: 0,
            dependencies: HashSet::new(),
            effects: indexmap::IndexMap::new(),
            handlers: indexmap::IndexMap::new(),
        }
    }

    /// Helper to create a simple function signature
    fn create_simple_signature(
        params: Vec<HirType>,
        returns: Vec<HirType>,
    ) -> HirFunctionSignature {
        HirFunctionSignature {
            params: params
                .into_iter()
                .enumerate()
                .map(|(i, ty)| HirParam {
                    id: HirId::new(),
                    name: make_name(&format!("param{}", i)),
                    ty,
                    attributes: ParamAttributes::default(),
                })
                .collect(),
            returns,
            type_params: vec![],
            const_params: vec![],
            lifetime_params: vec![],
            is_variadic: false,
            is_async: false,
            effects: vec![],
            is_pure: false,
        }
    }

    #[test]
    fn test_simple_arithmetic_function() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_arithmetic");

        // Create a simple function: fn add(a: i32, b: i32) -> i32 { return a + b; }
        let func_id = HirId::new();
        let param_a = HirId::new();
        let param_b = HirId::new();
        let result_id = HirId::new();
        let block_id = HirId::new();

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: result_id,
                    ty: HirType::I32,
                    left: param_a,
                    right: param_b,
                }],
                HirTerminator::Return {
                    values: vec![result_id],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("add"),
            signature: HirFunctionSignature {
                params: vec![
                    HirParam {
                        id: param_a,
                        name: make_name("a"),
                        ty: HirType::I32,
                        attributes: ParamAttributes::default(),
                    },
                    HirParam {
                        id: param_b,
                        name: make_name("b"),
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
            },
            entry_block: block_id,
            blocks,
            locals: HashMap::new(),
            values: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile module: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify the LLVM IR contains expected instructions
        assert!(
            llvm_ir.contains("add"),
            "LLVM IR should contain add instruction"
        );
        assert!(
            llvm_ir.contains("ret"),
            "LLVM IR should contain return instruction"
        );

        // JIT compile and execute the function
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            // Extract function name from the LLVM IR
            // Format: define i32 @"func_HirId(uuid)"(i32 %param_0, i32 %param_1)
            //                     ^--name starts  ^--name ends (first paren after closing quote)
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    // Find the closing quote if the name is quoted
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        // Name without quotes, stops at '('
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name in IR");

            println!("Function name: {}", func_name);

            // Get the function with the correct name
            // Function signature: fn(i32, i32) -> i32
            let add_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32, i32) -> i32>(&func_name)
                .expect("Failed to get function from execution engine");

            // Test: 5 + 7 = 12
            let result = add_fn.call(5, 7);
            assert_eq!(result, 12, "5 + 7 should equal 12");

            // Test: -3 + 10 = 7
            let result = add_fn.call(-3, 10);
            assert_eq!(result, 7, "-3 + 10 should equal 7");

            println!("✅ Simple arithmetic function: JIT execution verified!");
        }

        println!("✅ Simple arithmetic function compiled successfully");
        println!("LLVM IR:\n{}", llvm_ir);
    }

    #[test]
    fn test_conditional_branch() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_conditional");

        // Create function: fn max(a: i32, b: i32) -> i32 {
        //     if a > b { return a; } else { return b; }
        // }
        let func_id = HirId::new();
        let param_a = HirId::new();
        let param_b = HirId::new();
        let cmp_result = HirId::new();

        let entry_block = HirId::new();
        let true_block = HirId::new();
        let false_block = HirId::new();

        let mut blocks = HashMap::new();

        // Entry block: compare a > b
        blocks.insert(
            entry_block,
            make_block(
                entry_block,
                vec![HirInstruction::Binary {
                    op: BinaryOp::Gt,
                    result: cmp_result,
                    ty: HirType::Bool,
                    left: param_a,
                    right: param_b,
                }],
                HirTerminator::CondBranch {
                    condition: cmp_result,
                    true_target: true_block,
                    false_target: false_block,
                },
            ),
        );

        // True block: return a
        blocks.insert(
            true_block,
            make_block(
                true_block,
                vec![],
                HirTerminator::Return {
                    values: vec![param_a],
                },
            ),
        );

        // False block: return b
        blocks.insert(
            false_block,
            make_block(
                false_block,
                vec![],
                HirTerminator::Return {
                    values: vec![param_b],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("max"),
            signature: HirFunctionSignature {
                params: vec![
                    HirParam {
                        id: param_a,
                        name: make_name("a"),
                        ty: HirType::I32,
                        attributes: ParamAttributes::default(),
                    },
                    HirParam {
                        id: param_b,
                        name: make_name("b"),
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
            },
            entry_block,
            blocks,
            locals: HashMap::new(),
            values: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile conditional: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify the LLVM IR contains conditional branch
        assert!(
            llvm_ir.contains("icmp"),
            "LLVM IR should contain comparison instruction"
        );
        assert!(
            llvm_ir.contains("br i1"),
            "LLVM IR should contain conditional branch"
        );

        // JIT compile and execute
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let max_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32, i32) -> i32>(&func_name)
                .expect("Failed to get function");

            // Test max function: max(5, 3) = 5
            assert_eq!(max_fn.call(5, 3), 5, "max(5, 3) should be 5");

            // Test max function: max(2, 8) = 8
            assert_eq!(max_fn.call(2, 8), 8, "max(2, 8) should be 8");

            // Test max function: max(-5, -3) = -3
            assert_eq!(max_fn.call(-5, -3), -3, "max(-5, -3) should be -3");

            println!("✅ Conditional branch: JIT execution verified!");
        }

        println!("✅ Conditional branch compiled successfully");
        println!("LLVM IR:\n{}", llvm_ir);
    }

    #[test]
    fn test_switch_statement() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_switch");

        // Create function: fn classify(x: i32) -> i32 {
        //     switch x {
        //         1 => return 10;
        //         2 => return 20;
        //         3 => return 30;
        //         _ => return 0;
        //     }
        // }
        let func_id = HirId::new();
        let param_x = HirId::new();

        let entry_block = HirId::new();
        let case1_block = HirId::new();
        let case2_block = HirId::new();
        let case3_block = HirId::new();
        let default_block = HirId::new();

        let const_10 = HirId::new();
        let const_20 = HirId::new();
        let const_30 = HirId::new();
        let const_0 = HirId::new();

        let mut blocks = HashMap::new();

        // Entry block: switch on x
        blocks.insert(
            entry_block,
            make_block(
                entry_block,
                vec![],
                HirTerminator::Switch {
                    value: param_x,
                    default: default_block,
                    cases: vec![
                        (HirConstant::I32(1), case1_block),
                        (HirConstant::I32(2), case2_block),
                        (HirConstant::I32(3), case3_block),
                    ],
                },
            ),
        );

        // Case 1: return 10
        blocks.insert(
            case1_block,
            make_block(
                case1_block,
                vec![HirInstruction::Binary {
                    op: BinaryOp::Add,
                    result: const_10,
                    ty: HirType::I32,
                    left: param_x,
                    right: param_x, // Just to create a value
                }],
                HirTerminator::Return {
                    values: vec![const_10],
                },
            ),
        );

        // Case 2: return 20
        blocks.insert(
            case2_block,
            make_block(
                case2_block,
                vec![HirInstruction::Binary {
                    op: BinaryOp::Mul,
                    result: const_20,
                    ty: HirType::I32,
                    left: param_x,
                    right: param_x,
                }],
                HirTerminator::Return {
                    values: vec![const_20],
                },
            ),
        );

        // Case 3: return 30
        blocks.insert(
            case3_block,
            make_block(
                case3_block,
                vec![HirInstruction::Binary {
                    op: BinaryOp::Sub,
                    result: const_30,
                    ty: HirType::I32,
                    left: param_x,
                    right: param_x,
                }],
                HirTerminator::Return {
                    values: vec![const_30],
                },
            ),
        );

        // Default: return 0
        blocks.insert(
            default_block,
            make_block(
                default_block,
                vec![HirInstruction::Binary {
                    op: BinaryOp::And,
                    result: const_0,
                    ty: HirType::I32,
                    left: param_x,
                    right: param_x,
                }],
                HirTerminator::Return {
                    values: vec![const_0],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("classify"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_x,
                    name: make_name("x"),
                    ty: HirType::I32,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::I32],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block,
            blocks,
            locals: HashMap::new(),
            values: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile switch: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify the LLVM IR contains switch instruction
        assert!(
            llvm_ir.contains("switch i32"),
            "LLVM IR should contain switch instruction"
        );
        assert!(
            llvm_ir.contains("label"),
            "LLVM IR should contain case labels"
        );

        // JIT compile and execute
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let classify_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32) -> i32>(&func_name)
                .expect("Failed to get function");

            // Test switch cases
            // Case 1: x=1 returns x+x = 2
            let result = classify_fn.call(1);
            assert_eq!(result, 2, "classify(1) should return 1+1=2");

            // Case 2: x=2 returns x*x = 4
            let result = classify_fn.call(2);
            assert_eq!(result, 4, "classify(2) should return 2*2=4");

            // Case 3: x=3 returns x-x = 0
            let result = classify_fn.call(3);
            assert_eq!(result, 0, "classify(3) should return 3-3=0");

            // Default: x=5 returns x&x = 5
            let result = classify_fn.call(5);
            assert_eq!(result, 5, "classify(5) should return 5&5=5");

            println!("✅ Switch statement: JIT execution verified!");
        }

        println!("✅ Switch statement compiled successfully");
        println!("LLVM IR:\n{}", llvm_ir);
    }

    #[test]
    fn test_type_conversions() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_casts");

        // Create function that tests various casts:
        // fn test_casts(x: i64) -> f64 {
        //     let y: i32 = trunc x;
        //     let z: f64 = sitofp y;
        //     return z;
        // }
        let func_id = HirId::new();
        let param_x = HirId::new();
        let trunc_result = HirId::new();
        let sitofp_result = HirId::new();
        let block_id = HirId::new();

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![
                    // Truncate i64 to i32
                    HirInstruction::Cast {
                        op: CastOp::Trunc,
                        result: trunc_result,
                        ty: HirType::I32,
                        operand: param_x,
                    },
                    // Convert i32 to f64
                    HirInstruction::Cast {
                        op: CastOp::SiToFp,
                        result: sitofp_result,
                        ty: HirType::F64,
                        operand: trunc_result,
                    },
                ],
                HirTerminator::Return {
                    values: vec![sitofp_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_casts"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_x,
                    name: make_name("x"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::F64],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: block_id,
            blocks,
            locals: HashMap::new(),
            values: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile casts: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify the LLVM IR contains cast instructions
        assert!(
            llvm_ir.contains("trunc"),
            "LLVM IR should contain trunc instruction"
        );
        assert!(
            llvm_ir.contains("sitofp"),
            "LLVM IR should contain sitofp instruction"
        );

        // JIT compile and execute
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define double @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let cast_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i64) -> f64>(&func_name)
                .expect("Failed to get function");

            // Test: trunc(100) to i32, then sitofp to f64 = 100.0
            let result = cast_fn.call(100);
            assert_eq!(result, 100.0, "cast(100) should be 100.0");

            // Test: large value that gets truncated (keeps lower 32 bits)
            // 0x1_0000_0042 truncated to i32 = 0x42 = 66, then to f64 = 66.0
            let result = cast_fn.call(0x1_0000_0042);
            assert_eq!(result, 66.0, "cast(0x1_0000_0042) should be 66.0");

            // Test: negative value
            let result = cast_fn.call(-50);
            assert_eq!(result, -50.0, "cast(-50) should be -50.0");

            println!("✅ Type conversions: JIT execution verified!");
        }

        println!("✅ Type conversions compiled successfully");
        println!("LLVM IR:\n{}", llvm_ir);
    }

    #[test]
    fn test_memory_operations() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_memory");

        // Create function that tests memory operations:
        // fn test_memory(x: i32) -> i32 {
        //     let ptr = alloca i32;
        //     store x, ptr;
        //     let y = load ptr;
        //     return y;
        // }
        let func_id = HirId::new();
        let param_x = HirId::new();
        let ptr_result = HirId::new();
        let load_result = HirId::new();
        let block_id = HirId::new();

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![
                    // Allocate stack space for i32
                    HirInstruction::Alloca {
                        result: ptr_result,
                        ty: HirType::I32,
                        count: None,
                        align: 4,
                    },
                    // Store parameter to stack
                    HirInstruction::Store {
                        value: param_x,
                        ptr: ptr_result,
                        align: 4,
                        volatile: false,
                    },
                    // Load from stack
                    HirInstruction::Load {
                        result: load_result,
                        ty: HirType::I32,
                        ptr: ptr_result,
                        align: 4,
                        volatile: false,
                    },
                ],
                HirTerminator::Return {
                    values: vec![load_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_memory"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_x,
                    name: make_name("x"),
                    ty: HirType::I32,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::I32],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: block_id,
            blocks,
            locals: HashMap::new(),
            values: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile memory ops: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify the LLVM IR contains memory instructions
        assert!(
            llvm_ir.contains("alloca"),
            "LLVM IR should contain alloca instruction"
        );
        assert!(
            llvm_ir.contains("store"),
            "LLVM IR should contain store instruction"
        );
        assert!(
            llvm_ir.contains("load"),
            "LLVM IR should contain load instruction"
        );

        // JIT compile and execute
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let mem_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32) -> i32>(&func_name)
                .expect("Failed to get function");

            // Test: stores value to stack, then loads it back
            // Should return the same value that was passed in
            let result = mem_fn.call(42);
            assert_eq!(result, 42, "Memory test should return stored value 42");

            let result = mem_fn.call(-999);
            assert_eq!(result, -999, "Memory test should return stored value -999");

            let result = mem_fn.call(0);
            assert_eq!(result, 0, "Memory test should return stored value 0");

            println!("✅ Memory operations: JIT execution verified!");
        }

        println!("✅ Memory operations compiled successfully");
        println!("LLVM IR:\n{}", llvm_ir);
    }

    #[test]
    fn test_loop_compilation() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_loop");

        // Create function with a simple counting loop:
        // fn count_to_ten(start: i32) -> i32 {
        //   loop_header:
        //     if start < 10 goto loop_body else goto loop_exit
        //   loop_body:
        //     start = start + 1
        //     goto loop_header
        //   loop_exit:
        //     return start
        // }
        let func_id = HirId::new();
        let param_start = HirId::new();
        let cmp_result = HirId::new();
        let add_result = HirId::new();
        let const_10 = HirId::new();
        let const_1 = HirId::new();

        let loop_header = HirId::new();
        let loop_body = HirId::new();
        let loop_exit = HirId::new();

        let mut blocks = HashMap::new();

        // Loop header: check condition
        blocks.insert(
            loop_header,
            make_block(
                loop_header,
                vec![
                    // Create constant 10 for comparison
                    HirInstruction::Binary {
                        op: BinaryOp::Add,
                        result: const_10,
                        ty: HirType::I32,
                        left: param_start,
                        right: param_start, // Dummy operation to create a value
                    },
                    // Compare: start < 10
                    HirInstruction::Binary {
                        op: BinaryOp::Lt,
                        result: cmp_result,
                        ty: HirType::Bool,
                        left: param_start,
                        right: const_10,
                    },
                ],
                HirTerminator::CondBranch {
                    condition: cmp_result,
                    true_target: loop_body,
                    false_target: loop_exit,
                },
            ),
        );

        // Loop body: increment and loop back
        blocks.insert(
            loop_body,
            make_block(
                loop_body,
                vec![
                    // Create constant 1
                    HirInstruction::Binary {
                        op: BinaryOp::Sub,
                        result: const_1,
                        ty: HirType::I32,
                        left: param_start,
                        right: param_start, // Dummy: would be 0, but we need a value
                    },
                    // Increment: start + 1
                    HirInstruction::Binary {
                        op: BinaryOp::Add,
                        result: add_result,
                        ty: HirType::I32,
                        left: param_start,
                        right: const_1,
                    },
                ],
                HirTerminator::Branch {
                    target: loop_header,
                },
            ),
        );

        // Loop exit: return
        blocks.insert(
            loop_exit,
            make_block(
                loop_exit,
                vec![],
                HirTerminator::Return {
                    values: vec![param_start],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("count_to_ten"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_start,
                    name: make_name("start"),
                    ty: HirType::I32,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::I32],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: loop_header,
            blocks,
            locals: HashMap::new(),
            values: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(result.is_ok(), "Failed to compile loop: {:?}", result.err());

        let llvm_ir = result.unwrap();

        // Verify the LLVM IR contains loop structure
        assert!(
            llvm_ir.contains("br label"),
            "LLVM IR should contain unconditional branch"
        );
        assert!(
            llvm_ir.contains("br i1"),
            "LLVM IR should contain conditional branch"
        );

        // NOTE: This loop doesn't actually work correctly because it lacks proper phi nodes
        // to update the loop variable. The test just verifies that loop structures *compile*.
        // Once phi node incoming edges are implemented, we can add proper JIT execution tests.

        println!("✅ Loop compiled successfully (structure only, phi nodes TODO)");
        println!("LLVM IR:\n{}", llvm_ir);
        println!("⚠️  Note: Loop execution not tested - requires phi node incoming edges");
    }

    #[test]
    fn test_proper_loop_with_phi() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_loop_phi");

        // Create a proper counting loop with phi nodes:
        // fn count_up(start: i32, limit: i32) -> i32 {
        //   loop_header:
        //     i = phi [start, entry], [i_next, loop_body]
        //     if i < limit goto loop_body else goto loop_exit
        //   loop_body:
        //     i_next = i + 1
        //     goto loop_header
        //   loop_exit:
        //     return i
        // }
        let func_id = HirId::new();
        let param_start = HirId::new();
        let param_limit = HirId::new();
        let phi_i = HirId::new();
        let cmp_result = HirId::new();
        let i_next = HirId::new();
        let const_1 = HirId::new();

        let entry_block = HirId::new();
        let loop_header = HirId::new();
        let loop_body = HirId::new();
        let loop_exit = HirId::new();

        let mut blocks = HashMap::new();

        // Entry block: define constant 1, then branch to loop header
        blocks.insert(
            entry_block,
            make_block(
                entry_block,
                vec![],
                HirTerminator::Branch {
                    target: loop_header,
                },
            ),
        );

        // Loop header: phi node and condition check
        let phi_node = HirPhi {
            result: phi_i,
            ty: HirType::I32,
            incoming: vec![
                (param_start, entry_block), // First iteration: use start
                (i_next, loop_body),        // Later iterations: use incremented value
            ],
        };

        blocks.insert(
            loop_header,
            HirBlock {
                id: loop_header,
                label: None,
                phis: vec![phi_node],
                instructions: vec![
                    // Compare: i < limit
                    HirInstruction::Binary {
                        op: BinaryOp::Lt,
                        result: cmp_result,
                        ty: HirType::Bool,
                        left: phi_i,
                        right: param_limit,
                    },
                ],
                terminator: HirTerminator::CondBranch {
                    condition: cmp_result,
                    true_target: loop_body,
                    false_target: loop_exit,
                },
                dominance_frontier: HashSet::new(),
                predecessors: vec![entry_block, loop_body],
                successors: vec![loop_body, loop_exit],
            },
        );

        // Loop body: increment i
        blocks.insert(
            loop_body,
            make_block(
                loop_body,
                vec![
                    // i_next = i + 1
                    HirInstruction::Binary {
                        op: BinaryOp::Add,
                        result: i_next,
                        ty: HirType::I32,
                        left: phi_i,
                        right: const_1,
                    },
                ],
                HirTerminator::Branch {
                    target: loop_header,
                },
            ),
        );

        // Loop exit: return i
        blocks.insert(
            loop_exit,
            make_block(
                loop_exit,
                vec![],
                HirTerminator::Return {
                    values: vec![phi_i],
                },
            ),
        );

        // Define constant values
        let mut values = HashMap::new();
        values.insert(
            const_1,
            HirValue {
                id: const_1,
                ty: HirType::I32,
                kind: HirValueKind::Constant(HirConstant::I32(1)),
                uses: HashSet::new(),
                span: None,
            },
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("count_up"),
            signature: HirFunctionSignature {
                params: vec![
                    HirParam {
                        id: param_start,
                        name: make_name("start"),
                        ty: HirType::I32,
                        attributes: ParamAttributes::default(),
                    },
                    HirParam {
                        id: param_limit,
                        name: make_name("limit"),
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
            },
            entry_block,
            blocks,
            locals: HashMap::new(),
            values,
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile loop with phi: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify the LLVM IR contains phi node
        assert!(
            llvm_ir.contains("phi i32"),
            "LLVM IR should contain phi node"
        );
        assert!(
            llvm_ir.contains("br label"),
            "LLVM IR should contain unconditional branch"
        );
        assert!(
            llvm_ir.contains("br i1"),
            "LLVM IR should contain conditional branch"
        );

        // JIT compile and execute
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let count_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32, i32) -> i32>(&func_name)
                .expect("Failed to get function");

            // Test: count_up(0, 10) should return 10
            let result = count_fn.call(0, 10);
            assert_eq!(result, 10, "count_up(0, 10) should return 10");

            // Test: count_up(5, 8) should return 8
            let result = count_fn.call(5, 8);
            assert_eq!(result, 8, "count_up(5, 8) should return 8");

            // Test: count_up(7, 7) should return 7 (no iterations)
            let result = count_fn.call(7, 7);
            assert_eq!(result, 7, "count_up(7, 7) should return 7");

            // Test: count_up(10, 5) should return 10 (already >= limit)
            let result = count_fn.call(10, 5);
            assert_eq!(result, 10, "count_up(10, 5) should return 10");

            println!("✅ Proper loop with phi: JIT execution verified!");
        }

        println!("✅ Proper loop with phi nodes compiled and executed successfully!");
        println!("LLVM IR:\n{}", llvm_ir);
    }

    #[test]
    fn test_intrinsic_sqrt() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_sqrt");

        // Create a function: fn test_sqrt(x: f64) -> f64 { return sqrt(x); }
        let func_id = HirId::new();
        let param_x = HirId::new();
        let sqrt_result = HirId::new();
        let block_id = HirId::new();

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![HirInstruction::Call {
                    result: Some(sqrt_result),
                    callee: HirCallable::Intrinsic(Intrinsic::Sqrt),
                    args: vec![param_x],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                }],
                HirTerminator::Return {
                    values: vec![sqrt_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_sqrt"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_x,
                    name: make_name("x"),
                    ty: HirType::F64,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::F64],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: block_id,
            blocks,
            values: HashMap::new(),
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile sqrt intrinsic: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify LLVM IR contains sqrt intrinsic
        assert!(
            llvm_ir.contains("llvm.sqrt"),
            "Should contain sqrt intrinsic"
        );

        // JIT compile and execute
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define double @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let sqrt_fn = execution_engine
                .get_function::<unsafe extern "C" fn(f64) -> f64>(&func_name)
                .expect("Failed to get function");

            // Test: sqrt(4.0) = 2.0
            let result = sqrt_fn.call(4.0);
            assert!(
                (result - 2.0).abs() < 0.001,
                "sqrt(4.0) should be 2.0, got {}",
                result
            );

            // Test: sqrt(9.0) = 3.0
            let result = sqrt_fn.call(9.0);
            assert!(
                (result - 3.0).abs() < 0.001,
                "sqrt(9.0) should be 3.0, got {}",
                result
            );

            println!("✅ sqrt intrinsic: JIT execution verified!");
        }

        println!("✅ sqrt intrinsic test passed!");
    }

    #[test]
    fn test_intrinsic_ctpop() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_ctpop");

        // Create a function: fn test_ctpop(x: i32) -> i32 { return ctpop(x); }
        let func_id = HirId::new();
        let param_x = HirId::new();
        let ctpop_result = HirId::new();
        let block_id = HirId::new();

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![HirInstruction::Call {
                    result: Some(ctpop_result),
                    callee: HirCallable::Intrinsic(Intrinsic::Ctpop),
                    args: vec![param_x],
                    type_args: vec![],
                    const_args: vec![],
                    is_tail: false,
                }],
                HirTerminator::Return {
                    values: vec![ctpop_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_ctpop"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_x,
                    name: make_name("x"),
                    ty: HirType::I32,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::I32],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: block_id,
            blocks,
            values: HashMap::new(),
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        // Compile the module
        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile ctpop intrinsic: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();

        // Verify LLVM IR contains ctpop intrinsic
        assert!(
            llvm_ir.contains("llvm.ctpop"),
            "Should contain ctpop intrinsic"
        );

        // JIT compile and execute
        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let ctpop_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32) -> i32>(&func_name)
                .expect("Failed to get function");

            // Test: ctpop(0b1010) = 2
            assert_eq!(ctpop_fn.call(0b1010), 2, "ctpop(0b1010) should be 2");

            // Test: ctpop(0b1111) = 4
            assert_eq!(ctpop_fn.call(0b1111), 4, "ctpop(0b1111) should be 4");

            // Test: ctpop(0xFF) = 8
            assert_eq!(ctpop_fn.call(0xFF), 8, "ctpop(0xFF) should be 8");

            println!("✅ ctpop intrinsic: JIT execution verified!");
        }

        println!("✅ ctpop intrinsic test passed!");
    }

    #[test]
    fn test_intrinsic_malloc_free() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_malloc");

        // Create a function: fn test_malloc(size: i64) -> i64
        // Allocates memory, stores 42, loads it back, frees memory, returns loaded value
        let func_id = HirId::new();
        let param_size = HirId::new();
        let malloc_result = HirId::new();
        let const_42 = HirId::new();
        let load_result = HirId::new();
        let free_result = HirId::new();
        let block_id = HirId::new();

        // Create constant 42
        let mut values = HashMap::new();
        values.insert(
            const_42,
            HirValue {
                id: const_42,
                ty: HirType::I64,
                kind: HirValueKind::Constant(HirConstant::I64(42)),
                uses: HashSet::new(),
                span: None,
            },
        );

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![
                    // malloc_result = malloc(size)
                    HirInstruction::Call {
                        result: Some(malloc_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                        args: vec![param_size],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                    // store 42 to malloc_result
                    HirInstruction::Store {
                        value: const_42,
                        ptr: malloc_result,
                        align: 8,
                        volatile: false,
                    },
                    // load from malloc_result
                    HirInstruction::Load {
                        result: load_result,
                        ty: HirType::I64,
                        ptr: malloc_result,
                        align: 8,
                        volatile: false,
                    },
                    // free(malloc_result)
                    HirInstruction::Call {
                        result: Some(free_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Free),
                        args: vec![malloc_result],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                ],
                HirTerminator::Return {
                    values: vec![load_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_malloc"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_size,
                    name: make_name("size"),
                    ty: HirType::I64,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::I64],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: block_id,
            blocks,
            values,
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile malloc/free: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();
        assert!(llvm_ir.contains("@malloc"), "Should contain malloc");
        assert!(llvm_ir.contains("@free"), "Should contain free");

        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i64 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let malloc_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i64) -> i64>(&func_name)
                .expect("Failed to get function");

            // Test: allocate 8 bytes, should return 42
            let result = malloc_fn.call(8);
            assert_eq!(result, 42, "Should read back stored value 42");

            println!("✅ malloc/free intrinsics: JIT execution verified!");
        }

        println!("✅ malloc/free intrinsic test passed!");
    }

    #[test]
    fn test_intrinsic_math_ops() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_math");

        // Create a function: fn test_math(x: f64) -> f64
        // Returns sin(x) + cos(x) + pow(x, 2.0)
        let func_id = HirId::new();
        let param_x = HirId::new();
        let sin_result = HirId::new();
        let cos_result = HirId::new();
        let const_2 = HirId::new();
        let pow_result = HirId::new();
        let add1_result = HirId::new();
        let add2_result = HirId::new();
        let block_id = HirId::new();

        // Create constant 2.0
        let mut values = HashMap::new();
        values.insert(
            const_2,
            HirValue {
                id: const_2,
                ty: HirType::F64,
                kind: HirValueKind::Constant(HirConstant::F64(2.0)),
                uses: HashSet::new(),
                span: None,
            },
        );

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![
                    // sin_result = sin(x)
                    HirInstruction::Call {
                        result: Some(sin_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Sin),
                        args: vec![param_x],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                    // cos_result = cos(x)
                    HirInstruction::Call {
                        result: Some(cos_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Cos),
                        args: vec![param_x],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                    // pow_result = pow(x, 2.0)
                    HirInstruction::Call {
                        result: Some(pow_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Pow),
                        args: vec![param_x, const_2],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                    // add1 = sin_result + cos_result
                    HirInstruction::Binary {
                        result: add1_result,
                        op: BinaryOp::Add,
                        left: sin_result,
                        right: cos_result,
                        ty: HirType::F64,
                    },
                    // add2 = add1 + pow_result
                    HirInstruction::Binary {
                        result: add2_result,
                        op: BinaryOp::Add,
                        left: add1_result,
                        right: pow_result,
                        ty: HirType::F64,
                    },
                ],
                HirTerminator::Return {
                    values: vec![add2_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_math"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_x,
                    name: make_name("x"),
                    ty: HirType::F64,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::F64],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: block_id,
            blocks,
            values,
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile math intrinsics: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();
        assert!(llvm_ir.contains("llvm.sin"), "Should contain sin");
        assert!(llvm_ir.contains("llvm.cos"), "Should contain cos");
        assert!(llvm_ir.contains("llvm.pow"), "Should contain pow");

        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define double @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let math_fn = execution_engine
                .get_function::<unsafe extern "C" fn(f64) -> f64>(&func_name)
                .expect("Failed to get function");

            // Test with x = 1.0
            // Expected: sin(1.0) + cos(1.0) + pow(1.0, 2.0)
            let result = math_fn.call(1.0);
            let expected = 1.0_f64.sin() + 1.0_f64.cos() + 1.0_f64.powf(2.0);
            assert!(
                (result - expected).abs() < 0.001,
                "Expected {}, got {}",
                expected,
                result
            );

            println!("✅ sin/cos/pow intrinsics: JIT execution verified!");
        }

        println!("✅ Math intrinsics test passed!");
    }

    #[test]
    fn test_intrinsic_bit_ops() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_bitops");

        // Create a function: fn test_bitops(x: i32) -> i32
        // Returns ctlz(x) + cttz(x) + bswap(x)
        let func_id = HirId::new();
        let param_x = HirId::new();
        let ctlz_result = HirId::new();
        let cttz_result = HirId::new();
        let bswap_result = HirId::new();
        let add1_result = HirId::new();
        let add2_result = HirId::new();
        let block_id = HirId::new();

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![
                    // ctlz_result = ctlz(x)
                    HirInstruction::Call {
                        result: Some(ctlz_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Ctlz),
                        args: vec![param_x],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                    // cttz_result = cttz(x)
                    HirInstruction::Call {
                        result: Some(cttz_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Cttz),
                        args: vec![param_x],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                    // bswap_result = bswap(x)
                    HirInstruction::Call {
                        result: Some(bswap_result),
                        callee: HirCallable::Intrinsic(Intrinsic::Bswap),
                        args: vec![param_x],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                    // add1 = ctlz_result + cttz_result
                    HirInstruction::Binary {
                        result: add1_result,
                        op: BinaryOp::Add,
                        left: ctlz_result,
                        right: cttz_result,
                        ty: HirType::I32,
                    },
                    // add2 = add1 + bswap_result
                    HirInstruction::Binary {
                        result: add2_result,
                        op: BinaryOp::Add,
                        left: add1_result,
                        right: bswap_result,
                        ty: HirType::I32,
                    },
                ],
                HirTerminator::Return {
                    values: vec![add2_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_bitops"),
            signature: HirFunctionSignature {
                params: vec![HirParam {
                    id: param_x,
                    name: make_name("x"),
                    ty: HirType::I32,
                    attributes: ParamAttributes::default(),
                }],
                returns: vec![HirType::I32],
                type_params: vec![],
                const_params: vec![],
                lifetime_params: vec![],
                is_variadic: false,
                is_async: false,
            },
            entry_block: block_id,
            blocks,
            values: HashMap::new(),
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile bit intrinsics: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();
        assert!(llvm_ir.contains("llvm.ctlz"), "Should contain ctlz");
        assert!(llvm_ir.contains("llvm.cttz"), "Should contain cttz");
        assert!(llvm_ir.contains("llvm.bswap"), "Should contain bswap");

        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let bitops_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32) -> i32>(&func_name)
                .expect("Failed to get function");

            // Test with 0x00FF (leading zeros: 24, trailing zeros: 0, bswap: 0xFF000000)
            let result = bitops_fn.call(0x00FF);
            // ctlz(0x00FF) = 24, cttz(0x00FF) = 0, bswap(0x00FF) = 0xFF000000 = -16777216
            let expected = 24 + 0 + (0xFF000000_u32 as i32);
            assert_eq!(result, expected, "Expected {}, got {}", expected, result);

            println!("✅ ctlz/cttz/bswap intrinsics: JIT execution verified!");
        }

        println!("✅ Bit operations intrinsics test passed!");
    }

    #[test]
    fn test_struct_operations() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_struct");

        // Create a function: fn test_struct(a: i32, b: i32) -> i32
        // Creates a struct {i32, i32}, sets fields to (a, b), extracts and returns a + b
        let func_id = HirId::new();
        let param_a = HirId::new();
        let param_b = HirId::new();
        let undef_struct = HirId::new();
        let insert1_result = HirId::new();
        let insert2_result = HirId::new();
        let extract_a = HirId::new();
        let extract_b = HirId::new();
        let add_result = HirId::new();
        let block_id = HirId::new();

        // Define struct type: {i32, i32}
        let struct_ty = HirType::Struct(HirStructType {
            name: None,
            fields: vec![HirType::I32, HirType::I32],
            packed: false,
        });

        // Create an undefined struct value
        let mut values = HashMap::new();
        values.insert(
            undef_struct,
            HirValue {
                id: undef_struct,
                ty: struct_ty.clone(),
                kind: HirValueKind::Instruction,
                uses: HashSet::new(),
                span: None,
            },
        );

        let mut blocks = HashMap::new();
        blocks.insert(
            block_id,
            make_block(
                block_id,
                vec![
                    // insert1 = insertvalue undef, a, 0
                    HirInstruction::InsertValue {
                        result: insert1_result,
                        ty: struct_ty.clone(),
                        aggregate: undef_struct,
                        value: param_a,
                        indices: vec![0],
                    },
                    // insert2 = insertvalue insert1, b, 1
                    HirInstruction::InsertValue {
                        result: insert2_result,
                        ty: struct_ty.clone(),
                        aggregate: insert1_result,
                        value: param_b,
                        indices: vec![1],
                    },
                    // extract_a = extractvalue insert2, 0
                    HirInstruction::ExtractValue {
                        result: extract_a,
                        ty: HirType::I32,
                        aggregate: insert2_result,
                        indices: vec![0],
                    },
                    // extract_b = extractvalue insert2, 1
                    HirInstruction::ExtractValue {
                        result: extract_b,
                        ty: HirType::I32,
                        aggregate: insert2_result,
                        indices: vec![1],
                    },
                    // add_result = extract_a + extract_b
                    HirInstruction::Binary {
                        result: add_result,
                        op: BinaryOp::Add,
                        left: extract_a,
                        right: extract_b,
                        ty: HirType::I32,
                    },
                ],
                HirTerminator::Return {
                    values: vec![add_result],
                },
            ),
        );

        let function = HirFunction {
            id: func_id,
            name: make_name("test_struct"),
            signature: HirFunctionSignature {
                params: vec![
                    HirParam {
                        id: param_a,
                        name: make_name("a"),
                        ty: HirType::I32,
                        attributes: ParamAttributes::default(),
                    },
                    HirParam {
                        id: param_b,
                        name: make_name("b"),
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
            },
            entry_block: block_id,
            blocks,
            values,
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(func_id, function);

        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile struct operations: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();
        assert!(
            llvm_ir.contains("insertvalue"),
            "Should contain insertvalue"
        );
        assert!(
            llvm_ir.contains("extractvalue"),
            "Should contain extractvalue"
        );

        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            let func_name = llvm_ir
                .lines()
                .find(|line| line.contains("define i32 @"))
                .and_then(|line| {
                    let after_at = line.split('@').nth(1)?;
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).map(|s| s.to_string())
                    } else {
                        after_at.split('(').next().map(|s| s.to_string())
                    }
                })
                .expect("Could not find function name");

            let struct_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32, i32) -> i32>(&func_name)
                .expect("Failed to get function");

            // Test: struct with fields (10, 20) should return 30
            let result = struct_fn.call(10, 20);
            assert_eq!(result, 30, "Expected 30, got {}", result);

            // Test: struct with fields (100, 200) should return 300
            let result = struct_fn.call(100, 200);
            assert_eq!(result, 300, "Expected 300, got {}", result);

            println!("✅ Struct operations: JIT execution verified!");
        }

        println!("✅ Struct operations test passed!");
    }

    #[test]
    fn test_indirect_call() {
        let context = Context::create();
        let mut backend = LLVMBackend::new(&context, "test_indirect");

        // Create two functions:
        // 1. fn add(a: i32, b: i32) -> i32 { return a + b; }
        // 2. fn caller(f: fn(i32, i32) -> i32, x: i32, y: i32) -> i32 { return f(x, y); }

        let add_func_id = HirId::new();
        let caller_func_id = HirId::new();

        // === Create the "add" function ===
        let add_param_a = HirId::new();
        let add_param_b = HirId::new();
        let add_result = HirId::new();
        let add_block_id = HirId::new();

        let mut add_blocks = HashMap::new();
        add_blocks.insert(
            add_block_id,
            make_block(
                add_block_id,
                vec![HirInstruction::Binary {
                    result: add_result,
                    op: BinaryOp::Add,
                    left: add_param_a,
                    right: add_param_b,
                    ty: HirType::I32,
                }],
                HirTerminator::Return {
                    values: vec![add_result],
                },
            ),
        );

        let add_function = HirFunction {
            id: add_func_id,
            name: make_name("add"),
            signature: HirFunctionSignature {
                params: vec![
                    HirParam {
                        id: add_param_a,
                        name: make_name("a"),
                        ty: HirType::I32,
                        attributes: ParamAttributes::default(),
                    },
                    HirParam {
                        id: add_param_b,
                        name: make_name("b"),
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
            },
            entry_block: add_block_id,
            blocks: add_blocks,
            values: HashMap::new(),
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        // === Create the "caller" function ===
        let caller_param_f = HirId::new();
        let caller_param_x = HirId::new();
        let caller_param_y = HirId::new();
        let call_result = HirId::new();
        let caller_block_id = HirId::new();

        let mut caller_blocks = HashMap::new();
        caller_blocks.insert(
            caller_block_id,
            make_block(
                caller_block_id,
                vec![
                    // call_result = f(x, y)
                    HirInstruction::Call {
                        result: Some(call_result),
                        callee: HirCallable::Indirect(caller_param_f),
                        args: vec![caller_param_x, caller_param_y],
                        type_args: vec![],
                        const_args: vec![],
                        is_tail: false,
                    },
                ],
                HirTerminator::Return {
                    values: vec![call_result],
                },
            ),
        );

        // Define function pointer type: fn(i32, i32) -> i32
        let func_ptr_ty = HirType::Function(Box::new(HirFunctionType {
            params: vec![HirType::I32, HirType::I32],
            returns: vec![HirType::I32],
            lifetime_params: vec![],
            is_variadic: false,
        }));

        let caller_function = HirFunction {
            id: caller_func_id,
            name: make_name("caller"),
            signature: HirFunctionSignature {
                params: vec![
                    HirParam {
                        id: caller_param_f,
                        name: make_name("f"),
                        ty: func_ptr_ty.clone(),
                        attributes: ParamAttributes::default(),
                    },
                    HirParam {
                        id: caller_param_x,
                        name: make_name("x"),
                        ty: HirType::I32,
                        attributes: ParamAttributes::default(),
                    },
                    HirParam {
                        id: caller_param_y,
                        name: make_name("y"),
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
            },
            entry_block: caller_block_id,
            blocks: caller_blocks,
            values: HashMap::new(),
            locals: HashMap::new(),
            previous_version: None,
            is_external: false,
            calling_convention: CallingConvention::Fast,
            attributes: FunctionAttributes::default(),
        };

        let mut module = create_test_module();
        module.functions.insert(add_func_id, add_function);
        module.functions.insert(caller_func_id, caller_function);

        let result = backend.compile_module(&module);
        assert!(
            result.is_ok(),
            "Failed to compile indirect call: {:?}",
            result.err()
        );

        let llvm_ir = result.unwrap();
        println!("LLVM IR:\n{}", llvm_ir);
        assert!(llvm_ir.contains("call"), "Should contain call instruction");

        let execution_engine = backend
            .module()
            .create_jit_execution_engine(OptimizationLevel::None)
            .expect("Failed to create execution engine");

        unsafe {
            // Extract function names from LLVM IR
            // First function is add: define i32 @"func_HirId(...)"(i32 %param_0, i32 %param_1)
            // Second function is caller: define i32 @"func_HirId(...)"(ptr %param_0, i32 %param_1, i32 %param_2)

            let mut function_names = llvm_ir
                .lines()
                .filter(|line| line.starts_with("define i32 @"))
                .map(|line| {
                    let after_at = line.split('@').nth(1).unwrap();
                    if after_at.starts_with('"') {
                        after_at.split('"').nth(1).unwrap().to_string()
                    } else {
                        after_at.split('(').next().unwrap().to_string()
                    }
                });

            // First is add (takes two i32 params)
            let add_fn_name = function_names.next().expect("Could not find add function");
            // Second is caller (takes ptr + two i32 params)
            let caller_fn_name = function_names
                .next()
                .expect("Could not find caller function");

            let add_fn = execution_engine
                .get_function::<unsafe extern "C" fn(i32, i32) -> i32>(&add_fn_name)
                .expect("Failed to get add function");

            type CallerFn =
                unsafe extern "C" fn(unsafe extern "C" fn(i32, i32) -> i32, i32, i32) -> i32;
            let caller_fn = execution_engine
                .get_function::<CallerFn>(&caller_fn_name)
                .expect("Failed to get caller function");

            // Test: call caller with add function pointer
            let result = caller_fn.call(add_fn.as_raw(), 10, 20);
            assert_eq!(result, 30, "Expected 30, got {}", result);

            // Test: call caller with add function pointer, different args
            let result = caller_fn.call(add_fn.as_raw(), 100, 200);
            assert_eq!(result, 300, "Expected 300, got {}", result);

            println!("✅ Indirect call: JIT execution verified!");
        }

        println!("✅ Indirect call test passed!");
    }
}
