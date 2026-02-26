//! Debug test for continue statement in while loops
//! This test creates a minimal HIR representation to debug the infinite loop issue

use zyntax_compiler::cranelift_backend::CraneliftBackend;
use zyntax_compiler::hir::*;
use zyntax_typed_ast::{arena::AstArena, InternedString};

fn create_test_string(s: &str) -> InternedString {
    let mut arena = AstArena::new();
    arena.intern_string(s)
}

/// Test function that mimics:
/// ```
/// fn test_continue(n: i32) -> i32 {
///     var i = 0;
///     var sum = 0;
///     while (i < n) {
///         i = i + 1;
///         if (i == 2) {
///             continue;  // Skip adding 2 to sum
///         }
///         sum = sum + i;
///     }
///     return sum;
/// }
/// ```
/// With n=4, expected: 1 + 3 + 4 = 8 (skips 2)
#[test]
fn test_while_with_continue() {
    let func = create_continue_function();

    // Print the HIR structure for debugging
    println!("=== HIR Function Structure ===");
    println!("Function: {}", func.name);
    println!("Blocks:");
    for (block_id, block) in &func.blocks {
        println!("  Block {:?}:", block_id);
        println!("    Phis: {}", block.phis.len());
        for phi in &block.phis {
            println!("      {:?} = phi({:?})", phi.result, phi.incoming);
        }
        println!("    Instructions: {}", block.instructions.len());
        for inst in &block.instructions {
            println!("      {:?}", inst);
        }
        println!("    Terminator: {:?}", block.terminator);
    }

    use cranelift_module::Module;

    let func_id = func.id;
    let mut backend = CraneliftBackend::new().expect("Failed to create Cranelift backend");

    println!("\n=== Compiling HIR to Cranelift IR ===");
    backend
        .compile_function(func_id, &func)
        .expect("Failed to compile function");

    // Print the Cranelift IR
    println!("\n=== Cranelift IR (HIR test - WORKING) ===");
    println!("{}", backend.get_ir_string());

    backend
        .finalize_definitions()
        .expect("Failed to finalize definitions");

    let func_ptr = backend
        .get_function_ptr(func_id)
        .expect("Failed to get function pointer");

    let test_fn: extern "C" fn(i32) -> i32 = unsafe { std::mem::transmute(func_ptr) };

    // Test with n=4: should return 1 + 3 + 4 = 8 (skips 2)
    println!("\n=== Executing test_continue(4) ===");
    let result = test_fn(4);
    println!("Result: {}", result);
    assert_eq!(result, 8, "test_continue(4) should equal 8");

    println!("✅ While loop with continue executed correctly via JIT");
}

fn create_continue_function() -> HirFunction {
    let name = create_test_string("test_continue");

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

    // Create blocks
    let entry_block = func.entry_block;
    let loop_header = func.create_block();
    let loop_body = func.create_block();
    let if_check = func.create_block();
    let continue_block = func.create_block(); // Block for continue path
    let merge_block = func.create_block(); // Block for normal path (sum += i)
    let exit_block = func.create_block();

    let param_n = func.create_value(HirType::I32, HirValueKind::Parameter(0));

    // Constants
    let zero = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(0)));
    let one = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(1)));
    let two = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(2)));

    // Entry: i = 0, sum = 0, jump to loop header
    let entry = func.blocks.get_mut(&entry_block).unwrap();
    entry.set_terminator(HirTerminator::Branch {
        target: loop_header,
    });

    // Loop header: phi nodes for i and sum, check i < n
    let i_phi = func.create_value(HirType::I32, HirValueKind::Instruction);
    let sum_phi = func.create_value(HirType::I32, HirValueKind::Instruction);
    let cmp_i_lt_n = func.create_value(HirType::Bool, HirValueKind::Instruction);

    let cmp_inst = HirInstruction::Binary {
        op: BinaryOp::Lt,
        result: cmp_i_lt_n,
        ty: HirType::I32,
        left: i_phi,
        right: param_n,
    };

    let header = func.blocks.get_mut(&loop_header).unwrap();
    header.add_phi(HirPhi {
        result: i_phi,
        ty: HirType::I32,
        incoming: vec![
            (zero, entry_block),            // Initial: i = 0
            (HirId::new(), merge_block),    // Will be set to i_after_add
            (HirId::new(), continue_block), // Will be set to i_after_add
        ],
    });
    header.add_phi(HirPhi {
        result: sum_phi,
        ty: HirType::I32,
        incoming: vec![
            (zero, entry_block),            // Initial: sum = 0
            (HirId::new(), merge_block),    // Will be set to new_sum
            (HirId::new(), continue_block), // Will be set to sum_phi from prev iteration
        ],
    });
    header.add_instruction(cmp_inst);
    header.set_terminator(HirTerminator::CondBranch {
        condition: cmp_i_lt_n,
        true_target: loop_body,
        false_target: exit_block,
    });

    // Loop body: i = i + 1, then jump to if_check
    let i_plus_1 = func.create_value(HirType::I32, HirValueKind::Instruction);
    let add_inst = HirInstruction::Binary {
        op: BinaryOp::Add,
        result: i_plus_1,
        ty: HirType::I32,
        left: i_phi,
        right: one,
    };

    let body = func.blocks.get_mut(&loop_body).unwrap();
    body.add_instruction(add_inst);
    body.set_terminator(HirTerminator::Branch { target: if_check });

    // If check: if (i == 2) goto continue_block else goto merge_block
    let cmp_i_eq_2 = func.create_value(HirType::Bool, HirValueKind::Instruction);
    let eq_inst = HirInstruction::Binary {
        op: BinaryOp::Eq,
        result: cmp_i_eq_2,
        ty: HirType::I32,
        left: i_plus_1,
        right: two,
    };

    let if_block = func.blocks.get_mut(&if_check).unwrap();
    if_block.add_instruction(eq_inst);
    if_block.set_terminator(HirTerminator::CondBranch {
        condition: cmp_i_eq_2,
        true_target: continue_block,
        false_target: merge_block,
    });

    // Continue block: jump back to loop header (skip sum update)
    let cont_block = func.blocks.get_mut(&continue_block).unwrap();
    cont_block.set_terminator(HirTerminator::Branch {
        target: loop_header,
    });

    // Merge block: sum = sum + i, then jump back to loop header
    let new_sum = func.create_value(HirType::I32, HirValueKind::Instruction);
    let add_sum_inst = HirInstruction::Binary {
        op: BinaryOp::Add,
        result: new_sum,
        ty: HirType::I32,
        left: sum_phi,
        right: i_plus_1,
    };

    let merge = func.blocks.get_mut(&merge_block).unwrap();
    merge.add_instruction(add_sum_inst);
    merge.set_terminator(HirTerminator::Branch {
        target: loop_header,
    });

    // Fix phi node incoming values
    let header = func.blocks.get_mut(&loop_header).unwrap();
    header.phis[0].incoming[1] = (i_plus_1, merge_block);
    header.phis[0].incoming[2] = (i_plus_1, continue_block);
    header.phis[1].incoming[1] = (new_sum, merge_block);
    header.phis[1].incoming[2] = (sum_phi, continue_block); // Continue: sum unchanged, use phi value

    // Exit block: return sum
    let exit = func.blocks.get_mut(&exit_block).unwrap();
    exit.set_terminator(HirTerminator::Return {
        values: vec![sum_phi],
    });

    func
}
