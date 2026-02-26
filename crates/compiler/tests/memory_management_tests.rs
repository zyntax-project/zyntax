//! Tests for memory management infrastructure

use zyntax_compiler::{
    hir::*, ARCManager, AllocationInfo, DropManager, EscapeAnalysis, EscapeInfo, MemoryContext,
    MemoryStrategy, RefCountInfo,
};
use zyntax_typed_ast::arena::AstArena;

fn create_test_arena() -> AstArena {
    AstArena::new()
}

fn intern_str(arena: &mut AstArena, s: &str) -> zyntax_typed_ast::InternedString {
    arena.intern_string(s)
}

#[test]
fn test_arc_instrumentation() {
    let mut arena = create_test_arena();
    let mut arc_manager = ARCManager::new();

    // Create a simple function that allocates and uses pointers
    let signature = HirFunctionSignature {
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

    let mut func = HirFunction::new(intern_str(&mut arena, "test_arc"), signature);

    // Add a malloc instruction
    let ptr_type = HirType::Ptr(Box::new(HirType::I32));
    let malloc_result = func.create_value(ptr_type.clone(), HirValueKind::Instruction);
    let size_const = func.create_value(HirType::I64, HirValueKind::Constant(HirConstant::I64(4)));

    let malloc_inst = HirInstruction::Call {
        result: Some(malloc_result),
        callee: HirCallable::Intrinsic(Intrinsic::Malloc),
        args: vec![size_const],
        type_args: vec![],
        const_args: vec![],
        is_tail: false,
    };

    let block = func.blocks.get_mut(&func.entry_block).unwrap();
    block.add_instruction(malloc_inst);
    block.set_terminator(HirTerminator::Return {
        values: vec![malloc_result],
    });

    // Instrument the function
    let result = arc_manager.instrument_function(&mut func);
    assert!(result.is_ok());

    // Check that the function was marked as instrumented
    assert!(arc_manager.arc_functions.contains(&func.id));
}

#[test]
fn test_drop_manager() {
    let mut arena = create_test_arena();
    let mut drop_manager = DropManager::new();

    // Register a type with a destructor
    let custom_type = HirType::Struct(HirStructType {
        name: Some(intern_str(&mut arena, "CustomType")),
        fields: vec![HirType::Ptr(Box::new(HirType::I32))],
        packed: false,
    });

    let destructor_id = HirId::new();
    drop_manager.register_destructor(custom_type.clone(), destructor_id);

    // Create a function that uses this type
    let signature = HirFunctionSignature {
        params: vec![],
        returns: vec![],
        type_params: vec![],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let mut func = HirFunction::new(intern_str(&mut arena, "test_drop"), signature);

    // Create a value of the custom type
    let custom_value = func.create_value(custom_type, HirValueKind::Instruction);

    // Add a simple return
    let block = func.blocks.get_mut(&func.entry_block).unwrap();
    block.set_terminator(HirTerminator::Return { values: vec![] });

    // Insert drops
    let result = drop_manager.insert_drops(&mut func);
    assert!(result.is_ok());

    // Check that drop was identified
    assert!(drop_manager.needs_drop.contains(&custom_value));
}

#[test]
fn test_escape_analysis() {
    let mut arena = create_test_arena();
    let mut escape_analysis = EscapeAnalysis::new();

    // Create a function with local and escaping values
    let signature = HirFunctionSignature {
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

    let mut func = HirFunction::new(intern_str(&mut arena, "test_escape"), signature);

    // Create a local value (doesn't escape)
    let local_value = func.create_value(HirType::I32, HirValueKind::Constant(HirConstant::I32(42)));

    // Create a heap-allocated value (escapes via return)
    let heap_value = func.create_value(
        HirType::Ptr(Box::new(HirType::I32)),
        HirValueKind::Instruction,
    );

    // Return the heap value
    let block = func.blocks.get_mut(&func.entry_block).unwrap();
    block.set_terminator(HirTerminator::Return {
        values: vec![heap_value],
    });

    // Analyze escapes
    let result = escape_analysis.analyze(&func);
    assert!(result.is_ok());

    // Check results
    let local_info = escape_analysis.results.get(&local_value).unwrap();
    assert!(!local_info.escapes);
    assert!(!local_info.is_returned);

    let heap_info = escape_analysis.results.get(&heap_value).unwrap();
    assert!(heap_info.escapes);
    assert!(heap_info.is_returned);
}

#[test]
fn test_memory_context_reference_counting() {
    let mut ctx = MemoryContext::new(MemoryStrategy::ARC);

    let value1 = HirId::new();
    let value2 = HirId::new();

    // Track allocations
    ctx.track_allocation(
        value1,
        AllocationInfo {
            ty: HirType::Ptr(Box::new(HirType::I32)),
            size: Some(4),
            align: 4,
            is_stack: false,
            location: None,
        },
    );

    ctx.track_allocation(
        value2,
        AllocationInfo {
            ty: HirType::Array(Box::new(HirType::I32), 10),
            size: Some(40),
            align: 4,
            is_stack: true,
            location: None,
        },
    );

    // Test reference counting
    assert_eq!(ctx.ref_counts[&value1].count, 1);

    ctx.retain(value1);
    assert_eq!(ctx.ref_counts[&value1].count, 2);

    ctx.release(value1);
    assert_eq!(ctx.ref_counts[&value1].count, 1);

    ctx.release(value1);
    assert_eq!(ctx.ref_counts[&value1].count, 0);
    assert!(ctx.pending_drops.contains(&value1));
}

#[test]
fn test_stack_allocation_optimization() {
    let mut ctx = MemoryContext::new(MemoryStrategy::Ownership);

    let stack_value = HirId::new();
    let heap_value = HirId::new();

    // Add escape information
    ctx.escape_info.insert(
        stack_value,
        EscapeInfo {
            escapes: false,
            escape_targets: HashSet::new(),
            is_returned: false,
            stored_in_heap: false,
        },
    );

    ctx.escape_info.insert(
        heap_value,
        EscapeInfo {
            escapes: true,
            escape_targets: HashSet::new(),
            is_returned: true,
            stored_in_heap: false,
        },
    );

    // Check stack allocation eligibility
    assert!(ctx.can_stack_allocate(stack_value));
    assert!(!ctx.can_stack_allocate(heap_value));
}

#[test]
fn test_arc_needs_arc_type_checking() {
    let arc_manager = ARCManager::new();

    // Test primitive types (don't need ARC)
    assert!(!arc_manager.needs_arc(&HirType::I32));
    assert!(!arc_manager.needs_arc(&HirType::F64));
    assert!(!arc_manager.needs_arc(&HirType::Bool));

    // Test pointer and reference types (need ARC)
    assert!(arc_manager.needs_arc(&HirType::Ptr(Box::new(HirType::I32))));
    assert!(arc_manager.needs_arc(&HirType::Ref {
        lifetime: HirLifetime::anonymous(),
        pointee: Box::new(HirType::I32),
        mutable: false,
    }));

    // Test composite types
    let struct_with_ptr = HirType::Struct(HirStructType {
        name: None,
        fields: vec![HirType::I32, HirType::Ptr(Box::new(HirType::U8))],
        packed: false,
    });
    assert!(arc_manager.needs_arc(&struct_with_ptr));

    // Test closure types (need ARC)
    let closure_ty = HirType::Closure(Box::new(HirClosureType {
        function_type: HirFunctionType {
            params: vec![],
            returns: vec![],
            lifetime_params: vec![],
            is_variadic: false,
        },
        captures: vec![],
        call_mode: HirClosureCallMode::Fn,
    }));
    assert!(arc_manager.needs_arc(&closure_ty));
}

#[test]
fn test_drop_type_checking() {
    let drop_manager = DropManager::new();

    // Test primitive types (don't need drop)
    assert!(!drop_manager.needs_drop_type(&HirType::I32));
    assert!(!drop_manager.needs_drop_type(&HirType::F64));
    assert!(!drop_manager.needs_drop_type(&HirType::Bool));

    // Test pointer types (need drop - must free memory)
    assert!(drop_manager.needs_drop_type(&HirType::Ptr(Box::new(HirType::I32))));

    // Test array of droppable types
    let array_of_ptrs = HirType::Array(Box::new(HirType::Ptr(Box::new(HirType::I32))), 10);
    assert!(drop_manager.needs_drop_type(&array_of_ptrs));
}

#[test]
fn test_memory_strategy_selection() {
    // Test different memory strategies
    let manual_ctx = MemoryContext::new(MemoryStrategy::Manual);
    assert_eq!(manual_ctx.strategy, MemoryStrategy::Manual);

    let arc_ctx = MemoryContext::new(MemoryStrategy::ARC);
    assert_eq!(arc_ctx.strategy, MemoryStrategy::ARC);

    let ownership_ctx = MemoryContext::new(MemoryStrategy::Ownership);
    assert_eq!(ownership_ctx.strategy, MemoryStrategy::Ownership);
}

use std::collections::HashSet;
