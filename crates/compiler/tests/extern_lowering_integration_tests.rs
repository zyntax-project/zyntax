// Gap 11: Extern Function Lowering Integration Tests
//
// These tests verify that extern functions are properly lowered from TypedAST to HIR
// and that the backends handle them correctly.

use std::sync::{Arc, Mutex};
use zyntax_compiler::*;
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::*;

/// Create a simple extern function for testing
fn create_extern_function(
    arena: &mut AstArena,
    name: &str,
    cc: CallingConvention,
) -> TypedFunction {
    let span = Span::new(0, 10);
    let func_name = arena.intern_string(name);

    TypedFunction {
        name: func_name,
        params: vec![TypedParameter {
            name: arena.intern_string("x"),
            ty: Type::Primitive(PrimitiveType::I32),
            mutability: Mutability::Immutable,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span,
        }],
        type_params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        body: None, // Extern - no body
        visibility: Visibility::Public,
        is_async: false,
        is_external: true,
        calling_convention: cc,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    }
}

#[test]
fn test_extern_function_lowers_to_hir() {
    // Test that extern functions are properly lowered to HIR without errors
    let mut arena = AstArena::new();
    let extern_func = create_extern_function(&mut arena, "test_extern", CallingConvention::Cdecl);

    let span = Span::new(0, 10);
    let type_registry = TypeRegistry::new();
    let mut program = TypedProgram {
        declarations: vec![TypedNode::new(
            TypedDeclaration::Function(extern_func),
            Type::Primitive(PrimitiveType::I32), // Simplified
            span,
        )],
        span,
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    let type_registry = Arc::new(type_registry);
    let arena_arc = Arc::new(Mutex::new(arena));

    // Create lowering context
    let module_name = arena_arc.lock().unwrap().intern_string("test_module");
    let mut lowering_ctx = lowering::LoweringContext::new(
        module_name,
        type_registry,
        arena_arc,
        lowering::LoweringConfig::default(),
    );

    // Lower to HIR
    let result = lowering_ctx.lower_program(&mut program);

    // Should succeed
    assert!(
        result.is_ok(),
        "Failed to lower extern function: {:?}",
        result.err()
    );

    let hir_module = result.unwrap();

    // Verify we have 1 function
    assert_eq!(hir_module.functions.len(), 1);

    // Find the extern function
    let extern_func = hir_module.functions.values().next().unwrap();

    // Verify it's marked as external
    assert!(
        extern_func.is_external,
        "Extern function not marked as external in HIR"
    );

    // Verify it has the correct calling convention (C)
    assert_eq!(extern_func.calling_convention, hir::CallingConvention::C);

    // Verify it has no blocks (extern functions have no body)
    assert!(
        extern_func.blocks.is_empty(),
        "Extern function should have no blocks"
    );

    // Verify it has the correct signature
    assert_eq!(extern_func.signature.params.len(), 1);
    assert_eq!(extern_func.signature.returns.len(), 1);
}

#[test]
fn test_extern_function_compiles_with_cranelift() {
    // Test that extern functions can be compiled with Cranelift backend
    let mut arena = AstArena::new();
    let extern_func =
        create_extern_function(&mut arena, "cranelift_test", CallingConvention::Cdecl);

    let span = Span::new(0, 10);
    let type_registry = TypeRegistry::new();
    let mut program = TypedProgram {
        declarations: vec![TypedNode::new(
            TypedDeclaration::Function(extern_func),
            Type::Primitive(PrimitiveType::I32),
            span,
        )],
        span,
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    let type_registry = Arc::new(type_registry);
    let arena_arc = Arc::new(Mutex::new(arena));

    // Create lowering context and lower to HIR
    let module_name = arena_arc.lock().unwrap().intern_string("test_module");
    let mut lowering_ctx = lowering::LoweringContext::new(
        module_name,
        type_registry,
        arena_arc,
        lowering::LoweringConfig::default(),
    );
    let hir_module = lowering_ctx
        .lower_program(&mut program)
        .expect("Failed to lower to HIR");

    // Compile with Cranelift
    let mut cranelift_backend =
        cranelift_backend::CraneliftBackend::new().expect("Failed to create Cranelift backend");
    let result = cranelift_backend.compile_module(&hir_module);

    // Should succeed (extern functions just get declared, not compiled)
    assert!(
        result.is_ok(),
        "Cranelift compilation failed: {:?}",
        result.err()
    );
}

#[test]
fn test_calling_convention_conversion() {
    // Test that different calling conventions are properly converted
    let test_cases = vec![
        (
            "test_default",
            CallingConvention::Default,
            hir::CallingConvention::Fast,
        ),
        (
            "test_rust",
            CallingConvention::Rust,
            hir::CallingConvention::Fast,
        ),
        (
            "test_cdecl",
            CallingConvention::Cdecl,
            hir::CallingConvention::C,
        ),
        (
            "test_system",
            CallingConvention::System,
            hir::CallingConvention::System,
        ),
        (
            "test_stdcall",
            CallingConvention::Stdcall,
            hir::CallingConvention::C,
        ),
    ];

    for (name, typed_cc, expected_hir_cc) in test_cases {
        let mut arena = AstArena::new();
        let extern_func = create_extern_function(&mut arena, name, typed_cc);

        let span = Span::new(0, 10);
        let type_registry = TypeRegistry::new();
        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Function(extern_func),
                Type::Primitive(PrimitiveType::I32),
                span,
            )],
            span,
            source_files: vec![],
            type_registry: type_registry.clone(),
        };

        let type_registry = Arc::new(type_registry);
        let arena_arc = Arc::new(Mutex::new(arena));

        let module_name = arena_arc.lock().unwrap().intern_string("test_module");
        let mut lowering_ctx = lowering::LoweringContext::new(
            module_name,
            type_registry,
            arena_arc,
            lowering::LoweringConfig::default(),
        );
        let hir_module = lowering_ctx
            .lower_program(&mut program)
            .expect("Failed to lower");

        let hir_func = hir_module.functions.values().next().unwrap();
        assert_eq!(
            hir_func.calling_convention, expected_hir_cc,
            "Calling convention mismatch for {}",
            name
        );
    }
}

#[test]
fn test_non_extern_function_without_body_fails() {
    // Test that non-extern functions without a body fail during lowering
    let mut arena = AstArena::new();
    let span = Span::new(0, 10);
    let func_name = arena.intern_string("bad_func");

    let bad_func = TypedFunction {
        name: func_name,
        params: vec![],
        type_params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        body: None, // Regular function WITHOUT body
        visibility: Visibility::Public,
        is_async: false,
        is_external: false, // NOT extern
        calling_convention: CallingConvention::Default,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let type_registry = TypeRegistry::new();
    let mut program = TypedProgram {
        declarations: vec![TypedNode::new(
            TypedDeclaration::Function(bad_func),
            Type::Primitive(PrimitiveType::I32),
            span,
        )],
        span,
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    let type_registry = Arc::new(type_registry);
    let arena_arc = Arc::new(Mutex::new(arena));

    let module_name = arena_arc.lock().unwrap().intern_string("test_module");
    let mut lowering_ctx = lowering::LoweringContext::new(
        module_name,
        type_registry,
        arena_arc,
        lowering::LoweringConfig::default(),
    );

    let result = lowering_ctx.lower_program(&mut program);

    // Should fail
    assert!(
        result.is_err(),
        "Non-extern function without body should fail"
    );
}

#[test]
fn test_multiple_extern_functions() {
    // Test that we can lower multiple extern functions in one module
    let mut arena = AstArena::new();
    let span = Span::new(0, 10);

    let func1 = create_extern_function(&mut arena, "extern1", CallingConvention::Cdecl);
    let func2 = create_extern_function(&mut arena, "extern2", CallingConvention::System);
    let func3 = create_extern_function(&mut arena, "extern3", CallingConvention::Rust);

    let type_registry = TypeRegistry::new();
    let mut program = TypedProgram {
        declarations: vec![
            TypedNode::new(
                TypedDeclaration::Function(func1),
                Type::Primitive(PrimitiveType::I32),
                span,
            ),
            TypedNode::new(
                TypedDeclaration::Function(func2),
                Type::Primitive(PrimitiveType::I32),
                span,
            ),
            TypedNode::new(
                TypedDeclaration::Function(func3),
                Type::Primitive(PrimitiveType::I32),
                span,
            ),
        ],
        span,
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    let type_registry = Arc::new(type_registry);
    let arena_arc = Arc::new(Mutex::new(arena));

    let module_name = arena_arc.lock().unwrap().intern_string("test_module");
    let mut lowering_ctx = lowering::LoweringContext::new(
        module_name,
        type_registry,
        arena_arc,
        lowering::LoweringConfig::default(),
    );

    let result = lowering_ctx.lower_program(&mut program);
    assert!(result.is_ok());

    let hir_module = result.unwrap();
    assert_eq!(
        hir_module.functions.len(),
        3,
        "Should have 3 extern functions"
    );

    // All should be marked as external
    for func in hir_module.functions.values() {
        assert!(func.is_external);
        assert!(func.blocks.is_empty());
    }
}
