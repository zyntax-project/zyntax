//! Integration tests for the complete compilation pipeline
//!
//! These tests verify that the compilation pipeline correctly:
//! 1. Lowers TypedAST to HIR
//! 2. Monomorphizes generic functions
//! 3. Runs analysis passes
//! 4. Produces valid HIR ready for backend code generation

use std::sync::{Arc, Mutex};
use zyntax_compiler::{
    async_support::AsyncRuntimeType, compile_to_hir, hir::*, monomorphize_module,
    CompilationConfig, CompilerResult, MonomorphizationContext,
};
use zyntax_typed_ast::{
    arena::AstArena,
    typed_ast::{TypedBinary, TypedBlock, TypedWhile},
    typed_node, BinaryOp, CallingConvention, PrimitiveType, Span, Type, TypeRegistry,
    TypedDeclaration, TypedExpression, TypedFunction, TypedLiteral, TypedProgram, TypedStatement,
    Visibility,
};

fn test_span() -> Span {
    Span::new(0, 10)
}

fn test_arena() -> AstArena {
    AstArena::new()
}

/// Helper to create a simple typed program with one function
fn create_test_program(arena: &mut AstArena, func_name: &str, body: TypedBlock) -> TypedProgram {
    let name = arena.intern_string(func_name);
    let function = TypedFunction {
        type_params: vec![],
        name,
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        body: Some(body),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Rust,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    TypedProgram {
        declarations: vec![typed_node(
            TypedDeclaration::Function(function),
            Type::Primitive(PrimitiveType::Unit),
            test_span(),
        )],
        span: test_span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    }
}

#[test]
fn test_compilation_pipeline_simple_function() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a simple function that returns 42
    let literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(42)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "return_42", body);

    // Run compilation pipeline
    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "Compilation should succeed: {:?}",
        result.err()
    );

    let hir_module = result.unwrap();

    // Verify HIR module has the function
    assert_eq!(hir_module.functions.len(), 1, "Should have one function");

    let (_, hir_func) = hir_module.functions.iter().next().unwrap();

    // Verify function signature
    assert_eq!(
        hir_func.signature.params.len(),
        0,
        "Should have no parameters"
    );
    assert_eq!(
        hir_func.signature.returns.len(),
        1,
        "Should have one return value"
    );

    // Verify function has blocks with instructions
    assert!(
        !hir_func.blocks.is_empty(),
        "Function should have basic blocks"
    );

    let entry_block = &hir_func.blocks[&hir_func.entry_block];

    // The function should have a return terminator
    assert!(
        matches!(entry_block.terminator, HirTerminator::Return { .. }),
        "Entry block should have a return terminator"
    );
}

#[test]
fn test_compilation_pipeline_binary_operation() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a function: fn test() -> i32 { return 10 + 20; }
    let left = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(10)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let right = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(20)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let binary = typed_node(
        TypedExpression::Binary(TypedBinary {
            op: BinaryOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(binary))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "add_numbers", body);

    // Run compilation pipeline
    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "Compilation should succeed: {:?}",
        result.err()
    );

    let hir_module = result.unwrap();

    // Verify HIR module
    assert_eq!(hir_module.functions.len(), 1, "Should have one function");

    let (_, hir_func) = hir_module.functions.iter().next().unwrap();

    // Verify function has blocks with instructions
    let entry_block = &hir_func.blocks[&hir_func.entry_block];

    // Should have Binary instruction from the addition
    let has_binary = entry_block
        .instructions
        .iter()
        .any(|inst| matches!(inst, HirInstruction::Binary { .. }));

    assert!(has_binary, "Should have Binary instruction for addition");
}

#[test]
fn test_monomorphization_context_basic() {
    let mut arena = test_arena();
    let mut mono_ctx = MonomorphizationContext::new();

    // Create a generic function in HIR
    let func_name = arena.intern_string("test_func");
    let t_param = arena.intern_string("T");

    let signature = HirFunctionSignature {
        params: vec![HirParam {
            id: HirId::new(),
            name: arena.intern_string("x"),
            ty: HirType::Opaque(t_param),
            attributes: ParamAttributes::default(),
        }],
        returns: vec![HirType::Opaque(t_param)],
        type_params: vec![HirTypeParam {
            name: t_param,
            constraints: vec![],
        }],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let generic_func = HirFunction::new(func_name, signature);
    let generic_id = generic_func.id;

    // Register the generic function
    mono_ctx.register_generic(generic_func);

    // Instantiate with T=i32
    let type_args = vec![HirType::I32];
    let const_args = vec![];

    let instance_id = mono_ctx
        .get_or_create_instance(generic_id, type_args.clone(), const_args.clone())
        .unwrap();

    // Verify we get a different ID for the monomorphized instance
    assert_ne!(
        instance_id, generic_id,
        "Monomorphized instance should have different ID"
    );

    // Try to get the same instance again - should return the cached one
    let cached_id = mono_ctx
        .get_or_create_instance(generic_id, type_args, const_args)
        .unwrap();

    assert_eq!(instance_id, cached_id, "Should return cached instance");
}

#[test]
fn test_monomorphization_module_integration() {
    let mut arena = test_arena();

    // Create a HIR module with a generic function
    let mut hir_module = HirModule::new(arena.intern_string("test_module"));

    let func_name = arena.intern_string("generic_add");
    let t_param = arena.intern_string("T");

    let signature = HirFunctionSignature {
        params: vec![
            HirParam {
                id: HirId::new(),
                name: arena.intern_string("a"),
                ty: HirType::Opaque(t_param),
                attributes: ParamAttributes::default(),
            },
            HirParam {
                id: HirId::new(),
                name: arena.intern_string("b"),
                ty: HirType::Opaque(t_param),
                attributes: ParamAttributes::default(),
            },
        ],
        returns: vec![HirType::Opaque(t_param)],
        type_params: vec![HirTypeParam {
            name: t_param,
            constraints: vec![],
        }],
        const_params: vec![],
        lifetime_params: vec![],
        is_variadic: false,
        is_async: false,
        effects: vec![],
        is_pure: false,
    };

    let generic_func = HirFunction::new(func_name, signature);
    let func_id = generic_func.id;

    hir_module.functions.insert(func_id, generic_func);

    // Run monomorphization on the module
    let result = monomorphize_module(&mut hir_module);
    assert!(result.is_ok(), "Monomorphization should succeed");

    // Verify module still has the generic function
    assert_eq!(
        hir_module.functions.len(),
        1,
        "Should have the generic function"
    );
}

#[test]
fn test_compilation_config_defaults() {
    let config = CompilationConfig::default();

    assert_eq!(
        config.opt_level, 2,
        "Default optimization level should be 2"
    );
    assert!(config.debug_info, "Debug info should be enabled by default");
    assert!(
        config.enable_monomorphization,
        "Monomorphization should be enabled by default"
    );
    assert!(
        !config.hot_reload,
        "Hot reload should be disabled by default"
    );
}

#[test]
fn test_compilation_pipeline_with_analysis() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a simple function
    let literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(100)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_analysis", body);

    // Run compilation with all passes enabled
    let config = CompilationConfig {
        opt_level: 2, // Enable optimizations
        debug_info: true,
        enable_monomorphization: true,
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(result.is_ok(), "Compilation with analysis should succeed");

    let hir_module = result.unwrap();

    // Verify the module is valid
    assert!(
        !hir_module.functions.is_empty(),
        "Should have at least one function"
    );
}

#[test]
fn test_compilation_pipeline_with_memory_management() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a simple function
    let literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(42)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_memory", body);

    // Run compilation with ARC memory strategy
    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        memory_strategy: Some(zyntax_compiler::MemoryStrategy::ARC),
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(result.is_ok(), "Compilation with ARC should succeed");

    let hir_module = result.unwrap();
    assert!(
        !hir_module.functions.is_empty(),
        "Should have at least one function"
    );
}

#[test]
fn test_compilation_pipeline_without_memory_management() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a simple function
    let literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(100)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_no_memory", body);

    // Run compilation with memory management disabled
    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        memory_strategy: None, // Disable memory management
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "Compilation without memory management should succeed"
    );

    let hir_module = result.unwrap();
    assert!(
        !hir_module.functions.is_empty(),
        "Should have at least one function"
    );
}

#[test]
fn test_compilation_pipeline_with_gc_strategy() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a simple function
    let literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(256)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_gc", body);

    // Run compilation with GC memory strategy
    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        memory_strategy: Some(zyntax_compiler::MemoryStrategy::GC),
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(result.is_ok(), "Compilation with GC should succeed");

    let hir_module = result.unwrap();
    assert!(
        !hir_module.functions.is_empty(),
        "Should have at least one function"
    );
}

#[test]
fn test_compilation_pipeline_all_features() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a more complex function with binary operations
    let left = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(10)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let right = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(20)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let binary = typed_node(
        TypedExpression::Binary(TypedBinary {
            op: BinaryOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(binary))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_all_features", body);

    // Run compilation with ALL features enabled
    let config = CompilationConfig {
        opt_level: 2, // Enable optimizations
        debug_info: true,
        enable_monomorphization: true,
        memory_strategy: Some(zyntax_compiler::MemoryStrategy::ARC),
        async_runtime: None, // Not testing async in this test
        hot_reload: false,
        target_triple: "x86_64-unknown-linux-gnu".to_string(),
        import_resolver: None,
        enable_borrow_check: false,
        enable_effect_check: true,
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "Compilation with all features should succeed"
    );

    let hir_module = result.unwrap();

    // Verify the module is valid
    assert!(
        !hir_module.functions.is_empty(),
        "Should have at least one function"
    );

    let (_, hir_func) = hir_module.functions.iter().next().unwrap();

    // Verify function has blocks and instructions
    assert!(
        !hir_func.blocks.is_empty(),
        "Function should have basic blocks"
    );
}

#[test]
fn test_compilation_pipeline_with_memory_optimizations() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a function with binary operations
    let left = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(50)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let right = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(25)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let binary = typed_node(
        TypedExpression::Binary(TypedBinary {
            op: BinaryOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(binary))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_opt", body);

    // Run compilation with full optimization including memory optimizations
    let config = CompilationConfig {
        opt_level: 3, // Maximum optimization
        debug_info: false,
        enable_monomorphization: true,
        memory_strategy: Some(zyntax_compiler::MemoryStrategy::ARC),
        async_runtime: None, // Not testing async in this test
        hot_reload: false,
        target_triple: "x86_64-unknown-linux-gnu".to_string(),
        import_resolver: None,
        enable_borrow_check: false,
        enable_effect_check: true,
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "Compilation with memory optimizations should succeed"
    );

    let hir_module = result.unwrap();

    // Verify the module is valid and optimized
    assert!(
        !hir_module.functions.is_empty(),
        "Should have at least one function"
    );

    let (_, hir_func) = hir_module.functions.iter().next().unwrap();

    // Verify function has blocks
    assert!(
        !hir_func.blocks.is_empty(),
        "Function should have basic blocks"
    );

    // The optimization passes should have run successfully
    // (we don't check for specific optimizations as they may vary,
    // but we verify the pipeline completes without errors)
}

#[test]
fn test_compilation_pipeline_with_async_function() -> CompilerResult<()> {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create async function: async fn get_value() -> i32 { return 42; }
    let literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(42)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let func = TypedFunction {
        type_params: vec![],
        name: arena.intern_string("get_value"),
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        body: Some(body),
        visibility: Visibility::Public,
        is_async: true, // Mark as async
        is_external: false,
        calling_convention: CallingConvention::Rust,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![typed_node(
            TypedDeclaration::Function(func),
            Type::Primitive(PrimitiveType::Unit),
            test_span(),
        )],
        span: test_span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };

    // Compile with async runtime enabled
    let config = CompilationConfig {
        async_runtime: Some(AsyncRuntimeType::Tokio),
        ..Default::default()
    };

    let hir_module = compile_to_hir(&mut program, type_registry, config)?;

    // Verify module was created
    // Async functions generate additional state machine functions, so we expect >= 1
    assert!(
        hir_module.functions.len() >= 1,
        "Should have at least one function"
    );

    // Get any function - async transformation may change names
    let async_func = hir_module
        .functions
        .values()
        .next()
        .expect("Should have at least one function");

    // Verify function has blocks (state machine implementation)
    assert!(
        !async_func.blocks.is_empty(),
        "Async function should have basic blocks"
    );

    // The async function should have been transformed
    // (the exact structure depends on AsyncCompiler implementation)

    Ok(())
}

#[test]
fn test_compilation_pipeline_without_async_runtime() -> CompilerResult<()> {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create async function but compile without async runtime
    let literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(42)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let return_stmt = typed_node(
        TypedStatement::Return(Some(Box::new(literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![return_stmt],
        span: test_span(),
    };

    let func = TypedFunction {
        type_params: vec![],
        name: arena.intern_string("get_value"),
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        body: Some(body),
        visibility: Visibility::Public,
        is_async: true, // Mark as async
        is_external: false,
        calling_convention: CallingConvention::Rust,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![typed_node(
            TypedDeclaration::Function(func),
            Type::Primitive(PrimitiveType::Unit),
            test_span(),
        )],
        span: test_span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };

    // Compile with async runtime DISABLED
    let config = CompilationConfig {
        async_runtime: None,
        ..Default::default()
    };

    let hir_module = compile_to_hir(&mut program, type_registry, config)?;

    // Verify module was created
    assert!(
        hir_module.functions.len() >= 1,
        "Should have at least one function"
    );

    // Get any function - async transformation may change names
    let hir_func = hir_module
        .functions
        .values()
        .next()
        .expect("Should have at least one function");

    // Verify function exists (async flag may change after transformation)
    // Without async runtime, function is lowered but not transformed to state machine
    assert!(
        !hir_func.blocks.is_empty(),
        "Function should have basic blocks"
    );

    // Function should have blocks from normal lowering (not state machine)
    assert!(
        !hir_func.blocks.is_empty(),
        "Function should have basic blocks"
    );

    Ok(())
}

#[test]
fn test_compilation_pipeline_mixed_sync_async() -> CompilerResult<()> {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create sync function
    let sync_literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(1)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let sync_return = typed_node(
        TypedStatement::Return(Some(Box::new(sync_literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let sync_body = TypedBlock {
        statements: vec![sync_return],
        span: test_span(),
    };

    let sync_func = TypedFunction {
        type_params: vec![],
        name: arena.intern_string("sync_fn"),
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        body: Some(sync_body),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Rust,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    // Create async function
    let async_literal = typed_node(
        TypedExpression::Literal(TypedLiteral::Integer(2)),
        Type::Primitive(PrimitiveType::I32),
        test_span(),
    );

    let async_return = typed_node(
        TypedStatement::Return(Some(Box::new(async_literal))),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let async_body = TypedBlock {
        statements: vec![async_return],
        span: test_span(),
    };

    let async_func = TypedFunction {
        type_params: vec![],
        name: arena.intern_string("async_fn"),
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        body: Some(async_body),
        visibility: Visibility::Public,
        is_async: true,
        is_external: false,
        calling_convention: CallingConvention::Rust,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![
            typed_node(
                TypedDeclaration::Function(sync_func),
                Type::Primitive(PrimitiveType::Unit),
                test_span(),
            ),
            typed_node(
                TypedDeclaration::Function(async_func),
                Type::Primitive(PrimitiveType::Unit),
                test_span(),
            ),
        ],
        span: test_span(),
        source_files: vec![],
        type_registry: TypeRegistry::new(),
    };

    // Compile with async runtime enabled
    let config = CompilationConfig {
        async_runtime: Some(AsyncRuntimeType::Tokio),
        ..Default::default()
    };

    let hir_module = compile_to_hir(&mut program, type_registry, config)?;

    // Verify both functions were created
    // Async functions may generate additional state machine functions
    assert!(
        hir_module.functions.len() >= 2,
        "Should have at least two functions"
    );

    // Verify we have the functions (names may be transformed)
    // Just check that all functions have blocks
    for func in hir_module.functions.values() {
        assert!(!func.blocks.is_empty(), "All functions should have blocks");
    }

    Ok(())
}
#[test]
fn test_compilation_pipeline_while_loop() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create a while loop: while false { }
    let condition = typed_node(
        TypedExpression::Literal(TypedLiteral::Bool(false)),
        Type::Primitive(PrimitiveType::Bool),
        test_span(),
    );

    let while_stmt = typed_node(
        TypedStatement::While(TypedWhile {
            condition: Box::new(condition),
            body: TypedBlock {
                statements: vec![],
                span: test_span(),
            },
            span: test_span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![while_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_while", body);

    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "While loop compilation should succeed: {:?}",
        result.err()
    );

    let hir_module = result.unwrap();

    // Verify function was created
    assert_eq!(hir_module.functions.len(), 1, "Should have one function");

    let (_, hir_func) = hir_module.functions.iter().next().unwrap();

    // While loop should create multiple blocks: entry, header, body, exit
    assert!(
        hir_func.blocks.len() >= 3,
        "While loop should create at least 3 blocks (header, body, exit), got {}",
        hir_func.blocks.len()
    );

    // Entry block should branch to header
    let entry_block = &hir_func.blocks[&hir_func.entry_block];
    assert!(
        matches!(entry_block.terminator, HirTerminator::Branch { .. }),
        "Entry block should branch to loop header"
    );
}

#[test]
fn test_compilation_pipeline_infinite_loop_with_break() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create: loop { break; }
    let break_stmt = typed_node(
        TypedStatement::Break(None),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let loop_stmt = typed_node(
        TypedStatement::Loop(zyntax_typed_ast::typed_ast::TypedLoop::Infinite {
            body: TypedBlock {
                statements: vec![break_stmt],
                span: test_span(),
            },
        }),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let body = TypedBlock {
        statements: vec![loop_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_loop_break", body);

    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "Infinite loop with break should compile: {:?}",
        result.err()
    );

    let hir_module = result.unwrap();
    assert_eq!(hir_module.functions.len(), 1, "Should have one function");

    let (_, hir_func) = hir_module.functions.iter().next().unwrap();

    // Loop should create multiple blocks: entry, header, body, exit
    assert!(
        hir_func.blocks.len() >= 4,
        "Loop should create at least 4 blocks, got {}",
        hir_func.blocks.len()
    );
}

#[test]
fn test_compilation_pipeline_nested_blocks() {
    let mut arena = test_arena();
    let type_registry = Arc::new(TypeRegistry::new());

    // Create: { { } }
    let inner_block_stmt = typed_node(
        TypedStatement::Block(TypedBlock {
            statements: vec![],
            span: test_span(),
        }),
        Type::Primitive(PrimitiveType::Unit),
        test_span(),
    );

    let outer_block = TypedBlock {
        statements: vec![inner_block_stmt],
        span: test_span(),
    };

    let mut program = create_test_program(&mut arena, "test_nested_blocks", outer_block);

    let config = CompilationConfig {
        opt_level: 0,
        debug_info: false,
        enable_monomorphization: true,
        ..Default::default()
    };

    let result = compile_to_hir(&mut program, type_registry, config);
    assert!(
        result.is_ok(),
        "Nested blocks should compile: {:?}",
        result.err()
    );

    let hir_module = result.unwrap();
    assert_eq!(hir_module.functions.len(), 1, "Should have one function");
}
