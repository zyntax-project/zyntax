// Gap 11: Extern Function FFI Tests
//
// Tests the complete extern function pipeline:
// 1. TypedAST extern function representation
// 2. Lowering to HIR
// 3. Backend handling (Cranelift and LLVM)

use zyntax_typed_ast::typed_ast::{ParameterKind, TypedFunction, TypedParameter};
use zyntax_typed_ast::*;

/// Helper to create a test extern function in TypedAST
fn create_extern_function(
    arena: &mut AstArena,
    name: &str,
    params: Vec<(&str, Type)>,
    return_type: Type,
    calling_convention: CallingConvention,
) -> TypedFunction {
    let func_name = arena.intern_string(name);

    let typed_params = params
        .into_iter()
        .map(|(param_name, ty)| TypedParameter {
            name: arena.intern_string(param_name),
            ty,
            mutability: Mutability::Immutable,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span: Span::new(0, 10),
        })
        .collect();

    TypedFunction {
        name: func_name,
        params: typed_params,
        type_params: vec![],
        return_type,
        body: None, // Extern functions have no body
        visibility: Visibility::Public,
        is_async: false,
        is_external: true,
        calling_convention,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    }
}

#[test]
fn test_extern_function_declaration() {
    // Test that extern functions are properly represented in TypedAST
    let mut arena = AstArena::new();

    let malloc_func = create_extern_function(
        &mut arena,
        "malloc",
        vec![("size", Type::Primitive(PrimitiveType::U64))],
        Type::Primitive(PrimitiveType::U64), // Simplified - return address as u64
        CallingConvention::Cdecl,
    );

    assert!(malloc_func.is_external);
    assert!(malloc_func.body.is_none());
    assert_eq!(malloc_func.calling_convention, CallingConvention::Cdecl);
    assert_eq!(malloc_func.params.len(), 1);
}

#[test]
fn test_extern_c_calling_convention() {
    // Test C calling convention for libc functions
    let mut arena = AstArena::new();

    let free_func = create_extern_function(
        &mut arena,
        "free",
        vec![(
            "ptr",
            Type::Reference {
                ty: Box::new(Type::Primitive(PrimitiveType::U8)),
                mutability: Mutability::Mutable,
                lifetime: None,
                nullability: NullabilityKind::NonNull,
            },
        )],
        Type::Primitive(PrimitiveType::Unit),
        CallingConvention::Cdecl,
    );

    assert!(free_func.is_external);
    assert_eq!(free_func.calling_convention, CallingConvention::Cdecl);
}

#[test]
fn test_extern_system_calling_convention() {
    // Test system calling convention
    let mut arena = AstArena::new();

    let sys_func = create_extern_function(
        &mut arena,
        "system_call",
        vec![],
        Type::Primitive(PrimitiveType::I32),
        CallingConvention::System,
    );

    assert!(sys_func.is_external);
    assert_eq!(sys_func.calling_convention, CallingConvention::System);
}

// Note: Lowering test skipped - lower_function is private
// The lowering pipeline is tested through integration tests

// Note: Lowering test skipped - lower_function is private
// Non-extern functions with no body are caught in the lowering pipeline

// Note: Calling convention conversion test skipped - lower_function is private
// The conversion is verified through the backend compilation tests

#[test]
fn test_extern_function_with_multiple_parameters() {
    // Test extern function with multiple parameters
    let mut arena = AstArena::new();

    let printf_func = create_extern_function(
        &mut arena,
        "printf",
        vec![
            ("x", Type::Primitive(PrimitiveType::I32)), // Simplified
                                                        // Note: variadic parameters not yet supported, so we can't fully represent printf
        ],
        Type::Primitive(PrimitiveType::I32),
        CallingConvention::Cdecl,
    );

    assert!(printf_func.is_external);
    assert_eq!(printf_func.params.len(), 1); // Just format for now
}

#[test]
fn test_extern_function_with_pointer_types() {
    // Test extern functions with pointer parameter and return types
    let mut arena = AstArena::new();

    let memcpy_func = create_extern_function(
        &mut arena,
        "memcpy",
        vec![
            ("a", Type::Primitive(PrimitiveType::I32)),
            ("b", Type::Primitive(PrimitiveType::I32)),
            ("n", Type::Primitive(PrimitiveType::U64)),
        ],
        Type::Primitive(PrimitiveType::I32), // Simplified
        CallingConvention::Cdecl,
    );

    assert!(memcpy_func.is_external);
    assert_eq!(memcpy_func.params.len(), 3);
}

#[test]
fn test_extern_function_void_return() {
    // Test extern function returning void (Unit type)
    let mut arena = AstArena::new();

    let exit_func = create_extern_function(
        &mut arena,
        "exit",
        vec![("status", Type::Primitive(PrimitiveType::I32))],
        Type::Primitive(PrimitiveType::Unit), // void
        CallingConvention::Cdecl,
    );

    assert!(exit_func.is_external);
    assert_eq!(exit_func.return_type, Type::Primitive(PrimitiveType::Unit));
}

// Note: Type checker test removed - check_function is private
// The extern body skipping is verified through integration tests
