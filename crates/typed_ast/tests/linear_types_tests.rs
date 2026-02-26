//! Functional tests for the linear type system algorithms
//!
//! This module tests resource management, affine types, and borrowing with actual checking algorithms.

use string_interner::Symbol;
use zyntax_typed_ast::arena::InternedString;
use zyntax_typed_ast::linear_types::*;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type};
use zyntax_typed_ast::typed_ast::*;

fn create_test_program_with_variable(var_name: &str, var_type: Type) -> TypedProgram {
    let name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(1).unwrap());
    let span = Span::new(0, 10);

    TypedProgram {
        declarations: vec![TypedNode {
            node: TypedDeclaration::Variable(TypedVariable {
                name,
                ty: var_type.clone(),
                mutability: zyntax_typed_ast::type_registry::Mutability::Immutable,
                visibility: zyntax_typed_ast::type_registry::Visibility::Private,
                initializer: None,
            }),
            ty: Type::Primitive(PrimitiveType::Unit),
            span,
        }],
        span,
        ..Default::default()
    }
}

fn create_test_function() -> TypedProgram {
    let span = Span::new(0, 10);
    let func_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(2).unwrap());

    TypedProgram {
        declarations: vec![TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: func_name,
                params: vec![],
                type_params: vec![],
                return_type: Type::Primitive(PrimitiveType::Unit),
                body: Some(TypedBlock {
                    statements: vec![],
                    span,
                }),
                visibility: zyntax_typed_ast::type_registry::Visibility::Private,
                is_async: false,
                is_external: false,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: zyntax_typed_ast::type_registry::AsyncKind::Sync,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
            },
            span,
        }],
        span,
        ..Default::default()
    }
}

#[test]
fn test_linear_type_checker_creation() {
    let mut checker = LinearTypeChecker::new();

    // Test with a simple program containing no linear types
    let program = create_test_program_with_variable("x", Type::Primitive(PrimitiveType::I32));

    // Should succeed for non-linear types
    let result = checker.check_program(&program);
    assert!(
        result.is_ok(),
        "Linear type checker should handle non-linear types"
    );
}

#[test]
fn test_linear_type_checker_with_function() {
    let mut checker = LinearTypeChecker::new();

    // Test with a function program
    let program = create_test_function();

    // Should succeed for simple function
    let result = checker.check_program(&program);
    assert!(
        result.is_ok(),
        "Linear type checker should handle simple functions"
    );
}

#[test]
fn test_linear_type_checker_empty_program() {
    let mut checker = LinearTypeChecker::new();

    // Test with empty program
    let program = TypedProgram {
        declarations: vec![],
        span: Span::new(0, 0),
        ..Default::default()
    };

    // Should succeed for empty program
    let result = checker.check_program(&program);
    assert!(
        result.is_ok(),
        "Linear type checker should handle empty programs"
    );
}

#[test]
fn test_linearity_kind_variants() {
    // Test that different linearity kinds exist and are distinguishable
    assert!(LinearityKind::Linear != LinearityKind::Affine);
    assert!(LinearityKind::Affine != LinearityKind::Relevant);
    assert!(LinearityKind::Relevant != LinearityKind::Unrestricted);
    assert!(LinearityKind::Unrestricted != LinearityKind::Unique);
    assert!(LinearityKind::Unique != LinearityKind::Shared);
}

#[test]
fn test_resource_id_generation() {
    // Test that resource IDs are unique
    let id1 = ResourceId::next();
    let id2 = ResourceId::next();
    let id3 = ResourceId::next();

    assert!(id1 != id2);
    assert!(id2 != id3);
    assert!(id1 != id3);
}

#[test]
fn test_borrow_id_generation() {
    // Test that borrow IDs are unique
    let id1 = BorrowId::next();
    let id2 = BorrowId::next();
    let id3 = BorrowId::next();

    assert!(id1 != id2);
    assert!(id2 != id3);
    assert!(id1 != id3);
}

#[test]
fn test_borrow_kind_variants() {
    // Test that different borrow kinds exist
    assert!(BorrowKind::Shared != BorrowKind::Mutable);
    assert!(BorrowKind::Mutable != BorrowKind::Move);
    assert!(BorrowKind::Move != BorrowKind::Weak);
}

#[test]
fn test_resource_kind_variants() {
    // Test that different resource kinds exist
    assert!(ResourceKind::FileHandle != ResourceKind::NetworkConnection);
    assert!(ResourceKind::NetworkConnection != ResourceKind::Memory);
    assert!(ResourceKind::Memory != ResourceKind::Database);
    assert!(ResourceKind::Database != ResourceKind::Lock);
    assert!(ResourceKind::Lock != ResourceKind::Custom);
}

#[test]
fn test_scope_kind_variants() {
    // Test that different scope kinds exist
    assert!(ScopeKind::Function != ScopeKind::Block);
    assert!(ScopeKind::Block != ScopeKind::Loop);
    assert!(ScopeKind::Loop != ScopeKind::Conditional);
    assert!(ScopeKind::Conditional != ScopeKind::MatchArm);
    assert!(ScopeKind::MatchArm != ScopeKind::Async);
}

#[test]
fn test_linear_type_error_variants() {
    let span = Span::new(0, 10);
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(1).unwrap());

    // Test that different error types can be constructed
    let error1 = LinearTypeError::UsedMoreThanOnce {
        var: var_name,
        first_use: span,
        second_use: span,
        linearity: LinearityKind::Linear,
    };

    let error2 = LinearTypeError::NotUsed {
        var: var_name,
        declaration: span,
        linearity: LinearityKind::Linear,
    };

    let error3 = LinearTypeError::UsedAfterMove {
        var: var_name,
        move_span: span,
        use_span: span,
    };

    // Verify they're different error types
    assert!(std::mem::discriminant(&error1) != std::mem::discriminant(&error2));
    assert!(std::mem::discriminant(&error2) != std::mem::discriminant(&error3));
    assert!(std::mem::discriminant(&error1) != std::mem::discriminant(&error3));
}

#[test]
fn test_program_with_multiple_declarations() {
    let mut checker = LinearTypeChecker::new();
    let span = Span::new(0, 10);

    let program = TypedProgram {
        declarations: vec![
            TypedNode {
                node: TypedDeclaration::Variable(TypedVariable {
                    name: InternedString::from_symbol(
                        string_interner::DefaultSymbol::try_from_usize(1).unwrap(),
                    ),
                    ty: Type::Primitive(PrimitiveType::I32),
                    mutability: zyntax_typed_ast::type_registry::Mutability::Immutable,
                    visibility: zyntax_typed_ast::type_registry::Visibility::Private,
                    initializer: None,
                }),
                ty: Type::Primitive(PrimitiveType::Unit),
                span,
            },
            TypedNode {
                node: TypedDeclaration::Variable(TypedVariable {
                    name: InternedString::from_symbol(
                        string_interner::DefaultSymbol::try_from_usize(2).unwrap(),
                    ),
                    ty: Type::Primitive(PrimitiveType::String),
                    mutability: zyntax_typed_ast::type_registry::Mutability::Mutable,
                    visibility: zyntax_typed_ast::type_registry::Visibility::Public,
                    initializer: None,
                }),
                ty: Type::Primitive(PrimitiveType::Unit),
                span,
            },
        ],
        span,
        ..Default::default()
    };

    let result = checker.check_program(&program);
    assert!(
        result.is_ok(),
        "Linear type checker should handle multiple variable declarations"
    );
}

#[test]
fn test_linear_type_info_creation() {
    let type_id = zyntax_typed_ast::type_registry::TypeId::next();

    let file_handle_info = LinearTypeInfo::file_handle(type_id);
    assert_eq!(file_handle_info.linearity, LinearityKind::Linear);
    assert_eq!(
        file_handle_info.resource_kind,
        Some(ResourceKind::FileHandle)
    );

    let unique_ptr_info = LinearTypeInfo::unique_pointer(type_id);
    assert_eq!(unique_ptr_info.linearity, LinearityKind::Affine);
    assert_eq!(unique_ptr_info.resource_kind, Some(ResourceKind::Memory));

    let shared_ref_info = LinearTypeInfo::shared_reference(type_id);
    assert_eq!(shared_ref_info.linearity, LinearityKind::Shared);
    assert_eq!(shared_ref_info.resource_kind, None);
}

#[test]
fn test_lifetime_creation() {
    let lifetime1 = Lifetime::new(1);
    let lifetime2 = Lifetime::new(2);
    let static_lifetime = Lifetime::static_lifetime();

    assert!(lifetime1 != lifetime2);
    assert!(lifetime1 != static_lifetime);
    assert!(lifetime2 != static_lifetime);
}

#[test]
fn test_complex_program_structure() {
    let mut checker = LinearTypeChecker::new();
    let span = Span::new(0, 10);
    let func_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(1).unwrap());
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(2).unwrap());

    // Create a program with function containing variable declarations
    let program = TypedProgram {
        declarations: vec![TypedNode {
            node: TypedDeclaration::Function(TypedFunction {
                name: func_name,
                params: vec![],
                type_params: vec![],
                return_type: Type::Primitive(PrimitiveType::Unit),
                body: Some(TypedBlock {
                    statements: vec![TypedNode {
                        node: TypedStatement::Let(TypedLet {
                            name: var_name,
                            ty: Type::Primitive(PrimitiveType::I32),
                            mutability: zyntax_typed_ast::type_registry::Mutability::Immutable,
                            initializer: None,
                            span,
                        }),
                        ty: Type::Primitive(PrimitiveType::Unit),
                        span,
                    }],
                    span,
                }),
                visibility: zyntax_typed_ast::type_registry::Visibility::Private,
                is_async: false,
                is_external: false,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                link_name: None,
                ..Default::default()
            }),
            ty: Type::Function {
                params: vec![],
                return_type: Box::new(Type::Primitive(PrimitiveType::Unit)),
                is_varargs: false,
                has_named_params: false,
                has_default_params: false,
                async_kind: zyntax_typed_ast::type_registry::AsyncKind::Sync,
                calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
            },
            span,
        }],
        span,
        ..Default::default()
    };

    let result = checker.check_program(&program);
    assert!(
        result.is_ok(),
        "Linear type checker should handle complex program structures"
    );
}
