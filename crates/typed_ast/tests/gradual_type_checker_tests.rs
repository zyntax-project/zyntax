//! Tests for the gradual type checking system
//!
//! This module tests dynamic/static hybrid typing, runtime type checks, and type evidence.

// Note: These tests use internal types from gradual_type_checker
use string_interner::Symbol;
use zyntax_typed_ast::arena::InternedString;
use zyntax_typed_ast::gradual_type_checker::*;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{NullabilityKind, PrimitiveType, Type};

#[test]
fn test_gradual_compatibility_any_type() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);

    // Test that Any is compatible with everything without runtime checks
    let static_type = Type::Primitive(PrimitiveType::String);
    let any_type = Type::Any;

    let checks = checker
        .check_gradual_compatibility(&static_type, &any_type, BoundaryKind::DynamicToStatic, span)
        .unwrap();

    assert!(
        checks.is_empty(),
        "Any type should not require runtime checks"
    );

    // Test reverse: Any accepts everything
    let checks = checker
        .check_gradual_compatibility(&any_type, &static_type, BoundaryKind::StaticToDynamic, span)
        .unwrap();

    assert!(
        checks.is_empty(),
        "Any type should accept any value without checks"
    );
}

#[test]
fn test_dynamic_to_static_conversion() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);

    // Test that Dynamic to static type requires runtime checks
    let static_type = Type::Primitive(PrimitiveType::I32);
    let dynamic_type = Type::Dynamic;

    let checks = checker
        .check_gradual_compatibility(
            &static_type,
            &dynamic_type,
            BoundaryKind::DynamicToStatic,
            span,
        )
        .unwrap();

    // Should generate at least one runtime check
    assert!(
        !checks.is_empty(),
        "Dynamic to static should require runtime checks"
    );

    // Check that the generated check is a type assertion
    let first_check = &checks[0];
    match first_check.check_kind {
        RuntimeCheckKind::TypeAssertion => {}
        _ => panic!("Expected TypeAssertion for dynamic to static conversion"),
    }

    // Verify the check expects the correct type
    assert_eq!(first_check.expected_type, static_type);
}

#[test]
fn test_nullable_type_checks() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);

    // Test that Nullable types generate null checks first
    let nullable_string = Type::Nullable(Box::new(Type::Primitive(PrimitiveType::String)));
    let dynamic_type = Type::Dynamic;

    let checks = checker
        .check_gradual_compatibility(
            &nullable_string,
            &dynamic_type,
            BoundaryKind::DynamicToStatic,
            span,
        )
        .unwrap();

    // Should generate multiple checks: null check + type assertion
    assert!(
        checks.len() >= 2,
        "Nullable should generate null check and type check"
    );

    // First check should be null check
    match checks[0].check_kind {
        RuntimeCheckKind::NullCheck => {}
        _ => panic!("Expected NullCheck for nullable type"),
    }

    // Later checks should include type assertion for inner type
    let has_type_assertion = checks
        .iter()
        .any(|check| matches!(check.check_kind, RuntimeCheckKind::TypeAssertion));
    assert!(
        has_type_assertion,
        "Should have type assertion for inner type"
    );
}

#[test]
fn test_interface_duck_typing() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);
    // Create an interface type
    let interface_type = Type::Interface {
        methods: vec![],
        is_structural: true,
        nullability: Default::default(),
    };

    let dynamic_type = Type::Dynamic;

    // Check dynamic to interface should generate duck type checks
    let checks = checker
        .check_gradual_compatibility(
            &interface_type,
            &dynamic_type,
            BoundaryKind::DynamicToStatic,
            span,
        )
        .unwrap();

    // Should generate duck type check
    assert!(
        !checks.is_empty(),
        "Interface should require duck type checks"
    );

    let has_duck_check = checks
        .iter()
        .any(|check| matches!(check.check_kind, RuntimeCheckKind::DuckTypeCheck { .. }));
    assert!(
        has_duck_check,
        "Should generate duck type check for interface"
    );
}

#[test]
fn test_static_to_dynamic_no_checks() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);

    // Test that static to Dynamic requires no checks
    let static_type = Type::Primitive(PrimitiveType::I32);
    let dynamic_type = Type::Dynamic;

    let checks = checker
        .check_gradual_compatibility(
            &dynamic_type,
            &static_type,
            BoundaryKind::StaticToDynamic,
            span,
        )
        .unwrap();

    assert!(
        checks.is_empty(),
        "Dynamic should accept static types without checks"
    );
}

#[test]
fn test_unknown_type_handling() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);

    // Test that Unknown type requires special handling
    let static_type = Type::Primitive(PrimitiveType::String);
    let unknown_type = Type::Unknown;

    let checks = checker
        .check_gradual_compatibility(
            &static_type,
            &unknown_type,
            BoundaryKind::DynamicToStatic,
            span,
        )
        .unwrap();

    // Unknown should generate runtime checks
    assert!(
        !checks.is_empty(),
        "Unknown type should require runtime checks"
    );
}

#[test]
fn test_exact_type_match_no_checks() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);

    // Test that exact type matches require no runtime checks
    let type1 = Type::Primitive(PrimitiveType::I32);
    let type2 = Type::Primitive(PrimitiveType::I32);

    let checks = checker
        .check_gradual_compatibility(&type1, &type2, BoundaryKind::StaticToDynamic, span)
        .unwrap();

    assert!(
        checks.is_empty(),
        "Exact type match should not require runtime checks"
    );
}

#[test]
fn test_complex_type_with_multiple_checks() {
    let mut checker = GradualTypeChecker::new();
    let span = Span::new(0, 10);

    // Test complex nullable interface type
    let interface_type = Type::Interface {
        methods: vec![],
        is_structural: true,
        nullability: Default::default(),
    };

    let nullable_interface = Type::Nullable(Box::new(interface_type));
    let dynamic_type = Type::Dynamic;

    let checks = checker
        .check_gradual_compatibility(
            &nullable_interface,
            &dynamic_type,
            BoundaryKind::DynamicToStatic,
            span,
        )
        .unwrap();

    // Should generate multiple checks for nullable interface
    assert!(
        checks.len() >= 2,
        "Complex type should generate multiple checks"
    );

    // Should have both null check and duck type check
    let has_null_check = checks
        .iter()
        .any(|c| matches!(c.check_kind, RuntimeCheckKind::NullCheck));
    let has_duck_check = checks
        .iter()
        .any(|c| matches!(c.check_kind, RuntimeCheckKind::DuckTypeCheck { .. }));

    assert!(has_null_check, "Should have null check for nullable type");
    assert!(has_duck_check, "Should have duck type check for interface");
}
