//! Tests for Result<T, E> type - Gap 8 Phase 1
//!
//! These tests verify that:
//! 1. Result<T, E> can be registered in TypeRegistry
//! 2. Result types can be instantiated with concrete types
//! 3. Result types work with the type system

use zyntax_typed_ast::arena::AstArena;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, TypeKind, TypeRegistry, VariantFields};

#[test]
fn test_result_type_in_type_registry() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register Result<T, E>
    let result_type_id = registry.register_result_type(&mut arena, span);

    // Verify registration
    let result_name = arena.intern_string("Result");
    let type_def = registry.get_type_by_name(result_name).unwrap();

    assert_eq!(type_def.id, result_type_id);
    assert_eq!(type_def.type_params.len(), 2);

    println!("✅ Result<T, E> successfully registered in TypeRegistry");
}

#[test]
fn test_result_instantiation() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register Result<T, E>
    let result_type_id = registry.register_result_type(&mut arena, span);

    // Create Result<i32, String>
    let result_i32_string = Type::Named {
        id: result_type_id,
        type_args: vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::String),
        ],
        const_args: vec![],
        variance: vec![],
        nullability: Default::default(),
    };

    // Verify structure
    match result_i32_string {
        Type::Named { id, type_args, .. } => {
            assert_eq!(id, result_type_id);
            assert_eq!(type_args.len(), 2);
            assert_eq!(type_args[0], Type::Primitive(PrimitiveType::I32));
            assert_eq!(type_args[1], Type::Primitive(PrimitiveType::String));

            println!("✅ Result<i32, String> instantiated successfully");
        }
        _ => panic!("Expected Named type"),
    }
}

#[test]
fn test_result_enum_structure() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register Result<T, E>
    let result_type_id = registry.register_result_type(&mut arena, span);

    // Get the type definition
    let result_name = arena.intern_string("Result");
    let type_def = registry.get_type_by_name(result_name).unwrap();

    // Verify it's an enum with correct structure
    match &type_def.kind {
        TypeKind::Enum { variants } => {
            assert_eq!(variants.len(), 2);

            // Check Ok variant
            let ok_name = arena.intern_string("Ok");
            let ok_variant = variants
                .iter()
                .find(|v| v.name == ok_name)
                .expect("Ok variant should exist");

            match &ok_variant.fields {
                VariantFields::Tuple(fields) => {
                    assert_eq!(fields.len(), 1); // Contains T
                    println!("✅ Ok variant has correct structure (tuple with 1 field)");
                }
                _ => panic!("Ok should be a tuple variant"),
            }

            // Check Err variant
            let err_name = arena.intern_string("Err");
            let err_variant = variants
                .iter()
                .find(|v| v.name == err_name)
                .expect("Err variant should exist");

            match &err_variant.fields {
                VariantFields::Tuple(fields) => {
                    assert_eq!(fields.len(), 1); // Contains E
                    println!("✅ Err variant has correct structure (tuple with 1 field)");
                }
                _ => panic!("Err should be a tuple variant"),
            }

            println!("✅ Result<T, E> enum structure verified in TypeRegistry");
        }
        _ => panic!("Expected Enum kind"),
    }
}

#[test]
fn test_multiple_result_instantiations() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register Result<T, E>
    let result_type_id = registry.register_result_type(&mut arena, span);

    // Create Result<i32, String>
    let result_i32_string = Type::Named {
        id: result_type_id,
        type_args: vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::String),
        ],
        const_args: vec![],
        variance: vec![],
        nullability: Default::default(),
    };

    // Create Result<f64, i32>
    let result_f64_i32 = Type::Named {
        id: result_type_id,
        type_args: vec![
            Type::Primitive(PrimitiveType::F64),
            Type::Primitive(PrimitiveType::I32),
        ],
        const_args: vec![],
        variance: vec![],
        nullability: Default::default(),
    };

    // Both should reference the same generic Result type
    match (&result_i32_string, &result_f64_i32) {
        (
            Type::Named {
                id: id1,
                type_args: args1,
                ..
            },
            Type::Named {
                id: id2,
                type_args: args2,
                ..
            },
        ) => {
            assert_eq!(*id1, *id2); // Same Result type ID
            assert_ne!(args1, args2); // Different type arguments

            println!("✅ Multiple Result instantiations use same generic type");
            println!("   - Result<i32, String>");
            println!("   - Result<f64, i32>");
        }
        _ => panic!("Expected Named types"),
    }
}

#[test]
fn test_nested_result_types() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register Result<T, E>
    let result_type_id = registry.register_result_type(&mut arena, span);

    // Create Result<Result<i32, String>, String>
    let inner_result = Type::Named {
        id: result_type_id,
        type_args: vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::String),
        ],
        const_args: vec![],
        variance: vec![],
        nullability: Default::default(),
    };

    let outer_result = Type::Named {
        id: result_type_id,
        type_args: vec![inner_result.clone(), Type::Primitive(PrimitiveType::String)],
        const_args: vec![],
        variance: vec![],
        nullability: Default::default(),
    };

    // Verify nested structure
    match &outer_result {
        Type::Named { type_args, .. } => {
            assert_eq!(type_args.len(), 2);

            // Check inner Result
            match &type_args[0] {
                Type::Named {
                    id,
                    type_args: inner_type_args,
                    ..
                } => {
                    assert_eq!(*id, result_type_id);
                    assert_eq!(inner_type_args.len(), 2);
                    println!("✅ Nested Result<Result<i32, String>, String> verified");
                }
                _ => panic!("Expected inner type to be Named"),
            }
        }
        _ => panic!("Expected outer type to be Named"),
    }
}

#[test]
fn test_option_type_for_comparison() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register Option<T> (similar to Result but with one type param)
    let option_type_id = registry.register_option_type(&mut arena, span);

    // Verify registration
    let option_name = arena.intern_string("Option");
    let type_def = registry.get_type_by_name(option_name).unwrap();

    assert_eq!(type_def.id, option_type_id);
    assert_eq!(type_def.type_params.len(), 1); // Only T, not E

    println!("✅ Option<T> registered for comparison with Result<T, E>");
}
