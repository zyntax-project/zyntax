//! Tests for built-in types like Result<T, E> and Option<T>
//!
//! These tests verify that the Result and Option types can be registered
//! and used correctly in the type system.

use zyntax_typed_ast::arena::AstArena;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type, TypeKind, TypeRegistry, VariantFields};

#[test]
fn test_register_result_type() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register the Result<T, E> type
    let result_type_id = registry.register_result_type(&mut arena, span);

    // Verify the type was registered
    let result_name = arena.intern_string("Result");
    let type_def = registry
        .get_type_by_name(result_name)
        .expect("Result type should be registered");
    assert_eq!(type_def.id, result_type_id);
    assert_eq!(type_def.name, result_name);
    assert_eq!(type_def.type_params.len(), 2); // T and E

    // Verify it's an enum
    match &type_def.kind {
        TypeKind::Enum { variants } => {
            assert_eq!(variants.len(), 2);

            // Check Ok variant
            let ok_name = arena.intern_string("Ok");
            let ok_variant = variants.iter().find(|v| v.name == ok_name).unwrap();
            assert_eq!(ok_variant.discriminant, Some(0));
            match &ok_variant.fields {
                VariantFields::Tuple(fields) => {
                    assert_eq!(fields.len(), 1); // Contains T
                }
                _ => panic!("Expected Ok to be a tuple variant"),
            }

            // Check Err variant
            let err_name = arena.intern_string("Err");
            let err_variant = variants.iter().find(|v| v.name == err_name).unwrap();
            assert_eq!(err_variant.discriminant, Some(1));
            match &err_variant.fields {
                VariantFields::Tuple(fields) => {
                    assert_eq!(fields.len(), 1); // Contains E
                }
                _ => panic!("Expected Err to be a tuple variant"),
            }
        }
        _ => panic!("Expected Result to be an enum"),
    }
}

#[test]
fn test_register_option_type() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register the Option<T> type
    let option_type_id = registry.register_option_type(&mut arena, span);

    // Verify the type was registered
    let option_name = arena.intern_string("Option");
    let type_def = registry
        .get_type_by_name(option_name)
        .expect("Option type should be registered");
    assert_eq!(type_def.id, option_type_id);
    assert_eq!(type_def.name, option_name);
    assert_eq!(type_def.type_params.len(), 1); // T

    // Verify it's an enum
    match &type_def.kind {
        TypeKind::Enum { variants } => {
            assert_eq!(variants.len(), 2);

            // Check Some variant
            let some_name = arena.intern_string("Some");
            let some_variant = variants.iter().find(|v| v.name == some_name).unwrap();
            assert_eq!(some_variant.discriminant, Some(0));
            match &some_variant.fields {
                VariantFields::Tuple(fields) => {
                    assert_eq!(fields.len(), 1); // Contains T
                }
                _ => panic!("Expected Some to be a tuple variant"),
            }

            // Check None variant
            let none_name = arena.intern_string("None");
            let none_variant = variants.iter().find(|v| v.name == none_name).unwrap();
            assert_eq!(none_variant.discriminant, Some(1));
            match &none_variant.fields {
                VariantFields::Unit => {
                    // None has no fields
                }
                _ => panic!("Expected None to be a unit variant"),
            }
        }
        _ => panic!("Expected Option to be an enum"),
    }
}

#[test]
fn test_instantiate_result_with_concrete_types() {
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

    // Verify type structure
    match result_i32_string {
        Type::Named { id, type_args, .. } => {
            assert_eq!(id, result_type_id);
            assert_eq!(type_args.len(), 2);
            assert_eq!(type_args[0], Type::Primitive(PrimitiveType::I32));
            assert_eq!(type_args[1], Type::Primitive(PrimitiveType::String));
        }
        _ => panic!("Expected Named type"),
    }
}

#[test]
fn test_instantiate_option_with_concrete_type() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let span = Span::default();

    // Register Option<T>
    let option_type_id = registry.register_option_type(&mut arena, span);

    // Create Option<i32>
    let option_i32 = Type::Named {
        id: option_type_id,
        type_args: vec![Type::Primitive(PrimitiveType::I32)],
        const_args: vec![],
        variance: vec![],
        nullability: Default::default(),
    };

    // Verify type structure
    match option_i32 {
        Type::Named { id, type_args, .. } => {
            assert_eq!(id, option_type_id);
            assert_eq!(type_args.len(), 1);
            assert_eq!(type_args[0], Type::Primitive(PrimitiveType::I32));
        }
        _ => panic!("Expected Named type"),
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
    // This represents a nested result where the success value is itself a Result
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
    match outer_result {
        Type::Named { id, type_args, .. } => {
            assert_eq!(id, result_type_id);
            assert_eq!(type_args.len(), 2);

            // Check inner result
            match &type_args[0] {
                Type::Named {
                    id: inner_id,
                    type_args: inner_type_args,
                    ..
                } => {
                    assert_eq!(*inner_id, result_type_id);
                    assert_eq!(inner_type_args.len(), 2);
                }
                _ => panic!("Expected inner type to be a Named type"),
            }
        }
        _ => panic!("Expected outer type to be a Named type"),
    }
}
