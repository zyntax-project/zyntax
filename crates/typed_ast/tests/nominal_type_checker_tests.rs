//! Tests for the nominal type checking system
//!
//! This module tests inheritance, polymorphism, virtual method dispatch,
//! and interface implementation in the nominal type system.

// Note: These tests use internal types from nominal_type_checker
// use zyntax_typed_ast::*;
use zyntax_typed_ast::arena::InternedString;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::{
    nominal_type_checker::*, MethodSig, NullabilityKind, PrimitiveType, Type, TypeId, TypeParam,
    Variance, Visibility,
};
// use zyntax_typed_ast::type_registry::{Type, PrimitiveType, NullabilityKind, TypeVarKind, Variance};
use std::collections::HashMap;
use string_interner::Symbol;

fn create_test_string(s: &str) -> InternedString {
    use std::sync::{Mutex, OnceLock};
    use string_interner::DefaultStringInterner;

    static INTERNER: OnceLock<Mutex<DefaultStringInterner>> = OnceLock::new();

    let interner = INTERNER.get_or_init(|| Mutex::new(DefaultStringInterner::new()));
    let mut guard = interner.lock().unwrap();
    InternedString::from_symbol(guard.get_or_intern(s))
}

#[test]
fn test_nominal_type_checker_creation() {
    // Test that we can create a nominal type checker
    let checker = NominalTypeChecker::new();

    // Verify initial state
    assert!(checker.type_registry.is_empty());
    assert!(checker.trait_registry.is_empty());
    assert!(checker.impl_registry.is_empty());
    assert!(checker.inheritance_cache.is_empty());
    assert!(checker.variance_cache.is_empty());
    assert!(checker.vtable_cache.is_empty());
    assert!(checker.interface_dispatch_cache.is_empty());
    assert!(checker.compatibility_cache.is_empty());
}

#[test]
fn test_basic_type_registration() {
    let mut checker = NominalTypeChecker::new();

    // Create a basic type definition
    let type_id = TypeId::next();
    let type_def = TypeDefinition {
        id: type_id,
        name: create_test_string("TestClass"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(0, 10),
    };

    // Register the type
    let result = checker.register_type(type_def);
    assert!(result.is_ok());

    // Verify it was registered
    assert!(checker.type_registry.contains_key(&type_id));
    assert_eq!(checker.type_registry.len(), 1);
}

#[test]
fn test_inheritance_hierarchy() {
    let mut checker = NominalTypeChecker::new();

    // Create Object type (root)
    let object_id = TypeId::next();
    let object_type = TypeDefinition {
        id: object_id,
        name: create_test_string("Object"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(0, 10),
    };

    // Create Animal type extending Object
    let animal_id = TypeId::next();
    let animal_type = TypeDefinition {
        id: animal_id,
        name: create_test_string("Animal"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: Some(Type::Named {
            id: object_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: NullabilityKind::Unknown,
        }),
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(10, 20),
    };

    // Register types
    checker
        .register_type(object_type)
        .expect("Should register Object");
    checker
        .register_type(animal_type)
        .expect("Should register Animal");

    // Test inheritance checking

    let animal_type_instance = Type::Named {
        id: animal_id,
        type_args: vec![],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::Unknown,
    };

    let object_type_instance = Type::Named {
        id: object_id,
        type_args: vec![],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::Unknown,
    };

    // Animal should be assignable to Object (inheritance)
    let result = checker.is_subtype(&animal_type_instance, &object_type_instance);
    assert!(result.is_ok());
    assert!(result.unwrap());

    // Object should not be assignable to Animal
    let result = checker.is_subtype(&object_type_instance, &animal_type_instance);
    assert!(result.is_ok());
    assert!(!result.unwrap());
}

#[test]
fn test_trait_registration_and_implementation() {
    let mut checker = NominalTypeChecker::new();

    // Create a trait definition
    let trait_id = TypeId::next();
    let trait_def = TraitDefinition {
        id: trait_id,
        name: create_test_string("Drawable"),
        type_params: vec![],
        super_traits: vec![],
        methods: vec![MethodSig {
            name: create_test_string("draw"),
            type_params: vec![],
            params: vec![],
            return_type: Type::Primitive(PrimitiveType::Unit),
            where_clause: vec![],
            is_static: false,
            is_async: false,
            is_extension: false,
            visibility: Visibility::Public,
            span: Span::new(0, 10),
        }],
        associated_types: vec![],
        default_implementations: HashMap::new(),
        is_object_safe: true,
        visibility: Visibility::Public,
        span: Span::new(0, 20),
    };

    // Register the trait
    checker.register_trait(trait_def);
    // Note: register_trait returns () not Result

    // Verify trait was registered
    assert!(checker.trait_registry.contains_key(&trait_id));
    assert_eq!(checker.trait_registry.len(), 1);

    // Create a type that implements the trait
    let circle_id = TypeId::next();
    let circle_type = TypeDefinition {
        id: circle_id,
        name: create_test_string("Circle"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![MethodSig {
            name: create_test_string("draw"),
            type_params: vec![],
            params: vec![],
            return_type: Type::Primitive(PrimitiveType::Unit),
            where_clause: vec![],
            is_static: false,
            is_async: false,
            is_extension: false,
            visibility: Visibility::Public,
            span: Span::new(20, 30),
        }],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(20, 40),
    };

    // Register the implementing type
    let result = checker.register_type(circle_type);
    assert!(result.is_ok());

    // Create and register the trait implementation
    let impl_def = ImplDefinition {
        trait_id,
        implementing_type: Type::Named {
            id: circle_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: NullabilityKind::Unknown,
        },
        type_params: vec![],
        where_clause: vec![],
        methods: vec![],
        associated_types: HashMap::new(),
        span: Span::new(40, 50),
    };

    checker.register_impl(impl_def);

    // Test trait implementation checking
    let result = checker.implements_trait(circle_id, trait_id);
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_method_resolution() {
    let mut checker = NominalTypeChecker::new();

    // Create a type with a method
    let test_method = MethodSig {
        name: create_test_string("test_method"),
        type_params: vec![],
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::I32),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span: Span::new(0, 15),
    };

    let test_type_id = TypeId::next();
    let type_def = TypeDefinition {
        id: test_type_id,
        name: create_test_string("TestType"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![test_method],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(0, 20),
    };

    checker
        .register_type(type_def)
        .expect("Should register TestType");

    // Test method resolution
    let receiver_type = Type::Named {
        id: test_type_id,
        type_args: vec![],
        const_args: vec![],
        variance: vec![],
        nullability: NullabilityKind::Unknown,
    };

    let resolved = checker.resolve_method(&receiver_type, create_test_string("test_method"), &[]);
    assert!(resolved.is_ok());

    let method = resolved.unwrap();
    assert!(method.is_some());

    // Test non-existent method
    let resolved = checker.resolve_method(&receiver_type, create_test_string("non_existent"), &[]);
    assert!(resolved.is_ok());
    assert!(resolved.unwrap().is_none());
}

#[test]
fn test_virtual_method_table_generation() {
    let mut checker = NominalTypeChecker::new();

    // Create a type with virtual methods
    let virtual_method1 = MethodSig {
        name: create_test_string("virtual1"),
        type_params: vec![],
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::Unit),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span: Span::new(0, 10),
    };

    let virtual_method2 = MethodSig {
        name: create_test_string("virtual2"),
        type_params: vec![],
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::Unit),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span: Span::new(10, 20),
    };

    let virtual_type_id = TypeId::next();
    let type_def = TypeDefinition {
        id: virtual_type_id,
        name: create_test_string("VirtualType"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![virtual_method1, virtual_method2],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(0, 30),
    };

    checker
        .register_type(type_def)
        .expect("Should register VirtualType");

    // Test VTable generation
    let vtable_result = checker.build_virtual_method_table(virtual_type_id);
    assert!(vtable_result.is_ok());

    let vtable = vtable_result.unwrap();
    assert_eq!(vtable.type_id, virtual_type_id);
    assert_eq!(vtable.methods.len(), 2);

    // Verify methods are in the VTable
    assert!(vtable.methods.contains_key(&create_test_string("virtual1")));
    assert!(vtable.methods.contains_key(&create_test_string("virtual2")));
}

#[test]
fn test_error_cases() {
    let mut checker = NominalTypeChecker::new();

    // Test circular inheritance detection
    let type1_id = TypeId::next();
    let type2_id = TypeId::next();
    let type1 = TypeDefinition {
        id: type1_id,
        name: create_test_string("Type1"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: Some(Type::Named {
            id: type2_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: NullabilityKind::Unknown,
        }),
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(0, 10),
    };

    let type2 = TypeDefinition {
        id: type2_id,
        name: create_test_string("Type2"),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: Some(Type::Named {
            id: type1_id, // Creates cycle
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: NullabilityKind::Unknown,
        }),
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(10, 20),
    };

    // Register first type
    checker.register_type(type1).expect("Should register Type1");

    // Attempting to register second type should fail due to circular inheritance
    let result = checker.register_type(type2);
    assert!(result.is_err());

    match result.unwrap_err() {
        NominalTypeError::CircularInheritance { .. } => {
            // Expected error
        }
        other => panic!("Expected CircularInheritance error, got {:?}", other),
    }
}

#[test]
fn test_generic_type_handling() {
    let mut checker = NominalTypeChecker::new();

    // Create a generic type parameter
    let generic_param = TypeParam {
        name: create_test_string("T"),
        bounds: vec![],
        variance: Variance::Invariant,
        default: None,
        span: Span::new(0, 10),
    };

    // Create a generic type definition
    let generic_type_id = TypeId::next();
    let generic_type = TypeDefinition {
        id: generic_type_id,
        name: create_test_string("GenericType"),
        kind: TypeKind::Class,
        type_params: vec![generic_param],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span: Span::new(0, 20),
    };

    let result = checker.register_type(generic_type);
    assert!(result.is_ok());

    // Verify the generic type was registered with proper variance information
    let variance_info = checker.variance_cache.get(&generic_type_id);
    assert!(variance_info.is_some());
    assert_eq!(variance_info.unwrap().len(), 1);
    assert_eq!(variance_info.unwrap()[0], Variance::Invariant);
}
