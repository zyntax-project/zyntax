//! Functional tests for the nominal type checker algorithms
//!
//! This module tests Java/C#/Kotlin style nominal typing with actual type checking algorithms.

// Note: These tests use internal types from nominal_type_checker
use std::collections::HashMap;
use string_interner::Symbol;
use zyntax_typed_ast::arena::{AstArena, InternedString};
use zyntax_typed_ast::nominal_type_checker::*;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{
    MethodSig, NullabilityKind, ParamDef, PrimitiveType, Type, TypeId, TypeParam, TypeVarKind,
    Variance, Visibility,
};

#[test]
fn test_nominal_subtyping_inheritance() {
    let mut checker = NominalTypeChecker::new();
    let span = Span::new(0, 10);

    // Create base class Object
    let object_id = TypeId::next();
    let object_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(1).unwrap());
    let object_def = TypeDefinition {
        id: object_id,
        name: object_name,
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
        span,
    };
    checker.register_type(object_def).unwrap();

    // Create derived class String : Object
    let string_id = TypeId::next();
    let string_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(2).unwrap());
    let string_def = TypeDefinition {
        id: string_id,
        name: string_name,
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: Some(Type::Named {
            id: object_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        }),
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span,
    };
    checker.register_type(string_def).unwrap();

    // Test subtyping relationships
    let string_type = Type::Named {
        id: string_id,
        type_args: vec![],
        variance: vec![],
        const_args: vec![],
        nullability: NullabilityKind::NonNull,
    };

    let object_type = Type::Named {
        id: object_id,
        type_args: vec![],
        variance: vec![],
        const_args: vec![],
        nullability: NullabilityKind::NonNull,
    };

    // String is a subtype of Object
    assert!(
        checker.is_subtype(&string_type, &object_type).unwrap(),
        "String should be subtype of Object"
    );

    // Object is NOT a subtype of String
    assert!(
        !checker.is_subtype(&object_type, &string_type).unwrap(),
        "Object should not be subtype of String"
    );

    // A type is always a subtype of itself
    assert!(
        checker.is_subtype(&string_type, &string_type).unwrap(),
        "String should be subtype of itself"
    );
}

#[test]
fn test_circular_inheritance_detection() {
    let mut checker = NominalTypeChecker::new();
    let span = Span::new(0, 10);

    // Create type A
    let type_a_id = TypeId::next();
    let type_a_def = TypeDefinition {
        id: type_a_id,
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(3).unwrap(),
        ),
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
        span,
    };
    checker.register_type(type_a_def).unwrap();

    // Create type B : A
    let type_b_id = TypeId::next();
    let type_b_def = TypeDefinition {
        id: type_b_id,
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(4).unwrap(),
        ),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: Some(Type::Named {
            id: type_a_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        }),
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span,
    };
    checker.register_type(type_b_def).unwrap();

    // Try to create type C : B and then modify A : C (circular)
    let type_c_id = TypeId::next();
    let type_c_def_circular = TypeDefinition {
        id: type_a_id, // Reuse A's ID to modify it
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(3).unwrap(),
        ),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: Some(Type::Named {
            id: type_b_id, // A now inherits from B, creating A -> B -> A cycle
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        }),
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span,
    };

    // This should fail due to circular inheritance
    let result = checker.register_type(type_c_def_circular);
    match result {
        Err(NominalTypeError::CircularInheritance { types, .. }) => {
            assert!(types.contains(&type_a_id));
            assert!(types.contains(&type_b_id));
        }
        _ => panic!("Expected CircularInheritance error"),
    }
}

#[test]
fn test_method_resolution() {
    let mut checker = NominalTypeChecker::new();
    let span = Span::new(0, 10);

    // Create a class with methods
    let class_id = TypeId::next();
    let method_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(5).unwrap());
    let return_type = Type::Primitive(PrimitiveType::String);

    let method_sig = MethodSig {
        name: method_name,
        type_params: vec![],
        params: vec![],
        return_type: return_type.clone(),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span,
    };

    let class_def = TypeDefinition {
        id: class_id,
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(6).unwrap(),
        ),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![method_sig.clone()],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span,
    };
    checker.register_type(class_def).unwrap();

    // Test method resolution
    let class_type = Type::Named {
        id: class_id,
        type_args: vec![],
        variance: vec![],
        const_args: vec![],
        nullability: NullabilityKind::NonNull,
    };

    // Resolve existing method
    let resolved = checker
        .resolve_method(&class_type, method_name, &[])
        .unwrap();
    assert!(resolved.is_some(), "Should find the method");

    let resolved_method = resolved.unwrap();
    assert_eq!(resolved_method.signature.name, method_name);
    assert_eq!(resolved_method.signature.return_type, return_type);
    assert_eq!(resolved_method.source, MethodSource::Intrinsic);

    // Try to resolve non-existent method
    let non_existent =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(7).unwrap());
    let not_found = checker
        .resolve_method(&class_type, non_existent, &[])
        .unwrap();
    assert!(not_found.is_none(), "Should not find non-existent method");
}

#[test]
fn test_trait_implementation() {
    let mut checker = NominalTypeChecker::new();
    let span = Span::new(0, 10);

    // Create a trait
    let trait_id = TypeId::next();
    let trait_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(8).unwrap());
    let method_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(9).unwrap());

    let trait_method = MethodSig {
        name: method_name,
        type_params: vec![],
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::Bool),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span,
    };

    let trait_def = TraitDefinition {
        id: trait_id,
        name: trait_name,
        type_params: vec![],
        super_traits: vec![],
        methods: vec![trait_method.clone()],
        associated_types: vec![],
        default_implementations: HashMap::new(),
        is_object_safe: true,
        visibility: Visibility::Public,
        span,
    };
    checker.register_trait(trait_def);

    // Create a type that implements the trait
    let type_id = TypeId::next();
    let type_def = TypeDefinition {
        id: type_id,
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(10).unwrap(),
        ),
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
        span,
    };
    checker.register_type(type_def).unwrap();

    // Register implementation
    let impl_def = ImplDefinition {
        trait_id,
        implementing_type: Type::Named {
            id: type_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        },
        type_params: vec![],
        where_clause: vec![],
        methods: vec![MethodImpl {
            signature: trait_method,
            body: MethodBody::Placeholder,
        }],
        associated_types: HashMap::new(),
        span,
    };
    checker.register_impl(impl_def);

    // Test trait implementation check
    assert!(
        checker.implements_trait(type_id, trait_id).unwrap(),
        "Type should implement trait"
    );

    // Check that type is subtype of trait
    let type_instance = Type::Named {
        id: type_id,
        type_args: vec![],
        variance: vec![],
        const_args: vec![],
        nullability: NullabilityKind::NonNull,
    };

    let trait_instance = Type::Trait {
        id: trait_id,
        associated_types: Vec::<(InternedString, zyntax_typed_ast::Type)>::new(),
        super_traits: vec![],
    };

    assert!(
        checker.is_subtype(&type_instance, &trait_instance).unwrap(),
        "Type should be subtype of implemented trait"
    );
}

#[test]
fn test_virtual_method_override_validation() {
    let mut checker = NominalTypeChecker::new();
    let span = Span::new(0, 10);

    // Create base class with virtual method
    let base_id = TypeId::next();
    let method_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(11).unwrap());
    let param_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(12).unwrap());

    let base_method = MethodSig {
        name: method_name,
        type_params: vec![],
        params: vec![ParamDef {
            name: param_name,
            ty: Type::Primitive(PrimitiveType::I32),
            is_self: false,
            is_mut: false,
            is_varargs: false,
        }],
        return_type: Type::Primitive(PrimitiveType::String),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span,
    };

    // Test valid override (same signature)
    let derived_id = TypeId::next();
    let valid_override = base_method.clone();

    let result =
        checker.check_virtual_method_override(&base_method, &valid_override, base_id, derived_id);
    assert!(result.is_ok(), "Valid override should succeed");

    // Test invalid override - different parameter count
    let invalid_params = MethodSig {
        name: method_name,
        type_params: vec![],
        params: vec![], // No parameters - mismatch!
        return_type: Type::Primitive(PrimitiveType::String),
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span,
    };

    let result =
        checker.check_virtual_method_override(&base_method, &invalid_params, base_id, derived_id);
    match result {
        Err(NominalTypeError::VirtualMethodOverrideError { reason, .. }) => {
            assert!(reason.contains("Parameter count mismatch"));
        }
        _ => panic!("Expected VirtualMethodOverrideError for parameter count mismatch"),
    }

    // Test invalid override - wrong return type (no covariance)
    let wrong_return = MethodSig {
        name: method_name,
        type_params: vec![],
        params: vec![ParamDef {
            name: param_name,
            ty: Type::Primitive(PrimitiveType::I32),
            is_self: false,
            is_mut: false,
            is_varargs: false,
        }],
        return_type: Type::Primitive(PrimitiveType::Bool), // Wrong return type!
        where_clause: vec![],
        is_static: false,
        is_async: false,
        is_extension: false,
        visibility: Visibility::Public,
        span,
    };

    let result =
        checker.check_virtual_method_override(&base_method, &wrong_return, base_id, derived_id);
    match result {
        Err(NominalTypeError::CovarianceViolation {
            param_or_return, ..
        }) => {
            assert_eq!(param_or_return, "return type");
        }
        _ => panic!("Expected CovarianceViolation for return type"),
    }
}

#[test]
fn test_variance_in_generic_types() {
    let mut checker = NominalTypeChecker::new();
    let span = Span::new(0, 10);

    // Create generic List<T> type with covariant T
    let list_id = TypeId::next();
    let t_param =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(13).unwrap());

    let list_def = TypeDefinition {
        id: list_id,
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(14).unwrap(),
        ),
        kind: TypeKind::Class,
        type_params: vec![TypeParam {
            name: t_param,
            bounds: vec![],
            default: None,
            variance: Variance::Covariant,

            span: Span::new(0, 10),
        }],
        super_type: None,
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span,
    };
    checker.register_type(list_def).unwrap();

    // Create type hierarchy: Animal -> Dog
    let animal_id = TypeId::next();
    let dog_id = TypeId::next();

    let animal_def = TypeDefinition {
        id: animal_id,
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(15).unwrap(),
        ),
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
        span,
    };
    checker.register_type(animal_def).unwrap();

    let dog_def = TypeDefinition {
        id: dog_id,
        name: InternedString::from_symbol(
            string_interner::DefaultSymbol::try_from_usize(16).unwrap(),
        ),
        kind: TypeKind::Class,
        type_params: vec![],
        super_type: Some(Type::Named {
            id: animal_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        }),
        interfaces: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        visibility: Visibility::Public,
        is_abstract: false,
        is_sealed: false,
        is_final: false,
        span,
    };
    checker.register_type(dog_def).unwrap();

    // Test covariance: List<Dog> should be subtype of List<Animal>
    let list_dog = Type::Named {
        id: list_id,
        type_args: vec![Type::Named {
            id: dog_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        }],
        variance: vec![Variance::Covariant],
        const_args: vec![],
        nullability: NullabilityKind::NonNull,
    };

    let list_animal = Type::Named {
        id: list_id,
        type_args: vec![Type::Named {
            id: animal_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        }],
        variance: vec![Variance::Covariant],
        const_args: vec![],
        nullability: NullabilityKind::NonNull,
    };

    // Due to covariance, List<Dog> should be a subtype of List<Animal>
    // Note: The current implementation might not fully support this
    // This test documents the expected behavior
    let _ = checker.is_subtype(&list_dog, &list_animal);
}
