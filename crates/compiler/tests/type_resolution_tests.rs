//! Integration tests for named type resolution
//!
//! These tests verify that user-defined types (structs, enums) are correctly
//! resolved from the TypeRegistry and converted to appropriate HIR types.

use std::sync::{Arc, Mutex};
use zyntax_compiler::{
    hir::{HirModule, HirStructType, HirType, HirUnionType},
    lowering::{AstLowering, LoweringConfig, LoweringContext},
    CompilerResult,
};
use zyntax_typed_ast::{
    arena::AstArena, typed_ast::TypedBlock, typed_node, CallingConvention, ConstructorSig,
    FieldDef, MethodSig, Mutability, PrimitiveType, Span, Type, TypeConstraint, TypeDefinition,
    TypeId, TypeKind, TypeMetadata, TypeParam, TypeRegistry, TypedDeclaration, TypedFunction,
    TypedNode, TypedProgram, VariantDef, VariantFields, Visibility,
};

fn test_span() -> Span {
    Span::new(0, 10)
}

#[test]
fn test_struct_type_resolution() {
    let mut arena = AstArena::new();
    let mut type_registry = TypeRegistry::new();

    // Register a struct type: Point { x: i32, y: i32 }
    let point_name = arena.intern_string("Point");
    let x_name = arena.intern_string("x");
    let y_name = arena.intern_string("y");

    let point_id = type_registry.register_struct_type(
        point_name,
        vec![], // No type params
        vec![
            FieldDef {
                name: x_name,
                ty: Type::Primitive(PrimitiveType::I32),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                is_synthetic: false,
                span: test_span(),
                getter: None,
                setter: None,
            },
            FieldDef {
                name: y_name,
                ty: Type::Primitive(PrimitiveType::I32),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                is_synthetic: false,
                span: test_span(),
                getter: None,
                setter: None,
            },
        ],
        vec![], // No methods
        vec![], // No constructors
        TypeMetadata::default(),
        test_span(),
    );

    // Create a simple function with Point type
    let func_name = arena.intern_string("test_point");
    let function = TypedFunction {
        name: func_name,
        params: vec![],
        type_params: vec![],
        return_type: Type::Named {
            id: point_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        },
        body: Some(TypedBlock {
            statements: vec![],
            span: test_span(),
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![typed_node(
            TypedDeclaration::Function(function),
            Type::Primitive(PrimitiveType::Unit),
            test_span(),
        )],
        span: test_span(),
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    // Lower to HIR
    let type_registry = Arc::new(type_registry);
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("test_module");
    let arena_arc = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry.clone(), arena_arc, config);

    let result = ctx.lower_program(&mut program);
    assert!(
        result.is_ok(),
        "Failed to lower program with struct type: {:?}",
        result.err()
    );

    let module = result.unwrap();

    // Verify the function signature has struct type
    let func = module.functions.values().next().expect("No function found");
    let return_ty = &func.signature.returns[0];

    match return_ty {
        HirType::Struct(struct_ty) => {
            assert_eq!(struct_ty.name, Some(point_name), "Struct name mismatch");
            assert_eq!(struct_ty.fields.len(), 2, "Expected 2 fields");
            assert!(
                matches!(struct_ty.fields[0], HirType::I32),
                "Field 0 should be I32"
            );
            assert!(
                matches!(struct_ty.fields[1], HirType::I32),
                "Field 1 should be I32"
            );
        }
        other => panic!("Expected HirType::Struct, got {:?}", other),
    }
}

#[test]
fn test_enum_type_resolution() {
    let mut arena = AstArena::new();
    let mut type_registry = TypeRegistry::new();

    // Register an enum: Option { None, Some(T) }
    let option_name = arena.intern_string("Option");
    let none_name = arena.intern_string("None");
    let some_name = arena.intern_string("Some");

    let option_id = TypeId::next();
    let type_def = zyntax_typed_ast::TypeDefinition {
        id: option_id,
        name: option_name,
        kind: TypeKind::Enum {
            variants: vec![
                VariantDef {
                    name: none_name,
                    fields: VariantFields::Unit,
                    discriminant: Some(0),
                    span: test_span(),
                },
                VariantDef {
                    name: some_name,
                    fields: VariantFields::Tuple(vec![Type::Primitive(PrimitiveType::I32)]),
                    discriminant: Some(1),
                    span: test_span(),
                },
            ],
        },
        type_params: vec![], // No type params for this simplified version
        constraints: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        metadata: TypeMetadata::default(),
        span: test_span(),
    };
    type_registry.register_type(type_def);

    // Create a function returning Option type
    let func_name = arena.intern_string("test_option");
    let function = TypedFunction {
        name: func_name,
        params: vec![],
        type_params: vec![],
        return_type: Type::Named {
            id: option_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        },
        body: Some(TypedBlock {
            statements: vec![],
            span: test_span(),
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![typed_node(
            TypedDeclaration::Function(function),
            Type::Primitive(PrimitiveType::Unit),
            test_span(),
        )],
        span: test_span(),
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    // Lower to HIR
    let type_registry = Arc::new(type_registry);
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("test_module");
    let arena_arc = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry.clone(), arena_arc, config);

    let result = ctx.lower_program(&mut program);
    assert!(
        result.is_ok(),
        "Failed to lower program with enum type: {:?}",
        result.err()
    );

    let module = result.unwrap();

    // Verify the function signature has union type
    let func = module.functions.values().next().expect("No function found");
    let return_ty = &func.signature.returns[0];

    match return_ty {
        HirType::Union(union_ty) => {
            assert_eq!(union_ty.name, Some(option_name), "Union name mismatch");
            assert_eq!(union_ty.variants.len(), 2, "Expected 2 variants");

            // First variant (None) should be Void
            assert!(
                matches!(union_ty.variants[0].ty, HirType::Void),
                "None variant should be Void"
            );

            // Second variant (Some) should be a struct with one I32 field
            match &union_ty.variants[1].ty {
                HirType::Struct(s) => {
                    assert_eq!(s.fields.len(), 1, "Some variant should have 1 field");
                    assert!(
                        matches!(s.fields[0], HirType::I32),
                        "Some field should be I32"
                    );
                }
                other => panic!("Expected Struct for Some variant, got {:?}", other),
            }
        }
        other => panic!("Expected HirType::Union, got {:?}", other),
    }
}

#[test]
fn test_type_alias_resolution() {
    let mut arena = AstArena::new();
    let mut type_registry = TypeRegistry::new();

    // Register type alias: type Int = i32
    let int_alias_name = arena.intern_string("Int");

    type_registry.register_alias(int_alias_name, Type::Primitive(PrimitiveType::I32));

    // Create a TypeId for the alias (so we can reference it)
    let alias_id = TypeId::next();
    let alias_type_def = TypeDefinition {
        id: alias_id,
        name: int_alias_name,
        kind: TypeKind::Alias {
            target: Type::Primitive(PrimitiveType::I32),
        },
        type_params: vec![],
        constraints: vec![],
        fields: vec![],
        methods: vec![],
        constructors: vec![],
        metadata: TypeMetadata::default(),
        span: test_span(),
    };
    type_registry.register_type(alias_type_def);

    // Create a function using the alias
    let func_name = arena.intern_string("test_alias");
    let function = TypedFunction {
        name: func_name,
        params: vec![],
        type_params: vec![],
        return_type: Type::Named {
            id: alias_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        },
        body: Some(TypedBlock {
            statements: vec![],
            span: test_span(),
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![typed_node(
            TypedDeclaration::Function(function),
            Type::Primitive(PrimitiveType::Unit),
            test_span(),
        )],
        span: test_span(),
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    // Lower to HIR
    let type_registry = Arc::new(type_registry);
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("test_module");
    let arena_arc = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry.clone(), arena_arc, config);

    let result = ctx.lower_program(&mut program);
    assert!(
        result.is_ok(),
        "Failed to lower program with type alias: {:?}",
        result.err()
    );

    let module = result.unwrap();

    // Verify the alias was resolved to I32
    let func = module.functions.values().next().expect("No function found");
    let return_ty = &func.signature.returns[0];

    assert!(
        matches!(return_ty, HirType::I32),
        "Type alias should resolve to I32, got {:?}",
        return_ty
    );
}

#[test]
fn test_nested_struct_resolution() {
    let mut arena = AstArena::new();
    let mut type_registry = TypeRegistry::new();

    // Register inner struct: Inner { value: i32 }
    let inner_name = arena.intern_string("Inner");
    let value_name = arena.intern_string("value");

    let inner_id = type_registry.register_struct_type(
        inner_name,
        vec![], // No type params
        vec![FieldDef {
            name: value_name,
            ty: Type::Primitive(PrimitiveType::I32),
            visibility: Visibility::Public,
            mutability: Mutability::Immutable,
            is_static: false,
            is_synthetic: false,
            span: test_span(),
            getter: None,
            setter: None,
        }],
        vec![], // No methods
        vec![], // No constructors
        TypeMetadata::default(),
        test_span(),
    );

    // Register outer struct: Outer { inner: Inner }
    let outer_name = arena.intern_string("Outer");
    let inner_field_name = arena.intern_string("inner");

    let outer_id = type_registry.register_struct_type(
        outer_name,
        vec![], // No type params
        vec![FieldDef {
            name: inner_field_name,
            ty: Type::Named {
                id: inner_id,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            visibility: Visibility::Public,
            mutability: Mutability::Immutable,
            is_static: false,
            is_synthetic: false,
            span: test_span(),
            getter: None,
            setter: None,
        }],
        vec![], // No methods
        vec![], // No constructors
        TypeMetadata::default(),
        test_span(),
    );

    // Create a function using Outer type
    let func_name = arena.intern_string("test_nested");
    let function = TypedFunction {
        name: func_name,
        params: vec![],
        type_params: vec![],
        return_type: Type::Named {
            id: outer_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        },
        body: Some(TypedBlock {
            statements: vec![],
            span: test_span(),
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![typed_node(
            TypedDeclaration::Function(function),
            Type::Primitive(PrimitiveType::Unit),
            test_span(),
        )],
        span: test_span(),
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    // Lower to HIR
    let type_registry = Arc::new(type_registry);
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("test_module");
    let arena_arc = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry.clone(), arena_arc, config);

    let result = ctx.lower_program(&mut program);
    assert!(
        result.is_ok(),
        "Failed to lower program with nested struct: {:?}",
        result.err()
    );

    let module = result.unwrap();

    // Verify nested struct resolution
    let func = module.functions.values().next().expect("No function found");
    let return_ty = &func.signature.returns[0];

    match return_ty {
        HirType::Struct(outer_struct) => {
            assert_eq!(
                outer_struct.name,
                Some(outer_name),
                "Outer struct name mismatch"
            );
            assert_eq!(outer_struct.fields.len(), 1, "Outer should have 1 field");

            // Verify inner struct was also resolved
            match &outer_struct.fields[0] {
                HirType::Struct(inner_struct) => {
                    assert_eq!(
                        inner_struct.name,
                        Some(inner_name),
                        "Inner struct name mismatch"
                    );
                    assert_eq!(inner_struct.fields.len(), 1, "Inner should have 1 field");
                    assert!(
                        matches!(inner_struct.fields[0], HirType::I32),
                        "Inner field should be I32"
                    );
                }
                other => panic!("Expected nested struct, got {:?}", other),
            }
        }
        other => panic!("Expected HirType::Struct for outer, got {:?}", other),
    }
}

#[test]
fn test_multiple_struct_types() {
    let mut arena = AstArena::new();
    let mut type_registry = TypeRegistry::new();

    // Register multiple struct types
    let point_name = arena.intern_string("Point");
    let rect_name = arena.intern_string("Rectangle");

    let point_id = type_registry.register_struct_type(
        point_name,
        vec![], // No type params
        vec![
            FieldDef {
                name: arena.intern_string("x"),
                ty: Type::Primitive(PrimitiveType::F32),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                is_synthetic: false,
                span: test_span(),
                getter: None,
                setter: None,
            },
            FieldDef {
                name: arena.intern_string("y"),
                ty: Type::Primitive(PrimitiveType::F32),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                is_synthetic: false,
                span: test_span(),
                getter: None,
                setter: None,
            },
        ],
        vec![], // No methods
        vec![], // No constructors
        TypeMetadata::default(),
        test_span(),
    );

    let rect_id = type_registry.register_struct_type(
        rect_name,
        vec![], // No type params
        vec![
            FieldDef {
                name: arena.intern_string("width"),
                ty: Type::Primitive(PrimitiveType::F32),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                is_synthetic: false,
                span: test_span(),
                getter: None,
                setter: None,
            },
            FieldDef {
                name: arena.intern_string("height"),
                ty: Type::Primitive(PrimitiveType::F32),
                visibility: Visibility::Public,
                mutability: Mutability::Immutable,
                is_static: false,
                is_synthetic: false,
                span: test_span(),
                getter: None,
                setter: None,
            },
        ],
        vec![], // No methods
        vec![], // No constructors
        TypeMetadata::default(),
        test_span(),
    );

    // Create two functions with different struct types
    let func1 = TypedFunction {
        name: arena.intern_string("get_point"),
        params: vec![],
        type_params: vec![],
        return_type: Type::Named {
            id: point_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        },
        body: Some(TypedBlock {
            statements: vec![],
            span: test_span(),
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let func2 = TypedFunction {
        name: arena.intern_string("get_rect"),
        params: vec![],
        type_params: vec![],
        return_type: Type::Named {
            id: rect_id,
            type_args: vec![],
            const_args: vec![],
            variance: vec![],
            nullability: zyntax_typed_ast::NullabilityKind::NonNull,
        },
        body: Some(TypedBlock {
            statements: vec![],
            span: test_span(),
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        annotations: vec![],
        effects: vec![],
        is_pure: false,
    };

    let mut program = TypedProgram {
        declarations: vec![
            typed_node(
                TypedDeclaration::Function(func1),
                Type::Primitive(PrimitiveType::Unit),
                test_span(),
            ),
            typed_node(
                TypedDeclaration::Function(func2),
                Type::Primitive(PrimitiveType::Unit),
                test_span(),
            ),
        ],
        span: test_span(),
        source_files: vec![],
        type_registry: type_registry.clone(),
    };

    // Lower to HIR
    let type_registry = Arc::new(type_registry);
    let config = LoweringConfig::default();
    let module_name = arena.intern_string("test_module");
    let arena_arc = Arc::new(Mutex::new(arena));
    let mut ctx = LoweringContext::new(module_name, type_registry.clone(), arena_arc, config);

    let result = ctx.lower_program(&mut program);
    assert!(
        result.is_ok(),
        "Failed to lower program with multiple struct types: {:?}",
        result.err()
    );

    let module = result.unwrap();
    assert_eq!(module.functions.len(), 2, "Expected 2 functions");

    // Verify both struct types were resolved correctly
    for func in module.functions.values() {
        let return_ty = &func.signature.returns[0];
        match return_ty {
            HirType::Struct(s) => {
                assert!(
                    s.name == Some(point_name) || s.name == Some(rect_name),
                    "Unexpected struct name: {:?}",
                    s.name
                );
                assert_eq!(s.fields.len(), 2, "Expected 2 fields");
                assert!(matches!(s.fields[0], HirType::F32), "Field 0 should be F32");
                assert!(matches!(s.fields[1], HirType::F32), "Field 1 should be F32");
            }
            other => panic!("Expected struct type, got {:?}", other),
        }
    }
}
