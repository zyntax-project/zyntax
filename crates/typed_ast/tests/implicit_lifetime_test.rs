// Test to demonstrate implicit lifetime inference

use zyntax_typed_ast::arena::AstArena;
use zyntax_typed_ast::constraint_solver::ConstraintSolver;
use zyntax_typed_ast::type_registry::{Mutability, PrimitiveType, Type, TypeRegistry};
use zyntax_typed_ast::{AsyncKind, CallingConvention, NullabilityKind};

#[test]
fn test_implicit_lifetime_generation() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));

    // Create a reference type WITHOUT specifying a lifetime
    let ref_type_without_lifetime = Type::Reference {
        ty: Box::new(Type::Primitive(PrimitiveType::I32)),
        mutability: Mutability::Immutable,
        lifetime: None, // No lifetime specified!
        nullability: NullabilityKind::NonNull,
    };

    // Infer lifetimes - this should generate an implicit lifetime
    let ref_type_with_lifetime = solver.infer_lifetimes(&ref_type_without_lifetime, &mut arena);

    // Check that a lifetime was generated
    match &ref_type_with_lifetime {
        Type::Reference {
            lifetime: Some(lt), ..
        } => {
            println!("✅ Implicit lifetime generated: {:?}", lt.name);
            assert!(true, "Lifetime was successfully inferred");
        }
        Type::Reference { lifetime: None, .. } => {
            panic!("❌ No lifetime was generated");
        }
        _ => panic!("❌ Type changed unexpectedly"),
    }
}

#[test]
fn test_nested_reference_lifetime_inference() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));

    // Create nested references: &&i32 (without lifetimes)
    let nested_ref = Type::Reference {
        ty: Box::new(Type::Reference {
            ty: Box::new(Type::Primitive(PrimitiveType::I32)),
            mutability: Mutability::Immutable,
            lifetime: None,
            nullability: NullabilityKind::NonNull,
        }),
        mutability: Mutability::Immutable,
        lifetime: None,
        nullability: NullabilityKind::NonNull,
    };

    // Infer lifetimes
    let inferred = solver.infer_lifetimes(&nested_ref, &mut arena);

    // Check that both levels got lifetimes
    match &inferred {
        Type::Reference {
            ty: inner,
            lifetime: Some(outer_lt),
            ..
        } => {
            println!("✅ Outer lifetime generated: {:?}", outer_lt.name);

            match &**inner {
                Type::Reference {
                    lifetime: Some(inner_lt),
                    ..
                } => {
                    println!("✅ Inner lifetime generated: {:?}", inner_lt.name);
                    // The lifetimes should be different
                    assert_ne!(
                        outer_lt.name, inner_lt.name,
                        "Nested lifetimes should be distinct"
                    );
                }
                _ => panic!("❌ Inner reference missing lifetime"),
            }
        }
        _ => panic!("❌ Outer reference missing lifetime"),
    }
}

#[test]
fn test_function_parameter_lifetime_inference() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));

    // Create a function type with reference parameters (no lifetimes)
    let func_type = Type::Function {
        params: vec![
            zyntax_typed_ast::type_registry::ParamInfo {
                name: Some(arena.intern_string("x")),
                ty: Type::Reference {
                    ty: Box::new(Type::Primitive(PrimitiveType::I32)),
                    mutability: Mutability::Immutable,
                    lifetime: None,
                    nullability: NullabilityKind::NonNull,
                },
                is_optional: false,
                is_varargs: false,
                is_keyword_only: false,
                is_positional_only: false,
                is_out: false,
                is_ref: false,
                is_inout: false,
            },
            zyntax_typed_ast::type_registry::ParamInfo {
                name: Some(arena.intern_string("y")),
                ty: Type::Reference {
                    ty: Box::new(Type::Primitive(PrimitiveType::String)),
                    mutability: Mutability::Mutable,
                    lifetime: None,
                    nullability: NullabilityKind::NonNull,
                },
                is_optional: false,
                is_varargs: false,
                is_keyword_only: false,
                is_positional_only: false,
                is_out: false,
                is_ref: false,
                is_inout: false,
            },
        ],
        return_type: Box::new(Type::Reference {
            ty: Box::new(Type::Primitive(PrimitiveType::Bool)),
            mutability: Mutability::Immutable,
            lifetime: None,
            nullability: NullabilityKind::NonNull,
        }),
        is_varargs: false,
        has_named_params: false,
        has_default_params: false,
        async_kind: AsyncKind::Sync,
        calling_convention: CallingConvention::Default,
        nullability: NullabilityKind::NonNull,
    };

    // Infer lifetimes for the function
    let inferred = solver.infer_lifetimes(&func_type, &mut arena);

    match &inferred {
        Type::Function {
            params,
            return_type,
            ..
        } => {
            // Check first parameter
            match &params[0].ty {
                Type::Reference {
                    lifetime: Some(lt1),
                    ..
                } => {
                    println!("✅ First param lifetime: {:?}", lt1.name);
                }
                _ => panic!("❌ First parameter missing lifetime"),
            }

            // Check second parameter
            match &params[1].ty {
                Type::Reference {
                    lifetime: Some(lt2),
                    ..
                } => {
                    println!("✅ Second param lifetime: {:?}", lt2.name);
                }
                _ => panic!("❌ Second parameter missing lifetime"),
            }

            // Check return type
            match &**return_type {
                Type::Reference {
                    lifetime: Some(lt_ret),
                    ..
                } => {
                    println!("✅ Return type lifetime: {:?}", lt_ret.name);
                }
                _ => panic!("❌ Return type missing lifetime"),
            }

            println!("✅ All function lifetimes were inferred implicitly!");
        }
        _ => panic!("❌ Not a function type"),
    }
}

#[test]
fn test_explicit_lifetime_preserved() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));

    // Create a reference with an explicit lifetime
    let explicit_lifetime =
        zyntax_typed_ast::type_registry::Lifetime::named(arena.intern_string("'static"));

    let ref_with_explicit = Type::Reference {
        ty: Box::new(Type::Primitive(PrimitiveType::I32)),
        mutability: Mutability::Immutable,
        lifetime: Some(explicit_lifetime.clone()),
        nullability: NullabilityKind::NonNull,
    };

    // Infer lifetimes - should preserve the explicit lifetime
    let result = solver.infer_lifetimes(&ref_with_explicit, &mut arena);

    match &result {
        Type::Reference {
            lifetime: Some(lt), ..
        } => {
            assert_eq!(
                lt.name, explicit_lifetime.name,
                "Explicit lifetime should be preserved"
            );
            println!("✅ Explicit lifetime was preserved");
        }
        _ => panic!("❌ Lifetime was lost"),
    }
}
