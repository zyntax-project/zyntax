// Tests for higher-ranked trait bounds (for<'a>)

use std::collections::HashMap;
use zyntax_typed_ast::arena::AstArena;
use zyntax_typed_ast::constraint_solver::{Constraint, ConstraintSolver, SolverError};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{
    ImplDef, Lifetime, MethodImpl, MethodSig, ParamDef, PrimitiveType, TraitDef, Type, TypeBound,
    TypeConstraint, TypeId, TypeParam, TypeRegistry, Visibility,
};
use zyntax_typed_ast::InternedString;
use zyntax_typed_ast::{AsyncKind, CallingConvention, NullabilityKind};

#[test]
fn test_higher_ranked_trait_bound_basic() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(registry));

    // Create a trait that takes a lifetime parameter
    let fn_trait_name = arena.intern_string("Fn");

    // Add constraint: for<'a> F: Fn(&'a T) -> &'a U
    let lifetimes = vec![arena.intern_string("'a")];
    let func_type = Type::Primitive(PrimitiveType::I32); // Placeholder for function type

    let trait_bound = TypeBound::Trait {
        name: fn_trait_name,
        args: vec![], // Would include lifetime references in full implementation
    };

    solver.add_constraint(Constraint::HigherRankedBound {
        lifetimes,
        ty: func_type,
        bound: trait_bound,
        span: Span::new(0, 0),
    });

    let result = solver.solve();

    match &result {
        Ok(_) => println!("✅ Higher-ranked trait bound constraint solved"),
        Err(errors) => {
            println!("❌ Solver errors: {:?}", errors);
            for error in errors {
                println!("  - {:?}", error);
            }
        }
    }

    // The constraint should fail because we don't have Fn trait registered
    assert!(result.is_err());
}

#[test]
fn test_higher_ranked_lifetime_bound() {
    let mut arena = AstArena::new();
    let registry = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(registry));

    // Test: for<'a> T: 'a (T must be valid for any lifetime)
    let lifetimes = vec![arena.intern_string("'a")];
    let ty = Type::Primitive(PrimitiveType::String);

    let lifetime_bound = TypeBound::Lifetime(Lifetime::named(arena.intern_string("'a")));

    solver.add_constraint(Constraint::HigherRankedBound {
        lifetimes,
        ty,
        bound: lifetime_bound,
        span: Span::new(0, 0),
    });

    let result = solver.solve();
    assert!(
        result.is_ok(),
        "Higher-ranked lifetime bound should be solvable"
    );

    println!("✅ Higher-ranked lifetime bound works");
}

#[test]
fn test_nested_higher_ranked_bounds() {
    let mut arena = AstArena::new();
    let registry = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(registry));

    // Test: for<'a> T: for<'b> Trait<'a, 'b>
    let outer_lifetimes = vec![arena.intern_string("'a")];
    let inner_lifetimes = vec![arena.intern_string("'b")];

    let inner_bound = TypeBound::Trait {
        name: arena.intern_string("Trait"),
        args: vec![], // Would include lifetime args
    };

    let outer_bound = TypeBound::HigherRanked {
        lifetimes: inner_lifetimes,
        bound: Box::new(inner_bound),
    };

    solver.add_constraint(Constraint::HigherRankedBound {
        lifetimes: outer_lifetimes,
        ty: Type::Primitive(PrimitiveType::I32),
        bound: outer_bound,
        span: Span::new(0, 0),
    });

    let result = solver.solve();
    println!("Nested higher-ranked bounds result: {:?}", result);
}

#[test]
fn test_higher_ranked_bound_with_type_var() {
    let mut arena = AstArena::new();
    let registry = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(registry));

    // Create a type variable
    let type_var = solver.fresh_type_var();

    // Add higher-ranked bound to type variable
    let lifetimes = vec![arena.intern_string("'a")];
    let trait_bound = TypeBound::Trait {
        name: arena.intern_string("Send"),
        args: vec![],
    };

    solver.add_constraint(Constraint::HigherRankedBound {
        lifetimes,
        ty: type_var,
        bound: trait_bound,
        span: Span::new(0, 0),
    });

    let result = solver.solve();
    assert!(
        result.is_ok(),
        "Higher-ranked bounds on type variables should be deferred"
    );

    println!("✅ Higher-ranked bounds on type variables can be deferred");
}

#[test]
fn test_higher_ranked_fn_trait() {
    let mut arena = AstArena::new();
    let mut registry = TypeRegistry::new();

    // Register a Fn trait
    let fn_trait_id = TypeId::next();
    let fn_trait = TraitDef {
        id: fn_trait_id,
        name: arena.intern_string("Fn"),
        type_params: vec![],
        super_traits: vec![],
        methods: vec![MethodSig {
            name: arena.intern_string("call"),
            type_params: vec![],
            params: vec![
                ParamDef {
                    name: arena.intern_string("self"),
                    ty: Type::Primitive(PrimitiveType::Unit),
                    is_self: true,
                    is_mut: false,
                    is_varargs: false,
                },
                ParamDef {
                    name: arena.intern_string("arg"),
                    ty: Type::TypeVar(zyntax_typed_ast::type_registry::TypeVar::unbound(
                        arena.intern_string("T"),
                    )),
                    is_self: false,
                    is_mut: false,
                    is_varargs: false,
                },
            ],
            return_type: Type::TypeVar(zyntax_typed_ast::type_registry::TypeVar::unbound(
                arena.intern_string("U"),
            )),
            where_clause: vec![],
            is_static: false,
            is_async: false,
            visibility: Visibility::Public,
            span: Span::new(0, 0),
            is_extension: false,
        }],
        associated_types: vec![],
        is_object_safe: true,
        span: Span::new(0, 0),
    };
    registry.register_trait(fn_trait);

    let mut solver = ConstraintSolver::with_type_registry(Box::new(registry));

    // Test: for<'a> F: Fn(&'a T) -> &'a U
    let lifetimes = vec![arena.intern_string("'a")];
    let func_type = Type::Primitive(PrimitiveType::I32); // Placeholder

    let trait_bound = TypeBound::Trait {
        name: arena.intern_string("Fn"),
        args: vec![], // Would include lifetime-parameterized types
    };

    solver.add_constraint(Constraint::HigherRankedBound {
        lifetimes,
        ty: func_type,
        bound: trait_bound,
        span: Span::new(0, 0),
    });

    let result = solver.solve();

    // Should fail because I32 doesn't implement Fn
    assert!(result.is_err(), "I32 should not implement Fn trait");

    println!("✅ Higher-ranked Fn trait bound correctly rejects non-function types");
}

#[test]
fn test_higher_ranked_bound_formatting() {
    let mut arena = AstArena::new();
    let registry = TypeRegistry::new();
    let solver = ConstraintSolver::with_type_registry(Box::new(registry));

    // Create a higher-ranked bound constraint
    let lifetimes = vec![arena.intern_string("'a"), arena.intern_string("'b")];

    let trait_bound = TypeBound::Trait {
        name: arena.intern_string("Trait"),
        args: vec![],
    };

    let constraint = Constraint::HigherRankedBound {
        lifetimes,
        ty: Type::Primitive(PrimitiveType::I32),
        bound: trait_bound,
        span: Span::new(0, 0),
    };

    // Test that the constraint can be formatted
    let _formatted = format!("{:?}", constraint);

    println!("✅ Higher-ranked bound constraint formatting works");
}
