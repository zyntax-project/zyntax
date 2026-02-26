// Tests for lifetime inference and checking

use string_interner::Symbol;
use zyntax_typed_ast::arena::AstArena;
use zyntax_typed_ast::constraint_solver::{Constraint, ConstraintSolver, SolverError};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{Lifetime, Mutability, PrimitiveType, Type, TypeRegistry};
use zyntax_typed_ast::{AsyncKind, CallingConvention, NullabilityKind};

#[test]
fn test_lifetime_basics() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));

    // Create a lifetime
    let lifetime_a = Lifetime::named(arena.intern_string("'a"));
    let lifetime_b = Lifetime::named(arena.intern_string("'b"));

    // Add constraint: 'a: 'b ('a outlives 'b)
    solver.add_constraint(Constraint::LifetimeOutlives(
        lifetime_a.clone(),
        lifetime_b.clone(),
        Span::new(0, 0),
    ));

    let result = solver.solve();

    match &result {
        Ok(_) => println!("✅ Basic lifetime constraint solved successfully"),
        Err(errors) => {
            println!("❌ Solver errors: {:?}", errors);
            for error in errors {
                println!("  - {:?}", error);
            }
        }
    }

    assert!(
        result.is_ok(),
        "Lifetime outlives constraint should be solvable"
    );
}

#[test]
fn test_lifetime_cycle_detection() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));
    let span = Span::new(0, 0);

    // Create two lifetimes
    let lifetime_a = Lifetime::named(arena.intern_string("'a"));
    let lifetime_b = Lifetime::named(arena.intern_string("'b"));

    // Add first constraint: 'a: 'b
    solver.add_constraint(Constraint::LifetimeOutlives(
        lifetime_a.clone(),
        lifetime_b.clone(),
        span,
    ));

    // Solve should succeed with just one constraint
    let result1 = solver.solve();
    assert!(result1.is_ok(), "First constraint should be solvable");

    // Add conflicting constraint: 'b: 'a (this creates a cycle)
    solver.add_constraint(Constraint::LifetimeOutlives(
        lifetime_b.clone(),
        lifetime_a.clone(),
        span,
    ));

    // Now solve should fail due to the cycle
    let result2 = solver.solve();

    match &result2 {
        Ok(_) => println!("❌ Solver succeeded when it should have detected a cycle"),
        Err(errors) => {
            println!("✅ Solver detected errors: {:?}", errors);
            for error in errors {
                println!("  - {:?}", error);
            }
        }
    }

    assert!(result2.is_err(), "Lifetime cycle should be detected");

    if let Err(errors) = result2 {
        assert!(errors
            .iter()
            .any(|e| matches!(e, SolverError::LifetimeCycle { .. })));
        println!("✅ Lifetime cycle correctly detected");
    }
}

#[test]
fn test_type_outlives_lifetime() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));
    let span = Span::new(0, 0);

    // Create a lifetime
    let lifetime_a = Lifetime::named(arena.intern_string("'a"));

    // Create a reference type
    let ref_type = Type::Reference {
        ty: Box::new(Type::Primitive(PrimitiveType::I32)),
        mutability: Mutability::Immutable,
        lifetime: Some(lifetime_a.clone()),
        nullability: NullabilityKind::NonNull,
    };

    // Add constraint: &'a i32: 'a
    solver.add_constraint(Constraint::TypeOutlivesLifetime(ref_type, lifetime_a, span));

    let result = solver.solve();
    assert!(
        result.is_ok(),
        "Type outlives lifetime constraint should be solvable"
    );

    println!("✅ Type outlives lifetime constraint solved successfully");
}

#[test]
fn test_lifetime_constraint_formatting() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let solver = ConstraintSolver::with_type_registry(Box::new(env));

    // Create two lifetimes
    let lifetime_a = Lifetime::named(arena.intern_string("'a"));
    let lifetime_b = Lifetime::named(arena.intern_string("'b"));

    // Create constraints
    let outlives_constraint =
        Constraint::LifetimeOutlives(lifetime_a.clone(), lifetime_b.clone(), Span::new(0, 0));

    let type_outlives_constraint = Constraint::TypeOutlivesLifetime(
        Type::Primitive(PrimitiveType::I32),
        lifetime_a.clone(),
        Span::new(0, 0),
    );

    // Test that constraints can be formatted (this exercises the match arms)
    let _outlives_str = format!("{:?}", outlives_constraint);
    let _type_outlives_str = format!("{:?}", type_outlives_constraint);

    println!("✅ Lifetime constraint formatting works");
}

#[test]
fn test_lifetime_inference_api() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));

    // Test that we can generate fresh lifetimes
    let lifetime1 = solver.fresh_lifetime(&mut arena);
    let lifetime2 = solver.fresh_lifetime(&mut arena);

    // Lifetimes should be distinct
    assert_ne!(lifetime1.name, lifetime2.name);

    println!("✅ Fresh lifetime generation works");
}

#[test]
fn test_check_lifetime_constraints() {
    let mut arena = AstArena::new();
    let env = TypeRegistry::new();
    let mut solver = ConstraintSolver::with_type_registry(Box::new(env));
    let span = Span::new(0, 0);

    // Create three lifetimes
    let lifetime_a = Lifetime::named(arena.intern_string("'a"));
    let lifetime_b = Lifetime::named(arena.intern_string("'b"));
    let lifetime_c = Lifetime::named(arena.intern_string("'c"));

    // Add transitive constraints: 'a: 'b and 'b: 'c
    solver.add_constraint(Constraint::LifetimeOutlives(
        lifetime_a.clone(),
        lifetime_b.clone(),
        span,
    ));
    solver.add_constraint(Constraint::LifetimeOutlives(
        lifetime_b.clone(),
        lifetime_c.clone(),
        span,
    ));

    let result = solver.solve();
    assert!(
        result.is_ok(),
        "Transitive lifetime constraints should be solvable"
    );

    // Check lifetime constraints are satisfied
    let check_result = solver.check_lifetime_constraints();
    assert!(
        check_result.is_ok(),
        "Lifetime constraints should be satisfied"
    );

    println!("✅ Transitive lifetime constraints checked successfully");
}
