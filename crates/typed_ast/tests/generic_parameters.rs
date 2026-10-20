//! Bound parameters must not acquire a concrete type from inference in their bodies.

use zyntax_typed_ast::constraint_solver::{Constraint, ConstraintSolver};
use zyntax_typed_ast::type_inference::InferenceContext;
use zyntax_typed_ast::{InternedString, PrimitiveType, Span, Type, TypeRegistry, TypeVar};

#[test]
fn inference_variables_can_resolve_to_parameters_without_binding_the_parameter() {
    let parameter = Type::TypeVar(TypeVar::parameter(InternedString::new_global("T")));
    let mut inference = InferenceContext::new(Box::new(TypeRegistry::new()));
    let hole = inference.fresh_type_var();
    assert_eq!(
        inference.unify(parameter.clone(), hole.clone()).unwrap(),
        parameter
    );
    assert_eq!(inference.apply_substitutions(&hole), parameter);
    assert_eq!(
        inference.unify(Type::Never, parameter.clone()).unwrap(),
        parameter
    );
    assert!(
        inference
            .unify(parameter.clone(), Type::Primitive(PrimitiveType::F64))
            .is_err()
    );
    assert_eq!(inference.apply_substitutions(&parameter), parameter);
}

#[test]
fn constraint_solver_keeps_bound_parameters_rigid() {
    for reverse in [false, true] {
        let parameter = Type::TypeVar(TypeVar::parameter(InternedString::new_global("T")));
        let mut solver = ConstraintSolver::new();
        let hole = solver.fresh_type_var();
        let (left, right) = if reverse {
            (hole.clone(), parameter.clone())
        } else {
            (parameter.clone(), hole.clone())
        };
        solver.add_constraint(Constraint::Equal(left, right, Span::default()));
        let substitution = solver.solve().unwrap();
        assert_eq!(substitution.apply(&hole), parameter);
        solver.add_constraint(Constraint::Equal(
            parameter.clone(),
            Type::Primitive(PrimitiveType::F64),
            Span::default(),
        ));
        assert!(solver.solve().is_err());
    }
}
