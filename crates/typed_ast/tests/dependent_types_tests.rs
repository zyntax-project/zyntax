//! Tests for the dependent type system
//!
//! This module tests value-dependent types, refinement types, and dependent functions.

use string_interner::Symbol;
use zyntax_typed_ast::arena::InternedString;
use zyntax_typed_ast::dependent_types::*;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{ConstValue, Type, TypeVar, TypeVarId, TypeVarKind};
use zyntax_typed_ast::{NullabilityKind, PrimitiveType};

#[test]
fn test_refinement_type_well_formedness() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(1).unwrap());

    // Test positive integer refinement: {x: i32 | x > 0}
    let positive_int = DependentType::refinement(
        Type::Primitive(PrimitiveType::I32),
        var_name,
        RefinementPredicate::Comparison {
            op: ComparisonOp::Greater,
            left: Box::new(RefinementExpr::Variable(var_name)),
            right: Box::new(RefinementExpr::Constant(ConstValue::Int(0))),
        },
        span,
    );

    // Check that the type is well-formed
    let result = checker.check_well_formed(&positive_int);
    assert!(
        result.is_ok(),
        "Positive integer refinement should be well-formed"
    );
}

#[test]
fn test_refinement_with_unbound_variable() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(2).unwrap());
    let unbound_var =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(3).unwrap());

    // Test refinement that references an unbound variable
    let bad_refinement = DependentType::refinement(
        Type::Primitive(PrimitiveType::I32),
        var_name,
        RefinementPredicate::Comparison {
            op: ComparisonOp::Greater,
            left: Box::new(RefinementExpr::Variable(var_name)),
            right: Box::new(RefinementExpr::Variable(unbound_var)), // This variable is not bound
        },
        span,
    );

    let result = checker.check_well_formed(&bad_refinement);
    match result {
        Err(DependentTypeError::UnboundVariable { var, .. }) => {
            assert_eq!(
                var, unbound_var,
                "Should detect the correct unbound variable"
            );
        }
        _ => panic!("Expected UnboundVariable error"),
    }
}

#[test]
fn test_dependent_function_scope() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let param_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(4).unwrap());

    // Test dependent function: (n: i32) -> {x: i32 | x > n}
    let dep_function = DependentType::DependentFunction {
        param_name,
        param_type: Box::new(Type::Primitive(PrimitiveType::I32)),
        return_type: Box::new(DependentType::refinement(
            Type::Primitive(PrimitiveType::I32),
            InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(5).unwrap()),
            RefinementPredicate::Comparison {
                op: ComparisonOp::Greater,
                left: Box::new(RefinementExpr::Variable(InternedString::from_symbol(
                    string_interner::DefaultSymbol::try_from_usize(5).unwrap(),
                ))),
                right: Box::new(RefinementExpr::Variable(param_name)), // References the parameter
            },
            span,
        )),
        span,
    };

    // This should be well-formed because param_name is in scope in the return type
    let result = checker.check_well_formed(&dep_function);
    assert!(
        result.is_ok(),
        "Dependent function should be well-formed with parameter in scope"
    );
}

#[test]
fn test_dependent_pair_scoping() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let first_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(6).unwrap());
    let second_var =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(7).unwrap());

    // Test dependent pair: (x: i32, {y: i32 | y > x})
    let dep_pair = DependentType::DependentPair {
        first_name,
        first_type: Box::new(Type::Primitive(PrimitiveType::I32)),
        second_type: Box::new(DependentType::refinement(
            Type::Primitive(PrimitiveType::I32),
            second_var,
            RefinementPredicate::Comparison {
                op: ComparisonOp::Greater,
                left: Box::new(RefinementExpr::Variable(second_var)),
                right: Box::new(RefinementExpr::Variable(first_name)), // References first component
            },
            span,
        )),
        span,
    };

    // First component should be in scope for second component
    let result = checker.check_well_formed(&dep_pair);
    assert!(
        result.is_ok(),
        "Dependent pair should allow second to reference first"
    );
}

#[test]
fn test_singleton_type_validation() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);

    // Test singleton type for the value 42
    let singleton_42 = DependentType::singleton(
        ConstValue::Int(42),
        Type::Primitive(PrimitiveType::I32),
        span,
    );

    let result = checker.check_well_formed(&singleton_42);
    assert!(result.is_ok(), "Singleton type should be well-formed");
}

#[test]
fn test_path_dependent_type_resolution() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let module_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(8).unwrap());
    let type_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(9).unwrap());

    // Test path-dependent type: module.Type
    let path_dependent = DependentType::PathDependent {
        path: TypePath::Variable(module_name),
        type_name,
        span,
    };

    // Should pass basic well-formedness even if path isn't resolved yet
    let result = checker.check_well_formed(&path_dependent);
    assert!(
        result.is_ok(),
        "Path-dependent type should pass basic well-formedness"
    );
}

#[test]
fn test_conditional_type_branches() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);

    // Test conditional type with constant condition
    let conditional = DependentType::Conditional {
        condition: RefinementPredicate::Constant(true),
        then_type: Box::new(DependentType::Base(Type::Primitive(PrimitiveType::U32))),
        else_type: Box::new(DependentType::Base(Type::Primitive(PrimitiveType::I32))),
        span,
    };

    let result = checker.check_well_formed(&conditional);
    assert!(result.is_ok(), "Conditional type should be well-formed");
}

#[test]
fn test_recursive_type_detection() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(10).unwrap());

    // Test recursive type: μ(X: Type). List<X>
    let recursive_type = DependentType::Recursive {
        var_name,
        kind: Kind::Type,
        body: Box::new(DependentType::Base(Type::TypeVar(TypeVar {
            id: TypeVarId::next(),
            name: Some(var_name),
            kind: TypeVarKind::Type,
        }))),
        span,
    };

    let result = checker.check_well_formed(&recursive_type);
    // Should be ok for simple recursive reference
    assert!(
        result.is_ok(),
        "Simple recursive type should be well-formed"
    );
}

#[test]
fn test_universal_quantification() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(11).unwrap());

    // Test universal type: ∀(x: T). U(x)
    let universal = DependentType::Universal {
        var_name,
        var_type: Box::new(Type::Primitive(PrimitiveType::I32)),
        body: Box::new(DependentType::refinement(
            Type::Primitive(PrimitiveType::Bool),
            InternedString::from_symbol(
                string_interner::DefaultSymbol::try_from_usize(12).unwrap(),
            ),
            RefinementPredicate::Comparison {
                op: ComparisonOp::Greater,
                left: Box::new(RefinementExpr::Variable(var_name)),
                right: Box::new(RefinementExpr::Constant(ConstValue::Int(0))),
            },
            span,
        )),
        span,
    };

    let result = checker.check_well_formed(&universal);
    assert!(result.is_ok(), "Universal type should be well-formed");
}

#[test]
fn test_existential_quantification() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(13).unwrap());

    // Test existential type: ∃(x: T). U(x)
    let existential = DependentType::Existential {
        var_name,
        var_type: Box::new(Type::Primitive(PrimitiveType::I32)),
        body: Box::new(DependentType::Base(Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: Some(ConstValue::Variable(var_name)),
            nullability: NullabilityKind::NonNull,
        })),
        span,
    };

    let result = checker.check_well_formed(&existential);
    assert!(result.is_ok(), "Existential type should be well-formed");
}

#[test]
fn test_indexed_family_application() {
    let mut checker = DependentTypeChecker::new();
    let span = Span::new(0, 10);
    let family_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(14).unwrap());

    // Test indexed type application: Vec[10] (without pre-registering the family)
    let indexed = DependentType::IndexedFamily {
        family_name,
        indices: vec![DependentIndex::Value(ConstValue::Int(10))],
        span,
    };

    // Should fail because family isn't registered
    let result = checker.check_well_formed(&indexed);
    match result {
        Err(DependentTypeError::InvalidTypeFamily { family, .. }) => {
            assert_eq!(family, family_name, "Should report correct missing family");
        }
        _ => panic!("Expected InvalidTypeFamily error for unregistered family"),
    }
}

#[test]
fn test_complex_refinement_predicates() {
    let var_name =
        InternedString::from_symbol(string_interner::DefaultSymbol::try_from_usize(16).unwrap());

    // Test complex predicate: x > 0 AND x < 100 AND x % 2 == 0
    let positive = RefinementPredicate::Comparison {
        op: ComparisonOp::Greater,
        left: Box::new(RefinementExpr::Variable(var_name)),
        right: Box::new(RefinementExpr::Constant(ConstValue::Int(0))),
    };

    let less_than_100 = RefinementPredicate::Comparison {
        op: ComparisonOp::Less,
        left: Box::new(RefinementExpr::Variable(var_name)),
        right: Box::new(RefinementExpr::Constant(ConstValue::Int(100))),
    };

    let even = RefinementPredicate::Comparison {
        op: ComparisonOp::Equal,
        left: Box::new(RefinementExpr::Binary {
            op: ArithmeticOp::Mod,
            left: Box::new(RefinementExpr::Variable(var_name)),
            right: Box::new(RefinementExpr::Constant(ConstValue::Int(2))),
        }),
        right: Box::new(RefinementExpr::Constant(ConstValue::Int(0))),
    };

    let complex_predicate = RefinementPredicate::And(
        Box::new(RefinementPredicate::And(
            Box::new(positive),
            Box::new(less_than_100),
        )),
        Box::new(even),
    );

    // Test that complex predicate is properly structured
    match &complex_predicate {
        RefinementPredicate::And(left, right) => {
            match left.as_ref() {
                RefinementPredicate::And(_, _) => {}
                _ => panic!("Expected nested AND"),
            }
            match right.as_ref() {
                RefinementPredicate::Comparison { .. } => {}
                _ => panic!("Expected comparison for even check"),
            }
        }
        _ => panic!("Expected top-level AND"),
    }
}
