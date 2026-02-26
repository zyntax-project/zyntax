//! Tests for the const evaluator
//!
//! This module tests compile-time constant evaluation and constraint solving.

use zyntax_typed_ast::const_evaluator::*;
use zyntax_typed_ast::type_registry::{ConstBinaryOp, ConstValue, Type};
use zyntax_typed_ast::PrimitiveType;

#[test]
fn test_eval_simple_literals() {
    let mut evaluator = ConstEvaluator::new();

    // Test integer evaluation
    let int_val = ConstValue::int(42);
    let result = evaluator.eval_const_value(&int_val).unwrap();
    match result {
        ConstValue::Int(42) => {}
        _ => panic!("Expected Int(42), got {:?}", result),
    }

    // Test boolean evaluation
    let bool_val = ConstValue::bool(true);
    let result = evaluator.eval_const_value(&bool_val).unwrap();
    match result {
        ConstValue::Bool(true) => {}
        _ => panic!("Expected Bool(true), got {:?}", result),
    }

    // Test character evaluation
    let char_val = ConstValue::char('x');
    let result = evaluator.eval_const_value(&char_val).unwrap();
    match result {
        ConstValue::Char('x') => {}
        _ => panic!("Expected Char('x'), got {:?}", result),
    }
}

#[test]
fn test_eval_binary_operations() {
    let mut evaluator = ConstEvaluator::new();

    // Test integer addition
    let left = ConstValue::int(10);
    let right = ConstValue::int(5);
    let add_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Add,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&add_expr).unwrap();
    match result {
        ConstValue::Int(15) => {}
        _ => panic!("Expected Int(15), got {:?}", result),
    }

    // Test integer subtraction
    let left = ConstValue::int(10);
    let right = ConstValue::int(3);
    let sub_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Sub,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&sub_expr).unwrap();
    match result {
        ConstValue::Int(7) => {}
        _ => panic!("Expected Int(7), got {:?}", result),
    }

    // Test integer multiplication
    let left = ConstValue::int(6);
    let right = ConstValue::int(7);
    let mul_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Mul,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&mul_expr).unwrap();
    match result {
        ConstValue::Int(42) => {}
        _ => panic!("Expected Int(42), got {:?}", result),
    }
}

#[test]
fn test_eval_comparison_operations() {
    let mut evaluator = ConstEvaluator::new();

    // Test equality
    let left = ConstValue::int(5);
    let right = ConstValue::int(5);
    let eq_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Eq,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&eq_expr).unwrap();
    match result {
        ConstValue::Bool(true) => {}
        _ => panic!("Expected Bool(true), got {:?}", result),
    }

    // Test less than
    let left = ConstValue::int(3);
    let right = ConstValue::int(7);
    let lt_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Lt,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&lt_expr).unwrap();
    match result {
        ConstValue::Bool(true) => {}
        _ => panic!("Expected Bool(true), got {:?}", result),
    }

    // Test greater than (should be false)
    let left = ConstValue::int(3);
    let right = ConstValue::int(7);
    let gt_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Gt,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&gt_expr).unwrap();
    match result {
        ConstValue::Bool(false) => {}
        _ => panic!("Expected Bool(false), got {:?}", result),
    }
}

#[test]
fn test_eval_division_by_zero() {
    let mut evaluator = ConstEvaluator::new();

    let left = ConstValue::int(10);
    let right = ConstValue::int(0);
    let div_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Div,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&div_expr);
    match result {
        Err(ConstEvalError::DivisionByZero) => {}
        _ => panic!("Expected DivisionByZero error, got {:?}", result),
    }
}

#[test]
fn test_eval_type_mismatch_errors() {
    let mut evaluator = ConstEvaluator::new();

    // Test type mismatch - trying to add int and bool should fail
    let left = ConstValue::int(5);
    let right = ConstValue::bool(true);
    let add_expr = ConstValue::BinaryOp {
        op: ConstBinaryOp::Add,
        left: Box::new(left),
        right: Box::new(right),
    };

    let result = evaluator.eval_const_value(&add_expr);
    match result {
        Err(ConstEvalError::TypeMismatch { .. }) => {}
        _ => panic!("Expected TypeMismatch error, got {:?}", result),
    }
}

#[test]
fn test_eval_nested_expressions() {
    let mut evaluator = ConstEvaluator::new();

    // Test (5 + 3) * 2 = 16
    let inner_add = ConstValue::BinaryOp {
        op: ConstBinaryOp::Add,
        left: Box::new(ConstValue::int(5)),
        right: Box::new(ConstValue::int(3)),
    };

    let outer_mul = ConstValue::BinaryOp {
        op: ConstBinaryOp::Mul,
        left: Box::new(inner_add),
        right: Box::new(ConstValue::int(2)),
    };

    let result = evaluator.eval_const_value(&outer_mul).unwrap();
    match result {
        ConstValue::Int(16) => {}
        _ => panic!("Expected Int(16), got {:?}", result),
    }
}

#[test]
fn test_eval_literal_from_const_expr() {
    let mut evaluator = ConstEvaluator::new();

    let literal_expr = ConstExpr {
        kind: ConstExprKind::Literal(Literal::Integer(99)),
        ty: Type::Primitive(PrimitiveType::I64),
        span: zyntax_typed_ast::source::Span::new(0, 1),
    };

    let result = evaluator.eval_const_expr(&literal_expr).unwrap();
    match result {
        ConstValue::Int(99) => {}
        _ => panic!("Expected Int(99), got {:?}", result),
    }
}
