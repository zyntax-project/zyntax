//! Regression coverage for dominance-safe nested enum pattern lowering.

use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};
use zyntax_typed_ast::{
    Mutability, PrimitiveType, Span, Type, TypedASTBuilder, TypedDeclaration, TypedExpression,
    TypedLiteral, TypedMatchArm, TypedMatchExpr, TypedNode, TypedPattern, Visibility,
};

fn result_type() -> Type {
    Type::Result {
        ok_type: Box::new(Type::Optional(Box::new(Type::Primitive(
            PrimitiveType::I64,
        )))),
        err_type: Box::new(Type::Primitive(PrimitiveType::String)),
    }
}

#[derive(Clone, Copy)]
enum PatternShape {
    Enum,
    Constructor,
}

fn variant_pattern(
    builder: &mut TypedASTBuilder,
    enum_name: &str,
    variant: &str,
    mut fields: Vec<TypedNode<TypedPattern>>,
    shape: PatternShape,
    span: Span,
) -> TypedNode<TypedPattern> {
    match shape {
        PatternShape::Enum => builder.enum_pattern(enum_name, variant, fields, span),
        PatternShape::Constructor => {
            assert!(fields.len() <= 1, "Constructor stores one inner pattern");
            let inner = fields
                .pop()
                .unwrap_or_else(|| TypedNode::new(TypedPattern::Wildcard, Type::Any, span));
            TypedNode::new(
                TypedPattern::Constructor {
                    constructor: Type::Unresolved(builder.intern(variant)),
                    pattern: Box::new(inner),
                },
                Type::Never,
                span,
            )
        }
    }
}

fn observer_function(
    builder: &mut TypedASTBuilder,
    shape: PatternShape,
    span: Span,
) -> TypedNode<TypedDeclaration> {
    let i64_type = Type::Primitive(PrimitiveType::I64);
    let parameter = builder.parameter("outcome", result_type(), Mutability::Immutable, span);
    let scrutinee = builder.variable("outcome", result_type(), span);

    let value_pattern = TypedNode::new(
        TypedPattern::Identifier {
            name: builder.intern("value"),
            mutability: Mutability::Immutable,
        },
        i64_type.clone(),
        span,
    );
    let some_pattern = variant_pattern(builder, "Option", "Some", vec![value_pattern], shape, span);
    let ok_some = variant_pattern(builder, "Result", "Ok", vec![some_pattern], shape, span);
    let some_body = builder.variable("value", i64_type.clone(), span);

    let none_pattern = variant_pattern(builder, "Option", "None", vec![], shape, span);
    let ok_none = variant_pattern(builder, "Result", "Ok", vec![none_pattern], shape, span);
    let none_body = TypedNode::new(
        TypedExpression::Literal(TypedLiteral::Integer(-1)),
        i64_type.clone(),
        span,
    );
    let err_field = TypedNode::new(TypedPattern::Wildcard, Type::Never, span);
    let err_pattern = variant_pattern(builder, "Result", "Err", vec![err_field], shape, span);
    let err_body = TypedNode::new(
        TypedExpression::Literal(TypedLiteral::Integer(-2)),
        i64_type.clone(),
        span,
    );
    let matched = TypedNode::new(
        TypedExpression::Match(TypedMatchExpr {
            scrutinee: Box::new(scrutinee),
            arms: vec![
                TypedMatchArm {
                    pattern: Box::new(ok_some),
                    guard: None,
                    body: Box::new(some_body),
                },
                TypedMatchArm {
                    pattern: Box::new(ok_none),
                    guard: None,
                    body: Box::new(none_body),
                },
                TypedMatchArm {
                    pattern: Box::new(err_pattern),
                    guard: None,
                    body: Box::new(err_body),
                },
            ],
        }),
        i64_type.clone(),
        span,
    );
    let return_statement = builder.return_stmt(matched, span);
    let body = builder.block(vec![return_statement], span);
    builder.function(
        "inspect_result",
        vec![parameter],
        i64_type,
        body,
        Visibility::Private,
        false,
        span,
    )
}

fn entry_function(
    builder: &mut TypedASTBuilder,
    name: &str,
    outer_variant: &str,
    payload: TypedNode<TypedExpression>,
    span: Span,
) -> TypedNode<TypedDeclaration> {
    let i64_type = Type::Primitive(PrimitiveType::I64);
    let constructor = builder.variable(outer_variant, result_type(), span);
    let outcome = builder.call_positional(constructor, vec![payload], result_type(), span);
    let inspect = builder.variable("inspect_result", Type::Any, span);
    let result = builder.call_positional(inspect, vec![outcome], i64_type.clone(), span);
    let return_statement = builder.return_stmt(result, span);
    let body = builder.block(vec![return_statement], span);
    builder.function(
        name,
        vec![],
        i64_type,
        body,
        Visibility::Public,
        false,
        span,
    )
}

fn assert_nested_patterns(shape: PatternShape) {
    let mut builder = TypedASTBuilder::new();
    let span = Span::new(0, 0);
    let i64_type = Type::Primitive(PrimitiveType::I64);
    let optional_type = Type::Optional(Box::new(i64_type.clone()));

    let some_constructor = builder.variable("Some", optional_type.clone(), span);
    let some_value = TypedNode::new(
        TypedExpression::Literal(TypedLiteral::Integer(73)),
        i64_type,
        span,
    );
    let some_payload = builder.call_positional(
        some_constructor,
        vec![some_value],
        optional_type.clone(),
        span,
    );
    let none_payload = builder.variable("None", optional_type, span);
    let err_payload = TypedNode::new(
        TypedExpression::Literal(TypedLiteral::String(builder.intern("nested-pattern-error"))),
        Type::Primitive(PrimitiveType::String),
        span,
    );

    let observer = observer_function(&mut builder, shape, span);
    let ok_some = entry_function(&mut builder, "ok_some", "Ok", some_payload, span);
    let ok_none = entry_function(&mut builder, "ok_none", "Ok", none_payload, span);
    let err = entry_function(&mut builder, "err", "Err", err_payload, span);
    let program = builder.program(vec![observer, ok_some, ok_none, err], span);

    let mut runtime = ZyntaxRuntime::new().expect("runtime");
    runtime
        .compile_typed_program(program)
        .expect("nested enum pattern program should compile");
    for (name, expected) in [("ok_some", 73), ("ok_none", -1), ("err", -2)] {
        assert_eq!(
            runtime.call_raw(name, &[]).expect("entry should execute"),
            ZyntaxValue::Int(expected),
            "{name} selected the wrong nested pattern arm"
        );
    }
}

#[test]
fn inactive_outer_variant_does_not_evaluate_nested_payload_pattern() {
    assert_nested_patterns(PatternShape::Enum);
    assert_nested_patterns(PatternShape::Constructor);
}

#[test]
fn extracted_bool_payload_zero_extends_to_integer() {
    let mut builder = TypedASTBuilder::new();
    let span = Span::new(0, 0);
    let bool_type = Type::Primitive(PrimitiveType::Bool);
    let i64_type = Type::Primitive(PrimitiveType::I64);
    let bool_result_type = Type::Result {
        ok_type: Box::new(bool_type.clone()),
        err_type: Box::new(Type::Primitive(PrimitiveType::String)),
    };
    let parameter = builder.parameter(
        "outcome",
        bool_result_type.clone(),
        Mutability::Immutable,
        span,
    );
    let scrutinee = builder.variable("outcome", bool_result_type.clone(), span);
    let value_pattern = TypedNode::new(
        TypedPattern::Identifier {
            name: builder.intern("value"),
            mutability: Mutability::Immutable,
        },
        bool_type.clone(),
        span,
    );
    let ok_pattern = builder.enum_pattern("Result", "Ok", vec![value_pattern], span);
    let value = builder.variable("value", bool_type, span);
    let ok_body = builder.cast(value, i64_type.clone(), span);
    let err_field = TypedNode::new(TypedPattern::Wildcard, Type::Never, span);
    let err_pattern = builder.enum_pattern("Result", "Err", vec![err_field], span);
    let err_body = TypedNode::new(
        TypedExpression::Literal(TypedLiteral::Integer(-1)),
        i64_type.clone(),
        span,
    );
    let matched = TypedNode::new(
        TypedExpression::Match(TypedMatchExpr {
            scrutinee: Box::new(scrutinee),
            arms: vec![
                TypedMatchArm {
                    pattern: Box::new(ok_pattern),
                    guard: None,
                    body: Box::new(ok_body),
                },
                TypedMatchArm {
                    pattern: Box::new(err_pattern),
                    guard: None,
                    body: Box::new(err_body),
                },
            ],
        }),
        i64_type.clone(),
        span,
    );
    let return_statement = builder.return_stmt(matched, span);
    let body = builder.block(vec![return_statement], span);
    let observer = builder.function(
        "cast_bool_payload",
        vec![parameter],
        i64_type.clone(),
        body,
        Visibility::Private,
        false,
        span,
    );
    let truth = TypedNode::new(
        TypedExpression::Literal(TypedLiteral::Bool(true)),
        Type::Primitive(PrimitiveType::Bool),
        span,
    );
    let constructor = builder.variable("Ok", bool_result_type.clone(), span);
    let outcome = builder.call_positional(constructor, vec![truth], bool_result_type, span);
    let inspect = builder.variable("cast_bool_payload", Type::Any, span);
    let result = builder.call_positional(inspect, vec![outcome], i64_type.clone(), span);
    let return_statement = builder.return_stmt(result, span);
    let body = builder.block(vec![return_statement], span);
    let entry = builder.function(
        "cast_bool_entry",
        vec![],
        i64_type,
        body,
        Visibility::Public,
        false,
        span,
    );
    let program = builder.program(vec![observer, entry], span);

    let mut runtime = ZyntaxRuntime::new().expect("runtime");
    runtime
        .compile_typed_program(program)
        .expect("bool cast program should compile");
    assert_eq!(
        runtime
            .call_raw("cast_bool_entry", &[])
            .expect("cast entry should execute"),
        ZyntaxValue::Int(1)
    );
}
