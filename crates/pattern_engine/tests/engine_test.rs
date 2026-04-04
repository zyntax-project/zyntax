use pattern_engine::*;
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::TypedASTBuilder;

fn make_span() -> Span {
    Span::new(0, 0)
}

fn make_program_with_literal(value: i128) -> TypedProgram {
    let span = make_span();

    // Build: def main() { println(42) }
    let literal = TypedNode::new(
        TypedExpression::Literal(TypedLiteral::Integer(value)),
        Type::Primitive(PrimitiveType::I64),
        span,
    );

    let println_name = zyntax_typed_ast::InternedString::new_global("println");
    let callee = TypedNode::new(TypedExpression::Variable(println_name), Type::Any, span);

    let call = TypedNode::new(
        TypedExpression::Call(TypedCall {
            callee: Box::new(callee),
            positional_args: vec![literal],
            named_args: vec![],
            type_args: vec![],
        }),
        Type::Primitive(PrimitiveType::Unit),
        span,
    );

    let stmt = TypedNode::new(
        TypedStatement::Expression(Box::new(call)),
        Type::Primitive(PrimitiveType::Unit),
        span,
    );

    let main_name = zyntax_typed_ast::InternedString::new_global("main");
    let func = TypedFunction {
        name: main_name,
        type_params: vec![],
        params: vec![],
        return_type: Type::Primitive(PrimitiveType::Unit),
        body: Some(TypedBlock {
            statements: vec![stmt],
            span,
        }),
        visibility: zyntax_typed_ast::type_registry::Visibility::Public,
        is_async: false,
        is_pure: false,
        effects: vec![],
        ..Default::default()
    };

    let decl = TypedNode::new(
        TypedDeclaration::Function(func),
        Type::Primitive(PrimitiveType::Unit),
        span,
    );

    TypedProgram {
        declarations: vec![decl],
        span,
        source_files: vec![],
        type_registry: zyntax_typed_ast::TypeRegistry::new(),
    }
}

/// Find the first integer literal in the program
fn find_int_literal(program: &TypedProgram) -> Option<i128> {
    for decl in &program.declarations {
        if let TypedDeclaration::Function(func) = &decl.node {
            if let Some(body) = &func.body {
                for stmt in &body.statements {
                    if let TypedStatement::Expression(expr) = &stmt.node {
                        if let TypedExpression::Call(call) = &expr.node {
                            if let Some(arg) = call.positional_args.first() {
                                if let TypedExpression::Literal(TypedLiteral::Integer(v)) =
                                    &arg.node
                                {
                                    return Some(*v);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[test]
fn test_no_rewrites_returns_unchanged() {
    let mut program = make_program_with_literal(42);
    let registry = zyntax_typed_ast::TypeRegistry::new();

    let mut engine = PatternEngine::new(EngineConfig::default());
    engine.finalize().unwrap();

    let result = engine.run(&mut program, &registry);

    assert!(!result.changed);
    assert_eq!(result.rewrites_fired.len(), 0);
    assert_eq!(find_int_literal(&program), Some(42));
}

#[test]
fn test_literal_rewrite_fires() {
    let mut program = make_program_with_literal(42);
    let registry = zyntax_typed_ast::TypeRegistry::new();

    let mut engine = PatternEngine::new(EngineConfig::default());

    // Register a rewrite: IntLiteral(42) → IntLiteral(0)
    let pattern = Pattern::<TypedExpression>::new("match_42", |node, _ctx| {
        if let TypedExpression::Literal(TypedLiteral::Integer(42)) = &node.node {
            Some(Bindings::new())
        } else {
            None
        }
    });

    let rewrite = ExprRewrite::new(
        "replace_42_with_0",
        Priority::NORMALIZATION,
        pattern,
        |_bindings, _builder| {
            let replacement = TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(0)),
                Type::Primitive(PrimitiveType::I64),
                Span::new(0, 0),
            );
            RewriteOutput::ReplaceExpr(replacement)
        },
    );

    engine.register_expr_rewrite(rewrite);
    engine.finalize().unwrap();

    let result = engine.run(&mut program, &registry);

    assert!(result.changed);
    assert_eq!(result.rewrites_fired.len(), 1);
    assert_eq!(result.rewrites_fired[0].rewrite_name, "replace_42_with_0");
    // The literal should now be 0
    assert_eq!(find_int_literal(&program), Some(0));
}

#[test]
fn test_rewrite_is_idempotent() {
    let mut program = make_program_with_literal(42);
    let registry = zyntax_typed_ast::TypeRegistry::new();

    let pattern = Pattern::<TypedExpression>::new("match_42", |node, _ctx| {
        if let TypedExpression::Literal(TypedLiteral::Integer(42)) = &node.node {
            Some(Bindings::new())
        } else {
            None
        }
    });

    let rewrite = ExprRewrite::new(
        "replace_42_with_0",
        Priority::NORMALIZATION,
        pattern,
        |_bindings, _builder| {
            RewriteOutput::ReplaceExpr(TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(0)),
                Type::Primitive(PrimitiveType::I64),
                Span::new(0, 0),
            ))
        },
    );

    let mut engine = PatternEngine::new(EngineConfig::default());
    engine.register_expr_rewrite(rewrite);
    engine.finalize().unwrap();

    // First run
    let result1 = engine.run(&mut program, &registry);
    assert!(result1.changed);
    assert_eq!(find_int_literal(&program), Some(0));

    // Second run — should be a no-op (fixpoint)
    let result2 = engine.run(&mut program, &registry);
    assert!(!result2.changed);
    assert_eq!(result2.rewrites_fired.len(), 0);
}

#[test]
fn test_fixpoint_chain() {
    // Rewrite chain: 42 → 21 → 10 → fixpoint (10 doesn't match anything)
    let mut program = make_program_with_literal(42);
    let registry = zyntax_typed_ast::TypeRegistry::new();

    let rw1 = ExprRewrite::new(
        "42_to_21",
        Priority::NORMALIZATION,
        Pattern::new("match_42", |n, _| {
            matches!(&n.node, TypedExpression::Literal(TypedLiteral::Integer(42)))
                .then(Bindings::new)
        }),
        |_, _| {
            RewriteOutput::ReplaceExpr(TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(21)),
                Type::Primitive(PrimitiveType::I64),
                Span::new(0, 0),
            ))
        },
    );

    let rw2 = ExprRewrite::new(
        "21_to_10",
        Priority::NORMALIZATION,
        Pattern::new("match_21", |n, _| {
            matches!(&n.node, TypedExpression::Literal(TypedLiteral::Integer(21)))
                .then(Bindings::new)
        }),
        |_, _| {
            RewriteOutput::ReplaceExpr(TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(10)),
                Type::Primitive(PrimitiveType::I64),
                Span::new(0, 0),
            ))
        },
    );

    let mut engine = PatternEngine::new(EngineConfig::default());
    engine.register_expr_rewrite(rw1);
    engine.register_expr_rewrite(rw2);
    engine.finalize().unwrap();

    let result = engine.run(&mut program, &registry);

    assert!(result.changed);
    // Two rewrites fired across iterations: 42→21 (iter 1), 21→10 (iter 2)
    assert_eq!(result.rewrites_fired.len(), 2);
    assert_eq!(find_int_literal(&program), Some(10));
    assert_eq!(result.iterations, 3); // iter1: 42→21, iter2: 21→10, iter3: no change
}

#[test]
fn test_target_guard() {
    let mut program = make_program_with_literal(42);
    let registry = zyntax_typed_ast::TypeRegistry::new();

    // This rewrite only fires for NVPTX target
    let pattern = Pattern::<TypedExpression>::new("match_42", |node, _ctx| {
        if let TypedExpression::Literal(TypedLiteral::Integer(42)) = &node.node {
            Some(Bindings::new())
        } else {
            None
        }
    })
    .for_target(LoweringTarget::Nvptx);

    let rewrite = ExprRewrite::new("gpu_only_rewrite", Priority::TARGET, pattern, |_, _| {
        RewriteOutput::ReplaceExpr(TypedNode::new(
            TypedExpression::Literal(TypedLiteral::Integer(0)),
            Type::Primitive(PrimitiveType::I64),
            Span::new(0, 0),
        ))
    });

    // Engine targeting CPU — rewrite should NOT fire
    let mut engine = PatternEngine::new(EngineConfig {
        target: LoweringTarget::Cpu,
        ..Default::default()
    });
    engine.register_expr_rewrite(rewrite);
    engine.finalize().unwrap();

    let result = engine.run(&mut program, &registry);
    assert!(!result.changed);
    assert_eq!(find_int_literal(&program), Some(42));
}
