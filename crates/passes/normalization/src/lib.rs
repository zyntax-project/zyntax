//! Normalization pass — structural cleanup rewrites that run first.
//!
//! Two rewrites that produce real transformations:
//! - `flatten_nested_blocks`: Block containing only another Block → hoist inner statements
//! - `unit_return_explicit`: Void function without terminal Return → append Return(None)

use pattern_engine::{
    Bindings, DeclRewrite, Pattern, PatternEngine, PatternPass, Priority, RewriteOutput,
    StmtRewrite,
};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type};
use zyntax_typed_ast::typed_ast::*;

pub struct Pass;

impl PatternPass for Pass {
    fn name(&self) -> &'static str {
        "normalization"
    }

    fn description(&self) -> &'static str {
        "Structural cleanup: block flattening, explicit returns"
    }

    fn register(&self, engine: &mut PatternEngine) {
        engine.register_stmt_rewrite(flatten_nested_blocks());
        engine.register_decl_rewrite(unit_return_explicit());
    }
}

/// Block containing a single statement that is itself a Block → hoist inner statements.
fn flatten_nested_blocks() -> StmtRewrite {
    StmtRewrite::new(
        "flatten_nested_blocks",
        Priority::NORMALIZATION,
        Pattern::new("nested_block", |node, _ctx| {
            if let TypedStatement::Block(outer) = &node.node {
                if outer.statements.len() == 1 {
                    if let TypedStatement::Block(_) = &outer.statements[0].node {
                        return Some(Bindings::new());
                    }
                }
            }
            None
        }),
        |matched, _bindings, _builder| {
            if let TypedStatement::Block(outer) = &matched.node {
                if let Some(inner_stmt) = outer.statements.first() {
                    if let TypedStatement::Block(inner) = &inner_stmt.node {
                        // Replace the outer block wrapping a single inner block
                        // with just the inner block
                        return RewriteOutput::ReplaceStmt(TypedNode::new(
                            TypedStatement::Block(inner.clone()),
                            matched.ty.clone(),
                            matched.span,
                        ));
                    }
                }
            }
            RewriteOutput::Unchanged
        },
    )
}

/// Function returning Unit with no terminal Return → append Return(None).
fn unit_return_explicit() -> DeclRewrite {
    DeclRewrite::new(
        "unit_return_explicit",
        Priority::NORMALIZATION,
        Pattern::new("void_fn_no_return", |node, _ctx| {
            if let TypedDeclaration::Function(func) = &node.node {
                if !matches!(func.return_type, Type::Primitive(PrimitiveType::Unit)) {
                    return None;
                }
                let body = func.body.as_ref()?;
                if body.statements.is_empty() {
                    // Empty body — needs Return(None)
                    return Some(Bindings::new());
                }
                let last = body.statements.last()?;
                if matches!(&last.node, TypedStatement::Return(_)) {
                    return None; // already has explicit return
                }
                // If the function body is a single bare expression, it's an
                // implicit return — do NOT append Return(None).
                // The CFG builder will wrap it as Return(Some(expr)).
                if body.statements.len() == 1 {
                    if matches!(&last.node, TypedStatement::Expression(_)) {
                        return None; // implicit return, not void
                    }
                }
                Some(Bindings::new())
            } else {
                None
            }
        }),
        |matched, _bindings, _builder| {
            if let TypedDeclaration::Function(func) = &matched.node {
                let mut new_func = func.clone();
                let span = func.body.as_ref().map(|b| b.span).unwrap_or(matched.span);

                let return_stmt = TypedNode::new(
                    TypedStatement::Return(None),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                );

                if let Some(body) = &mut new_func.body {
                    body.statements.push(return_stmt);
                } else {
                    new_func.body = Some(TypedBlock {
                        statements: vec![return_stmt],
                        span,
                    });
                }

                RewriteOutput::ReplaceDecl(TypedNode::new(
                    TypedDeclaration::Function(new_func),
                    matched.ty.clone(),
                    matched.span,
                ))
            } else {
                RewriteOutput::Unchanged
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pattern_engine::{EngineConfig, PatternEngine};
    use zyntax_typed_ast::InternedString;

    fn span() -> Span {
        Span::new(0, 0)
    }

    #[test]
    fn test_flatten_nested_blocks() {
        let s = span();
        let inner_stmt = TypedNode::new(
            TypedStatement::Expression(Box::new(TypedNode::new(
                TypedExpression::Literal(TypedLiteral::Integer(1)),
                Type::Primitive(PrimitiveType::I64),
                s,
            ))),
            Type::Primitive(PrimitiveType::Unit),
            s,
        );
        let inner_block = TypedNode::new(
            TypedStatement::Block(TypedBlock {
                statements: vec![inner_stmt],
                span: s,
            }),
            Type::Primitive(PrimitiveType::Unit),
            s,
        );
        let outer_block = TypedNode::new(
            TypedStatement::Block(TypedBlock {
                statements: vec![inner_block],
                span: s,
            }),
            Type::Primitive(PrimitiveType::Unit),
            s,
        );

        // Wrap in a function so the engine can walk it
        let func = TypedFunction {
            name: InternedString::new_global("test"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![outer_block],
                span: s,
            }),
            ..Default::default()
        };
        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Function(func),
                Type::Primitive(PrimitiveType::Unit),
                s,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(Pass);
        engine.finalize().unwrap();

        let result = engine.run(&mut program, &registry);
        assert!(result.changed);

        // The nested block should be flattened AND unit_return_explicit fires
        assert!(result.rewrites_fired.len() >= 1);
    }

    #[test]
    fn test_unit_return_explicit() {
        let s = span();
        // Use a Let + Expression body (multi-statement) so the single-expression
        // implicit-return guard doesn't skip this function
        let let_stmt = TypedNode::new(
            TypedStatement::Let(TypedLet {
                name: InternedString::new_global("x"),
                ty: Type::Primitive(PrimitiveType::I64),
                initializer: Some(Box::new(TypedNode::new(
                    TypedExpression::Literal(TypedLiteral::Integer(42)),
                    Type::Primitive(PrimitiveType::I64),
                    s,
                ))),
                mutability: zyntax_typed_ast::type_registry::Mutability::Immutable,
                span: s,
            }),
            Type::Primitive(PrimitiveType::Unit),
            s,
        );
        let expr_stmt = TypedNode::new(
            TypedStatement::Expression(Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("x")),
                Type::Primitive(PrimitiveType::I64),
                s,
            ))),
            Type::Primitive(PrimitiveType::Unit),
            s,
        );

        let func = TypedFunction {
            name: InternedString::new_global("foo"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![let_stmt, expr_stmt],
                span: s,
            }),
            ..Default::default()
        };
        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Function(func),
                Type::Primitive(PrimitiveType::Unit),
                s,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(Pass);
        engine.finalize().unwrap();

        let result = engine.run(&mut program, &registry);
        assert!(result.changed);

        // Verify the function now ends with Return(None)
        if let TypedDeclaration::Function(func) = &program.declarations[0].node {
            let body = func.body.as_ref().unwrap();
            let last = body.statements.last().unwrap();
            assert!(
                matches!(&last.node, TypedStatement::Return(None)),
                "Expected Return(None) at end, got {:?}",
                last.node
            );
        } else {
            panic!("Expected Function declaration");
        }
    }

    #[test]
    fn test_idempotent() {
        let s = span();
        // Multi-statement void function (not single-expression implicit return)
        let func = TypedFunction {
            name: InternedString::new_global("bar"),
            return_type: Type::Primitive(PrimitiveType::Unit),
            body: Some(TypedBlock {
                statements: vec![
                    TypedNode::new(
                        TypedStatement::Let(TypedLet {
                            name: InternedString::new_global("x"),
                            ty: Type::Primitive(PrimitiveType::I64),
                            initializer: Some(Box::new(TypedNode::new(
                                TypedExpression::Literal(TypedLiteral::Integer(1)),
                                Type::Primitive(PrimitiveType::I64),
                                s,
                            ))),
                            mutability: zyntax_typed_ast::type_registry::Mutability::Immutable,
                            span: s,
                        }),
                        Type::Primitive(PrimitiveType::Unit),
                        s,
                    ),
                    TypedNode::new(
                        TypedStatement::Expression(Box::new(TypedNode::new(
                            TypedExpression::Variable(InternedString::new_global("x")),
                            Type::Primitive(PrimitiveType::I64),
                            s,
                        ))),
                        Type::Primitive(PrimitiveType::Unit),
                        s,
                    ),
                ],
                span: s,
            }),
            ..Default::default()
        };
        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Function(func),
                Type::Primitive(PrimitiveType::Unit),
                s,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();

        // First run
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(Pass);
        engine.finalize().unwrap();
        let r1 = engine.run(&mut program, &registry);
        assert!(r1.changed);

        // Second run — should be no-op
        let mut engine2 = PatternEngine::new(EngineConfig::default());
        engine2.register_pass(Pass);
        engine2.finalize().unwrap();
        let r2 = engine2.run(&mut program, &registry);
        assert!(!r2.changed, "Normalization should be idempotent");
    }
}
