//! Rewrite: Await(inner) → poll dispatch
//!
//! Each `await expr` becomes a state transition:
//!   match poll(expr) {
//!     Ready(v) => v,
//!     Pending => { self.state = AwaitN; return Pending; }
//!   }
//!
//! The state enum variant index is tracked per-function by the state machine builder.
//! This rewrite handles the expression-level transformation.

use pattern_engine::{Bindings, ExprRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::source::Span;
use zyntax_typed_ast::type_registry::{PrimitiveType, Type};
use zyntax_typed_ast::typed_ast::*;
use zyntax_typed_ast::InternedString;

pub fn await_expr_to_poll() -> ExprRewrite {
    ExprRewrite::new(
        "await_expr_to_poll",
        Priority::SEMANTIC,
        Pattern::new("await_expr", |node, _ctx| {
            matches!(&node.node, TypedExpression::Await(_)).then(Bindings::new)
        }),
        |matched, _bindings, _builder| {
            if let TypedExpression::Await(inner) = &matched.node {
                // Replace await(expr) with a call to poll(expr).
                // In a full implementation this would be a Match expression
                // dispatching on Poll::Ready/Poll::Pending. For now, we desugar
                // to a direct call to the poll function, which the lowering
                // pipeline will handle.
                let poll_name = InternedString::new_global("__poll");
                let callee = TypedNode::new(
                    TypedExpression::Variable(poll_name),
                    Type::Any,
                    matched.span,
                );
                let call = TypedExpression::Call(TypedCall {
                    callee: Box::new(callee),
                    positional_args: vec![*inner.clone()],
                    named_args: vec![],
                    type_args: vec![],
                });
                RewriteOutput::ReplaceExpr(TypedNode::new(call, matched.ty.clone(), matched.span))
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

    #[test]
    fn test_await_rewritten_to_poll_call() {
        let span = Span::new(0, 0);
        let await_expr = TypedNode::new(
            TypedExpression::Await(Box::new(TypedNode::new(
                TypedExpression::Variable(InternedString::new_global("future")),
                Type::Primitive(PrimitiveType::I64),
                span,
            ))),
            Type::Primitive(PrimitiveType::I64),
            span,
        );

        // Wrap in a function body so the engine can walk it
        let func = TypedFunction {
            name: InternedString::new_global("test_fn"),
            is_async: true,
            return_type: Type::Primitive(PrimitiveType::I64),
            body: Some(TypedBlock {
                statements: vec![TypedNode::new(
                    TypedStatement::Expression(Box::new(await_expr)),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                )],
                span,
            }),
            ..Default::default()
        };

        let mut program = TypedProgram {
            declarations: vec![TypedNode::new(
                TypedDeclaration::Function(func),
                Type::Primitive(PrimitiveType::Unit),
                span,
            )],
            ..Default::default()
        };

        let registry = zyntax_typed_ast::TypeRegistry::new();
        let mut engine = PatternEngine::new(EngineConfig::default());
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(super::super::Pass);
        engine.finalize().unwrap();

        let result = engine.run(&mut program, &registry);
        assert!(result.changed);

        // The await should be rewritten — check that no Await expressions remain
        // (the async_fn_to_state_machine also fires, expanding the function)
        let has_await = program.declarations.iter().any(|d| {
            if let TypedDeclaration::Function(f) = &d.node {
                if let Some(body) = &f.body {
                    return body.statements.iter().any(|s| {
                        if let TypedStatement::Expression(e) = &s.node {
                            matches!(&e.node, TypedExpression::Await(_))
                        } else {
                            false
                        }
                    });
                }
            }
            false
        });
        assert!(!has_await, "No Await expressions should remain");
    }
}
