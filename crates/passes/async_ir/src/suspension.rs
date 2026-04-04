//! Rewrite: Await(inner) → Match(poll_dispatch)

use pattern_engine::{Bindings, ExprRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::typed_ast::*;

/// Match Await expressions.
pub fn await_expr_to_poll() -> ExprRewrite {
    ExprRewrite::new(
        "await_expr_to_poll",
        Priority::SEMANTIC,
        Pattern::new("await_expr", |node, _ctx| {
            if let TypedExpression::Await(_inner) = &node.node {
                let mut bindings = Bindings::new();
                bindings.bind_span("span", node.span);
                return Some(bindings);
            }
            None
        }),
        |_bindings, _builder| {
            // TODO: Build poll dispatch match expression.
            RewriteOutput::Unchanged
        },
    )
}
