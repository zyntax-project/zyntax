//! Rewrite: async Function → state machine (Enum + Class + poll Function)

use pattern_engine::{Bindings, DeclRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::typed_ast::*;

/// Match async functions that contain Await expressions.
pub fn async_fn_to_state_machine() -> DeclRewrite {
    DeclRewrite::new(
        "async_fn_to_state_machine",
        Priority::SEMANTIC,
        Pattern::new("async_fn", |node, _ctx| {
            if let TypedDeclaration::Function(func) = &node.node {
                if func.is_async && func.body.is_some() {
                    let mut bindings = Bindings::new();
                    bindings.bind_span("span", node.span);
                    return Some(bindings);
                }
            }
            None
        }),
        |_bindings, _builder| {
            // TODO: Build state machine from async function body.
            // Requires: suspension point analysis, capture analysis,
            // state enum synthesis, poll function generation.
            RewriteOutput::Unchanged
        },
    )
}
