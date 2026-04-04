//! Rewrite: Call to effect operation → handler dispatch through vtable
//!
//! A call like:
//!   let x = sample(normal_dist)
//!
//! Where `sample` resolves to a TypedEffectOp, becomes:
//!   let x = handler_vtable.sample_fn(normal_dist)
//!
//! The handler_vtable reference is resolved from the handler in scope.
//! Effect operations are identified by matching call names against
//! TypedEffect.operations declared in the program.

use pattern_engine::{Bindings, ExprRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::typed_ast::*;

/// Match Call expressions where the callee is a variable whose name
/// matches a known effect operation. Since the pattern engine walks the
/// program after type checking, effect op names are in scope.
///
/// The actual resolution of which handler is in scope requires the
/// EffectSystem scope stack — this is wired through MatchContext.effects.
pub fn effect_op_to_dispatch() -> ExprRewrite {
    ExprRewrite::new(
        "effect_op_to_dispatch",
        Priority::SEMANTIC,
        Pattern::new("effect_op_call", |node, _ctx| {
            // Match calls where the callee is a Variable referencing an effect op.
            // For now, this is a structural match — the actual effect-op resolution
            // requires checking the program's Effect declarations, which the
            // MatchContext doesn't currently expose directly.
            //
            // TODO: Add a method to MatchContext that checks if a name resolves
            // to an effect operation (by querying the TypedProgram's declarations).
            if let TypedExpression::Call(call) = &node.node {
                if let TypedExpression::Variable(_name) = &call.callee.node {
                    // Placeholder: don't match anything yet until MatchContext
                    // exposes effect operation lookup.
                    return None;
                }
            }
            None
        }),
        |_bindings, _builder| RewriteOutput::Unchanged,
    )
}
