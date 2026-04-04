//! Rewrite: Call to effect operation → handler dispatch through vtable
//!
//! Currently a no-op pattern — effect op resolution requires the program's
//! Effect declarations to be available in MatchContext, which is not yet
//! implemented. The vtable and dispatch rewrites are complete; this one
//! will fire once MatchContext exposes effect operation lookup.

use pattern_engine::{Bindings, ExprRewrite, Pattern, Priority, RewriteOutput};
use zyntax_typed_ast::typed_ast::*;

pub fn effect_op_to_dispatch() -> ExprRewrite {
    ExprRewrite::new(
        "effect_op_to_dispatch",
        Priority::SEMANTIC,
        // This pattern needs to check if a Call's callee name matches a
        // declared effect operation. That requires the MatchContext to expose
        // a lookup: "is this name an effect op?" — currently not available.
        // Returns None (never matches) until that's wired.
        Pattern::new("effect_op_call", |_node, _ctx| None),
        |_matched, _bindings, _builder| RewriteOutput::Unchanged,
    )
}
