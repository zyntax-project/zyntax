use crate::bindings::Bindings;
use crate::context::{LoweringTarget, MatchContext};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use zyntax_typed_ast::typed_ast::TypedNode;

/// Unique pattern identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatternId(pub u32);

static NEXT_PATTERN_ID: AtomicU32 = AtomicU32::new(1);

impl PatternId {
    pub fn fresh() -> Self {
        PatternId(NEXT_PATTERN_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A typed predicate that either matches (returning Bindings) or fails.
/// Patterns are composable via combinators.
pub struct Pattern<T: 'static> {
    pub id: PatternId,
    pub name: &'static str,
    predicate: Arc<dyn Fn(&TypedNode<T>, &MatchContext) -> Option<Bindings> + Send + Sync>,
}

impl<T: 'static> Pattern<T> {
    pub fn new(
        name: &'static str,
        f: impl Fn(&TypedNode<T>, &MatchContext) -> Option<Bindings> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: PatternId::fresh(),
            name,
            predicate: Arc::new(f),
        }
    }

    /// Try to match this pattern against a node.
    pub fn try_match(&self, node: &TypedNode<T>, ctx: &MatchContext) -> Option<Bindings> {
        (self.predicate)(node, ctx)
    }

    /// Both patterns must match; bindings are merged.
    pub fn and(self, other: Pattern<T>) -> Pattern<T> {
        let p1 = self.predicate;
        let p2 = other.predicate;
        Pattern {
            id: PatternId::fresh(),
            name: self.name,
            predicate: Arc::new(move |node, ctx| {
                let mut b1 = p1(node, ctx)?;
                let b2 = p2(node, ctx)?;
                b1.merge(b2);
                Some(b1)
            }),
        }
    }

    /// First match wins; bindings from winner returned.
    pub fn or(self, other: Pattern<T>) -> Pattern<T> {
        let p1 = self.predicate;
        let p2 = other.predicate;
        Pattern {
            id: PatternId::fresh(),
            name: self.name,
            predicate: Arc::new(move |node, ctx| p1(node, ctx).or_else(|| p2(node, ctx))),
        }
    }

    /// Match only when a guard predicate on the context holds.
    pub fn when(self, guard: impl Fn(&MatchContext) -> bool + Send + Sync + 'static) -> Pattern<T> {
        let inner = self.predicate;
        Pattern {
            id: self.id,
            name: self.name,
            predicate: Arc::new(
                move |node, ctx| {
                    if guard(ctx) {
                        inner(node, ctx)
                    } else {
                        None
                    }
                },
            ),
        }
    }

    /// Match only for a specific lowering target.
    pub fn for_target(self, target: LoweringTarget) -> Pattern<T> {
        self.when(move |ctx| ctx.target == target)
    }
}
