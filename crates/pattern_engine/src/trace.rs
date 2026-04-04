use crate::rewrite::RewriteId;
use std::collections::HashSet;
use zyntax_typed_ast::source::Span;

/// Record of a single rewrite that fired during engine execution.
#[derive(Debug, Clone)]
pub struct FiredRewrite {
    pub rewrite_id: RewriteId,
    pub rewrite_name: &'static str,
    pub node_span: Span,
    pub iteration: usize,
}

/// Set of rewrite IDs that have fired in the current iteration.
#[derive(Debug, Clone, Default)]
pub struct FiredSet {
    ids: HashSet<RewriteId>,
}

impl FiredSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: RewriteId) {
        self.ids.insert(id);
    }

    pub fn contains(&self, id: &RewriteId) -> bool {
        self.ids.contains(id)
    }

    pub fn clear(&mut self) {
        self.ids.clear();
    }
}
