use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use zyntax_typed_ast::source::Span;

/// Unique identifier for a TypedAST node within a single engine run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

static NEXT_ID: AtomicU32 = AtomicU32::new(1);

impl NodeId {
    pub fn fresh() -> Self {
        NodeId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Reset the counter (call at the start of each engine run).
    pub fn reset() {
        NEXT_ID.store(1, Ordering::Relaxed);
    }
}

/// Maps Spans to NodeIds. Built by walking the TypedProgram once.
/// Synthesized nodes (from rewrites) get fresh IDs via `NodeId::fresh()`.
pub struct NodeIdMap {
    span_to_id: HashMap<Span, NodeId>,
}

impl NodeIdMap {
    pub fn new() -> Self {
        Self {
            span_to_id: HashMap::new(),
        }
    }

    /// Assign an ID for a span. If the span already has an ID, return it.
    pub fn id_for_span(&mut self, span: Span) -> NodeId {
        *self.span_to_id.entry(span).or_insert_with(NodeId::fresh)
    }

    /// Look up an existing ID by span.
    pub fn get(&self, span: &Span) -> Option<NodeId> {
        self.span_to_id.get(span).copied()
    }

    /// Build the map by walking all declarations in a program.
    pub fn build(program: &zyntax_typed_ast::TypedProgram) -> Self {
        let mut map = Self::new();
        for decl in &program.declarations {
            map.id_for_span(decl.span);
        }
        map
    }
}

impl Default for NodeIdMap {
    fn default() -> Self {
        Self::new()
    }
}
