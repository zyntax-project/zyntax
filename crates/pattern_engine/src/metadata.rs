use crate::node_id::NodeId;
use std::any::{Any, TypeId};
use std::collections::HashMap;

/// External metadata table for attaching target-specific data to TypedAST nodes.
///
/// Keyed by `(NodeId, TypeId_of_metadata)`. This keeps GPU thread hierarchy,
/// FPGA clock domains, and other target-specific concepts out of the core AST.
pub struct MetadataTable {
    entries: HashMap<NodeId, HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
}

impl MetadataTable {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn insert<M: Any + Send + Sync>(&mut self, node: NodeId, metadata: M) {
        self.entries
            .entry(node)
            .or_default()
            .insert(TypeId::of::<M>(), Box::new(metadata));
    }

    pub fn get<M: Any + Send + Sync>(&self, node: NodeId) -> Option<&M> {
        self.entries
            .get(&node)?
            .get(&TypeId::of::<M>())?
            .downcast_ref()
    }

    pub fn remove<M: Any + Send + Sync>(&mut self, node: NodeId) -> Option<M> {
        let type_map = self.entries.get_mut(&node)?;
        let boxed = type_map.remove(&TypeId::of::<M>())?;
        boxed.downcast().ok().map(|b| *b)
    }

    pub fn has<M: Any + Send + Sync>(&self, node: NodeId) -> bool {
        self.entries
            .get(&node)
            .and_then(|m| m.get(&TypeId::of::<M>()))
            .is_some()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for MetadataTable {
    fn default() -> Self {
        Self::new()
    }
}
