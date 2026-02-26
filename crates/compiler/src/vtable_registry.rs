//! # Vtable Registry
//!
//! Central registry for managing virtual method tables (vtables) for trait dispatch.
//!
//! ## Architecture
//!
//! The vtable registry maintains:
//! - Mapping: (trait_id, type_id) → vtable_global_id
//! - Mapping: (impl_def_id, method_name) → function_id
//! - Cache of generated vtables
//! - Super-trait vtable relationships
//!
//! ## Usage
//!
//! ```ignore
//! let mut registry = VtableRegistry::new();
//!
//! // Register method implementations
//! registry.register_method(impl_id, method_name, function_id);
//!
//! // Generate or lookup vtable
//! let vtable_id = registry.get_or_create_vtable(trait_id, type_id, ...)?;
//!
//! // Lookup for upcasting
//! let super_vtable = registry.get_super_trait_vtable(sub_trait_id, super_trait_id, type_id)?;
//! ```

use crate::hir::{HirGlobal, HirId, HirType, HirVTable};
use crate::CompilerResult;
use std::collections::HashMap;
use zyntax_typed_ast::{InternedString, TypeId};

/// Central registry for vtables and method-to-function mappings
#[derive(Debug, Clone)]
pub struct VtableRegistry {
    /// Mapping: (trait_id, type_id) → vtable_global_id
    vtable_cache: HashMap<(TypeId, TypeId), HirId>,

    /// Mapping: (trait_id, type_id, method_name) → function_id
    method_implementations: HashMap<(TypeId, TypeId, InternedString), HirId>,

    /// Stored vtables (vtable_id → HirVTable)
    vtables: HashMap<HirId, HirVTable>,

    /// Stored vtable globals (vtable_id → HirGlobal)
    vtable_globals: HashMap<HirId, HirGlobal>,

    /// Super-trait relationships: (sub_trait_id, super_trait_id, type_id) → vtable_id
    super_trait_vtables: HashMap<(TypeId, TypeId, TypeId), HirId>,
}

impl VtableRegistry {
    /// Create a new empty vtable registry
    pub fn new() -> Self {
        Self {
            vtable_cache: HashMap::new(),
            method_implementations: HashMap::new(),
            vtables: HashMap::new(),
            vtable_globals: HashMap::new(),
            super_trait_vtables: HashMap::new(),
        }
    }

    /// Register a method implementation
    ///
    /// Maps (trait_id, type_id, method_name) → function_id so vtable generation
    /// can find the actual function to call.
    ///
    /// # Arguments
    /// * `trait_id` - Trait being implemented
    /// * `type_id` - Type implementing the trait
    /// * `method_name` - Name of the method
    /// * `function_id` - HIR function ID for this method implementation
    pub fn register_method(
        &mut self,
        trait_id: TypeId,
        type_id: TypeId,
        method_name: InternedString,
        function_id: HirId,
    ) {
        self.method_implementations
            .insert((trait_id, type_id, method_name), function_id);
    }

    /// Lookup method implementation function ID
    ///
    /// Returns the HIR function ID for a method implementation, or None if not registered.
    pub fn get_method_function(
        &self,
        trait_id: TypeId,
        type_id: TypeId,
        method_name: InternedString,
    ) -> Option<HirId> {
        self.method_implementations
            .get(&(trait_id, type_id, method_name))
            .copied()
    }

    /// Register a vtable
    ///
    /// Stores a vtable and its global definition in the registry.
    /// This is called after generating a vtable to cache it for future lookups.
    pub fn register_vtable(
        &mut self,
        trait_id: TypeId,
        type_id: TypeId,
        vtable: HirVTable,
        vtable_global: HirGlobal,
    ) -> HirId {
        let vtable_id = vtable.id;

        // Store in cache
        self.vtable_cache.insert((trait_id, type_id), vtable_id);

        // Store vtable and global
        self.vtables.insert(vtable_id, vtable);
        self.vtable_globals.insert(vtable_id, vtable_global);

        vtable_id
    }

    /// Get or create a vtable for (trait_id, type_id) pair
    ///
    /// Checks cache first, generates new vtable if not found.
    /// This is the main entry point for vtable management.
    pub fn get_or_create_vtable(
        &mut self,
        trait_id: TypeId,
        type_id: TypeId,
    ) -> CompilerResult<HirId> {
        // Check cache
        if let Some(&vtable_id) = self.vtable_cache.get(&(trait_id, type_id)) {
            return Ok(vtable_id);
        }

        // Not in cache - caller needs to generate it
        // Return error to signal vtable generation needed
        Err(crate::CompilerError::Analysis(format!(
            "Vtable for trait {:?} on type {:?} not yet generated. Call generate_and_register_vtable first.",
            trait_id, type_id
        )))
    }

    /// Get a vtable by ID
    pub fn get_vtable(&self, vtable_id: HirId) -> Option<&HirVTable> {
        self.vtables.get(&vtable_id)
    }

    /// Get a vtable global by ID
    pub fn get_vtable_global(&self, vtable_id: HirId) -> Option<&HirGlobal> {
        self.vtable_globals.get(&vtable_id)
    }

    /// Get all registered vtable globals
    ///
    /// Used during module finalization to emit all vtables as module globals.
    pub fn get_all_vtable_globals(&self) -> Vec<&HirGlobal> {
        self.vtable_globals.values().collect()
    }

    /// Register a super-trait vtable relationship
    ///
    /// When type T implements SubTrait (which extends SuperTrait), we need two vtables:
    /// - vtable for (SubTrait, T)
    /// - vtable for (SuperTrait, T)
    ///
    /// This tracks the relationship so upcasting can find the super-trait vtable.
    pub fn register_super_trait_vtable(
        &mut self,
        sub_trait_id: TypeId,
        super_trait_id: TypeId,
        type_id: TypeId,
        super_vtable_id: HirId,
    ) {
        self.super_trait_vtables
            .insert((sub_trait_id, super_trait_id, type_id), super_vtable_id);
    }

    /// Get super-trait vtable for upcasting
    ///
    /// Given a trait object of type SubTrait on concrete type T,
    /// return the vtable ID for SuperTrait on T.
    pub fn get_super_trait_vtable(
        &self,
        sub_trait_id: TypeId,
        super_trait_id: TypeId,
        type_id: TypeId,
    ) -> CompilerResult<HirId> {
        self.super_trait_vtables
            .get(&(sub_trait_id, super_trait_id, type_id))
            .copied()
            .ok_or_else(|| {
                crate::CompilerError::Analysis(format!(
                    "No super-trait vtable found for {:?} → {:?} on type {:?}",
                    sub_trait_id, super_trait_id, type_id
                ))
            })
    }

    /// Check if a vtable exists for (trait_id, type_id)
    pub fn has_vtable(&self, trait_id: TypeId, type_id: TypeId) -> bool {
        self.vtable_cache.contains_key(&(trait_id, type_id))
    }

    /// Get statistics about registry contents
    pub fn stats(&self) -> VtableRegistryStats {
        VtableRegistryStats {
            total_vtables: self.vtables.len(),
            total_methods: self.method_implementations.len(),
            total_super_trait_relationships: self.super_trait_vtables.len(),
        }
    }
}

impl Default for VtableRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about vtable registry
#[derive(Debug, Clone, Copy)]
pub struct VtableRegistryStats {
    pub total_vtables: usize,
    pub total_methods: usize,
    pub total_super_trait_relationships: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vtable_registry_creation() {
        let registry = VtableRegistry::new();
        let stats = registry.stats();

        assert_eq!(stats.total_vtables, 0);
        assert_eq!(stats.total_methods, 0);
        assert_eq!(stats.total_super_trait_relationships, 0);
    }

    #[test]
    fn test_method_registration() {
        use zyntax_typed_ast::AstArena;

        let mut arena = AstArena::new();
        let method_name = arena.intern_string("test_method");

        let mut registry = VtableRegistry::new();
        let trait_id = TypeId::new(1);
        let type_id = TypeId::new(2);
        let function_id = HirId::new();

        // Register method
        registry.register_method(trait_id, type_id, method_name, function_id);

        // Verify registration
        assert_eq!(
            registry.get_method_function(trait_id, type_id, method_name),
            Some(function_id)
        );

        // Stats should reflect registration
        let stats = registry.stats();
        assert_eq!(stats.total_methods, 1);
    }

    #[test]
    fn test_vtable_cache_miss() {
        let mut registry = VtableRegistry::new();

        let trait_id = TypeId::new(1);
        let type_id = TypeId::new(2);

        // Should error when vtable not in cache
        assert!(registry.get_or_create_vtable(trait_id, type_id).is_err());
    }

    #[test]
    fn test_has_vtable() {
        let registry = VtableRegistry::new();

        let trait_id = TypeId::new(1);
        let type_id = TypeId::new(2);

        assert!(!registry.has_vtable(trait_id, type_id));
    }

    #[test]
    fn test_vtable_registration_and_lookup() {
        use zyntax_typed_ast::AstArena;

        let mut arena = AstArena::new();
        let trait_name = arena.intern_string("TestTrait");

        let mut registry = VtableRegistry::new();
        let trait_id = TypeId::new(1);
        let type_id = TypeId::new(2);

        // Create vtable
        let vtable = HirVTable {
            id: HirId::new(),
            trait_id,
            for_type: HirType::I32,
            methods: vec![],
        };
        let vtable_id = vtable.id;

        // Create global with all required fields
        let global = HirGlobal {
            id: vtable_id,
            name: trait_name,
            ty: HirType::Ptr(Box::new(HirType::Void)),
            initializer: None,
            is_const: true,
            is_thread_local: false,
            linkage: crate::hir::Linkage::Internal,
            visibility: crate::hir::Visibility::Default,
        };

        // Register
        let registered_id = registry.register_vtable(trait_id, type_id, vtable, global);
        assert_eq!(registered_id, vtable_id);

        // Verify cache
        assert!(registry.has_vtable(trait_id, type_id));

        // Verify lookup
        let cached_vtable_id = registry.get_or_create_vtable(trait_id, type_id).unwrap();
        assert_eq!(cached_vtable_id, vtable_id);

        // Verify retrieval
        assert!(registry.get_vtable(vtable_id).is_some());
        assert!(registry.get_vtable_global(vtable_id).is_some());

        // Stats
        let stats = registry.stats();
        assert_eq!(stats.total_vtables, 1);
    }

    #[test]
    fn test_super_trait_vtable_lookup() {
        let mut registry = VtableRegistry::new();

        let sub_trait_id = TypeId::new(1);
        let super_trait_id = TypeId::new(2);
        let type_id = TypeId::new(3);
        let super_vtable_id = HirId::new();

        // Register super-trait relationship
        registry.register_super_trait_vtable(
            sub_trait_id,
            super_trait_id,
            type_id,
            super_vtable_id,
        );

        // Verify lookup succeeds
        let result = registry.get_super_trait_vtable(sub_trait_id, super_trait_id, type_id);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), super_vtable_id);

        // Verify stats
        let stats = registry.stats();
        assert_eq!(stats.total_super_trait_relationships, 1);
    }

    #[test]
    fn test_super_trait_vtable_missing() {
        let registry = VtableRegistry::new();

        let sub_trait_id = TypeId::new(1);
        let super_trait_id = TypeId::new(2);
        let type_id = TypeId::new(3);

        // Should error when relationship not registered
        let result = registry.get_super_trait_vtable(sub_trait_id, super_trait_id, type_id);
        assert!(result.is_err());
    }
}
