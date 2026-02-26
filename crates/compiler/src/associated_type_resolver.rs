//! # Associated Type Resolver
//!
//! Resolves associated type projections to concrete types during HIR lowering.
//!
//! ## Overview
//!
//! Associated types in traits allow implementations to specify concrete types:
//!
//! ```ignore
//! trait Iterator {
//!     type Item;
//!     fn next(&mut self) -> Option<Self::Item>;
//! }
//!
//! impl Iterator for Vec<i32> {
//!     type Item = i32;
//!     fn next(&mut self) -> Option<i32> { ... }
//! }
//! ```
//!
//! This resolver takes projections like `<Vec<i32> as Iterator>::Item` and
//! resolves them to concrete types like `i32` by looking up the trait implementation.
//!
//! ## Usage
//!
//! ```ignore
//! let resolver = AssociatedTypeResolver::new(type_registry, impl_registry);
//!
//! // Resolve <Vec<i32> as Iterator>::Item
//! let item_type = resolver.resolve(
//!     Iterator_trait_id,
//!     &Type::Vec(Box::new(Type::I32)),
//!     "Item"
//! )?;
//!
//! assert_eq!(item_type, Type::I32);
//! ```

use crate::{CompilerError, CompilerResult};
use std::collections::HashMap;
use std::sync::Arc;
use zyntax_typed_ast::{ImplDef, InternedString, Type, TypeId};

/// Resolves associated type projections to concrete types
///
/// This resolver is used during HIR lowering to convert associated type projections
/// like `<T as Trait>::Assoc` into concrete types based on trait implementations.
#[derive(Debug, Clone)]
pub struct AssociatedTypeResolver {
    /// Registry of trait implementations
    /// Key: (trait_id, implementing_type_id)
    /// Value: ImplDef containing associated type bindings
    impl_registry: HashMap<(TypeId, TypeId), Arc<ImplDef>>,
}

impl AssociatedTypeResolver {
    /// Create a new resolver with an empty impl registry
    pub fn new() -> Self {
        Self {
            impl_registry: HashMap::new(),
        }
    }

    /// Register a trait implementation
    ///
    /// This makes the impl available for associated type resolution.
    ///
    /// # Arguments
    ///
    /// * `trait_id` - The trait being implemented
    /// * `for_type_id` - The type implementing the trait
    /// * `impl_def` - The implementation definition with associated type bindings
    pub fn register_impl(&mut self, trait_id: TypeId, for_type_id: TypeId, impl_def: Arc<ImplDef>) {
        self.impl_registry.insert((trait_id, for_type_id), impl_def);
    }

    /// Resolve an associated type projection to a concrete type
    ///
    /// Given a projection like `<T as Trait>::Assoc`, returns the concrete type
    /// from the trait implementation.
    ///
    /// # Arguments
    ///
    /// * `trait_id` - The trait containing the associated type
    /// * `self_ty` - The implementing type (e.g., Vec<i32>)
    /// * `assoc_name` - The associated type name (e.g., "Item")
    ///
    /// # Returns
    ///
    /// The concrete type from the implementation's associated type binding.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No implementation of the trait for the given type exists
    /// - The associated type is not defined in the implementation
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Resolve <Vec<i32> as Iterator>::Item
    /// let item_type = resolver.resolve(
    ///     Iterator_trait_id,
    ///     &Type::Concrete(Vec<i32>),
    ///     "Item"
    /// )?;
    /// // Returns Type::I32
    /// ```
    pub fn resolve(
        &self,
        trait_id: TypeId,
        self_ty: &Type,
        assoc_name: InternedString,
    ) -> CompilerResult<Type> {
        // Extract the type ID from the type
        // For complex types, we need to normalize them first
        let for_type_id = self.extract_type_id(self_ty)?;

        // Look up the implementation
        let impl_def = self.find_impl(trait_id, for_type_id).ok_or_else(|| {
            CompilerError::Analysis(format!(
                "No implementation of trait {:?} found for type {:?}",
                trait_id, self_ty
            ))
        })?;

        // Look up the associated type binding in the impl
        let concrete_type = impl_def.associated_types.get(&assoc_name).ok_or_else(|| {
            CompilerError::Analysis(format!(
                "Associated type '{}' not found in implementation of trait {:?} for type {:?}",
                assoc_name, trait_id, self_ty
            ))
        })?;

        Ok(concrete_type.clone())
    }

    /// Find an implementation for a given trait and type
    ///
    /// # Arguments
    ///
    /// * `trait_id` - The trait being implemented
    /// * `for_type_id` - The implementing type
    ///
    /// # Returns
    ///
    /// The implementation definition if found, None otherwise.
    fn find_impl(&self, trait_id: TypeId, for_type_id: TypeId) -> Option<&Arc<ImplDef>> {
        self.impl_registry.get(&(trait_id, for_type_id))
    }

    /// Extract a TypeId from a Type
    ///
    /// For simple types, this is straightforward. For complex types like
    /// generics, we need to extract the base type.
    ///
    /// # Arguments
    ///
    /// * `ty` - The type to extract an ID from
    ///
    /// # Returns
    ///
    /// The TypeId if the type can be identified.
    ///
    /// # Errors
    ///
    /// Returns an error if the type cannot be identified (e.g., type variables).
    fn extract_type_id(&self, ty: &Type) -> CompilerResult<TypeId> {
        match ty {
            // Named types have TypeIds directly
            Type::Named { id, .. } => Ok(*id),

            // Primitive types don't have type IDs - they would need special handling
            // In a full implementation, we might create synthetic TypeIds for primitives
            Type::Primitive(_) => Err(CompilerError::Analysis(format!(
                "Primitive types cannot have trait implementations yet: {:?}",
                ty
            ))),

            // Type variables and other abstract types cannot be resolved yet
            _ => Err(CompilerError::Analysis(format!(
                "Cannot extract type ID from type: {:?}",
                ty
            ))),
        }
    }

    /// Get all registered implementations
    ///
    /// Returns an iterator over (trait_id, for_type_id, impl_def) tuples.
    pub fn impls(&self) -> impl Iterator<Item = (TypeId, TypeId, &Arc<ImplDef>)> {
        self.impl_registry
            .iter()
            .map(|((trait_id, for_type_id), impl_def)| (*trait_id, *for_type_id, impl_def))
    }

    /// Clear all registered implementations
    ///
    /// Useful for testing or when recompiling a module.
    pub fn clear(&mut self) {
        self.impl_registry.clear();
    }

    /// Get the number of registered implementations
    pub fn impl_count(&self) -> usize {
        self.impl_registry.len()
    }
}

impl Default for AssociatedTypeResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::arena::AstArena;
    use zyntax_typed_ast::{AssociatedTypeDef, MethodImpl, PrimitiveType, TypeBound};

    fn create_test_impl(
        arena: &mut AstArena,
        trait_id: TypeId,
        for_type_id: TypeId,
        assoc_type_name: &str,
        assoc_type_value: Type,
    ) -> ImplDef {
        let mut associated_types = HashMap::new();
        associated_types.insert(arena.intern_string(assoc_type_name), assoc_type_value);

        ImplDef {
            trait_id,
            for_type: Type::Named {
                id: for_type_id,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            type_args: vec![],
            methods: vec![],
            associated_types,
            where_clause: vec![],
            span: Default::default(),
        }
    }

    #[test]
    fn test_register_and_find_impl() {
        let mut arena = AstArena::new();
        let mut resolver = AssociatedTypeResolver::new();

        let iterator_trait = TypeId::new(1);
        let vec_type = TypeId::new(2);

        let impl_def = create_test_impl(
            &mut arena,
            iterator_trait,
            vec_type,
            "Item",
            Type::Primitive(PrimitiveType::I32),
        );

        resolver.register_impl(iterator_trait, vec_type, Arc::new(impl_def));

        assert_eq!(resolver.impl_count(), 1);
        assert!(resolver.find_impl(iterator_trait, vec_type).is_some());
    }

    #[test]
    fn test_resolve_simple_associated_type() {
        let mut arena = AstArena::new();
        let mut resolver = AssociatedTypeResolver::new();

        let iterator_trait = TypeId::new(1);
        let vec_type = TypeId::new(2);

        let impl_def = create_test_impl(
            &mut arena,
            iterator_trait,
            vec_type,
            "Item",
            Type::Primitive(PrimitiveType::I32),
        );

        resolver.register_impl(iterator_trait, vec_type, Arc::new(impl_def));

        // Resolve <Vec as Iterator>::Item
        let result = resolver.resolve(
            iterator_trait,
            &Type::Named {
                id: vec_type,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            arena.intern_string("Item"),
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Type::Primitive(PrimitiveType::I32));
    }

    #[test]
    fn test_resolve_missing_impl() {
        let mut arena = AstArena::new();
        let resolver = AssociatedTypeResolver::new();

        let iterator_trait = TypeId::new(1);
        let vec_type = TypeId::new(2);

        // Try to resolve without registering an impl
        let result = resolver.resolve(
            iterator_trait,
            &Type::Named {
                id: vec_type,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            arena.intern_string("Item"),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_missing_associated_type() {
        let mut arena = AstArena::new();
        let mut resolver = AssociatedTypeResolver::new();

        let iterator_trait = TypeId::new(1);
        let vec_type = TypeId::new(2);

        let impl_def = create_test_impl(
            &mut arena,
            iterator_trait,
            vec_type,
            "Item",
            Type::Primitive(PrimitiveType::I32),
        );

        resolver.register_impl(iterator_trait, vec_type, Arc::new(impl_def));

        // Try to resolve a different associated type
        let result = resolver.resolve(
            iterator_trait,
            &Type::Named {
                id: vec_type,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            arena.intern_string("Output"), // Wrong name
        );

        // Should error because impl only has "Item", not "Output"
        assert!(
            result.is_err(),
            "Expected error for missing associated type, got: {:?}",
            result
        );
    }

    #[test]
    fn test_multiple_impls() {
        let mut arena = AstArena::new();
        let mut resolver = AssociatedTypeResolver::new();

        let iterator_trait = TypeId::new(1);
        let vec_i32 = TypeId::new(2);
        let vec_string = TypeId::new(3);

        // Register impl Iterator for Vec<i32>
        let impl1 = create_test_impl(
            &mut arena,
            iterator_trait,
            vec_i32,
            "Item",
            Type::Primitive(PrimitiveType::I32),
        );
        resolver.register_impl(iterator_trait, vec_i32, Arc::new(impl1));

        // Register impl Iterator for Vec<String>
        let impl2 = create_test_impl(
            &mut arena,
            iterator_trait,
            vec_string,
            "Item",
            Type::Primitive(PrimitiveType::String),
        );
        resolver.register_impl(iterator_trait, vec_string, Arc::new(impl2));

        // Resolve for Vec<i32>
        let result1 = resolver.resolve(
            iterator_trait,
            &Type::Named {
                id: vec_i32,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            arena.intern_string("Item"),
        );
        assert_eq!(result1.unwrap(), Type::Primitive(PrimitiveType::I32));

        // Resolve for Vec<String>
        let result2 = resolver.resolve(
            iterator_trait,
            &Type::Named {
                id: vec_string,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: zyntax_typed_ast::NullabilityKind::NonNull,
            },
            arena.intern_string("Item"),
        );
        assert_eq!(result2.unwrap(), Type::Primitive(PrimitiveType::String));
    }

    #[test]
    fn test_clear() {
        let mut arena = AstArena::new();
        let mut resolver = AssociatedTypeResolver::new();

        let iterator_trait = TypeId::new(1);
        let vec_type = TypeId::new(2);

        let impl_def = create_test_impl(
            &mut arena,
            iterator_trait,
            vec_type,
            "Item",
            Type::Primitive(PrimitiveType::I32),
        );
        resolver.register_impl(iterator_trait, vec_type, Arc::new(impl_def));

        assert_eq!(resolver.impl_count(), 1);

        resolver.clear();

        assert_eq!(resolver.impl_count(), 0);
    }
}
