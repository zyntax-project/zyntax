//! Nominal Type System Checker
//!
//! Implements Java/C#/Kotlin style nominal typing where types are equal
//! only if they have the same name and are in the same inheritance hierarchy.

use crate::*;
// use crate::universal_type_system::*;
use crate::source::Span;
use crate::{arena::InternedString, TypeId};
use std::collections::{HashMap, HashSet, VecDeque};

/// Nominal type checker for inheritance-based type systems
pub struct NominalTypeChecker {
    /// Type registry containing all type definitions
    pub type_registry: HashMap<TypeId, TypeDefinition>,

    /// Trait/interface registry
    pub trait_registry: HashMap<TypeId, TraitDefinition>,

    /// Implementation registry (type -> implemented traits)
    pub impl_registry: HashMap<TypeId, Vec<ImplDefinition>>,

    /// Inheritance hierarchy cache
    pub inheritance_cache: HashMap<TypeId, HashSet<TypeId>>,

    /// Variance cache for generic parameters
    pub variance_cache: HashMap<TypeId, Vec<Variance>>,

    /// Virtual method table cache
    pub vtable_cache: HashMap<TypeId, VirtualMethodTable>,

    /// Interface dispatch table cache
    pub interface_dispatch_cache: HashMap<(TypeId, TypeId), InterfaceDispatchTable>,

    /// Type compatibility cache for performance
    pub compatibility_cache: HashMap<(Type, Type), bool>,
}

/// Type definition in the nominal system
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub id: TypeId,
    pub name: InternedString,
    pub kind: TypeKind,
    pub type_params: Vec<TypeParam>,
    pub super_type: Option<Type>,
    pub interfaces: Vec<Type>,
    pub fields: Vec<FieldDef>,
    pub methods: Vec<MethodSig>,
    pub constructors: Vec<ConstructorSig>,
    pub visibility: Visibility,
    pub is_abstract: bool,
    pub is_sealed: bool,
    pub is_final: bool,
    pub span: Span,
}

/// Different kinds of nominal types
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Class,
    Interface,
    Struct, // C# struct, Kotlin data class
    Enum,
    Record,     // C# record, Java record
    Delegate,   // C# delegate
    Annotation, // Java annotation
    Trait,      // Rust trait (when used nominally)
}

/// Trait/interface definition
#[derive(Debug, Clone)]
pub struct TraitDefinition {
    pub id: TypeId,
    pub name: InternedString,
    pub type_params: Vec<TypeParam>,
    pub super_traits: Vec<Type>,
    pub methods: Vec<MethodSig>,
    pub associated_types: Vec<AssociatedTypeDef>,
    pub default_implementations: HashMap<InternedString, DefaultImpl>,
    pub is_object_safe: bool,
    pub visibility: Visibility,
    pub span: Span,
}

/// Associated type definition
#[derive(Debug, Clone)]
pub struct AssociatedTypeDef {
    pub name: InternedString,
    pub bounds: Vec<TypeBound>,
    pub default: Option<Type>,
}

/// Default implementation for trait methods
#[derive(Debug, Clone)]
pub struct DefaultImpl {
    pub method_name: InternedString,
    pub body: MethodBody, // Would be actual AST in real implementation
}

/// Method body placeholder
#[derive(Debug, Clone)]
pub enum MethodBody {
    Abstract,
    Default,
    Native,
    Placeholder, // For this implementation
}

/// Virtual method table for a type
#[derive(Debug, Clone)]
pub struct VirtualMethodTable {
    /// Type this vtable belongs to
    pub type_id: TypeId,
    /// Virtual method slots
    pub methods: HashMap<InternedString, VirtualMethodSlot>,
    /// Parent vtable (for inheritance)
    pub parent: Option<Box<VirtualMethodTable>>,
}

/// A slot in the virtual method table
#[derive(Debug, Clone)]
pub struct VirtualMethodSlot {
    /// Method signature
    pub signature: MethodSig,
    /// Implementation type ID (which type provides this implementation)
    pub implementation_type: TypeId,
    /// Slot index for efficient dispatch
    pub slot_index: usize,
    /// Whether this method is virtual/overridable
    pub is_virtual: bool,
    /// Whether this method is abstract
    pub is_abstract: bool,
}

/// Interface dispatch table for efficient interface method calls
#[derive(Debug, Clone)]
pub struct InterfaceDispatchTable {
    /// Interface trait ID
    pub trait_id: TypeId,
    /// Implementing type ID
    pub type_id: TypeId,
    /// Method implementations
    pub method_implementations: HashMap<InternedString, MethodSig>,
}

/// Polymorphism information for method resolution
#[derive(Debug, Clone)]
pub struct PolymorphismInfo {
    /// The actual runtime type
    pub runtime_type: TypeId,
    /// The static type used for type checking
    pub static_type: Type,
    /// Available virtual methods
    pub virtual_methods: Vec<InternedString>,
    /// Available interface methods
    pub interface_methods: HashMap<TypeId, Vec<InternedString>>,
}

/// Implementation definition
#[derive(Debug, Clone)]
pub struct ImplDefinition {
    pub trait_id: TypeId,
    pub implementing_type: Type,
    pub type_params: Vec<TypeParam>,
    pub where_clause: Vec<TypeConstraint>,
    pub methods: Vec<MethodImpl>,
    pub associated_types: HashMap<InternedString, Type>,
    pub span: Span,
}

/// Method implementation
#[derive(Debug, Clone)]
pub struct MethodImpl {
    pub signature: MethodSig,
    pub body: MethodBody,
}

/// Constructor signature
#[derive(Debug, Clone)]
pub struct ConstructorSig {
    pub type_params: Vec<TypeParam>,
    pub params: Vec<ParamDef>,
    pub where_clause: Vec<TypeConstraint>,
    pub visibility: Visibility,
    pub span: Span,
}

/// Errors that can occur in nominal type checking
#[derive(Debug, Clone, PartialEq)]
pub enum NominalTypeError {
    /// Type not found in registry
    UnknownType { id: TypeId, span: Span },

    /// Trait not found in registry
    UnknownTrait { id: TypeId, span: Span },

    /// Circular inheritance detected
    CircularInheritance { types: Vec<TypeId>, span: Span },

    /// Type doesn't implement required trait
    TraitNotImplemented {
        ty: Type,
        trait_id: TypeId,
        span: Span,
    },

    /// Method not found in type
    MethodNotFound {
        ty: Type,
        method_name: InternedString,
        span: Span,
    },

    /// Invalid variance usage
    VarianceViolation {
        param: InternedString,
        expected: Variance,
        actual: Variance,
        span: Span,
    },

    /// Sealed class inheritance violation
    SealedInheritance {
        base_type: TypeId,
        derived_type: TypeId,
        span: Span,
    },

    /// Abstract method not implemented
    AbstractMethodNotImplemented {
        method: InternedString,
        in_type: TypeId,
        span: Span,
    },

    /// Interface segregation violation (too many unrelated methods)
    InterfaceSegregationViolation { interface: TypeId, span: Span },

    /// Virtual method override violation
    VirtualMethodOverrideError {
        method: InternedString,
        base_type: TypeId,
        derived_type: TypeId,
        reason: String,
        span: Span,
    },

    /// Multiple inheritance not allowed (for languages that don't support it)
    MultipleInheritanceNotAllowed {
        type_id: TypeId,
        parents: Vec<TypeId>,
        span: Span,
    },

    /// Diamond inheritance problem
    DiamondInheritance {
        type_id: TypeId,
        conflicting_types: Vec<TypeId>,
        span: Span,
    },

    /// Covariance/contravariance violation in method overrides
    CovarianceViolation {
        method: InternedString,
        param_or_return: String,
        expected: Type,
        actual: Type,
        span: Span,
    },

    /// Liskov substitution principle violation
    LiskovSubstitutionViolation {
        base_type: TypeId,
        derived_type: TypeId,
        reason: String,
        span: Span,
    },
}

impl NominalTypeChecker {
    pub fn new() -> Self {
        Self {
            type_registry: HashMap::new(),
            trait_registry: HashMap::new(),
            impl_registry: HashMap::new(),
            inheritance_cache: HashMap::new(),
            variance_cache: HashMap::new(),
            vtable_cache: HashMap::new(),
            interface_dispatch_cache: HashMap::new(),
            compatibility_cache: HashMap::new(),
        }
    }

    /// Register a new type definition
    pub fn register_type(&mut self, type_def: TypeDefinition) -> Result<(), NominalTypeError> {
        // Check for circular inheritance before registering
        if let Some(super_type) = &type_def.super_type {
            if let Type::Named { id: super_id, .. } = super_type {
                if self.would_create_cycle(type_def.id, *super_id)? {
                    return Err(NominalTypeError::CircularInheritance {
                        types: vec![type_def.id, *super_id],
                        span: type_def.span,
                    });
                }
            }
        }

        // Cache variance information
        let variances: Vec<Variance> = type_def
            .type_params
            .iter()
            .map(|param| param.variance)
            .collect();
        self.variance_cache.insert(type_def.id, variances);

        self.type_registry.insert(type_def.id, type_def);
        Ok(())
    }

    /// Register a new trait definition
    pub fn register_trait(&mut self, trait_def: TraitDefinition) {
        self.trait_registry.insert(trait_def.id, trait_def);
    }

    /// Register an implementation
    pub fn register_impl(&mut self, impl_def: ImplDefinition) {
        let implementing_type_id = match &impl_def.implementing_type {
            Type::Named { id, .. } => *id,
            _ => return, // Only named types can implement traits in nominal system
        };

        self.impl_registry
            .entry(implementing_type_id)
            .or_insert_with(Vec::new)
            .push(impl_def);
    }

    /// Check if one type is a subtype of another in the nominal system
    pub fn is_subtype(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<bool, NominalTypeError> {
        match (sub_type, super_type) {
            // Same type is always a subtype
            (a, b) if a == b => Ok(true),

            // Named type subtyping through inheritance
            (
                Type::Named {
                    id: sub_id,
                    type_args: sub_args,
                    ..
                },
                Type::Named {
                    id: super_id,
                    type_args: super_args,
                    ..
                },
            ) => {
                if sub_id == super_id {
                    // Same nominal type - check variance of type arguments
                    self.check_type_arg_variance(*sub_id, sub_args, super_args)
                } else {
                    // Different nominal types - check inheritance hierarchy
                    let is_subtype = self.is_in_inheritance_hierarchy(*sub_id, *super_id)?;
                    if is_subtype && !sub_args.is_empty() && !super_args.is_empty() {
                        // TODO: Handle variance in inheritance with generics
                        // This is complex and requires careful variance checking
                        Ok(true)
                    } else {
                        Ok(is_subtype)
                    }
                }
            }

            // Interface implementation checking
            (Type::Named { id: type_id, .. }, Type::Trait { id: trait_id, .. }) => {
                self.implements_trait(*type_id, *trait_id)
            }

            // Other cases delegate to structural/gradual checkers
            _ => Ok(false),
        }
    }

    /// Check if a type implements a trait
    pub fn implements_trait(
        &self,
        type_id: TypeId,
        trait_id: TypeId,
    ) -> Result<bool, NominalTypeError> {
        if let Some(impls) = self.impl_registry.get(&type_id) {
            Ok(impls.iter().any(|impl_def| impl_def.trait_id == trait_id))
        } else {
            Ok(false)
        }
    }

    /// Get all traits implemented by a type
    pub fn get_implemented_traits(&self, type_id: TypeId) -> Vec<TypeId> {
        if let Some(impls) = self.impl_registry.get(&type_id) {
            impls.iter().map(|impl_def| impl_def.trait_id).collect()
        } else {
            Vec::new()
        }
    }

    /// Check if one type is in the inheritance hierarchy of another
    fn is_in_inheritance_hierarchy(
        &mut self,
        sub_id: TypeId,
        super_id: TypeId,
    ) -> Result<bool, NominalTypeError> {
        // Check cache first
        if let Some(cached) = self.inheritance_cache.get(&sub_id) {
            return Ok(cached.contains(&super_id));
        }

        // Compute inheritance hierarchy
        let mut hierarchy = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(sub_id);

        while let Some(current_id) = queue.pop_front() {
            if current_id == super_id {
                self.inheritance_cache.insert(sub_id, hierarchy);
                return Ok(true);
            }

            if hierarchy.contains(&current_id) {
                continue; // Already processed
            }
            hierarchy.insert(current_id);

            // Get the type definition
            if let Some(type_def) = self.type_registry.get(&current_id) {
                // Add super type to queue
                if let Some(Type::Named { id: parent_id, .. }) = &type_def.super_type {
                    queue.push_back(*parent_id);
                }

                // Add interfaces to queue
                for interface in &type_def.interfaces {
                    if let Type::Named {
                        id: interface_id, ..
                    } = interface
                    {
                        queue.push_back(*interface_id);
                    }
                }
            }
        }

        self.inheritance_cache.insert(sub_id, hierarchy);
        Ok(false)
    }

    /// Check variance of type arguments
    fn check_type_arg_variance(
        &self,
        type_id: TypeId,
        sub_args: &[Type],
        super_args: &[Type],
    ) -> Result<bool, NominalTypeError> {
        if sub_args.len() != super_args.len() {
            return Ok(false);
        }

        let variances = self
            .variance_cache
            .get(&type_id)
            .cloned()
            .unwrap_or_default();

        for ((sub_arg, super_arg), variance) in
            sub_args.iter().zip(super_args.iter()).zip(variances.iter())
        {
            let compatible = match variance {
                Variance::Covariant => {
                    // T<A> <: T<B> if A <: B
                    self.is_subtype_recursive(sub_arg, super_arg)?
                }
                Variance::Contravariant => {
                    // T<A> <: T<B> if B <: A (reversed)
                    self.is_subtype_recursive(super_arg, sub_arg)?
                }
                Variance::Invariant => {
                    // T<A> <: T<B> if A = B
                    sub_arg == super_arg
                }
                Variance::Bivariant => {
                    // T<A> <: T<B> always (unsafe)
                    true
                }
            };

            if !compatible {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Recursive subtype checking helper (immutable — no inheritance cache updates)
    fn is_subtype_recursive(
        &self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<bool, NominalTypeError> {
        // Same type is always a subtype
        if sub_type == super_type {
            return Ok(true);
        }

        match (sub_type, super_type) {
            // Never is a subtype of everything (bottom type)
            (Type::Never, _) => Ok(true),

            // Everything is a subtype of Any (top type)
            (_, Type::Any) => Ok(true),

            // Primitive widening: i8 <: i16 <: i32 <: i64
            (Type::Primitive(sub_prim), Type::Primitive(super_prim)) => {
                use crate::type_registry::PrimitiveType;
                let widening_compatible = matches!(
                    (sub_prim, super_prim),
                    (
                        PrimitiveType::I8,
                        PrimitiveType::I16 | PrimitiveType::I32 | PrimitiveType::I64
                    ) | (PrimitiveType::I16, PrimitiveType::I32 | PrimitiveType::I64)
                        | (PrimitiveType::I32, PrimitiveType::I64)
                        | (
                            PrimitiveType::U8,
                            PrimitiveType::U16 | PrimitiveType::U32 | PrimitiveType::U64
                        )
                        | (PrimitiveType::U16, PrimitiveType::U32 | PrimitiveType::U64)
                        | (PrimitiveType::U32, PrimitiveType::U64)
                        | (PrimitiveType::F32, PrimitiveType::F64)
                );
                Ok(widening_compatible)
            }

            // Named type subtyping: check inheritance hierarchy (read-only)
            (
                Type::Named {
                    id: sub_id,
                    type_args: sub_args,
                    ..
                },
                Type::Named {
                    id: super_id,
                    type_args: super_args,
                    ..
                },
            ) => {
                if sub_id == super_id {
                    // Same nominal type — check variance of type arguments
                    self.check_type_arg_variance(*sub_id, sub_args, super_args)
                } else {
                    // Check cached hierarchy (read-only — no BFS here to avoid &mut self)
                    if let Some(cached) = self.inheritance_cache.get(sub_id) {
                        Ok(cached.contains(super_id))
                    } else {
                        Ok(false)
                    }
                }
            }

            // Named implements trait
            (Type::Named { id: type_id, .. }, Type::Trait { id: trait_id, .. }) => {
                self.implements_trait(*type_id, *trait_id)
            }

            // Optional subtyping: T <: Optional(T)
            (inner, Type::Optional(opt_inner)) => self.is_subtype_recursive(inner, opt_inner),

            // Array covariance: Array<A> <: Array<B> if A <: B
            (
                Type::Array {
                    element_type: sub_elem,
                    ..
                },
                Type::Array {
                    element_type: super_elem,
                    ..
                },
            ) => self.is_subtype_recursive(sub_elem, super_elem),

            // Reference subtyping: shared references are covariant
            (
                Type::Reference {
                    ty: sub_inner,
                    mutability: sub_mut,
                    ..
                },
                Type::Reference {
                    ty: super_inner,
                    mutability: super_mut,
                    ..
                },
            ) => {
                use crate::Mutability;
                match (sub_mut, super_mut) {
                    // &mut T <: &T (mutable ref can be used as immutable)
                    (Mutability::Mutable, Mutability::Immutable) => {
                        self.is_subtype_recursive(sub_inner, super_inner)
                    }
                    // &T <: &T (same mutability, covariant)
                    (Mutability::Immutable, Mutability::Immutable) => {
                        self.is_subtype_recursive(sub_inner, super_inner)
                    }
                    // &mut T <: &mut T (invariant for mutable)
                    (Mutability::Mutable, Mutability::Mutable) => Ok(sub_inner == super_inner),
                    // &T </: &mut T
                    _ => Ok(false),
                }
            }

            // Tuple subtyping: element-wise
            (Type::Tuple(sub_elems), Type::Tuple(super_elems)) => {
                if sub_elems.len() != super_elems.len() {
                    return Ok(false);
                }
                for (sub_e, super_e) in sub_elems.iter().zip(super_elems.iter()) {
                    if !self.is_subtype_recursive(sub_e, super_e)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            // Function subtyping: contravariant params, covariant return
            (
                Type::Function {
                    params: sub_params,
                    return_type: sub_ret,
                    ..
                },
                Type::Function {
                    params: super_params,
                    return_type: super_ret,
                    ..
                },
            ) => {
                if sub_params.len() != super_params.len() {
                    return Ok(false);
                }
                // Return type: covariant
                if !self.is_subtype_recursive(sub_ret, super_ret)? {
                    return Ok(false);
                }
                // Parameters: contravariant
                for (sub_p, super_p) in sub_params.iter().zip(super_params.iter()) {
                    if !self.is_subtype_recursive(&super_p.ty, &sub_p.ty)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            _ => Ok(false),
        }
    }

    /// Check if adding an inheritance relationship would create a cycle
    fn would_create_cycle(
        &self,
        derived_id: TypeId,
        base_id: TypeId,
    ) -> Result<bool, NominalTypeError> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(base_id);

        while let Some(current_id) = queue.pop_front() {
            if current_id == derived_id {
                return Ok(true); // Cycle detected
            }

            if !visited.insert(current_id) {
                continue; // Already visited
            }

            // Get the type definition and check its super types
            if let Some(type_def) = self.type_registry.get(&current_id) {
                if let Some(Type::Named { id: parent_id, .. }) = &type_def.super_type {
                    queue.push_back(*parent_id);
                }
            }
        }

        Ok(false)
    }

    /// Resolve method call on a nominal type
    pub fn resolve_method(
        &self,
        receiver_type: &Type,
        method_name: InternedString,
        type_args: &[Type],
    ) -> Result<Option<ResolvedMethod>, NominalTypeError> {
        match receiver_type {
            Type::Named {
                id: type_id,
                type_args: receiver_args,
                ..
            } => {
                // First check intrinsic methods
                if let Some(type_def) = self.type_registry.get(type_id) {
                    for method in &type_def.methods {
                        if method.name == method_name {
                            return Ok(Some(ResolvedMethod {
                                signature: method.clone(),
                                receiver_type: receiver_type.clone(),
                                instantiated_type_args: type_args.to_vec(),
                                source: MethodSource::Intrinsic,
                            }));
                        }
                    }

                    // Then check inherited methods
                    if let Some(super_type) = &type_def.super_type {
                        if let Ok(Some(method)) =
                            self.resolve_method(super_type, method_name, type_args)
                        {
                            return Ok(Some(method));
                        }
                    }

                    // Finally check interface methods
                    for interface in &type_def.interfaces {
                        if let Ok(Some(method)) =
                            self.resolve_method(interface, method_name, type_args)
                        {
                            return Ok(Some(method));
                        }
                    }
                }

                // Check trait implementations
                if let Some(impls) = self.impl_registry.get(type_id) {
                    for impl_def in impls {
                        for method_impl in &impl_def.methods {
                            if method_impl.signature.name == method_name {
                                return Ok(Some(ResolvedMethod {
                                    signature: method_impl.signature.clone(),
                                    receiver_type: receiver_type.clone(),
                                    instantiated_type_args: type_args.to_vec(),
                                    source: MethodSource::TraitImpl(impl_def.trait_id),
                                }));
                            }
                        }
                    }
                }

                Ok(None)
            }

            Type::Trait { id: trait_id, .. } => {
                if let Some(trait_def) = self.trait_registry.get(trait_id) {
                    for method in &trait_def.methods {
                        if method.name == method_name {
                            return Ok(Some(ResolvedMethod {
                                signature: method.clone(),
                                receiver_type: receiver_type.clone(),
                                instantiated_type_args: type_args.to_vec(),
                                source: MethodSource::Trait(*trait_id),
                            }));
                        }
                    }
                }
                Ok(None)
            }

            _ => Ok(None),
        }
    }

    /// Validate that all abstract methods are implemented
    pub fn validate_concrete_type(&self, type_id: TypeId) -> Result<(), Vec<NominalTypeError>> {
        let mut errors = Vec::new();

        if let Some(type_def) = self.type_registry.get(&type_id) {
            if type_def.is_abstract {
                return Ok(()); // Abstract types don't need to implement everything
            }

            // Collect all abstract methods that need implementation
            let mut required_methods = HashSet::new();
            let abstract_methods = self.collect_abstract_methods(type_id);
            required_methods.extend(abstract_methods);

            // Check that all required methods are implemented
            let implemented_methods: HashSet<InternedString> =
                type_def.methods.iter().map(|m| m.name).collect();

            for required_method in required_methods {
                if !implemented_methods.contains(&required_method) {
                    errors.push(NominalTypeError::AbstractMethodNotImplemented {
                        method: required_method,
                        in_type: type_id,
                        span: type_def.span,
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Resolved method information
#[derive(Debug, Clone)]
pub struct ResolvedMethod {
    pub signature: MethodSig,
    pub receiver_type: Type,
    pub instantiated_type_args: Vec<Type>,
    pub source: MethodSource,
}

/// Source of a resolved method
#[derive(Debug, Clone, PartialEq)]
pub enum MethodSource {
    Intrinsic,         // Method defined directly in the type
    Inherited(TypeId), // Method inherited from base class
    TraitImpl(TypeId), // Method from trait implementation
    Trait(TypeId),     // Method from trait definition
    Extension,         // Extension method
}

impl NominalTypeChecker {
    // === Enhanced Nominal Type Checking Methods ===

    /// Build virtual method table for a type
    pub fn build_virtual_method_table(
        &mut self,
        type_id: TypeId,
    ) -> Result<VirtualMethodTable, NominalTypeError> {
        if let Some(vtable) = self.vtable_cache.get(&type_id).cloned() {
            return Ok(vtable);
        }

        let type_def =
            self.type_registry
                .get(&type_id)
                .cloned()
                .ok_or(NominalTypeError::UnknownType {
                    id: type_id,
                    span: crate::source::Span::new(0, 0),
                })?;

        let mut vtable = VirtualMethodTable {
            type_id,
            methods: HashMap::new(),
            parent: None,
        };

        // Build parent vtable first if there's inheritance
        if let Some(super_type) = &type_def.super_type {
            if let Type::Named { id: parent_id, .. } = super_type {
                let parent_vtable = self.build_virtual_method_table(*parent_id)?;
                vtable.parent = Some(Box::new(parent_vtable));
            }
        }

        // Add methods from this type
        let mut slot_index = 0;
        for method in &type_def.methods {
            let slot = VirtualMethodSlot {
                signature: method.clone(),
                implementation_type: type_id,
                slot_index,
                is_virtual: !method.is_static,
                is_abstract: false, // Would need to check method body or flags
            };
            vtable.methods.insert(method.name, slot);
            slot_index += 1;
        }

        // Cache the vtable
        self.vtable_cache.insert(type_id, vtable.clone());
        Ok(vtable)
    }

    /// Build interface dispatch table for efficient interface method resolution
    pub fn build_interface_dispatch_table(
        &mut self,
        type_id: TypeId,
        trait_id: TypeId,
    ) -> Result<InterfaceDispatchTable, NominalTypeError> {
        let key = (type_id, trait_id);
        if let Some(table) = self.interface_dispatch_cache.get(&key).cloned() {
            return Ok(table);
        }

        let mut table = InterfaceDispatchTable {
            trait_id,
            type_id,
            method_implementations: HashMap::new(),
        };

        // Find implementations for this type-trait combination
        if let Some(impls) = self.impl_registry.get(&type_id) {
            for impl_def in impls {
                if impl_def.trait_id == trait_id {
                    for method_impl in &impl_def.methods {
                        table
                            .method_implementations
                            .insert(method_impl.signature.name, method_impl.signature.clone());
                    }
                    break;
                }
            }
        }

        // Cache the table
        self.interface_dispatch_cache.insert(key, table.clone());
        Ok(table)
    }

    /// Check virtual method override validity
    pub fn check_virtual_method_override(
        &mut self,
        base_method: &MethodSig,
        derived_method: &MethodSig,
        base_type: TypeId,
        derived_type: TypeId,
    ) -> Result<(), NominalTypeError> {
        // Check method signature compatibility
        if base_method.name != derived_method.name {
            return Err(NominalTypeError::VirtualMethodOverrideError {
                method: base_method.name,
                base_type,
                derived_type,
                reason: "Method name mismatch".to_string(),
                span: derived_method.span,
            });
        }

        // Check parameter count
        if base_method.params.len() != derived_method.params.len() {
            return Err(NominalTypeError::VirtualMethodOverrideError {
                method: base_method.name,
                base_type,
                derived_type,
                reason: "Parameter count mismatch".to_string(),
                span: derived_method.span,
            });
        }

        // Check parameter types (contravariance)
        for (base_param, derived_param) in
            base_method.params.iter().zip(derived_method.params.iter())
        {
            if !self.is_contravariant_compatible(&derived_param.ty, &base_param.ty) {
                return Err(NominalTypeError::CovarianceViolation {
                    method: base_method.name,
                    param_or_return: "parameter".to_string(),
                    expected: base_param.ty.clone(),
                    actual: derived_param.ty.clone(),
                    span: derived_method.span,
                });
            }
        }

        // Check return type (covariance)
        if !self.is_covariant_compatible(&derived_method.return_type, &base_method.return_type) {
            return Err(NominalTypeError::CovarianceViolation {
                method: base_method.name,
                param_or_return: "return type".to_string(),
                expected: base_method.return_type.clone(),
                actual: derived_method.return_type.clone(),
                span: derived_method.span,
            });
        }

        Ok(())
    }

    /// Check covariant compatibility (return types)
    fn is_covariant_compatible(&mut self, derived: &Type, base: &Type) -> bool {
        // Derived type must be a subtype of base type
        self.is_subtype(derived, base).unwrap_or(false)
    }

    /// Check contravariant compatibility (parameter types)
    fn is_contravariant_compatible(&mut self, derived: &Type, base: &Type) -> bool {
        // Base type must be a subtype of derived type (reversed for contravariance)
        self.is_subtype(base, derived).unwrap_or(false)
    }

    /// Check Liskov Substitution Principle compliance
    pub fn check_liskov_substitution(
        &mut self,
        base_type: TypeId,
        derived_type: TypeId,
    ) -> Result<(), NominalTypeError> {
        let base_def =
            self.type_registry
                .get(&base_type)
                .cloned()
                .ok_or(NominalTypeError::UnknownType {
                    id: base_type,
                    span: crate::source::Span::new(0, 0),
                })?;

        let derived_def = self.type_registry.get(&derived_type).cloned().ok_or(
            NominalTypeError::UnknownType {
                id: derived_type,
                span: crate::source::Span::new(0, 0),
            },
        )?;

        // Check that derived type can substitute base type
        // 1. All public methods of base must be available in derived
        for base_method in &base_def.methods {
            if base_method.visibility == Visibility::Public {
                let found = derived_def.methods.iter().any(|derived_method| {
                    derived_method.name == base_method.name
                        && self
                            .check_virtual_method_override(
                                base_method,
                                derived_method,
                                base_type,
                                derived_type,
                            )
                            .is_ok()
                });

                if !found {
                    return Err(NominalTypeError::LiskovSubstitutionViolation {
                        base_type,
                        derived_type,
                        reason: format!("Method {:?} not properly overridden", base_method.name),
                        span: derived_def.span,
                    });
                }
            }
        }

        // 2. Derived type must not strengthen preconditions
        // 3. Derived type must not weaken postconditions
        // (These would require more sophisticated analysis)

        Ok(())
    }

    /// Resolve polymorphic method call
    pub fn resolve_polymorphic_method(
        &self,
        receiver_type: &Type,
        method_name: InternedString,
        is_virtual_call: bool,
    ) -> Result<Option<(MethodSig, MethodSource)>, NominalTypeError> {
        match receiver_type {
            Type::Named { id: type_id, .. } => {
                // First check for direct methods
                if let Some(type_def) = self.type_registry.get(type_id) {
                    for method in &type_def.methods {
                        if method.name == method_name {
                            return Ok(Some((method.clone(), MethodSource::Intrinsic)));
                        }
                    }

                    // Check inherited methods if virtual call
                    if is_virtual_call {
                        if let Some(vtable) = self.vtable_cache.get(type_id) {
                            if let Some(slot) = vtable.methods.get(&method_name) {
                                return Ok(Some((
                                    slot.signature.clone(),
                                    MethodSource::Inherited(slot.implementation_type),
                                )));
                            }
                        }
                    }

                    // Check interface methods
                    for interface in &type_def.interfaces {
                        if let Type::Named {
                            id: interface_id, ..
                        } = interface
                        {
                            // Note: This assumes interface_id represents a trait
                            // In a real implementation, we'd need proper type/trait ID management
                            let trait_id = TypeId::new(interface_id.as_u32());
                            if let Some(trait_def) = self.trait_registry.get(&trait_id) {
                                for method in &trait_def.methods {
                                    if method.name == method_name {
                                        return Ok(Some((
                                            method.clone(),
                                            MethodSource::TraitImpl(trait_id),
                                        )));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    /// Check for diamond inheritance problems
    pub fn check_diamond_inheritance(&self, type_id: TypeId) -> Result<(), NominalTypeError> {
        let mut visited = HashSet::new();
        let mut path = Vec::new();
        let mut conflicts = Vec::new();

        self.check_diamond_inheritance_recursive(type_id, &mut visited, &mut path, &mut conflicts)?;

        if !conflicts.is_empty() {
            return Err(NominalTypeError::DiamondInheritance {
                type_id,
                conflicting_types: conflicts,
                span: crate::source::Span::new(0, 0),
            });
        }

        Ok(())
    }

    fn check_diamond_inheritance_recursive(
        &self,
        current_type: TypeId,
        visited: &mut HashSet<TypeId>,
        path: &mut Vec<TypeId>,
        conflicts: &mut Vec<TypeId>,
    ) -> Result<(), NominalTypeError> {
        if path.contains(&current_type) {
            conflicts.push(current_type);
            return Ok(());
        }

        if visited.contains(&current_type) {
            return Ok(());
        }

        visited.insert(current_type);
        path.push(current_type);

        if let Some(type_def) = self.type_registry.get(&current_type) {
            // Check superclass
            if let Some(super_type) = &type_def.super_type {
                if let Type::Named { id: super_id, .. } = super_type {
                    self.check_diamond_inheritance_recursive(*super_id, visited, path, conflicts)?;
                }
            }

            // Check interfaces (potential for diamond problems)
            for interface in &type_def.interfaces {
                if let Type::Named {
                    id: interface_id, ..
                } = interface
                {
                    self.check_diamond_inheritance_recursive(
                        *interface_id,
                        visited,
                        path,
                        conflicts,
                    )?;
                }
            }
        }

        path.pop();
        Ok(())
    }

    /// Get polymorphism information for a type
    pub fn get_polymorphism_info(
        &self,
        type_id: TypeId,
    ) -> Result<PolymorphismInfo, NominalTypeError> {
        let type_def = self
            .type_registry
            .get(&type_id)
            .ok_or(NominalTypeError::UnknownType {
                id: type_id,
                span: crate::source::Span::new(0, 0),
            })?;

        let static_type = Type::Named {
            id: type_id,
            type_args: vec![], // Would need actual type args in real usage
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        };

        // Collect virtual methods
        let virtual_methods: Vec<InternedString> = type_def
            .methods
            .iter()
            .filter(|m| !m.is_static)
            .map(|m| m.name)
            .collect();

        // Collect interface methods
        let mut interface_methods = HashMap::new();
        for interface in &type_def.interfaces {
            if let Type::Named { id: type_id, .. } = interface {
                // Convert TypeId to TypeId for trait lookup
                let trait_id = TypeId::new(type_id.as_u32());
                if let Some(trait_def) = self.trait_registry.get(&trait_id) {
                    let methods: Vec<InternedString> =
                        trait_def.methods.iter().map(|m| m.name).collect();
                    interface_methods.insert(trait_id, methods);
                }
            }
        }

        Ok(PolymorphismInfo {
            runtime_type: type_id,
            static_type,
            virtual_methods,
            interface_methods,
        })
    }

    /// Validate type hierarchy for correctness
    pub fn validate_type_hierarchy(
        &mut self,
        type_id: TypeId,
    ) -> Result<(), Vec<NominalTypeError>> {
        let mut errors = Vec::new();

        // Check diamond inheritance
        if let Err(e) = self.check_diamond_inheritance(type_id) {
            errors.push(e);
        }

        // Check abstract method implementations
        if let Some(type_def) = self.type_registry.get(&type_id) {
            if !type_def.is_abstract {
                // Check that all abstract methods are implemented
                let abstract_methods = self.collect_abstract_methods(type_id);
                for abstract_method in abstract_methods {
                    if !self.has_concrete_implementation(type_id, abstract_method) {
                        errors.push(NominalTypeError::AbstractMethodNotImplemented {
                            method: abstract_method,
                            in_type: type_id,
                            span: type_def.span,
                        });
                    }
                }
            }
        }

        // Check Liskov substitution principle
        if let Some(type_def) = self.type_registry.get(&type_id) {
            if let Some(super_type) = &type_def.super_type {
                if let Type::Named { id: super_id, .. } = super_type {
                    if let Err(e) = self.check_liskov_substitution(*super_id, type_id) {
                        errors.push(e);
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn collect_abstract_methods(&self, type_id: TypeId) -> Vec<InternedString> {
        let mut abstract_methods = Vec::new();
        let mut visited = HashSet::new();
        self.collect_abstract_methods_recursive(type_id, &mut abstract_methods, &mut visited);
        abstract_methods
    }

    fn collect_abstract_methods_recursive(
        &self,
        type_id: TypeId,
        methods: &mut Vec<InternedString>,
        visited: &mut HashSet<TypeId>,
    ) {
        if visited.contains(&type_id) {
            return;
        }
        visited.insert(type_id);

        if let Some(type_def) = self.type_registry.get(&type_id) {
            // Add abstract methods from this type
            for method in &type_def.methods {
                // For interface methods, assume they're abstract unless they have default impls
                let is_abstract = type_def.kind == TypeKind::Interface;
                if is_abstract && !methods.contains(&method.name) {
                    methods.push(method.name);
                }
            }

            // Recursively check superclass
            if let Some(super_type) = &type_def.super_type {
                if let Type::Named { id: super_id, .. } = super_type {
                    self.collect_abstract_methods_recursive(*super_id, methods, visited);
                }
            }

            // Check interfaces for abstract methods
            for interface in &type_def.interfaces {
                if let Type::Named {
                    id: interface_id, ..
                } = interface
                {
                    self.collect_abstract_methods_recursive(*interface_id, methods, visited);
                }
            }
        }
    }

    fn has_concrete_implementation(&self, type_id: TypeId, method_name: InternedString) -> bool {
        if let Some(type_def) = self.type_registry.get(&type_id) {
            // Check if this type has a concrete implementation
            for method in &type_def.methods {
                if method.name == method_name {
                    // Assume non-interface methods are concrete
                    let is_concrete = type_def.kind != TypeKind::Interface;
                    if is_concrete {
                        return true;
                    }
                }
            }

            // Check inherited implementations
            if let Some(super_type) = &type_def.super_type {
                if let Type::Named { id: super_id, .. } = super_type {
                    return self.has_concrete_implementation(*super_id, method_name);
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::AstArena;

    #[test]
    fn test_basic_nominal_subtyping() {
        let mut checker = NominalTypeChecker::new();
        let mut arena = AstArena::new();

        // Create Object type
        let object_id = TypeId::next();
        let object_type = TypeDefinition {
            id: object_id,
            name: arena.intern_string("Object"),
            kind: TypeKind::Class,
            type_params: vec![],
            super_type: None,
            interfaces: vec![],
            fields: vec![],
            methods: vec![],
            constructors: vec![],
            visibility: Visibility::Public,
            is_abstract: false,
            is_sealed: false,
            is_final: false,
            span: crate::source::Span::new(0, 0),
        };

        checker.register_type(object_type).unwrap();

        // Create String type that inherits from Object
        let string_id = TypeId::next();
        let string_type = TypeDefinition {
            id: string_id,
            name: arena.intern_string("String"),
            kind: TypeKind::Class,
            type_params: vec![],
            super_type: Some(Type::Named {
                id: object_id,
                type_args: vec![],
                variance: vec![],
                const_args: vec![],
                nullability: NullabilityKind::NonNull,
            }),
            interfaces: vec![],
            fields: vec![],
            methods: vec![],
            constructors: vec![],
            visibility: Visibility::Public,
            is_abstract: false,
            is_sealed: false,
            is_final: false,
            span: crate::source::Span::new(0, 0),
        };

        checker.register_type(string_type).unwrap();

        // Test subtyping
        let string_instance = Type::Named {
            id: string_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        };

        let object_instance = Type::Named {
            id: object_id,
            type_args: vec![],
            variance: vec![],
            const_args: vec![],
            nullability: NullabilityKind::NonNull,
        };

        assert!(checker
            .is_subtype(&string_instance, &object_instance)
            .unwrap());
        assert!(!checker
            .is_subtype(&object_instance, &string_instance)
            .unwrap());
    }
}
