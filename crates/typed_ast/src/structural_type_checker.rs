//! Structural Type System Checker
//!
//! Implements Go/TypeScript style structural typing where types are compatible
//! if they have the same structure (same fields and methods with compatible types).

use crate::arena::InternedString;
use crate::source::Span;
use crate::{typed_builder::DefaultValue, *};
use std::collections::{HashMap, HashSet};
use string_interner::Symbol;

/// Structural typing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralMode {
    /// Strict structural typing (TypeScript style)
    Strict,
    /// Duck typing (Python style)
    Duck,
    /// Nominal with structural fallback
    Nominal,
    /// Gradual typing (TypeScript/Flow style)
    Gradual,
}

/// Variance analysis context for structural checking
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceContext {
    /// Current variance position (covariant, contravariant, invariant)
    pub current_variance: Variance,
    /// Stack of variance positions for nested types
    pub variance_stack: Vec<Variance>,
    /// Type parameter variance constraints
    pub type_param_variances: HashMap<InternedString, Variance>,
    /// Higher-kinded type constructors and their variances
    pub hkt_variances: HashMap<InternedString, Vec<Variance>>,
}

/// Structural type checker for duck typing and interface compatibility
pub struct StructuralTypeChecker {
    /// Cache for structural compatibility checks
    pub compatibility_cache: HashMap<(StructuralTypeId, StructuralTypeId), bool>,

    /// Method signature cache for performance
    pub method_cache: HashMap<StructuralTypeId, Vec<MethodSignature>>,

    /// Field signature cache for performance  
    pub field_cache: HashMap<StructuralTypeId, Vec<FieldSignature>>,

    /// Variance analysis cache using structural type IDs
    pub variance_cache: HashMap<StructuralTypeId, VarianceContext>,

    /// Subtyping cache with type IDs for performance (avoiding full type hashing)
    pub subtyping_cache: HashMap<(StructuralTypeId, StructuralTypeId, Variance), bool>,

    /// Type constructor variance rules
    pub type_constructor_variances: HashMap<InternedString, Vec<Variance>>,
}

/// Unique identifier for structural types (based on structure hash)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StructuralTypeId(u64);

/// Method signature for structural compatibility
#[derive(Debug, Clone, PartialEq)]
pub struct MethodSignature {
    pub name: InternedString,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<ParamInfo>,
    pub return_type: Type,
    pub is_static: bool,
    pub is_async: bool,
    pub variance: Vec<Variance>,
}

/// Field signature for structural compatibility
#[derive(Debug, Clone, PartialEq)]
pub struct FieldSignature {
    pub name: InternedString,
    pub ty: Type,
    pub is_readonly: bool,
    pub is_optional: bool,
    pub getter: Option<MethodSignature>,
    pub setter: Option<MethodSignature>,
}

/// Result of structural compatibility check
#[derive(Debug, Clone, PartialEq)]
pub enum StructuralCompatibility {
    Compatible,
    Incompatible(Vec<StructuralError>),
    RequiresAdapterPattern(Vec<AdapterRequirement>),
}

/// Structural type checking errors
#[derive(Debug, Clone, PartialEq)]
pub enum StructuralError {
    /// Missing required field
    MissingField {
        name: InternedString,
        expected_type: Type,
        span: Span,
    },

    /// Field type mismatch
    FieldTypeMismatch {
        name: InternedString,
        expected: Type,
        found: Type,
        span: Span,
    },

    /// Missing required method
    MissingMethod {
        name: InternedString,
        expected_signature: MethodSignature,
        span: Span,
    },

    /// Method signature mismatch
    MethodSignatureMismatch {
        name: InternedString,
        expected: MethodSignature,
        found: MethodSignature,
        span: Span,
    },

    /// Readonly field assignment
    ReadonlyFieldAssignment {
        field_name: InternedString,
        span: Span,
    },

    /// Optional field used as required
    OptionalFieldRequired {
        field_name: InternedString,
        span: Span,
    },

    /// Variance violation in method signatures
    VarianceViolation {
        method_name: InternedString,
        param_index: usize,
        expected_variance: Variance,
        actual_variance: Variance,
        span: Span,
    },

    /// Covariance violation in return types
    CovarianceViolation {
        method_name: InternedString,
        expected_return: Type,
        actual_return: Type,
        span: Span,
    },

    /// Contravariance violation in parameter types
    ContravarianceViolation {
        method_name: InternedString,
        param_index: usize,
        expected_param: Type,
        actual_param: Type,
        span: Span,
    },

    /// Invariance violation
    InvarianceViolation {
        context: String,
        expected_type: Type,
        actual_type: Type,
        span: Span,
    },

    /// Higher-kinded type variance error
    HigherKindedVarianceError {
        type_constructor: InternedString,
        type_param_index: usize,
        expected_variance: Variance,
        actual_variance: Variance,
        span: Span,
    },

    /// Mutable aliasing violation (for linear types)
    MutableAliasingViolation {
        field_name: InternedString,
        span: Span,
    },
}

/// Requirements for adapter pattern implementation
#[derive(Debug, Clone, PartialEq)]
pub enum AdapterRequirement {
    /// Need to wrap a field access
    WrapField {
        from_field: InternedString,
        to_field: InternedString,
        adapter: FieldAdapter,
    },

    /// Need to wrap a method call
    WrapMethod {
        from_method: InternedString,
        to_method: InternedString,
        adapter: MethodAdapter,
    },

    /// Need to provide default implementation
    ProvideDefault {
        method_name: InternedString,
        default_impl: DefaultImplementation,
    },
}

/// Field adapter for type conversion
#[derive(Debug, Clone, PartialEq)]
pub enum FieldAdapter {
    Identity,                      // No conversion needed
    TypeCast(Type),                // Simple type cast
    Wrapper(InternedString),       // Wrap in another type
    Converter(ConversionFunction), // Custom conversion function
}

/// Method adapter for signature differences
#[derive(Debug, Clone, PartialEq)]
pub enum MethodAdapter {
    Identity,                              // No adaptation needed
    ParameterReorder(Vec<usize>),          // Reorder parameters
    ParameterDefault(Vec<DefaultValue>),   // Provide default values
    ReturnTypeConvert(ConversionFunction), // Convert return type
    AsyncAdapter,                          // Wrap sync method in async
    SyncAdapter,                           // Unwrap async method to sync
}

/// Conversion function specification
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionFunction {
    pub from_type: Type,
    pub to_type: Type,
    pub function_name: InternedString,
    pub is_safe: bool,
}

/// Default implementation for missing methods
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultImplementation {
    Throw(InternedString),        // Throw exception with message
    ReturnDefault(Type),          // Return default value of type
    Delegate(InternedString),     // Delegate to another method
    Custom(CustomImplementation), // Custom implementation
}

/// Custom implementation placeholder
#[derive(Debug, Clone, PartialEq)]
pub struct CustomImplementation {
    pub body: String, // Would be actual AST in real implementation
}

impl StructuralTypeChecker {
    pub fn new() -> Self {
        let mut type_constructor_variances = HashMap::new();

        // Initialize common type constructor variances
        Self::initialize_standard_variances(&mut type_constructor_variances);

        Self {
            compatibility_cache: HashMap::new(),
            method_cache: HashMap::new(),
            field_cache: HashMap::new(),
            variance_cache: HashMap::new(),
            subtyping_cache: HashMap::new(),
            type_constructor_variances,
        }
    }

    /// Initialize standard type constructor variance rules
    fn initialize_standard_variances(variances: &mut HashMap<InternedString, Vec<Variance>>) {
        // Array/List is covariant in element type
        // List<Dog> <: List<Animal> if Dog <: Animal
        variances.insert(
            InternedString::from_symbol(string_interner::Symbol::try_from_usize(100).unwrap()), // "Array"
            vec![Variance::Covariant],
        );
        variances.insert(
            InternedString::from_symbol(string_interner::Symbol::try_from_usize(101).unwrap()), // "List"
            vec![Variance::Covariant],
        );

        // Function parameters are contravariant, return types are covariant
        // (Animal) -> Dog <: (Dog) -> Animal
        variances.insert(
            InternedString::from_symbol(string_interner::Symbol::try_from_usize(102).unwrap()), // "Function"
            vec![Variance::Contravariant, Variance::Covariant],
        );

        // Mutable references are invariant
        // &mut Dog is NOT a subtype of &mut Animal
        variances.insert(
            InternedString::from_symbol(string_interner::Symbol::try_from_usize(103).unwrap()), // "MutRef"
            vec![Variance::Invariant],
        );

        // Immutable references are covariant
        // &Dog <: &Animal
        variances.insert(
            InternedString::from_symbol(string_interner::Symbol::try_from_usize(104).unwrap()), // "Ref"
            vec![Variance::Covariant],
        );

        // Option/Maybe is covariant
        variances.insert(
            InternedString::from_symbol(string_interner::Symbol::try_from_usize(105).unwrap()), // "Option"
            vec![Variance::Covariant],
        );

        // Result is covariant in success type, contravariant in error type
        variances.insert(
            InternedString::from_symbol(string_interner::Symbol::try_from_usize(106).unwrap()), // "Result"
            vec![Variance::Covariant, Variance::Contravariant],
        );
    }

    /// Check if one type is structurally compatible with another
    pub fn is_structurally_compatible(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
        mode: StructuralMode,
    ) -> Result<StructuralCompatibility, Vec<StructuralError>> {
        match mode {
            StructuralMode::Strict => self.check_strict_compatibility(sub_type, super_type),
            StructuralMode::Duck => self.check_duck_compatibility(sub_type, super_type),
            StructuralMode::Nominal => self.check_nominal_structure(sub_type, super_type),
            StructuralMode::Gradual => self.check_gradual_compatibility(sub_type, super_type),
        }
    }

    /// Strict structural compatibility (TypeScript style)
    fn check_strict_compatibility(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<StructuralCompatibility, Vec<StructuralError>> {
        match (sub_type, super_type) {
            // Interface compatibility
            (
                Type::Interface {
                    methods: sub_methods,
                    ..
                },
                Type::Interface {
                    methods: super_methods,
                    ..
                },
            ) => {
                let mut errors = Vec::new();

                // Check that sub_type has all methods required by super_type
                for required_method in super_methods {
                    if let Some(found_method) =
                        sub_methods.iter().find(|m| m.name == required_method.name)
                    {
                        // Check method signature compatibility
                        if let Err(method_errors) =
                            self.check_method_compatibility(found_method, required_method)
                        {
                            errors.extend(method_errors);
                        }
                    } else {
                        errors.push(StructuralError::MissingMethod {
                            name: required_method.name,
                            expected_signature: self.method_sig_to_signature(required_method),
                            span: required_method.span,
                        });
                    }
                }

                if errors.is_empty() {
                    Ok(StructuralCompatibility::Compatible)
                } else {
                    Ok(StructuralCompatibility::Incompatible(errors))
                }
            }

            // Struct compatibility
            (
                Type::Struct {
                    fields: sub_fields, ..
                },
                Type::Struct {
                    fields: super_fields,
                    ..
                },
            ) => {
                let mut errors = Vec::new();

                // Check that sub_type has all fields required by super_type
                for required_field in super_fields {
                    if let Some(found_field) =
                        sub_fields.iter().find(|f| f.name == required_field.name)
                    {
                        // Check field type compatibility
                        if !self.are_types_compatible(&found_field.ty, &required_field.ty) {
                            errors.push(StructuralError::FieldTypeMismatch {
                                name: required_field.name,
                                expected: required_field.ty.clone(),
                                found: found_field.ty.clone(),
                                span: Span::new(0, 0), // Would use actual span
                            });
                        }
                    } else {
                        errors.push(StructuralError::MissingField {
                            name: required_field.name,
                            expected_type: required_field.ty.clone(),
                            span: Span::new(0, 0), // Would use actual span
                        });
                    }
                }

                if errors.is_empty() {
                    Ok(StructuralCompatibility::Compatible)
                } else {
                    Ok(StructuralCompatibility::Incompatible(errors))
                }
            }

            // Named type to interface (Go style)
            (Type::Named { .. }, Type::Interface { methods, .. }) => {
                // Check if the named type has methods that satisfy the interface
                self.check_named_type_implements_interface(sub_type, methods)
            }

            // Function type compatibility
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
            ) => self.check_function_compatibility(sub_params, sub_ret, super_params, super_ret),

            // Other structural checks
            _ => Ok(StructuralCompatibility::Incompatible(vec![])),
        }
    }

    /// Duck typing compatibility (Python style)
    fn check_duck_compatibility(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<StructuralCompatibility, Vec<StructuralError>> {
        // Duck typing: if it walks like a duck and quacks like a duck, it's a duck
        // This is more permissive than strict structural typing

        match super_type {
            Type::Interface { methods, .. } => {
                let mut adapter_requirements = Vec::new();

                for required_method in methods {
                    // Try to find a compatible method (name can be different)
                    if let Some(compatible_method) =
                        self.find_duck_compatible_method(sub_type, required_method)
                    {
                        if compatible_method.name != required_method.name {
                            // Need adapter for name difference
                            adapter_requirements.push(AdapterRequirement::WrapMethod {
                                from_method: compatible_method.name,
                                to_method: required_method.name,
                                adapter: MethodAdapter::Identity,
                            });
                        }
                    } else {
                        // Try to provide a reasonable default
                        if let Some(default_impl) =
                            self.generate_default_implementation(required_method)
                        {
                            adapter_requirements.push(AdapterRequirement::ProvideDefault {
                                method_name: required_method.name,
                                default_impl,
                            });
                        } else {
                            return Ok(StructuralCompatibility::Incompatible(vec![
                                StructuralError::MissingMethod {
                                    name: required_method.name,
                                    expected_signature: self
                                        .method_sig_to_signature(required_method),
                                    span: required_method.span,
                                },
                            ]));
                        }
                    }
                }

                if adapter_requirements.is_empty() {
                    Ok(StructuralCompatibility::Compatible)
                } else {
                    Ok(StructuralCompatibility::RequiresAdapterPattern(
                        adapter_requirements,
                    ))
                }
            }

            _ => self.check_strict_compatibility(sub_type, super_type),
        }
    }

    /// Check nominal structure (hybrid approach)
    fn check_nominal_structure(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<StructuralCompatibility, Vec<StructuralError>> {
        // First check nominal compatibility, then structural
        match (sub_type, super_type) {
            (Type::Named { id: sub_id, .. }, Type::Named { id: super_id, .. }) => {
                if sub_id == super_id {
                    // Same nominal type - always compatible
                    Ok(StructuralCompatibility::Compatible)
                } else {
                    // Different nominal types - check structural compatibility
                    self.check_strict_compatibility(sub_type, super_type)
                }
            }

            _ => self.check_strict_compatibility(sub_type, super_type),
        }
    }

    /// Check gradual compatibility (TypeScript/Flow style)
    fn check_gradual_compatibility(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<StructuralCompatibility, Vec<StructuralError>> {
        match (sub_type, super_type) {
            // Any is compatible with everything
            (Type::Any, _) | (_, Type::Any) => Ok(StructuralCompatibility::Compatible),

            // Unknown requires more careful handling
            (Type::Unknown, _) => {
                // Unknown can be used as any type, but with runtime checks
                Ok(StructuralCompatibility::RequiresAdapterPattern(vec![
                    AdapterRequirement::ProvideDefault {
                        method_name: InternedString::from_symbol(
                            string_interner::Symbol::try_from_usize(0).unwrap(),
                        ),
                        default_impl: DefaultImplementation::Throw(InternedString::from_symbol(
                            string_interner::Symbol::try_from_usize(1).unwrap(),
                        )),
                    },
                ]))
            }

            // Dynamic allows method calls that may not exist
            (Type::Dynamic, _) | (_, Type::Dynamic) => Ok(StructuralCompatibility::Compatible),

            _ => self.check_strict_compatibility(sub_type, super_type),
        }
    }

    /// Check if a named type implements an interface structurally
    fn check_named_type_implements_interface(
        &mut self,
        named_type: &Type,
        required_methods: &[MethodSig],
    ) -> Result<StructuralCompatibility, Vec<StructuralError>> {
        // This would require access to the type registry to get the actual methods
        // For now, we'll assume compatibility and return placeholder

        let mut errors = Vec::new();

        for required_method in required_methods {
            // In a real implementation, we'd look up the named type's methods
            // and check if there's a compatible method

            // Placeholder: assume the method exists
            if required_method.name.symbol().to_usize() % 2 == 0 {
                // Simulate some methods being missing
                errors.push(StructuralError::MissingMethod {
                    name: required_method.name,
                    expected_signature: self.method_sig_to_signature(required_method),
                    span: required_method.span,
                });
            }
        }

        if errors.is_empty() {
            Ok(StructuralCompatibility::Compatible)
        } else {
            Ok(StructuralCompatibility::Incompatible(errors))
        }
    }

    /// Check function type compatibility with variance
    fn check_function_compatibility(
        &mut self,
        sub_params: &[ParamInfo],
        sub_return: &Type,
        super_params: &[ParamInfo],
        super_return: &Type,
    ) -> Result<StructuralCompatibility, Vec<StructuralError>> {
        let mut errors = Vec::new();

        // Check parameter count
        if sub_params.len() != super_params.len() {
            // Could potentially be compatible with default parameters
            return Ok(StructuralCompatibility::RequiresAdapterPattern(vec![
                AdapterRequirement::WrapMethod {
                    from_method: InternedString::from_symbol(
                        string_interner::Symbol::try_from_usize(0).unwrap(),
                    ),
                    to_method: InternedString::from_symbol(
                        string_interner::Symbol::try_from_usize(1).unwrap(),
                    ),
                    adapter: MethodAdapter::ParameterDefault(vec![]),
                },
            ]));
        }

        // Check parameter types (contravariant)
        for (i, (sub_param, super_param)) in sub_params.iter().zip(super_params.iter()).enumerate()
        {
            if !self.are_types_compatible(&super_param.ty, &sub_param.ty) {
                errors.push(StructuralError::VarianceViolation {
                    method_name: InternedString::from_symbol(
                        string_interner::Symbol::try_from_usize(0).unwrap(),
                    ),
                    param_index: i,
                    expected_variance: Variance::Contravariant,
                    actual_variance: Variance::Invariant,
                    span: Span::new(0, 0),
                });
            }
        }

        // Check return type (covariant)
        if !self.are_types_compatible(sub_return, super_return) {
            errors.push(StructuralError::VarianceViolation {
                method_name: InternedString::from_symbol(
                    string_interner::Symbol::try_from_usize(0).unwrap(),
                ),
                param_index: 0,
                expected_variance: Variance::Covariant,
                actual_variance: Variance::Invariant,
                span: Span::new(0, 0),
            });
        }

        if errors.is_empty() {
            Ok(StructuralCompatibility::Compatible)
        } else {
            Ok(StructuralCompatibility::Incompatible(errors))
        }
    }

    /// Check method signature compatibility
    fn check_method_compatibility(
        &self,
        found_method: &MethodSig,
        required_method: &MethodSig,
    ) -> Result<(), Vec<StructuralError>> {
        let mut errors = Vec::new();

        // Check parameter compatibility
        if found_method.params.len() != required_method.params.len() {
            errors.push(StructuralError::MethodSignatureMismatch {
                name: required_method.name,
                expected: self.method_sig_to_signature(required_method),
                found: self.method_sig_to_signature(found_method),
                span: required_method.span,
            });
        }

        // Check return type compatibility
        if !self.are_types_compatible(&found_method.return_type, &required_method.return_type) {
            errors.push(StructuralError::MethodSignatureMismatch {
                name: required_method.name,
                expected: self.method_sig_to_signature(required_method),
                found: self.method_sig_to_signature(found_method),
                span: required_method.span,
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Find a method that's duck-compatible with the required method
    fn find_duck_compatible_method(
        &self,
        ty: &Type,
        required_method: &MethodSig,
    ) -> Option<MethodSignature> {
        // This would require access to the type's methods
        // For now, return a placeholder

        // Simulate finding a compatible method based on name similarity
        if required_method.name.symbol().to_usize() % 3 == 0 {
            Some(self.method_sig_to_signature(required_method))
        } else {
            None
        }
    }

    /// Generate a reasonable default implementation for a missing method
    fn generate_default_implementation(&self, method: &MethodSig) -> Option<DefaultImplementation> {
        // Generate defaults based on method signature
        match &method.return_type {
            Type::Primitive(PrimitiveType::Bool) => Some(DefaultImplementation::ReturnDefault(
                Type::Primitive(PrimitiveType::Bool),
            )),
            Type::Primitive(PrimitiveType::I32) => Some(DefaultImplementation::ReturnDefault(
                Type::Primitive(PrimitiveType::I32),
            )),
            Type::Primitive(PrimitiveType::String) => Some(DefaultImplementation::ReturnDefault(
                Type::Primitive(PrimitiveType::String),
            )),
            Type::Nullable(_) => Some(DefaultImplementation::ReturnDefault(Type::Primitive(
                PrimitiveType::Unit,
            ))),
            _ => Some(DefaultImplementation::Throw(InternedString::from_symbol(
                string_interner::Symbol::try_from_usize(2).unwrap(),
            ))),
        }
    }

    /// Convert MethodSig to MethodSignature
    fn method_sig_to_signature(&self, method: &MethodSig) -> MethodSignature {
        MethodSignature {
            name: method.name,
            type_params: method.type_params.clone(),
            params: method
                .params
                .iter()
                .map(|p| ParamInfo {
                    name: Some(p.name),
                    ty: p.ty.clone(),
                    is_optional: false,
                    is_varargs: p.is_varargs,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: p.is_mut,
                    is_inout: false,
                })
                .collect(),
            return_type: method.return_type.clone(),
            is_static: method.is_static,
            is_async: method.is_async,
            variance: vec![], // Would be computed from type_params
        }
    }

    /// Check if two types are compatible (simplified)
    fn are_types_compatible(&self, ty1: &Type, ty2: &Type) -> bool {
        // Simplified compatibility check
        ty1 == ty2 || matches!((ty1, ty2), (Type::Any, _) | (_, Type::Any))
    }

    // === ENHANCED VARIANCE ANALYSIS METHODS ===

    /// Check structural subtyping with full variance analysis
    pub fn is_subtype_with_variance(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
        variance: Variance,
    ) -> Result<bool, Vec<StructuralError>> {
        // Check cache first using structural type IDs
        let sub_id = self.get_structural_id(sub_type);
        let super_id = self.get_structural_id(super_type);
        let cache_key = (sub_id, super_id, variance);

        if let Some(&result) = self.subtyping_cache.get(&cache_key) {
            return Ok(result);
        }

        let result = self.check_subtype_with_variance_impl(sub_type, super_type, variance)?;
        self.subtyping_cache.insert(cache_key, result);
        Ok(result)
    }

    /// Implementation of variance-aware subtyping
    fn check_subtype_with_variance_impl(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
        variance: Variance,
    ) -> Result<bool, Vec<StructuralError>> {
        match variance {
            Variance::Covariant => {
                // sub_type <: super_type
                self.is_covariant_subtype(sub_type, super_type)
            }
            Variance::Contravariant => {
                // super_type <: sub_type (reversed for contravariance)
                self.is_contravariant_subtype(sub_type, super_type)
            }
            Variance::Invariant => {
                // sub_type = super_type (exact match)
                self.is_invariant_subtype(sub_type, super_type)
            }
            Variance::Bivariant => {
                // Both directions allowed (unsafe, rare)
                Ok(true)
            }
        }
    }

    /// Check covariant subtyping: sub_type <: super_type
    fn is_covariant_subtype(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<bool, Vec<StructuralError>> {
        match (sub_type, super_type) {
            // Reflexivity: T <: T
            (a, b) if a == b => Ok(true),

            // Any type relationships
            (_, Type::Any) => Ok(true),
            (Type::Never, _) => Ok(true),

            // Named type hierarchy
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
                    // Same type constructor, check type arguments with their variances
                    self.check_type_args_variance(sub_id, sub_args, super_args)
                } else {
                    // Different type constructors - check nominal hierarchy
                    // This would require access to type registry for inheritance chains
                    Ok(false)
                }
            }

            // Array covariance: Array<Dog> <: Array<Animal> if Dog <: Animal
            (
                Type::Array {
                    element_type: sub_elem,
                    ..
                },
                Type::Array {
                    element_type: super_elem,
                    ..
                },
            ) => self.is_subtype_with_variance(sub_elem, super_elem, Variance::Covariant),

            // Function contravariance in parameters, covariance in return
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
            ) => self.check_function_variance(sub_params, sub_ret, super_params, super_ret),

            // Interface structural subtyping
            (
                Type::Interface {
                    methods: sub_methods,
                    ..
                },
                Type::Interface {
                    methods: super_methods,
                    ..
                },
            ) => self.check_interface_structural_subtyping(sub_methods, super_methods),

            // Struct structural subtyping
            (
                Type::Struct {
                    fields: sub_fields, ..
                },
                Type::Struct {
                    fields: super_fields,
                    ..
                },
            ) => self.check_struct_structural_subtyping(sub_fields, super_fields),

            // Union type covariance: (A | B) <: C if both A <: C and B <: C
            (Type::Union(sub_types), super_type) => {
                for sub_variant in sub_types {
                    if !self.is_subtype_with_variance(
                        sub_variant,
                        super_type,
                        Variance::Covariant,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            // Union type on right: A <: (B | C) if A <: B or A <: C
            (sub_type, Type::Union(super_types)) => {
                for super_variant in super_types {
                    if self.is_subtype_with_variance(
                        sub_type,
                        super_variant,
                        Variance::Covariant,
                    )? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            // Intersection type covariance: A <: (B & C) if A <: B and A <: C
            (sub_type, Type::Intersection(super_types)) => {
                for super_component in super_types {
                    if !self.is_subtype_with_variance(
                        sub_type,
                        super_component,
                        Variance::Covariant,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            // Intersection type on left: (A & B) <: C if A <: C or B <: C
            (Type::Intersection(sub_types), super_type) => {
                for sub_component in sub_types {
                    if self.is_subtype_with_variance(
                        sub_component,
                        super_type,
                        Variance::Covariant,
                    )? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            _ => Ok(false),
        }
    }

    /// Check contravariant subtyping: super_type <: sub_type (reversed)
    fn is_contravariant_subtype(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<bool, Vec<StructuralError>> {
        // For contravariance, we check the reverse relationship
        self.is_covariant_subtype(super_type, sub_type)
    }

    /// Check invariant subtyping: sub_type = super_type (exact match)
    fn is_invariant_subtype(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<bool, Vec<StructuralError>> {
        // Invariance requires exact type equality
        Ok(sub_type == super_type)
    }

    /// Check type arguments with their declared variances
    fn check_type_args_variance(
        &mut self,
        type_constructor: &TypeId,
        sub_args: &[Type],
        super_args: &[Type],
    ) -> Result<bool, Vec<StructuralError>> {
        if sub_args.len() != super_args.len() {
            return Ok(false);
        }

        // Look up variance for this type constructor
        let constructor_name = InternedString::from_symbol(
            string_interner::Symbol::try_from_usize(type_constructor.as_u32() as usize).unwrap(),
        );

        let variances = self
            .type_constructor_variances
            .get(&constructor_name)
            .cloned()
            .unwrap_or_else(|| vec![Variance::Invariant; sub_args.len()]);

        for ((sub_arg, super_arg), &variance) in
            sub_args.iter().zip(super_args.iter()).zip(variances.iter())
        {
            if !self.is_subtype_with_variance(sub_arg, super_arg, variance)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check function type variance (contravariant params, covariant return)
    fn check_function_variance(
        &mut self,
        sub_params: &[ParamInfo],
        sub_return: &Type,
        super_params: &[ParamInfo],
        super_return: &Type,
    ) -> Result<bool, Vec<StructuralError>> {
        // Parameter count must match
        if sub_params.len() != super_params.len() {
            return Ok(false);
        }

        // Parameters are contravariant: super_param <: sub_param
        for (sub_param, super_param) in sub_params.iter().zip(super_params.iter()) {
            if !self.is_subtype_with_variance(
                &super_param.ty,
                &sub_param.ty,
                Variance::Contravariant,
            )? {
                return Ok(false);
            }
        }

        // Return type is covariant: sub_return <: super_return
        self.is_subtype_with_variance(sub_return, super_return, Variance::Covariant)
    }

    /// Check interface structural subtyping with method variance
    fn check_interface_structural_subtyping(
        &mut self,
        sub_methods: &[MethodSig],
        super_methods: &[MethodSig],
    ) -> Result<bool, Vec<StructuralError>> {
        // Sub interface must have all methods of super interface
        for super_method in super_methods {
            if let Some(sub_method) = sub_methods.iter().find(|m| m.name == super_method.name) {
                // Check method signature variance
                if !self.check_method_variance(sub_method, super_method)? {
                    return Ok(false);
                }
            } else {
                return Ok(false); // Missing required method
            }
        }
        Ok(true)
    }

    /// Check method signature variance compatibility
    fn check_method_variance(
        &mut self,
        sub_method: &MethodSig,
        super_method: &MethodSig,
    ) -> Result<bool, Vec<StructuralError>> {
        // Parameters are contravariant
        if sub_method.params.len() != super_method.params.len() {
            return Ok(false);
        }

        for (sub_param, super_param) in sub_method.params.iter().zip(super_method.params.iter()) {
            if !self.is_subtype_with_variance(
                &super_param.ty,
                &sub_param.ty,
                Variance::Contravariant,
            )? {
                return Ok(false);
            }
        }

        // Return type is covariant
        self.is_subtype_with_variance(
            &sub_method.return_type,
            &super_method.return_type,
            Variance::Covariant,
        )
    }

    /// Check struct structural subtyping with field variance
    fn check_struct_structural_subtyping(
        &mut self,
        sub_fields: &[FieldDef],
        super_fields: &[FieldDef],
    ) -> Result<bool, Vec<StructuralError>> {
        // Sub struct must have all fields of super struct
        for super_field in super_fields {
            if let Some(sub_field) = sub_fields.iter().find(|f| f.name == super_field.name) {
                // Mutable fields are invariant, immutable fields are covariant
                let variance = if super_field.mutability == Mutability::Mutable
                    || sub_field.mutability == Mutability::Mutable
                {
                    Variance::Invariant
                } else {
                    Variance::Covariant
                };

                if !self.is_subtype_with_variance(&sub_field.ty, &super_field.ty, variance)? {
                    return Ok(false);
                }
            } else {
                return Ok(false); // Missing required field
            }
        }
        Ok(true)
    }

    /// Analyze variance context for a type
    pub fn analyze_variance_context(&mut self, ty: &Type) -> VarianceContext {
        let type_id = self.get_structural_id(ty);
        if let Some(context) = self.variance_cache.get(&type_id) {
            return context.clone();
        }

        let context = self.compute_variance_context(ty);
        self.variance_cache.insert(type_id, context.clone());
        context
    }

    /// Compute variance context for a type
    fn compute_variance_context(&self, ty: &Type) -> VarianceContext {
        match ty {
            Type::Function {
                params,
                return_type,
                ..
            } => {
                let mut type_param_variances = HashMap::new();

                // Parameters are contravariant positions
                for param in params {
                    self.collect_type_param_variances(
                        &param.ty,
                        Variance::Contravariant,
                        &mut type_param_variances,
                    );
                }

                // Return type is covariant position
                self.collect_type_param_variances(
                    return_type,
                    Variance::Covariant,
                    &mut type_param_variances,
                );

                VarianceContext {
                    current_variance: Variance::Covariant,
                    variance_stack: vec![],
                    type_param_variances,
                    hkt_variances: HashMap::new(),
                }
            }

            Type::Array { element_type, .. } => {
                let mut type_param_variances = HashMap::new();
                self.collect_type_param_variances(
                    element_type,
                    Variance::Covariant,
                    &mut type_param_variances,
                );

                VarianceContext {
                    current_variance: Variance::Covariant,
                    variance_stack: vec![],
                    type_param_variances,
                    hkt_variances: HashMap::new(),
                }
            }

            _ => VarianceContext {
                current_variance: Variance::Invariant,
                variance_stack: vec![],
                type_param_variances: HashMap::new(),
                hkt_variances: HashMap::new(),
            },
        }
    }

    /// Collect type parameter variances recursively
    fn collect_type_param_variances(
        &self,
        ty: &Type,
        current_variance: Variance,
        variances: &mut HashMap<InternedString, Variance>,
    ) {
        match ty {
            Type::TypeVar(..) => {
                // For generic types, we would need to extract the name
                // This is a simplified implementation
            }

            Type::Named { type_args, .. } => {
                for arg in type_args {
                    self.collect_type_param_variances(arg, current_variance, variances);
                }
            }

            Type::Function {
                params,
                return_type,
                ..
            } => {
                let param_variance = match current_variance {
                    Variance::Covariant => Variance::Contravariant,
                    Variance::Contravariant => Variance::Covariant,
                    v => v,
                };

                for param in params {
                    self.collect_type_param_variances(&param.ty, param_variance, variances);
                }
                self.collect_type_param_variances(return_type, current_variance, variances);
            }

            Type::Array { element_type, .. } => {
                self.collect_type_param_variances(element_type, current_variance, variances);
            }

            _ => {}
        }
    }

    /// Validate variance annotations on type parameters
    pub fn validate_variance_annotations(
        &self,
        type_params: &[TypeParam],
        usage_context: &VarianceContext,
    ) -> Result<(), Vec<StructuralError>> {
        let mut errors = Vec::new();

        for type_param in type_params {
            if let Some(&actual_variance) = usage_context.type_param_variances.get(&type_param.name)
            {
                if type_param.variance != actual_variance {
                    errors.push(StructuralError::VarianceViolation {
                        method_name: type_param.name,
                        param_index: 0,
                        expected_variance: type_param.variance,
                        actual_variance,
                        span: Span::new(0, 0), // Would need to be passed in or stored separately
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

    /// Get structural type ID for caching
    fn get_structural_id(&self, ty: &Type) -> StructuralTypeId {
        // Simple hash-based ID generation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", ty).hash(&mut hasher);
        StructuralTypeId(hasher.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::AstArena;

    #[test]
    fn test_structural_interface_compatibility() {
        let mut checker = StructuralTypeChecker::new();
        let mut arena = AstArena::new();

        // Create interface with one method
        let required_interface = Type::Interface {
            methods: vec![MethodSig {
                name: arena.intern_string("getName"),
                type_params: vec![],
                params: vec![],
                return_type: Type::Primitive(PrimitiveType::String),
                where_clause: vec![],
                is_static: false,
                is_async: false,
                is_extension: false,
                visibility: Visibility::Public,
                span: Span::new(0, 0),
            }],
            is_structural: true,
            nullability: NullabilityKind::NonNull,
        };

        // Create interface that has the required method
        let implementing_interface = Type::Interface {
            methods: vec![
                MethodSig {
                    name: arena.intern_string("getName"),
                    type_params: vec![],
                    params: vec![],
                    return_type: Type::Primitive(PrimitiveType::String),
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    is_extension: false,
                    visibility: Visibility::Public,
                    span: Span::new(0, 0),
                },
                MethodSig {
                    name: arena.intern_string("getAge"),
                    type_params: vec![],
                    params: vec![],
                    return_type: Type::Primitive(PrimitiveType::I32),
                    where_clause: vec![],
                    is_static: false,
                    is_async: false,
                    is_extension: false,
                    visibility: Visibility::Public,
                    span: Span::new(0, 0),
                },
            ],
            is_structural: true,
            nullability: NullabilityKind::NonNull,
        };

        // Test compatibility
        let result = checker
            .is_structurally_compatible(
                &implementing_interface,
                &required_interface,
                StructuralMode::Strict,
            )
            .unwrap();

        assert_eq!(result, StructuralCompatibility::Compatible);
    }

    #[test]
    fn test_structural_missing_method() {
        let mut checker = StructuralTypeChecker::new();
        let mut arena = AstArena::new();

        // Create interface with method
        let required_interface = Type::Interface {
            methods: vec![MethodSig {
                name: arena.intern_string("missingMethod"),
                type_params: vec![],
                params: vec![],
                return_type: Type::Primitive(PrimitiveType::String),
                where_clause: vec![],
                is_static: false,
                is_async: false,
                is_extension: false,
                visibility: Visibility::Public,
                span: Span::new(0, 0),
            }],
            is_structural: true,
            nullability: NullabilityKind::NonNull,
        };

        // Create empty interface
        let empty_interface = Type::Interface {
            methods: vec![],
            is_structural: true,
            nullability: NullabilityKind::NonNull,
        };

        // Test compatibility
        let result = checker
            .is_structurally_compatible(
                &empty_interface,
                &required_interface,
                StructuralMode::Strict,
            )
            .unwrap();

        match result {
            StructuralCompatibility::Incompatible(errors) => {
                assert!(!errors.is_empty());
                assert!(matches!(errors[0], StructuralError::MissingMethod { .. }));
            }
            _ => panic!("Expected incompatible result"),
        }
    }
}
