//! # Type Inference Engine
//!
//! Implements Hindley-Milner style type inference with extensions for:
//! - Subtyping
//! - Higher-kinded types
//! - Type classes/traits
//! - Gradual typing

use crate::arena::InternedString;
use crate::type_registry::{
    Mutability, ParamInfo, PrimitiveType, Type, TypeVar, TypeVarId, TypeVarKind,
};
use std::collections::{HashMap, VecDeque};

/// Type inference context
pub struct InferenceContext {
    /// Type registry
    pub registry: Box<crate::type_registry::TypeRegistry>,
    /// Type variable substitutions
    substitutions: HashMap<TypeVarId, Type>,
    /// Constraints to be solved
    constraints: VecDeque<Constraint>,
    /// Current type variable counter
    next_var: u32,
    /// Inference options
    options: InferenceOptions,
}

/// Options for type inference
#[derive(Debug, Clone)]
pub struct InferenceOptions {
    /// Allow implicit conversions
    pub allow_implicit_conversions: bool,
    /// Use gradual typing (allow Any type)
    pub gradual_typing: bool,
    /// Infer variance automatically
    pub infer_variance: bool,
    /// Maximum inference iterations
    pub max_iterations: usize,
}

impl Default for InferenceOptions {
    fn default() -> Self {
        Self {
            allow_implicit_conversions: true,
            gradual_typing: false,
            infer_variance: true,
            max_iterations: 1000,
        }
    }
}

/// Type constraint
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Type equality: T1 = T2
    Equal(Type, Type),
    /// Subtype: T1 <: T2
    Subtype(Type, Type),
    /// Type class constraint: T : Trait
    HasTrait(Type, InternedString, Vec<Type>),
    /// Field access: T has field 'name' of type U
    HasField(Type, InternedString, Type),
    /// Method constraint: T has method 'name'
    HasMethod(Type, InternedString, Vec<Type>, Type),
    /// Indexable constraint: T[K] = V (T is indexable by K, yielding V)
    Indexable(Type, Type, Type),
    /// Numeric constraint
    IsNumeric(Type),
    /// Reference constraint
    IsRef(Type, Mutability),
}

/// Type inference error
#[derive(Debug, Clone)]
pub enum InferenceError {
    /// Type mismatch
    TypeMismatch { expected: Type, found: Type },
    /// Unresolved type variable
    UnresolvedTypeVar(TypeVarId),
    /// Infinite type
    InfiniteType(TypeVarId, Type),
    /// Missing trait implementation
    MissingTrait {
        ty: Type,
        trait_name: InternedString,
    },
    /// Unknown field
    UnknownField { ty: Type, field: InternedString },
    /// Unknown method
    UnknownMethod { ty: Type, method: InternedString },
    /// Ambiguous type
    AmbiguousType(Vec<Type>),
    /// Constraint solving failed
    UnsolvableConstraint(Constraint),
}

impl InferenceContext {
    pub fn new(registry: Box<crate::type_registry::TypeRegistry>) -> Self {
        Self {
            registry,
            substitutions: HashMap::new(),
            constraints: VecDeque::new(),
            next_var: 1,
            options: InferenceOptions::default(),
        }
    }

    pub fn with_options(
        registry: Box<crate::type_registry::TypeRegistry>,
        options: InferenceOptions,
    ) -> Self {
        Self {
            registry,
            substitutions: HashMap::new(),
            constraints: VecDeque::new(),
            next_var: 1,
            options,
        }
    }

    /// Create a fresh type variable
    pub fn fresh_type_var(&mut self) -> Type {
        let id = TypeVarId::next();
        self.next_var += 1;
        Type::TypeVar(TypeVar {
            id,
            name: None,
            kind: TypeVarKind::Type,
        })
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push_back(constraint);
    }

    /// Unify two types
    pub fn unify(&mut self, t1: Type, t2: Type) -> Result<Type, InferenceError> {
        let t1 = self.apply_substitutions(&t1);
        let t2 = self.apply_substitutions(&t2);

        // Debug: catch problematic unification
        if matches!(&t1, Type::Primitive(crate::PrimitiveType::Unit))
            && matches!(&t2, Type::Named { .. })
        {
            eprintln!(
                "[UNIFY DEBUG] Attempting to unify Unit with Named: {:?}",
                t2
            );
            eprintln!(
                "[UNIFY DEBUG] Call location: {}:{}:{}",
                file!(),
                line!(),
                column!()
            );
            // This will fail and generate the error
        }
        if matches!(&t2, Type::Primitive(crate::PrimitiveType::Unit))
            && matches!(&t1, Type::Named { .. })
        {
            eprintln!(
                "[UNIFY DEBUG] Attempting to unify Named with Unit: {:?}",
                t1
            );
            eprintln!(
                "[UNIFY DEBUG] Call location: {}:{}:{}",
                file!(),
                line!(),
                column!()
            );
        }

        match (&t1, &t2) {
            // Same type
            (Type::Primitive(p1), Type::Primitive(p2)) if p1 == p2 => Ok(t1),

            // Type variables
            (Type::TypeVar(v), t) | (t, Type::TypeVar(v)) => {
                if let Type::TypeVar(v2) = t {
                    if v.id == v2.id {
                        return Ok(t1);
                    }
                }

                // Occurs check
                if self.occurs_check(v.id, t) {
                    return Err(InferenceError::InfiniteType(v.id, t.clone()));
                }

                self.substitutions.insert(v.id, t.clone());
                Ok(t.clone())
            }

            // Named types
            (
                Type::Named {
                    id: id1,
                    type_args: args1,
                    ..
                },
                Type::Named {
                    id: id2,
                    type_args: args2,
                    ..
                },
            ) => {
                if id1 != id2 || args1.len() != args2.len() {
                    return Err(InferenceError::TypeMismatch {
                        expected: t1,
                        found: t2,
                    });
                }

                let mut unified_args = Vec::new();
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    unified_args.push(self.unify(a1.clone(), a2.clone())?);
                }

                Ok(Type::Named {
                    id: *id1,
                    type_args: unified_args,
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                })
            }

            // Function types
            (
                Type::Function {
                    params: p1,
                    return_type: r1,
                    is_varargs: v1,
                    has_named_params: n1,
                    has_default_params: d1,
                    ..
                },
                Type::Function {
                    params: p2,
                    return_type: r2,
                    is_varargs: v2,
                    has_named_params: n2,
                    has_default_params: d2,
                    ..
                },
            ) => {
                if p1.len() != p2.len() || v1 != v2 {
                    return Err(InferenceError::TypeMismatch {
                        expected: t1,
                        found: t2,
                    });
                }

                let mut unified_params = Vec::new();
                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    let unified_param = ParamInfo {
                        name: param1.name.or(param2.name),
                        ty: self.unify(param1.ty.clone(), param2.ty.clone())?,
                        is_optional: param1.is_optional || param2.is_optional,
                        is_varargs: param1.is_varargs || param2.is_varargs,
                        is_keyword_only: param1.is_keyword_only || param2.is_keyword_only,
                        is_positional_only: param1.is_positional_only && param2.is_positional_only,
                        is_out: param1.is_out || param2.is_out,
                        is_ref: param1.is_ref || param2.is_ref,
                        is_inout: param1.is_inout || param2.is_inout,
                    };
                    unified_params.push(unified_param);
                }

                let unified_return = self.unify((**r1).clone(), (**r2).clone())?;

                Ok(Type::Function {
                    params: unified_params,
                    return_type: Box::new(unified_return),
                    is_varargs: *v1,
                    has_named_params: *n1 || *n2,
                    has_default_params: *d1 || *d2,
                    async_kind: crate::type_registry::AsyncKind::default(),
                    calling_convention: crate::type_registry::CallingConvention::default(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                })
            }

            // Tuple types
            (Type::Tuple(elems1), Type::Tuple(elems2)) => {
                if elems1.len() != elems2.len() {
                    return Err(InferenceError::TypeMismatch {
                        expected: t1,
                        found: t2,
                    });
                }

                let mut unified_elems = Vec::new();
                for (e1, e2) in elems1.iter().zip(elems2.iter()) {
                    unified_elems.push(self.unify(e1.clone(), e2.clone())?);
                }

                Ok(Type::Tuple(unified_elems))
            }

            // Array types
            (
                Type::Array {
                    element_type: e1,
                    size: s1,
                    ..
                },
                Type::Array {
                    element_type: e2,
                    size: s2,
                    ..
                },
            ) => {
                if s1 != s2 {
                    return Err(InferenceError::TypeMismatch {
                        expected: t1,
                        found: t2,
                    });
                }

                let unified_elem = self.unify((**e1).clone(), (**e2).clone())?;
                Ok(Type::Array {
                    element_type: Box::new(unified_elem),
                    size: s1.clone(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                })
            }

            // Reference types
            (
                Type::Reference {
                    ty: t1,
                    mutability: m1,
                    lifetime: l1,
                    ..
                },
                Type::Reference {
                    ty: t2,
                    mutability: m2,
                    lifetime: l2,
                    ..
                },
            ) => {
                if m1 != m2 {
                    return Err(InferenceError::TypeMismatch {
                        expected: (**t1).clone(),
                        found: (**t2).clone(),
                    });
                }

                let unified_ty = self.unify((**t1).clone(), (**t2).clone())?;

                // TODO: Properly handle lifetime unification
                let lifetime = l1.clone().or_else(|| l2.clone());

                Ok(Type::Reference {
                    ty: Box::new(unified_ty),
                    mutability: *m1,
                    lifetime,
                    nullability: crate::type_registry::NullabilityKind::default(),
                })
            }

            // Optional types
            (Type::Optional(t1), Type::Optional(t2)) => {
                let unified = self.unify((**t1).clone(), (**t2).clone())?;
                Ok(Type::Optional(Box::new(unified)))
            }

            // Gradual typing: Any unifies with anything
            (Type::Any, t) | (t, Type::Any) if self.options.gradual_typing => Ok(t.clone()),

            // Any type (for error recovery)
            (Type::Any, t) | (t, Type::Any) => Ok(t.clone()),

            // Error type propagates
            (Type::Error, _) | (_, Type::Error) => Ok(Type::Error),

            // Otherwise, type mismatch
            _ => Err(InferenceError::TypeMismatch {
                expected: t1,
                found: t2,
            }),
        }
    }

    /// Apply type substitutions
    pub fn apply_substitutions(&self, ty: &Type) -> Type {
        match ty {
            Type::TypeVar(v) => {
                if let Some(subst) = self.substitutions.get(&v.id) {
                    self.apply_substitutions(subst)
                } else {
                    ty.clone()
                }
            }
            Type::Named {
                id,
                type_args,
                const_args,
                variance,
                nullability,
            } => Type::Named {
                id: *id,
                type_args: type_args
                    .iter()
                    .map(|arg| self.apply_substitutions(arg))
                    .collect(),
                const_args: const_args.clone(),
                variance: variance.clone(),
                nullability: *nullability,
            },
            Type::Function {
                params,
                return_type,
                is_varargs,
                has_named_params,
                has_default_params,
                async_kind,
                calling_convention,
                nullability,
            } => Type::Function {
                params: params
                    .iter()
                    .map(|p| ParamInfo {
                        name: p.name,
                        ty: self.apply_substitutions(&p.ty),
                        is_optional: p.is_optional,
                        is_varargs: p.is_varargs,
                        is_keyword_only: p.is_keyword_only,
                        is_positional_only: p.is_positional_only,
                        is_out: p.is_out,
                        is_ref: p.is_ref,
                        is_inout: p.is_inout,
                    })
                    .collect(),
                return_type: Box::new(self.apply_substitutions(return_type)),
                is_varargs: *is_varargs,
                has_named_params: *has_named_params,
                has_default_params: *has_default_params,
                async_kind: *async_kind,
                calling_convention: *calling_convention,
                nullability: *nullability,
            },
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.apply_substitutions(e)).collect())
            }
            Type::Array {
                element_type,
                size,
                nullability,
            } => Type::Array {
                element_type: Box::new(self.apply_substitutions(element_type)),
                size: size.clone(),
                nullability: *nullability,
            },
            Type::Reference {
                ty,
                mutability,
                lifetime,
                nullability,
            } => Type::Reference {
                ty: Box::new(self.apply_substitutions(ty)),
                mutability: *mutability,
                lifetime: lifetime.clone(),
                nullability: *nullability,
            },
            Type::Optional(t) => Type::Optional(Box::new(self.apply_substitutions(t))),
            Type::Union(types) => {
                Type::Union(types.iter().map(|t| self.apply_substitutions(t)).collect())
            }
            Type::Intersection(types) => {
                Type::Intersection(types.iter().map(|t| self.apply_substitutions(t)).collect())
            }
            _ => ty.clone(),
        }
    }

    /// Occurs check to prevent infinite types
    fn occurs_check(&self, var: TypeVarId, ty: &Type) -> bool {
        match ty {
            Type::TypeVar(v) => v.id == var,
            Type::Named { type_args, .. } => {
                type_args.iter().any(|arg| self.occurs_check(var, arg))
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                params.iter().any(|p| self.occurs_check(var, &p.ty))
                    || self.occurs_check(var, return_type)
            }
            Type::Tuple(elems) => elems.iter().any(|e| self.occurs_check(var, e)),
            Type::Array { element_type, .. } => self.occurs_check(var, element_type),
            Type::Reference { ty, .. } => self.occurs_check(var, ty),
            Type::Optional(t) => self.occurs_check(var, t),
            Type::Union(types) | Type::Intersection(types) => {
                types.iter().any(|t| self.occurs_check(var, t))
            }
            _ => false,
        }
    }

    /// Solve all constraints
    pub fn solve_constraints(&mut self) -> Result<(), InferenceError> {
        let mut iterations = 0;

        while let Some(constraint) = self.constraints.pop_front() {
            if iterations >= self.options.max_iterations {
                return Err(InferenceError::UnsolvableConstraint(constraint));
            }
            iterations += 1;

            match constraint {
                Constraint::Equal(t1, t2) => {
                    self.unify(t1, t2)?;
                }

                Constraint::Subtype(sub, sup) => {
                    self.check_subtype(&sub, &sup)?;
                }

                Constraint::HasTrait(ty, trait_name, args) => {
                    self.check_trait_impl(&ty, trait_name, &args)?;
                }

                Constraint::HasField(ty, field, field_ty) => {
                    self.check_field(&ty, field, &field_ty)?;
                }

                Constraint::HasMethod(ty, method, params, ret) => {
                    self.check_method(&ty, method, &params, &ret)?;
                }

                Constraint::IsNumeric(ty) => {
                    self.check_numeric(&ty)?;
                }

                Constraint::IsRef(ty, mutability) => {
                    self.check_reference(&ty, mutability)?;
                }

                Constraint::Indexable(container_ty, index_ty, element_ty) => {
                    self.check_indexable(&container_ty, &index_ty, &element_ty)?;
                }
            }
        }

        Ok(())
    }

    /// Check if sub is a subtype of sup
    fn check_subtype(&mut self, sub: &Type, sup: &Type) -> Result<(), InferenceError> {
        let sub = self.apply_substitutions(sub);
        let sup = self.apply_substitutions(sup);

        match (&sub, &sup) {
            // Same type is subtype of itself
            _ if sub == sup => Ok(()),

            // Any is supertype of everything (gradual typing)
            (_, Type::Any) if self.options.gradual_typing => Ok(()),

            // Never is subtype of everything
            (Type::Never, _) => Ok(()),

            // Numeric coercions
            (Type::Primitive(p1), Type::Primitive(p2)) => {
                if self.options.allow_implicit_conversions {
                    self.check_numeric_conversion(*p1, *p2)
                } else {
                    Err(InferenceError::TypeMismatch {
                        expected: sup,
                        found: sub,
                    })
                }
            }

            // Reference subtyping (covariant)
            (
                Type::Reference {
                    ty: t1,
                    mutability: Mutability::Immutable,
                    ..
                },
                Type::Reference {
                    ty: t2,
                    mutability: Mutability::Immutable,
                    ..
                },
            ) => self.check_subtype(t1, t2),

            // Array subtyping (covariant for immutable)
            (
                Type::Array {
                    element_type: e1,
                    size: s1,
                    ..
                },
                Type::Array {
                    element_type: e2,
                    size: s2,
                    ..
                },
            ) if s1 == s2 => self.check_subtype(e1, e2),

            // TODO: Add more subtyping rules
            _ => Err(InferenceError::TypeMismatch {
                expected: sup,
                found: sub,
            }),
        }
    }

    /// Check numeric conversion
    fn check_numeric_conversion(
        &self,
        from: PrimitiveType,
        to: PrimitiveType,
    ) -> Result<(), InferenceError> {
        use PrimitiveType::*;

        let allowed = match (from, to) {
            // Widening conversions
            (I8, I16) | (I8, I32) | (I8, I64) | (I8, I128) => true,
            (I16, I32) | (I16, I64) | (I16, I128) => true,
            (I32, I64) | (I32, I128) => true,
            (I64, I128) => true,

            (U8, U16) | (U8, U32) | (U8, U64) | (U8, U128) => true,
            (U16, U32) | (U16, U64) | (U16, U128) => true,
            (U32, U64) | (U32, U128) => true,
            (U64, U128) => true,

            (F32, F64) => true,

            // Int to float
            (I8 | I16 | I32, F32 | F64) => true,
            (I64, F64) => true,
            (U8 | U16 | U32, F32 | F64) => true,
            (U64, F64) => true,

            _ => false,
        };

        if allowed {
            Ok(())
        } else {
            Err(InferenceError::TypeMismatch {
                expected: Type::Primitive(to),
                found: Type::Primitive(from),
            })
        }
    }

    /// Check trait implementation
    fn check_trait_impl(
        &self,
        _ty: &Type,
        _trait_name: InternedString,
        _args: &[Type],
    ) -> Result<(), InferenceError> {
        // TODO: Implement trait checking
        Ok(())
    }

    /// Check field access
    fn check_field(
        &self,
        _ty: &Type,
        _field: InternedString,
        _expected_ty: &Type,
    ) -> Result<(), InferenceError> {
        // TODO: Implement field checking
        Ok(())
    }

    /// Check method
    fn check_method(
        &self,
        _ty: &Type,
        _method: InternedString,
        _params: &[Type],
        _ret: &Type,
    ) -> Result<(), InferenceError> {
        // TODO: Implement method checking
        Ok(())
    }

    /// Check if type is numeric
    fn check_numeric(&self, ty: &Type) -> Result<(), InferenceError> {
        match ty {
            Type::Primitive(p) if p.is_numeric() => Ok(()),
            Type::TypeVar(_) => {
                // Add numeric constraint for later
                Ok(())
            }
            _ => Err(InferenceError::TypeMismatch {
                expected: Type::Primitive(PrimitiveType::I32), // Example numeric type
                found: ty.clone(),
            }),
        }
    }

    /// Check if type is a reference
    fn check_reference(&self, ty: &Type, expected_mut: Mutability) -> Result<(), InferenceError> {
        match ty {
            Type::Reference { mutability, .. } if *mutability == expected_mut => Ok(()),
            Type::TypeVar(_) => Ok(()), // Will be resolved later
            _ => Err(InferenceError::TypeMismatch {
                expected: Type::Reference {
                    ty: Box::new(Type::Any),
                    mutability: expected_mut,
                    lifetime: None,
                    nullability: crate::type_registry::NullabilityKind::default(),
                },
                found: ty.clone(),
            }),
        }
    }

    /// Check that a type is indexable by an index type, yielding an element type
    fn check_indexable(
        &mut self,
        container_ty: &Type,
        index_ty: &Type,
        element_ty: &Type,
    ) -> Result<(), InferenceError> {
        match container_ty {
            Type::Array { element_type, .. } => {
                // Arrays must be indexed by integers
                if !matches!(
                    index_ty,
                    Type::Primitive(PrimitiveType::I32) | Type::TypeVar(_)
                ) {
                    return Err(InferenceError::TypeMismatch {
                        expected: Type::Primitive(PrimitiveType::I32),
                        found: index_ty.clone(),
                    });
                }
                // Element type must match
                if element_type.as_ref() != element_ty && !matches!(element_ty, Type::TypeVar(_)) {
                    return Err(InferenceError::TypeMismatch {
                        expected: element_type.as_ref().clone(),
                        found: element_ty.clone(),
                    });
                }
                Ok(())
            }

            Type::Named { id, type_args, .. } => {
                // For named types, check if they implement indexing traits
                // This should be done through the type environment's trait system

                // First, try to resolve the type definition
                if let Some(_type_def) = self.registry.get_type_by_id(*id) {
                    // Check if this type has indexing implemented through traits
                    // This is where we'd check for Index<T> trait implementations
                    // For now, we'll defer to constraint solving by creating unification constraints

                    // If it's a generic type, we might be able to infer from type parameters
                    if type_args.len() == 1 {
                        // Single type parameter - could be indexable by integer, yielding element type
                        // Create constraints rather than hardcoding
                        match self.unify(index_ty.clone(), Type::Primitive(PrimitiveType::I32)) {
                            Ok(_) => {
                                // If index is integer, element type should be the type parameter
                                self.unify(element_ty.clone(), type_args[0].clone())?;
                                Ok(())
                            }
                            Err(_) => {
                                // Not indexable by integer
                                Err(InferenceError::TypeMismatch {
                                    expected: Type::Array {
                                        element_type: Box::new(Type::Any),
                                        size: None,
                                        nullability: crate::type_registry::NullabilityKind::default(
                                        ),
                                    },
                                    found: container_ty.clone(),
                                })
                            }
                        }
                    } else if type_args.len() == 2 {
                        // Two type parameters - could be map-like (key, value)
                        // Try to unify index with first type parameter
                        match self.unify(index_ty.clone(), type_args[0].clone()) {
                            Ok(_) => {
                                // If successful, element type should be the second type parameter
                                self.unify(element_ty.clone(), type_args[1].clone())?;
                                Ok(())
                            }
                            Err(_) => {
                                // Not indexable by the key type
                                Err(InferenceError::TypeMismatch {
                                    expected: type_args[0].clone(),
                                    found: index_ty.clone(),
                                })
                            }
                        }
                    } else {
                        // Other generic arities - not standard indexable patterns
                        Err(InferenceError::TypeMismatch {
                            expected: Type::Array {
                                element_type: Box::new(Type::Never),
                                size: None,
                                nullability: crate::type_registry::NullabilityKind::default(),
                            },
                            found: container_ty.clone(),
                        })
                    }
                } else {
                    // Type not found - defer to constraint solving
                    // This allows for external type definitions
                    Ok(())
                }
            }

            Type::Primitive(PrimitiveType::String) => {
                // Strings indexed by integers yield characters
                if !matches!(
                    index_ty,
                    Type::Primitive(PrimitiveType::I32) | Type::TypeVar(_)
                ) {
                    return Err(InferenceError::TypeMismatch {
                        expected: Type::Primitive(PrimitiveType::I32),
                        found: index_ty.clone(),
                    });
                }
                if !matches!(
                    element_ty,
                    Type::Primitive(PrimitiveType::Char) | Type::TypeVar(_)
                ) {
                    return Err(InferenceError::TypeMismatch {
                        expected: Type::Primitive(PrimitiveType::Char),
                        found: element_ty.clone(),
                    });
                }
                Ok(())
            }

            // Slices are represented as dynamic arrays (Array with size: None)
            // This case is already handled by the Array case above
            Type::TypeVar(_) => {
                // Type variables are handled during unification
                Ok(())
            }

            _ => {
                // Type is not indexable
                Err(InferenceError::TypeMismatch {
                    expected: Type::Array {
                        element_type: Box::new(Type::Never),
                        size: None,
                        nullability: crate::type_registry::NullabilityKind::default(),
                    },
                    found: container_ty.clone(),
                })
            }
        }
    }

    /// Add a field access constraint: object_type has field field_name of type field_type
    pub fn add_has_field_constraint(
        &mut self,
        object_type: Type,
        field_name: InternedString,
        field_type: Type,
    ) -> Result<(), InferenceError> {
        self.add_constraint(Constraint::HasField(object_type, field_name, field_type));
        Ok(())
    }

    /// Add an indexable constraint: container_type[index_type] = element_type
    pub fn add_indexable_constraint(
        &mut self,
        container_type: Type,
        index_type: Type,
        element_type: Type,
    ) -> Result<(), InferenceError> {
        self.add_constraint(Constraint::Indexable(
            container_type,
            index_type,
            element_type,
        ));
        Ok(())
    }
}
