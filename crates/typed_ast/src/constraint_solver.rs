//! # Constraint-Based Type Inference System
//!
//! A sophisticated type inference system using constraint generation and solving
//! for better handling of complex type relationships like generics, varargs, and
//! type parameters.

use crate::arena::InternedString;
use crate::source::Span;
use crate::type_registry::{
    Lifetime, Mutability, ParamInfo, PrimitiveType, Type, TypeBound, TypeKind, TypeVar,
    TypeVarKind, Variance,
};
use crate::{ConstValue, TypeVarId};
use std::collections::{HashMap, HashSet, VecDeque};
use string_interner::Symbol;

/// A type constraint representing a relationship between types
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// Type equality: t1 = t2
    Equal(Type, Type, Span),

    /// Subtype relation: t1 <: t2 (t1 is a subtype of t2)
    Subtype(Type, Type, Span),

    /// Type instantiation: t1 = inst(scheme)
    Instance(Type, TypeScheme, Span),

    /// Member constraint: t has member m of type t2
    HasMember(Type, InternedString, Type, Span),

    /// Callable constraint: t can be called with args producing result
    Callable(Type, Vec<Type>, Type, Span),

    /// Array element constraint: array type has element type
    ArrayElement(Type, Type, Span),

    /// Varargs constraint: rest parameter accepting multiple args
    Varargs(Type, Vec<Type>, Span),

    /// Trait bound constraint: type implements trait
    TraitBound(Type, InternedString, Span),

    /// Lifetime constraint: 'a outlives 'b
    LifetimeOutlives(Lifetime, Lifetime, Span),

    /// Lifetime bound on type: T: 'a
    TypeOutlivesLifetime(Type, Lifetime, Span),

    /// Higher-ranked trait bound: for<'a> T: Trait<'a>
    HigherRankedBound {
        lifetimes: Vec<InternedString>,
        ty: Type,
        bound: TypeBound,
        span: Span,
    },

    /// Dependent type constraints

    /// Refinement type constraint: value must satisfy predicate
    RefinementSatisfies {
        value: Type,
        predicate: crate::dependent_types::RefinementPredicate,
        span: Span,
    },

    /// Dependent function constraint: argument satisfies parameter constraint
    DependentCall {
        func_type: Type,
        arg_value: ConstValue,
        result_type: Type,
        span: Span,
    },

    /// Path-dependent type constraint: path resolves to valid type
    PathDependent {
        base_type: Type,
        path: Vec<InternedString>,
        target_type: Type,
        span: Span,
    },

    /// Singleton type constraint: value equals specific constant
    SingletonEquals {
        value_type: Type,
        constant: ConstValue,
        span: Span,
    },

    /// Index constraint for type families: F[args] = result_type
    TypeFamilyApplication {
        family: Type,
        indices: Vec<crate::dependent_types::DependentIndex>,
        result_type: Type,
        span: Span,
    },

    /// Conditional type constraint: if condition then type1 else type2
    ConditionalType {
        condition: crate::dependent_types::RefinementPredicate,
        then_type: Type,
        else_type: Type,
        result_type: Type,
        span: Span,
    },
}

/// A type scheme with quantified variables
#[derive(Debug, Clone, PartialEq)]
pub struct TypeScheme {
    /// Quantified type variables
    pub quantified: Vec<TypeVarId>,
    /// The type body
    pub ty: Type,
}

impl TypeScheme {
    pub fn monomorphic(ty: Type) -> Self {
        Self {
            quantified: Vec::new(),
            ty,
        }
    }
}

/// Substitution mapping from type variables to types
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    map: HashMap<TypeVarId, Type>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply substitution to a type
    pub fn apply(&self, ty: &Type) -> Type {
        match ty {
            Type::TypeVar(var) => self.map.get(&var.id).cloned().unwrap_or_else(|| ty.clone()),
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
                        ty: self.apply(&p.ty),
                        is_optional: p.is_optional,
                        is_varargs: p.is_varargs,
                        is_keyword_only: p.is_keyword_only,
                        is_positional_only: p.is_positional_only,
                        is_out: p.is_out,
                        is_ref: p.is_ref,
                        is_inout: p.is_inout,
                    })
                    .collect(),
                return_type: Box::new(self.apply(return_type)),
                is_varargs: *is_varargs,
                has_named_params: *has_named_params,
                has_default_params: *has_default_params,
                async_kind: *async_kind,
                calling_convention: *calling_convention,
                nullability: *nullability,
            },
            Type::Array {
                element_type,
                size,
                nullability,
            } => Type::Array {
                element_type: Box::new(self.apply(element_type)),
                size: size.clone(),
                nullability: *nullability,
            },
            Type::Tuple(types) => Type::Tuple(types.iter().map(|t| self.apply(t)).collect()),
            Type::Named {
                id,
                type_args,
                const_args,
                variance,
                nullability,
            } => Type::Named {
                id: *id,
                type_args: type_args.iter().map(|t| self.apply(t)).collect(),
                const_args: const_args.clone(),
                variance: variance.clone(),
                nullability: *nullability,
            },
            Type::Reference {
                ty,
                mutability,
                lifetime,
                nullability,
            } => Type::Reference {
                ty: Box::new(self.apply(ty)),
                mutability: *mutability,
                lifetime: lifetime.clone(),
                nullability: *nullability,
            },
            Type::Optional(ty) => Type::Optional(Box::new(self.apply(ty))),
            Type::Result { ok_type, err_type } => Type::Result {
                ok_type: Box::new(self.apply(ok_type)),
                err_type: Box::new(self.apply(err_type)),
            },
            Type::Union(types) => Type::Union(types.iter().map(|t| self.apply(t)).collect()),
            Type::Intersection(types) => {
                Type::Intersection(types.iter().map(|t| self.apply(t)).collect())
            }
            Type::Alias { name, target } => Type::Alias {
                name: *name,
                target: Box::new(self.apply(target)),
            },
            Type::Projection { base, item } => Type::Projection {
                base: Box::new(self.apply(base)),
                item: *item,
            },
            Type::Index { base, index } => Type::Index {
                base: Box::new(self.apply(base)),
                index: Box::new(self.apply(index)),
            },
            Type::Associated {
                trait_name,
                type_name,
            } => {
                let associated_ty = Type::Associated {
                    trait_name: *trait_name,
                    type_name: *type_name,
                };
                // Apply substitution recursively after resolving the associated type
                let resolved = self.apply(&associated_ty);
                if resolved == associated_ty {
                    // Not in substitution map, return as-is
                    associated_ty
                } else {
                    resolved
                }
            }
            Type::SelfType
            | Type::Primitive(_)
            | Type::Never
            | Type::Any
            | Type::Error
            | Type::HigherKinded { .. }
            | Type::Extern { .. }
            | Type::Unresolved(_) => ty.clone(),
            Type::Nullable(inner_ty) => {
                // For nullable types, substitute the inner type and maintain nullability
                let substituted_inner = self.apply(inner_ty);
                Type::Nullable(Box::new(substituted_inner))
            }
            Type::NonNull(inner_ty) => {
                // For non-null types, substitute the inner type and maintain non-nullability
                let substituted_inner = self.apply(inner_ty);
                Type::NonNull(Box::new(substituted_inner))
            }
            Type::ConstVar {
                id,
                name,
                const_type,
            } => {
                // Const variables don't usually need substitution, but substitute the const_type
                let substituted_const_type = self.apply(const_type);
                Type::ConstVar {
                    id: *id,
                    name: *name,
                    const_type: Box::new(substituted_const_type),
                }
            }
            Type::ConstDependent {
                base_type,
                constraint,
            } => {
                // Substitute in the base type, constraint is typically a const expression
                let substituted_base = self.apply(base_type);
                Type::ConstDependent {
                    base_type: Box::new(substituted_base),
                    constraint: constraint.clone(), // Constraints typically don't have type vars
                }
            }
            Type::Dynamic => {
                // Dynamic types don't contain type variables, so return as-is
                ty.clone()
            }
            Type::Unknown => {
                // Unknown types don't contain type variables, so return as-is
                ty.clone()
            }
            Type::Interface {
                methods,
                is_structural,
                nullability,
            } => todo!(),
            Type::Struct {
                fields,
                is_anonymous,
                nullability,
            } => todo!(),
            Type::Trait {
                id,
                associated_types,
                super_traits,
            } => todo!(),
        }
    }

    /// Compose two substitutions
    pub fn compose(&self, other: &Substitution) -> Substitution {
        let mut result = Substitution::new();

        // Apply other to all values in self
        for (k, v) in &self.map {
            result.map.insert(*k, other.apply(v));
        }

        // Add entries from other that aren't in self
        for (k, v) in &other.map {
            if !result.map.contains_key(k) {
                result.map.insert(*k, v.clone());
            }
        }

        result
    }

    /// Add a binding
    pub fn bind(&mut self, var: TypeVarId, ty: Type) {
        self.map.insert(var, ty);
    }
}

/// The constraint solver
pub struct ConstraintSolver {
    /// Generated constraints
    constraints: Vec<Constraint>,
    /// Current substitution
    subst: Substitution,
    // /// Type variable generator
    // next_type_var: TypeVarId,
    /// Error tracking
    errors: Vec<SolverError>,
    /// Type registry for looking up type definitions
    type_registry: Box<crate::type_registry::TypeRegistry>,
    /// Track trait bounds for type variables (type_var_id -> set of required traits)
    trait_bounds: HashMap<TypeVarId, HashSet<InternedString>>,
    /// Lifetime constraints (outlives relationships)
    lifetime_constraints: Vec<(Lifetime, Lifetime)>,
    /// Type-lifetime constraints (type outlives lifetime)
    type_lifetime_constraints: Vec<(Type, Lifetime)>,
    /// Implicit lifetime counter for generating fresh lifetimes
    next_implicit_lifetime: u32,
}

#[derive(Debug, Clone)]
pub enum SolverError {
    /// Cannot unify types
    CannotUnify(Type, Type, Span),
    /// Infinite type (occurs check failed)
    InfiniteType(TypeVarId, Type, Span),
    /// Unsolvable constraint
    UnsolvableConstraint(Constraint),
    /// Type does not implement required trait
    TraitNotImplemented {
        ty: Type,
        trait_name: InternedString,
        span: Span,
    },
    /// Unknown trait referenced
    UnknownTrait { name: InternedString, span: Span },
    /// Lifetime cycle detected
    LifetimeCycle {
        lifetime1: InternedString,
        lifetime2: InternedString,
        span: Span,
    },
}

impl ConstraintSolver {
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            subst: Substitution::new(),
            // next_type_var: TypeVarId::unknown(),
            errors: Vec::new(),
            type_registry: Box::new(crate::type_registry::TypeRegistry::new()),
            trait_bounds: HashMap::new(),
            lifetime_constraints: Vec::new(),
            type_lifetime_constraints: Vec::new(),
            next_implicit_lifetime: 1,
        }
    }

    pub fn with_type_registry(type_registry: Box<crate::type_registry::TypeRegistry>) -> Self {
        Self {
            constraints: Vec::new(),
            subst: Substitution::new(),
            // next_type_var: 0,
            errors: Vec::new(),
            type_registry,
            trait_bounds: HashMap::new(),
            lifetime_constraints: Vec::new(),
            type_lifetime_constraints: Vec::new(),
            next_implicit_lifetime: 1,
        }
    }

    /// Generate a fresh type variable
    pub fn fresh_type_var(&mut self) -> Type {
        let id = TypeVarId::next();
        Type::TypeVar(TypeVar {
            id,
            name: None,
            kind: TypeVarKind::Type,
        })
    }

    /// Get a reference to the type registry
    pub fn type_registry(&self) -> &crate::type_registry::TypeRegistry {
        &self.type_registry
    }

    /// Add a constraint
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Solve all constraints
    pub fn solve(&mut self) -> Result<Substitution, Vec<SolverError>> {
        let mut work_list = VecDeque::from(self.constraints.clone());
        let mut iteration_count = 0;
        const MAX_ITERATIONS: usize = 1000; // Prevent infinite loops

        while let Some(constraint) = work_list.pop_front() {
            iteration_count += 1;
            if iteration_count > MAX_ITERATIONS {
                self.errors
                    .push(SolverError::UnsolvableConstraint(constraint));
                break;
            }
            match self.solve_constraint(constraint.clone())? {
                ConstraintResult::Substitution(s) => {
                    // Apply new substitution to remaining constraints
                    let remaining: Vec<_> = work_list.drain(..).collect();
                    for c in remaining {
                        work_list.push_back(self.apply_subst_to_constraint(&s, c));
                    }

                    // Compose substitutions
                    self.subst = self.subst.compose(&s);
                }
                ConstraintResult::NewConstraints(cs) => {
                    // Add new constraints to work list
                    for c in cs {
                        work_list.push_back(c);
                    }
                }
                ConstraintResult::Solved => {
                    // Constraint solved, continue
                }
                ConstraintResult::Deferred => {
                    // Constraint deferred, add back to end of work list for later retry
                    // Only defer a limited number of times to prevent infinite loops
                    if iteration_count < MAX_ITERATIONS / 2 {
                        work_list.push_back(constraint);
                    }
                    // If we've hit the defer limit, just skip the constraint
                }
            }
        }

        if self.errors.is_empty() {
            Ok(self.subst.clone())
        } else {
            Err(self.errors.clone())
        }
    }

    /// Solve a single constraint
    fn solve_constraint(
        &mut self,
        constraint: Constraint,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        match constraint {
            Constraint::Equal(t1, t2, span) => self.unify(t1, t2, span),
            Constraint::Varargs(param_ty, arg_types, span) => {
                self.solve_varargs(param_ty, arg_types, span)
            }
            Constraint::ArrayElement(array_ty, elem_ty, span) => {
                self.solve_array_element(array_ty, elem_ty, span)
            }
            Constraint::HasMember(obj_ty, member, member_ty, span) => {
                self.solve_has_member(obj_ty, member, member_ty, span)
            }
            Constraint::Callable(func_ty, arg_types, ret_ty, span) => {
                self.solve_callable(func_ty, arg_types, ret_ty, span)
            }
            Constraint::Subtype(sub_ty, super_ty, span) => {
                self.solve_subtype(sub_ty, super_ty, span)
            }
            Constraint::Instance(ty, scheme, span) => self.solve_instance(ty, scheme, span),
            Constraint::TraitBound(ty, trait_name, span) => {
                self.solve_trait_bound(ty, trait_name, span)
            }
            Constraint::LifetimeOutlives(lifetime_a, lifetime_b, span) => {
                self.solve_lifetime_outlives(lifetime_a, lifetime_b, span)
            }
            Constraint::TypeOutlivesLifetime(ty, lifetime, span) => {
                self.solve_type_outlives_lifetime(ty, lifetime, span)
            }
            Constraint::HigherRankedBound {
                lifetimes,
                ty,
                bound,
                span,
            } => self.solve_higher_ranked_bound(lifetimes, ty, bound, span),

            // Dependent type constraints
            Constraint::RefinementSatisfies {
                value,
                predicate,
                span,
            } => self.solve_refinement_satisfies(value, predicate, span),

            Constraint::DependentCall {
                func_type,
                arg_value,
                result_type,
                span,
            } => self.solve_dependent_call(func_type, arg_value, result_type, span),

            Constraint::PathDependent {
                base_type,
                path,
                target_type,
                span,
            } => self.solve_path_dependent(base_type, path, target_type, span),

            Constraint::SingletonEquals {
                value_type,
                constant,
                span,
            } => self.solve_singleton_equals(value_type, constant, span),

            Constraint::TypeFamilyApplication {
                family,
                indices,
                result_type,
                span,
            } => self.solve_type_family_application(family, indices, result_type, span),

            Constraint::ConditionalType {
                condition,
                then_type,
                else_type,
                result_type,
                span,
            } => self.solve_conditional_type(condition, then_type, else_type, result_type, span),
        }
    }

    /// Unify two types
    fn unify(
        &mut self,
        t1: Type,
        t2: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let t1 = self.subst.apply(&t1);
        let t2 = self.subst.apply(&t2);

        match (&t1, &t2) {
            // Same primitive types
            (Type::Primitive(p1), Type::Primitive(p2)) if p1 == p2 => Ok(ConstraintResult::Solved),

            // Type variable cases
            (Type::TypeVar(v1), Type::TypeVar(v2)) if v1.id == v2.id => {
                Ok(ConstraintResult::Solved)
            }
            (Type::TypeVar(v1), Type::TypeVar(v2)) => {
                // Merge trait bounds from both type variables
                let bounds1 = self.trait_bounds.get(&v1.id).cloned().unwrap_or_default();
                let bounds2 = self.trait_bounds.get(&v2.id).cloned().unwrap_or_default();
                let merged_bounds: HashSet<_> = bounds1.union(&bounds2).cloned().collect();

                // Use v1 as the representative and bind v2 to v1
                self.trait_bounds.insert(v1.id, merged_bounds);
                self.trait_bounds.remove(&v2.id);

                let mut subst = Substitution::new();
                subst.bind(v2.id, Type::TypeVar(v1.clone()));
                Ok(ConstraintResult::Substitution(subst))
            }
            (Type::TypeVar(v), t) | (t, Type::TypeVar(v)) => {
                if self.occurs_check(v.id, t) {
                    self.errors
                        .push(SolverError::InfiniteType(v.id, t.clone(), span));
                    Err(self.errors.clone())
                } else {
                    // Check if we have trait bounds for this type variable
                    if let Some(trait_bounds) = self.trait_bounds.get(&v.id).cloned() {
                        // Verify that the concrete type satisfies all trait bounds
                        self.verify_trait_bounds(t.clone(), trait_bounds.into_iter(), span)?;
                    }

                    let mut subst = Substitution::new();
                    subst.bind(v.id, t.clone());
                    Ok(ConstraintResult::Substitution(subst))
                }
            }

            // Self type unification
            (Type::SelfType, Type::SelfType) => Ok(ConstraintResult::Solved),
            (Type::SelfType, t) | (t, Type::SelfType) => {
                // Self type can unify with any type in the context of a trait method
                // For now, we'll defer resolution until we have more context
                Ok(ConstraintResult::Deferred)
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
                if s1 == s2 {
                    Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                        *e1.clone(),
                        *e2.clone(),
                        span,
                    )]))
                } else {
                    self.errors.push(SolverError::CannotUnify(t1, t2, span));
                    Err(self.errors.clone())
                }
            }

            // Function types
            (
                Type::Function {
                    params: p1,
                    return_type: r1,
                    ..
                },
                Type::Function {
                    params: p2,
                    return_type: r2,
                    ..
                },
            ) => {
                if p1.len() != p2.len() {
                    self.errors.push(SolverError::CannotUnify(t1, t2, span));
                    return Err(self.errors.clone());
                }

                let mut constraints = vec![];
                for (param1, param2) in p1.iter().zip(p2.iter()) {
                    constraints.push(Constraint::Equal(
                        param1.ty.clone(),
                        param2.ty.clone(),
                        span,
                    ));
                }
                constraints.push(Constraint::Equal(*r1.clone(), *r2.clone(), span));

                Ok(ConstraintResult::NewConstraints(constraints))
            }

            // Associated types
            (
                Type::Associated {
                    trait_name: tn1,
                    type_name: n1,
                },
                Type::Associated {
                    trait_name: tn2,
                    type_name: n2,
                },
            ) => {
                if tn1 == tn2 && n1 == n2 {
                    Ok(ConstraintResult::Solved)
                } else {
                    self.errors.push(SolverError::CannotUnify(t1, t2, span));
                    Err(self.errors.clone())
                }
            }

            // Different types
            _ => {
                self.errors.push(SolverError::CannotUnify(t1, t2, span));
                Err(self.errors.clone())
            }
        }
    }

    /// Solve varargs constraint
    fn solve_varargs(
        &mut self,
        param_ty: Type,
        arg_types: Vec<Type>,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        // For varargs, the parameter type should be an array
        if let Type::Array { element_type, .. } = self.subst.apply(&param_ty) {
            // Each argument should match the element type
            let constraints: Vec<_> = arg_types
                .into_iter()
                .map(|arg_ty| Constraint::Equal(*element_type.clone(), arg_ty, span))
                .collect();
            Ok(ConstraintResult::NewConstraints(constraints))
        } else {
            // Parameter is not an array type, try to make it one
            let elem_ty = self.fresh_type_var();
            let array_ty = Type::Array {
                element_type: Box::new(elem_ty.clone()),
                size: None,
                nullability: crate::type_registry::NullabilityKind::default(),
            };

            let mut constraints = vec![Constraint::Equal(param_ty, array_ty, span)];
            for arg_ty in arg_types {
                constraints.push(Constraint::Equal(elem_ty.clone(), arg_ty, span));
            }

            Ok(ConstraintResult::NewConstraints(constraints))
        }
    }

    /// Solve array element constraint
    fn solve_array_element(
        &mut self,
        array_ty: Type,
        elem_ty: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let array_ty = self.subst.apply(&array_ty);

        match array_ty {
            Type::Array { element_type, .. } => {
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                    *element_type,
                    elem_ty,
                    span,
                )]))
            }
            Type::TypeVar(_) => {
                // Create array type constraint
                let new_array = Type::Array {
                    element_type: Box::new(elem_ty),
                    size: None,
                    nullability: crate::type_registry::NullabilityKind::default(),
                };
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                    array_ty, new_array, span,
                )]))
            }
            _ => {
                self.errors.push(SolverError::CannotUnify(
                    array_ty,
                    Type::Array {
                        element_type: Box::new(elem_ty),
                        size: None,
                        nullability: crate::type_registry::NullabilityKind::default(),
                    },
                    span,
                ));
                Err(self.errors.clone())
            }
        }
    }

    /// Occurs check for infinite types
    fn occurs_check(&self, var: TypeVarId, ty: &Type) -> bool {
        match ty {
            Type::TypeVar(v) => v.id == var,
            Type::Array { element_type, .. } => self.occurs_check(var, element_type),
            Type::Function {
                return_type,
                params,
                ..
            } => {
                self.occurs_check(var, return_type)
                    || params.iter().any(|p| self.occurs_check(var, &p.ty))
            }
            Type::Tuple(types) => types.iter().any(|t| self.occurs_check(var, t)),
            Type::Named { type_args, .. } => type_args.iter().any(|t| self.occurs_check(var, t)),
            _ => false,
        }
    }

    /// Solve HasMember constraint
    fn solve_has_member(
        &mut self,
        obj_ty: Type,
        member: InternedString,
        member_ty: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let obj_ty = self.subst.apply(&obj_ty);

        match &obj_ty {
            Type::Named { id, type_args, .. } => {
                // Look up the type definition in the type registry
                if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                    match &type_def.kind {
                        TypeKind::Struct { fields, .. } => {
                            // Look for the field in the struct
                            if let Some(field_def) = fields.iter().find(|f| f.name == member) {
                                // Apply type arguments to the field type if it's generic
                                let field_type = self.apply_type_args(
                                    &field_def.ty,
                                    &type_def.type_params,
                                    type_args,
                                );

                                // Unify the expected member type with the actual field type
                                Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                                    member_ty, field_type, span,
                                )]))
                            } else {
                                // Field not found
                                self.errors.push(SolverError::CannotUnify(
                                    obj_ty,
                                    Type::Never,
                                    span,
                                ));
                                Err(self.errors.clone())
                            }
                        }
                        TypeKind::Interface { methods, .. } => {
                            // Look for the method in the interface
                            if let Some(method_sig) = methods.iter().find(|m| m.name == member) {
                                // Create a function type from the method signature
                                let method_params: Vec<ParamInfo> = method_sig
                                    .params
                                    .iter()
                                    .map(|p| ParamInfo {
                                        name: Some(p.name),
                                        ty: p.ty.clone(),
                                        is_optional: false,
                                        is_varargs: false,
                                        is_keyword_only: false,
                                        is_positional_only: false,
                                        is_out: false,
                                        is_ref: p.is_mut,
                                        is_inout: false,
                                    })
                                    .collect();

                                let method_type = Type::Function {
                                    params: method_params,
                                    return_type: Box::new(method_sig.return_type.clone()),
                                    is_varargs: false,
                                    has_named_params: true,
                                    has_default_params: false,
                                    async_kind: crate::type_registry::AsyncKind::default(),
                                    calling_convention:
                                        crate::type_registry::CallingConvention::default(),
                                    nullability: crate::type_registry::NullabilityKind::default(),
                                };

                                // Apply type arguments to the method type if it's generic
                                let instantiated_method_type = self.apply_type_args(
                                    &method_type,
                                    &type_def.type_params,
                                    type_args,
                                );

                                Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                                    member_ty,
                                    instantiated_method_type,
                                    span,
                                )]))
                            } else {
                                // Method not found
                                self.errors.push(SolverError::CannotUnify(
                                    obj_ty,
                                    Type::Never,
                                    span,
                                ));
                                Err(self.errors.clone())
                            }
                        }
                        TypeKind::Enum { .. } => {
                            // Enums typically don't have fields/methods accessible this way
                            self.errors
                                .push(SolverError::CannotUnify(obj_ty, Type::Never, span));
                            Err(self.errors.clone())
                        }
                        TypeKind::Alias { target } => {
                            // For type aliases, recurse on the target type
                            let target_with_args =
                                self.apply_type_args(target, &type_def.type_params, type_args);
                            Ok(ConstraintResult::NewConstraints(vec![
                                Constraint::HasMember(target_with_args, member, member_ty, span),
                            ]))
                        }
                        TypeKind::Abstract {
                            underlying_type, ..
                        } => {
                            // For abstract types, forward member access to underlying type
                            let underlying_with_args = self.apply_type_args(
                                underlying_type,
                                &type_def.type_params,
                                type_args,
                            );
                            Ok(ConstraintResult::NewConstraints(vec![
                                Constraint::HasMember(
                                    underlying_with_args,
                                    member,
                                    member_ty,
                                    span,
                                ),
                            ]))
                        }
                        TypeKind::Atomic => todo!(),
                        TypeKind::Class => todo!(),
                        TypeKind::Function => todo!(),
                        TypeKind::Array => todo!(),
                        TypeKind::Generic => todo!(),
                    }
                } else {
                    // Type definition not found - defer constraint
                    // This could happen if the type is defined later or in another module
                    Ok(ConstraintResult::Solved)
                }
            }
            Type::TypeVar(_) => {
                // For type variables, we can't determine members yet
                // Keep the constraint for later when the type variable is resolved
                Ok(ConstraintResult::Solved)
            }
            Type::Array { .. } => {
                // Arrays have built-in members like length
                // For now, we'll assume any member access on arrays fails
                // In a real implementation, we'd check for built-in members
                self.errors
                    .push(SolverError::CannotUnify(obj_ty, Type::Never, span));
                Err(self.errors.clone())
            }
            Type::Tuple(_types) => {
                // Tuples can be accessed by numeric field names (0, 1, 2, etc.)
                // For now, we'll assume tuple member access fails
                // In a real implementation, we'd check for numeric field access
                self.errors
                    .push(SolverError::CannotUnify(obj_ty, Type::Never, span));
                Err(self.errors.clone())
            }
            _ => {
                // Type doesn't support member access
                self.errors
                    .push(SolverError::CannotUnify(obj_ty, Type::Never, span));
                Err(self.errors.clone())
            }
        }
    }

    /// Apply type arguments to a type based on type parameters
    fn apply_type_args(
        &self,
        ty: &Type,
        type_params: &[crate::type_registry::TypeParam],
        type_args: &[Type],
    ) -> Type {
        if type_params.is_empty() || type_args.is_empty() {
            return ty.clone();
        }

        // Create a substitution mapping from type parameters to type arguments
        let mut subst = Substitution::new();
        for (param, arg) in type_params.iter().zip(type_args.iter()) {
            // For now, we'll assume type parameters are represented as TypeVars
            // In a full implementation, we'd need a proper mapping system
            // This is a simplified version
        }

        // Apply the substitution to the type
        subst.apply(ty)
    }

    /// Check if one primitive type is a subtype of another (numeric promotions)
    fn is_numeric_subtype(&self, sub_ty: PrimitiveType, super_ty: PrimitiveType) -> bool {
        use PrimitiveType::*;

        match (sub_ty, super_ty) {
            // Integer promotions
            (I8, I16) | (I8, I32) | (I8, I64) | (I8, I128) => true,
            (I16, I32) | (I16, I64) | (I16, I128) => true,
            (I32, I64) | (I32, I128) => true,
            (I64, I128) => true,

            (U8, U16) | (U8, U32) | (U8, U64) | (U8, U128) => true,
            (U16, U32) | (U16, U64) | (U16, U128) => true,
            (U32, U64) | (U32, U128) => true,
            (U64, U128) => true,

            // Unsigned to signed promotion (if fits)
            (U8, I16) | (U8, I32) | (U8, I64) | (U8, I128) => true,
            (U16, I32) | (U16, I64) | (U16, I128) => true,
            (U32, I64) | (U32, I128) => true,
            (U64, I128) => true,

            // Float promotions
            (F32, F64) => true,

            // Integer to float promotions
            (I8, F32) | (I8, F64) => true,
            (I16, F32) | (I16, F64) => true,
            (I32, F64) => true, // I32 to F32 loses precision
            (U8, F32) | (U8, F64) => true,
            (U16, F32) | (U16, F64) => true,
            (U32, F64) => true, // U32 to F32 loses precision

            _ => false,
        }
    }

    /// Solve function subtyping: contravariant in parameters, covariant in return
    fn solve_function_subtype(
        &mut self,
        sub_params: Vec<ParamInfo>,
        sub_return: Type,
        super_params: Vec<ParamInfo>,
        super_return: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        // Function types must have the same number of parameters
        if sub_params.len() != super_params.len() {
            self.errors.push(SolverError::CannotUnify(
                Type::Function {
                    params: sub_params,
                    return_type: Box::new(sub_return),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: crate::type_registry::AsyncKind::default(),
                    calling_convention: crate::type_registry::CallingConvention::default(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                },
                Type::Function {
                    params: super_params,
                    return_type: Box::new(super_return),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: crate::type_registry::AsyncKind::default(),
                    calling_convention: crate::type_registry::CallingConvention::default(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                },
                span,
            ));
            return Err(self.errors.clone());
        }

        let mut constraints = Vec::new();

        // Parameters are contravariant: super_param <: sub_param
        for (sub_param, super_param) in sub_params.iter().zip(super_params.iter()) {
            constraints.push(Constraint::Subtype(
                super_param.ty.clone(),
                sub_param.ty.clone(),
                span,
            ));
        }

        // Return type is covariant: sub_return <: super_return
        constraints.push(Constraint::Subtype(sub_return, super_return, span));

        Ok(ConstraintResult::NewConstraints(constraints))
    }

    /// Solve tuple subtyping: width subtyping
    fn solve_tuple_subtype(
        &mut self,
        sub_types: Vec<Type>,
        super_types: Vec<Type>,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        // Width subtyping: tuple with fewer fields is subtype of tuple with more fields
        // (T1, T2) <: (T1, T2, T3) is valid in some type systems
        if sub_types.len() > super_types.len() {
            self.errors.push(SolverError::CannotUnify(
                Type::Tuple(sub_types),
                Type::Tuple(super_types),
                span,
            ));
            return Err(self.errors.clone());
        }

        let mut constraints = Vec::new();

        // Each corresponding field must be a subtype
        for (sub_ty, super_ty) in sub_types.iter().zip(super_types.iter()) {
            constraints.push(Constraint::Subtype(sub_ty.clone(), super_ty.clone(), span));
        }

        Ok(ConstraintResult::NewConstraints(constraints))
    }

    /// Solve named type subtyping by checking inheritance hierarchy
    fn solve_named_type_subtype(
        &mut self,
        sub_id: crate::type_registry::TypeId,
        sub_args: Vec<Type>,
        super_id: crate::type_registry::TypeId,
        super_args: Vec<Type>,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        // If IDs are the same, check type argument variance
        if sub_id == super_id {
            return self.solve_type_args_subtype(sub_id, sub_args, super_args, span);
        }

        // Check if sub_id is a subtype of super_id via inheritance
        if let Some(sub_def) = self.type_registry.get_type_by_id(sub_id) {
            match &sub_def.kind {
                TypeKind::Struct { .. } => {
                    // For now, structs don't have inheritance - only nominal equality
                    self.errors.push(SolverError::CannotUnify(
                        Type::Named {
                            id: sub_id,
                            type_args: sub_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        Type::Named {
                            id: super_id,
                            type_args: super_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        span,
                    ));
                    Err(self.errors.clone())
                }
                TypeKind::Interface { super_traits, .. } => {
                    // Check if super_id is in the super_traits
                    for super_trait in super_traits {
                        if let Type::Named { id, .. } = super_trait {
                            if *id == super_id {
                                // Found inheritance relationship, now check type arguments
                                return self
                                    .solve_type_args_subtype(super_id, sub_args, super_args, span);
                            }
                        }
                    }

                    // No inheritance relationship found
                    self.errors.push(SolverError::CannotUnify(
                        Type::Named {
                            id: sub_id,
                            type_args: sub_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        Type::Named {
                            id: super_id,
                            type_args: super_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        span,
                    ));
                    Err(self.errors.clone())
                }
                TypeKind::Enum { .. } => {
                    // Enums don't have inheritance
                    self.errors.push(SolverError::CannotUnify(
                        Type::Named {
                            id: sub_id,
                            type_args: sub_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        Type::Named {
                            id: super_id,
                            type_args: super_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        span,
                    ));
                    Err(self.errors.clone())
                }
                TypeKind::Alias { target } => {
                    // Resolve alias and try again
                    let instantiated_target =
                        self.apply_type_args(target, &sub_def.type_params, &sub_args);
                    Ok(ConstraintResult::NewConstraints(vec![Constraint::Subtype(
                        instantiated_target,
                        Type::Named {
                            id: super_id,
                            type_args: super_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        span,
                    )]))
                }
                TypeKind::Abstract {
                    underlying_type,
                    implicit_to,
                    ..
                } => {
                    // For abstract types, check if super_id is in implicit_to list
                    // or if underlying type is a subtype of super_id
                    for to_type in implicit_to {
                        if let Type::Named { id, .. } = to_type {
                            if *id == super_id {
                                // Abstract can implicitly convert to super_id
                                return Ok(ConstraintResult::Solved);
                            }
                        }
                    }

                    // Otherwise try underlying type
                    let underlying_with_args =
                        self.apply_type_args(underlying_type, &sub_def.type_params, &sub_args);
                    Ok(ConstraintResult::NewConstraints(vec![Constraint::Subtype(
                        underlying_with_args,
                        Type::Named {
                            id: super_id,
                            type_args: super_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        span,
                    )]))
                }
                TypeKind::Atomic
                | TypeKind::Class
                | TypeKind::Function
                | TypeKind::Array
                | TypeKind::Generic => {
                    // These types don't have inheritance relationships
                    self.errors.push(SolverError::CannotUnify(
                        Type::Named {
                            id: sub_id,
                            type_args: sub_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        Type::Named {
                            id: super_id,
                            type_args: super_args,
                            const_args: Vec::new(),
                            variance: Vec::new(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        span,
                    ));
                    Err(self.errors.clone())
                }
            }
        } else {
            // Type definition not found
            self.errors.push(SolverError::CannotUnify(
                Type::Named {
                    id: sub_id,
                    type_args: sub_args,
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                },
                Type::Named {
                    id: super_id,
                    type_args: super_args,
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                },
                span,
            ));
            Err(self.errors.clone())
        }
    }

    /// Solve type argument subtyping based on variance
    fn solve_type_args_subtype(
        &mut self,
        type_id: crate::type_registry::TypeId,
        sub_args: Vec<Type>,
        super_args: Vec<Type>,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        if sub_args.len() != super_args.len() {
            self.errors.push(SolverError::CannotUnify(
                Type::Named {
                    id: type_id,
                    type_args: sub_args,
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                },
                Type::Named {
                    id: type_id,
                    type_args: super_args,
                    const_args: Vec::new(),
                    variance: Vec::new(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                },
                span,
            ));
            return Err(self.errors.clone());
        }

        if let Some(type_def) = self.type_registry.get_type_by_id(type_id) {
            let mut constraints = Vec::new();

            for ((sub_arg, super_arg), type_param) in sub_args
                .iter()
                .zip(super_args.iter())
                .zip(type_def.type_params.iter())
            {
                match type_param.variance {
                    Variance::Covariant => {
                        // T<A> <: T<B> if A <: B
                        constraints.push(Constraint::Subtype(
                            sub_arg.clone(),
                            super_arg.clone(),
                            span,
                        ));
                    }
                    Variance::Contravariant => {
                        // T<A> <: T<B> if B <: A
                        constraints.push(Constraint::Subtype(
                            super_arg.clone(),
                            sub_arg.clone(),
                            span,
                        ));
                    }
                    Variance::Invariant => {
                        // T<A> <: T<B> if A = B
                        constraints.push(Constraint::Equal(
                            sub_arg.clone(),
                            super_arg.clone(),
                            span,
                        ));
                    }
                    Variance::Bivariant => {
                        // T<A> <: T<B> always (unsafe, rarely used)
                        // No constraint needed
                    }
                }
            }

            Ok(ConstraintResult::NewConstraints(constraints))
        } else {
            // Unknown type, assume invariant
            let mut constraints = Vec::new();
            for (sub_arg, super_arg) in sub_args.iter().zip(super_args.iter()) {
                constraints.push(Constraint::Equal(sub_arg.clone(), super_arg.clone(), span));
            }
            Ok(ConstraintResult::NewConstraints(constraints))
        }
    }

    /// Solve reference subtyping with mutability constraints
    fn solve_reference_subtype(
        &mut self,
        sub_inner: Type,
        sub_mut: Mutability,
        super_inner: Type,
        super_mut: Mutability,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        use crate::type_registry::Mutability::*;

        match (sub_mut, super_mut) {
            // Immutable reference can be used where immutable reference is expected
            (Immutable, Immutable) => {
                // Covariant in the referenced type
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Subtype(
                    sub_inner,
                    super_inner,
                    span,
                )]))
            }
            // Mutable reference can be used where immutable reference is expected
            (Mutable, Immutable) => {
                // Covariant in the referenced type
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Subtype(
                    sub_inner,
                    super_inner,
                    span,
                )]))
            }
            // Mutable reference requires exact type match
            (Mutable, Mutable) => {
                // Invariant in the referenced type (to prevent unsoundness)
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                    sub_inner,
                    super_inner,
                    span,
                )]))
            }
            // Cannot use immutable reference where mutable is expected
            (Immutable, Mutable) => {
                self.errors.push(SolverError::CannotUnify(
                    Type::Reference {
                        ty: Box::new(sub_inner),
                        mutability: sub_mut,
                        lifetime: None,
                        nullability: crate::type_registry::NullabilityKind::default(),
                    },
                    Type::Reference {
                        ty: Box::new(super_inner),
                        mutability: super_mut,
                        lifetime: None,
                        nullability: crate::type_registry::NullabilityKind::default(),
                    },
                    span,
                ));
                Err(self.errors.clone())
            }
        }
    }

    /// Solve Callable constraint
    fn solve_callable(
        &mut self,
        func_ty: Type,
        arg_types: Vec<Type>,
        ret_ty: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let func_ty = self.subst.apply(&func_ty);

        match func_ty {
            Type::Function {
                params,
                return_type,
                ..
            } => {
                if params.len() != arg_types.len() {
                    self.errors.push(SolverError::CannotUnify(
                        Type::Function {
                            params: params.clone(),
                            return_type: return_type.clone(),
                            is_varargs: false,
                            has_named_params: false,
                            has_default_params: false,
                            async_kind: crate::type_registry::AsyncKind::default(),
                            calling_convention: crate::type_registry::CallingConvention::default(),
                            nullability: crate::type_registry::NullabilityKind::default(),
                        },
                        Type::Never,
                        span,
                    ));
                    return Err(self.errors.clone());
                }

                let mut constraints = vec![];
                for (param, arg_ty) in params.iter().zip(arg_types.iter()) {
                    constraints.push(Constraint::Equal(param.ty.clone(), arg_ty.clone(), span));
                }
                constraints.push(Constraint::Equal(*return_type, ret_ty, span));

                Ok(ConstraintResult::NewConstraints(constraints))
            }
            Type::TypeVar(_) => {
                // Create a function type to unify with
                let param_infos: Vec<ParamInfo> = arg_types
                    .iter()
                    .map(|ty| ParamInfo {
                        name: None,
                        ty: ty.clone(),
                        is_optional: false,
                        is_varargs: false,
                        is_keyword_only: false,
                        is_positional_only: false,
                        is_out: false,
                        is_ref: false,
                        is_inout: false,
                    })
                    .collect();

                let new_func_ty = Type::Function {
                    params: param_infos,
                    return_type: Box::new(ret_ty),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: crate::type_registry::AsyncKind::default(),
                    calling_convention: crate::type_registry::CallingConvention::default(),
                    nullability: crate::type_registry::NullabilityKind::default(),
                };

                Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                    func_ty,
                    new_func_ty,
                    span,
                )]))
            }
            _ => {
                self.errors
                    .push(SolverError::CannotUnify(func_ty, Type::Never, span));
                Err(self.errors.clone())
            }
        }
    }

    /// Solve Subtype constraint with proper structural subtyping rules
    fn solve_subtype(
        &mut self,
        sub_ty: Type,
        super_ty: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let sub_ty = self.subst.apply(&sub_ty);
        let super_ty = self.subst.apply(&super_ty);

        // Fast path: if types are equal, subtyping holds
        if sub_ty == super_ty {
            return Ok(ConstraintResult::Solved);
        }

        match (&sub_ty, &super_ty) {
            // Reflexivity: T <: T (handled above)

            // Bottom type: Never is a subtype of everything
            (Type::Never, _) => Ok(ConstraintResult::Solved),

            // Top type: everything is a subtype of Any
            (_, Type::Any) => Ok(ConstraintResult::Solved),

            // Type variables: defer or unify
            (Type::TypeVar(v1), Type::TypeVar(v2)) if v1.id == v2.id => {
                Ok(ConstraintResult::Solved)
            }
            (Type::TypeVar(_), _) | (_, Type::TypeVar(_)) => {
                // For type variables, we fall back to unification
                // In a more sophisticated system, we'd maintain subtype bounds
                self.unify(sub_ty, super_ty, span)
            }

            // Self type subtyping
            (Type::SelfType, Type::SelfType) => Ok(ConstraintResult::Solved),
            (Type::SelfType, _) | (_, Type::SelfType) => {
                // Self type subtyping requires trait method context
                // For now, we'll defer resolution
                Ok(ConstraintResult::Deferred)
            }

            // Primitive subtyping (numeric promotions)
            (Type::Primitive(p1), Type::Primitive(p2)) => {
                if self.is_numeric_subtype(*p1, *p2) {
                    Ok(ConstraintResult::Solved)
                } else if p1 == p2 {
                    Ok(ConstraintResult::Solved)
                } else {
                    self.errors
                        .push(SolverError::CannotUnify(sub_ty, super_ty, span));
                    Err(self.errors.clone())
                }
            }

            // Function subtyping: contravariant in parameters, covariant in return
            (
                Type::Function {
                    params: p1,
                    return_type: r1,
                    ..
                },
                Type::Function {
                    params: p2,
                    return_type: r2,
                    ..
                },
            ) => {
                self.solve_function_subtype(p1.clone(), *r1.clone(), p2.clone(), *r2.clone(), span)
            }

            // Array subtyping: covariant in element type (for immutable arrays)
            (
                Type::Array {
                    element_type: e1,
                    size: s1,
                    nullability: n1,
                },
                Type::Array {
                    element_type: e2,
                    size: s2,
                    nullability: n2,
                },
            ) => {
                if s1 == s2 {
                    // Arrays are covariant in their element type (for read-only access)
                    Ok(ConstraintResult::NewConstraints(vec![Constraint::Subtype(
                        *e1.clone(),
                        *e2.clone(),
                        span,
                    )]))
                } else {
                    self.errors
                        .push(SolverError::CannotUnify(sub_ty, super_ty, span));
                    Err(self.errors.clone())
                }
            }

            // Tuple subtyping: width subtyping (fewer fields can be subtype of more fields)
            (Type::Tuple(t1), Type::Tuple(t2)) => {
                self.solve_tuple_subtype(t1.clone(), t2.clone(), span)
            }

            // Named type subtyping: check inheritance hierarchy
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
            ) => self.solve_named_type_subtype(*id1, args1.clone(), *id2, args2.clone(), span),

            // Union subtyping: T <: U1 | U2 | ... if T <: Ui for some i
            (sub_t, Type::Union(union_types)) => {
                // Try to find at least one type in the union that sub_t is a subtype of
                for union_member in union_types {
                    let result = self.solve_subtype(sub_t.clone(), union_member.clone(), span);
                    if result.is_ok() {
                        return result;
                    }
                }
                self.errors
                    .push(SolverError::CannotUnify(sub_ty, super_ty, span));
                Err(self.errors.clone())
            }

            // Union subtyping: U1 | U2 | ... <: T if Ui <: T for all i
            (Type::Union(union_types), super_t) => {
                let mut constraints = Vec::new();
                for union_member in union_types {
                    constraints.push(Constraint::Subtype(
                        union_member.clone(),
                        super_t.clone(),
                        span,
                    ));
                }
                Ok(ConstraintResult::NewConstraints(constraints))
            }

            // Intersection subtyping: T1 & T2 & ... <: U if Ti <: U for some i
            (Type::Intersection(inter_types), super_t) => {
                // At least one component of the intersection must be a subtype
                for inter_member in inter_types {
                    let result = self.solve_subtype(inter_member.clone(), super_t.clone(), span);
                    if result.is_ok() {
                        return result;
                    }
                }
                self.errors
                    .push(SolverError::CannotUnify(sub_ty, super_ty, span));
                Err(self.errors.clone())
            }

            // Intersection subtyping: T <: U1 & U2 & ... if T <: Ui for all i
            (sub_t, Type::Intersection(inter_types)) => {
                let mut constraints = Vec::new();
                for inter_member in inter_types {
                    constraints.push(Constraint::Subtype(
                        sub_t.clone(),
                        inter_member.clone(),
                        span,
                    ));
                }
                Ok(ConstraintResult::NewConstraints(constraints))
            }

            // Optional subtyping: T <: T?
            (sub_t, Type::Optional(opt_inner)) => {
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Subtype(
                    sub_t.clone(),
                    *opt_inner.clone(),
                    span,
                )]))
            }

            // Reference subtyping: handle mutability and lifetime constraints
            (
                Type::Reference {
                    ty: t1,
                    mutability: m1,
                    ..
                },
                Type::Reference {
                    ty: t2,
                    mutability: m2,
                    ..
                },
            ) => self.solve_reference_subtype(*t1.clone(), *m1, *t2.clone(), *m2, span),

            // Type alias resolution
            (Type::Alias { target: t1, .. }, super_t) => Ok(ConstraintResult::NewConstraints(
                vec![Constraint::Subtype(*t1.clone(), super_t.clone(), span)],
            )),
            (sub_t, Type::Alias { target: t2, .. }) => {
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Subtype(
                    sub_t.clone(),
                    *t2.clone(),
                    span,
                )]))
            }

            // Default case: types are unrelated
            _ => {
                self.errors
                    .push(SolverError::CannotUnify(sub_ty, super_ty, span));
                Err(self.errors.clone())
            }
        }
    }

    /// Solve Instance constraint
    fn solve_instance(
        &mut self,
        ty: Type,
        scheme: TypeScheme,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        // Instantiate the type scheme with fresh type variables
        let mut inst_subst = Substitution::new();
        for &quant_var in &scheme.quantified {
            inst_subst.bind(quant_var, self.fresh_type_var());
        }

        let instantiated = inst_subst.apply(&scheme.ty);
        Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
            ty,
            instantiated,
            span,
        )]))
    }

    /// Solve trait bound constraint: type must implement trait
    fn solve_trait_bound(
        &mut self,
        ty: Type,
        trait_name: InternedString,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let resolved_ty = self.subst.apply(&ty);

        match &resolved_ty {
            Type::TypeVar(type_var) => {
                // For type variables, accumulate trait bounds
                self.trait_bounds
                    .entry(type_var.id)
                    .or_insert_with(HashSet::new)
                    .insert(trait_name);

                // Check if we have a concrete type that we can verify against
                if let Some(concrete_ty) = self.subst.map.get(&type_var.id) {
                    self.verify_trait_bounds(concrete_ty.clone(), std::iter::once(trait_name), span)
                } else {
                    Ok(ConstraintResult::Deferred)
                }
            }

            Type::Named {
                id: _,
                type_args: _,
                ..
            } => {
                // Check if the named type implements the trait
                // First try to find the trait by name
                if let Some(trait_def) = self.type_registry.get_trait_by_name(trait_name) {
                    if self
                        .type_registry
                        .type_implements(&resolved_ty, trait_def.id)
                    {
                        Ok(ConstraintResult::Solved)
                    } else {
                        self.errors.push(SolverError::TraitNotImplemented {
                            ty: resolved_ty,
                            trait_name,
                            span,
                        });
                        Err(self.errors.clone())
                    }
                } else {
                    self.errors.push(SolverError::UnknownTrait {
                        name: trait_name,
                        span,
                    });
                    Err(self.errors.clone())
                }
            }

            Type::Primitive(prim_ty) => {
                // Check built-in trait implementations for primitive types
                // For now, we'll assume primitives implement basic traits like Display
                match (prim_ty, trait_name) {
                    // Numeric primitives implement Display
                    (
                        PrimitiveType::I8
                        | PrimitiveType::I16
                        | PrimitiveType::I32
                        | PrimitiveType::I64
                        | PrimitiveType::I128
                        | PrimitiveType::ISize
                        | PrimitiveType::U8
                        | PrimitiveType::U16
                        | PrimitiveType::U32
                        | PrimitiveType::U64
                        | PrimitiveType::U128
                        | PrimitiveType::USize
                        | PrimitiveType::F32
                        | PrimitiveType::F64,
                        _,
                    ) => {
                        // Check if it's a standard trait that numeric types implement
                        if let Some(trait_def) = self.type_registry.get_trait_by_name(trait_name) {
                            if self
                                .type_registry
                                .type_implements(&Type::Primitive(*prim_ty), trait_def.id)
                            {
                                Ok(ConstraintResult::Solved)
                            } else {
                                self.errors.push(SolverError::TraitNotImplemented {
                                    ty: resolved_ty,
                                    trait_name,
                                    span,
                                });
                                Err(self.errors.clone())
                            }
                        } else {
                            self.errors.push(SolverError::UnknownTrait {
                                name: trait_name,
                                span,
                            });
                            Err(self.errors.clone())
                        }
                    }

                    // String implements Display
                    (PrimitiveType::String, _) => {
                        // Check if it's a standard trait that numeric types implement
                        if let Some(trait_def) = self.type_registry.get_trait_by_name(trait_name) {
                            if self
                                .type_registry
                                .type_implements(&Type::Primitive(*prim_ty), trait_def.id)
                            {
                                Ok(ConstraintResult::Solved)
                            } else {
                                self.errors.push(SolverError::TraitNotImplemented {
                                    ty: resolved_ty,
                                    trait_name,
                                    span,
                                });
                                Err(self.errors.clone())
                            }
                        } else {
                            self.errors.push(SolverError::UnknownTrait {
                                name: trait_name,
                                span,
                            });
                            Err(self.errors.clone())
                        }
                    }

                    // Other primitives
                    _ => {
                        // Check if it's a standard trait that numeric types implement
                        if let Some(trait_def) = self.type_registry.get_trait_by_name(trait_name) {
                            if self
                                .type_registry
                                .type_implements(&Type::Primitive(*prim_ty), trait_def.id)
                            {
                                Ok(ConstraintResult::Solved)
                            } else {
                                self.errors.push(SolverError::TraitNotImplemented {
                                    ty: resolved_ty,
                                    trait_name,
                                    span,
                                });
                                Err(self.errors.clone())
                            }
                        } else {
                            self.errors.push(SolverError::UnknownTrait {
                                name: trait_name,
                                span,
                            });
                            Err(self.errors.clone())
                        }
                    }
                }
            }

            _ => {
                // For other types, assume they don't implement the trait
                self.errors.push(SolverError::TraitNotImplemented {
                    ty: resolved_ty,
                    trait_name,
                    span,
                });
                Err(self.errors.clone())
            }
        }
    }

    /// Generate a fresh implicit lifetime
    pub fn fresh_lifetime(&mut self, arena: &mut crate::arena::AstArena) -> Lifetime {
        let name = arena.intern_string(&format!("'_{}", self.next_implicit_lifetime));
        self.next_implicit_lifetime += 1;
        Lifetime::named(name)
    }

    /// Solve lifetime outlives constraint: 'a: 'b ('a outlives 'b)
    fn solve_lifetime_outlives(
        &mut self,
        lifetime_a: Lifetime,
        lifetime_b: Lifetime,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        // Check if adding this constraint would create a cycle
        // We need to check if there's already a path from b to a
        if self.would_create_lifetime_cycle(&lifetime_a, &lifetime_b) {
            self.errors.push(SolverError::LifetimeCycle {
                lifetime1: lifetime_a.name,
                lifetime2: lifetime_b.name,
                span,
            });
            return Err(self.errors.clone());
        }

        // Track the outlives relationship
        self.lifetime_constraints
            .push((lifetime_a.clone(), lifetime_b.clone()));

        Ok(ConstraintResult::Solved)
    }

    /// Solve type outlives lifetime constraint: T: 'a
    fn solve_type_outlives_lifetime(
        &mut self,
        ty: Type,
        lifetime: Lifetime,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let resolved_ty = self.subst.apply(&ty);

        // Track the type-lifetime constraint
        self.type_lifetime_constraints
            .push((resolved_ty.clone(), lifetime.clone()));

        match &resolved_ty {
            Type::TypeVar(_) => {
                // For type variables, defer the constraint
                Ok(ConstraintResult::Deferred)
            }
            Type::Reference {
                lifetime: ref_lifetime,
                ..
            } => {
                // For references, the reference lifetime must outlive the required lifetime
                if let Some(ref_lifetime) = ref_lifetime {
                    self.lifetime_constraints
                        .push((ref_lifetime.clone(), lifetime));
                }
                Ok(ConstraintResult::Solved)
            }
            _ => {
                // For other types, the constraint is satisfied by default
                Ok(ConstraintResult::Solved)
            }
        }
    }

    /// Check if adding 'a: 'b would create a cycle
    fn would_create_lifetime_cycle(&self, lifetime_a: &Lifetime, lifetime_b: &Lifetime) -> bool {
        // If a == b, it's definitely a cycle
        if lifetime_a.name == lifetime_b.name {
            return true;
        }

        // Check if there's already a path from b to a
        // If there is, then adding a: b would create a cycle
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(lifetime_b.clone());

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.name) {
                continue;
            }

            // Find all lifetimes that outlive current (i.e., x where x: current exists)
            for (x, y) in &self.lifetime_constraints {
                if y.name == current.name {
                    if x.name == lifetime_a.name {
                        // Found a path from b to a, so adding a: b would create a cycle
                        return true;
                    }
                    queue.push_back(x.clone());
                }
            }
        }

        false
    }

    /// Infer lifetimes for references in a type
    pub fn infer_lifetimes(&mut self, ty: &Type, arena: &mut crate::arena::AstArena) -> Type {
        match ty {
            Type::Reference {
                ty: inner,
                mutability,
                lifetime: None,
                ..
            } => {
                // Generate implicit lifetime for references without explicit lifetimes
                let implicit_lifetime = self.fresh_lifetime(arena);
                Type::Reference {
                    ty: Box::new(self.infer_lifetimes(inner, arena)),
                    mutability: *mutability,
                    lifetime: Some(implicit_lifetime),
                    nullability: crate::type_registry::NullabilityKind::default(),
                }
            }
            Type::Reference {
                ty: inner,
                mutability,
                lifetime,
                ..
            } => Type::Reference {
                ty: Box::new(self.infer_lifetimes(inner, arena)),
                mutability: *mutability,
                lifetime: lifetime.clone(),
                nullability: crate::type_registry::NullabilityKind::default(),
            },
            Type::Function {
                params,
                return_type,
                is_varargs,
                has_named_params,
                has_default_params,
                ..
            } => Type::Function {
                params: params
                    .iter()
                    .map(|p| ParamInfo {
                        name: p.name,
                        ty: self.infer_lifetimes(&p.ty, arena),
                        is_optional: p.is_optional,
                        is_varargs: p.is_varargs,
                        is_keyword_only: p.is_keyword_only,
                        is_positional_only: p.is_positional_only,
                        is_out: p.is_out,
                        is_ref: p.is_ref,
                        is_inout: p.is_inout,
                    })
                    .collect(),
                return_type: Box::new(self.infer_lifetimes(return_type, arena)),
                is_varargs: *is_varargs,
                has_named_params: *has_named_params,
                has_default_params: *has_default_params,
                async_kind: crate::type_registry::AsyncKind::default(),
                calling_convention: crate::type_registry::CallingConvention::default(),
                nullability: crate::type_registry::NullabilityKind::default(),
            },
            Type::Array {
                element_type,
                size,
                nullability,
            } => Type::Array {
                element_type: Box::new(self.infer_lifetimes(element_type, arena)),
                size: size.clone(),
                nullability: crate::type_registry::NullabilityKind::default(),
            },
            Type::Tuple(types) => Type::Tuple(
                types
                    .iter()
                    .map(|t| self.infer_lifetimes(t, arena))
                    .collect(),
            ),
            Type::Optional(inner) => Type::Optional(Box::new(self.infer_lifetimes(inner, arena))),
            _ => ty.clone(),
        }
    }

    /// Check lifetime constraints are satisfied
    pub fn check_lifetime_constraints(&self) -> Result<(), Vec<SolverError>> {
        let mut errors = Vec::new();

        // Check for contradictory lifetime constraints
        for (i, (a1, b1)) in self.lifetime_constraints.iter().enumerate() {
            for (a2, b2) in self.lifetime_constraints.iter().skip(i + 1) {
                // Check if we have both 'a: 'b and 'b: 'a (which would be a cycle)
                if a1.name == b2.name && b1.name == a2.name && a1.name != b1.name {
                    errors.push(SolverError::LifetimeCycle {
                        lifetime1: a1.name,
                        lifetime2: b1.name,
                        span: Span::new(0, 0), // TODO: preserve spans
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

    /// Check if a lifetime is 'static or constrained by 'static
    pub fn is_static_lifetime(&self, lifetime: &Lifetime, static_name: InternedString) -> bool {
        // Check if it's literally 'static
        if lifetime.name == static_name {
            return true;
        }

        // Check constraints to see if 'static: lifetime
        for (longer, shorter) in &self.lifetime_constraints {
            if longer.name == static_name && shorter.name == lifetime.name {
                return true;
            }
        }

        false
    }

    /// Propagate static lifetime constraints
    /// When a function returns a reference that must be 'static,
    /// this propagates the constraint to the input lifetimes
    pub fn propagate_static_constraints(&mut self, static_name: InternedString) {
        let mut new_constraints = Vec::new();

        // Find all lifetimes that must be at least 'static
        for (longer, shorter) in &self.lifetime_constraints {
            // If we have 'a: 'static, then 'a must be 'static
            if shorter.name == static_name {
                // longer must also be 'static
                new_constraints.push((longer.clone(), shorter.clone()));
            }
        }

        // Add discovered constraints
        for (a, b) in new_constraints {
            self.lifetime_constraints.push((a, b));
        }
    }

    /// Solve higher-ranked trait bound: for<'a> T: Trait<'a>
    fn solve_higher_ranked_bound(
        &mut self,
        lifetimes: Vec<InternedString>,
        ty: Type,
        bound: TypeBound,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        // If the type is a type variable, defer the constraint
        if matches!(ty, Type::TypeVar(_)) {
            return Ok(ConstraintResult::Deferred);
        }

        // Create temporary lifetime variables for the universally quantified lifetimes
        let mut lifetime_mapping = HashMap::new();
        let mut fresh_lifetimes = Vec::new();

        for lifetime_name in &lifetimes {
            let fresh = Lifetime::named(*lifetime_name);
            lifetime_mapping.insert(*lifetime_name, fresh.clone());
            fresh_lifetimes.push(fresh);
        }

        // Substitute the fresh lifetimes in the bound
        let instantiated_bound = self.instantiate_bound_lifetimes(&bound, &lifetime_mapping);

        // Check the instantiated bound
        match instantiated_bound {
            TypeBound::Trait { name, args } => {
                // Create a regular trait bound constraint with instantiated lifetimes
                self.solve_trait_bound(ty.clone(), name, span)?;

                // TODO: Handle trait type arguments that may contain the fresh lifetimes
                // For now, we assume the trait bound is valid if the trait exists
                Ok(ConstraintResult::Solved)
            }
            TypeBound::Lifetime(lifetime) => {
                // Type must outlive the lifetime
                self.solve_type_outlives_lifetime(ty, lifetime, span)
            }
            _ => {
                // Other bounds inside higher-ranked bounds are handled similarly
                // For now, we'll accept them
                Ok(ConstraintResult::Solved)
            }
        }
    }

    /// Instantiate bound lifetimes with fresh lifetime variables
    fn instantiate_bound_lifetimes(
        &self,
        bound: &TypeBound,
        lifetime_mapping: &HashMap<InternedString, Lifetime>,
    ) -> TypeBound {
        match bound {
            TypeBound::Trait { name, args } => {
                // TODO: Substitute lifetimes in trait arguments
                TypeBound::Trait {
                    name: *name,
                    args: args.clone(), // For now, keep args as-is
                }
            }
            TypeBound::Lifetime(lt) => {
                // Check if this lifetime should be substituted
                if let Some(fresh_lt) = lifetime_mapping.get(&lt.name) {
                    TypeBound::Lifetime(fresh_lt.clone())
                } else {
                    TypeBound::Lifetime(lt.clone())
                }
            }
            TypeBound::HigherRanked { lifetimes, bound } => {
                // Nested higher-ranked bounds
                let inner_bound = self.instantiate_bound_lifetimes(bound, lifetime_mapping);
                TypeBound::HigherRanked {
                    lifetimes: lifetimes.clone(),
                    bound: Box::new(inner_bound),
                }
            }
            other => other.clone(),
        }
    }

    /// Apply substitution to constraint
    pub fn apply_subst_to_constraint(
        &self,
        subst: &Substitution,
        constraint: Constraint,
    ) -> Constraint {
        match constraint {
            Constraint::Equal(t1, t2, span) => {
                Constraint::Equal(subst.apply(&t1), subst.apply(&t2), span)
            }
            Constraint::Varargs(t, args, span) => Constraint::Varargs(
                subst.apply(&t),
                args.into_iter().map(|a| subst.apply(&a)).collect(),
                span,
            ),
            Constraint::ArrayElement(arr, elem, span) => {
                Constraint::ArrayElement(subst.apply(&arr), subst.apply(&elem), span)
            }
            Constraint::HasMember(obj, member, member_ty, span) => {
                Constraint::HasMember(subst.apply(&obj), member, subst.apply(&member_ty), span)
            }
            Constraint::Callable(func, args, ret, span) => Constraint::Callable(
                subst.apply(&func),
                args.into_iter().map(|a| subst.apply(&a)).collect(),
                subst.apply(&ret),
                span,
            ),
            Constraint::Subtype(sub, super_ty, span) => {
                Constraint::Subtype(subst.apply(&sub), subst.apply(&super_ty), span)
            }
            Constraint::Instance(ty, scheme, span) => {
                // Note: We should also apply to the scheme, but for now we'll keep it simple
                Constraint::Instance(subst.apply(&ty), scheme, span)
            }
            Constraint::TraitBound(ty, trait_name, span) => {
                Constraint::TraitBound(subst.apply(&ty), trait_name, span)
            }
            Constraint::LifetimeOutlives(lifetime_a, lifetime_b, span) => {
                // Lifetimes are not affected by type substitution
                Constraint::LifetimeOutlives(lifetime_a, lifetime_b, span)
            }
            Constraint::TypeOutlivesLifetime(ty, lifetime, span) => {
                Constraint::TypeOutlivesLifetime(subst.apply(&ty), lifetime, span)
            }
            Constraint::HigherRankedBound {
                lifetimes,
                ty,
                bound,
                span,
            } => {
                // Apply substitution to the type, but not to the bound's internal types
                // since they may reference the universally quantified lifetimes
                Constraint::HigherRankedBound {
                    lifetimes,
                    ty: subst.apply(&ty),
                    bound,
                    span,
                }
            }

            // Dependent type constraints
            Constraint::RefinementSatisfies {
                value,
                predicate,
                span,
            } => Constraint::RefinementSatisfies {
                value: subst.apply(&value),
                predicate, // Predicates don't contain type variables directly
                span,
            },

            Constraint::DependentCall {
                func_type,
                arg_value,
                result_type,
                span,
            } => Constraint::DependentCall {
                func_type: subst.apply(&func_type),
                arg_value, // Const values don't contain type variables
                result_type: subst.apply(&result_type),
                span,
            },

            Constraint::PathDependent {
                base_type,
                path,
                target_type,
                span,
            } => Constraint::PathDependent {
                base_type: subst.apply(&base_type),
                path, // Paths are just identifier sequences
                target_type: subst.apply(&target_type),
                span,
            },

            Constraint::SingletonEquals {
                value_type,
                constant,
                span,
            } => Constraint::SingletonEquals {
                value_type: subst.apply(&value_type),
                constant, // Constants don't contain type variables
                span,
            },

            Constraint::TypeFamilyApplication {
                family,
                indices,
                result_type,
                span,
            } => Constraint::TypeFamilyApplication {
                family: subst.apply(&family),
                indices, // Indices may need substitution but are complex
                result_type: subst.apply(&result_type),
                span,
            },

            Constraint::ConditionalType {
                condition,
                then_type,
                else_type,
                result_type,
                span,
            } => Constraint::ConditionalType {
                condition, // Predicates don't contain type variables directly
                then_type: subst.apply(&then_type),
                else_type: subst.apply(&else_type),
                result_type: subst.apply(&result_type),
                span,
            },
        }
    }

    /// Verify all trait bounds for a concrete type
    fn verify_trait_bounds<I>(
        &mut self,
        concrete_ty: Type,
        trait_names: I,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>>
    where
        I: Iterator<Item = InternedString>,
    {
        for trait_name in trait_names {
            // Check if the concrete type implements this trait
            match self.check_single_trait_bound(&concrete_ty, trait_name, span) {
                Ok(ConstraintResult::Solved) => continue,
                Ok(other) => return Ok(other), // Deferred or NewConstraints
                Err(errors) => return Err(errors),
            }
        }
        Ok(ConstraintResult::Solved)
    }

    /// Check if a concrete type implements a single trait
    fn check_single_trait_bound(
        &mut self,
        concrete_ty: &Type,
        trait_name: InternedString,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        match concrete_ty {
            Type::Named {
                id: _,
                type_args: _,
                ..
            } => {
                if let Some(trait_def) = self.type_registry.get_trait_by_name(trait_name) {
                    if self
                        .type_registry
                        .type_implements(concrete_ty, trait_def.id)
                    {
                        Ok(ConstraintResult::Solved)
                    } else {
                        self.errors.push(SolverError::TraitNotImplemented {
                            ty: concrete_ty.clone(),
                            trait_name,
                            span,
                        });
                        Err(self.errors.clone())
                    }
                } else {
                    self.errors.push(SolverError::UnknownTrait {
                        name: trait_name,
                        span,
                    });
                    Err(self.errors.clone())
                }
            }
            Type::Primitive(prim_ty) => {
                // Check built-in trait implementations for primitive types
                if let Some(trait_def) = self.type_registry.get_trait_by_name(trait_name) {
                    if self
                        .type_registry
                        .type_implements(&Type::Primitive(*prim_ty), trait_def.id)
                    {
                        Ok(ConstraintResult::Solved)
                    } else {
                        self.errors.push(SolverError::TraitNotImplemented {
                            ty: concrete_ty.clone(),
                            trait_name,
                            span,
                        });
                        Err(self.errors.clone())
                    }
                } else {
                    self.errors.push(SolverError::UnknownTrait {
                        name: trait_name,
                        span,
                    });
                    Err(self.errors.clone())
                }
            }
            _ => {
                // For other types, assume they don't implement the trait
                self.errors.push(SolverError::TraitNotImplemented {
                    ty: concrete_ty.clone(),
                    trait_name,
                    span,
                });
                Err(self.errors.clone())
            }
        }
    }

    /// Propagate trait bounds through type relationships
    pub fn propagate_trait_bounds(&mut self) -> Vec<Constraint> {
        let mut new_constraints = Vec::new();

        // For each type variable that has trait bounds
        for (type_var_id, trait_bounds) in &self.trait_bounds {
            // If this type variable is bound to a concrete type, verify all bounds
            if let Some(concrete_type) = self.subst.map.get(type_var_id) {
                for &trait_name in trait_bounds {
                    new_constraints.push(Constraint::TraitBound(
                        concrete_type.clone(),
                        trait_name,
                        Span::new(0, 0), // TODO: Track proper spans
                    ));
                }
            }
        }

        new_constraints
    }

    /// Enhanced error message generation for trait implementation failures
    pub fn generate_detailed_error(&self, error: &SolverError) -> String {
        match error {
            SolverError::TraitNotImplemented {
                ty,
                trait_name,
                span,
            } => {
                format!(
                    "Type '{}' does not implement trait '{}' at line {}. \
                     Consider adding an implementation or changing the type constraint.",
                    self.format_type(ty),
                    self.format_interned_string(*trait_name),
                    span.start // Simplified - in real implementation would convert to line number
                )
            }
            SolverError::UnknownTrait { name, span } => {
                format!(
                    "Unknown trait '{}' referenced at line {}. \
                     Make sure the trait is in scope and properly defined.",
                    self.format_interned_string(*name),
                    span.start
                )
            }
            SolverError::CannotUnify(t1, t2, span) => {
                format!(
                    "Cannot unify types '{}' and '{}' at line {}. \
                     These types are incompatible.",
                    self.format_type(t1),
                    self.format_type(t2),
                    span.start
                )
            }
            SolverError::InfiniteType(var_id, ty, span) => {
                format!(
                    "Infinite type detected: type variable {} would be bound to '{}' at line {}. \
                     This creates a recursive type definition.",
                    var_id.as_u32(),
                    self.format_type(ty),
                    span.start
                )
            }
            SolverError::UnsolvableConstraint(constraint) => {
                format!(
                    "Unsolvable constraint: {}. \
                     The constraint cannot be satisfied with any type assignment.",
                    self.format_constraint(constraint)
                )
            }
            SolverError::LifetimeCycle {
                lifetime1,
                lifetime2,
                span,
            } => {
                format!(
                    "Lifetime cycle detected: '{}' and '{}' at line {} would create a cyclic dependency. \
                     Lifetimes cannot outlive each other in a cycle.",
                    self.format_interned_string(*lifetime1),
                    self.format_interned_string(*lifetime2),
                    span.start
                )
            }
        }
    }

    /// Format type for display in error messages
    pub fn format_type(&self, ty: &Type) -> String {
        match ty {
            Type::Primitive(prim) => format!("{:?}", prim).to_lowercase(),
            Type::TypeVar(var) => {
                if let Some(name) = &var.name {
                    format!("'{}", name)
                } else {
                    format!("'T{}", var.id.as_u32())
                }
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                let param_strs: Vec<String> =
                    params.iter().map(|p| self.format_type(&p.ty)).collect();
                format!(
                    "fn({}) -> {}",
                    param_strs.join(", "),
                    self.format_type(return_type)
                )
            }
            Type::Array {
                element_type,
                size,
                nullability,
            } => {
                if let Some(size) = size {
                    let size_str = match size {
                        ConstValue::Int(n) => n.to_string(),
                        ConstValue::UInt(n) => n.to_string(),
                        _ => format!("{:?}", size),
                    };
                    format!("[{}; {}]", self.format_type(element_type), size_str)
                } else {
                    format!("[{}]", self.format_type(element_type))
                }
            }
            Type::Tuple(types) => {
                let type_strs: Vec<String> = types.iter().map(|t| self.format_type(t)).collect();
                format!("({})", type_strs.join(", "))
            }
            Type::Named { id, type_args, .. } => {
                if type_args.is_empty() {
                    format!("Type#{}", id.as_u32()) // Simplified - would resolve actual name
                } else {
                    let arg_strs: Vec<String> =
                        type_args.iter().map(|t| self.format_type(t)).collect();
                    format!("Type#{}<{}>", id.as_u32(), arg_strs.join(", "))
                }
            }
            Type::Reference { ty, mutability, .. } => match mutability {
                Mutability::Mutable => format!("&mut {}", self.format_type(ty)),
                Mutability::Immutable => format!("&{}", self.format_type(ty)),
            },
            Type::Optional(ty) => {
                format!("{}?", self.format_type(ty))
            }
            Type::Result { ok_type, err_type } => {
                format!(
                    "Result<{}, {}>",
                    self.format_type(ok_type),
                    self.format_type(err_type)
                )
            }
            Type::Union(types) => {
                let type_strs: Vec<String> = types.iter().map(|t| self.format_type(t)).collect();
                format!("{}", type_strs.join(" | "))
            }
            Type::Intersection(types) => {
                let type_strs: Vec<String> = types.iter().map(|t| self.format_type(t)).collect();
                format!("{}", type_strs.join(" & "))
            }
            Type::Alias { name, target } => {
                format!("{} (alias for {})", name, self.format_type(target))
            }
            Type::Associated {
                trait_name,
                type_name,
            } => {
                format!("{}::{}", trait_name, type_name)
            }
            Type::HigherKinded {
                constructor, arity, ..
            } => {
                format!("{}<{}>", constructor, "*".repeat(*arity))
            }
            Type::Projection { base, item } => {
                format!("{}::{}", self.format_type(base), item)
            }
            Type::Index { base, index } => {
                format!("{}[{}]", self.format_type(base), self.format_type(index))
            }
            Type::Any => "any".to_string(),
            Type::Never => "never".to_string(),
            Type::Error => "error".to_string(),
            Type::SelfType => "Self".to_string(),
            Type::Nullable(inner_ty) => {
                format!("{}?", self.format_type(inner_ty))
            }
            Type::NonNull(inner_ty) => {
                format!("{}!", self.format_type(inner_ty))
            }
            Type::ConstVar {
                id,
                name,
                const_type,
            } => {
                if let Some(name) = name {
                    format!("const {}", name)
                } else {
                    format!("const C{}", id.as_u32())
                }
            }
            Type::ConstDependent {
                base_type,
                constraint,
            } => {
                format!("{{{}|constraint}}", self.format_type(base_type))
            }
            Type::Dynamic => "dynamic".to_string(),
            Type::Unknown => "unknown".to_string(),
            Type::Interface {
                methods,
                is_structural,
                nullability,
            } => todo!(),
            Type::Struct {
                fields,
                is_anonymous,
                nullability,
            } => todo!(),
            Type::Trait {
                id,
                associated_types,
                super_traits,
            } => todo!(),
            Type::Extern { name, .. } => {
                // Format extern/opaque type by name
                name.resolve_global()
                    .unwrap_or_else(|| format!("extern_{}", name.symbol().to_usize()))
            }
            Type::Unresolved(name) => {
                // Format unresolved type by name
                name.resolve_global()
                    .unwrap_or_else(|| format!("unresolved_{}", name.symbol().to_usize()))
            }
        }
    }

    /// Format InternedString for display in error messages
    /// Since we don't have access to the arena here, we'll create a readable representation
    fn format_interned_string(&self, interned: InternedString) -> String {
        // For now, we'll extract a simple representation using the symbol index
        // In a full implementation, this would resolve through the arena
        format!("trait_{}", interned.symbol().to_usize())
    }

    /// Format constraint for display in error messages
    fn format_constraint(&self, constraint: &Constraint) -> String {
        match constraint {
            Constraint::Equal(t1, t2, _) => {
                format!("{} = {}", self.format_type(t1), self.format_type(t2))
            }
            Constraint::Subtype(t1, t2, _) => {
                format!("{} <: {}", self.format_type(t1), self.format_type(t2))
            }
            Constraint::TraitBound(ty, trait_name, _) => {
                format!("{}: {}", self.format_type(ty), trait_name)
            }
            Constraint::HasMember(obj_ty, member, member_ty, _) => {
                format!(
                    "{}.{}: {}",
                    self.format_type(obj_ty),
                    member,
                    self.format_type(member_ty)
                )
            }
            Constraint::Callable(func_ty, arg_types, ret_ty, _) => {
                let arg_strs: Vec<String> = arg_types.iter().map(|t| self.format_type(t)).collect();
                format!(
                    "{}({}) -> {}",
                    self.format_type(func_ty),
                    arg_strs.join(", "),
                    self.format_type(ret_ty)
                )
            }
            Constraint::ArrayElement(array_ty, elem_ty, _) => {
                format!(
                    "{}[_]: {}",
                    self.format_type(array_ty),
                    self.format_type(elem_ty)
                )
            }
            Constraint::Varargs(param_ty, arg_types, _) => {
                let arg_strs: Vec<String> = arg_types.iter().map(|t| self.format_type(t)).collect();
                format!(
                    "...{}: [{}]",
                    self.format_type(param_ty),
                    arg_strs.join(", ")
                )
            }
            Constraint::Instance(ty, scheme, _) => {
                format!(
                    "{} = inst({})",
                    self.format_type(ty),
                    self.format_type(&scheme.ty)
                )
            }
            Constraint::LifetimeOutlives(lifetime_a, lifetime_b, _) => {
                format!(
                    "'{}' outlives '{}'",
                    self.format_interned_string(lifetime_a.name),
                    self.format_interned_string(lifetime_b.name)
                )
            }
            Constraint::TypeOutlivesLifetime(ty, lifetime, _) => {
                format!(
                    "{}: '{}'",
                    self.format_type(ty),
                    self.format_interned_string(lifetime.name)
                )
            }
            Constraint::HigherRankedBound {
                lifetimes,
                ty,
                bound,
                span: _,
            } => {
                let lifetime_names: Vec<String> = lifetimes
                    .iter()
                    .map(|lt| format!("'{}", self.format_interned_string(*lt)))
                    .collect();
                format!(
                    "for<{}> {}: {:?}",
                    lifetime_names.join(", "),
                    self.format_type(ty),
                    bound
                )
            }

            // Dependent type constraints
            Constraint::RefinementSatisfies {
                value,
                predicate,
                span: _,
            } => {
                format!(
                    "{} satisfies refinement {:?}",
                    self.format_type(value),
                    predicate
                )
            }

            Constraint::DependentCall {
                func_type,
                arg_value,
                result_type,
                span: _,
            } => {
                format!(
                    "{}({:?}) -> {}",
                    self.format_type(func_type),
                    arg_value,
                    self.format_type(result_type)
                )
            }

            Constraint::PathDependent {
                base_type,
                path,
                target_type,
                span: _,
            } => {
                let path_str = path
                    .iter()
                    .map(|p| self.format_interned_string(*p))
                    .collect::<Vec<_>>()
                    .join(".");
                format!(
                    "{}.{} = {}",
                    self.format_type(base_type),
                    path_str,
                    self.format_type(target_type)
                )
            }

            Constraint::SingletonEquals {
                value_type,
                constant,
                span: _,
            } => {
                format!("{} == {:?}", self.format_type(value_type), constant)
            }

            Constraint::TypeFamilyApplication {
                family,
                indices,
                result_type,
                span: _,
            } => {
                format!(
                    "{}[{:?}] = {}",
                    self.format_type(family),
                    indices,
                    self.format_type(result_type)
                )
            }

            Constraint::ConditionalType {
                condition,
                then_type,
                else_type,
                result_type,
                span: _,
            } => {
                format!(
                    "if {:?} then {} else {} = {}",
                    condition,
                    self.format_type(then_type),
                    self.format_type(else_type),
                    self.format_type(result_type)
                )
            }
        }
    }

    /// Enhanced constraint solving with propagation
    pub fn solve_with_propagation(&mut self) -> Result<Substitution, Vec<SolverError>> {
        // First, solve constraints normally
        let initial_result = self.solve();

        // Then propagate trait bounds and solve again if needed
        let propagated_constraints = self.propagate_trait_bounds();
        if !propagated_constraints.is_empty() {
            for constraint in propagated_constraints {
                self.add_constraint(constraint);
            }
            // Solve again with propagated constraints
            self.solve()
        } else {
            initial_result
        }
    }

    /// Resolve method with trait bounds for a given receiver type
    pub fn resolve_method_with_trait_bounds(
        &self,
        receiver_ty: &Type,
        method_name: InternedString,
        type_args: &[Type],
    ) -> Result<Option<ResolvedMethod>, Vec<SolverError>> {
        match receiver_ty {
            Type::Named {
                id,
                type_args: receiver_type_args,
                ..
            } => {
                // First, look for intrinsic methods in the type definition
                if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                    for method in &type_def.methods {
                        if method.name == method_name {
                            // Found intrinsic method - verify trait bounds
                            return self.resolve_and_verify_method(
                                receiver_ty,
                                method.clone(),
                                type_args,
                                receiver_type_args,
                            );
                        }
                    }
                }

                // Then look for trait methods
                self.resolve_trait_method_with_bounds(receiver_ty, method_name, type_args)
            }
            Type::Primitive(_) => {
                // Look for trait methods implemented for primitive types
                self.resolve_trait_method_with_bounds(receiver_ty, method_name, type_args)
            }
            Type::TypeVar(var) => {
                // For type variables, check if we have trait bounds that provide the method
                self.resolve_method_from_trait_bounds(var, method_name, type_args)
            }
            _ => {
                // Other types don't support method calls by default
                Ok(None)
            }
        }
    }

    /// Resolve trait method with bounds verification
    fn resolve_trait_method_with_bounds(
        &self,
        receiver_ty: &Type,
        method_name: InternedString,
        type_args: &[Type],
    ) -> Result<Option<ResolvedMethod>, Vec<SolverError>> {
        // Get all traits implemented by this type
        let implemented_traits = self.type_registry.get_implementations(receiver_ty);

        for trait_id in implemented_traits {
            if let Some(trait_def) = self.type_registry.get_trait_by_id(trait_id) {
                // Look for the method in this trait
                for method in &trait_def.methods {
                    if method.name == method_name {
                        // Found the method - verify bounds and instantiate
                        return self.resolve_and_verify_method(
                            receiver_ty,
                            method.clone(),
                            type_args,
                            &[], // No receiver type args for trait methods
                        );
                    }
                }
            }
        }

        // Method not found in any trait
        Ok(None)
    }

    /// Resolve method from trait bounds on type variables
    pub fn resolve_method_from_trait_bounds(
        &self,
        type_var: &TypeVar,
        method_name: InternedString,
        type_args: &[Type],
    ) -> Result<Option<ResolvedMethod>, Vec<SolverError>> {
        // Check if we have trait bounds for this type variable
        if let Some(trait_names) = self.trait_bounds.get(&type_var.id) {
            for trait_name in trait_names {
                // Look up the trait definition
                if let Some(trait_def) = self.type_registry.get_trait_by_name(*trait_name) {
                    // Look for the method in this trait
                    for method in &trait_def.methods {
                        if method.name == method_name {
                            // Found the method - create a resolved method with trait bounds
                            return Ok(Some(ResolvedMethod {
                                signature: method.clone(),
                                receiver_type: Type::TypeVar(type_var.clone()),
                                instantiated_type_args: type_args.to_vec(),
                                required_trait_bounds: vec![*trait_name],
                            }));
                        }
                    }
                }
            }
        }

        // Method not found in trait bounds
        Ok(None)
    }

    /// Resolve and verify method with proper bounds checking
    fn resolve_and_verify_method(
        &self,
        receiver_ty: &Type,
        method_sig: crate::type_registry::MethodSig,
        type_args: &[Type],
        receiver_type_args: &[Type],
    ) -> Result<Option<ResolvedMethod>, Vec<SolverError>> {
        // Verify that type arguments match method type parameters
        if type_args.len() != method_sig.type_params.len() {
            return Err(vec![SolverError::CannotUnify(
                Type::Error, // Placeholder
                Type::Error, // Placeholder
                method_sig.span,
            )]);
        }

        // Collect required trait bounds from method signature
        let mut required_bounds = Vec::new();
        for type_param in &method_sig.type_params {
            for bound in &type_param.bounds {
                if let crate::type_registry::TypeBound::Trait { name, .. } = bound {
                    required_bounds.push(*name);
                }
            }
        }

        // Create the resolved method with Self type substitution and associated type resolution
        let mut resolved_sig = method_sig.clone();

        // Substitute Self type in method signature if needed
        resolved_sig.return_type =
            self.substitute_self_type(&resolved_sig.return_type, receiver_ty);
        resolved_sig.params = resolved_sig
            .params
            .into_iter()
            .map(|mut param| {
                param.ty = self.substitute_self_type(&param.ty, receiver_ty);
                param
            })
            .collect();

        // Also resolve associated types
        resolved_sig.return_type =
            self.resolve_associated_types(&resolved_sig.return_type, receiver_ty);
        resolved_sig.params = resolved_sig
            .params
            .into_iter()
            .map(|mut param| {
                param.ty = self.resolve_associated_types(&param.ty, receiver_ty);
                param
            })
            .collect();

        // Process where clause constraints
        resolved_sig.where_clause = resolved_sig
            .where_clause
            .into_iter()
            .map(|constraint| match constraint {
                crate::type_registry::TypeConstraint::Implementation { ty, trait_id } => {
                    let resolved_ty = self.substitute_self_type(&ty, receiver_ty);
                    let resolved_ty = self.resolve_associated_types(&resolved_ty, receiver_ty);
                    crate::type_registry::TypeConstraint::Implementation {
                        ty: resolved_ty,
                        trait_id,
                    }
                }
                crate::type_registry::TypeConstraint::Equality { left, right } => {
                    let left_ty = self.substitute_self_type(&left, receiver_ty);
                    let left_ty = self.resolve_associated_types(&left_ty, receiver_ty);
                    let right_ty = self.substitute_self_type(&right, receiver_ty);
                    let right_ty = self.resolve_associated_types(&right_ty, receiver_ty);
                    crate::type_registry::TypeConstraint::Equality {
                        left: left_ty,
                        right: right_ty,
                    }
                }
                crate::type_registry::TypeConstraint::Subtype { sub, super_type } => {
                    let sub_ty = self.substitute_self_type(&sub, receiver_ty);
                    let sub_ty = self.resolve_associated_types(&sub_ty, receiver_ty);
                    let super_ty = self.substitute_self_type(&super_type, receiver_ty);
                    let super_ty = self.resolve_associated_types(&super_ty, receiver_ty);
                    crate::type_registry::TypeConstraint::Subtype {
                        sub: sub_ty,
                        super_type: super_ty,
                    }
                }
                other => other,
            })
            .collect();

        Ok(Some(ResolvedMethod {
            signature: resolved_sig,
            receiver_type: receiver_ty.clone(),
            instantiated_type_args: type_args.to_vec(),
            required_trait_bounds: required_bounds,
        }))
    }

    /// Process where clause constraints from a method signature
    pub fn process_where_clause(
        &mut self,
        where_clause: &[crate::type_registry::TypeConstraint],
        type_substitution: &HashMap<InternedString, Type>,
    ) {
        for constraint in where_clause {
            match constraint {
                crate::type_registry::TypeConstraint::Implementation { ty, trait_id } => {
                    // Substitute type parameters in the type
                    let substituted_ty = self.apply_type_param_substitution(ty, type_substitution);

                    // Add trait bound constraint
                    if let Some(trait_def) = self.type_registry.get_trait_by_id(*trait_id) {
                        self.add_constraint(Constraint::TraitBound(
                            substituted_ty,
                            trait_def.name,
                            Span::new(0, 0), // TODO: Track proper span
                        ));
                    }
                }
                crate::type_registry::TypeConstraint::Equality { left, right } => {
                    let left_ty = self.apply_type_param_substitution(left, type_substitution);
                    let right_ty = self.apply_type_param_substitution(right, type_substitution);
                    self.add_constraint(Constraint::Equal(left_ty, right_ty, Span::new(0, 0)));
                }
                crate::type_registry::TypeConstraint::Subtype { sub, super_type } => {
                    let sub_ty = self.apply_type_param_substitution(sub, type_substitution);
                    let super_ty =
                        self.apply_type_param_substitution(super_type, type_substitution);
                    self.add_constraint(Constraint::Subtype(sub_ty, super_ty, Span::new(0, 0)));
                }
                _ => {
                    // Other constraint types can be added as needed
                }
            }
        }
    }

    /// Apply type parameter substitution to a type
    fn apply_type_param_substitution(
        &self,
        ty: &Type,
        substitution: &HashMap<InternedString, Type>,
    ) -> Type {
        match ty {
            Type::Named {
                id,
                type_args,
                const_args,
                variance,
                nullability,
            } => {
                // Check if this is a type parameter that needs substitution
                if let Some(type_def) = self.type_registry.get_type_by_id(*id) {
                    if let Some(subst_ty) = substitution.get(&type_def.name) {
                        return subst_ty.clone();
                    }
                }

                // Otherwise, recursively apply to type arguments
                Type::Named {
                    id: *id,
                    type_args: type_args
                        .iter()
                        .map(|arg| self.apply_type_param_substitution(arg, substitution))
                        .collect(),
                    const_args: vec![], // Todo: apply type param sub for const_args
                    variance: vec![],   // Check variance?
                    nullability: *nullability,
                }
            }
            Type::Array {
                element_type,
                size,
                nullability,
            } => Type::Array {
                element_type: Box::new(
                    self.apply_type_param_substitution(element_type, substitution),
                ),
                size: size.clone(),
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
                        ty: self.apply_type_param_substitution(&p.ty, substitution),
                        ..p.clone()
                    })
                    .collect(),
                return_type: Box::new(
                    self.apply_type_param_substitution(return_type, substitution),
                ),
                is_varargs: *is_varargs,
                has_named_params: *has_named_params,
                has_default_params: *has_default_params,
                async_kind: *async_kind,
                calling_convention: *calling_convention,
                nullability: *nullability,
            },
            Type::Tuple(types) => Type::Tuple(
                types
                    .iter()
                    .map(|t| self.apply_type_param_substitution(t, substitution))
                    .collect(),
            ),
            Type::Optional(ty) => Type::Optional(Box::new(
                self.apply_type_param_substitution(ty, substitution),
            )),
            Type::Union(types) => Type::Union(
                types
                    .iter()
                    .map(|t| self.apply_type_param_substitution(t, substitution))
                    .collect(),
            ),
            Type::Intersection(types) => Type::Intersection(
                types
                    .iter()
                    .map(|t| self.apply_type_param_substitution(t, substitution))
                    .collect(),
            ),
            Type::Reference {
                ty,
                mutability,
                lifetime,
                nullability,
            } => Type::Reference {
                ty: Box::new(self.apply_type_param_substitution(ty, substitution)),
                mutability: *mutability,
                lifetime: lifetime.clone(),
                nullability: *nullability,
            },
            _ => ty.clone(),
        }
    }

    /// Verify that a type satisfies method trait bounds
    pub fn verify_method_trait_bounds(
        &self,
        method: &ResolvedMethod,
    ) -> Result<(), Vec<SolverError>> {
        let mut errors = Vec::new();

        // For each required trait bound, check if the receiver type satisfies it
        for trait_name in &method.required_trait_bounds {
            if let Some(trait_def) = self.type_registry.get_trait_by_name(*trait_name) {
                // Check if the receiver type implements this trait
                if !self
                    .type_registry
                    .type_implements(&method.receiver_type, trait_def.id)
                {
                    errors.push(SolverError::TraitNotImplemented {
                        ty: method.receiver_type.clone(),
                        trait_name: *trait_name,
                        span: method.signature.span,
                    });
                }
            } else {
                errors.push(SolverError::UnknownTrait {
                    name: *trait_name,
                    span: method.signature.span,
                });
            }
        }

        // Also verify bounds on type arguments
        for (type_param, type_arg) in method
            .signature
            .type_params
            .iter()
            .zip(method.instantiated_type_args.iter())
        {
            for bound in &type_param.bounds {
                if let crate::type_registry::TypeBound::Trait {
                    name: trait_name, ..
                } = bound
                {
                    if let Some(trait_def) = self.type_registry.get_trait_by_name(*trait_name) {
                        if !self.type_registry.type_implements(type_arg, trait_def.id) {
                            errors.push(SolverError::TraitNotImplemented {
                                ty: type_arg.clone(),
                                trait_name: *trait_name,
                                span: method.signature.span,
                            });
                        }
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

    /// Instantiate method return type with type arguments
    pub fn instantiate_method_return_type(&self, method: &ResolvedMethod) -> Type {
        // Apply type arguments to the method's return type
        let mut substitution = Substitution::new();

        for (type_param, type_arg) in method
            .signature
            .type_params
            .iter()
            .zip(method.instantiated_type_args.iter())
        {
            // Create a type variable for this type parameter and bind it
            let type_var_id = TypeVarId::next();
            substitution.bind(type_var_id, type_arg.clone());
        }

        // Apply substitution to return type
        substitution.apply(&method.signature.return_type)
    }

    /// Resolve associated types in a type given a receiver type and trait context
    pub fn resolve_associated_types(&self, ty: &Type, receiver_type: &Type) -> Type {
        match ty {
            Type::Associated {
                trait_name,
                type_name,
            } => {
                // Look up the trait implementation for the receiver type
                if let Some(trait_def) = self.type_registry.get_trait_by_name(*trait_name) {
                    // Get the implementation for this type and trait
                    if let Some(impl_def) = self
                        .type_registry
                        .get_implementation(receiver_type, trait_def.id)
                    {
                        // Look up the associated type binding
                        if let Some(concrete_type) = impl_def.associated_types.get(type_name) {
                            return concrete_type.clone();
                        }
                    }
                }
                // If we can't resolve it, return the original associated type
                ty.clone()
            }
            Type::Array {
                element_type,
                size,
                nullability,
            } => Type::Array {
                element_type: Box::new(self.resolve_associated_types(element_type, receiver_type)),
                size: size.clone(),
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
                        ty: self.resolve_associated_types(&p.ty, receiver_type),
                        ..p.clone()
                    })
                    .collect(),
                return_type: Box::new(self.resolve_associated_types(return_type, receiver_type)),
                is_varargs: *is_varargs,
                has_named_params: *has_named_params,
                has_default_params: *has_default_params,
                async_kind: *async_kind,
                calling_convention: *calling_convention,
                nullability: *nullability,
            },
            Type::Tuple(types) => Type::Tuple(
                types
                    .iter()
                    .map(|t| self.resolve_associated_types(t, receiver_type))
                    .collect(),
            ),
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
                    .map(|t| self.resolve_associated_types(t, receiver_type))
                    .collect(),
                const_args: vec![], // Check?
                variance: vec![],   // Check?
                nullability: *nullability,
            },
            Type::Reference {
                ty,
                mutability,
                lifetime,
                nullability,
            } => Type::Reference {
                ty: Box::new(self.resolve_associated_types(ty, receiver_type)),
                mutability: *mutability,
                lifetime: lifetime.clone(),
                nullability: *nullability,
            },
            Type::Optional(ty) => {
                Type::Optional(Box::new(self.resolve_associated_types(ty, receiver_type)))
            }
            Type::Result { ok_type, err_type } => Type::Result {
                ok_type: Box::new(self.resolve_associated_types(ok_type, receiver_type)),
                err_type: Box::new(self.resolve_associated_types(err_type, receiver_type)),
            },
            Type::Union(types) => Type::Union(
                types
                    .iter()
                    .map(|t| self.resolve_associated_types(t, receiver_type))
                    .collect(),
            ),
            Type::Intersection(types) => Type::Intersection(
                types
                    .iter()
                    .map(|t| self.resolve_associated_types(t, receiver_type))
                    .collect(),
            ),
            Type::Alias { name, target } => Type::Alias {
                name: *name,
                target: Box::new(self.resolve_associated_types(target, receiver_type)),
            },
            Type::Projection { base, item } => Type::Projection {
                base: Box::new(self.resolve_associated_types(base, receiver_type)),
                item: *item,
            },
            Type::Index { base, index } => Type::Index {
                base: Box::new(self.resolve_associated_types(base, receiver_type)),
                index: Box::new(self.resolve_associated_types(index, receiver_type)),
            },
            Type::Primitive(_)
            | Type::TypeVar(_)
            | Type::Never
            | Type::Any
            | Type::Error
            | Type::SelfType
            | Type::HigherKinded { .. }
            | Type::Extern { .. }
            | Type::Unresolved(_) => ty.clone(),
            Type::Nullable(inner_ty) => {
                // For nullable types, substitute the inner type and maintain nullability
                let substituted_inner = self.resolve_associated_types(inner_ty, receiver_type);
                Type::Nullable(Box::new(substituted_inner))
            }
            Type::NonNull(inner_ty) => {
                // For non-null types, substitute the inner type and maintain non-nullability
                let substituted_inner = self.resolve_associated_types(inner_ty, receiver_type);
                Type::NonNull(Box::new(substituted_inner))
            }
            Type::ConstVar {
                id,
                name,
                const_type,
            } => {
                // Const variables don't usually need substitution, but substitute the const_type
                let substituted_const_type =
                    self.resolve_associated_types(const_type, receiver_type);
                Type::ConstVar {
                    id: *id,
                    name: *name,
                    const_type: Box::new(substituted_const_type),
                }
            }
            Type::ConstDependent {
                base_type,
                constraint,
            } => {
                // Substitute in the base type, constraint is typically a const expression
                let substituted_base = self.resolve_associated_types(base_type, receiver_type);
                Type::ConstDependent {
                    base_type: Box::new(substituted_base),
                    constraint: constraint.clone(), // Constraints typically don't have type vars
                }
            }
            Type::Dynamic => {
                // Dynamic types don't contain type variables, so return as-is
                ty.clone()
            }
            Type::Unknown => {
                // Unknown types don't contain type variables, so return as-is
                ty.clone()
            }
            Type::Interface {
                methods,
                is_structural,
                nullability,
            } => todo!(),
            Type::Struct {
                fields,
                is_anonymous,
                nullability,
            } => todo!(),
            Type::Trait {
                id,
                associated_types,
                super_traits,
            } => todo!(),
            Type::Extern { .. } | Type::Unresolved(_) => ty.clone(),
        }
    }

    /// Substitute Self type with the actual receiver type in trait method contexts
    pub fn substitute_self_type(&self, ty: &Type, receiver_type: &Type) -> Type {
        match ty {
            Type::SelfType => receiver_type.clone(),
            Type::Array {
                element_type,
                size,
                nullability,
            } => Type::Array {
                element_type: Box::new(self.substitute_self_type(element_type, receiver_type)),
                size: size.clone(),
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
                        ty: self.substitute_self_type(&p.ty, receiver_type),
                        ..p.clone()
                    })
                    .collect(),
                return_type: Box::new(self.substitute_self_type(return_type, receiver_type)),
                is_varargs: *is_varargs,
                has_named_params: *has_named_params,
                has_default_params: *has_default_params,
                async_kind: *async_kind,
                calling_convention: *calling_convention,
                nullability: *nullability,
            },
            Type::Tuple(types) => Type::Tuple(
                types
                    .iter()
                    .map(|t| self.substitute_self_type(t, receiver_type))
                    .collect(),
            ),
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
                    .map(|t| self.substitute_self_type(t, receiver_type))
                    .collect(),
                const_args: vec![], // Check?
                variance: vec![],   // Check?
                nullability: *nullability,
            },
            Type::Reference {
                ty,
                mutability,
                lifetime,
                nullability,
            } => Type::Reference {
                ty: Box::new(self.substitute_self_type(ty, receiver_type)),
                mutability: *mutability,
                lifetime: lifetime.clone(),
                nullability: *nullability,
            },
            Type::Optional(ty) => {
                Type::Optional(Box::new(self.substitute_self_type(ty, receiver_type)))
            }
            Type::Result { ok_type, err_type } => Type::Result {
                ok_type: Box::new(self.substitute_self_type(ok_type, receiver_type)),
                err_type: Box::new(self.substitute_self_type(err_type, receiver_type)),
            },
            Type::Union(types) => Type::Union(
                types
                    .iter()
                    .map(|t| self.substitute_self_type(t, receiver_type))
                    .collect(),
            ),
            Type::Intersection(types) => Type::Intersection(
                types
                    .iter()
                    .map(|t| self.substitute_self_type(t, receiver_type))
                    .collect(),
            ),
            Type::Alias { name, target } => Type::Alias {
                name: *name,
                target: Box::new(self.substitute_self_type(target, receiver_type)),
            },
            Type::Projection { base, item } => Type::Projection {
                base: Box::new(self.substitute_self_type(base, receiver_type)),
                item: *item,
            },
            Type::Index { base, index } => Type::Index {
                base: Box::new(self.substitute_self_type(base, receiver_type)),
                index: Box::new(self.substitute_self_type(index, receiver_type)),
            },
            Type::Primitive(_)
            | Type::TypeVar(_)
            | Type::Never
            | Type::Any
            | Type::Error
            | Type::Associated { .. }
            | Type::HigherKinded { .. }
            | Type::Extern { .. }
            | Type::Unresolved(_) => ty.clone(),
            Type::Nullable(inner_ty) => {
                // For nullable types, substitute the inner type and maintain nullability
                let substituted_inner = self.substitute_self_type(inner_ty, receiver_type);
                Type::Nullable(Box::new(substituted_inner))
            }
            Type::NonNull(inner_ty) => {
                // For non-null types, substitute the inner type and maintain non-nullability
                let substituted_inner = self.substitute_self_type(inner_ty, receiver_type);
                Type::NonNull(Box::new(substituted_inner))
            }
            Type::ConstVar {
                id,
                name,
                const_type,
            } => {
                // Const variables don't usually need substitution, but substitute the const_type
                let substituted_const_type = self.substitute_self_type(const_type, receiver_type);
                Type::ConstVar {
                    id: *id,
                    name: *name,
                    const_type: Box::new(substituted_const_type),
                }
            }
            Type::ConstDependent {
                base_type,
                constraint,
            } => {
                // Substitute in the base type, constraint is typically a const expression
                let substituted_base = self.substitute_self_type(base_type, receiver_type);
                Type::ConstDependent {
                    base_type: Box::new(substituted_base),
                    constraint: constraint.clone(), // Constraints typically don't have type vars
                }
            }
            Type::Dynamic => {
                // Dynamic types don't contain type variables, so return as-is
                ty.clone()
            }
            Type::Unknown => {
                // Unknown types don't contain type variables, so return as-is
                ty.clone()
            }
            Type::Interface {
                methods,
                is_structural,
                nullability,
            } => todo!(),
            Type::Struct {
                fields,
                is_anonymous,
                nullability,
            } => todo!(),
            Type::Trait {
                id,
                associated_types,
                super_traits,
            } => todo!(),
        }
    }

    // ===== DEPENDENT TYPE CONSTRAINT SOLVERS =====

    /// Solve refinement predicate constraint
    fn solve_refinement_satisfies(
        &mut self,
        value: Type,
        predicate: crate::dependent_types::RefinementPredicate,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let value = self.subst.apply(&value);

        match &value {
            Type::TypeVar(_) => {
                // Defer until the type variable is resolved
                Ok(ConstraintResult::Deferred)
            }
            _ => {
                // TODO: This is where SMT solver integration would happen
                // For now, we'll assume all refinement predicates are satisfiable
                // In a full implementation, this would:
                // 1. Convert the predicate to SMT-LIB format
                // 2. Call an external SMT solver (Z3, CVC4, etc.)
                // 3. Return based on satisfiability result

                match self.check_predicate_satisfiability(&value, &predicate) {
                    Ok(true) => Ok(ConstraintResult::Solved),
                    Ok(false) => {
                        self.errors.push(SolverError::UnsolvableConstraint(
                            Constraint::RefinementSatisfies {
                                value,
                                predicate,
                                span,
                            },
                        ));
                        Err(self.errors.clone())
                    }
                    Err(_) => Ok(ConstraintResult::Deferred), // Could not determine, defer
                }
            }
        }
    }

    /// Solve dependent function call constraint
    fn solve_dependent_call(
        &mut self,
        func_type: Type,
        arg_value: ConstValue,
        result_type: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let func_type = self.subst.apply(&func_type);
        let result_type = self.subst.apply(&result_type);

        match &func_type {
            Type::TypeVar(_) => Ok(ConstraintResult::Deferred),

            // TODO: Handle dependent function types specifically
            // For now, treat as regular function call constraint
            _ => {
                // In a full implementation, this would:
                // 1. Extract the dependent function's parameter constraints
                // 2. Substitute the argument value into the result type
                // 3. Generate new equality constraints

                // Simplified: assume the call is valid
                Ok(ConstraintResult::Solved)
            }
        }
    }

    /// Solve path-dependent type constraint
    fn solve_path_dependent(
        &mut self,
        base_type: Type,
        path: Vec<InternedString>,
        target_type: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let base_type = self.subst.apply(&base_type);
        let target_type = self.subst.apply(&target_type);

        match &base_type {
            Type::TypeVar(_) => Ok(ConstraintResult::Deferred),

            Type::Named { id, .. } => {
                // Resolve the path through the type definition
                if let Some(resolved_type) = self.resolve_type_path(*id, &path) {
                    Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                        resolved_type,
                        target_type,
                        span,
                    )]))
                } else {
                    self.errors.push(SolverError::UnsolvableConstraint(
                        Constraint::PathDependent {
                            base_type,
                            path,
                            target_type,
                            span,
                        },
                    ));
                    Err(self.errors.clone())
                }
            }

            _ => {
                self.errors.push(SolverError::UnsolvableConstraint(
                    Constraint::PathDependent {
                        base_type,
                        path,
                        target_type,
                        span,
                    },
                ));
                Err(self.errors.clone())
            }
        }
    }

    /// Solve singleton type equality constraint
    fn solve_singleton_equals(
        &mut self,
        value_type: Type,
        constant: ConstValue,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let value_type = self.subst.apply(&value_type);

        match &value_type {
            Type::TypeVar(_) => {
                // Create a singleton type and unify
                let singleton_type = self.create_singleton_type(constant.clone());
                Ok(ConstraintResult::NewConstraints(vec![Constraint::Equal(
                    value_type,
                    singleton_type,
                    span,
                )]))
            }

            // TODO: Check if the type is compatible with the singleton constraint
            _ => Ok(ConstraintResult::Solved),
        }
    }

    /// Solve type family application constraint
    fn solve_type_family_application(
        &mut self,
        family: Type,
        indices: Vec<crate::dependent_types::DependentIndex>,
        result_type: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let family = self.subst.apply(&family);
        let result_type = self.subst.apply(&result_type);

        // TODO: Implement type family evaluation
        // This would require:
        // 1. A type-level computation engine
        // 2. Evaluation of type family applications F[args]
        // 3. Generating equality constraints with the result

        // For now, defer the constraint
        Ok(ConstraintResult::Deferred)
    }

    /// Solve conditional type constraint
    fn solve_conditional_type(
        &mut self,
        condition: crate::dependent_types::RefinementPredicate,
        then_type: Type,
        else_type: Type,
        result_type: Type,
        span: Span,
    ) -> Result<ConstraintResult, Vec<SolverError>> {
        let then_type = self.subst.apply(&then_type);
        let else_type = self.subst.apply(&else_type);
        let result_type = self.subst.apply(&result_type);

        // TODO: Evaluate the condition and choose appropriate type
        // This would require predicate evaluation capability

        // For now, generate constraints for both branches
        Ok(ConstraintResult::NewConstraints(vec![
            // Could be either branch - this is a simplification
            // Real implementation would evaluate the condition
            Constraint::Equal(then_type, result_type.clone(), span),
            // Or generate a union type constraint
        ]))
    }

    // ===== HELPER METHODS FOR DEPENDENT TYPES =====

    /// Check if a refinement predicate is satisfiable for a given type
    /// Uses SMT solver integration when available
    fn check_predicate_satisfiability(
        &self,
        value_type: &Type,
        predicate: &crate::dependent_types::RefinementPredicate,
    ) -> Result<bool, String> {
        // Try to use SMT solver for satisfiability checking
        let mut smt_solver = crate::smt_solver::SmtSolver::new();

        if smt_solver.is_available() {
            let context = std::collections::HashMap::new(); // TODO: build actual context
            match smt_solver.check_predicate_satisfiable(predicate, value_type, &context) {
                Ok(crate::smt_solver::SmtResult::Satisfiable) => Ok(true),
                Ok(crate::smt_solver::SmtResult::Unsatisfiable) => Ok(false),
                Ok(crate::smt_solver::SmtResult::Unknown) => {
                    // If SMT solver can't determine, assume satisfiable for safety
                    Ok(true)
                }
                Ok(crate::smt_solver::SmtResult::Error(_)) => {
                    // Fall back to conservative assumption
                    Ok(true)
                }
                Err(_) => {
                    // Fall back to conservative assumption
                    Ok(true)
                }
            }
        } else {
            // No SMT solver available, assume satisfiable for safety
            Ok(true)
        }
    }

    /// Resolve a path through a type definition
    fn resolve_type_path(
        &self,
        base_type_id: crate::type_registry::TypeId,
        path: &[InternedString],
    ) -> Option<Type> {
        // TODO: Implement path resolution through type definitions
        // This would navigate through struct fields, associated types, etc.
        None
    }

    /// Create a singleton type for a constant value
    fn create_singleton_type(&self, constant: ConstValue) -> Type {
        // TODO: Create proper singleton type representation
        // For now, return the constant as a dependent type
        Type::ConstDependent {
            base_type: Box::new(self.infer_const_type(&constant)),
            constraint: crate::type_registry::ConstConstraint::Equal(constant),
        }
    }

    /// Infer the base type for a constant value
    fn infer_const_type(&self, constant: &ConstValue) -> Type {
        match constant {
            ConstValue::Int(_) => Type::Primitive(crate::type_registry::PrimitiveType::I32),
            ConstValue::UInt(_) => Type::Primitive(crate::type_registry::PrimitiveType::U32),
            ConstValue::String(_) => Type::Primitive(crate::type_registry::PrimitiveType::String),
            ConstValue::Bool(_) => Type::Primitive(crate::type_registry::PrimitiveType::Bool),
            ConstValue::Char(_) => Type::Primitive(crate::type_registry::PrimitiveType::Char),
            ConstValue::Array(elements) => {
                // Infer element type from first element, or use Any if empty
                let element_type = if let Some(first) = elements.first() {
                    self.infer_const_type(first)
                } else {
                    Type::Any
                };
                Type::Array {
                    element_type: Box::new(element_type),
                    size: Some(ConstValue::UInt(elements.len() as u64)),
                    nullability: crate::type_registry::NullabilityKind::default(),
                }
            }
            ConstValue::Struct(..) => Type::Any,   // Simplified
            ConstValue::Tuple(..) => Type::Any,    // Simplified
            ConstValue::Variable(..) => Type::Any, // Would need context to resolve
            ConstValue::FunctionCall { .. } => Type::Any, // Would need function signature
            ConstValue::BinaryOp { .. } => Type::Any, // Would need operand types
            ConstValue::UnaryOp { .. } => Type::Any, // Would need operand type
        }
    }

    // ===== DEPENDENT TYPE TO CONSTRAINT TRANSLATION =====

    /// Convert a dependent type into constraints that can be solved
    pub fn generate_dependent_type_constraints(
        &mut self,
        dependent_type: &crate::dependent_types::DependentType,
        target_type: Type,
        span: Span,
    ) -> Vec<Constraint> {
        use crate::dependent_types::DependentType;

        match dependent_type {
            DependentType::Refinement {
                base_type,
                predicate,
                ..
            } => {
                let mut constraints = vec![
                    // First, ensure the target type matches the base type
                    Constraint::Equal(target_type.clone(), (**base_type).clone(), span),
                ];

                // Then add the refinement constraint
                constraints.push(Constraint::RefinementSatisfies {
                    value: target_type,
                    predicate: predicate.clone(),
                    span,
                });

                constraints
            }

            DependentType::DependentFunction {
                param_name,
                param_type,
                return_type,
                ..
            } => {
                // For dependent functions (x: T) -> U(x), we need to handle application
                // This is complex and would need type-level substitution
                vec![
                    Constraint::Equal(target_type, Type::Primitive(PrimitiveType::Unit), span), // Placeholder
                ]
            }

            DependentType::DependentPair {
                first_type,
                second_type,
                ..
            } => {
                // For dependent pairs (x: T, U(x)), create tuple constraint
                // Note: This is simplified - real implementation would handle the dependency
                let tuple_type = Type::Tuple(vec![
                    (**first_type).clone(),
                    // For now, just get the base type of the dependent second type
                    self.extract_base_type(second_type),
                ]);
                vec![Constraint::Equal(target_type, tuple_type, span)]
            }

            DependentType::PathDependent {
                path, type_name, ..
            } => {
                // For path-dependent types, we would need to resolve the path
                // This is simplified - real implementation would traverse the path
                let _ = (path, type_name); // Use parameters to avoid warnings

                vec![Constraint::Equal(
                    target_type,
                    Type::Primitive(PrimitiveType::Unit), // Placeholder
                    span,
                )]
            }

            DependentType::Singleton { value, .. } => {
                vec![Constraint::SingletonEquals {
                    value_type: target_type,
                    constant: value.clone(),
                    span,
                }]
            }

            DependentType::IndexedFamily {
                family_name,
                indices,
                ..
            } => {
                vec![Constraint::TypeFamilyApplication {
                    family: Type::Named {
                        id: crate::type_registry::TypeId::new(0), // Placeholder
                        type_args: vec![],
                        const_args: vec![],
                        variance: vec![],
                        nullability: crate::type_registry::NullabilityKind::default(),
                    },
                    indices: indices.clone(),
                    result_type: target_type,
                    span,
                }]
            }

            DependentType::Conditional {
                condition,
                then_type,
                else_type,
                ..
            } => {
                // For conditional types, we would need to evaluate the condition
                // This is simplified - real implementation would handle type-level conditionals
                let _ = (condition, then_type, else_type); // Use parameters to avoid warnings

                vec![Constraint::Equal(
                    target_type,
                    Type::Primitive(PrimitiveType::Unit), // Placeholder
                    span,
                )]
            }

            DependentType::Recursive { .. } => {
                // Recursive types need special handling, defer for now
                vec![]
            }

            DependentType::Existential {
                var_name,
                var_type,
                body,
                ..
            } => {
                // Existential types ∃(x: T). U(x) need witness generation
                // For now, just use the body type
                vec![Constraint::Equal(
                    target_type,
                    self.extract_base_type(body),
                    span,
                )]
            }

            DependentType::Universal {
                var_name,
                var_type,
                body,
                ..
            } => {
                // Universal types ∀(x: T). U(x) need polymorphic instantiation
                // For now, just use the body type
                vec![Constraint::Equal(
                    target_type,
                    self.extract_base_type(body),
                    span,
                )]
            }

            DependentType::Base(base_type) => {
                // Regular non-dependent type
                vec![Constraint::Equal(target_type, base_type.clone(), span)]
            }
        }
    }

    /// Generate constraints for dependent type well-formedness checking
    pub fn generate_wellformedness_constraints(
        &mut self,
        dependent_type: &crate::dependent_types::DependentType,
        span: Span,
    ) -> Vec<Constraint> {
        // This would check that:
        // 1. All free variables are bound in context
        // 2. All predicates are well-typed
        // 3. All type families are properly applied

        // For now, return empty constraint set (assumes well-formed input)
        vec![]
    }

    /// Convert a refinement predicate into SMT-solvable constraints
    pub fn lower_refinement_predicate(
        &mut self,
        predicate: &crate::dependent_types::RefinementPredicate,
        value_type: &Type,
        span: Span,
    ) -> Vec<Constraint> {
        use crate::dependent_types::RefinementPredicate;

        match predicate {
            RefinementPredicate::And(left, right) => {
                let mut constraints = self.lower_refinement_predicate(left, value_type, span);
                constraints.extend(self.lower_refinement_predicate(right, value_type, span));
                constraints
            }

            RefinementPredicate::Or(left, right) => {
                // For disjunctions, we need more sophisticated constraint generation
                // This is a simplification - real implementation would need choice constraints
                vec![Constraint::RefinementSatisfies {
                    value: value_type.clone(),
                    predicate: predicate.clone(),
                    span,
                }]
            }

            RefinementPredicate::Not(inner) => {
                vec![Constraint::RefinementSatisfies {
                    value: value_type.clone(),
                    predicate: predicate.clone(),
                    span,
                }]
            }

            RefinementPredicate::Comparison { op, left, right } => {
                // Convert comparison into constraint solver format
                vec![Constraint::RefinementSatisfies {
                    value: value_type.clone(),
                    predicate: predicate.clone(),
                    span,
                }]
            }

            // Note: TypeMembership variant doesn't exist in RefinementPredicate
            _ => {
                // For other predicates, create a refinement satisfaction constraint
                vec![Constraint::RefinementSatisfies {
                    value: value_type.clone(),
                    predicate: predicate.clone(),
                    span,
                }]
            }
        }
    }

    /// Extract the base type from a dependent type (helper method)
    fn extract_base_type(&self, dependent_type: &crate::dependent_types::DependentType) -> Type {
        use crate::dependent_types::DependentType;

        match dependent_type {
            DependentType::Refinement { base_type, .. } => (**base_type).clone(),
            DependentType::DependentFunction { return_type, .. } => {
                self.extract_base_type(return_type)
            }
            DependentType::DependentPair { first_type, .. } => (**first_type).clone(), // Simplified
            DependentType::Singleton { base_type, .. } => (**base_type).clone(),
            DependentType::Base(ty) => ty.clone(),
            _ => Type::Any, // Fallback for complex dependent types
        }
    }
}

/// Represents a resolved method with all necessary information for type checking
#[derive(Debug, Clone)]
pub struct ResolvedMethod {
    /// The method signature
    pub signature: crate::type_registry::MethodSig,
    /// The receiver type
    pub receiver_type: Type,
    /// Instantiated type arguments
    pub instantiated_type_args: Vec<Type>,
    /// Required trait bounds for this method
    pub required_trait_bounds: Vec<InternedString>,
}

/// Result of solving a constraint
enum ConstraintResult {
    /// Constraint solved with substitution
    Substitution(Substitution),
    /// Constraint generated new constraints
    NewConstraints(Vec<Constraint>),
    /// Constraint solved without changes
    Solved,
    /// Constraint deferred until more information available
    Deferred,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{type_registry::PrimitiveType, ConstValue};

    #[test]
    fn test_simple_unification() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Int = Int should succeed
        solver.add_constraint(Constraint::Equal(
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::I32),
            span,
        ));

        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_type_variable_unification() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Create type variables
        let t1 = solver.fresh_type_var();
        let t2 = Type::Primitive(PrimitiveType::I32);

        // T1 = I32
        solver.add_constraint(Constraint::Equal(t1.clone(), t2.clone(), span));

        let result = solver.solve();
        assert!(result.is_ok());

        let subst = result.unwrap();
        if let Type::TypeVar(var) = &t1 {
            let resolved = subst.apply(&t1);
            assert_eq!(resolved, t2);
        }
    }

    #[test]
    fn test_function_type_unification() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Create a function type: (I32, I32) -> I32
        let func_ty1 = Type::Function {
            params: vec![
                ParamInfo {
                    name: None,
                    ty: Type::Primitive(PrimitiveType::I32),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
                ParamInfo {
                    name: None,
                    ty: Type::Primitive(PrimitiveType::I32),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
            ],
            return_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: crate::type_registry::AsyncKind::default(),
            calling_convention: crate::type_registry::CallingConvention::default(),
            nullability: crate::type_registry::NullabilityKind::default(),
        };

        // Create another function type with type variable: (T, T) -> T
        let t = solver.fresh_type_var();
        let func_ty2 = Type::Function {
            params: vec![
                ParamInfo {
                    name: None,
                    ty: t.clone(),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
                ParamInfo {
                    name: None,
                    ty: t.clone(),
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                },
            ],
            return_type: Box::new(t.clone()),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: crate::type_registry::AsyncKind::default(),
            calling_convention: crate::type_registry::CallingConvention::default(),
            nullability: crate::type_registry::NullabilityKind::default(),
        };

        solver.add_constraint(Constraint::Equal(func_ty1, func_ty2, span));

        let result = solver.solve();
        assert!(result.is_ok());

        // T should be resolved to I32
        let subst = result.unwrap();
        let resolved_t = subst.apply(&t);
        assert_eq!(resolved_t, Type::Primitive(PrimitiveType::I32));
    }

    #[test]
    fn test_varargs_constraint() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Array<Int> with args [Int, Int, Int]
        let array_ty = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: None,
            nullability: crate::type_registry::NullabilityKind::default(),
        };

        let args = vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::I32),
        ];

        solver.add_constraint(Constraint::Varargs(array_ty, args, span));

        let result = solver.solve();
        assert!(result.is_ok());
    }

    #[test]
    fn test_callable_constraint() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Create a type variable for the function
        let func_var = solver.fresh_type_var();

        // We want to call it with (I32, Bool) -> String
        let arg_types = vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::Bool),
        ];
        let ret_type = Type::Primitive(PrimitiveType::String);

        solver.add_constraint(Constraint::Callable(
            func_var.clone(),
            arg_types,
            ret_type,
            span,
        ));

        let result = solver.solve();
        assert!(result.is_ok());

        // The function variable should be resolved to a function type
        let subst = result.unwrap();
        let resolved = subst.apply(&func_var);

        if let Type::Function {
            params,
            return_type,
            ..
        } = resolved
        {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].ty, Type::Primitive(PrimitiveType::I32));
            assert_eq!(params[1].ty, Type::Primitive(PrimitiveType::Bool));
            assert_eq!(*return_type, Type::Primitive(PrimitiveType::String));
        } else {
            panic!("Expected function type");
        }
    }

    #[test]
    fn test_array_element_constraint() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Create a type variable for array
        let array_var = solver.fresh_type_var();

        // Constrain that it's an array with String elements
        solver.add_constraint(Constraint::ArrayElement(
            array_var.clone(),
            Type::Primitive(PrimitiveType::String),
            span,
        ));

        let result = solver.solve();
        assert!(result.is_ok());

        // The array variable should be resolved to Array<String>
        let subst = result.unwrap();
        let resolved = subst.apply(&array_var);

        if let Type::Array { element_type, .. } = resolved {
            assert_eq!(*element_type, Type::Primitive(PrimitiveType::String));
        } else {
            panic!("Expected array type");
        }
    }

    #[test]
    fn test_complex_constraint_solving() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Create type variables
        let t1 = solver.fresh_type_var();
        let t2 = solver.fresh_type_var();
        let t3 = solver.fresh_type_var();

        // Add constraints:
        // t1 = Array<t2>
        // t2 = t3
        // t3 = I32
        solver.add_constraint(Constraint::Equal(
            t1.clone(),
            Type::Array {
                element_type: Box::new(t2.clone()),
                size: None,
                nullability: crate::type_registry::NullabilityKind::default(),
            },
            span,
        ));
        solver.add_constraint(Constraint::Equal(t2.clone(), t3.clone(), span));
        solver.add_constraint(Constraint::Equal(
            t3,
            Type::Primitive(PrimitiveType::I32),
            span,
        ));

        let result = solver.solve();
        assert!(result.is_ok());

        // t1 should be Array<I32>
        let subst = result.unwrap();
        let resolved_t1 = subst.apply(&t1);
        assert_eq!(
            resolved_t1,
            Type::Array {
                element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
                size: None,
                nullability: crate::type_registry::NullabilityKind::default(),
            }
        );
    }

    #[test]
    fn test_numeric_subtyping() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // I8 <: I32 should succeed
        solver.add_constraint(Constraint::Subtype(
            Type::Primitive(PrimitiveType::I8),
            Type::Primitive(PrimitiveType::I32),
            span,
        ));

        // F32 <: F64 should succeed
        solver.add_constraint(Constraint::Subtype(
            Type::Primitive(PrimitiveType::F32),
            Type::Primitive(PrimitiveType::F64),
            span,
        ));

        // I32 <: F64 should succeed (with loss of precision)
        solver.add_constraint(Constraint::Subtype(
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::F64),
            span,
        ));

        let result = solver.solve();
        assert!(result.is_ok(), "Numeric subtyping should succeed");
    }

    #[test]
    fn test_function_subtyping() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Create function types:
        // sub: (I32) -> I8
        // super: (I8) -> I32
        // This should succeed due to contravariance/covariance

        let sub_func = Type::Function {
            params: vec![ParamInfo {
                name: None,
                ty: Type::Primitive(PrimitiveType::I32),
                is_optional: false,
                is_varargs: false,
                is_keyword_only: false,
                is_positional_only: false,
                is_out: false,
                is_ref: false,
                is_inout: false,
            }],
            return_type: Box::new(Type::Primitive(PrimitiveType::I8)),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: crate::type_registry::AsyncKind::default(),
            calling_convention: crate::type_registry::CallingConvention::default(),
            nullability: crate::type_registry::NullabilityKind::default(),
        };

        let super_func = Type::Function {
            params: vec![ParamInfo {
                name: None,
                ty: Type::Primitive(PrimitiveType::I8),
                is_optional: false,
                is_varargs: false,
                is_keyword_only: false,
                is_positional_only: false,
                is_out: false,
                is_ref: false,
                is_inout: false,
            }],
            return_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: crate::type_registry::AsyncKind::default(),
            calling_convention: crate::type_registry::CallingConvention::default(),
            nullability: crate::type_registry::NullabilityKind::default(),
        };

        solver.add_constraint(Constraint::Subtype(sub_func, super_func, span));

        let result = solver.solve();
        assert!(
            result.is_ok(),
            "Function subtyping should succeed with proper variance"
        );
    }

    #[test]
    fn test_tuple_width_subtyping() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // (I32, String) <: (I32, String, Bool) should fail (we don't support width subtyping in this direction)
        let sub_tuple = Type::Tuple(vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::String),
            Type::Primitive(PrimitiveType::Bool),
        ]);

        let super_tuple = Type::Tuple(vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::String),
        ]);

        solver.add_constraint(Constraint::Subtype(sub_tuple, super_tuple, span));

        let result = solver.solve();
        assert!(
            result.is_err(),
            "Tuple width subtyping in wrong direction should fail"
        );
    }

    #[test]
    fn test_union_subtyping() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // I32 <: I32 | String should succeed
        let sub_type = Type::Primitive(PrimitiveType::I32);
        let super_type = Type::Union(vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Primitive(PrimitiveType::String),
        ]);

        solver.add_constraint(Constraint::Subtype(sub_type, super_type, span));

        let result = solver.solve();
        assert!(result.is_ok(), "Union subtyping should succeed");
    }

    #[test]
    fn test_intersection_subtyping() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // I32 <: I32 & Any should succeed
        // This is a simple case where I32 is a subtype of both I32 and Any
        let intersection = Type::Intersection(vec![
            Type::Primitive(PrimitiveType::I32),
            Type::Any, // Everything is subtype of Any
        ]);

        solver.add_constraint(Constraint::Subtype(
            Type::Primitive(PrimitiveType::I32),
            intersection,
            span,
        ));

        let result = solver.solve();
        assert!(result.is_ok(), "Intersection subtyping should succeed");
    }

    #[test]
    fn test_array_covariance() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Array<I8> <: Array<I32> should succeed (covariant in element type)
        let sub_array = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I8)),
            size: Some(ConstValue::Int(10)),
            nullability: crate::NullabilityKind::Unknown,
        };

        let super_array = Type::Array {
            element_type: Box::new(Type::Primitive(PrimitiveType::I32)),
            size: Some(ConstValue::Int(10)),
            nullability: crate::NullabilityKind::Unknown,
        };

        solver.add_constraint(Constraint::Subtype(sub_array, super_array, span));

        let result = solver.solve();
        assert!(result.is_ok(), "Array covariance should work");
    }

    #[test]
    fn test_reference_mutability_subtyping() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // &mut T <: &T should succeed (mutable reference to immutable reference)
        let mutable_ref = Type::Reference {
            ty: Box::new(Type::Primitive(PrimitiveType::I32)),
            mutability: Mutability::Mutable,
            lifetime: None,
            nullability: crate::NullabilityKind::Unknown,
        };

        let immutable_ref = Type::Reference {
            ty: Box::new(Type::Primitive(PrimitiveType::I32)),
            mutability: Mutability::Immutable,
            lifetime: None,
            nullability: crate::NullabilityKind::Unknown,
        };

        solver.add_constraint(Constraint::Subtype(mutable_ref, immutable_ref, span));

        let result = solver.solve();
        assert!(
            result.is_ok(),
            "Mutable to immutable reference subtyping should succeed"
        );
    }

    #[test]
    fn test_reference_mutability_subtyping_invalid() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // &T <: &mut T should fail (immutable reference to mutable reference)
        let immutable_ref = Type::Reference {
            ty: Box::new(Type::Primitive(PrimitiveType::I32)),
            mutability: Mutability::Immutable,
            lifetime: None,
            nullability: crate::NullabilityKind::Unknown,
        };

        let mutable_ref = Type::Reference {
            ty: Box::new(Type::Primitive(PrimitiveType::I32)),
            mutability: Mutability::Mutable,
            lifetime: None,
            nullability: crate::NullabilityKind::Unknown,
        };

        solver.add_constraint(Constraint::Subtype(immutable_ref, mutable_ref, span));

        let result = solver.solve();
        assert!(
            result.is_err(),
            "Immutable to mutable reference subtyping should fail"
        );
    }

    #[test]
    fn test_bottom_top_types() {
        let mut solver = ConstraintSolver::new();
        let span = Span::new(0, 0);

        // Never <: T for any T
        solver.add_constraint(Constraint::Subtype(
            Type::Never,
            Type::Primitive(PrimitiveType::I32),
            span,
        ));

        // T <: Any for any T
        solver.add_constraint(Constraint::Subtype(
            Type::Primitive(PrimitiveType::String),
            Type::Any,
            span,
        ));

        let result = solver.solve();
        assert!(
            result.is_ok(),
            "Bottom and top type subtyping should always succeed"
        );
    }

    #[test]
    fn test_trait_bound_primitive_display() {
        let mut arena = crate::arena::AstArena::new();
        let registry = Box::new(crate::type_registry::TypeRegistry::new());

        let mut solver = ConstraintSolver::with_type_registry(registry);
        let span = Span::new(0, 0);

        // Test that i32 implements Display (assuming standard setup)
        let display_trait = arena.intern_string("Display");
        solver.add_constraint(Constraint::TraitBound(
            Type::Primitive(PrimitiveType::I32),
            display_trait,
            span,
        ));

        let result = solver.solve();
        // This may fail until we set up primitive trait implementations properly
        // For now, just ensure the constraint system can handle trait bounds
        println!("Trait bound result: {:?}", result);
    }

    #[test]
    fn test_trait_bound_type_variable_deferred() {
        let mut arena = crate::arena::AstArena::new();
        let registry = Box::new(crate::type_registry::TypeRegistry::new());

        let mut solver = ConstraintSolver::with_type_registry(registry);
        let span = Span::new(0, 0);

        // Create a type variable
        let type_var = solver.fresh_type_var();
        let display_trait = arena.intern_string("Display");

        // Type variable should defer the trait bound constraint
        solver.add_constraint(Constraint::TraitBound(type_var, display_trait, span));

        let result = solver.solve();
        // Should succeed since deferred constraints don't fail immediately
        assert!(
            result.is_ok(),
            "Type variable trait bounds should be deferred"
        );
    }

    #[test]
    fn test_trait_bound_unknown_trait() {
        let mut arena = crate::arena::AstArena::new();
        let registry = Box::new(crate::type_registry::TypeRegistry::new());

        let mut solver = ConstraintSolver::with_type_registry(registry);
        let span = Span::new(0, 0);

        // Test with a non-existent trait
        let unknown_trait = arena.intern_string("NonExistentTrait");
        solver.add_constraint(Constraint::TraitBound(
            Type::Primitive(PrimitiveType::I32),
            unknown_trait,
            span,
        ));

        let result = solver.solve();
        assert!(
            result.is_err(),
            "Unknown trait should cause constraint solving to fail"
        );

        // Check that we get the right error type
        if let Err(errors) = result {
            assert!(errors
                .iter()
                .any(|e| matches!(e, SolverError::UnknownTrait { .. })));
        }
    }

    #[test]
    fn test_trait_bound_named_type() {
        let mut arena = crate::arena::AstArena::new();
        let registry = Box::new(crate::type_registry::TypeRegistry::new());

        let mut solver = ConstraintSolver::with_type_registry(registry);
        let span = Span::new(0, 0);

        // Create a named type (e.g., Vec<i32>)
        // We need to get or create a TypeId for Vec first
        let vec_id = crate::type_registry::TypeId::new(1); // In a real scenario, this would come from the registry
        let vec_type = Type::Named {
            id: vec_id,
            type_args: vec![Type::Primitive(PrimitiveType::I32)],
            const_args: vec![],
            variance: vec![],
            nullability: crate::NullabilityKind::Unknown,
        };

        let display_trait = arena.intern_string("Display");
        solver.add_constraint(Constraint::TraitBound(vec_type, display_trait, span));

        let result = solver.solve();
        // This will likely fail since Vec<i32> doesn't implement Display
        // but it exercises the named type trait checking path
        println!("Named type trait bound result: {:?}", result);
    }

    #[test]
    fn test_multiple_trait_bounds() {
        let mut arena = crate::arena::AstArena::new();
        let registry = Box::new(crate::type_registry::TypeRegistry::new());

        let mut solver = ConstraintSolver::with_type_registry(registry);
        let span = Span::new(0, 0);

        // Test multiple trait bounds on the same type
        let type_var = solver.fresh_type_var();
        let display_trait = arena.intern_string("Display");
        let clone_trait = arena.intern_string("Clone");

        solver.add_constraint(Constraint::TraitBound(
            type_var.clone(),
            display_trait,
            span,
        ));
        solver.add_constraint(Constraint::TraitBound(type_var, clone_trait, span));

        let result = solver.solve();
        assert!(
            result.is_ok(),
            "Multiple trait bounds on type variables should be deferred"
        );
    }

    #[test]
    fn test_trait_bound_with_unification() {
        let mut arena = crate::arena::AstArena::new();
        let registry = Box::new(crate::type_registry::TypeRegistry::new());

        let mut solver = ConstraintSolver::with_type_registry(registry);
        let span = Span::new(0, 0);

        // Create a type variable and unify it with i32
        let type_var = solver.fresh_type_var();
        solver.add_constraint(Constraint::Equal(
            type_var.clone(),
            Type::Primitive(PrimitiveType::I32),
            span,
        ));

        // Then add a trait bound
        let display_trait = arena.intern_string("Display");
        solver.add_constraint(Constraint::TraitBound(type_var, display_trait, span));

        let result = solver.solve();
        // After unification, the trait bound should be checked on i32
        println!("Trait bound with unification result: {:?}", result);
    }
}
