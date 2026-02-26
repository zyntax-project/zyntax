//! Gradual Type System Checker
//!
//! Implements Python/Haxe/TypeScript style gradual typing where types can be
//! checked statically or dynamically, with smooth interoperability between
//! typed and untyped code.

use crate::arena::InternedString;
use crate::source::Span;
use crate::{const_evaluator::Literal, *};
use std::collections::{HashMap, HashSet};

/// Gradual type checker for dynamic/static hybrid systems
pub struct GradualTypeChecker {
    /// Runtime type check insertion points
    pub runtime_checks: HashMap<CheckPointId, RuntimeCheck>,

    /// Type evidence tracking for flow-sensitive typing
    pub type_evidence: HashMap<VariableId, TypeEvidence>,

    /// Dynamic dispatch targets
    pub dynamic_targets: HashMap<CallSiteId, Vec<PossibleTarget>>,

    /// Blame tracking for error reporting
    pub blame_map: HashMap<CheckPointId, BlameInfo>,

    /// Type precision levels
    pub precision_levels: HashMap<TypeId, PrecisionLevel>,
}

/// Unique identifier for runtime check insertion points
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckPointId(u32);

/// Unique identifier for variables in gradual typing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VariableId(u32);

/// Unique identifier for call sites
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallSiteId(u32);

/// Runtime type check specification
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeCheck {
    pub check_kind: RuntimeCheckKind,
    pub expected_type: Type,
    pub fallback_behavior: FallbackBehavior,
    pub optimization_hint: OptimizationHint,
    pub span: Span,
}

/// Different kinds of runtime checks
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeCheckKind {
    /// Type assertion: assert that value has specific type
    TypeAssertion,

    /// Type guard: check type and narrow in conditional
    TypeGuard,

    /// Cast: convert between compatible types
    Cast { is_safe: bool },

    /// Duck type check: verify object has required methods/fields
    DuckTypeCheck {
        required_members: Vec<MemberRequirement>,
    },

    /// Interface check: verify object implements interface
    InterfaceCheck { interface_id: TypeId },

    /// Null check: verify value is not null
    NullCheck,

    /// Undefined check: verify value is defined (JavaScript style)
    UndefinedCheck,

    /// Array bounds check: verify array access is valid
    BoundsCheck,

    /// Property existence check: verify property exists on object
    PropertyCheck { property_name: InternedString },

    /// Method existence check: verify method exists and is callable
    MethodCheck {
        method_name: InternedString,
        expected_signature: Option<FunctionSignature>,
    },
}

/// Member requirement for duck typing checks
#[derive(Debug, Clone, PartialEq)]
pub struct MemberRequirement {
    pub name: InternedString,
    pub kind: MemberKind,
    pub expected_type: Type,
    pub is_optional: bool,
}

/// Different kinds of object members
#[derive(Debug, Clone, PartialEq)]
pub enum MemberKind {
    Field,
    Method,
    Property { has_getter: bool, has_setter: bool },
}

/// Behavior when runtime check fails
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackBehavior {
    /// Throw exception with error message
    Throw(String),

    /// Return default value of expected type
    ReturnDefault,

    /// Continue with dynamic typing (no check)
    ContinueDynamic,

    /// Coerce to expected type if possible
    CoerceType,

    /// Call custom error handler
    CallHandler(InternedString),

    /// Gradual mode: issue warning but continue
    Warn(String),
}

/// Optimization hints for runtime checks
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationHint {
    /// Check is likely to succeed (hot path)
    LikelySuccess,

    /// Check is likely to fail (error path)
    LikelyFailure,

    /// Check can be cached for this object
    Cacheable,

    /// Check can be eliminated if proven statically
    EliminateIfProven,

    /// Check is performance critical
    PerformanceCritical,

    /// Check can be deferred to usage site
    DeferToUsage,
}

/// Type evidence for flow-sensitive typing
#[derive(Debug, Clone)]
pub struct TypeEvidence {
    pub variable_id: VariableId,
    pub evidence_chain: Vec<Evidence>,
    pub current_type: Type,
    pub confidence: ConfidenceLevel,
}

/// Individual piece of type evidence
#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    pub kind: EvidenceKind,
    pub resulting_type: Type,
    pub location: Span,
    pub confidence: ConfidenceLevel,
}

/// Different kinds of type evidence
#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceKind {
    /// Assignment: x = value
    Assignment(Type),

    /// Type guard: if isinstance(x, Type)
    TypeGuard(Type),

    /// Method call: x.method() succeeded
    MethodCall {
        method_name: InternedString,
        return_type: Type,
    },

    /// Field access: x.field succeeded
    FieldAccess {
        field_name: InternedString,
        field_type: Type,
    },

    /// Duck typing: x behaves like Type
    DuckTyping(Type),

    /// Runtime check: runtime assertion passed
    RuntimeAssertion(Type),

    /// Pattern match: x matches pattern
    PatternMatch(Pattern),

    /// Null check: x is not null
    NonNull,

    /// Truthiness: x is truthy/falsy
    Truthiness(bool),
}

/// Pattern for pattern matching evidence
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Type(Type),
    Literal(Literal),
    Destructure {
        constructor: InternedString,
        fields: Vec<Pattern>,
    },
    Variable(InternedString),
    Wildcard,
}

/// Confidence level for type evidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConfidenceLevel {
    /// Type is guaranteed by static analysis
    Certain = 100,

    /// Type is very likely based on strong evidence
    VeryHigh = 90,

    /// Type is likely based on good evidence
    High = 75,

    /// Type is somewhat likely
    Medium = 50,

    /// Type is possible but uncertain
    Low = 25,

    /// Type is just a guess
    VeryLow = 10,

    /// No confidence, complete uncertainty
    Unknown = 0,
}

/// Possible target for dynamic dispatch
#[derive(Debug, Clone)]
pub struct PossibleTarget {
    pub target_type: Type,
    pub method_signature: crate::structural_type_checker::MethodSignature,
    pub probability: f64,
    pub call_cost: CallCost,
}

/// Cost model for dynamic calls
#[derive(Debug, Clone)]
pub struct CallCost {
    pub lookup_cost: u32,     // Cost of method lookup
    pub dispatch_cost: u32,   // Cost of dispatch
    pub conversion_cost: u32, // Cost of type conversions
    pub check_cost: u32,      // Cost of runtime checks
}

/// Blame information for error reporting
#[derive(Debug, Clone)]
pub struct BlameInfo {
    pub blamed_component: BlamedComponent,
    pub error_message: String,
    pub suggested_fix: Option<String>,
    pub related_locations: Vec<Span>,
}

/// What to blame when a gradual typing error occurs
#[derive(Debug, Clone, PartialEq)]
pub enum BlamedComponent {
    StaticCode(Span),  // Blame statically typed code
    DynamicCode(Span), // Blame dynamically typed code
    Boundary(Span),    // Blame static/dynamic boundary
    RuntimeSystem,     // Blame runtime type system
    UserCode(Span),    // Blame user code in general
}

/// Precision level for type information
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecisionLevel {
    /// Exact type information
    Exact,

    /// Upper bound (subtype relationship)
    UpperBound,

    /// Lower bound (supertype relationship)
    LowerBound,

    /// Shape information (structural)
    Shape,

    /// Tag information (nominal)
    Tag,

    /// No precision (Any)
    None,
}

/// Gradual typing errors
#[derive(Debug, Clone, PartialEq)]
pub enum GradualTypeError {
    /// Dynamic access on non-dynamic type
    InvalidDynamicAccess {
        accessed_type: Type,
        member_name: InternedString,
        span: Span,
    },

    /// Static/dynamic boundary violation
    BoundaryViolation {
        expected_type: Type,
        actual_type: Type,
        boundary_kind: BoundaryKind,
        span: Span,
    },

    /// Runtime check failed
    RuntimeCheckFailed {
        check: RuntimeCheck,
        actual_value: String, // String representation
        span: Span,
    },

    /// Insufficient type evidence
    InsufficientEvidence {
        variable: VariableId,
        required_confidence: ConfidenceLevel,
        actual_confidence: ConfidenceLevel,
        span: Span,
    },

    /// Contradictory type evidence
    ContradictoryEvidence {
        variable: VariableId,
        evidence1: Evidence,
        evidence2: Evidence,
        span: Span,
    },

    /// Dynamic dispatch failed
    DispatchFailed {
        call_site: CallSiteId,
        receiver_type: Type,
        method_name: InternedString,
        span: Span,
    },

    /// Precision loss in type conversion
    PrecisionLoss {
        from_type: Type,
        to_type: Type,
        lost_precision: PrecisionLevel,
        span: Span,
    },
}

/// Different kinds of static/dynamic boundaries
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryKind {
    StaticToDynamic, // Passing static value to dynamic context
    DynamicToStatic, // Passing dynamic value to static context
    Import,          // Importing from dynamically typed module
    Export,          // Exporting to dynamically typed module
    Callback,        // Dynamic callback from static code
    Reflection,      // Reflection/metaprogramming boundary
}

/// Function signature for method checks
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub type_params: Vec<TypeParam>,
    pub params: Vec<ParamInfo>,
    pub return_type: Type,
    pub async_kind: AsyncKind,
    pub where_clause: Vec<TypeConstraint>,
}

/// Runtime type information for dynamic dispatch
#[derive(Debug, Clone)]
pub struct RuntimeTypeInfo {
    pub type_id: TypeId,
    pub type_name: InternedString,
    pub methods: HashMap<InternedString, MethodInfo>,
    pub fields: HashMap<InternedString, FieldInfo>,
    pub interfaces: Vec<TypeId>,
    pub metadata: TypeMetadata,
}

/// Runtime method information
#[derive(Debug, Clone)]
pub struct MethodInfo {
    pub signature: FunctionSignature,
    pub implementation: MethodImpl,
    pub dispatch_cost: u32,
    pub is_cached: bool,
}

/// Method implementation type
#[derive(Debug, Clone)]
pub enum MethodImpl {
    Native(InternedString), // Native implementation name
    Dynamic(CallSiteId),    // Dynamic dispatch target
    Wrapper(WrapperKind),   // Generated wrapper
    Missing,                // Method not found (for error reporting)
}

/// Wrapper types for gradual typing
#[derive(Debug, Clone)]
pub enum WrapperKind {
    TypeCheck,  // Adds runtime type checks
    Coercion,   // Performs type coercions
    Monitoring, // Adds performance monitoring
    Blame,      // Adds blame tracking
}

/// Runtime field information
#[derive(Debug, Clone)]
pub struct FieldInfo {
    pub field_type: Type,
    pub accessor: FieldAccessor,
    pub is_mutable: bool,
    pub metadata: FieldMetadata,
}

/// Field accessor type
#[derive(Debug, Clone)]
pub enum FieldAccessor {
    Direct(u32),            // Direct offset access
    Getter(InternedString), // Getter method name
    Dynamic,                // Dynamic property access
}

/// Type metadata for runtime
#[derive(Debug, Clone)]
pub struct TypeMetadata {
    pub is_sealed: bool,
    pub is_final: bool,
    pub supports_dynamic: bool,
    pub version: u32,
    pub source_location: Option<Span>,
}

/// Field metadata
#[derive(Debug, Clone)]
pub struct FieldMetadata {
    pub is_lazy: bool,
    pub is_cached: bool,
    pub validator: Option<InternedString>,
}

/// Gradual typing mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradualMode {
    /// Static mode: no dynamic behavior allowed
    Static,
    /// Permissive mode: allow dynamic with warnings
    Permissive,
    /// Dynamic mode: full dynamic typing
    Dynamic,
    /// Transitional mode: migrating from dynamic to static
    Transitional,
}

/// Type migration information
#[derive(Debug, Clone)]
pub struct TypeMigration {
    pub from_type: Type,
    pub to_type: Type,
    pub migration_path: Vec<MigrationStep>,
    pub confidence: ConfidenceLevel,
}

/// Migration step in gradual typing
#[derive(Debug, Clone)]
pub enum MigrationStep {
    AddTypeAnnotation(Span, Type),
    AddRuntimeCheck(CheckPointId, RuntimeCheck),
    RefactorCode(Span, String),
    UpdateInterface(TypeId),
}

impl GradualTypeChecker {
    pub fn new() -> Self {
        Self {
            runtime_checks: HashMap::new(),
            type_evidence: HashMap::new(),
            dynamic_targets: HashMap::new(),
            blame_map: HashMap::new(),
            precision_levels: HashMap::new(),
        }
    }

    /// Check compatibility between static and dynamic types
    pub fn check_gradual_compatibility(
        &mut self,
        static_type: &Type,
        dynamic_value: &Type,
        boundary: BoundaryKind,
        span: Span,
    ) -> Result<Vec<RuntimeCheck>, GradualTypeError> {
        match (static_type, dynamic_value) {
            // Any is compatible with everything
            (_, Type::Any) | (Type::Any, _) => Ok(vec![]),

            // Dynamic is compatible but requires runtime checks
            (static_ty, Type::Dynamic) => self.generate_dynamic_to_static_checks(static_ty, span),

            (Type::Dynamic, _) => {
                // Dynamic can accept anything, no checks needed
                Ok(vec![])
            }

            // Unknown requires careful handling
            (static_ty, Type::Unknown) => self.generate_unknown_checks(static_ty, span),

            // Exact type match
            (a, b) if a == b => Ok(vec![]),

            // Structural compatibility with runtime verification
            _ => self.check_structural_with_runtime(static_type, dynamic_value, span),
        }
    }

    /// Generate runtime checks for dynamic to static conversion
    fn generate_dynamic_to_static_checks(
        &mut self,
        target_type: &Type,
        span: Span,
    ) -> Result<Vec<RuntimeCheck>, GradualTypeError> {
        let mut checks = Vec::new();

        match target_type {
            Type::Primitive(prim) => {
                checks.push(RuntimeCheck {
                    check_kind: RuntimeCheckKind::TypeAssertion,
                    expected_type: target_type.clone(),
                    fallback_behavior: FallbackBehavior::Throw(format!(
                        "Expected {}, got dynamic value",
                        self.format_primitive(*prim)
                    )),
                    optimization_hint: OptimizationHint::EliminateIfProven,
                    span,
                });
            }

            Type::Nullable(inner) => {
                // Check for null first, then check inner type
                checks.push(RuntimeCheck {
                    check_kind: RuntimeCheckKind::NullCheck,
                    expected_type: target_type.clone(),
                    fallback_behavior: FallbackBehavior::ContinueDynamic,
                    optimization_hint: OptimizationHint::LikelySuccess,
                    span,
                });

                checks.extend(self.generate_dynamic_to_static_checks(inner, span)?);
            }

            Type::Interface { methods, .. } => {
                // Generate duck typing check
                let requirements: Vec<MemberRequirement> = methods
                    .iter()
                    .map(|method| MemberRequirement {
                        name: method.name,
                        kind: MemberKind::Method,
                        expected_type: Type::Function {
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
                            return_type: Box::new(method.return_type.clone()),
                            async_kind: if method.is_async {
                                AsyncKind::Async
                            } else {
                                AsyncKind::Sync
                            },
                            is_varargs: false,
                            calling_convention: CallingConvention::Default,
                            nullability: NullabilityKind::NonNull,
                            has_named_params: false,
                            has_default_params: false,
                        },
                        is_optional: false,
                    })
                    .collect();

                checks.push(RuntimeCheck {
                    check_kind: RuntimeCheckKind::DuckTypeCheck {
                        required_members: requirements,
                    },
                    expected_type: target_type.clone(),
                    fallback_behavior: FallbackBehavior::Throw(
                        "Object does not implement required interface".to_string(),
                    ),
                    optimization_hint: OptimizationHint::Cacheable,
                    span,
                });
            }

            Type::Function {
                params,
                return_type,
                ..
            } => {
                // Check that value is callable with correct signature
                checks.push(RuntimeCheck {
                    check_kind: RuntimeCheckKind::MethodCheck {
                        method_name: InternedString::from_symbol(
                            string_interner::Symbol::try_from_usize(0).unwrap(),
                        ), // "__call__" or similar
                        expected_signature: Some(FunctionSignature {
                            type_params: vec![],
                            params: params.clone(),
                            return_type: *return_type.clone(),
                            async_kind: AsyncKind::Sync,
                            where_clause: vec![],
                        }),
                    },
                    expected_type: target_type.clone(),
                    fallback_behavior: FallbackBehavior::Throw(
                        "Value is not callable with expected signature".to_string(),
                    ),
                    optimization_hint: OptimizationHint::PerformanceCritical,
                    span,
                });
            }

            _ => {
                // Generic type assertion
                checks.push(RuntimeCheck {
                    check_kind: RuntimeCheckKind::TypeAssertion,
                    expected_type: target_type.clone(),
                    fallback_behavior: FallbackBehavior::Throw("Type assertion failed".to_string()),
                    optimization_hint: OptimizationHint::EliminateIfProven,
                    span,
                });
            }
        }

        Ok(checks)
    }

    /// Generate checks for unknown type values
    fn generate_unknown_checks(
        &mut self,
        target_type: &Type,
        span: Span,
    ) -> Result<Vec<RuntimeCheck>, GradualTypeError> {
        // Unknown is more restrictive than Any - we need to prove compatibility
        let mut checks = self.generate_dynamic_to_static_checks(target_type, span)?;

        // Add additional safety checks for Unknown
        checks.push(RuntimeCheck {
            check_kind: RuntimeCheckKind::UndefinedCheck,
            expected_type: target_type.clone(),
            fallback_behavior: FallbackBehavior::Throw("Value is undefined".to_string()),
            optimization_hint: OptimizationHint::LikelySuccess,
            span,
        });

        Ok(checks)
    }

    /// Check structural compatibility with runtime verification
    fn check_structural_with_runtime(
        &mut self,
        static_type: &Type,
        dynamic_type: &Type,
        span: Span,
    ) -> Result<Vec<RuntimeCheck>, GradualTypeError> {
        // Use structural typing principles but add runtime checks
        // where static analysis is insufficient

        let mut checks = Vec::new();

        match (static_type, dynamic_type) {
            (Type::Named { id: static_id, .. }, Type::Named { id: dynamic_id, .. }) => {
                if static_id != dynamic_id {
                    // Different nominal types - need runtime type check
                    checks.push(RuntimeCheck {
                        check_kind: RuntimeCheckKind::TypeAssertion,
                        expected_type: static_type.clone(),
                        fallback_behavior: FallbackBehavior::Throw(
                            "Nominal type mismatch".to_string(),
                        ),
                        optimization_hint: OptimizationHint::EliminateIfProven,
                        span,
                    });
                }
            }

            _ => {
                // Default to type assertion
                checks.push(RuntimeCheck {
                    check_kind: RuntimeCheckKind::TypeAssertion,
                    expected_type: static_type.clone(),
                    fallback_behavior: FallbackBehavior::CoerceType,
                    optimization_hint: OptimizationHint::LikelySuccess,
                    span,
                });
            }
        }

        Ok(checks)
    }

    /// Add type evidence for flow-sensitive typing
    pub fn add_evidence(&mut self, variable: VariableId, evidence: Evidence) {
        // Get current type and evidence chain to avoid borrowing conflicts
        let (current_type, current_evidence_chain) = {
            let type_evidence =
                self.type_evidence
                    .entry(variable)
                    .or_insert_with(|| TypeEvidence {
                        variable_id: variable,
                        evidence_chain: Vec::new(),
                        current_type: Type::Unknown,
                        confidence: ConfidenceLevel::Unknown,
                    });
            (
                type_evidence.current_type.clone(),
                type_evidence.evidence_chain.clone(),
            )
        };

        // Calculate new values
        let new_type = self.combine_type_with_evidence(&current_type, &evidence);
        let new_confidence = self.calculate_confidence(&current_evidence_chain, &evidence);

        // Update the evidence
        let type_evidence = self.type_evidence.get_mut(&variable).unwrap();
        type_evidence.evidence_chain.push(evidence);
        type_evidence.current_type = new_type;
        type_evidence.confidence = new_confidence;
    }

    /// Combine existing type information with new evidence
    fn combine_type_with_evidence(&self, current_type: &Type, evidence: &Evidence) -> Type {
        match &evidence.kind {
            EvidenceKind::Assignment(ref new_type) => {
                // Assignment evidence overrides previous type
                new_type.clone()
            }

            EvidenceKind::TypeGuard(ref guarded_type) => {
                // Type guard narrows the type
                self.intersect_types(current_type, guarded_type)
            }

            EvidenceKind::NonNull => {
                // Remove null from union type
                self.remove_null_from_type(current_type)
            }

            EvidenceKind::MethodCall { return_type, .. } => {
                // Method call evidence suggests object has that method
                current_type.clone() // Keep current type, just add evidence
            }

            _ => current_type.clone(),
        }
    }

    /// Calculate confidence level based on evidence chain
    fn calculate_confidence(
        &self,
        evidence_chain: &[Evidence],
        new_evidence: &Evidence,
    ) -> ConfidenceLevel {
        let base_confidence = match new_evidence.kind {
            EvidenceKind::Assignment(_) => ConfidenceLevel::VeryHigh,
            EvidenceKind::TypeGuard(_) => ConfidenceLevel::High,
            EvidenceKind::RuntimeAssertion(_) => ConfidenceLevel::Certain,
            EvidenceKind::MethodCall { .. } => ConfidenceLevel::Medium,
            EvidenceKind::DuckTyping(_) => ConfidenceLevel::Low,
            _ => ConfidenceLevel::Medium,
        };

        // Adjust confidence based on evidence chain length and consistency
        let chain_length = evidence_chain.len();
        let consistency_bonus = if chain_length > 0 {
            let consistent = evidence_chain.iter().all(|e| {
                self.evidence_supports_type(&e.resulting_type, &new_evidence.resulting_type)
            });
            if consistent {
                10
            } else {
                -20
            }
        } else {
            0
        };

        let final_confidence = (base_confidence as i32 + consistency_bonus).max(0).min(100);
        match final_confidence {
            90..=100 => ConfidenceLevel::Certain,
            75..=89 => ConfidenceLevel::VeryHigh,
            60..=74 => ConfidenceLevel::High,
            40..=59 => ConfidenceLevel::Medium,
            20..=39 => ConfidenceLevel::Low,
            10..=19 => ConfidenceLevel::VeryLow,
            _ => ConfidenceLevel::Unknown,
        }
    }

    /// Check if evidence supports a type hypothesis
    fn evidence_supports_type(&self, evidence_type: &Type, hypothesis: &Type) -> bool {
        // Simplified compatibility check
        evidence_type == hypothesis
            || matches!((evidence_type, hypothesis), (Type::Any, _) | (_, Type::Any))
    }

    /// Intersect two types (type narrowing)
    fn intersect_types(&self, type1: &Type, type2: &Type) -> Type {
        match (type1, type2) {
            (Type::Union(types1), _) => {
                // Filter union types
                let filtered: Vec<Type> = types1
                    .iter()
                    .filter(|t| self.is_subtype_of(t, type2))
                    .cloned()
                    .collect();

                if filtered.len() == 1 {
                    filtered[0].clone()
                } else if filtered.is_empty() {
                    Type::Never
                } else {
                    Type::Union(filtered)
                }
            }

            (_, Type::Union(_)) => self.intersect_types(type2, type1),

            _ => {
                if self.is_subtype_of(type2, type1) {
                    type2.clone()
                } else if self.is_subtype_of(type1, type2) {
                    type1.clone()
                } else {
                    Type::Never
                }
            }
        }
    }

    /// Remove null from a type
    fn remove_null_from_type(&self, ty: &Type) -> Type {
        match ty {
            Type::Nullable(inner) => *inner.clone(),
            Type::Union(types) => {
                let non_null_types: Vec<Type> = types
                    .iter()
                    .filter(|t| !matches!(t, Type::Primitive(PrimitiveType::Unit))) // Assuming Unit represents null
                    .cloned()
                    .collect();

                if non_null_types.len() == 1 {
                    non_null_types[0].clone()
                } else {
                    Type::Union(non_null_types)
                }
            }
            _ => ty.clone(),
        }
    }

    /// Simple subtype check
    fn is_subtype_of(&self, sub: &Type, sup: &Type) -> bool {
        sub == sup || matches!((sub, sup), (_, Type::Any) | (Type::Never, _))
    }

    /// Format primitive type for error messages
    fn format_primitive(&self, prim: PrimitiveType) -> String {
        format!("{:?}", prim).to_lowercase()
    }

    /// Get current type of variable with confidence
    pub fn get_variable_type(&self, variable: VariableId) -> Option<(Type, ConfidenceLevel)> {
        self.type_evidence
            .get(&variable)
            .map(|evidence| (evidence.current_type.clone(), evidence.confidence))
    }

    /// Generate blame information for error
    pub fn generate_blame(&mut self, error: &GradualTypeError) -> BlameInfo {
        match error {
            GradualTypeError::BoundaryViolation {
                boundary_kind,
                span,
                ..
            } => match boundary_kind {
                BoundaryKind::StaticToDynamic => BlameInfo {
                    blamed_component: BlamedComponent::StaticCode(*span),
                    error_message: "Static code is too restrictive for dynamic context".to_string(),
                    suggested_fix: Some("Consider using Any or adding runtime checks".to_string()),
                    related_locations: vec![],
                },
                BoundaryKind::DynamicToStatic => BlameInfo {
                    blamed_component: BlamedComponent::DynamicCode(*span),
                    error_message: "Dynamic value doesn't meet static type requirements"
                        .to_string(),
                    suggested_fix: Some("Add type assertions or guards".to_string()),
                    related_locations: vec![],
                },
                _ => BlameInfo {
                    blamed_component: BlamedComponent::Boundary(*span),
                    error_message: "Type mismatch at static/dynamic boundary".to_string(),
                    suggested_fix: Some("Add explicit type conversion".to_string()),
                    related_locations: vec![],
                },
            },

            _ => BlameInfo {
                blamed_component: BlamedComponent::RuntimeSystem,
                error_message: "Gradual typing error".to_string(),
                suggested_fix: None,
                related_locations: vec![],
            },
        }
    }

    // === ENHANCED GRADUAL TYPING METHODS ===

    /// Generate runtime type information for a type
    pub fn generate_runtime_type_info(
        &mut self,
        type_id: TypeId,
        type_def: &Type,
    ) -> RuntimeTypeInfo {
        let type_name = self.get_type_name(type_id);
        let mut methods = HashMap::new();
        let mut fields = HashMap::new();
        let mut interfaces = Vec::new();

        // Extract methods and fields based on type definition
        match type_def {
            Type::Named { .. } => {
                // Would need type registry access for full info
                // This is a simplified version
            }
            Type::Struct {
                fields: field_defs, ..
            } => {
                for field_def in field_defs {
                    fields.insert(
                        field_def.name,
                        FieldInfo {
                            field_type: field_def.ty.clone(),
                            accessor: FieldAccessor::Direct(0), // Would calculate offset
                            is_mutable: field_def.mutability == Mutability::Mutable,
                            metadata: FieldMetadata {
                                is_lazy: false,
                                is_cached: false,
                                validator: None,
                            },
                        },
                    );
                }
            }
            Type::Interface {
                methods: method_defs,
                ..
            } => {
                for method_def in method_defs {
                    methods.insert(
                        method_def.name,
                        MethodInfo {
                            signature: FunctionSignature {
                                type_params: method_def.type_params.clone(),
                                params: method_def
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
                                return_type: method_def.return_type.clone(),
                                async_kind: if method_def.is_async {
                                    AsyncKind::Async
                                } else {
                                    AsyncKind::Sync
                                },
                                where_clause: method_def.where_clause.clone(),
                            },
                            implementation: MethodImpl::Native(method_def.name),
                            dispatch_cost: 1,
                            is_cached: false,
                        },
                    );
                }
            }
            _ => {}
        }

        RuntimeTypeInfo {
            type_id,
            type_name,
            methods,
            fields,
            interfaces,
            metadata: TypeMetadata {
                is_sealed: false,
                is_final: false,
                supports_dynamic: true,
                version: 1,
                source_location: None,
            },
        }
    }

    /// Generate optimized runtime checks based on mode
    pub fn generate_optimized_checks(
        &mut self,
        value_type: &Type,
        expected_type: &Type,
        mode: GradualMode,
        span: Span,
    ) -> Vec<RuntimeCheck> {
        match mode {
            GradualMode::Static => {
                // In static mode, fail fast with detailed checks
                self.generate_strict_checks(value_type, expected_type, span)
            }
            GradualMode::Dynamic => {
                // In dynamic mode, minimal checks
                vec![]
            }
            GradualMode::Permissive => {
                // In permissive mode, check but warn instead of fail
                let mut checks = self
                    .generate_dynamic_to_static_checks(expected_type, span)
                    .unwrap_or_default();
                for check in &mut checks {
                    check.fallback_behavior = FallbackBehavior::Warn(format!(
                        "Type mismatch in permissive mode: expected {:?}",
                        expected_type
                    ));
                }
                checks
            }
            GradualMode::Transitional => {
                // In transitional mode, generate migration-friendly checks
                self.generate_transitional_checks(value_type, expected_type, span)
            }
        }
    }

    /// Generate strict runtime checks for static mode
    fn generate_strict_checks(
        &mut self,
        value_type: &Type,
        expected_type: &Type,
        span: Span,
    ) -> Vec<RuntimeCheck> {
        let mut checks = Vec::new();

        // Add type assertion
        checks.push(RuntimeCheck {
            check_kind: RuntimeCheckKind::TypeAssertion,
            expected_type: expected_type.clone(),
            fallback_behavior: FallbackBehavior::Throw(format!(
                "Strict mode type violation: expected {:?}, got {:?}",
                expected_type, value_type
            )),
            optimization_hint: OptimizationHint::EliminateIfProven,
            span,
        });

        // Add null check if needed
        if !matches!(expected_type, Type::Nullable(_)) {
            checks.push(RuntimeCheck {
                check_kind: RuntimeCheckKind::NullCheck,
                expected_type: expected_type.clone(),
                fallback_behavior: FallbackBehavior::Throw(
                    "Null value in non-nullable context".to_string(),
                ),
                optimization_hint: OptimizationHint::LikelySuccess,
                span,
            });
        }

        checks
    }

    /// Generate transitional checks for migration
    fn generate_transitional_checks(
        &mut self,
        value_type: &Type,
        expected_type: &Type,
        span: Span,
    ) -> Vec<RuntimeCheck> {
        let mut checks = Vec::new();

        // Try to coerce if possible
        checks.push(RuntimeCheck {
            check_kind: RuntimeCheckKind::Cast { is_safe: false },
            expected_type: expected_type.clone(),
            fallback_behavior: FallbackBehavior::CoerceType,
            optimization_hint: OptimizationHint::Cacheable,
            span,
        });

        // Add monitoring to track migrations
        checks.push(RuntimeCheck {
            check_kind: RuntimeCheckKind::TypeAssertion,
            expected_type: expected_type.clone(),
            fallback_behavior: FallbackBehavior::Warn(format!(
                "Migration warning: converting {:?} to {:?}",
                value_type, expected_type
            )),
            optimization_hint: OptimizationHint::DeferToUsage,
            span,
        });

        checks
    }

    /// Analyze dynamic dispatch targets and optimize
    pub fn analyze_dynamic_dispatch(
        &mut self,
        call_site: CallSiteId,
        receiver_type: &Type,
        method_name: InternedString,
    ) -> Vec<PossibleTarget> {
        let mut targets = Vec::new();

        // Analyze receiver type to find possible implementations
        match receiver_type {
            Type::Union(types) => {
                // For union types, each variant is a possible target
                for (i, variant) in types.iter().enumerate() {
                    if let Some(target) = self.find_method_in_type(variant, method_name) {
                        targets.push(PossibleTarget {
                            target_type: variant.clone(),
                            method_signature: target,
                            probability: 1.0 / types.len() as f64,
                            call_cost: CallCost {
                                lookup_cost: 10 + i as u32,
                                dispatch_cost: 5,
                                conversion_cost: 0,
                                check_cost: 5,
                            },
                        });
                    }
                }
            }
            Type::Dynamic => {
                // For dynamic types, we can't predict targets
                // Return a generic dynamic dispatch target
                targets.push(PossibleTarget {
                    target_type: Type::Dynamic,
                    method_signature: crate::structural_type_checker::MethodSignature {
                        name: method_name,
                        type_params: vec![],
                        params: vec![],
                        return_type: Type::Any,
                        is_static: false,
                        is_async: false,
                        variance: vec![],
                    },
                    probability: 1.0,
                    call_cost: CallCost {
                        lookup_cost: 100,
                        dispatch_cost: 50,
                        conversion_cost: 20,
                        check_cost: 30,
                    },
                });
            }
            _ => {
                // For concrete types, single target
                if let Some(target) = self.find_method_in_type(receiver_type, method_name) {
                    targets.push(PossibleTarget {
                        target_type: receiver_type.clone(),
                        method_signature: target,
                        probability: 1.0,
                        call_cost: CallCost {
                            lookup_cost: 1,
                            dispatch_cost: 1,
                            conversion_cost: 0,
                            check_cost: 0,
                        },
                    });
                }
            }
        }

        // Cache the analysis
        self.dynamic_targets.insert(call_site, targets.clone());
        targets
    }

    /// Find method in a type (simplified)
    fn find_method_in_type(
        &self,
        ty: &Type,
        method_name: InternedString,
    ) -> Option<crate::structural_type_checker::MethodSignature> {
        match ty {
            Type::Interface { methods, .. } => {
                methods.iter().find(|m| m.name == method_name).map(|m| {
                    crate::structural_type_checker::MethodSignature {
                        name: m.name,
                        type_params: m.type_params.clone(),
                        params: m
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
                        return_type: m.return_type.clone(),
                        is_static: m.is_static,
                        is_async: m.is_async,
                        variance: vec![],
                    }
                })
            }
            _ => None,
        }
    }

    /// Generate type migration plan
    pub fn generate_migration_plan(
        &mut self,
        from_type: &Type,
        to_type: &Type,
        span: Span,
    ) -> TypeMigration {
        let mut migration_path = Vec::new();

        // Analyze what needs to change
        match (from_type, to_type) {
            (Type::Any, concrete_type) | (Type::Dynamic, concrete_type) => {
                // Migrating from dynamic to static
                migration_path.push(MigrationStep::AddTypeAnnotation(
                    span,
                    concrete_type.clone(),
                ));

                let check_id = CheckPointId::next();
                migration_path.push(MigrationStep::AddRuntimeCheck(
                    check_id,
                    RuntimeCheck {
                        check_kind: RuntimeCheckKind::TypeAssertion,
                        expected_type: concrete_type.clone(),
                        fallback_behavior: FallbackBehavior::Warn("Migration check".to_string()),
                        optimization_hint: OptimizationHint::DeferToUsage,
                        span,
                    },
                ));
            }
            (Type::Nullable(inner), non_nullable) if !matches!(non_nullable, Type::Nullable(_)) => {
                // Migrating from nullable to non-nullable
                migration_path.push(MigrationStep::RefactorCode(
                    span,
                    "Add null checks before use".to_string(),
                ));
                migration_path.push(MigrationStep::AddTypeAnnotation(span, non_nullable.clone()));
            }
            _ => {
                // Generic migration
                migration_path.push(MigrationStep::AddTypeAnnotation(span, to_type.clone()));
            }
        }

        TypeMigration {
            from_type: from_type.clone(),
            to_type: to_type.clone(),
            migration_path,
            confidence: self.calculate_migration_confidence(from_type, to_type),
        }
    }

    /// Calculate confidence for type migration
    fn calculate_migration_confidence(&self, from_type: &Type, to_type: &Type) -> ConfidenceLevel {
        match (from_type, to_type) {
            // Same type - certain
            (a, b) if a == b => ConfidenceLevel::Certain,

            // Any to concrete - medium confidence
            (Type::Any, _) => ConfidenceLevel::Medium,

            // Dynamic to concrete - low confidence
            (Type::Dynamic, _) => ConfidenceLevel::Low,

            // Nullable to non-nullable - high confidence if evidence
            (Type::Nullable(inner), outer) if inner.as_ref() == outer => ConfidenceLevel::High,

            _ => ConfidenceLevel::VeryLow,
        }
    }

    /// Get type name for a type ID (placeholder)
    fn get_type_name(&self, type_id: TypeId) -> InternedString {
        InternedString::from_symbol(
            string_interner::Symbol::try_from_usize(type_id.as_u32() as usize).unwrap(),
        )
    }
}

// Helper implementations for IDs
impl CheckPointId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl VariableId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl CallSiteId {
    pub fn next() -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT_ID: AtomicU32 = AtomicU32::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradual_compatibility_any() {
        let mut checker = GradualTypeChecker::new();

        let static_type = Type::Primitive(PrimitiveType::I32);
        let dynamic_type = Type::Any;

        let checks = checker
            .check_gradual_compatibility(
                &static_type,
                &dynamic_type,
                BoundaryKind::DynamicToStatic,
                Span::new(0, 0),
            )
            .unwrap();

        // Any should be compatible with no checks needed
        assert!(checks.is_empty());
    }

    #[test]
    fn test_dynamic_to_static_checks() {
        let mut checker = GradualTypeChecker::new();

        let static_type = Type::Primitive(PrimitiveType::String);
        let dynamic_type = Type::Dynamic;

        let checks = checker
            .check_gradual_compatibility(
                &static_type,
                &dynamic_type,
                BoundaryKind::DynamicToStatic,
                Span::new(0, 0),
            )
            .unwrap();

        // Should generate runtime type check
        assert!(!checks.is_empty());
        assert!(matches!(
            checks[0].check_kind,
            RuntimeCheckKind::TypeAssertion
        ));
    }

    #[test]
    fn test_type_evidence_flow() {
        let mut checker = GradualTypeChecker::new();
        let var_id = VariableId::next();

        // Add assignment evidence
        checker.add_evidence(
            var_id,
            Evidence {
                kind: EvidenceKind::Assignment(Type::Primitive(PrimitiveType::String)),
                resulting_type: Type::Primitive(PrimitiveType::String),
                location: Span::new(0, 0),
                confidence: ConfidenceLevel::High,
            },
        );

        let (current_type, confidence) = checker.get_variable_type(var_id).unwrap();
        assert_eq!(current_type, Type::Primitive(PrimitiveType::String));
        assert!(confidence >= ConfidenceLevel::High);
    }
}
