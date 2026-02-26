//! Multi-Paradigm Type Checker
//!
//! A unified type checker that uses the optimized TypeRegistry as its foundation
//! and optionally enables advanced features like structural typing, gradual typing,
//! dependent types, linear types, and effect systems when needed.
//!
//! ## Design Philosophy
//!
//! This system treats the existing TypeRegistry and type_inference as the
//! battle-tested, optimized foundation. Advanced features are opt-in extensions
//! that build on top of this solid base.
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Default: Uses optimized TypeRegistry + type_inference
//! let mut checker = TypeChecker::new();
//!
//! // Explicit paradigm selection
//! let mut checker = TypeChecker::with_paradigm(Paradigm::Structural);
//!
//! // Multiple paradigms
//! let mut checker = TypeChecker::with_paradigms(vec![
//!     Paradigm::Nominal,
//!     Paradigm::Gradual,
//! ]);
//! ```

use crate::constraint_solver::ConstraintSolver;
use crate::error::AstError;
use crate::source::Span;
use crate::type_inference::{InferenceContext, InferenceError};
use crate::type_registry::{Type, TypeDefinition, TypeId, TypeRegistry};
use crate::typed_ast::{TypedExpression, TypedFunction, TypedProgram};

// Advanced checkers (lazy-loaded)
use crate::const_evaluator::ConstEvaluator;
use crate::dependent_types::DependentTypeChecker;
use crate::effect_system::EffectSystem;
use crate::gradual_type_checker::GradualTypeChecker;
use crate::linear_types::LinearTypeChecker;
use crate::nominal_type_checker::NominalTypeChecker;
use crate::structural_type_checker::StructuralTypeChecker;
use crate::{AstArena, StructuralMode};

use std::collections::HashMap;

/// Language paradigms supported by the type checker
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Paradigm {
    /// Traditional nominal typing (Java/C#/Rust) - uses optimized TypeRegistry
    Nominal,

    /// Structural typing (Go/TypeScript) - duck typing and shape matching
    Structural {
        /// Allow duck typing compatibility
        duck_typing: bool,
        /// Strict structural matching
        strict: bool,
    },

    /// Gradual typing (Python/JavaScript) - mix of static and dynamic
    Gradual {
        /// How to handle 'any' type propagation
        any_propagation: GradualMode,
        /// Generate runtime type checks
        runtime_checks: bool,
    },

    /// Dependent types (Agda/Idris) - types that depend on values
    Dependent {
        /// Enable const generics
        const_generics: bool,
        /// Enable refinement types
        refinement_types: bool,
    },

    /// Linear types - resource management and memory safety
    Linear {
        /// Enable affine types (at most once)
        affine_types: bool,
        /// Enable borrowing system
        borrowing: bool,
    },

    /// Effect systems - track computational side effects
    Effects {
        /// Enable effect inference
        inference: bool,
        /// Enable effect handlers
        handlers: bool,
    },
}

/// Gradual typing modes
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GradualMode {
    /// Strict gradual typing
    Strict,
    /// Lenient - allow more implicit conversions
    Lenient,
    /// Conservative - minimize dynamic checks
    Conservative,
}

/// Configuration for the type checker
#[derive(Debug, Clone)]
pub struct TypeCheckerConfig {
    /// Primary paradigms to enable
    pub paradigms: Vec<Paradigm>,

    /// Enable automatic paradigm detection
    pub auto_detect: bool,

    /// Performance settings
    pub performance: PerformanceConfig,

    /// Error reporting settings
    pub diagnostics: DiagnosticConfig,
}

/// Performance configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Enable caching for type checking results
    pub enable_caching: bool,

    /// Maximum constraint solving iterations
    pub max_iterations: usize,

    /// Enable parallel type checking where possible
    pub parallel: bool,
}

/// Diagnostic configuration
#[derive(Debug, Clone)]
pub struct DiagnosticConfig {
    /// Provide suggestions for type errors
    pub suggestions: bool,

    /// Show intermediate inference steps
    pub verbose_inference: bool,

    /// Include performance statistics
    pub performance_stats: bool,
}

impl Default for TypeCheckerConfig {
    fn default() -> Self {
        Self {
            paradigms: vec![Paradigm::Nominal], // Default to optimized TypeRegistry
            auto_detect: true,
            performance: PerformanceConfig {
                enable_caching: true,
                max_iterations: 1000,
                parallel: false,
            },
            diagnostics: DiagnosticConfig {
                suggestions: true,
                verbose_inference: false,
                performance_stats: false,
            },
        }
    }
}

/// Unified type checker that builds on TypeRegistry foundation
pub struct TypeChecker {
    /// Battle-tested foundation - always present for optimization
    core_registry: TypeRegistry,
    core_inference: InferenceContext,
    core_solver: ConstraintSolver,

    /// Configuration
    config: TypeCheckerConfig,

    /// Advanced checkers (lazy-loaded based on paradigms)
    nominal_checker: Option<NominalTypeChecker>,
    structural_checker: Option<StructuralTypeChecker>,
    gradual_checker: Option<GradualTypeChecker>,
    linear_checker: Option<LinearTypeChecker>,
    effect_system: Option<EffectSystem>,
    dependent_checker: Option<DependentTypeChecker>,
    const_evaluator: Option<ConstEvaluator>,

    /// Unified caching layer
    type_cache: HashMap<TypeCacheKey, Type>,
    inference_cache: HashMap<InferenceCacheKey, Result<Type, InferenceError>>,
}

/// Cache keys for type checking results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TypeCacheKey {
    expression_hash: u64,
    paradigm: Paradigm,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InferenceCacheKey {
    expression_id: u64,
    context_hash: u64,
}

/// Result type for type checking operations
pub type TypeResult<T> = Result<T, TypeCheckError>;

/// Unified error type for type checking
#[derive(Debug, Clone)]
pub enum TypeCheckError {
    /// Errors from the core type system (TypeRegistry)
    Core(InferenceError),

    /// Errors from structural typing
    Structural(Vec<crate::structural_type_checker::StructuralError>),

    /// Errors from gradual typing
    Gradual(crate::gradual_type_checker::GradualTypeError),

    /// Errors from linear typing
    Linear(crate::linear_types::LinearTypeError),

    /// Errors from effect system
    Effects(crate::effect_system::EffectError),

    /// Errors from dependent types
    Dependent(crate::dependent_types::DependentTypeError),

    /// Errors from nominal type checking
    Nominal(crate::nominal_type_checker::NominalTypeError),

    /// Errors from const evaluation
    ConstEval(crate::const_evaluator::ConstEvalError),

    /// Configuration or integration errors
    System(String),
}

impl From<InferenceError> for TypeCheckError {
    fn from(error: InferenceError) -> Self {
        TypeCheckError::Core(error)
    }
}

impl TypeChecker {
    /// Create a new type checker with default configuration
    /// Uses optimized TypeRegistry as the foundation
    pub fn new() -> Self {
        Self::with_config(TypeCheckerConfig::default())
    }

    /// Create a type checker with specific paradigm
    pub fn with_paradigm(paradigm: Paradigm) -> Self {
        let config = TypeCheckerConfig {
            paradigms: vec![paradigm],
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a type checker with multiple paradigms
    pub fn with_paradigms(paradigms: Vec<Paradigm>) -> Self {
        let config = TypeCheckerConfig {
            paradigms,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a type checker with full configuration
    pub fn with_config(config: TypeCheckerConfig) -> Self {
        Self {
            core_registry: TypeRegistry::new(),
            core_inference: InferenceContext::new(Box::new(TypeRegistry::new())),
            core_solver: ConstraintSolver::new(),
            config,
            nominal_checker: None,
            structural_checker: None,
            gradual_checker: None,
            linear_checker: None,
            effect_system: None,
            dependent_checker: None,
            const_evaluator: None,
            type_cache: HashMap::new(),
            inference_cache: HashMap::new(),
        }
    }

    /// Main type checking entry point
    pub fn check_expression(
        &mut self,
        expr: &crate::typed_ast::TypedNode<TypedExpression>,
    ) -> TypeResult<Type> {
        // Check cache first (if enabled)
        if self.config.performance.enable_caching {
            if let Some(cached_result) = self.check_cache(expr) {
                return cached_result;
            }
        }

        // Determine which paradigms are needed
        let required_paradigms = if self.config.auto_detect {
            self.detect_required_paradigms(expr)
        } else {
            self.config.paradigms.clone()
        };

        // Route to appropriate checker(s)
        let result = self.check_with_paradigms(expr, &required_paradigms);

        // Cache result (if enabled)
        if self.config.performance.enable_caching {
            self.cache_result(expr, &required_paradigms, &result);
        }

        result
    }

    /// Check a complete program
    pub fn check_program(&mut self, program: &TypedProgram) -> TypeResult<()> {
        // Always start with the optimized core for basic validation
        // Note: InferenceContext doesn't have check_program, so we validate declarations
        for decl in &program.declarations {
            self.check_declaration(&decl.node)?;
        }

        // Apply additional paradigm checks as needed
        // Clone paradigms to avoid borrow conflicts
        let paradigms = self.config.paradigms.clone();
        for paradigm in paradigms {
            match paradigm {
                Paradigm::Nominal => {
                    // Already checked by core inference
                    continue;
                }
                Paradigm::Linear { .. } => {
                    let checker = self.get_or_create_linear_checker();
                    checker
                        .check_program(program)
                        .map_err(TypeCheckError::Linear)?;
                }
                Paradigm::Effects { .. } => {
                    let checker = self.get_or_create_effect_system();
                    checker
                        .check_program(program)
                        .map_err(TypeCheckError::Effects)?;
                }
                _ => {
                    // Other paradigms checked per-expression
                    continue;
                }
            }
        }

        Ok(())
    }

    /// Register a type (delegates to optimized TypeRegistry)
    pub fn register_type(&mut self, type_def: TypeDefinition) -> TypeId {
        self.core_registry.register_type(type_def)
    }

    /// Get type by ID (uses optimized TypeRegistry lookup)
    pub fn get_type_by_id(&self, id: TypeId) -> Option<&TypeDefinition> {
        self.core_registry.get_type_by_id(id)
    }

    /// Check if type implements trait (uses optimized cache)
    pub fn type_implements(&self, for_type: &Type, trait_id: TypeId) -> bool {
        self.core_registry.type_implements(for_type, trait_id)
    }

    /// Get type registry for direct access when needed
    pub fn type_registry(&self) -> &TypeRegistry {
        &self.core_registry
    }

    /// Get mutable type registry for registration operations
    pub fn type_registry_mut(&mut self) -> &mut TypeRegistry {
        &mut self.core_registry
    }

    // Private implementation methods

    /// Check a declaration
    fn check_declaration(&mut self, decl: &crate::typed_ast::TypedDeclaration) -> TypeResult<()> {
        use crate::typed_ast::TypedDeclaration;

        match decl {
            TypedDeclaration::Function(func) => {
                // Check function body
                if let Some(ref body) = func.body {
                    for stmt in &body.statements {
                        self.check_statement(stmt)?;
                    }
                }
                Ok(())
            }
            TypedDeclaration::Variable(var) => {
                // Check variable initializer if present
                if let Some(init) = &var.initializer {
                    self.check_expression(init)?;
                }
                Ok(())
            }
            TypedDeclaration::TypeAlias(_)
            | TypedDeclaration::Class(_)
            | TypedDeclaration::Interface(_)
            | TypedDeclaration::Impl(_)
            | TypedDeclaration::Enum(_)
            | TypedDeclaration::Module(_)
            | TypedDeclaration::Import(_)
            | TypedDeclaration::Extern(_)
            | TypedDeclaration::Effect(_)
            | TypedDeclaration::EffectHandler(_) => {
                // Type declarations are handled by the registry
                // Extern declarations are resolved to runtime symbols
                // Effect declarations define algebraic effect types
                Ok(())
            }
        }
    }

    /// Check a statement
    fn check_statement(
        &mut self,
        stmt: &crate::typed_ast::TypedNode<crate::typed_ast::TypedStatement>,
    ) -> TypeResult<()> {
        use crate::typed_ast::{TypedIf, TypedLet, TypedMatch, TypedStatement, TypedWhile};

        match &stmt.node {
            TypedStatement::Expression(expr) => {
                self.check_expression(expr)?;
                Ok(())
            }
            TypedStatement::Let(TypedLet { initializer, .. }) => {
                if let Some(init) = initializer {
                    self.check_expression(init)?;
                }
                Ok(())
            }
            TypedStatement::Return(expr) => {
                if let Some(expr) = expr {
                    self.check_expression(expr)?;
                }
                Ok(())
            }
            TypedStatement::While(TypedWhile {
                condition, body, ..
            }) => {
                self.check_expression(condition)?;
                for stmt in &body.statements {
                    self.check_statement(stmt)?;
                }
                Ok(())
            }
            TypedStatement::If(TypedIf {
                condition,
                then_block,
                else_block,
                ..
            }) => {
                self.check_expression(condition)?;
                for stmt in &then_block.statements {
                    self.check_statement(stmt)?;
                }
                if let Some(else_block) = else_block {
                    for stmt in &else_block.statements {
                        self.check_statement(stmt)?;
                    }
                }
                Ok(())
            }
            TypedStatement::Match(TypedMatch { scrutinee, arms }) => {
                self.check_expression(scrutinee)?;
                for arm in arms {
                    self.check_expression(&arm.body)?;
                }
                Ok(())
            }
            TypedStatement::Block(block) => {
                for stmt in &block.statements {
                    self.check_statement(stmt)?;
                }
                Ok(())
            }
            TypedStatement::Break(_) | TypedStatement::Continue => Ok(()),
            TypedStatement::Throw(expr) => {
                self.check_expression(expr)?;
                Ok(())
            }
            _ => {
                // For other statement types, just return Ok for now
                Ok(())
            }
        }
    }

    /// Check expression with specific paradigms
    fn check_with_paradigms(
        &mut self,
        expr: &crate::typed_ast::TypedNode<TypedExpression>,
        paradigms: &[Paradigm],
    ) -> TypeResult<Type> {
        // Always start with the optimized core inference
        // Since InferenceContext doesn't have infer_expression_type, we just use the expression's type
        let base_type = expr.ty.clone();

        // Apply additional paradigm checks
        let mut current_type = base_type;

        for paradigm in paradigms {
            current_type = match paradigm {
                Paradigm::Nominal => {
                    // Use enhanced nominal type checker for polymorphism and inheritance
                    // For now, just use the core registry functionality
                    // TODO: Add more sophisticated nominal checking when needed

                    current_type
                }
                Paradigm::Structural {
                    duck_typing,
                    strict,
                } => {
                    let universal_type = &current_type;
                    let expr_universal_type = &expr.ty;

                    let checker = self.get_or_create_structural_checker();

                    let compatibility = checker
                        .is_structurally_compatible(
                            &expr_universal_type,
                            &universal_type,
                            if *duck_typing {
                                StructuralMode::Duck
                            } else if *strict {
                                StructuralMode::Strict
                            } else {
                                StructuralMode::Nominal
                            },
                        )
                        .map_err(TypeCheckError::Structural)?;

                    // Check if types are compatible
                    match compatibility {
                        crate::structural_type_checker::StructuralCompatibility::Compatible => {
                            // Return the original Type since we're using Type as our unified API
                            current_type
                        }
                        crate::structural_type_checker::StructuralCompatibility::Incompatible(errors) => {
                            // This should not happen as we already handled errors above
                            return Err(TypeCheckError::Structural(errors));
                        }
                        crate::structural_type_checker::StructuralCompatibility::RequiresAdapterPattern(_) => {
                            // For now, treat adapter requirements as compatible
                            current_type
                        }
                    }
                }
                Paradigm::Gradual { .. } => {
                    let universal_type = &current_type;
                    let expr_universal_type = &expr.ty;

                    let checker = self.get_or_create_gradual_checker();

                    checker
                        .check_gradual_compatibility(
                            &expr_universal_type,
                            &universal_type,
                            crate::gradual_type_checker::BoundaryKind::StaticToDynamic,
                            expr.span,
                        )
                        .map_err(TypeCheckError::Gradual)?;

                    // Return the original Type
                    current_type
                }
                Paradigm::Dependent {
                    const_generics,
                    refinement_types,
                } => {
                    // For dependent types, we need to check if the expression involves
                    // dependent types and perform appropriate checking
                    let checker = self.get_or_create_dependent_checker();

                    // Check if the current type involves dependent types
                    if let Some(dependent_type) = self.extract_dependent_type(&current_type) {
                        // Generate constraints for dependent type checking
                        let span = crate::source::Span::new(0, 0); // TODO: get actual span
                        let constraints = self.core_solver.generate_dependent_type_constraints(
                            &dependent_type,
                            current_type.clone(),
                            span,
                        );

                        // Add constraints to the solver
                        for constraint in constraints {
                            self.core_solver.add_constraint(constraint);
                        }

                        // Check well-formedness if enabled
                        if *refinement_types {
                            let _wellformedness_constraints = self
                                .core_solver
                                .generate_wellformedness_constraints(&dependent_type, span);
                            // TODO: Add wellformedness constraints to solver
                        }
                    }

                    // Handle const generics if enabled
                    if *const_generics {
                        if let Some(const_constraints) =
                            self.extract_const_constraints(&current_type)
                        {
                            for constraint in const_constraints {
                                self.core_solver.add_constraint(constraint);
                            }
                        }
                    }

                    current_type
                }
                _ => {
                    // Other paradigms are program-level, not expression-level
                    current_type
                }
            };
        }

        Ok(current_type)
    }

    /// Detect which paradigms are needed based on expression complexity
    fn detect_required_paradigms(
        &self,
        _expr: &crate::typed_ast::TypedNode<TypedExpression>,
    ) -> Vec<Paradigm> {
        // For now, return configured paradigms
        // TODO: Implement smart detection based on expression features
        self.config.paradigms.clone()
    }

    /// Check if result is cached
    fn check_cache(
        &self,
        _expr: &crate::typed_ast::TypedNode<TypedExpression>,
    ) -> Option<TypeResult<Type>> {
        // TODO: Implement caching based on expression hash
        None
    }

    /// Cache a type checking result
    fn cache_result(
        &mut self,
        _expr: &crate::typed_ast::TypedNode<TypedExpression>,
        _paradigms: &[Paradigm],
        _result: &TypeResult<Type>,
    ) {
        // TODO: Implement result caching
    }

    /// Lazy-load nominal type checker
    fn get_or_create_nominal_checker(&mut self) -> &mut NominalTypeChecker {
        if self.nominal_checker.is_none() {
            self.nominal_checker = Some(NominalTypeChecker::new());
        }
        self.nominal_checker.as_mut().unwrap()
    }

    /// Lazy-load structural type checker
    fn get_or_create_structural_checker(&mut self) -> &mut StructuralTypeChecker {
        if self.structural_checker.is_none() {
            self.structural_checker = Some(StructuralTypeChecker::new());
        }
        self.structural_checker.as_mut().unwrap()
    }

    /// Lazy-load gradual type checker
    fn get_or_create_gradual_checker(&mut self) -> &mut GradualTypeChecker {
        if self.gradual_checker.is_none() {
            self.gradual_checker = Some(GradualTypeChecker::new());
        }
        self.gradual_checker.as_mut().unwrap()
    }

    /// Lazy-load linear type checker
    fn get_or_create_linear_checker(&mut self) -> &mut LinearTypeChecker {
        if self.linear_checker.is_none() {
            self.linear_checker = Some(LinearTypeChecker::new());
        }
        self.linear_checker.as_mut().unwrap()
    }

    /// Lazy-load effect system
    fn get_or_create_effect_system(&mut self) -> &mut EffectSystem {
        if self.effect_system.is_none() {
            self.effect_system = Some(EffectSystem::new());
        }
        self.effect_system.as_mut().unwrap()
    }

    /// Lazy-load dependent type checker
    fn get_or_create_dependent_checker(&mut self) -> &mut DependentTypeChecker {
        if self.dependent_checker.is_none() {
            self.dependent_checker = Some(DependentTypeChecker::new());
        }
        self.dependent_checker.as_mut().unwrap()
    }

    /// Lazy-load const evaluator
    fn get_or_create_const_evaluator(&mut self) -> &mut ConstEvaluator {
        if self.const_evaluator.is_none() {
            self.const_evaluator = Some(ConstEvaluator::new());
        }
        self.const_evaluator.as_mut().unwrap()
    }

    // /// Convert Type to UniversalType for compatibility with checkers that use UniversalType
    // fn type_to_universal_type(&self, ty: &Type) -> crate::universal_type_system::UniversalType {

    //     use crate::type_registry::PrimitiveType;

    //     match ty {
    //         Type::Primitive(prim) => {
    //             let universal_prim = match prim {
    //                 PrimitiveType::I8 => UniversalPrimitive::I8,
    //                 PrimitiveType::I16 => UniversalPrimitive::I16,
    //                 PrimitiveType::I32 => UniversalPrimitive::I32,
    //                 PrimitiveType::I64 => UniversalPrimitive::I64,
    //                 PrimitiveType::I128 => UniversalPrimitive::I128,
    //                 PrimitiveType::U8 => UniversalPrimitive::U8,
    //                 PrimitiveType::U16 => UniversalPrimitive::U16,
    //                 PrimitiveType::U32 => UniversalPrimitive::U32,
    //                 PrimitiveType::U64 => UniversalPrimitive::U64,
    //                 PrimitiveType::U128 => UniversalPrimitive::U128,
    //                 PrimitiveType::F32 => UniversalPrimitive::F32,
    //                 PrimitiveType::F64 => UniversalPrimitive::F64,
    //                 PrimitiveType::Bool => UniversalPrimitive::Bool,
    //                 PrimitiveType::Char => UniversalPrimitive::Char,
    //                 PrimitiveType::String => UniversalPrimitive::String,
    //                 PrimitiveType::ISize => UniversalPrimitive::ISize,
    //                 PrimitiveType::USize => UniversalPrimitive::USize,
    //                 PrimitiveType::Unit => UniversalPrimitive::Unit,
    //             };
    //             UniversalType::Primitive(universal_prim)
    //         }
    //         Type::Any => UniversalType::Any,
    //         Type::Never => UniversalType::Never,
    //         Type::Error => UniversalType::Error,
    //         Type::Dynamic => UniversalType::Dynamic,
    //         Type::Unknown => UniversalType::Unknown,
    //         Type::Function { params, return_type, nullability, calling_convention, async_kind, .. } => {
    //             use crate::{ParamInfo, AsyncKind};

    //             let universal_params: Vec<ParamInfo> = params.iter().map(|p| ParamInfo {
    //                 name: p.name,
    //                 ty: self.type_to_universal_type(&p.ty),
    //                 is_optional: p.is_optional,
    //                 is_varargs: p.is_varargs,
    //                 is_keyword_only: p.is_keyword_only,
    //                 is_positional_only: p.is_positional_only,
    //                 is_out: p.is_out,
    //                 is_ref: p.is_ref,
    //                 is_inout: p.is_inout,
    //             }).collect();

    //             let universal_return = Box::new(self.type_to_universal_type(return_type));

    //             // // Convert AsyncKind from type_registry to universal_type_system
    //             // let universal_async_kind = match async_kind {
    //             //     crate::type_registry::AsyncKind::Sync => AsyncKind::Sync,
    //             //     crate::type_registry::AsyncKind::Async => AsyncKind::Async,
    //             //     crate::type_registry::AsyncKind::Future => AsyncKind::Future(universal_return.clone()),
    //             //     crate::type_registry::AsyncKind::Task => AsyncKind::Task(universal_return.clone()),
    //             //     crate::type_registry::AsyncKind::Promise => AsyncKind::Promise(universal_return.clone()),
    //             //     crate::type_registry::AsyncKind::Coroutine => AsyncKind::Coroutine(universal_return.clone()),
    //             //     crate::type_registry::AsyncKind::Generator => AsyncKind::Generator {
    //             //         yield_type: Box::new(UniversalType::Unknown),
    //             //         return_type: universal_return.clone(),
    //             //     },
    //             // };

    //             UniversalType::Function {
    //                 params: universal_params,
    //                 return_type: universal_return,
    //                 async_kind: universal_async_kind,
    //                 calling_convention: match calling_convention {
    //                     crate::type_registry::CallingConvention::Default => crate::universal_type_system::CallingConvention::Default,
    //                     crate::type_registry::CallingConvention::Cdecl => crate::universal_type_system::CallingConvention::Cdecl,
    //                     crate::type_registry::CallingConvention::Stdcall => crate::universal_type_system::CallingConvention::Stdcall,
    //                     crate::type_registry::CallingConvention::Fastcall => crate::universal_type_system::CallingConvention::Fastcall,
    //                     crate::type_registry::CallingConvention::Vectorcall => crate::universal_type_system::CallingConvention::Vectorcall,
    //                     crate::type_registry::CallingConvention::Thiscall => crate::universal_type_system::CallingConvention::Thiscall,
    //                     crate::type_registry::CallingConvention::Rust => crate::universal_type_system::CallingConvention::Default,
    //                     crate::type_registry::CallingConvention::System => crate::universal_type_system::CallingConvention::Default,
    //                 },
    //                 nullability: match nullability {
    //                     crate::type_registry::NullabilityKind::NonNull => crate::universal_type_system::NullabilityKind::NonNull,
    //                     crate::type_registry::NullabilityKind::Nullable => crate::universal_type_system::NullabilityKind::Nullable,
    //                     crate::type_registry::NullabilityKind::Unknown => crate::universal_type_system::NullabilityKind::Unknown,
    //                     crate::type_registry::NullabilityKind::Platform => crate::universal_type_system::NullabilityKind::Platform,
    //                 },
    //             }
    //         },
    //         Type::Array { element_type, size, nullability, .. } => {
    //             let universal_element = Box::new(self.type_to_universal_type(element_type));
    //             // Convert ConstValue from type_registry to universal_type_system
    //             let universal_size = size.as_ref().map(|s| match s {
    //                 crate::type_registry::ConstValue::Int(i) => crate::universal_type_system::ConstValue::Integer(*i),
    //                 crate::type_registry::ConstValue::UInt(u) => crate::universal_type_system::ConstValue::UInteger(*u),
    //                 crate::type_registry::ConstValue::Bool(b) => crate::universal_type_system::ConstValue::Boolean(*b),
    //                 crate::type_registry::ConstValue::String(s) => crate::universal_type_system::ConstValue::String(*s),
    //                 crate::type_registry::ConstValue::Char(c) => crate::universal_type_system::ConstValue::Char(*c),
    //                 _ => crate::universal_type_system::ConstValue::Unevaluated(crate::universal_type_system::ConstExpr {
    //                     kind: crate::universal_type_system::ConstExprKind::Literal(crate::universal_type_system::Literal::Placeholder),
    //                     span: crate::source::Span::new(0, 0),
    //                 }),
    //             });
    //             UniversalType::Array {
    //                 element_type: universal_element,
    //                 size: universal_size,
    //                 nullability: match nullability {
    //                     crate::type_registry::NullabilityKind::NonNull => crate::universal_type_system::NullabilityKind::NonNull,
    //                     crate::type_registry::NullabilityKind::Nullable => crate::universal_type_system::NullabilityKind::Nullable,
    //                     crate::type_registry::NullabilityKind::Unknown => crate::universal_type_system::NullabilityKind::Unknown,
    //                     crate::type_registry::NullabilityKind::Platform => crate::universal_type_system::NullabilityKind::Platform,
    //                 },
    //             }
    //         },
    //         Type::Tuple(fields) => {
    //             let universal_fields = fields.iter().map(|f| self.type_to_universal_type(f)).collect();
    //             UniversalType::Tuple(universal_fields)
    //         },
    //         Type::Named { id, type_args, const_args, variance, nullability } => {
    //             // Convert named types to nominal types in universal system
    //             UniversalType::Named {
    //                 id: crate::universal_type_system::TypeId::new(id.as_u32()),
    //                 type_args: type_args.iter().map(|t| self.type_to_universal_type(t)).collect(),
    //                 const_args: const_args.iter().map(|c| match c {
    //                     crate::type_registry::ConstValue::Int(i) => crate::universal_type_system::ConstValue::Integer(*i),
    //                     crate::type_registry::ConstValue::UInt(u) => crate::universal_type_system::ConstValue::UInteger(*u),
    //                     crate::type_registry::ConstValue::Bool(b) => crate::universal_type_system::ConstValue::Boolean(*b),
    //                     crate::type_registry::ConstValue::String(s) => crate::universal_type_system::ConstValue::String(*s),
    //                     crate::type_registry::ConstValue::Char(c) => crate::universal_type_system::ConstValue::Char(*c),
    //                     _ => crate::universal_type_system::ConstValue::Unevaluated(crate::universal_type_system::ConstExpr {
    //                         kind: crate::universal_type_system::ConstExprKind::Literal(crate::universal_type_system::Literal::Placeholder),
    //                         span: crate::source::Span::new(0, 0),
    //                     }),
    //                 }).collect(),
    //                 variance: variance.iter().map(|v| match v {
    //                     crate::type_registry::Variance::Covariant => crate::universal_type_system::Variance::Covariant,
    //                     crate::type_registry::Variance::Contravariant => crate::universal_type_system::Variance::Contravariant,
    //                     crate::type_registry::Variance::Invariant => crate::universal_type_system::Variance::Invariant,
    //                     crate::type_registry::Variance::Bivariant => crate::universal_type_system::Variance::Bivariant,
    //                 }).collect(),
    //                 nullability: match nullability {
    //                     crate::type_registry::NullabilityKind::NonNull => crate::universal_type_system::NullabilityKind::NonNull,
    //                     crate::type_registry::NullabilityKind::Nullable => crate::universal_type_system::NullabilityKind::Nullable,
    //                     crate::type_registry::NullabilityKind::Unknown => crate::universal_type_system::NullabilityKind::Unknown,
    //                     crate::type_registry::NullabilityKind::Platform => crate::universal_type_system::NullabilityKind::Platform,
    //                 },
    //             }
    //         },
    //         Type::TypeVar(var_id) => {
    //             // Convert TypeVarId to TypeVar structure for universal system
    //             UniversalType::TypeVar(crate::universal_type_system::TypeVar {
    //                 id: crate::universal_type_system::TypeVarId::new(var_id.id.as_u32()),
    //                 name: None,
    //                 kind: crate::universal_type_system::TypeVarKind::Type,
    //                 bounds: vec![],
    //                 nullability: crate::universal_type_system::NullabilityKind::Unknown,
    //             })
    //         },
    //         Type::Nullable(inner) => {
    //             let universal_inner = Box::new(self.type_to_universal_type(inner));
    //             UniversalType::Nullable(universal_inner)
    //         },
    //         Type::NonNull(inner) => {
    //             // Non-null is just the inner type in universal system
    //             self.type_to_universal_type(inner)
    //         },
    //         _ => {
    //             // For remaining complex types, return Any for now
    //             // Could be extended for specific use cases
    //             UniversalType::Any
    //         }
    //     }
    // }

    /// Extract dependent type information from a regular type
    fn extract_dependent_type(&self, ty: &Type) -> Option<crate::dependent_types::DependentType> {
        use crate::dependent_types::DependentType;
        use crate::type_registry::Type;

        match ty {
            Type::ConstDependent {
                base_type,
                constraint,
            } => {
                // Convert ConstDependent type to refinement type
                let predicate = self.const_constraint_to_predicate(constraint);
                Some(DependentType::Refinement {
                    base_type: base_type.clone(),
                    variable: AstArena::new().intern_string("x"), // Placeholder variable
                    predicate,
                    span: crate::source::Span::new(0, 0),
                })
            }

            // Add more type conversions as needed
            _ => None,
        }
    }

    /// Extract const generic constraints from a type
    fn extract_const_constraints(
        &self,
        ty: &Type,
    ) -> Option<Vec<crate::constraint_solver::Constraint>> {
        use crate::constraint_solver::Constraint;
        use crate::type_registry::Type;

        match ty {
            Type::ConstDependent { constraint, .. } => {
                // Convert const constraint to solver constraint
                // This is simplified - real implementation would be more sophisticated
                Some(vec![])
            }

            Type::Array {
                size: Some(size_const),
                ..
            } => {
                // Array size constraints
                let span = crate::source::Span::new(0, 0);
                Some(vec![Constraint::SingletonEquals {
                    value_type: Type::Primitive(crate::type_registry::PrimitiveType::USize),
                    constant: size_const.clone(),
                    span,
                }])
            }

            _ => None,
        }
    }

    /// Convert const constraint to refinement predicate
    fn const_constraint_to_predicate(
        &self,
        constraint: &crate::type_registry::ConstConstraint,
    ) -> crate::dependent_types::RefinementPredicate {
        use crate::dependent_types::RefinementPredicate;
        use crate::type_registry::ConstConstraint;

        match constraint {
            ConstConstraint::Equal(value) => RefinementPredicate::Comparison {
                op: crate::dependent_types::ComparisonOp::Equal,
                left: Box::new(crate::dependent_types::RefinementExpr::Variable(
                    AstArena::new().intern_string("x"),
                )),
                right: Box::new(crate::dependent_types::RefinementExpr::Constant(
                    value.clone(),
                )),
            },
            ConstConstraint::Range { min, max } => {
                // Create a range predicate: min <= x <= max
                let min_pred = RefinementPredicate::Comparison {
                    op: crate::dependent_types::ComparisonOp::LessEqual,
                    left: Box::new(crate::dependent_types::RefinementExpr::Constant(
                        min.clone(),
                    )),
                    right: Box::new(crate::dependent_types::RefinementExpr::Variable(
                        AstArena::new().intern_string("x"),
                    )),
                };

                let max_pred = RefinementPredicate::Comparison {
                    op: crate::dependent_types::ComparisonOp::LessEqual,
                    left: Box::new(crate::dependent_types::RefinementExpr::Variable(
                        AstArena::new().intern_string("x"),
                    )),
                    right: Box::new(crate::dependent_types::RefinementExpr::Constant(
                        max.clone(),
                    )),
                };

                RefinementPredicate::And(Box::new(min_pred), Box::new(max_pred))
            }
            ConstConstraint::Predicate(_) => {
                // For complex predicates, use a placeholder
                RefinementPredicate::Constant(true)
            }
            ConstConstraint::And(const_constraints) => todo!(),
            ConstConstraint::Or(const_constraints) => todo!(),
        }
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for creating common configurations
impl TypeChecker {
    /// Create a type checker for Rust-like languages
    pub fn for_rust_like() -> Self {
        Self::with_paradigms(vec![
            Paradigm::Nominal,
            Paradigm::Linear {
                affine_types: true,
                borrowing: true,
            },
            Paradigm::Effects {
                inference: true,
                handlers: false,
            },
        ])
    }

    /// Create a type checker for Go-like languages
    pub fn for_go_like() -> Self {
        Self::with_paradigm(Paradigm::Structural {
            duck_typing: true,
            strict: false,
        })
    }

    /// Create a type checker for TypeScript-like languages
    pub fn for_typescript_like() -> Self {
        Self::with_paradigms(vec![
            Paradigm::Structural {
                duck_typing: true,
                strict: false,
            },
            Paradigm::Gradual {
                any_propagation: GradualMode::Lenient,
                runtime_checks: true,
            },
        ])
    }

    /// Create a type checker for Python-like languages
    pub fn for_python_like() -> Self {
        Self::with_paradigm(Paradigm::Gradual {
            any_propagation: GradualMode::Lenient,
            runtime_checks: true,
        })
    }

    /// Create a type checker for Haskell/Agda-like languages
    pub fn for_functional_like() -> Self {
        Self::with_paradigms(vec![
            Paradigm::Nominal,
            Paradigm::Dependent {
                const_generics: true,
                refinement_types: true,
            },
            Paradigm::Effects {
                inference: true,
                handlers: true,
            },
        ])
    }
}
