//! # Type Checker
//!
//! Type checking for the TypedAST, working with our incremental design.
//! Verifies type correctness and mutability constraints.
//!
//! Integrated with diagnostic reporting system for comprehensive error reporting.

use crate::arena::InternedString;
use crate::constraint_solver::{Constraint as SolverConstraint, ConstraintSolver};
use crate::diagnostics::{codes, DiagnosticCollector};
use crate::multi_paradigm_checker::Paradigm;
use crate::source::Span;
use crate::type_inference::{InferenceContext, InferenceError};
use crate::type_registry::{
    CallingConvention, MethodSig, Mutability, ParamInfo, PrimitiveType, Type, TypeBound, Visibility,
};
use crate::{typed_ast::*, AsyncKind};
use std::collections::HashMap;

/// Type checking context
pub struct TypeChecker {
    /// Type inference context
    inference: InferenceContext,
    /// Constraint solver for advanced type inference
    constraint_solver: ConstraintSolver,
    /// Local variable types and mutability
    locals: Vec<HashMap<InternedString, (Type, Mutability)>>,
    /// Return type of current function
    return_type: Option<Type>,
    /// Current loop type (for break expressions)
    loop_type: Option<Type>,
    /// Type checking options
    #[allow(dead_code)]
    options: TypeCheckOptions,
    /// Diagnostic collector for error reporting
    diagnostics: DiagnosticCollector,
    /// Active paradigms for multi-paradigm type checking
    paradigms: Vec<Paradigm>,
    /// Nominal type checker (lazy-loaded when Nominal paradigm active)
    nominal_checker: Option<crate::nominal_type_checker::NominalTypeChecker>,
    /// Structural type checker (lazy-loaded when Structural paradigm active)
    structural_checker: Option<crate::structural_type_checker::StructuralTypeChecker>,
    /// Linear type checker (lazy-loaded when Linear paradigm active)
    linear_checker: Option<crate::linear_types::LinearTypeChecker>,
}

/// Type checking options
#[derive(Debug, Clone)]
pub struct TypeCheckOptions {
    /// Strict null checking
    pub strict_nulls: bool,
    /// Strict function types
    pub strict_functions: bool,
    /// No implicit any
    pub no_implicit_any: bool,
    /// Check unreachable code
    pub check_unreachable: bool,
}

impl Default for TypeCheckOptions {
    fn default() -> Self {
        Self {
            strict_nulls: true,
            strict_functions: true,
            no_implicit_any: true,
            check_unreachable: true,
        }
    }
}

/// Type checking error
#[derive(Debug)]
pub enum TypeError {
    /// Inference error
    InferenceError(InferenceError),
    /// Undefined variable
    UndefinedVariable(InternedString),
    /// Type mismatch
    TypeMismatch { expected: Type, found: Type },
    /// Assignment to immutable variable
    ImmutableVariable(InternedString),
    /// Invalid return type
    InvalidReturn { expected: Type, found: Type },
    /// Break outside loop
    BreakOutsideLoop,
    /// Missing return
    MissingReturn(Type),
    /// Too many arguments provided to function
    TooManyArguments { expected: usize, found: usize },
    /// Too few arguments provided to function
    TooFewArguments { expected: usize, found: usize },
    /// Named arguments not allowed for this function
    NamedArgumentsNotAllowed,
    /// Parameter not found for named argument
    UnknownParameter(InternedString),
    /// Expression is not assignable (for out/ref/inout parameters)
    NotAssignable,
    /// Invalid parameter kind for argument
    InvalidParameterKind { param_name: InternedString },
    /// Trait not implemented for type
    TraitNotImplemented {
        ty: Type,
        trait_name: InternedString,
        span: Span,
    },
    /// Method not found for type
    MethodNotFound {
        receiver_ty: Type,
        method_name: InternedString,
        span: Span,
    },
    /// Suffix function not implemented for abstract type
    MissingSuffixFunction {
        type_name: String,
        suffix: String,
        available_suffixes: Vec<String>,
        span: Span,
    },
    /// Arity mismatch (generic - covers both too many and too few)
    ArityMismatch {
        expected: usize,
        found: usize,
        span: Span,
    },
}

impl From<InferenceError> for TypeError {
    fn from(err: InferenceError) -> Self {
        TypeError::InferenceError(err)
    }
}

impl TypeChecker {
    pub fn new(registry: Box<crate::type_registry::TypeRegistry>) -> Self {
        // Clone the registry so both inference context and constraint solver can use it
        let registry_for_solver = Box::new((*registry).clone());
        Self {
            inference: InferenceContext::new(registry),
            constraint_solver: ConstraintSolver::with_type_registry(registry_for_solver),
            locals: vec![HashMap::new()],
            return_type: None,
            loop_type: None,
            options: TypeCheckOptions::default(),
            diagnostics: DiagnosticCollector::new(),
            paradigms: vec![Paradigm::Nominal],
            nominal_checker: None,
            structural_checker: None,
            linear_checker: None,
        }
    }

    pub fn with_options(
        registry: Box<crate::type_registry::TypeRegistry>,
        options: TypeCheckOptions,
    ) -> Self {
        // Clone the registry so both inference context and constraint solver can use it
        let registry_for_solver = Box::new((*registry).clone());
        Self {
            inference: InferenceContext::new(registry),
            constraint_solver: ConstraintSolver::with_type_registry(registry_for_solver),
            locals: vec![HashMap::new()],
            return_type: None,
            loop_type: None,
            options,
            diagnostics: DiagnosticCollector::new(),
            paradigms: vec![Paradigm::Nominal],
            nominal_checker: None,
            structural_checker: None,
            linear_checker: None,
        }
    }

    /// Set active paradigms for multi-paradigm type checking
    pub fn set_paradigms(&mut self, paradigms: Vec<Paradigm>) {
        self.paradigms = paradigms;
    }

    /// Check if a specific paradigm is active
    pub fn has_paradigm(&self, paradigm: &Paradigm) -> bool {
        self.paradigms.contains(paradigm)
    }

    /// Check if structural typing is active
    fn is_structural_active(&self) -> bool {
        self.paradigms
            .iter()
            .any(|p| matches!(p, Paradigm::Structural { .. }))
    }

    /// Check if linear typing is active
    fn is_linear_active(&self) -> bool {
        self.paradigms
            .iter()
            .any(|p| matches!(p, Paradigm::Linear { .. }))
    }

    /// Get or create the nominal type checker
    fn get_nominal_checker(&mut self) -> &mut crate::nominal_type_checker::NominalTypeChecker {
        if self.nominal_checker.is_none() {
            self.nominal_checker = Some(crate::nominal_type_checker::NominalTypeChecker::new());
        }
        self.nominal_checker.as_mut().unwrap()
    }

    /// Get or create the structural type checker
    fn get_structural_checker(
        &mut self,
    ) -> &mut crate::structural_type_checker::StructuralTypeChecker {
        if self.structural_checker.is_none() {
            self.structural_checker =
                Some(crate::structural_type_checker::StructuralTypeChecker::new());
        }
        self.structural_checker.as_mut().unwrap()
    }

    /// Get or create the linear type checker
    fn get_linear_checker(&mut self) -> &mut crate::linear_types::LinearTypeChecker {
        if self.linear_checker.is_none() {
            self.linear_checker = Some(crate::linear_types::LinearTypeChecker::new());
        }
        self.linear_checker.as_mut().unwrap()
    }

    /// Run a linear type checking pass over the full program
    pub fn check_program_linear(&mut self, program: &TypedProgram) -> Result<(), TypeError> {
        if !self.is_linear_active() {
            return Ok(());
        }
        let checker = self.get_linear_checker();
        checker
            .check_program(program)
            .map_err(|e| TypeError::TypeMismatch {
                expected: Type::Primitive(PrimitiveType::Unit),
                found: Type::Error,
            })
    }

    /// Check structural compatibility between two types (when structural paradigm is active)
    pub fn check_structural_compatibility(
        &mut self,
        sub_type: &Type,
        super_type: &Type,
    ) -> Result<bool, TypeError> {
        if !self.is_structural_active() {
            return Ok(sub_type == super_type);
        }

        let mode = self
            .paradigms
            .iter()
            .find_map(|p| match p {
                Paradigm::Structural {
                    duck_typing,
                    strict,
                } => Some(if *duck_typing {
                    crate::StructuralMode::Duck
                } else if *strict {
                    crate::StructuralMode::Strict
                } else {
                    crate::StructuralMode::Nominal
                }),
                _ => None,
            })
            .unwrap_or(crate::StructuralMode::Nominal);

        let checker = self.get_structural_checker();
        match checker.is_structurally_compatible(sub_type, super_type, mode) {
            Ok(crate::structural_type_checker::StructuralCompatibility::Compatible) => Ok(true),
            Ok(
                crate::structural_type_checker::StructuralCompatibility::RequiresAdapterPattern(_),
            ) => Ok(true),
            Ok(crate::structural_type_checker::StructuralCompatibility::Incompatible(_)) => {
                Ok(false)
            }
            Err(_errors) => Ok(false),
        }
    }

    /// Get reference to the diagnostic collector
    pub fn diagnostics(&self) -> &DiagnosticCollector {
        &self.diagnostics
    }

    /// Take ownership of the diagnostic collector
    pub fn take_diagnostics(self) -> DiagnosticCollector {
        self.diagnostics
    }

    /// Check if there were any errors during type checking
    pub fn has_errors(&self) -> bool {
        self.diagnostics.has_errors()
    }

    /// Helper to emit diagnostic for TypeError
    fn emit_type_error(&mut self, error: TypeError, span: Span) {
        match error {
            TypeError::InferenceError(inf_err) => {
                self.emit_inference_error(inf_err, span);
            }
            TypeError::UndefinedVariable(_name) => {
                self.diagnostics
                    .error("cannot find value in this scope")
                    .code(codes::E0002)
                    .primary(span, "not found in this scope")
                    .help("consider importing this value or checking the spelling")
                    .emit()
                    .ok();
            }
            TypeError::TypeMismatch { expected, found } => {
                let expected_str = self.format_type(&expected);
                let found_str = self.format_type(&found);
                self.diagnostics
                    .error("mismatched types")
                    .code(codes::E0001)
                    .primary(
                        span,
                        format!("expected `{}`, found `{}`", expected_str, found_str),
                    )
                    .emit()
                    .ok();
            }
            TypeError::ImmutableVariable(_name) => {
                self.diagnostics
                    .error("cannot assign to immutable variable")
                    .primary(span, "cannot assign twice to immutable variable")
                    .help("consider making this binding mutable")
                    .emit()
                    .ok();
            }
            TypeError::InvalidReturn { expected, found } => {
                let expected_str = self.format_type(&expected);
                let found_str = self.format_type(&found);
                self.diagnostics
                    .error("mismatched return type")
                    .primary(
                        span,
                        format!("expected `{}`, found `{}`", expected_str, found_str),
                    )
                    .emit()
                    .ok();
            }
            TypeError::BreakOutsideLoop => {
                self.diagnostics
                    .error("`break` outside of loop")
                    .primary(span, "cannot break outside of a loop")
                    .emit()
                    .ok();
            }
            TypeError::MissingReturn(expected_type) => {
                let expected_str = self.format_type(&expected_type);
                self.diagnostics
                    .error("missing return statement")
                    .code(codes::E0301)
                    .primary(
                        span,
                        format!("expected `{}` because of return type", expected_str),
                    )
                    .help("add a return statement")
                    .emit()
                    .ok();
            }
            TypeError::TooManyArguments { expected, found } => {
                self.diagnostics
                    .error("this function takes too many arguments")
                    .code(codes::E0003)
                    .primary(
                        span,
                        format!("expected {} arguments, found {}", expected, found),
                    )
                    .emit()
                    .ok();
            }
            TypeError::TooFewArguments { expected, found } => {
                self.diagnostics
                    .error("this function takes too few arguments")
                    .code(codes::E0004)
                    .primary(
                        span,
                        format!("expected {} arguments, found {}", expected, found),
                    )
                    .emit()
                    .ok();
            }
            TypeError::NamedArgumentsNotAllowed => {
                self.diagnostics
                    .error("named arguments are not allowed for this function")
                    .code(codes::E0007)
                    .primary(span, "named arguments not supported")
                    .emit()
                    .ok();
            }
            TypeError::UnknownParameter(_name) => {
                self.diagnostics
                    .error("no parameter with this name")
                    .code(codes::E0006)
                    .primary(span, "unknown parameter name")
                    .emit()
                    .ok();
            }
            TypeError::NotAssignable => {
                self.diagnostics
                    .error("cannot assign to this expression")
                    .code(codes::E0005)
                    .primary(span, "not assignable")
                    .help("only variables, fields, and indexed expressions can be assigned to")
                    .emit()
                    .ok();
            }
            TypeError::InvalidParameterKind {
                param_name: _param_name,
            } => {
                self.diagnostics
                    .error("invalid use of parameter")
                    .primary(span, "invalid parameter usage")
                    .emit()
                    .ok();
            }
            TypeError::TraitNotImplemented {
                ty,
                trait_name,
                span: error_span,
            } => {
                let ty_str = self.format_type(&ty);
                // For now, use debug formatting for trait name - ideally would use arena
                let trait_str = format!("{:?}", trait_name);
                self.diagnostics
                    .error("trait not implemented")
                    .code(codes::E0277)
                    .primary(
                        error_span.clone(),
                        format!(
                            "the trait `{}` is not implemented for `{}`",
                            trait_str, ty_str
                        ),
                    )
                    .help("consider implementing the trait for this type")
                    .emit()
                    .ok();
            }
            TypeError::MethodNotFound {
                receiver_ty,
                method_name,
                span: error_span,
            } => {
                let ty_str = self.format_type(&receiver_ty);
                // For now, use debug formatting for method name - ideally would use arena
                let method_str = format!("{:?}", method_name);
                self.diagnostics
                    .error("method not found")
                    .primary(
                        error_span.clone(),
                        format!(
                            "no method named `{}` found for type `{}`",
                            method_str, ty_str
                        ),
                    )
                    .help("check the method name or import the necessary trait")
                    .emit()
                    .ok();
            }
            TypeError::ArityMismatch {
                expected,
                found,
                span: error_span,
            } => {
                self.diagnostics
                    .error("argument count mismatch")
                    .primary(
                        error_span.clone(),
                        format!("expected {} arguments, found {}", expected, found),
                    )
                    .emit()
                    .ok();
            }
            TypeError::MissingSuffixFunction {
                type_name,
                suffix,
                available_suffixes,
                span: error_span,
            } => {
                let suffix_list = available_suffixes.join(", ");
                self.diagnostics
                    .error(format!("suffix '{}' not registered for abstract type '{}'", suffix, type_name))
                    .code(codes::E0002)
                    .primary(
                        error_span.clone(),
                        format!("unknown suffix '{}' for abstract type", suffix),
                    )
                    .help(format!(
                        "Available suffixes for '{}': [{}]. Use one of these or add this suffix to the abstract type declaration.",
                        type_name, suffix_list
                    ))
                    .emit()
                    .ok();
            }
        }
    }

    /// Helper to emit diagnostic for SolverError  
    fn emit_solver_error(&mut self, error: crate::constraint_solver::SolverError, span: Span) {
        match error {
            crate::constraint_solver::SolverError::CannotUnify(t1, t2, _error_span) => {
                let t1_str = self.format_type(&t1);
                let t2_str = self.format_type(&t2);
                self.diagnostics
                    .error("constraint solver: cannot unify types")
                    .code(codes::E0001)
                    .primary(span, format!("cannot unify `{}` with `{}`", t1_str, t2_str))
                    .help("check that all type constraints are consistent")
                    .emit()
                    .ok();
            }
            crate::constraint_solver::SolverError::InfiniteType(type_id, ty, _error_span) => {
                let ty_str = self.format_type(&ty);
                self.diagnostics
                    .error("constraint solver: infinite type detected")
                    .primary(
                        span,
                        format!("type variable {:?} occurs in `{}`", type_id, ty_str),
                    )
                    .help("this would create an infinitely recursive type")
                    .emit()
                    .ok();
            }
            crate::constraint_solver::SolverError::UnsolvableConstraint(constraint) => {
                self.diagnostics
                    .error("constraint solver: unsolvable constraint")
                    .primary(
                        span,
                        format!("constraint `{:?}` cannot be satisfied", constraint),
                    )
                    .emit()
                    .ok();
            }
            crate::constraint_solver::SolverError::TraitNotImplemented {
                ty,
                trait_name,
                span: error_span,
            } => {
                let ty_str = self.format_type(&ty);
                // For now, use debug formatting for trait name - ideally would use arena
                let trait_str = format!("{:?}", trait_name);
                self.diagnostics
                    .error("trait not implemented")
                    .code(codes::E0277)
                    .primary(
                        error_span,
                        format!(
                            "the trait `{}` is not implemented for `{}`",
                            trait_str, ty_str
                        ),
                    )
                    .help("consider implementing the trait for this type")
                    .emit()
                    .ok();
            }
            crate::constraint_solver::SolverError::UnknownTrait {
                name,
                span: error_span,
            } => {
                // For now, use debug formatting for trait name - ideally would use arena
                let trait_str = format!("{:?}", name);
                self.diagnostics
                    .error("unknown trait")
                    .code(codes::E0405)
                    .primary(
                        error_span,
                        format!("cannot find trait `{}` in this scope", trait_str),
                    )
                    .help("make sure the trait is imported or defined")
                    .emit()
                    .ok();
            }
            crate::constraint_solver::SolverError::LifetimeCycle {
                lifetime1,
                lifetime2,
                span: error_span,
            } => {
                let lt1_str = format!("{:?}", lifetime1);
                let lt2_str = format!("{:?}", lifetime2);
                self.diagnostics
                    .error("lifetime cycle detected")
                    .code(codes::E0309)
                    .primary(
                        error_span,
                        format!("lifetimes `{}` and `{}` form a cycle", lt1_str, lt2_str),
                    )
                    .help("lifetimes cannot outlive each other in a cycle")
                    .emit()
                    .ok();
            }
        }
    }

    /// Helper to emit diagnostic for InferenceError
    fn emit_inference_error(&mut self, error: InferenceError, span: Span) {
        match error {
            InferenceError::TypeMismatch { expected, found } => {
                // Never is the bottom type — suppress mismatches involving Never
                // (e.g., from return statements in if/match branches)
                if matches!(&expected, Type::Never) || matches!(&found, Type::Never) {
                    return;
                }
                let expected_str = self.format_type(&expected);
                let found_str = self.format_type(&found);
                self.diagnostics
                    .error("type mismatch")
                    .code(codes::E0001)
                    .primary(
                        span,
                        format!("expected `{}`, found `{}`", expected_str, found_str),
                    )
                    .emit()
                    .ok();
            }
            InferenceError::UnresolvedTypeVar(_type_id) => {
                self.diagnostics
                    .error("cannot infer type")
                    .primary(span, "type variable could not be resolved")
                    .help("try adding type annotations")
                    .emit()
                    .ok();
            }
            InferenceError::InfiniteType(_type_id, ty) => {
                let ty_str = self.format_type(&ty);
                self.diagnostics
                    .error("recursive type")
                    .primary(span, format!("recursive type occurs in `{}`", ty_str))
                    .emit()
                    .ok();
            }
            InferenceError::MissingTrait {
                ty,
                trait_name: _trait_name,
            } => {
                self.diagnostics
                    .error(format!(
                        "trait is not implemented for `{}`",
                        self.format_type(&ty)
                    ))
                    .primary(span, "trait not implemented")
                    .emit()
                    .ok();
            }
            InferenceError::UnknownField { ty, field: _field } => {
                self.diagnostics
                    .error(format!("no field on type `{}`", self.format_type(&ty)))
                    .code(codes::E0008)
                    .primary(span, "unknown field")
                    .emit()
                    .ok();
            }
            InferenceError::UnknownMethod {
                ty,
                method: _method,
            } => {
                self.diagnostics
                    .error(format!("no method found for `{}`", self.format_type(&ty)))
                    .primary(span, "method not found")
                    .emit()
                    .ok();
            }
            InferenceError::AmbiguousType(types) => {
                let type_list = types
                    .iter()
                    .map(|t| self.format_type(t))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.diagnostics
                    .error("ambiguous type")
                    .primary(span, format!("could be any of: {}", type_list))
                    .help("try adding type annotations to disambiguate")
                    .emit()
                    .ok();
            }
            InferenceError::UnsolvableConstraint(_) => {
                self.diagnostics
                    .error("constraint solving failed")
                    .primary(span, "could not satisfy type constraints")
                    .emit()
                    .ok();
            }
        }
    }

    /// Format a type for display in diagnostics
    fn format_type(&self, ty: &Type) -> String {
        // Simplified type formatting - can be improved
        format!("{:?}", ty)
    }

    /// Push a new scope
    fn push_scope(&mut self) {
        self.locals.push(HashMap::new());
    }

    /// Pop a scope
    fn pop_scope(&mut self) {
        self.locals.pop();
    }

    /// Add a local variable with mutability
    fn add_local(&mut self, name: InternedString, ty: Type, mutability: Mutability) {
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, (ty, mutability));
        }
    }

    /// Look up a local variable
    fn lookup_local(&self, name: InternedString) -> Option<&(Type, Mutability)> {
        for scope in self.locals.iter().rev() {
            if let Some(info) = scope.get(&name) {
                return Some(info);
            }
        }
        None
    }

    /// Add a constraint to the advanced constraint solver
    fn add_constraint(&mut self, constraint: SolverConstraint) {
        self.constraint_solver.add_constraint(constraint);
    }

    /// Generate a fresh type variable using the constraint solver
    fn fresh_type_var(&mut self) -> Type {
        self.constraint_solver.fresh_type_var()
    }

    /// Type check a program
    /// Returns () on success, diagnostics are collected in self.diagnostics
    pub fn check_program(&mut self, program: &TypedProgram) {
        // First pass: Add all function declarations to scope
        for decl_node in &program.declarations {
            if let TypedDeclaration::Function(func) = &decl_node.node {
                // Create function type from the function signature
                let func_type = Type::Function {
                    params: func
                        .params
                        .iter()
                        .map(|param| crate::type_registry::ParamInfo {
                            name: Some(param.name),
                            ty: param.ty.clone(),
                            is_optional: matches!(
                                param.kind,
                                crate::typed_ast::ParameterKind::Optional
                            ),
                            is_varargs: matches!(param.kind, crate::typed_ast::ParameterKind::Rest),
                            is_keyword_only: false,
                            is_positional_only: false,
                            is_out: matches!(param.kind, crate::typed_ast::ParameterKind::Out),
                            is_ref: matches!(param.kind, crate::typed_ast::ParameterKind::Ref),
                            is_inout: matches!(param.kind, crate::typed_ast::ParameterKind::InOut),
                        })
                        .collect(),
                    return_type: Box::new(func.return_type.clone()),
                    is_varargs: func
                        .params
                        .iter()
                        .any(|p| matches!(p.kind, crate::typed_ast::ParameterKind::Rest)),
                    has_named_params: !func.params.is_empty(),
                    has_default_params: func
                        .params
                        .iter()
                        .any(|p| matches!(p.kind, crate::typed_ast::ParameterKind::Optional)),
                    async_kind: if !func.is_async {
                        crate::AsyncKind::Sync
                    } else {
                        AsyncKind::Async
                    },
                    calling_convention: crate::CallingConvention::Default,
                    nullability: crate::NullabilityKind::Unknown,
                };

                // Add function to current scope
                self.add_local(func.name, func_type, Mutability::Immutable);
            }
        }

        // Second pass: Type check all declarations
        for decl_node in &program.declarations {
            self.check_declaration(&decl_node.node, decl_node.span);
        }

        // Solve all constraints - first try the legacy system, then the advanced solver
        if let Err(inference_err) = self.inference.solve_constraints() {
            self.emit_inference_error(inference_err, program.span);
        }

        // Also solve constraints with the advanced constraint solver
        if let Err(solver_errors) = self.constraint_solver.solve() {
            for error in solver_errors {
                self.emit_solver_error(error, program.span);
            }
        }
    }

    /// Apply resolved types from inference to the program AST
    ///
    /// This method walks the TypedProgram and replaces Type::Any instances
    /// with their resolved types from the inference context.
    /// Call this after check_program() to propagate inferred types.
    pub fn apply_inferred_types(&self, program: &mut TypedProgram) {
        for decl in &mut program.declarations {
            self.apply_types_to_declaration(&mut decl.node);
        }
    }

    fn apply_types_to_declaration(&self, decl: &mut TypedDeclaration) {
        match decl {
            TypedDeclaration::Function(func) => self.apply_types_to_function(func),
            TypedDeclaration::Variable(var) => {
                var.ty = self.inference.apply_substitutions(&var.ty);
                if let Some(init) = &mut var.initializer {
                    self.apply_types_to_expr_node(init);
                }
            }
            // TODO: Handle other declaration types as needed
            _ => {}
        }
    }

    fn apply_types_to_function(&self, func: &mut crate::typed_ast::TypedFunction) {
        // Apply to return type
        func.return_type = self.inference.apply_substitutions(&func.return_type);

        // Apply to parameters
        for param in &mut func.params {
            param.ty = self.inference.apply_substitutions(&param.ty);
        }

        // Apply to function body
        if let Some(body) = &mut func.body {
            for stmt in &mut body.statements {
                self.apply_types_to_statement(stmt);
            }
        }
    }

    fn apply_types_to_statement(
        &self,
        stmt_node: &mut crate::typed_ast::TypedNode<crate::typed_ast::TypedStatement>,
    ) {
        use crate::typed_ast::TypedStatement;

        // Apply to the node's type
        stmt_node.ty = self.inference.apply_substitutions(&stmt_node.ty);

        match &mut stmt_node.node {
            TypedStatement::Expression(expr) => {
                self.apply_types_to_expr_node(expr);
            }
            TypedStatement::Let(let_stmt) => {
                let_stmt.ty = self.inference.apply_substitutions(&let_stmt.ty);
                if let Some(init) = &mut let_stmt.initializer {
                    self.apply_types_to_expr_node(init);
                }
            }
            TypedStatement::Return(opt_value) => {
                if let Some(val) = opt_value {
                    self.apply_types_to_expr_node(val);
                }
            }
            TypedStatement::Block(block) => {
                for stmt in &mut block.statements {
                    self.apply_types_to_statement(stmt);
                }
            }
            TypedStatement::If(if_stmt) => {
                self.apply_types_to_expr_node(&mut if_stmt.condition);
                for stmt in &mut if_stmt.then_block.statements {
                    self.apply_types_to_statement(stmt);
                }
                if let Some(else_block) = &mut if_stmt.else_block {
                    for stmt in &mut else_block.statements {
                        self.apply_types_to_statement(stmt);
                    }
                }
            }
            TypedStatement::While(while_stmt) => {
                self.apply_types_to_expr_node(&mut while_stmt.condition);
                for stmt in &mut while_stmt.body.statements {
                    self.apply_types_to_statement(stmt);
                }
            }
            // TODO: Handle other statement types
            _ => {}
        }
    }

    fn apply_types_to_expr_node(
        &self,
        expr_node: &mut crate::typed_ast::TypedNode<crate::typed_ast::TypedExpression>,
    ) {
        use crate::typed_ast::TypedExpression;

        // Apply to the node's type
        expr_node.ty = self.inference.apply_substitutions(&expr_node.ty);

        match &mut expr_node.node {
            TypedExpression::Call(call) => {
                // Apply to callee
                self.apply_types_to_expr_node(&mut call.callee);
                // Apply to arguments
                for arg in &mut call.positional_args {
                    self.apply_types_to_expr_node(arg);
                }
                for arg in &mut call.named_args {
                    self.apply_types_to_expr_node(&mut arg.value);
                }
            }
            TypedExpression::Binary(binary) => {
                self.apply_types_to_expr_node(&mut binary.left);
                self.apply_types_to_expr_node(&mut binary.right);
            }
            TypedExpression::Unary(unary) => {
                self.apply_types_to_expr_node(&mut unary.operand);
            }
            TypedExpression::Variable(_) => {
                // Variable name only, type is in the node
            }
            TypedExpression::Literal(_) => {
                // Literal value only, type is in the node
            }
            TypedExpression::Field(field) => {
                self.apply_types_to_expr_node(&mut field.object);
            }
            TypedExpression::Index(index) => {
                self.apply_types_to_expr_node(&mut index.object);
                self.apply_types_to_expr_node(&mut index.index);
            }
            TypedExpression::Array(elements) => {
                for elem in elements {
                    self.apply_types_to_expr_node(elem);
                }
            }
            TypedExpression::Tuple(elements) => {
                for elem in elements {
                    self.apply_types_to_expr_node(elem);
                }
            }
            TypedExpression::MethodCall(method_call) => {
                self.apply_types_to_expr_node(&mut method_call.receiver);
                for arg in &mut method_call.positional_args {
                    self.apply_types_to_expr_node(arg);
                }
            }
            // TODO: Handle other expression types
            _ => {}
        }
    }

    /// Type check a declaration
    pub fn check_declaration(&mut self, decl: &TypedDeclaration, span: Span) {
        match decl {
            TypedDeclaration::Function(func) => {
                if let Err(err) = self.check_function(func) {
                    self.emit_type_error(err, span);
                }
            }
            TypedDeclaration::Variable(var) => {
                if let Err(err) = self.check_variable_decl(var) {
                    self.emit_type_error(err, span);
                }
            }
            TypedDeclaration::Impl(impl_block) => {
                // Type check impl block
                if let Err(err) = self.check_impl_block(impl_block) {
                    self.emit_type_error(err, span);
                }
            }
            // Import and Class declarations are handled elsewhere
            TypedDeclaration::Import(_) => {
                // Import declarations are processed at the module level
            }
            TypedDeclaration::Class(class) => {
                // Class declarations (structs/abstract types) are registered in the TypeRegistry
                // Validate abstract types with suffixes
                self.check_abstract_type_suffixes(class.name, span);
            }
            TypedDeclaration::TypeAlias(alias) => {
                // Type aliases might be abstract types
                self.check_abstract_type_suffixes(alias.name, span);
            }
            // TODO: Implement other declaration types
            TypedDeclaration::Interface(_)
            | TypedDeclaration::Enum(_)
            | TypedDeclaration::Module(_) => {
                // Placeholder for now - emit warning for unimplemented features
                self.diagnostics
                    .warning("declaration type not yet fully implemented")
                    .primary(span, "this declaration type is not yet supported")
                    .note("this is a known limitation and will be implemented in the future")
                    .emit()
                    .ok();
            }
            TypedDeclaration::Extern(_) => {
                // Extern declarations are validated at the grammar level,
                // their methods are resolved to runtime symbols
            }
            TypedDeclaration::Effect(_effect) => {
                // Effect declarations define algebraic effect types
                // Type checking for effect operations will be added later
            }
            TypedDeclaration::EffectHandler(_handler) => {
                // Effect handler declarations provide implementations for effects
                // Type checking for handler implementations will be added later
            }
        }
    }

    /// Validate that abstract types with Suffixes only use numeric underlying types
    fn check_abstract_type_suffixes(&mut self, type_name: InternedString, span: Span) {
        let registry = self.constraint_solver.type_registry();

        // Find the type definition by name
        if let Some(type_def) = registry.get_type_by_name(type_name) {
            // Check if this is an abstract type with suffixes
            if let crate::type_registry::TypeKind::Abstract {
                underlying_type,
                suffixes,
                ..
            } = &type_def.kind
            {
                if !suffixes.is_empty() {
                    // Check if underlying type is numeric
                    let is_numeric = matches!(
                        underlying_type,
                        Type::Primitive(PrimitiveType::I8)
                            | Type::Primitive(PrimitiveType::I16)
                            | Type::Primitive(PrimitiveType::I32)
                            | Type::Primitive(PrimitiveType::I64)
                            | Type::Primitive(PrimitiveType::U8)
                            | Type::Primitive(PrimitiveType::U16)
                            | Type::Primitive(PrimitiveType::U32)
                            | Type::Primitive(PrimitiveType::U64)
                            | Type::Primitive(PrimitiveType::F32)
                            | Type::Primitive(PrimitiveType::F64)
                    );

                    if !is_numeric {
                        let type_name_str = type_name
                            .resolve_global()
                            .unwrap_or_else(|| "Unknown".to_string());
                        let underlying_type_str = match underlying_type {
                            Type::Primitive(prim) => format!("{:?}", prim),
                            Type::Named { id, .. } => {
                                if let Some(ut_def) = registry.get_type_by_id(*id) {
                                    ut_def
                                        .name
                                        .resolve_global()
                                        .unwrap_or_else(|| "Unknown".to_string())
                                } else {
                                    "Unknown".to_string()
                                }
                            }
                            _ => "non-numeric type".to_string(),
                        };

                        self.diagnostics
                            .error(format!(
                                "Suffixes can only be used with numeric underlying types, but '{}' has underlying type '{}'",
                                type_name_str, underlying_type_str
                            ))
                            .code(codes::E0002)
                            .primary(span, "invalid use of Suffixes on non-numeric abstract type")
                            .help("Suffixes are restricted to abstract types that wrap numeric primitives (i8, i16, i32, i64, u8, u16, u32, u64, f32, f64)")
                            .note("Example: abstract Duration(i64) with Suffixes(\"ms, s\") for unit literals like 1000ms")
                            .emit()
                            .ok();
                    }
                }
            }
        }
    }

    /// Type check an impl block
    fn check_impl_block(&mut self, impl_block: &TypedTraitImpl) -> Result<(), TypeError> {
        // Type check each method in the impl block
        for method in &impl_block.methods {
            // FIRST: Add constraints for self parameters so inference can propagate through body
            for param in &method.params {
                if param.is_self
                    && (matches!(param.ty, Type::Any) || matches!(param.ty, Type::Unresolved(_)))
                {
                    // Unify the parameter's Any/Unresolved type with the implementing type
                    // This allows the inference to propagate to all usages in the body
                    self.inference
                        .unify(param.ty.clone(), impl_block.for_type.clone())?;
                }
            }

            // Convert method params to function params with resolved types
            let params: Vec<TypedParameter> = method
                .params
                .iter()
                .map(|p| {
                    let resolved_ty = if p.is_self
                        && (matches!(p.ty, Type::Any) || matches!(p.ty, Type::Unresolved(_)))
                    {
                        impl_block.for_type.clone()
                    } else {
                        p.ty.clone()
                    };

                    TypedParameter {
                        name: p.name,
                        ty: resolved_ty,
                        mutability: p.mutability,
                        kind: p.kind.clone(),
                        default_value: p.default_value.clone(),
                        attributes: p.attributes.clone(),
                        span: p.span,
                    }
                })
                .collect();

            // Return types should be explicitly typed in source
            let resolved_return_type = method.return_type.clone();

            // Create a function-like structure from the method to reuse function checking
            let func = TypedFunction {
                name: method.name,
                annotations: vec![],
                effects: vec![],
                type_params: vec![],
                params,
                return_type: resolved_return_type,
                body: method.body.clone(),
                visibility: Visibility::Public,
                is_async: method.is_async,
                is_pure: false,
                is_external: false,
                calling_convention: CallingConvention::Default,
                link_name: None,
            };

            self.check_function(&func)?;
        }

        Ok(())
    }

    /// Type check a function
    fn check_function(&mut self, func: &TypedFunction) -> Result<(), TypeError> {
        self.push_scope();

        // Add parameters to scope
        for param in &func.params {
            self.add_local(param.name, param.ty.clone(), param.mutability);
        }

        // Set return type
        let old_return = self.return_type.take();
        self.return_type = Some(func.return_type.clone());

        // Check body
        // Skip body checking for extern functions
        if func.is_external {
            return Ok(());
        }

        let body = func.body.as_ref().ok_or_else(|| {
            InferenceError::TypeMismatch {
                expected: Type::Primitive(PrimitiveType::Unit),
                found: Type::Never, // Non-extern function without body is error
            }
        })?;

        let body_ty = self.check_block(body)?;

        // Check return type matches
        // Skip if return type is Unit (no constraint needed) or if body type is Unit
        // (body may return via explicit `return` statements inside match/while/for blocks
        // which aren't fully tracked yet)
        if func.return_type != Type::Primitive(PrimitiveType::Unit)
            && body_ty != Type::Primitive(PrimitiveType::Unit)
        {
            self.inference.unify(body_ty, func.return_type.clone())?;
        }

        self.return_type = old_return;
        self.pop_scope();

        Ok(())
    }

    /// Type check a variable declaration
    fn check_variable_decl(&mut self, var: &TypedVariable) -> Result<(), TypeError> {
        if let Some(init_node) = &var.initializer {
            let init_ty = self.check_expression(&init_node.node)?;
            self.inference.unify(init_ty, var.ty.clone())?;
        }

        self.add_local(var.name, var.ty.clone(), var.mutability);
        Ok(())
    }

    /// Type check a statement
    pub fn check_statement(&mut self, stmt: &TypedStatement) -> Result<Type, TypeError> {
        match stmt {
            TypedStatement::Expression(expr_node) => {
                // Check the expression for type correctness, but statement returns Unit
                self.check_expression(&expr_node.node)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::Let(let_stmt) => {
                self.check_let_statement(let_stmt)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::Return(expr_opt) => {
                let ty = if let Some(expr_node) = expr_opt {
                    self.check_expression(&expr_node.node)?
                } else {
                    Type::Primitive(PrimitiveType::Unit)
                };

                if let Some(ret_ty) = &self.return_type {
                    self.inference.unify(ty.clone(), ret_ty.clone())?;
                }

                Ok(Type::Never)
            }
            TypedStatement::Yield(expr_node) => {
                // Yield is currently validated as an expression-producing statement.
                // Context-specific validation (e.g., only within compute reductions)
                // happens in lowering/runtime passes.
                self.check_expression(&expr_node.node)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::If(if_stmt) => self.check_if_statement(if_stmt),
            TypedStatement::While(while_stmt) => {
                self.check_while_statement(while_stmt)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::Block(block) => self.check_block(block),
            TypedStatement::Coroutine(coroutine) => {
                self.check_coroutine_statement(coroutine)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::Defer(defer_stmt) => {
                self.check_defer_statement(defer_stmt)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::Select(select_stmt) => {
                self.check_select_statement(select_stmt)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::ForCStyle(for_c_stmt) => {
                self.check_for_c_style_statement(for_c_stmt)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            TypedStatement::Loop(loop_stmt) => {
                self.check_loop_statement(loop_stmt)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
            // TODO: Implement other statement types
            TypedStatement::For(_)
            | TypedStatement::Match(_)
            | TypedStatement::Try(_)
            | TypedStatement::Throw(_)
            | TypedStatement::Break(_)
            | TypedStatement::Continue
            | TypedStatement::LetPattern(_) => {
                // Placeholder for now
                Ok(Type::Primitive(PrimitiveType::Unit))
            }
        }
    }

    /// Type check a let statement
    fn check_let_statement(&mut self, let_stmt: &TypedLet) -> Result<(), TypeError> {
        if let Some(init_node) = &let_stmt.initializer {
            let init_ty = self.check_expression(&init_node.node)?;
            self.inference.unify(init_ty, let_stmt.ty.clone())?;
        }

        self.add_local(let_stmt.name, let_stmt.ty.clone(), let_stmt.mutability);
        Ok(())
    }

    /// Type check an if statement
    fn check_if_statement(&mut self, if_stmt: &TypedIf) -> Result<Type, TypeError> {
        // Check condition is boolean
        let cond_ty = self.check_expression(&if_stmt.condition.node)?;
        self.inference
            .unify(cond_ty, Type::Primitive(PrimitiveType::Bool))?;

        // Check branches
        let then_ty = self.check_block(&if_stmt.then_block)?;

        if let Some(else_block) = &if_stmt.else_block {
            let else_ty = self.check_block(else_block)?;
            self.inference.unify(then_ty.clone(), else_ty)?;
            Ok(then_ty)
        } else {
            Ok(Type::Primitive(PrimitiveType::Unit))
        }
    }

    /// Type check a while statement
    fn check_while_statement(&mut self, while_stmt: &TypedWhile) -> Result<(), TypeError> {
        // Check condition
        let cond_ty = self.check_expression(&while_stmt.condition.node)?;
        self.inference
            .unify(cond_ty, Type::Primitive(PrimitiveType::Bool))?;

        // Check body with loop type
        let old_loop = self.loop_type.take();
        self.loop_type = Some(Type::Primitive(PrimitiveType::Unit));

        self.check_block(&while_stmt.body)?;

        self.loop_type = old_loop;
        Ok(())
    }

    /// Type check a block
    fn check_block(&mut self, block: &TypedBlock) -> Result<Type, TypeError> {
        self.push_scope();

        let mut last_ty = Type::Primitive(PrimitiveType::Unit);

        for stmt_node in &block.statements {
            last_ty = self.check_statement(&stmt_node.node)?;
        }

        self.pop_scope();
        Ok(last_ty)
    }

    /// Type check an expression
    pub fn check_expression(&mut self, expr: &TypedExpression) -> Result<Type, TypeError> {
        match expr {
            TypedExpression::Literal(lit) => Ok(self.literal_type(lit)),

            TypedExpression::Variable(name) => {
                if let Some((ty, _mutability)) = self.lookup_local(*name) {
                    Ok(ty.clone())
                } else {
                    // In lenient mode, unknown variables (likely external functions)
                    // get a fresh type variable instead of an error.
                    // This supports gradual typing where external symbols may not be
                    // fully typed. The actual resolution happens at codegen time.
                    if !self.options.no_implicit_any {
                        Ok(self.inference.fresh_type_var())
                    } else {
                        Err(TypeError::UndefinedVariable(*name))
                    }
                }
            }

            TypedExpression::Binary(bin) => self.check_binary_expr(bin),

            TypedExpression::Unary(un) => self.check_unary_expr(un),

            TypedExpression::Call(call) => self.check_call_expr(call),

            TypedExpression::Field(field) => self.check_field_expr(field),

            TypedExpression::Index(index) => self.check_index_expr(index),

            TypedExpression::Array(elements) => self.check_array_expr(elements),
            TypedExpression::Tuple(elements) => self.check_tuple_expr(elements),
            TypedExpression::Struct(struct_lit) => self.check_struct_expr(struct_lit),
            TypedExpression::Lambda(lambda) => self.check_lambda_expr(lambda),
            TypedExpression::Match(match_expr) => self.check_match_expr(match_expr),
            TypedExpression::If(if_expr) => self.check_if_expr(if_expr),
            TypedExpression::Cast(cast) => self.check_cast_expr(cast),
            TypedExpression::Await(expr) => {
                // Check inner expression; async types not yet fully tracked
                let _ = self.check_expression(&expr.node)?;
                Ok(self.inference.fresh_type_var())
            }
            TypedExpression::Try(expr) => self.check_try_expr(expr),
            TypedExpression::Reference(reference) => self.check_reference_expr(reference),
            TypedExpression::Dereference(expr) => self.check_dereference_expr(expr),
            TypedExpression::Range(range) => self.check_range_expr(range),
            TypedExpression::Block(block) => self.check_block(block),
            TypedExpression::ListComprehension(comp) => self.check_list_comprehension_expr(comp),
            TypedExpression::Slice(slice) => self.check_slice_expr(slice),
            TypedExpression::Compute(_) => {
                // Compute expressions are SIMD/kernel-specific; type checked at lowering
                Ok(self.inference.fresh_type_var())
            }
            TypedExpression::MethodCall(method_call) => self.check_method_call(method_call),
            TypedExpression::ImportModifier(import) => {
                // The type is resolved to the target_type (e.g., Image, AudioBuffer)
                Ok(Type::Unresolved(import.target_type))
            }
            TypedExpression::Path(_) => {
                // Path expressions need type resolution at a later stage
                Ok(self.inference.fresh_type_var())
            }
        }
    }

    /// Get type of a literal
    fn literal_type(&self, lit: &TypedLiteral) -> Type {
        match lit {
            TypedLiteral::Integer(_) => Type::Primitive(PrimitiveType::I32), // Default
            TypedLiteral::Float(_) => Type::Primitive(PrimitiveType::F64),   // Default
            TypedLiteral::Bool(_) => Type::Primitive(PrimitiveType::Bool),
            TypedLiteral::String(_) => Type::Primitive(PrimitiveType::String),
            TypedLiteral::Char(_) => Type::Primitive(PrimitiveType::Char),
            TypedLiteral::Unit => Type::Primitive(PrimitiveType::Unit),
            TypedLiteral::Null => Type::Optional(Box::new(Type::Unknown)),
            TypedLiteral::Undefined => Type::Unknown,
        }
    }

    /// Type check binary expression
    fn check_binary_expr(&mut self, bin: &TypedBinary) -> Result<Type, TypeError> {
        let left_ty = self.check_expression(&bin.left.node)?;
        let right_ty = self.check_expression(&bin.right.node)?;

        use BinaryOp::*;
        match bin.op {
            // Arithmetic operators
            Add | Sub | Mul | MatMul | Div | Rem => {
                self.inference.unify(left_ty.clone(), right_ty)?;
                Ok(left_ty)
            }

            // Comparison operators
            Eq | Ne | Lt | Le | Gt | Ge => {
                self.inference.unify(left_ty, right_ty)?;
                Ok(Type::Primitive(PrimitiveType::Bool))
            }

            // Logical operators
            And | Or => {
                self.inference
                    .unify(left_ty, Type::Primitive(PrimitiveType::Bool))?;
                self.inference
                    .unify(right_ty, Type::Primitive(PrimitiveType::Bool))?;
                Ok(Type::Primitive(PrimitiveType::Bool))
            }

            // Bitwise operators
            BitAnd | BitOr | BitXor | Shl | Shr => {
                self.inference.unify(left_ty.clone(), right_ty)?;
                Ok(left_ty)
            }

            // Assignment
            Assign => {
                // Check mutability
                if let TypedExpression::Variable(name) = &bin.left.node {
                    if let Some((_ty, mutability)) = self.lookup_local(*name) {
                        if *mutability != Mutability::Mutable {
                            return Err(TypeError::ImmutableVariable(*name));
                        }
                    }
                }

                self.inference.unify(left_ty, right_ty)?;
                Ok(Type::Primitive(PrimitiveType::Unit))
            }

            // Zig error handling operators
            Orelse => {
                // `a orelse b` - unwrap optional or use default
                // Left should be optional type, right is the default value
                // Result type is the unwrapped type
                Ok(right_ty)
            }
            Catch => {
                // `a catch b` - unwrap error union or use default
                // Left should be error union type, right is the default value
                // Result type is the success type
                Ok(right_ty)
            }
        }
    }

    /// Type check unary expression
    fn check_unary_expr(&mut self, un: &TypedUnary) -> Result<Type, TypeError> {
        let operand_ty = self.check_expression(&un.operand.node)?;

        use UnaryOp::*;
        match un.op {
            Plus | Minus => Ok(operand_ty),
            Not => {
                self.inference
                    .unify(operand_ty, Type::Primitive(PrimitiveType::Bool))?;
                Ok(Type::Primitive(PrimitiveType::Bool))
            }
            BitNot => Ok(operand_ty),
        }
    }

    /// Type check call expression
    fn check_call_expr(&mut self, call: &TypedCall) -> Result<Type, TypeError> {
        // Special check for abstract type suffix constructors (e.g., Duration::from_ms)
        // These are generated from suffixed literals like "1000ms"
        if let TypedExpression::Variable(var_name) = &call.callee.node {
            let name_str = var_name.resolve_global().unwrap_or_default();
            if let Some((type_name_str, suffix_part)) = name_str.split_once("::from_") {
                // This looks like a suffix constructor call
                // Check all types in the registry to find if any abstract type matches this name
                let registry = self.constraint_solver.type_registry();

                for type_def in registry.get_all_types() {
                    // Check if this type's name matches
                    let type_name_resolved = type_def.name.resolve_global().unwrap_or_default();
                    if type_name_resolved == type_name_str {
                        // Found the type, check if it's abstract with suffixes
                        if let crate::type_registry::TypeKind::Abstract { suffixes, .. } =
                            &type_def.kind
                        {
                            // Check if this suffix is registered for this abstract type
                            if !suffixes.contains(&suffix_part.to_string()) {
                                return Err(TypeError::MissingSuffixFunction {
                                    type_name: type_name_str.to_string(),
                                    suffix: suffix_part.to_string(),
                                    available_suffixes: suffixes.clone(),
                                    span: call.callee.span,
                                });
                            }
                            // If suffix is valid but the from_{suffix} function isn't implemented,
                            // that will be caught when we try to resolve the variable below
                        }
                        break;
                    }
                }
            }
        }

        // Type check the callee
        let callee_type = self.check_expression(&call.callee.node)?;

        // Extract function type
        let func_type = match &callee_type {
            Type::Function {
                params,
                return_type,
                has_named_params,
                has_default_params,
                ..
            } => Some((
                params.clone(),
                return_type.as_ref().clone(),
                *has_named_params,
                *has_default_params,
            )),
            // In lenient mode, unknown types (type variables, Any, Unknown) are allowed
            // for function calls. We just check argument expressions and return a fresh type var.
            Type::TypeVar(_) | Type::Any | Type::Unknown => {
                // Check argument expressions for side effects and type consistency
                for arg in &call.positional_args {
                    self.check_expression(&arg.node)?;
                }
                for arg in &call.named_args {
                    self.check_expression(&arg.value.node)?;
                }
                // Return fresh type variable since we don't know the return type
                return Ok(self.inference.fresh_type_var());
            }
            _ => {
                if !self.options.no_implicit_any {
                    // Lenient mode: allow unknown call types
                    for arg in &call.positional_args {
                        self.check_expression(&arg.node)?;
                    }
                    for arg in &call.named_args {
                        self.check_expression(&arg.value.node)?;
                    }
                    return Ok(self.inference.fresh_type_var());
                }
                return Err(TypeError::TypeMismatch {
                    expected: Type::Function {
                        params: vec![],
                        return_type: Box::new(Type::Any),
                        is_varargs: false,
                        has_named_params: false,
                        has_default_params: false,
                        async_kind: AsyncKind::Sync,
                        calling_convention: crate::CallingConvention::Default,
                        nullability: crate::NullabilityKind::Unknown,
                    },
                    found: callee_type,
                });
            }
        };

        // If we don't have function type info, early return already happened above
        let func_type = match func_type {
            Some(ft) => ft,
            None => return Ok(self.inference.fresh_type_var()),
        };

        let (expected_params, return_type, has_named_params, has_default_params) = func_type;

        // Validate argument counts and types
        self.validate_function_arguments(
            call,
            &expected_params,
            has_named_params,
            has_default_params,
        )?;

        // Type check all positional arguments
        let mut rest_param_index = None;
        for (i, param) in expected_params.iter().enumerate() {
            if param.is_varargs {
                rest_param_index = Some(i);
                break;
            }
        }

        for (i, arg) in call.positional_args.iter().enumerate() {
            let arg_type = self.check_expression(&arg.node)?;

            if let Some(rest_idx) = rest_param_index {
                if i >= rest_idx {
                    // This argument goes to the rest parameter
                    if let Some(param) = expected_params.get(rest_idx) {
                        self.validate_parameter_assignment(param, &arg_type, &arg.node)?;
                    }
                } else if let Some(param) = expected_params.get(i) {
                    // Regular parameter before rest
                    self.validate_parameter_assignment(param, &arg_type, &arg.node)?;
                }
            } else if let Some(param) = expected_params.get(i) {
                // No rest parameter, regular validation
                self.validate_parameter_assignment(param, &arg_type, &arg.node)?;
            }
        }

        // Type check all named arguments
        for named_arg in &call.named_args {
            let arg_type = self.check_expression(&named_arg.value.node)?;

            // Find the parameter with this name
            let param = expected_params
                .iter()
                .find(|p| p.name == Some(named_arg.name))
                .ok_or_else(|| TypeError::UnknownParameter(named_arg.name))?;

            self.validate_parameter_assignment(param, &arg_type, &named_arg.value.node)?;
        }

        // Validate generic type arguments if present
        for generic_arg in &call.type_args {
            // For now, just ensure they're valid types
            // TODO: Add constraint checking
            let _ = generic_arg;
        }

        Ok(return_type)
    }

    /// Validate function call arguments against parameters
    fn validate_function_arguments(
        &mut self,
        call: &TypedCall,
        expected_params: &[ParamInfo],
        has_named_params: bool,
        has_default_params: bool,
    ) -> Result<(), TypeError> {
        use crate::typed_ast::ParameterKind;

        let positional_count = call.positional_args.len();
        let named_count = call.named_args.len();
        let total_provided = positional_count + named_count;

        // Count required parameters (those without defaults)
        let required_count = expected_params
            .iter()
            .filter(|p| !p.is_optional && !p.is_varargs)
            .count();

        // Count optional parameters
        let optional_count = expected_params
            .iter()
            .filter(|p| p.is_optional || p.is_varargs)
            .count();

        let max_params = expected_params.len();

        // Check if we have a rest parameter
        let has_rest = expected_params.iter().any(|p| p.is_varargs);

        // Validate argument count
        if !has_rest && total_provided > max_params {
            return Err(TypeError::TooManyArguments {
                expected: max_params,
                found: total_provided,
            });
        }

        if total_provided < required_count {
            return Err(TypeError::TooFewArguments {
                expected: required_count,
                found: total_provided,
            });
        }

        // Validate named arguments are allowed
        if !call.named_args.is_empty() && !has_named_params {
            return Err(TypeError::NamedArgumentsNotAllowed);
        }

        Ok(())
    }

    /// Validate assignment to a specific parameter type
    fn validate_parameter_assignment(
        &mut self,
        param: &ParamInfo,
        arg_type: &Type,
        arg_expr: &TypedExpression,
    ) -> Result<(), TypeError> {
        // Check if out/ref/inout parameter requires assignable expression
        if param.is_out || param.is_ref || param.is_inout {
            self.validate_assignable_expression(arg_expr)?;
        }

        if param.is_varargs {
            // Rest parameters accept the element type
            if let Type::Array { element_type, .. } = &param.ty {
                // For rest parameters, individual arguments should match the element type
                self.inference
                    .unify(arg_type.clone(), element_type.as_ref().clone())?;
            } else {
                // Rest parameter should be array type
                return Err(TypeError::TypeMismatch {
                    expected: Type::Array {
                        element_type: Box::new(Type::Any),
                        size: None,
                        nullability: crate::NullabilityKind::Unknown,
                    },
                    found: param.ty.clone(),
                });
            }
        } else {
            // Regular parameter - just check type compatibility
            self.inference.unify(arg_type.clone(), param.ty.clone())?;
        }

        Ok(())
    }

    /// Validate that an expression is assignable (for out/ref/inout parameters)
    fn validate_assignable_expression(&self, expr: &TypedExpression) -> Result<(), TypeError> {
        match expr {
            TypedExpression::Variable(_) => Ok(()), // Variables are assignable
            TypedExpression::Field(_) => Ok(()),    // Field access is assignable
            TypedExpression::Index(_) => Ok(()),    // Array/map indexing is assignable
            TypedExpression::Dereference(_) => Ok(()), // Pointer dereference is assignable
            _ => Err(TypeError::NotAssignable),
        }
    }

    /// Type check field expression
    fn check_field_expr(&mut self, field: &TypedFieldAccess) -> Result<Type, TypeError> {
        let object_type = self.check_expression(&field.object.node)?;

        match &object_type {
            Type::Named { id, type_args, .. } => {
                // Look up the type definition — clone data out to avoid borrow conflicts
                use crate::type_registry::TypeRegistry;
                let lookup = self.inference.registry.get_type_by_id(*id).map(|td| {
                    let kind = td.kind.clone();
                    let type_params = td.type_params.clone();
                    kind
                });
                let type_params = self
                    .inference
                    .registry
                    .get_type_by_id(*id)
                    .map(|td| td.type_params.clone())
                    .unwrap_or_default();
                let type_args = type_args.clone();

                if let Some(kind) = lookup {
                    match &kind {
                        crate::type_registry::TypeKind::Struct { fields, .. } => {
                            // Look up the field in the struct definition
                            if let Some(field_def) = fields.iter().find(|f| f.name == field.field) {
                                // Apply type arguments to the field type if needed
                                let field_type = if !type_args.is_empty() && !type_params.is_empty()
                                {
                                    TypeRegistry::substitute_type_params(
                                        &field_def.ty,
                                        &type_params,
                                        &type_args,
                                    )
                                } else {
                                    field_def.ty.clone()
                                };
                                Ok(field_type)
                            } else {
                                Err(TypeError::UnknownParameter(field.field))
                            }
                        }
                        crate::type_registry::TypeKind::Interface { methods, .. } => {
                            // Look for a method or property with this name
                            if let Some(method) = methods.iter().find(|m| m.name == field.field) {
                                let return_ty = if !type_args.is_empty() && !type_params.is_empty()
                                {
                                    TypeRegistry::substitute_type_params(
                                        &method.return_type,
                                        &type_params,
                                        &type_args,
                                    )
                                } else {
                                    method.return_type.clone()
                                };
                                Ok(return_ty)
                            } else {
                                Err(TypeError::UnknownParameter(field.field))
                            }
                        }
                        _ => {
                            // Other type kinds don't support field access
                            Err(TypeError::TypeMismatch {
                                expected: Type::Never, // We don't have a specific struct type to expect
                                found: object_type,
                            })
                        }
                    }
                } else {
                    // Type not found in environment - add constraint for inference
                    let field_type = self.inference.fresh_type_var();
                    self.inference.add_has_field_constraint(
                        object_type,
                        field.field,
                        field_type.clone(),
                    )?;
                    Ok(field_type)
                }
            }

            Type::TypeVar(_) => {
                // For type variables, we can't determine the field type statically
                // Create a fresh type variable and add a constraint
                let field_type = self.inference.fresh_type_var();

                // Add a constraint that this type has this field
                self.inference.add_has_field_constraint(
                    object_type,
                    field.field,
                    field_type.clone(),
                )?;

                Ok(field_type)
            }

            _ => {
                // Type doesn't support field access
                Err(TypeError::TypeMismatch {
                    expected: Type::Any, // We don't have a specific struct type to expect
                    found: object_type,
                })
            }
        }
    }

    /// Type check index expression using symbol-based trait resolution
    fn check_index_expr(&mut self, index: &TypedIndex) -> Result<Type, TypeError> {
        let container_type = self.check_expression(&index.object.node)?;
        let index_type = self.check_expression(&index.index.node)?;

        match &container_type {
            Type::Array { element_type, .. } => {
                // Built-in array type: indexed by integers, returns element type
                self.inference
                    .unify(index_type, Type::Primitive(PrimitiveType::I32))?;
                Ok(element_type.as_ref().clone())
            }

            Type::Primitive(PrimitiveType::String) => {
                // Built-in string type: indexed by integers, returns characters
                self.inference
                    .unify(index_type, Type::Primitive(PrimitiveType::I32))?;
                Ok(Type::Primitive(PrimitiveType::Char))
            }

            Type::Named {
                id: _,
                type_args: _,
                ..
            } => {
                // For named types, we'll add a constraint and let the solver handle it
                let element_type = self.inference.fresh_type_var();
                self.inference.add_indexable_constraint(
                    container_type.clone(),
                    index_type,
                    element_type.clone(),
                )?;
                Ok(element_type)
            }

            Type::TypeVar(_) => {
                // For type variables, use constraint-based inference
                let element_type = self.inference.fresh_type_var();
                self.inference.add_indexable_constraint(
                    container_type,
                    index_type,
                    element_type.clone(),
                )?;
                Ok(element_type)
            }

            _ => {
                // Type doesn't support indexing
                Err(TypeError::TypeMismatch {
                    expected: Type::Array {
                        element_type: Box::new(Type::Any),
                        size: None,
                        nullability: crate::NullabilityKind::Unknown,
                    },
                    found: container_type,
                })
            }
        }
    }

    // ========================================================================
    // Expression type checking implementations
    // ========================================================================

    /// Type check array literal — unify all element types
    fn check_array_expr(
        &mut self,
        elements: &[TypedNode<TypedExpression>],
    ) -> Result<Type, TypeError> {
        if elements.is_empty() {
            return Ok(Type::Array {
                element_type: Box::new(self.inference.fresh_type_var()),
                size: None,
                nullability: crate::NullabilityKind::NonNull,
            });
        }
        let mut elem_type = self.check_expression(&elements[0].node)?;
        for elem in &elements[1..] {
            let ty = self.check_expression(&elem.node)?;
            elem_type = self.inference.unify(elem_type.clone(), ty)?;
        }
        Ok(Type::Array {
            element_type: Box::new(elem_type),
            size: None,
            nullability: crate::NullabilityKind::NonNull,
        })
    }

    /// Type check tuple literal — each element can be a different type
    fn check_tuple_expr(
        &mut self,
        elements: &[TypedNode<TypedExpression>],
    ) -> Result<Type, TypeError> {
        let mut types = Vec::with_capacity(elements.len());
        for elem in elements {
            types.push(self.check_expression(&elem.node)?);
        }
        Ok(Type::Tuple(types))
    }

    /// Type check struct literal — look up type, validate fields
    fn check_struct_expr(&mut self, struct_lit: &TypedStructLiteral) -> Result<Type, TypeError> {
        // Look up the type by name in the registry — clone data out to avoid borrow conflicts
        let type_info = self
            .inference
            .registry
            .get_type_by_name(struct_lit.name)
            .map(|td| {
                let id = td.id;
                let field_defs: Vec<_> =
                    if let crate::type_registry::TypeKind::Struct { fields, .. } = &td.kind {
                        fields.iter().map(|f| (f.name, f.ty.clone())).collect()
                    } else {
                        vec![]
                    };
                (id, field_defs)
            });

        if let Some((type_id, field_defs)) = type_info {
            // Check each field initializer
            for field_init in &struct_lit.fields {
                let init_ty = self.check_expression(&field_init.value.node)?;
                if let Some((_, field_ty)) = field_defs.iter().find(|(n, _)| *n == field_init.name)
                {
                    let _ = self.inference.unify(init_ty, field_ty.clone());
                }
            }

            Ok(Type::Named {
                id: type_id,
                type_args: vec![],
                const_args: vec![],
                variance: vec![],
                nullability: crate::NullabilityKind::NonNull,
            })
        } else {
            // Type not found — check fields anyway and return fresh type var
            for field_init in &struct_lit.fields {
                let _ = self.check_expression(&field_init.value.node)?;
            }
            Ok(self.inference.fresh_type_var())
        }
    }

    /// Type check lambda/closure expression
    fn check_lambda_expr(&mut self, lambda: &TypedLambda) -> Result<Type, TypeError> {
        self.push_scope();

        // Add parameters to scope
        let mut param_types = Vec::new();
        for param in &lambda.params {
            let param_type = param
                .ty
                .clone()
                .unwrap_or_else(|| self.inference.fresh_type_var());
            param_types.push(param_type.clone());
            self.add_local(param.name, param_type, crate::Mutability::Immutable);
        }

        // Check body
        let return_type = match &lambda.body {
            TypedLambdaBody::Expression(expr) => self.check_expression(&expr.node)?,
            TypedLambdaBody::Block(block) => self.check_block(block)?,
        };

        self.pop_scope();

        Ok(Type::Function {
            params: param_types
                .into_iter()
                .map(|ty| crate::ParamInfo {
                    name: None,
                    ty,
                    is_optional: false,
                    is_varargs: false,
                    is_keyword_only: false,
                    is_positional_only: false,
                    is_out: false,
                    is_ref: false,
                    is_inout: false,
                })
                .collect(),
            return_type: Box::new(return_type),
            is_varargs: false,
            has_named_params: false,
            has_default_params: false,
            async_kind: crate::AsyncKind::Sync,
            calling_convention: crate::CallingConvention::Default,
            nullability: crate::NullabilityKind::NonNull,
        })
    }

    /// Type check match expression — unify all arm body types
    fn check_match_expr(&mut self, match_expr: &TypedMatchExpr) -> Result<Type, TypeError> {
        let scrutinee_ty = self.check_expression(&match_expr.scrutinee.node)?;

        let mut result_type: Option<Type> = None;

        for arm in &match_expr.arms {
            self.push_scope();

            // Check pattern against scrutinee type (adds bindings to scope)
            self.check_pattern_against_type(&arm.pattern.node, &scrutinee_ty);

            // Check guard if present
            if let Some(guard) = &arm.guard {
                let guard_ty = self.check_expression(&guard.node)?;
                let _ = self
                    .inference
                    .unify(guard_ty, Type::Primitive(PrimitiveType::Bool));
            }

            // Check arm body
            let body_ty = self.check_expression(&arm.body.node)?;

            // Unify with previous arm types
            if let Some(ref prev_ty) = result_type {
                result_type = Some(
                    self.inference
                        .unify(prev_ty.clone(), body_ty)
                        .unwrap_or(prev_ty.clone()),
                );
            } else {
                result_type = Some(body_ty);
            }

            self.pop_scope();
        }

        Ok(result_type.unwrap_or(Type::Never))
    }

    /// Type check if expression — condition must be Bool, unify branches
    fn check_if_expr(&mut self, if_expr: &TypedIfExpr) -> Result<Type, TypeError> {
        let cond_ty = self.check_expression(&if_expr.condition.node)?;
        let _ = self
            .inference
            .unify(cond_ty, Type::Primitive(PrimitiveType::Bool));

        let then_ty = self.check_expression(&if_expr.then_branch.node)?;
        let else_ty = self.check_expression(&if_expr.else_branch.node)?;

        Ok(self
            .inference
            .unify(then_ty.clone(), else_ty)
            .unwrap_or(then_ty))
    }

    /// Type check cast expression — return target type directly
    fn check_cast_expr(&mut self, cast: &TypedCast) -> Result<Type, TypeError> {
        let _ = self.check_expression(&cast.expr.node)?;
        Ok(cast.target_type.clone())
    }

    /// Type check try expression — extract ok_type from Result
    fn check_try_expr(&mut self, expr: &TypedNode<TypedExpression>) -> Result<Type, TypeError> {
        let ty = self.check_expression(&expr.node)?;
        match ty {
            Type::Result { ok_type, .. } => Ok(*ok_type),
            _ => Ok(self.inference.fresh_type_var()),
        }
    }

    /// Type check reference expression — wrap in Reference type
    fn check_reference_expr(&mut self, reference: &TypedReference) -> Result<Type, TypeError> {
        let inner_ty = self.check_expression(&reference.expr.node)?;
        Ok(Type::Reference {
            ty: Box::new(inner_ty),
            mutability: reference.mutability,
            lifetime: None,
            nullability: crate::NullabilityKind::NonNull,
        })
    }

    /// Type check dereference expression — unwrap Reference type
    fn check_dereference_expr(
        &mut self,
        expr: &TypedNode<TypedExpression>,
    ) -> Result<Type, TypeError> {
        let ty = self.check_expression(&expr.node)?;
        match ty {
            Type::Reference { ty: inner, .. } => Ok(*inner),
            _ => {
                // In lenient mode, return fresh type var
                Ok(self.inference.fresh_type_var())
            }
        }
    }

    /// Type check range expression — unify start/end types
    fn check_range_expr(&mut self, range: &TypedRange) -> Result<Type, TypeError> {
        let mut range_type = self.inference.fresh_type_var();

        if let Some(start) = &range.start {
            let start_ty = self.check_expression(&start.node)?;
            range_type = self
                .inference
                .unify(range_type, start_ty)
                .unwrap_or(Type::Primitive(PrimitiveType::I32));
        }
        if let Some(end) = &range.end {
            let end_ty = self.check_expression(&end.node)?;
            range_type = self
                .inference
                .unify(range_type, end_ty)
                .unwrap_or(Type::Primitive(PrimitiveType::I32));
        }

        // Return as a named Range type or just the element type
        // (Range type depends on language — for now return fresh var with constraint)
        Ok(self.inference.fresh_type_var())
    }

    /// Type check list comprehension — [expr for var in iter if cond]
    fn check_list_comprehension_expr(
        &mut self,
        comp: &TypedListComprehension,
    ) -> Result<Type, TypeError> {
        let iter_ty = self.check_expression(&comp.iterator.node)?;

        // Extract element type from iterator
        let elem_ty = match &iter_ty {
            Type::Array { element_type, .. } => *element_type.clone(),
            _ => self.inference.fresh_type_var(),
        };

        self.push_scope();
        self.add_local(comp.variable, elem_ty, crate::Mutability::Immutable);

        // Check filter condition if present
        if let Some(cond) = &comp.condition {
            let cond_ty = self.check_expression(&cond.node)?;
            let _ = self
                .inference
                .unify(cond_ty, Type::Primitive(PrimitiveType::Bool));
        }

        // Check output expression
        let output_ty = self.check_expression(&comp.output_expr.node)?;
        self.pop_scope();

        Ok(Type::Array {
            element_type: Box::new(output_ty),
            size: None,
            nullability: crate::NullabilityKind::NonNull,
        })
    }

    /// Type check slice expression — preserves container type
    fn check_slice_expr(&mut self, slice: &TypedSlice) -> Result<Type, TypeError> {
        let obj_ty = self.check_expression(&slice.object.node)?;

        // Check indices are numeric
        if let Some(start) = &slice.start {
            let _ = self.check_expression(&start.node)?;
        }
        if let Some(end) = &slice.end {
            let _ = self.check_expression(&end.node)?;
        }
        if let Some(step) = &slice.step {
            let _ = self.check_expression(&step.node)?;
        }

        // Slice returns the same container type
        Ok(obj_ty)
    }

    /// Check pattern against expected type — adds variable bindings to scope
    fn check_pattern_against_type(&mut self, pattern: &TypedPattern, expected: &Type) {
        match pattern {
            TypedPattern::Wildcard => {
                // Matches anything, no bindings
            }
            TypedPattern::Identifier { name, mutability } => {
                // Bind variable to expected type in current scope
                self.add_local(*name, expected.clone(), *mutability);
            }
            TypedPattern::Literal(_) => {
                // Literal patterns should match the expected type
                // Full validation would unify literal type with expected
            }
            TypedPattern::Tuple(patterns) => {
                if let Type::Tuple(types) = expected {
                    for (pat, ty) in patterns.iter().zip(types.iter()) {
                        self.check_pattern_against_type(&pat.node, ty);
                    }
                }
            }
            TypedPattern::Struct { fields, .. } => {
                // For struct patterns, try to match field names to expected type fields
                for field in fields {
                    let fresh = self.inference.fresh_type_var();
                    self.check_pattern_against_type(&field.pattern.node, &fresh);
                }
            }
            TypedPattern::Enum { fields, .. } => {
                // For enum patterns, bind any field patterns
                for field in fields {
                    let fresh = self.inference.fresh_type_var();
                    self.check_pattern_against_type(&field.node, &fresh);
                }
            }
            TypedPattern::Array(patterns) => {
                let elem_ty = if let Type::Array { element_type, .. } = expected {
                    *element_type.clone()
                } else {
                    self.inference.fresh_type_var()
                };
                for pat in patterns {
                    self.check_pattern_against_type(&pat.node, &elem_ty);
                }
            }
            TypedPattern::Or(alternatives) => {
                for alt in alternatives {
                    self.check_pattern_against_type(&alt.node, expected);
                }
            }
            TypedPattern::Guard { pattern, .. } => {
                self.check_pattern_against_type(&pattern.node, expected);
            }
            TypedPattern::Rest { name, mutability } => {
                if let Some(name) = name {
                    self.add_local(*name, expected.clone(), *mutability);
                }
            }
            TypedPattern::Reference { pattern, .. } => {
                if let Type::Reference { ty: inner, .. } = expected {
                    self.check_pattern_against_type(&pattern.node, inner);
                } else {
                    self.check_pattern_against_type(&pattern.node, expected);
                }
            }
            TypedPattern::Box(inner) => {
                self.check_pattern_against_type(&inner.node, expected);
            }
            _ => {
                // Slice, Range patterns — handle minimally
            }
        }
    }

    /// Type check coroutine statement
    fn check_coroutine_statement(&mut self, coroutine: &TypedCoroutine) -> Result<(), TypeError> {
        // Check the coroutine body
        self.check_expression(&coroutine.body.node)?;

        // Check parameters if any
        for param in &coroutine.params {
            self.check_expression(&param.node)?;
        }

        // Validate coroutine kind semantics
        match coroutine.kind {
            CoroutineKind::Goroutine => {
                // Go goroutines should return unit type
                // TODO: Add more specific goroutine validation
            }
            CoroutineKind::Async => {
                // Async blocks should return Future<T>
                // TODO: Add async type validation
            }
            CoroutineKind::Generator => {
                // Generators should yield values
                // TODO: Add generator validation
            }
            CoroutineKind::Custom { .. } => {
                // Custom coroutines have user-defined semantics
            }
        }

        Ok(())
    }

    /// Type check defer statement
    fn check_defer_statement(&mut self, defer_stmt: &TypedDefer) -> Result<(), TypeError> {
        // Defer body should be a valid expression (usually a function call)
        self.check_expression(&defer_stmt.body.node)?;
        Ok(())
    }

    /// Type check select statement
    fn check_select_statement(&mut self, select_stmt: &TypedSelect) -> Result<(), TypeError> {
        // Check all select arms
        for arm in &select_stmt.arms {
            self.check_select_arm(arm)?;
        }

        // Check default case if present
        if let Some(default_block) = &select_stmt.default {
            self.check_block(default_block)?;
        }

        Ok(())
    }

    /// Type check select arm
    fn check_select_arm(&mut self, arm: &TypedSelectArm) -> Result<(), TypeError> {
        // Check the operation
        match &arm.operation {
            TypedSelectOperation::Receive { channel, pattern } => {
                let channel_ty = self.check_expression(&channel.node)?;
                // TODO: Validate channel is actually a channel type

                if let Some(pattern_node) = pattern {
                    // TODO: Check pattern matches channel element type
                    self.check_pattern(&pattern_node.node)?;
                }
            }
            TypedSelectOperation::Send { channel, value } => {
                let channel_ty = self.check_expression(&channel.node)?;
                let value_ty = self.check_expression(&value.node)?;
                // TODO: Validate channel can accept value type
            }
            TypedSelectOperation::Timeout { duration } => {
                let duration_ty = self.check_expression(&duration.node)?;
                // TODO: Validate duration is a time type
            }
        }

        // Check the arm body
        self.check_block(&arm.body)?;

        Ok(())
    }

    /// Type check C-style for statement
    fn check_for_c_style_statement(&mut self, for_stmt: &TypedForCStyle) -> Result<(), TypeError> {
        // Check initialization statement if present
        if let Some(init) = &for_stmt.init {
            self.check_statement(&init.node)?;
        }

        // Check condition if present (should be boolean)
        if let Some(condition) = &for_stmt.condition {
            let condition_ty = self.check_expression(&condition.node)?;
            self.inference
                .unify(condition_ty, Type::Primitive(PrimitiveType::Bool))?;
        }

        // Check update expression if present
        if let Some(update) = &for_stmt.update {
            self.check_expression(&update.node)?;
        }

        // Check loop body
        self.check_block(&for_stmt.body)?;

        Ok(())
    }

    /// Type check unified loop statement
    fn check_loop_statement(&mut self, loop_stmt: &TypedLoop) -> Result<(), TypeError> {
        match loop_stmt {
            TypedLoop::ForEach {
                pattern,
                iterator,
                body,
            } => {
                let iterator_ty = self.check_expression(&iterator.node)?;
                // TODO: Check that iterator implements iterable interface
                self.check_pattern(&pattern.node)?;
                self.check_block(body)?;
            }
            TypedLoop::ForCStyle {
                init,
                condition,
                update,
                body,
            } => {
                if let Some(init) = init {
                    self.check_statement(&init.node)?;
                }
                if let Some(condition) = condition {
                    let condition_ty = self.check_expression(&condition.node)?;
                    self.inference
                        .unify(condition_ty, Type::Primitive(PrimitiveType::Bool))?;
                }
                if let Some(update) = update {
                    self.check_expression(&update.node)?;
                }
                self.check_block(body)?;
            }
            TypedLoop::While { condition, body } => {
                let condition_ty = self.check_expression(&condition.node)?;
                self.inference
                    .unify(condition_ty, Type::Primitive(PrimitiveType::Bool))?;
                self.check_block(body)?;
            }
            TypedLoop::DoWhile { body, condition } => {
                self.check_block(body)?;
                let condition_ty = self.check_expression(&condition.node)?;
                self.inference
                    .unify(condition_ty, Type::Primitive(PrimitiveType::Bool))?;
            }
            TypedLoop::Infinite { body } => {
                self.check_block(body)?;
            }
        }

        Ok(())
    }

    /// Basic pattern checking (simplified for now)
    fn check_pattern(&mut self, _pattern: &TypedPattern) -> Result<Type, TypeError> {
        // TODO: Implement comprehensive pattern type checking
        Ok(self.inference.fresh_type_var())
    }

    /// Check method call with trait bounds verification
    fn check_method_call(&mut self, method_call: &TypedMethodCall) -> Result<Type, TypeError> {
        // Check the receiver type
        let receiver_ty = self.check_expression(&method_call.receiver.node)?;

        // Check method arguments
        let mut arg_types = Vec::new();
        for arg in &method_call.positional_args {
            arg_types.push(self.check_expression(&arg.node)?);
        }

        // Handle named arguments
        for named_arg in &method_call.named_args {
            arg_types.push(self.check_expression(&named_arg.value.node)?);
        }

        // Look up method definition using constraint solver's enhanced resolution
        match self.constraint_solver.resolve_method_with_trait_bounds(
            &receiver_ty,
            method_call.method,
            &method_call.type_args,
        ) {
            Ok(Some(resolved_method)) => {
                // Verify trait bounds for this method
                if let Err(errors) = self
                    .constraint_solver
                    .verify_method_trait_bounds(&resolved_method)
                {
                    for error in errors {
                        self.emit_solver_error(error, method_call.receiver.span);
                    }
                    return Ok(self.inference.fresh_type_var()); // Error recovery
                }

                // Check argument compatibility
                self.check_method_arguments(
                    &resolved_method.signature,
                    &arg_types,
                    &method_call.named_args,
                )?;

                // Return the method's return type, instantiated with type arguments
                Ok(self
                    .constraint_solver
                    .instantiate_method_return_type(&resolved_method))
            }
            Ok(None) => {
                // Method not found - generate appropriate error
                self.report_method_not_found(&receiver_ty, method_call.method)?;
                Ok(self.inference.fresh_type_var()) // Error recovery
            }
            Err(errors) => {
                // Method resolution failed due to constraint errors
                for error in errors {
                    self.emit_solver_error(error, method_call.receiver.span);
                }
                Ok(self.inference.fresh_type_var()) // Error recovery
            }
        }
    }

    /// Resolve method signature for a given receiver type and method name
    fn resolve_method(
        &mut self,
        receiver_ty: &Type,
        method_name: InternedString,
        _type_args: &[Type],
    ) -> Result<Option<MethodSig>, TypeError> {
        match receiver_ty {
            Type::Named {
                id: _,
                type_args: _,
                ..
            } => {
                // Look for method in the type definition (intrinsic methods)
                // For now, types don't have intrinsic methods, so we skip this step

                // Look for trait methods implemented for this type
                self.resolve_trait_method(receiver_ty, method_name)
            }
            Type::Primitive(_) => {
                // Look for trait methods implemented for primitive types
                self.resolve_trait_method(receiver_ty, method_name)
            }
            Type::TypeVar(_) => {
                // For type variables, we need to defer until more information is available
                // For now, return None and let constraint solving handle it
                Ok(None)
            }
            _ => {
                // Other types don't support method calls for now
                Ok(None)
            }
        }
    }

    /// Resolve trait method for a given type by searching trait implementations
    fn resolve_trait_method(
        &self,
        receiver_ty: &Type,
        method_name: InternedString,
    ) -> Result<Option<MethodSig>, TypeError> {
        let registry = &self.inference.registry;

        // 1. Check inherent methods on the type definition itself
        match receiver_ty {
            Type::Named { id, .. } => {
                if let Some(type_def) = registry.get_type_by_id(*id) {
                    for method in &type_def.methods {
                        if method.name == method_name {
                            return Ok(Some(method.clone()));
                        }
                    }
                }
            }
            _ => {}
        }

        // 2. Get all traits implemented by this type and search their methods
        let implemented_traits = registry.get_implementations(receiver_ty);

        for trait_id in &implemented_traits {
            // Check trait definition's declared methods
            if let Some(trait_def) = registry.get_trait_by_id(*trait_id) {
                for method in &trait_def.methods {
                    if method.name == method_name {
                        return Ok(Some(method.clone()));
                    }
                }
            }
        }

        // 3. Fall back to searching impl methods directly
        //    (impl blocks may add methods not declared in the trait definition,
        //     e.g. extern trait impls with 0 declared methods)
        for (_trait_id, impls) in registry.iter_implementations() {
            for impl_def in impls {
                if registry.impl_matches_type(&impl_def.for_type, receiver_ty) {
                    for method_impl in &impl_def.methods {
                        if method_impl.signature.name == method_name {
                            return Ok(Some(method_impl.signature.clone()));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Verify that the receiver type satisfies all trait bounds required by the method
    fn verify_method_trait_bounds(
        &mut self,
        receiver_ty: &Type,
        method_sig: &MethodSig,
    ) -> Result<(), TypeError> {
        for type_param in &method_sig.type_params {
            for bound in &type_param.bounds {
                match bound {
                    TypeBound::Trait {
                        name: trait_name, ..
                    } => {
                        // Look up the trait by name, get its ID
                        let trait_id = self
                            .inference
                            .registry
                            .get_trait_by_name(*trait_name)
                            .map(|td| td.id);

                        if let Some(trait_id) = trait_id {
                            if !self
                                .inference
                                .registry
                                .type_implements(receiver_ty, trait_id)
                            {
                                return Err(TypeError::TraitNotImplemented {
                                    ty: receiver_ty.clone(),
                                    trait_name: *trait_name,
                                    span: method_sig.span,
                                });
                            }
                        }
                        // If trait not found in registry, skip (may be external/unresolved)
                    }
                    _ => {
                        // Other bound types (Sized, Copy, Send, Sync, Lifetime, etc.)
                        // are handled by dedicated checkers or deferred
                    }
                }
            }
        }
        Ok(())
    }

    /// Check method argument compatibility with method signature
    fn check_method_arguments(
        &mut self,
        method_sig: &MethodSig,
        arg_types: &[Type],
        _named_args: &[TypedNamedArg],
    ) -> Result<(), TypeError> {
        // Exclude 'self' parameter from arity check — callers only provide non-self args
        let non_self_params: Vec<_> = method_sig.params.iter().filter(|p| !p.is_self).collect();

        // Determine arity bounds:
        // - Required: params that are not varargs and not Optional-typed
        // - Maximum: all non-self params (or unlimited if varargs present)
        let has_varargs = non_self_params.iter().any(|p| p.is_varargs);

        let required_count = non_self_params
            .iter()
            .filter(|p| !p.is_varargs && !matches!(p.ty, Type::Optional(_)))
            .count();

        let max_count = if has_varargs {
            usize::MAX
        } else {
            non_self_params.len()
        };

        let provided = arg_types.len();

        if provided < required_count || provided > max_count {
            return Err(TypeError::ArityMismatch {
                expected: non_self_params.len(),
                found: provided,
                span: method_sig.span,
            });
        }

        // Check type compatibility for each provided argument
        for (i, arg_ty) in arg_types.iter().enumerate() {
            if i < non_self_params.len() {
                let param = &non_self_params[i];
                // If param type is Optional(T) but arg is not Optional, accept T directly
                let target_ty = match &param.ty {
                    Type::Optional(inner) if !matches!(arg_ty, Type::Optional(_)) => *inner.clone(),
                    other => other.clone(),
                };
                self.inference
                    .unify(arg_ty.clone(), target_ty)
                    .map_err(|_| TypeError::TypeMismatch {
                        expected: param.ty.clone(),
                        found: arg_ty.clone(),
                    })?;
            }
            // If i >= non_self_params.len() and has_varargs, skip type checking
            // (varargs are dynamically typed)
        }

        Ok(())
    }

    /// Instantiate method return type with provided type arguments
    fn instantiate_method_return_type(&self, method_sig: &MethodSig, type_args: &[Type]) -> Type {
        if type_args.is_empty() || method_sig.type_params.is_empty() {
            // No type arguments provided or method not generic
            return method_sig.return_type.clone();
        }

        // Create substitution map from type parameters to type arguments
        let mut substitution = std::collections::HashMap::new();
        for (param, arg) in method_sig.type_params.iter().zip(type_args.iter()) {
            substitution.insert(param.name, arg.clone());
        }

        // Apply substitution to return type
        self.apply_type_substitution(&method_sig.return_type, &substitution)
    }

    /// Apply type substitution to a type
    fn apply_type_substitution(
        &self,
        ty: &Type,
        substitution: &std::collections::HashMap<InternedString, Type>,
    ) -> Type {
        match ty {
            Type::Named {
                id,
                type_args,
                const_args,
                variance,
                nullability,
            } => {
                // Named types don't participate in this kind of substitution
                // Only type variables would be substituted
                Type::Named {
                    id: *id,
                    type_args: type_args
                        .iter()
                        .map(|arg| self.apply_type_substitution(arg, substitution))
                        .collect(),
                    const_args: vec![], // Check?
                    variance: vec![],   // Check?
                    nullability: *nullability,
                }
            }
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
                        ty: self.apply_type_substitution(&p.ty, substitution),
                        is_optional: p.is_optional,
                        is_varargs: p.is_varargs,
                        is_keyword_only: p.is_keyword_only,
                        is_positional_only: p.is_positional_only,
                        is_out: p.is_out,
                        is_ref: p.is_ref,
                        is_inout: p.is_inout,
                    })
                    .collect(),
                return_type: Box::new(self.apply_type_substitution(return_type, substitution)),
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
                element_type: Box::new(self.apply_type_substitution(element_type, substitution)),
                size: size.clone(),
                nullability: *nullability,
            },
            Type::Tuple(types) => Type::Tuple(
                types
                    .iter()
                    .map(|t| self.apply_type_substitution(t, substitution))
                    .collect(),
            ),
            _ => ty.clone(), // For primitive types and other non-substitutable types
        }
    }

    /// Report method not found error
    fn report_method_not_found(
        &mut self,
        receiver_ty: &Type,
        method_name: InternedString,
    ) -> Result<(), TypeError> {
        Err(TypeError::MethodNotFound {
            receiver_ty: receiver_ty.clone(),
            method_name,
            span: Span::new(0, 0), // TODO: Get proper span
        })
    }
}
