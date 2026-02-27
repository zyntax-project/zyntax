//! # Zyntax Compiler Infrastructure
//!
//! This crate provides the compilation pipeline from TypedAST to executable code,
//! supporting both JIT (via Cranelift) and AOT (via LLVM) compilation targets.
//!
//! ## Architecture
//!
//! The compiler uses a multi-tier approach:
//! - **HIR (High-level IR)**: Platform-agnostic representation with CFG and SSA
//! - **Cranelift Backend**: Fast JIT compilation with hot-reloading support
//! - **LLVM Backend**: Optimized AOT compilation to native targets
//!
//! ## Design Principles
//!
//! - Shared HIR ensures consistent semantics across backends
//! - CFG and SSA form enable standard compiler optimizations
//! - Type information preserved for optimization opportunities
//! - Memory safety validated before code generation

pub mod analysis;
pub mod associated_type_resolver; // Associated type resolution for trait dispatch
pub mod async_support;
pub mod borrow_check; // HIR-level borrow checking pass
pub mod bytecode; // HIR bytecode serialization/deserialization
pub mod cfg;
pub mod const_eval;
pub mod effect_analysis; // Effect inference and checking for algebraic effects
pub mod effect_codegen; // Code generation support for algebraic effects
pub mod effect_handler_resolution; // Handler resolution for effect dispatch
pub mod hir;
pub mod hir_builder; // HIR Builder API for direct HIR construction
pub mod hir_dump; // CLIF-inspired HIR text dump for debugging
pub mod lowering;
pub mod memory_management;
pub mod memory_optimization; // Memory-aware optimizations
pub mod memory_pass;
pub mod monomorphize;
pub mod optimization;
pub mod pattern_matching;
pub mod runtime;
pub mod ssa;
pub mod stdlib; // Standard library implementation using HIR Builder
pub mod trait_lowering; // Trait/interface lowering to HIR
pub mod typed_cfg; // New: TypedAST-aware CFG builder
pub mod vtable_registry; // Vtable management and caching // Async runtime (executor, task, waker)

pub mod cranelift_backend;

#[cfg(feature = "llvm-backend")]
pub mod llvm_backend;

#[cfg(feature = "llvm-backend")]
pub mod llvm_jit_backend; // LLVM MCJIT for hot-path optimization

pub mod plugin; // Plugin system for frontend runtime registration
pub mod profiling; // Runtime profiling for tiered compilation
pub mod tiered_backend; // Tiered JIT/AOT compilation system
pub mod zpack;
pub mod zrtl; // ZRTL (Zyntax Runtime Library) dynamic plugin format // ZPack package format for distributing modules + runtimes

// Re-export key types
pub use async_support::{
    AsyncCapture, AsyncCaptureMode, AsyncCompiler, AsyncRuntime, AsyncRuntimeType, AsyncState,
    AsyncStateMachine, AsyncTerminator,
};
pub use borrow_check::{
    run_borrow_check, validate_borrow_check, BorrowCheckResult, BorrowError, BorrowWarning,
    HirBorrowChecker,
};
pub use cfg::{BasicBlock, CfgEdge, ControlFlowGraph};
pub use const_eval::{ConstEvalContext, ConstEvaluator};
pub use cranelift_backend::{
    cranelift_skipped_function_count, reset_cranelift_skipped_function_count,
};
pub use effect_analysis::{
    analyze_effects, analyze_effects_with_call_graph, get_function_effect_summary,
    has_effect_errors, EffectAnalyzer, EffectCallGraph, EffectError, EffectErrorKind,
    EffectOccurrence, EffectSite, EffectSummary, EffectWarning, EffectWarningKind,
    FunctionEffectAnalysis, HandlerScope, ModuleEffectAnalysis,
};
pub use effect_codegen::{
    analyze_handle_effect, analyze_perform_effect, get_handler_ops_info, mangle_handler_op_name,
    mangle_handler_state_name, EffectCodegenContext, HandleEffectCodegen, HandlerOpInfo,
    HandlerStackEntry, HandlerStateInfo, PerformEffectCodegen, PerformStrategy, StateVarInfo,
};
pub use effect_handler_resolution::{
    can_inline_handler, get_optimization_level, resolve_handlers, resolve_handlers_with_analysis,
    FunctionHandlerResolution, HandlerOptimization, HandlerResolution, HandlerResolver,
    HandlerScopeNode, HandlerScopeTree, ModuleHandlerResolution, PerformSite, ResolutionStats,
    ResolvedHandler,
};
pub use hir::{HirBlock, HirFunction, HirId, HirInstruction, HirModule, HirValue};
pub use hir_builder::HirBuilder; // HIR Builder API
use log::debug;
pub use lowering::{
    lowering_skipped_function_count,
    reset_lowering_skipped_function_count,
    AstLowering,
    BuiltinResolver,
    ChainedResolver,
    ExportedSymbol,
    ImportContext,
    ImportError,
    ImportManager,
    // Re-export import resolver types for convenience
    ImportResolver,
    LoweringConfig,
    LoweringContext,
    LoweringPipeline,
    ModuleArchitecture,
    ResolvedImport,
    SymbolKind,
};
pub use memory_management::{
    convert_linearity_kind,
    ARCManager,
    AllocationInfo,
    CleanupAction,
    CleanupInfo,
    DropManager,
    EscapeAnalysis,
    EscapeInfo,
    HirLinearityKind,
    MemoryContext,
    MemoryStrategy,
    RefCountInfo,
    ScopeCleanupInfo,
    StackPromotionPass,
    // Unified cleanup types
    UnifiedCleanupBehavior,
    UnifiedCleanupManager,
};
pub use memory_optimization::MemoryOptimizationPass;
pub use memory_pass::MemoryManagementPass;
pub use monomorphize::{monomorphize_module, MonomorphizationContext};
pub use pattern_matching::{check_exhaustiveness, DecisionNode, PatternMatchCompiler};
pub use ssa::{PhiNode, SsaBuilder, SsaForm};
pub use typed_cfg::{TypedBasicBlock, TypedCfgBuilder, TypedControlFlowGraph, TypedTerminator};

use optimization::OptimizationPass;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use zyntax_typed_ast::TypedProgram;

#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("Lowering error: {0}")]
    Lowering(String),

    #[error("Analysis error: {0}")]
    Analysis(String),

    #[error("Optimization error: {0}")]
    Optimization(String),

    #[error("Code generation error: {0}")]
    CodeGen(String),

    #[error("Backend error: {0}")]
    Backend(String),
}

// Implement From for inkwell BuilderError (only when llvm-backend feature is enabled)
#[cfg(feature = "llvm-backend")]
impl From<inkwell::builder::BuilderError> for CompilerError {
    fn from(err: inkwell::builder::BuilderError) -> Self {
        CompilerError::CodeGen(format!("LLVM builder error: {}", err))
    }
}

pub type CompilerResult<T> = Result<T, CompilerError>;

/// Compilation pipeline configuration
#[derive(Clone)]
pub struct CompilationConfig {
    /// Optimization level (0-3)
    pub opt_level: u8,
    /// Enable debug information
    pub debug_info: bool,
    /// Target triple for code generation
    pub target_triple: String,
    /// Enable hot-reloading support (Cranelift only)
    pub hot_reload: bool,
    /// Enable monomorphization of generic functions
    pub enable_monomorphization: bool,
    /// Memory management strategy
    pub memory_strategy: Option<memory_management::MemoryStrategy>,
    /// Async runtime configuration
    pub async_runtime: Option<async_support::AsyncRuntimeType>,
    /// Import resolver for resolving import statements during compilation
    pub import_resolver: Option<Arc<dyn ImportResolver>>,
    /// Enable HIR-level borrow checking (validates ownership/borrowing at HIR level)
    pub enable_borrow_check: bool,
    /// Enable effect analysis and checking (validates algebraic effect usage)
    pub enable_effect_check: bool,
}

impl std::fmt::Debug for CompilationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompilationConfig")
            .field("opt_level", &self.opt_level)
            .field("debug_info", &self.debug_info)
            .field("target_triple", &self.target_triple)
            .field("hot_reload", &self.hot_reload)
            .field("enable_monomorphization", &self.enable_monomorphization)
            .field("memory_strategy", &self.memory_strategy)
            .field("async_runtime", &self.async_runtime)
            .field(
                "import_resolver",
                &self.import_resolver.as_ref().map(|r| r.resolver_name()),
            )
            .field("enable_borrow_check", &self.enable_borrow_check)
            .field("enable_effect_check", &self.enable_effect_check)
            .finish()
    }
}

impl Default for CompilationConfig {
    fn default() -> Self {
        Self {
            opt_level: 2,
            debug_info: true,
            target_triple: "x86_64-unknown-linux-gnu".to_string(),
            hot_reload: false,
            enable_monomorphization: true,
            memory_strategy: Some(memory_management::MemoryStrategy::ARC),
            async_runtime: Some(async_support::AsyncRuntimeType::Tokio),
            import_resolver: None,
            enable_borrow_check: false, // Disabled by default for now
            enable_effect_check: true,  // Enable effect checking by default
        }
    }
}

/// Main compilation pipeline: TypedAST → HIR → Optimized HIR
///
/// This function orchestrates the complete compilation process:
/// Register all impl blocks from a TypedProgram into its type registry
/// This must be called before lowering starts
pub fn register_impl_blocks(
    program: &mut zyntax_typed_ast::TypedProgram,
) -> Result<(), CompilerError> {
    use zyntax_typed_ast::type_registry::{ImplDef, MethodImpl, MethodSig, ParamDef};
    use zyntax_typed_ast::typed_ast::TypedDeclaration;
    use zyntax_typed_ast::Type;

    // First pass: Register all traits
    for decl in program.declarations.iter() {
        if let TypedDeclaration::Interface(trait_decl) = &decl.node {
            let trait_def = zyntax_typed_ast::type_registry::TraitDef {
                id: zyntax_typed_ast::type_registry::TypeId::next(),
                name: trait_decl.name,
                methods: vec![], // Simplified for now
                associated_types: vec![],
                super_traits: vec![],
                type_params: vec![],
                is_object_safe: true,
                span: trait_decl.span,
            };
            program.type_registry.register_trait(trait_def);
        }
    }

    // Second pass: Register impl blocks
    let mut impl_count = 0;
    for decl in program.declarations.iter() {
        if let TypedDeclaration::Impl(impl_block) = &decl.node {
            impl_count += 1;

            // Check if this is an inherent impl (empty trait name)
            let trait_name_str = impl_block
                .trait_name
                .resolve_global()
                .unwrap_or_else(|| String::new());
            let is_inherent = trait_name_str.is_empty();

            if is_inherent {
                // For inherent impls, register methods directly on the type
                // Resolve the implementing type
                let implementing_type_id = match &impl_block.for_type {
                    Type::Unresolved(name) => {
                        // Try to look up by resolved name if InternedString doesn't match
                        let name_str = name.resolve_global().unwrap_or_default();
                        let by_name = program.type_registry.get_type_by_name(*name);
                        if let Some(type_def) = by_name {
                            Some(type_def.id)
                        } else {
                            // Try with a fresh InternedString from the name string
                            let fresh_name =
                                zyntax_typed_ast::InternedString::new_global(&name_str);
                            if let Some(type_def) =
                                program.type_registry.get_type_by_name(fresh_name)
                            {
                                Some(type_def.id)
                            } else {
                                None
                            }
                        }
                    }
                    Type::Extern { name, .. } => {
                        // Look up extern type by name
                        let name_str = name.resolve_global().unwrap_or_default();
                        if let Some(type_def) = program.type_registry.get_type_by_name(*name) {
                            Some(type_def.id)
                        } else {
                            // Try with a fresh InternedString
                            let fresh_name =
                                zyntax_typed_ast::InternedString::new_global(&name_str);
                            if let Some(type_def) =
                                program.type_registry.get_type_by_name(fresh_name)
                            {
                                Some(type_def.id)
                            } else {
                                None
                            }
                        }
                    }
                    Type::Named { id, .. } => Some(*id),
                    _ => None,
                };

                if let Some(type_id) = implementing_type_id {
                    // Build MethodSig list for inherent methods
                    for method in &impl_block.methods {
                        let params: Vec<ParamDef> = method
                            .params
                            .iter()
                            .map(|p| ParamDef {
                                name: p.name,
                                ty: p.ty.clone(),
                                is_self: p.is_self,
                                is_varargs: false,
                                is_mut: matches!(
                                    p.mutability,
                                    zyntax_typed_ast::type_registry::Mutability::Mutable
                                ),
                            })
                            .collect();

                        let type_params: Vec<zyntax_typed_ast::type_registry::TypeParam> = method
                            .type_params
                            .iter()
                            .map(|tp| zyntax_typed_ast::type_registry::TypeParam {
                                name: tp.name,
                                bounds: vec![],
                                variance: zyntax_typed_ast::type_registry::Variance::Invariant,
                                default: tp.default.clone(),
                                span: tp.span,
                            })
                            .collect();

                        let method_sig = MethodSig {
                            name: method.name,
                            type_params,
                            params,
                            return_type: method.return_type.clone(),
                            where_clause: vec![],
                            is_static: false,
                            is_async: method.is_async,
                            visibility: zyntax_typed_ast::type_registry::Visibility::Public,
                            span: method.span,
                            is_extension: false,
                        };

                        // Register the inherent method on the type
                        program
                            .type_registry
                            .add_method_to_type(type_id, method_sig.clone());
                    }
                }
                continue;
            }

            // Find the trait by name - be lenient if trait not found
            let trait_def = match program
                .type_registry
                .get_trait_by_name(impl_block.trait_name)
            {
                Some(t) => t,
                None => {
                    // Try to resolve trait name for better warning message
                    let trait_name_str = impl_block
                        .trait_name
                        .resolve_global()
                        .unwrap_or_else(|| format!("{:?}", impl_block.trait_name));
                    log::warn!(
                        "Trait '{}' not found in registry, skipping impl block",
                        trait_name_str
                    );
                    continue;
                }
            };
            let trait_id = trait_def.id;

            // Resolve the implementing type
            let implementing_type = match &impl_block.for_type {
                Type::Unresolved(name) => {
                    if let Some(type_def) = program.type_registry.get_type_by_name(*name) {
                        Type::Named {
                            id: type_def.id,
                            type_args: vec![],
                            const_args: vec![],
                            variance: vec![],
                            nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                        }
                    } else {
                        impl_block.for_type.clone()
                    }
                }
                _ => impl_block.for_type.clone(),
            };

            // Build MethodImpl list
            let method_impls: Vec<MethodImpl> = impl_block
                .methods
                .iter()
                .map(|method| {
                    let params: Vec<ParamDef> = method
                        .params
                        .iter()
                        .map(|p| ParamDef {
                            name: p.name,
                            ty: p.ty.clone(),
                            is_self: p.is_self,
                            is_varargs: false,
                            is_mut: matches!(
                                p.mutability,
                                zyntax_typed_ast::type_registry::Mutability::Mutable
                            ),
                        })
                        .collect();

                    let type_params: Vec<zyntax_typed_ast::type_registry::TypeParam> = method
                        .type_params
                        .iter()
                        .map(|tp| {
                            zyntax_typed_ast::type_registry::TypeParam {
                                name: tp.name,
                                bounds: vec![], // Simplified for now
                                variance: zyntax_typed_ast::type_registry::Variance::Invariant,
                                default: tp.default.clone(),
                                span: tp.span,
                            }
                        })
                        .collect();

                    let method_sig = MethodSig {
                        name: method.name,
                        type_params,
                        params,
                        return_type: method.return_type.clone(),
                        where_clause: vec![],
                        is_static: false,
                        is_async: method.is_async,
                        visibility: zyntax_typed_ast::type_registry::Visibility::Public,
                        span: method.span,
                        is_extension: false,
                    };

                    MethodImpl {
                        signature: method_sig,
                        is_default: false,
                    }
                })
                .collect();

            // Create and register ImplDef
            let impl_def = ImplDef {
                trait_id,
                for_type: implementing_type.clone(),
                type_args: vec![],
                methods: method_impls,
                associated_types: std::collections::HashMap::new(),
                where_clause: vec![],
                span: impl_block.span,
            };

            program
                .type_registry
                .register_implementation(impl_def.clone());
        }
    }

    Ok(())
}

/// Run linear type checking on a TypedProgram
///
/// This function validates ownership, borrowing, and lifetime constraints
/// before lowering to HIR. It ensures:
/// - Linear types are used exactly once
/// - Affine types are used at most once
/// - Borrow rules are respected (no mutable aliases)
/// - Resources are properly cleaned up
///
/// Returns errors if linear type constraints are violated.
pub fn run_linear_type_check(
    program: &zyntax_typed_ast::TypedProgram,
) -> Result<(), CompilerError> {
    use zyntax_typed_ast::LinearTypeChecker;

    eprintln!("[LINEAR_CHECK] Running linear type checking...");

    let mut checker = LinearTypeChecker::new();

    match checker.check_program(program) {
        Ok(()) => {
            eprintln!("[LINEAR_CHECK] Linear type checking passed");
            Ok(())
        }
        Err(e) => {
            eprintln!("[LINEAR_CHECK] Linear type error: {:?}", e);
            Err(CompilerError::Analysis(format!(
                "Linear type error: {:?}",
                e
            )))
        }
    }
}

/// Generate automatic trait implementations for abstract types
/// Abstract types wrapping numeric primitives automatically inherit numeric operation traits
pub fn generate_abstract_trait_impls(
    program: &mut zyntax_typed_ast::TypedProgram,
) -> Result<(), CompilerError> {
    use std::collections::HashMap;
    use zyntax_typed_ast::arena::InternedString;
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::{
        ImplDef, MethodImpl, MethodSig, NullabilityKind, PrimitiveType, TypeKind, Variance,
    };
    use zyntax_typed_ast::{
        BinaryOp, Type, TypedBinary, TypedBlock, TypedDeclaration, TypedExpression, TypedFunction,
        TypedImplAssociatedType, TypedMethod, TypedNode, TypedParameter, TypedStatement,
        TypedTraitImpl, TypedUnary, UnaryOp,
    };

    // Binary operation traits that should be automatically implemented for numeric types
    let binary_traits = vec![
        ("Add", "add", BinaryOp::Add),
        ("Sub", "sub", BinaryOp::Sub),
        ("Mul", "mul", BinaryOp::Mul),
        ("Div", "div", BinaryOp::Div),
        ("Mod", "mod", BinaryOp::Rem),
    ];

    // Comparison traits
    let comparison_traits = vec![
        ("Eq", vec![("eq", BinaryOp::Eq), ("ne", BinaryOp::Ne)]),
        (
            "Ord",
            vec![
                ("lt", BinaryOp::Lt),
                ("gt", BinaryOp::Gt),
                ("le", BinaryOp::Le),
                ("ge", BinaryOp::Ge),
            ],
        ),
    ];

    // Unary operation traits
    let unary_traits = vec![("Neg", "neg", UnaryOp::Minus)];

    // Iterate through all registered types
    let type_ids: Vec<_> = program
        .type_registry
        .get_all_types()
        .filter_map(|type_def| {
            if matches!(&type_def.kind, TypeKind::Abstract { .. }) {
                Some((type_def.id, type_def.name, type_def.kind.clone()))
            } else {
                None
            }
        })
        .collect();

    let mut generated_impls: Vec<TypedNode<TypedDeclaration>> = Vec::new();

    for (type_id, type_name, kind) in type_ids {
        if let TypeKind::Abstract {
            underlying_type, ..
        } = &kind
        {
            // Check if underlying type is numeric
            let is_numeric = matches!(
                underlying_type,
                Type::Primitive(
                    PrimitiveType::I8
                        | PrimitiveType::I16
                        | PrimitiveType::I32
                        | PrimitiveType::I64
                        | PrimitiveType::U8
                        | PrimitiveType::U16
                        | PrimitiveType::U32
                        | PrimitiveType::U64
                        | PrimitiveType::F32
                        | PrimitiveType::F64
                )
            );

            if is_numeric {
                let abstract_type = Type::Named {
                    id: type_id,
                    type_args: vec![],
                    const_args: vec![],
                    variance: vec![],
                    nullability: NullabilityKind::NonNull,
                };

                // Generate binary operation trait implementations
                for (trait_name, method_name, op) in &binary_traits {
                    if let Some((impl_decl, func_decl)) = generate_binary_trait_impl(
                        program,
                        &abstract_type,
                        underlying_type,
                        trait_name,
                        method_name,
                        *op,
                    )? {
                        generated_impls.push(impl_decl);
                        generated_impls.push(func_decl);
                    }
                }

                // Generate comparison trait implementations
                for (trait_name, methods) in &comparison_traits {
                    if let Some((impl_decl, func_decls)) = generate_comparison_trait_impl(
                        program,
                        &abstract_type,
                        underlying_type,
                        trait_name,
                        methods,
                    )? {
                        generated_impls.push(impl_decl);
                        generated_impls.extend(func_decls);
                    }
                }

                // Generate unary operation trait implementations
                for (trait_name, method_name, op) in &unary_traits {
                    if let Some((impl_decl, func_decl)) = generate_unary_trait_impl(
                        program,
                        &abstract_type,
                        underlying_type,
                        trait_name,
                        method_name,
                        *op,
                    )? {
                        generated_impls.push(impl_decl);
                        generated_impls.push(func_decl);
                    }
                }
            }
        }
    }

    // Add generated implementations to the program
    let count = generated_impls.len();
    program.declarations.extend(generated_impls);

    Ok(())
}

/// Generate a binary operation trait implementation (Add, Sub, Mul, Div, Mod)
/// Returns a tuple of (impl_block, function_decl) where function_decl is the standalone function
fn generate_binary_trait_impl(
    program: &mut zyntax_typed_ast::TypedProgram,
    abstract_type: &zyntax_typed_ast::Type,
    underlying_type: &zyntax_typed_ast::Type,
    trait_name: &str,
    method_name: &str,
    op: zyntax_typed_ast::BinaryOp,
) -> Result<
    Option<(
        zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>,
        zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>,
    )>,
    CompilerError,
> {
    use zyntax_typed_ast::arena::InternedString;
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::Mutability;
    use zyntax_typed_ast::{
        ParameterKind, TypedBinary, TypedBlock, TypedDeclaration, TypedExpression, TypedFunction,
        TypedImplAssociatedType, TypedMethod, TypedNode, TypedParameter, TypedStatement,
        TypedTraitImpl,
    };

    let trait_name_interned = InternedString::new_global(trait_name);

    // Check if trait exists
    if program
        .type_registry
        .get_trait_by_name(trait_name_interned)
        .is_none()
    {
        return Ok(None);
    }

    let method_name_interned = InternedString::new_global(method_name);
    let self_param = InternedString::new_global("self");
    let rhs_param = InternedString::new_global("rhs");

    // Create method: fn {method_name}(self, rhs: Self) -> Self { TypeName { value: self.value {op} rhs.value } }
    use zyntax_typed_ast::TypedMethodParam;

    // Get type name for struct literal
    let type_name_for_literal = if let zyntax_typed_ast::Type::Named { id, .. } = abstract_type {
        program
            .type_registry
            .get_type_by_id(*id)
            .map(|t| t.name)
            .unwrap_or_else(|| InternedString::new_global("Unknown"))
    } else {
        InternedString::new_global("Unknown")
    };

    let method = TypedMethod {
        name: method_name_interned,
        type_params: vec![],
        params: vec![
            TypedMethodParam {
                name: self_param,
                ty: abstract_type.clone(),
                mutability: Mutability::Immutable,
                is_self: true,
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
                span: Span::default(),
            },
            TypedMethodParam {
                name: rhs_param,
                ty: abstract_type.clone(),
                mutability: Mutability::Immutable,
                is_self: false,
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
                span: Span::default(),
            },
        ],
        return_type: abstract_type.clone(),
        body: Some(TypedBlock {
            statements: vec![TypedNode::new(
                TypedStatement::Expression(Box::new(TypedNode::new(
                    // Create struct literal: TypeName { value: self.value {op} rhs.value }
                    TypedExpression::Struct(zyntax_typed_ast::TypedStructLiteral {
                        name: type_name_for_literal,
                        fields: vec![zyntax_typed_ast::TypedFieldInit {
                            name: InternedString::new_global("value"),
                            value: Box::new(TypedNode::new(
                                TypedExpression::Binary(TypedBinary {
                                    // Extract self.value
                                    left: Box::new(TypedNode::new(
                                        TypedExpression::Field(
                                            zyntax_typed_ast::TypedFieldAccess {
                                                object: Box::new(TypedNode::new(
                                                    TypedExpression::Variable(self_param),
                                                    abstract_type.clone(),
                                                    Span::default(),
                                                )),
                                                field: InternedString::new_global("value"),
                                            },
                                        ),
                                        underlying_type.clone(),
                                        Span::default(),
                                    )),
                                    op,
                                    // Extract rhs.value
                                    right: Box::new(TypedNode::new(
                                        TypedExpression::Field(
                                            zyntax_typed_ast::TypedFieldAccess {
                                                object: Box::new(TypedNode::new(
                                                    TypedExpression::Variable(rhs_param),
                                                    abstract_type.clone(),
                                                    Span::default(),
                                                )),
                                                field: InternedString::new_global("value"),
                                            },
                                        ),
                                        underlying_type.clone(),
                                        Span::default(),
                                    )),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                        }],
                    }),
                    abstract_type.clone(),
                    Span::default(),
                ))),
                zyntax_typed_ast::Type::Any,
                Span::default(),
            )],
            span: Span::default(),
        }),
        visibility: zyntax_typed_ast::type_registry::Visibility::Public,
        is_static: false,
        is_async: false,
        is_override: false,
        span: Span::default(),
    };

    // Create impl block
    let impl_block = TypedTraitImpl {
        trait_name: trait_name_interned,
        trait_type_args: vec![abstract_type.clone()], // impl Add<Self> for Self
        for_type: abstract_type.clone(),
        methods: vec![method],
        associated_types: vec![TypedImplAssociatedType {
            name: InternedString::new_global("Output"),
            ty: abstract_type.clone(),
            span: Span::default(),
        }],
        span: Span::default(),
    };

    let impl_decl = TypedNode::new(
        TypedDeclaration::Impl(impl_block),
        zyntax_typed_ast::Type::Any,
        Span::default(),
    );

    // Create standalone function with mangled name: {TypeName}${method}
    // Note: Operator overloading uses 2-part names (TypeName$method) not 3-part (TypeName$Trait$method)
    // This matches the naming convention in ssa.rs:try_operator_trait_dispatch()
    let type_name_str = if let zyntax_typed_ast::Type::Named { id, .. } = abstract_type {
        program
            .type_registry
            .get_type_by_id(*id)
            .map(|t| t.name.resolve_global().unwrap_or_default())
            .unwrap_or_default()
    } else {
        "UnknownType".to_string()
    };

    let mangled_name = format!("{}${}", type_name_str, method_name);
    let mangled_name_interned = InternedString::new_global(&mangled_name);

    let standalone_func = TypedFunction {
        name: mangled_name_interned,
        annotations: vec![],
        effects: vec![],
        type_params: vec![],
        params: vec![
            TypedParameter {
                name: self_param,
                ty: abstract_type.clone(),
                mutability: Mutability::Immutable,
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
                span: Span::default(),
            },
            TypedParameter {
                name: rhs_param,
                ty: abstract_type.clone(),
                mutability: Mutability::Immutable,
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
                span: Span::default(),
            },
        ],
        return_type: abstract_type.clone(),
        body: Some(TypedBlock {
            statements: vec![TypedNode::new(
                TypedStatement::Expression(Box::new(TypedNode::new(
                    // Create struct literal: TypeName { value: self.value {op} rhs.value }
                    TypedExpression::Struct(zyntax_typed_ast::TypedStructLiteral {
                        name: type_name_for_literal,
                        fields: vec![zyntax_typed_ast::TypedFieldInit {
                            name: InternedString::new_global("value"),
                            value: Box::new(TypedNode::new(
                                TypedExpression::Binary(TypedBinary {
                                    // Extract self.value
                                    left: Box::new(TypedNode::new(
                                        TypedExpression::Field(
                                            zyntax_typed_ast::TypedFieldAccess {
                                                object: Box::new(TypedNode::new(
                                                    TypedExpression::Variable(self_param),
                                                    abstract_type.clone(),
                                                    Span::default(),
                                                )),
                                                field: InternedString::new_global("value"),
                                            },
                                        ),
                                        underlying_type.clone(),
                                        Span::default(),
                                    )),
                                    op,
                                    // Extract rhs.value
                                    right: Box::new(TypedNode::new(
                                        TypedExpression::Field(
                                            zyntax_typed_ast::TypedFieldAccess {
                                                object: Box::new(TypedNode::new(
                                                    TypedExpression::Variable(rhs_param),
                                                    abstract_type.clone(),
                                                    Span::default(),
                                                )),
                                                field: InternedString::new_global("value"),
                                            },
                                        ),
                                        underlying_type.clone(),
                                        Span::default(),
                                    )),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                        }],
                    }),
                    abstract_type.clone(),
                    Span::default(),
                ))),
                zyntax_typed_ast::Type::Any,
                Span::default(),
            )],
            span: Span::default(),
        }),
        visibility: zyntax_typed_ast::type_registry::Visibility::Public,
        is_async: false,
        is_pure: false,
        is_external: false,
        calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
        link_name: None,
    };

    let func_decl = TypedNode::new(
        TypedDeclaration::Function(standalone_func),
        zyntax_typed_ast::Type::Any,
        Span::default(),
    );

    Ok(Some((impl_decl, func_decl)))
}

/// Generate a comparison trait implementation (Eq, Ord)
/// Returns (impl_block, vec of standalone functions)
fn generate_comparison_trait_impl(
    program: &mut zyntax_typed_ast::TypedProgram,
    abstract_type: &zyntax_typed_ast::Type,
    underlying_type: &zyntax_typed_ast::Type,
    trait_name: &str,
    methods: &[(&str, zyntax_typed_ast::BinaryOp)],
) -> Result<
    Option<(
        zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>,
        Vec<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>>,
    )>,
    CompilerError,
> {
    use zyntax_typed_ast::arena::InternedString;
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::{
        CallingConvention, Mutability, PrimitiveType, Visibility,
    };
    use zyntax_typed_ast::{
        ParameterKind, TypedBinary, TypedBlock, TypedDeclaration, TypedExpression, TypedFunction,
        TypedMethod, TypedMethodParam, TypedNode, TypedParameter, TypedStatement, TypedTraitImpl,
    };

    let trait_name_interned = InternedString::new_global(trait_name);

    // Check if trait exists
    if program
        .type_registry
        .get_trait_by_name(trait_name_interned)
        .is_none()
    {
        return Ok(None);
    }

    let self_param = InternedString::new_global("self");
    let rhs_param = InternedString::new_global("rhs");

    // Get type name for mangling
    let type_name = if let zyntax_typed_ast::Type::Named { id, .. } = abstract_type {
        program
            .type_registry
            .get_type_by_id(*id)
            .ok_or_else(|| CompilerError::Analysis("Type not found in registry".to_string()))?
            .name
    } else {
        return Err(CompilerError::Analysis(
            "Abstract type must be Named".to_string(),
        ));
    };
    let type_name_str = type_name.resolve_global().unwrap_or_default();

    // Generate impl block methods and standalone functions
    let mut typed_methods = Vec::new();
    let mut standalone_functions = Vec::new();

    for (method_name, op) in methods {
        let method_name_interned = InternedString::new_global(method_name);

        // Generate TypedMethod for impl block
        typed_methods.push(TypedMethod {
            name: method_name_interned,
            type_params: vec![],
            params: vec![
                TypedMethodParam {
                    name: self_param,
                    ty: abstract_type.clone(),
                    mutability: Mutability::Immutable,
                    is_self: true,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::default(),
                },
                TypedMethodParam {
                    name: rhs_param,
                    ty: abstract_type.clone(),
                    mutability: Mutability::Immutable,
                    is_self: false,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::default(),
                },
            ],
            return_type: zyntax_typed_ast::Type::Primitive(PrimitiveType::Bool),
            body: Some(TypedBlock {
                statements: vec![TypedNode::new(
                    TypedStatement::Expression(Box::new(TypedNode::new(
                        TypedExpression::Binary(TypedBinary {
                            // Extract self.value
                            left: Box::new(TypedNode::new(
                                TypedExpression::Field(zyntax_typed_ast::TypedFieldAccess {
                                    object: Box::new(TypedNode::new(
                                        TypedExpression::Variable(self_param),
                                        abstract_type.clone(),
                                        Span::default(),
                                    )),
                                    field: InternedString::new_global("value"),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                            op: *op,
                            // Extract rhs.value
                            right: Box::new(TypedNode::new(
                                TypedExpression::Field(zyntax_typed_ast::TypedFieldAccess {
                                    object: Box::new(TypedNode::new(
                                        TypedExpression::Variable(rhs_param),
                                        abstract_type.clone(),
                                        Span::default(),
                                    )),
                                    field: InternedString::new_global("value"),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                        }),
                        zyntax_typed_ast::Type::Primitive(PrimitiveType::Bool),
                        Span::default(),
                    ))),
                    zyntax_typed_ast::Type::Any,
                    Span::default(),
                )],
                span: Span::default(),
            }),
            visibility: Visibility::Public,
            is_static: false,
            is_async: false,
            is_override: false,
            span: Span::default(),
        });

        // Generate standalone function with 2-part mangled name (TypeName$method)
        let mangled_name = format!("{}${}", type_name_str, method_name);
        let mangled_name_interned = InternedString::new_global(&mangled_name);

        let standalone_func = TypedFunction {
            name: mangled_name_interned,
            annotations: vec![],
            effects: vec![],
            type_params: vec![],
            params: vec![
                TypedParameter {
                    name: self_param,
                    ty: abstract_type.clone(),
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::default(),
                },
                TypedParameter {
                    name: rhs_param,
                    ty: abstract_type.clone(),
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::default(),
                },
            ],
            return_type: zyntax_typed_ast::Type::Primitive(PrimitiveType::Bool),
            body: Some(TypedBlock {
                statements: vec![TypedNode::new(
                    TypedStatement::Expression(Box::new(TypedNode::new(
                        TypedExpression::Binary(TypedBinary {
                            // Extract self.value
                            left: Box::new(TypedNode::new(
                                TypedExpression::Field(zyntax_typed_ast::TypedFieldAccess {
                                    object: Box::new(TypedNode::new(
                                        TypedExpression::Variable(self_param),
                                        abstract_type.clone(),
                                        Span::default(),
                                    )),
                                    field: InternedString::new_global("value"),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                            op: *op,
                            // Extract rhs.value
                            right: Box::new(TypedNode::new(
                                TypedExpression::Field(zyntax_typed_ast::TypedFieldAccess {
                                    object: Box::new(TypedNode::new(
                                        TypedExpression::Variable(rhs_param),
                                        abstract_type.clone(),
                                        Span::default(),
                                    )),
                                    field: InternedString::new_global("value"),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                        }),
                        zyntax_typed_ast::Type::Primitive(PrimitiveType::Bool),
                        Span::default(),
                    ))),
                    zyntax_typed_ast::Type::Any,
                    Span::default(),
                )],
                span: Span::default(),
            }),
            visibility: Visibility::Public,
            is_async: false,
            is_pure: false,
            is_external: false,
            calling_convention: CallingConvention::Default,
            link_name: None,
        };

        standalone_functions.push(TypedNode::new(
            TypedDeclaration::Function(standalone_func),
            zyntax_typed_ast::Type::Any,
            Span::default(),
        ));
    }

    // Create impl block
    let impl_block = TypedTraitImpl {
        trait_name: trait_name_interned,
        trait_type_args: vec![abstract_type.clone()], // impl Eq<Self> for Self
        for_type: abstract_type.clone(),
        methods: typed_methods,
        associated_types: vec![],
        span: Span::default(),
    };

    let impl_decl = TypedNode::new(
        TypedDeclaration::Impl(impl_block),
        zyntax_typed_ast::Type::Any,
        Span::default(),
    );

    Ok(Some((impl_decl, standalone_functions)))
}

/// Generate a unary operation trait implementation (Neg)
/// Returns (impl_block, standalone_function)
fn generate_unary_trait_impl(
    program: &mut zyntax_typed_ast::TypedProgram,
    abstract_type: &zyntax_typed_ast::Type,
    underlying_type: &zyntax_typed_ast::Type,
    trait_name: &str,
    method_name: &str,
    op: zyntax_typed_ast::UnaryOp,
) -> Result<
    Option<(
        zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>,
        zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>,
    )>,
    CompilerError,
> {
    use zyntax_typed_ast::arena::InternedString;
    use zyntax_typed_ast::source::Span;
    use zyntax_typed_ast::type_registry::{CallingConvention, Mutability, Visibility};
    use zyntax_typed_ast::{
        ParameterKind, TypedBlock, TypedDeclaration, TypedExpression, TypedFunction,
        TypedImplAssociatedType, TypedMethod, TypedMethodParam, TypedNode, TypedParameter,
        TypedStatement, TypedTraitImpl, TypedUnary,
    };

    let trait_name_interned = InternedString::new_global(trait_name);

    // Check if trait exists
    if program
        .type_registry
        .get_trait_by_name(trait_name_interned)
        .is_none()
    {
        return Ok(None);
    }

    let method_name_interned = InternedString::new_global(method_name);
    let self_param = InternedString::new_global("self");

    // Get type name for mangling and struct literal construction
    let type_name = if let zyntax_typed_ast::Type::Named { id, .. } = abstract_type {
        program
            .type_registry
            .get_type_by_id(*id)
            .ok_or_else(|| CompilerError::Analysis("Type not found in registry".to_string()))?
            .name
    } else {
        return Err(CompilerError::Analysis(
            "Abstract type must be Named".to_string(),
        ));
    };
    let type_name_str = type_name.resolve_global().unwrap_or_default();

    // Create method: fn {method_name}(self) -> Self { TypeName { value: {op}self.value } }
    let method = TypedMethod {
        name: method_name_interned,
        type_params: vec![],
        params: vec![TypedMethodParam {
            name: self_param,
            ty: abstract_type.clone(),
            mutability: Mutability::Immutable,
            is_self: true,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span: Span::default(),
        }],
        return_type: abstract_type.clone(),
        body: Some(TypedBlock {
            statements: vec![TypedNode::new(
                TypedStatement::Expression(Box::new(TypedNode::new(
                    // Create struct literal: TypeName { value: {op}self.value }
                    TypedExpression::Struct(zyntax_typed_ast::TypedStructLiteral {
                        name: type_name,
                        fields: vec![zyntax_typed_ast::TypedFieldInit {
                            name: InternedString::new_global("value"),
                            value: Box::new(TypedNode::new(
                                TypedExpression::Unary(TypedUnary {
                                    op,
                                    operand: Box::new(TypedNode::new(
                                        // Extract self.value
                                        TypedExpression::Field(
                                            zyntax_typed_ast::TypedFieldAccess {
                                                object: Box::new(TypedNode::new(
                                                    TypedExpression::Variable(self_param),
                                                    abstract_type.clone(),
                                                    Span::default(),
                                                )),
                                                field: InternedString::new_global("value"),
                                            },
                                        ),
                                        underlying_type.clone(),
                                        Span::default(),
                                    )),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                        }],
                    }),
                    abstract_type.clone(),
                    Span::default(),
                ))),
                zyntax_typed_ast::Type::Any,
                Span::default(),
            )],
            span: Span::default(),
        }),
        visibility: Visibility::Public,
        is_static: false,
        is_async: false,
        is_override: false,
        span: Span::default(),
    };

    // Create impl block
    let impl_block = TypedTraitImpl {
        trait_name: trait_name_interned,
        trait_type_args: vec![],
        for_type: abstract_type.clone(),
        methods: vec![method],
        associated_types: vec![TypedImplAssociatedType {
            name: InternedString::new_global("Output"),
            ty: abstract_type.clone(),
            span: Span::default(),
        }],
        span: Span::default(),
    };

    let impl_decl = TypedNode::new(
        TypedDeclaration::Impl(impl_block),
        zyntax_typed_ast::Type::Any,
        Span::default(),
    );

    // Generate standalone function with 2-part mangled name (TypeName$method)
    let mangled_name = format!("{}${}", type_name_str, method_name);
    let mangled_name_interned = InternedString::new_global(&mangled_name);

    let standalone_func = TypedFunction {
        name: mangled_name_interned,
        annotations: vec![],
        effects: vec![],
        type_params: vec![],
        params: vec![TypedParameter {
            name: self_param,
            ty: abstract_type.clone(),
            mutability: Mutability::Immutable,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span: Span::default(),
        }],
        return_type: abstract_type.clone(),
        body: Some(TypedBlock {
            statements: vec![TypedNode::new(
                TypedStatement::Expression(Box::new(TypedNode::new(
                    // Create struct literal: TypeName { value: {op}self.value }
                    TypedExpression::Struct(zyntax_typed_ast::TypedStructLiteral {
                        name: type_name,
                        fields: vec![zyntax_typed_ast::TypedFieldInit {
                            name: InternedString::new_global("value"),
                            value: Box::new(TypedNode::new(
                                TypedExpression::Unary(TypedUnary {
                                    op,
                                    operand: Box::new(TypedNode::new(
                                        // Extract self.value
                                        TypedExpression::Field(
                                            zyntax_typed_ast::TypedFieldAccess {
                                                object: Box::new(TypedNode::new(
                                                    TypedExpression::Variable(self_param),
                                                    abstract_type.clone(),
                                                    Span::default(),
                                                )),
                                                field: InternedString::new_global("value"),
                                            },
                                        ),
                                        underlying_type.clone(),
                                        Span::default(),
                                    )),
                                }),
                                underlying_type.clone(),
                                Span::default(),
                            )),
                        }],
                    }),
                    abstract_type.clone(),
                    Span::default(),
                ))),
                zyntax_typed_ast::Type::Any,
                Span::default(),
            )],
            span: Span::default(),
        }),
        visibility: Visibility::Public,
        is_async: false,
        is_pure: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
    };

    let func_decl = TypedNode::new(
        TypedDeclaration::Function(standalone_func),
        zyntax_typed_ast::Type::Any,
        Span::default(),
    );

    debug!(
        "[AUTO_TRAITS] Generated {} impl for {:?}",
        trait_name, abstract_type
    );
    debug!(
        "[AUTO_TRAITS] Generated standalone function '{}' for {} trait",
        mangled_name, trait_name
    );

    Ok(Some((impl_decl, func_decl)))
}

/// 1. Lower TypedAST to HIR (with generic definitions intact)
/// 2. Monomorphize generic functions and types (if enabled)
/// 3. Run analysis passes (liveness, alias analysis, etc.)
/// 4. Run optimization passes
/// 5. Return optimized HIR ready for backend code generation
pub fn compile_to_hir(
    program: &mut TypedProgram,
    type_registry: Arc<zyntax_typed_ast::TypeRegistry>,
    config: CompilationConfig,
) -> CompilerResult<HirModule> {
    // Step 1: Lower TypedAST → HIR
    let lowering_config = lowering::LoweringConfig {
        debug_info: config.debug_info,
        opt_level: config.opt_level,
        target_triple: config.target_triple.clone(),
        hot_reload: config.hot_reload,
        strict_mode: false, // Default to non-strict
        import_resolver: config.import_resolver.clone(),
        builtins: indexmap::IndexMap::new(), // Empty - callers with grammar should use runtime directly
    };

    // Create arena for string interning (needed for async transformation)
    let mut arena = zyntax_typed_ast::arena::AstArena::new();

    // Extract module name from program or use default
    let module_name = arena.intern_string("main");

    let arena_arc = Arc::new(Mutex::new(arena));
    let mut lowering_ctx = lowering::LoweringContext::new(
        module_name,
        type_registry.clone(),
        arena_arc,
        lowering_config,
    );
    let mut hir_module = lowering_ctx.lower_program(program)?;

    // Step 2: Async transformation (if async runtime is configured)
    if let Some(runtime_type) = config.async_runtime {
        // Transform async functions into state machines
        let async_func_ids: Vec<HirId> = hir_module
            .functions
            .values()
            .filter(|f| f.signature.is_async)
            .map(|f| f.id)
            .collect();

        if !async_func_ids.is_empty() {
            let mut async_compiler = async_support::AsyncCompiler::new();

            // Create async runtime with proper HIR IDs
            let runtime = async_support::AsyncRuntime {
                runtime_type,
                executor: HirId::new(),
                future_trait: HirId::new(),
                waker_type: HirId::new(),
            };
            async_compiler.set_runtime(runtime);

            // Transform each async function
            for func_id in async_func_ids {
                let original_func = hir_module
                    .functions
                    .get(&func_id)
                    .ok_or_else(|| CompilerError::Backend("Async function not found".into()))?
                    .clone();

                // Compile to state machine
                let state_machine = async_compiler.compile_async_function(&original_func)?;

                // Generate poll function (internal implementation)
                let poll_func = async_compiler.generate_poll_function(
                    &state_machine,
                    &mut *lowering_ctx.arena.lock().unwrap(),
                    &original_func,
                )?;

                // Generate async entry function that returns Promise
                let entry_func = async_compiler.generate_async_entry_function(
                    &state_machine,
                    &poll_func,
                    &original_func,
                    &mut *lowering_ctx.arena.lock().unwrap(),
                )?;

                // Add poll function to module (internal implementation detail)
                hir_module.functions.insert(poll_func.id, poll_func);

                // Replace the original async function with the entry function that returns Promise
                hir_module.functions.insert(func_id, entry_func);
            }
        }
    }

    // Step 3: Monomorphization (if enabled and generic functions exist)
    if config.enable_monomorphization {
        let has_generics = hir_module
            .functions
            .values()
            .any(|f| !f.signature.type_params.is_empty() || !f.signature.const_params.is_empty());

        if has_generics {
            monomorphize_module(&mut hir_module)?;
        }
    }

    // Step 3: Analysis passes
    let mut analysis_runner = analysis::AnalysisRunner::new(hir_module.clone());
    let analysis = analysis_runner.run_all()?;

    // Step 3b: HIR borrow checking (if enabled)
    if config.enable_borrow_check {
        let borrow_result = borrow_check::run_borrow_check(&hir_module, Some(&analysis))?;
        borrow_check::validate_borrow_check(&borrow_result)?;
    }

    // Step 3c: Effect analysis and checking (if enabled)
    if config.enable_effect_check {
        let effect_analysis =
            effect_analysis::analyze_effects_with_call_graph(&hir_module, &analysis.call_graph)?;

        // Report errors if any effect constraints are violated
        if effect_analysis::has_effect_errors(&effect_analysis) {
            let error_messages: Vec<String> = effect_analysis
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect();
            return Err(CompilerError::Analysis(format!(
                "Effect checking failed:\n  - {}",
                error_messages.join("\n  - ")
            )));
        }

        // Log warnings (but don't fail compilation)
        for warning in &effect_analysis.warnings {
            debug!("[EFFECT_CHECK] Warning: {}", warning.message);
        }
    }

    // Step 4: Memory management pass (if strategy is configured)
    if let Some(strategy) = config.memory_strategy {
        let mut memory_pass = memory_pass::MemoryManagementPass::new(strategy);
        memory_pass.run(&mut hir_module, &analysis)?;
    }

    // Step 5: Optimization passes (if opt_level > 0)
    if config.opt_level > 0 {
        let opt_level = match config.opt_level {
            0 => optimization::OptLevel::None,
            1 => optimization::OptLevel::Less,
            2 => optimization::OptLevel::Default,
            _ => optimization::OptLevel::More,
        };

        let mut opt_pipeline = optimization::OptimizationPipeline::new(opt_level);
        opt_pipeline.run(&mut hir_module, &analysis)?;

        // Step 5b: Memory-specific optimizations (if memory strategy is enabled)
        if config.memory_strategy.is_some() {
            let mut memory_opt = memory_optimization::MemoryOptimizationPass::new();
            memory_opt.run(&mut hir_module, &analysis)?;
        }
    }

    Ok(hir_module)
}

/// Compile a TypedProgram to JIT executable code using Cranelift backend
///
/// This is a convenience function that:
/// 1. Compiles TypedAST to HIR (via compile_to_hir)
/// 2. Generates native code using Cranelift JIT
/// 3. Returns a Cranelift backend ready to execute functions
#[cfg(feature = "cranelift")]
pub fn compile_to_jit(
    program: &mut TypedProgram,
    type_registry: Arc<zyntax_typed_ast::TypeRegistry>,
    config: CompilationConfig,
) -> CompilerResult<cranelift_backend::CraneliftBackend> {
    // Compile to HIR
    let hir_module = compile_to_hir(program, type_registry, config)?;

    // Generate native code with Cranelift
    let mut backend = cranelift_backend::CraneliftBackend::new()
        .map_err(|e| CompilerError::Backend(format!("Failed to initialize Cranelift: {}", e)))?;

    backend
        .compile_module(&hir_module)
        .map_err(|e| CompilerError::CodeGen(format!("Cranelift compilation failed: {}", e)))?;

    Ok(backend)
}
