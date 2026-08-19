#![allow(unused, dead_code, deprecated)]

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

pub mod affine_loop; // Closed-form affine reduction loops (acc += invariant over a counted loop)
pub mod aggregate_split; // Replace struct round-trips with direct field Load/Store (HIR-level SROA)
pub mod alloca_promote; // Alloca → Malloc promotion for escaping allocations (pairs with drop_insert)
pub mod analysis;
pub mod associated_type_resolver; // Associated type resolution for trait dispatch
pub mod async_support;
pub mod auto_vectorize;
pub mod borrow_check; // HIR-level borrow checking pass
pub mod builtin_class; // Wrapper-class dispatch for compiler-known built-in types (Fiber, future SimdVector, etc.)
pub mod bytecode; // HIR bytecode serialization/deserialization
pub mod cast_classify; // Pure classification of source/target coercions → CastKind
pub mod cfg;
pub mod cfg_simplify;
pub mod const_eval;
pub mod const_fold;
pub mod cse;
pub mod dce; // Reachability-based dead-code elimination at the function level
pub mod drop_insert; // Speculative drop-site analysis: insert free() for non-escaping mallocs
pub mod effect_analysis; // Effect inference and checking for algebraic effects
pub mod effect_codegen; // Code generation support for algebraic effects
pub mod effect_handler_resolution; // Handler resolution for effect dispatch
pub mod fiber_backend; // `FiberCfg` trait + global install slot for fiber primitives
pub mod fiber_lowering; // First-class fiber HIR ops → Call::Symbol("krio_fiber_*") rewrite
pub mod fma_contract; // FMA contraction: rewrite fadd(fmul a b, c) → fma(a, b, c)
pub mod hir;
pub mod hir_builder; // HIR Builder API for direct HIR construction
pub mod hir_dump; // CLIF-inspired HIR text dump for debugging
pub mod hir_interp; // HIR bytecode interpreter (universal Tier 0; always available)
pub mod inline;
pub mod licm;
pub mod load_cse;
pub mod loop_vectorize;
pub mod lowering;
pub mod memory_management;
pub mod memory_optimization; // Memory-aware optimizations
pub mod memory_pass;
pub mod monomorphize;
pub mod move_insert; // Owning parameters become `Move` the borrow check can see
pub mod optimization;
pub mod parallel_safe; // Which counted loops have independent iterations
pub mod pattern_matching;
pub mod phi_prune;
pub mod pure_call_pre;
pub mod purity;
pub mod reduction_vectorize;
pub(crate) mod return_infer; // Return types for declarations that don't state one
pub mod runtime;
pub mod scalar_replace_alloc; // Eliminate non-escaping Call(Intrinsic::Malloc) allocations (heap SROA)
pub mod ssa;
pub mod stdlib; // Standard library implementation using HIR Builder
pub mod target_vector; // How wide a vector the target accepts, and lanes per element
pub mod tco; // Tail-call optimisation marker
pub mod trait_lowering; // Trait/interface lowering to HIR
pub mod typed_cfg; // New: TypedAST-aware CFG builder
pub mod value; // Unified runtime value type (used by interp + embed)
pub mod vtable_registry; // Vtable management and caching // Async runtime (executor, task, waker)

#[cfg(feature = "cranelift-backend")]
pub mod cranelift_backend;

#[cfg(feature = "llvm-backend")]
pub mod llvm_backend;

#[cfg(feature = "llvm-backend")]
pub mod llvm_jit_backend; // LLVM MCJIT for hot-path optimization

#[cfg(feature = "llvm-backend")]
pub mod llvm_link; // AOT-via-object: trampolines + system linker + dlopen

#[cfg(feature = "wasm-jit")]
pub mod wasm_backend; // HIR → wasm hot-function tier-1 JIT (Phase E)

#[cfg(feature = "cranelift-backend")]
pub mod beadie_adapter; // beadie::JitBackend wrappers for our backends
pub mod osr; // On-stack replacement infrastructure (probe, registry, layout)
pub mod plugin; // Plugin system for frontend runtime registration
pub mod profiling; // Runtime profiling for tiered compilation
pub mod reload; // Hot-reload fingerprints, call cells, reports
pub mod string_intrinsics; // Runtime intrinsics for string operations (e.g. string-eq for BinaryOp::Eq)
#[cfg(feature = "cranelift-backend")]
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
#[cfg(feature = "cranelift-backend")]
pub use cranelift_backend::{
    cranelift_skipped_function_count, reset_cranelift_skipped_function_count,
};
pub use dce::reachable_function_ids;
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
    /// Extern aliases to inject into `extern_link_names` before SSA
    /// lowering. Each entry maps an in-source identifier the program
    /// will reference (e.g. `__signal_set_i32`) to the registered
    /// host symbol name (e.g. `$Signal$set_i32`). Without an entry
    /// here a call to that identifier lowers as
    /// `HirCallable::Function(...)` and the wasm backend currently
    /// rejects internal-function dispatch (E.5.2+ work). Adding the
    /// alias routes the call through `HirCallable::Symbol(name)`,
    /// which the wasm backend imports as
    /// `(import "extern" "<name>@<arity>" …)`. The interpreter and
    /// Cranelift backends both treat Symbol calls as FFI lookups
    /// against the runtime's registered-symbol table, so the same
    /// declaration works across native + wasm.
    pub builtins: indexmap::IndexMap<String, String>,
    /// When `true`, the SSA-level legacy `transform_async_function`
    /// pass is skipped so a downstream `krio_adapter` post-pass owns
    /// the async → poll-fn state-machine transform. The wasm shim
    /// sets this so Phase I.2's cooperative-await rewrite reaches
    /// raw `Intrinsic::Await` calls (which the legacy transform
    /// would otherwise eat into a Promise-polling fallback before
    /// krio runs). Defaults to `false` for backwards compatibility
    /// with existing native callers; flip on when calling
    /// `apply_krio_async_lowering` afterward.
    pub use_krio_async: bool,
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
            builtins: indexmap::IndexMap::new(),
            use_krio_async: false,
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
                        // Extern types have runtime_prefix names like "$Tensor",
                        // but TypeDefinitions are registered under the bare name "Tensor"
                        let name_str = name.resolve_global().unwrap_or_default();
                        let bare_name = name_str.strip_prefix('$').unwrap_or(&name_str);
                        if let Some(type_def) = program.type_registry.get_type_by_name(*name) {
                            Some(type_def.id)
                        } else {
                            // Try with bare name (strip $ prefix)
                            let bare_interned =
                                zyntax_typed_ast::InternedString::new_global(bare_name);
                            if let Some(type_def) =
                                program.type_registry.get_type_by_name(bare_interned)
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
                                is_const: tp.is_const,
                                const_ty: tp.const_ty.clone(),
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
                                is_const: tp.is_const,
                                const_ty: tp.const_ty.clone(),
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
        annotations: vec![],
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
        module: None,
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
        with_handlers: vec![],
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
                ownership: Default::default(),
            },
            TypedParameter {
                name: rhs_param,
                ty: abstract_type.clone(),
                mutability: Mutability::Immutable,
                kind: ParameterKind::Regular,
                default_value: None,
                attributes: vec![],
                span: Span::default(),
                ownership: Default::default(),
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
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
        link_name: None,
        module: None,
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
            annotations: vec![],
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
            with_handlers: vec![],
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
                    ownership: Default::default(),
                },
                TypedParameter {
                    name: rhs_param,
                    ty: abstract_type.clone(),
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::default(),
                    ownership: Default::default(),
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
            is_fiber: false,
            is_pure: false,
            is_external: false,
            calling_convention: CallingConvention::Default,
            link_name: None,
            module: None,
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
        module: None,
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
        annotations: vec![],
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
        module: None,
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
        with_handlers: vec![],
        type_params: vec![],
        params: vec![TypedParameter {
            name: self_param,
            ty: abstract_type.clone(),
            mutability: Mutability::Immutable,
            kind: ParameterKind::Regular,
            default_value: None,
            attributes: vec![],
            span: Span::default(),
            ownership: Default::default(),
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
        is_fiber: false,
        is_pure: false,
        is_external: false,
        calling_convention: CallingConvention::Default,
        link_name: None,
        module: None,
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
        // Forward caller-supplied extern aliases. The wasm shim
        // uses this to inject test-only externs (e.g. `__zw_test_double`)
        // so source-level calls lower as `HirCallable::Symbol`,
        // which is the only callee variant the wasm backend
        // currently emits. Callers that don't need any aliases
        // (the default) pass an empty map.
        builtins: config.builtins.clone(),
        use_krio_async: config.use_krio_async,
        entry_names: Vec::new(),
    };

    // Create arena for string interning (needed for async transformation)
    let mut arena = zyntax_typed_ast::arena::AstArena::new();

    // Extract module name from program or use default
    let module_name = arena.intern_string("main");

    // Run pattern engine (term-rewriting passes on TypedAST)
    {
        let mut engine = pattern_engine::PatternEngine::new(pattern_engine::EngineConfig {
            target: pattern_engine::LoweringTarget::Cpu,
            max_iterations: 64,
            trace: cfg!(debug_assertions),
            verify_after: false,
        });
        engine.register_pass(normalization_pass::Pass);
        engine.register_pass(algebraic_effects_pass::Pass);
        engine.finalize().map_err(|e| {
            CompilerError::Analysis(format!("Pattern engine finalize error: {}", e))
        })?;
        let result = engine.run(program, &type_registry);
        if result.changed {
            log::debug!(
                "[pattern_engine] {} rewrites fired in {} iterations",
                result.rewrites_fired.len(),
                result.iterations
            );
        }
    }

    let arena_arc = Arc::new(Mutex::new(arena));
    let mut lowering_ctx = lowering::LoweringContext::new(
        module_name,
        type_registry.clone(),
        arena_arc,
        lowering_config,
    );
    let mut hir_module = lowering_ctx.lower_program(program)?;

    // Display any lowering diagnostics (warnings/errors) using the proper formatter
    lowering_ctx.display_diagnostics(program);

    // Step 2: Async transformation (if async runtime is configured)
    // Skipped when `use_krio_async` is set so a downstream krio post-pass
    // can do the captures-lift transform instead. Mirrors the parallel
    // skip at `lowering.rs::lower_function` (see line ~2134).
    if let Some(runtime_type) = config.async_runtime.filter(|_| !config.use_krio_async) {
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
        // Through `check_ownership`, not the raw check: the moves an
        // owning parameter implies have to exist before anything can be
        // found, and they are made on a copy so what gets compiled is
        // unchanged.
        let borrow_result = borrow_check::check_ownership(&hir_module)?;
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

/// Counters from `run_interp_safe_opts`. Useful for tests + the
/// `--print-opt-stats` style debugging surfaces.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InterpOptStats {
    pub alloca_promote: alloca_promote::PromoteStats,
    pub const_fold: const_fold::FoldStats,
    pub cse: cse::CseStats,
    pub fma_contract: fma_contract::FmaStats,
    pub load_cse: load_cse::LoadCseStats,
    pub aggregate_split: aggregate_split::AggregateSplitStats,
    pub scalar_replace_alloc: scalar_replace_alloc::ScalarReplaceAllocStats,
    pub licm: licm::LicmStats,
    pub affine_loop: affine_loop::AffineLoopStats,
    pub inline: inline::InlineStats,
    pub loop_vectorize: loop_vectorize::VectorizeStats,
    pub reduction_vectorize: reduction_vectorize::ReductionStats,
    pub auto_vectorize: auto_vectorize::AutoVectorizeStats,
    pub cfg_simplify: cfg_simplify::CfgSimplifyStats,
    pub drop_insert: drop_insert::DropStats,
    pub tco: tco::TcoStats,
    pub recursive_inline: inline::RecursiveInlineStats,
    pub pure_call_pre: pure_call_pre::PureCallPreStats,
    pub phi_prune: phi_prune::PhiPruneStats,
}

/// Run the subset of HIR optimization passes that are safe for the
/// BC interpreter — every pass that only touches SSA value `kind` in
/// place or substitutes uses through a sweep. Skips the legacy
/// `DeadCodeElimination` pass (which reads `HirValue.uses` directly
/// and over-eliminates when the lowering doesn't populate that
/// def-use chain, leaving dangling refs) and the legacy
/// `SimplifyCfg` (half-done, finds blocks but doesn't merge them).
///
/// Passes run in a sensible order:
///   1. const_fold — gives downstream passes a tighter SSA to work
///                   with (more invariant operands, more CSE keys)
///   2. cse        — fold-then-cse exposes more redundancy than the
///                   other way around
///   3. inline     — leaf inlines expand the per-function instruction
///                   pool that the post-pass const_fold + cse can chew
///   4. licm       — wants stable SSA; runs after inlining
///   5. loop_vectorize — pattern matcher prefers a clean inner loop
///   6. const_fold (again) — catch any new constants exposed by the
///                            previous passes (inline often does)
///   7. cse (again)         — and the dedup that follows
///
/// All passes are individually fixed-point already; the outer order
/// is a single sweep — we don't iterate the *order* itself because
/// the rounds-2 pair already mops up the easy carryover. Anything
/// more elaborate would need real cost-modelled scheduling.
///
/// Returns per-pass counters so callers can log / assert what work
/// was done.
/// Compile a module to LLVM IR text, owning the context for the call.
///
/// The backend borrows a context the caller has to create, which means
/// naming `inkwell` to inspect what was generated. This exists so a
/// caller that only wants to read the IR does not take that dependency.
#[cfg(feature = "llvm-backend")]
pub fn compile_module_to_llvm_ir(module: &HirModule, name: &str) -> CompilerResult<String> {
    let context = inkwell::context::Context::create();
    let mut backend = llvm_backend::LLVMBackend::new(&context, name);
    backend.compile_module(module)
}

pub fn run_interp_safe_opts(module: &mut HirModule) -> InterpOptStats {
    let mut stats = InterpOptStats::default();

    // Alloca → Malloc promotion runs ONCE up front, before the
    // fixed-point sweep. Two reasons it goes here:
    //   * Promoting an escaping Alloca to a `Call(Intrinsic::Malloc)`
    //     introduces a new call boundary that CSE / LICM treat as a
    //     memory barrier. Doing it before the fixed-point means
    //     subsequent passes see the final allocation shape and can
    //     reason about it consistently (rather than re-deriving
    //     across each round).
    //   * The promotion is idempotent — a single pass identifies
    //     every escaping Alloca and rewrites it. Running again would
    //     find zero work; no benefit to iterating.
    let ap = alloca_promote::run_module(module);
    stats.alloca_promote.allocas_scanned += ap.allocas_scanned;
    stats.alloca_promote.promoted += ap.promoted;
    stats.alloca_promote.kept_on_stack += ap.kept_on_stack;
    stats.alloca_promote.skipped_unsupported += ap.skipped_unsupported;

    // Purity inference feeds the pure-call CSE inside the sweep: with
    // `signature.is_pure` populated, `cse::eliminate_module` can collapse
    // two identical calls to an effect-free function into one. Runs once
    // up front; it's re-run after recursive inlining (below), which
    // rewrites bodies and can expose freshly-shared pure calls.
    purity::infer_module(module);

    // Outer fixed-point: keeps iterating the whole sweep until none
    // of the passes report new work. Compounding example: inline
    // exposes new constant operands → const_fold creates a Constant
    // → cse can now collapse a now-equivalent pair → that may free
    // another callee for inlining (one call's args becoming
    // syntactically identical to another's). Capped at 8 outer
    // rounds — well above what real source needs but a guard against
    // pathological pass-feedback loops.
    //
    // Wall-clock circuit breaker: if any single round exceeds 5 s,
    // bail out of further rounds. HIR is valid at any round boundary
    // (every pass leaves a well-formed module), so this converts a
    // potential hang into "slightly less-optimized code". Safety net
    // for the relaxed inliner — the relaxations open new shapes the
    // downstream passes haven't been audited against.
    const ROUND_WALL_CLOCK_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);
    for _ in 0..8 {
        let round_start = web_time::Instant::now();
        let cf = const_fold::fold_module(module);
        let cs = cse::eliminate_module(module);
        // load_cse runs after value-cse so canonical pointer ids are
        // already chased — if two GEPs cse'd to one, the load_cse
        // pass sees both loads using the same canonical ptr id.
        let lcse = load_cse::run_module(module);
        // aggregate_split runs after load_cse so the struct-typed
        // Loads it targets are the canonical ones (load_cse may
        // have collapsed sibling Loads of the same pointer).
        // Eliminates the load-modify-store cycle on struct values
        // produced by `let mut b = arr[i]; b.x = …; arr[i] = b`.
        let ags = aggregate_split::run_module(module);
        // scalar_replace_alloc runs after aggregate_split:
        //   * aggregate_split has just rewritten any struct-typed
        //     round-trips into direct GEP+Load/Store. That exposes the
        //     malloc-result+GEP+Load/Store shape that this pass needs.
        //   * Runs BEFORE drop_insert (which only fires after the
        //     fixed-point), so eliminated mallocs never get a paired
        //     Free synthesised.
        // Cranelift's mem2reg cannot promote heap allocations
        // (Call results are opaque); this is the HIR-only path.
        let sra = scalar_replace_alloc::run_module(module);
        let il = inline::run_module(module);
        let lc = licm::run_module(module);
        // Before the loops are matched against a shape. A variable live
        // across a loop but never reassigned in it still carries a phi,
        // which reads as a definition in the header and makes a buffer
        // the function allocated itself look like it changes every
        // iteration. Collapsing those first is what lets such a loop be
        // recognised at all.
        let ppf = phi_prune::run_module(module);
        stats.phi_prune.removed += ppf.removed;
        stats.phi_prune.rounds = stats.phi_prune.rounds.max(ppf.rounds);
        let lv = loop_vectorize::run_module(module);
        // Reduction vectorization runs alongside loop_vectorize; the
        // two recognise disjoint patterns (store-to-array vs.
        // accumulator) so they can't double-fire on the same loop.
        let rv = reduction_vectorize::run_module(module);
        // FMA contraction runs after the vectorizers, not before them.
        // A multiply feeding an add is the shape both the loop matcher
        // and this pass want, and whichever runs first takes it: fusing
        // to a single call leaves the matcher a call it cannot lower to
        // lanes, and a scaled kernel such as `y[i] = a * x[i] + y[i]`
        // stays scalar. Fusing afterwards costs nothing, because the
        // pass contracts a vector-typed multiply and add just as
        // readily as a scalar pair, so the loop ends up both widened
        // and fused.
        //
        // It still runs after const_fold + cse, which is why it sits
        // inside the round rather than after it: a multiply of
        // constants has already collapsed, and a pattern CSE could
        // eliminate outright is not contracted first.
        //
        // Temporary investigation gate: `ZYNTAX_DISABLE_FMA=1` skips
        // the pass entirely so before/after HIR can be diffed and
        // exec-time effect measured on Apple-Silicon. Remove once the
        // hot-loop FMA-helps/hurts question is closed.
        let fma_disabled = std::env::var("ZYNTAX_DISABLE_FMA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let fma = if fma_disabled {
            fma_contract::FmaStats::default()
        } else {
            fma_contract::run_module(module)
        };
        // cfg_simplify runs last in the round — `const_fold`'s
        // CondBranch-on-known-Bool collapse routinely turns
        // conditional branches into unconditional ones, which makes
        // the target block a straight-line successor ready for
        // merging.
        let cs_cfg = cfg_simplify::run_module(module);

        let made_progress = cf.folded > 0
            || cs.eliminated > 0
            || fma.contracted > 0
            || lcse.eliminated > 0
            || ags.round_trips_removed > 0
            || ags.field_reads_only > 0
            || sra.mallocs_eliminated > 0
            || il.inlined > 0
            || lc.hoisted > 0
            || lv.vectorized > 0
            || rv.vectorized > 0
            || cs_cfg.merged > 0;

        // Accumulate stats from this round.
        stats.const_fold.folded += cf.folded;
        stats.const_fold.iterations = stats.const_fold.iterations.max(cf.iterations);
        stats.cse.eliminated += cs.eliminated;
        stats.cse.rewrites += cs.rewrites;
        stats.fma_contract.contracted += fma.contracted;
        stats.load_cse.eliminated += lcse.eliminated;
        stats.aggregate_split.round_trips_removed += ags.round_trips_removed;
        stats.aggregate_split.field_accesses_emitted += ags.field_accesses_emitted;
        stats.aggregate_split.field_reads_only += ags.field_reads_only;
        stats.scalar_replace_alloc.candidates_examined += sra.candidates_examined;
        stats.scalar_replace_alloc.mallocs_eliminated += sra.mallocs_eliminated;
        stats.scalar_replace_alloc.frees_eliminated += sra.frees_eliminated;
        stats.scalar_replace_alloc.escapes_skipped += sra.escapes_skipped;
        stats.inline.inlined += il.inlined;
        stats.inline.call_sites_visited += il.call_sites_visited;
        stats.inline.skipped_too_large += il.skipped_too_large;
        stats.inline.skipped_multi_block += il.skipped_multi_block;
        stats.inline.skipped_unsupported += il.skipped_unsupported;
        stats.inline.skipped_indirect += il.skipped_indirect;
        stats.inline.inlined_multiblock_alloca += il.inlined_multiblock_alloca;
        stats.inline.inlined_multiblock_intrinsic_call += il.inlined_multiblock_intrinsic_call;
        stats.inline.skipped_caller_budget += il.skipped_caller_budget;
        stats.inline.skipped_per_pass_cap += il.skipped_per_pass_cap;
        stats.inline.skipped_post_inline_overflow += il.skipped_post_inline_overflow;
        stats.licm.hoisted += lc.hoisted;
        stats.licm.loops_visited += lc.loops_visited;
        stats.licm.loops_skipped_no_preheader += lc.loops_skipped_no_preheader;
        stats.loop_vectorize.vectorized += lv.vectorized;
        stats.loop_vectorize.loops_visited += lv.loops_visited;
        stats.loop_vectorize.skipped_shape += lv.skipped_shape;
        stats.loop_vectorize.skipped_no_induction += lv.skipped_no_induction;
        stats.loop_vectorize.skipped_op_unsupported += lv.skipped_op_unsupported;
        stats.reduction_vectorize.vectorized += rv.vectorized;
        stats.reduction_vectorize.loops_visited += rv.loops_visited;
        stats.reduction_vectorize.skipped_shape += rv.skipped_shape;
        stats.reduction_vectorize.skipped_op_unsupported += rv.skipped_op_unsupported;
        stats.cfg_simplify.merged += cs_cfg.merged;

        if !made_progress {
            break;
        }

        if round_start.elapsed() > ROUND_WALL_CLOCK_BUDGET {
            // Round took too long — almost certainly a runaway loop
            // in one of the relaxed passes. Stop iterating; the
            // module is still well-formed.
            break;
        }
    }

    // Affine reduction-loop closed-forming runs ONCE after the
    // fixed-point sweep, before the vectorizers. Ordering rationale:
    //   * After the sweep: it needs const_fold to have propagated
    //     loop-invariant constants (e.g. `factor → 1.0`) into the
    //     recurrence, cse/inline to have settled (the free-function
    //     kernel's `step()` call is inlined into the body first), and
    //     fma_contract to have canonicalised `a*1 + b` into the single
    //     `Fma(acc, one, b)` form the recogniser matches. Running it
    //     earlier would see a half-canonicalised body and bail.
    //   * Before auto_vectorize: an affine accumulator is a serial,
    //     loop-carried recurrence — not vectorizable — so closing it to
    //     its constant closed form is strictly better than handing the
    //     loop to the vectorizer, and removing the loop first keeps the
    //     vectorizer from wasting a pass on it.
    // The transform fires only when every input is a compile-time
    // constant and (for floats) the closed form is provably bit-exact,
    // so it introduces no rounding the serial loop wouldn't also
    // produce. See `affine_loop` module docs for the soundness proof.
    let al = affine_loop::run_module(module);
    stats.affine_loop.folded += al.folded;
    stats.affine_loop.loops_visited += al.loops_visited;
    stats.affine_loop.skipped_shape += al.skipped_shape;
    stats.affine_loop.skipped_no_preheader += al.skipped_no_preheader;
    stats.affine_loop.skipped_unrecognized_body += al.skipped_unrecognized_body;
    stats.affine_loop.skipped_trip_count += al.skipped_trip_count;
    stats.affine_loop.skipped_recurrence += al.skipped_recurrence;
    stats.affine_loop.skipped_float_inexact += al.skipped_float_inexact;

    // auto_vectorize runs ONCE after the fixed-point sweep terminates.
    // Vectorization is not fixed-point-monotonic — vector instructions
    // don't expose new const-fold/cse opportunities — so a single pass
    // after const_fold/cse/inline/licm have all settled gives the
    // cleanest IV/stride pattern matcher input. This is the generalised
    // successor to `loop_vectorize` + `reduction_vectorize` which still
    // run inside the sweep for now (transition period — see module
    // docs).
    let av = auto_vectorize::run_module(module);
    stats.auto_vectorize.vectorized += av.vectorized;
    stats.auto_vectorize.loops_visited += av.loops_visited;
    stats.auto_vectorize.rejected_shape += av.rejected_shape;
    stats.auto_vectorize.rejected_hazard += av.rejected_hazard;
    stats.auto_vectorize.rejected_cost += av.rejected_cost;
    stats.auto_vectorize.rejected_no_iv += av.rejected_no_iv;
    stats.auto_vectorize.rejected_trip_count += av.rejected_trip_count;

    // Recursive inlining runs ONCE after the fixed-point sweep.
    // Trying to put it inside the fixed-point would invite divergence
    // — the regular inliner's iterative budget cap interacts badly
    // with a pass that's deliberately tripling IR size in one shot.
    // Two ordering reasons for placing it here:
    //   * After the fixed-point: const_fold + cse + cfg_simplify have
    //     already canonicalised the body, so the snapshot the
    //     recursive inliner clones is the smallest legal version.
    //     The post-inline IR is correspondingly cleaner — fewer
    //     redundant phis, no dead branches threaded through the
    //     copies.
    //   * Before tco / drop_insert: inlining can convert a previously
    //     non-tail Call into a tail-position one (the inlined Return
    //     becomes the caller's Return), so tco runs after and catches
    //     the new shapes. drop_insert needs to see the final post-
    //     inline body to place Frees at the correct life-range ends.
    // Drop phis nothing reads before the remaining passes run. SSA
    // placement gives every variable written in a loop a phi, including
    // ones declared inside the body whose previous-iteration value is
    // never observed. Each survivor becomes a loop-carried value in the
    // backend and holds a register for the whole loop, which is what
    // pushes the register allocator into spilling a real one.
    let pp = phi_prune::run_module(module);
    stats.phi_prune.removed += pp.removed;
    stats.phi_prune.rounds = stats.phi_prune.rounds.max(pp.rounds);

    let ri = inline::run_module_recursive(module);
    stats.recursive_inline.functions_visited += ri.functions_visited;
    stats.recursive_inline.self_calls_inlined += ri.self_calls_inlined;
    stats.recursive_inline.skipped_too_large += ri.skipped_too_large;
    stats.recursive_inline.skipped_too_many_sites += ri.skipped_too_many_sites;
    stats.recursive_inline.skipped_unsupported += ri.skipped_unsupported;

    // Recursive self-inlining above can leave two inlined copies of a
    // body sharing a pure sub-call — e.g. the depth-1 inline of `fib`
    // makes both the `fib(n-1)` and `fib(n-2)` expansions call
    // `fib(n-3)`. Re-infer purity (inlining rewrote bodies), then:
    //   * `pure_call_pre` hoists the shared call across the two base-case
    //     branches to their common dominator (the cross-branch case
    //     dominator-CSE can't reach), gated on speculation-safety, and
    //   * `cse` collapses any remaining duplicate that already sits in a
    //     dominance relationship.
    purity::infer_module(module);
    let pcp = pure_call_pre::run_module(module);
    stats.pure_call_pre.hoisted += pcp.hoisted;
    stats.pure_call_pre.groups_visited += pcp.groups_visited;
    let post_ri_cse = cse::eliminate_module(module);
    stats.cse.eliminated += post_ri_cse.eliminated;
    stats.cse.rewrites += post_ri_cse.rewrites;

    // Tail-call marker runs after the fixed-point sweep but before
    // drop_insert. Two ordering reasons:
    //   * After the fixed-point: inlining can convert a non-tail Call
    //     into a tail-shape one (the inlined return becomes the
    //     caller's return), and const_fold / cse / cfg_simplify may
    //     also clear away intervening instructions that were blocking
    //     a tail shape. Running tco after all of those settle catches
    //     every shape they enable.
    //   * Before drop_insert: drop_insert places `Free` calls between
    //     Call and Return for any non-escaping malloc, breaking the
    //     "Call is the last instruction" precondition. Running tco
    //     first marks every valid shape; the drop_insert pass can
    //     then add Free calls without those marks getting lost.
    let tc = tco::run_module(module);
    stats.tco.candidates_visited += tc.candidates_visited;
    stats.tco.marked += tc.marked;

    // Drop-site insertion runs *after* the optimization fixed-point.
    // Order matters two ways:
    //   * Inserting Free calls earlier would create new memory
    //     barriers that load_cse would have to treat conservatively,
    //     suppressing legitimate redundancy elimination.
    //   * Running drop-insert on the *optimised* IR is what gives
    //     speculative drop-site analysis its precision: const_fold
    //     and cse may have replaced uses, eliminating spurious
    //     "use" sites that would otherwise extend the live-range
    //     past the real last use. Running it once at the end means
    //     each malloc gets its tightest legal drop position.
    let di = drop_insert::run_module(module);
    stats.drop_insert.mallocs_scanned += di.mallocs_scanned;
    stats.drop_insert.frees_inserted += di.frees_inserted;
    stats.drop_insert.escapes_skipped += di.escapes_skipped;
    stats.drop_insert.multi_block_skipped += di.multi_block_skipped;
    stats.drop_insert.no_use_skipped += di.no_use_skipped;

    stats
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
