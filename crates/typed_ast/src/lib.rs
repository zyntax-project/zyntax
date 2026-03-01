#![allow(unused, dead_code, deprecated)]

//! # Zyntax TypedAST
//!
//! A robust TypedAST that serves as the common intermediate representation
//! for multiple statically-typed languages including Rust, Java, C#,
//! TypeScript, and Haxe.
//!
//! ## Overview
//!
//! This crate provides a fully-typed abstract syntax tree that other language
//! implementations can target. It includes:
//!
//! - **Complete Type System**: Parametric polymorphism, subtyping, type inference
//! - **Multi-Language Support**: Designed to handle features from various languages
//! - **Type Inference Engine**: Hindley-Milner style inference with extensions
//! - **Type Checking**: Sound type checking with support for gradual typing
//! - **Memory Efficiency**: Arena allocation and string interning
//!
//! ## Architecture
//!
//! Languages compile to this TypedAST in phases:
//! 1. Parse source language to its native AST
//! 2. Lower to Zyntax TypedAST with type annotations
//! 3. Run type inference to resolve unknown types
//! 4. Type check for correctness
//! 5. Optimize and transform the TypedAST
//! 6. Generate target code
//!
//! ## Usage
//!
//! The TypedASTBuilder provides a fluent interface for constructing typed AST nodes:
//!
//! ```rust,no_run
//! use zyntax_typed_ast::{TypedASTBuilder, Type, PrimitiveType, Mutability, Span};
//!
//! let mut builder = TypedASTBuilder::new();
//! let span = Span::new(0, 10);
//!
//! // Build typed expressions
//! let int_expr = builder.int_literal(42, span);
//! let str_expr = builder.string_literal("hello", span);
//! let bool_expr = builder.bool_literal(true, span);
//!
//! // Enhanced parameter system
//! let regular_param = builder.parameter("x", Type::Primitive(PrimitiveType::I32), Mutability::Immutable, span);
//! let default_value = builder.int_literal(100, span);
//! let optional_param = builder.optional_parameter("y", Type::Primitive(PrimitiveType::I32), Mutability::Immutable, default_value, span);
//! let out_param = builder.out_parameter("result", Type::Primitive(PrimitiveType::I32), span);
//!
//! // Function calls with named arguments
//! let callee = builder.variable("func", Type::Primitive(PrimitiveType::I32), span);
//! let call = builder.call_named(callee, vec![("x", int_expr), ("y", str_expr)], Type::Primitive(PrimitiveType::I32), span);
//! ```

pub mod arena;

pub mod diagnostics;
pub mod error;
pub mod source;

pub mod advanced_analysis;
pub mod ast_convert;
pub mod constraint_solver;
pub mod import_resolver;
pub mod smt_solver;
pub mod type_checker;
pub mod type_inference;
pub mod type_registry;
pub mod typed_ast;
pub mod typed_builder;

// Universal Type System - supporting multiple language paradigms
// pub(crate) mod universal_type_system; // Internal use only - deprecated
pub mod const_evaluator;
pub mod dependent_types;
pub mod effect_system;
pub mod gradual_type_checker;
pub mod linear_types;
pub mod multi_paradigm_checker;
pub mod nominal_type_checker;
pub mod structural_type_checker;

#[cfg(test)]
pub mod test_dsl;

// Re-exports for convenience
pub use arena::{ArenaStatistics, AstArena, InternedString, MemoryUsage};

pub use diagnostics::{
    codes, Annotation, AnnotationStyle, ConsoleDiagnosticDisplay, Diagnostic, DiagnosticBuilder,
    DiagnosticCode, DiagnosticCollector, DiagnosticDisplay, DiagnosticLevel, Suggestion,
    SuggestionApplicability,
};
pub use error::{AstError, ErrorReporter, ValidationError};
pub use source::{Location, SourceFile, SourceMap, Span};

// Unified Type System exports (enhanced TypeRegistry with universal features)
pub use type_registry::{
    AssociatedTypeDef,
    AsyncKind,
    CallingConvention,
    ConstBinaryOp,
    ConstConstraint,
    ConstPredicate,
    ConstUnaryOp,
    ConstValue,
    ConstVarId,
    ConstructorSig,
    FieldDef,
    ImplDef,
    Kind,
    Lifetime,
    LifetimeBound,
    MethodImpl,
    MethodSig,
    Mutability,
    // Universal type system features
    NullabilityKind,
    ParamDef,
    ParamInfo,
    PrimitiveType,
    TraitDef,
    Type,
    TypeBound,
    TypeConstraint,
    TypeDefinition,
    TypeId,
    TypeKind,
    TypeMetadata,
    TypeParam,
    TypeRegistry,
    TypeVar,
    TypeVarId,
    TypeVarKind,
    Variance,
    VariantDef,
    VariantFields,
    Visibility,
};

// Operator traits feature exports
#[cfg(feature = "operator_traits")]
pub use type_registry::BuiltinTraitIds;

pub use typed_ast::{
    typed_node,
    BinaryOp,
    ParameterKind,
    // Annotation types
    TypedAnnotation,
    TypedAnnotationArg,
    TypedAnnotationValue,
    TypedBinary,
    TypedBlock,
    TypedCall,
    // Cast type
    TypedCast,
    // Class/Struct/Enum types
    TypedClass,
    TypedComputeExpr,
    TypedComputeModifier,
    TypedDeclaration,
    // Defer type
    TypedDefer,
    // Algebraic effect types
    TypedEffect,
    TypedEffectHandler,
    TypedEffectHandlerImpl,
    TypedEffectOp,
    TypedEnum,
    TypedExpression,
    // Extern types for external declarations
    TypedExtern,
    TypedExternClass,
    TypedExternEnum,
    TypedExternEnumVariant,
    TypedExternMethod,
    TypedExternProperty,
    TypedExternStruct,
    TypedExternTypeDef,
    TypedField,
    TypedFieldAccess,
    TypedFieldInit,
    TypedFieldPattern,
    TypedFor,
    TypedFunction,
    TypedIf,
    // If expression type
    TypedIfExpr,
    TypedImplAssociatedType,
    // Import types
    TypedImport,
    TypedImportItem,
    TypedImportModifier,
    TypedIndex,
    TypedInterface,
    TypedKernelAttr,
    // Lambda types
    TypedLambda,
    TypedLambdaBody,
    TypedLambdaParam,
    TypedLet,
    TypedLetPattern,
    // List comprehension, slice, import modifier, and path types
    TypedListComprehension,
    TypedLiteral,
    TypedLiteralPattern,
    TypedMatch,
    TypedMatchArm,
    TypedMatchExpr,
    TypedMethod,
    // Method types
    TypedMethodCall,
    TypedMethodParam,
    TypedModule,
    TypedNamedArg,
    TypedNode,
    // Additional types for parser generation
    TypedParameter,
    TypedPath,
    // Pattern types
    TypedPattern,
    TypedProgram,
    // Range type
    TypedRange,
    // Reference type
    TypedReference,
    TypedSlice,
    TypedStatement,
    // Struct type
    TypedStructLiteral,
    // Trait implementation types
    TypedTraitImpl,
    TypedTypeAlias,
    TypedTypeParam,
    TypedUnary,
    TypedVariable,
    TypedVariant,
    TypedVariantFields,
    TypedWhile,
    UnaryOp,
};

pub use type_inference::{Constraint, InferenceContext, InferenceError, InferenceOptions};

pub use constraint_solver::{
    Constraint as SolverConstraint, ConstraintSolver, SolverError, Substitution, TypeScheme,
};

pub use type_checker::{TypeCheckOptions, TypeChecker, TypeError};

pub use typed_builder::TypedASTBuilder;

pub use advanced_analysis::{
    AnalysisContext, AnalysisError, AnalysisPhase, AnalysisResult, ControlFlowError,
    ControlFlowGraph, DataFlowError, DataFlowGraph, LifetimeAnalysis, LifetimeError,
    OwnershipAnalysis, OwnershipError,
};

// Note: UniversalType features have been merged into the unified Type enum above
// Note: universal_type_system is used internally by some type checkers but should not be used directly

pub use nominal_type_checker::{
    ImplDefinition, MethodSource, NominalTypeChecker, NominalTypeError, ResolvedMethod,
    TraitDefinition,
};

pub use structural_type_checker::{
    AdapterRequirement, FieldSignature, MethodSignature, StructuralCompatibility, StructuralError,
    StructuralMode, StructuralTypeChecker,
};

pub use gradual_type_checker::{
    BoundaryKind, ConfidenceLevel, Evidence, EvidenceKind, FallbackBehavior, GradualTypeChecker,
    GradualTypeError, RuntimeCheck, RuntimeCheckKind, TypeEvidence,
};

pub use const_evaluator::{ConstEvalError, ConstEvalResult, ConstEvaluator};

pub use dependent_types::{
    ArithmeticOp, ComparisonOp, DependentIndex, DependentType, DependentTypeChecker,
    DependentTypeError, DependentTypeResult, ParamKind, RefinementExpr, RefinementFunction,
    RefinementPredicate, TypeFamily, TypeFamilyParam, TypePath, UnaryOp as RefinementUnaryOp,
};

pub use linear_types::{
    BorrowChecker, BorrowId, BorrowInfo, BorrowKind, BorrowingRules, CleanupBehavior,
    LinearConstraint, LinearTypeChecker, LinearTypeError, LinearTypeInfo, LinearTypeResult,
    LinearityKind, ResourceId, ResourceInfo, ResourceKind, ResourceTracker, ScopeKind, UsageInfo,
};

pub use effect_system::{
    CompositionOperator, Effect, EffectCapability, EffectConstraint, EffectError, EffectHandler,
    EffectHandlerId, EffectId, EffectInferenceContext, EffectIntensity, EffectKind, EffectRegion,
    EffectRegionId, EffectRequirement, EffectResult, EffectScopeKind, EffectSet, EffectSignature,
    EffectSystem, EffectTransformation, EffectTypeInfo, EffectVar, EffectVarId, EffectVarKind,
    EffectVariance, HandlerType,
};

pub use import_resolver::{
    BuiltinResolver, ChainedResolver, CompiledCacheFormat, CompiledModuleCache, EntryPointResolver,
    ExportedSymbol, ImportContext, ImportError, ImportManager, ImportResolver, ModuleArchitecture,
    ModuleSource, ResolvedImport, SymbolKind,
};

// Note: universal_type_system is deprecated and only used internally by some type checkers

/// Quick way to create a new TypedAST builder
pub fn typed_builder() -> TypedASTBuilder {
    TypedASTBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_ast::{BinaryOp, TypedPattern};

    #[test]
    fn test_typed_builder_comprehensive() {
        let mut builder = typed_builder();
        let span = Span::new(0, 10);

        // Test enhanced parameter system
        let param1 = builder.parameter(
            "x",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            span,
        );
        let default_val = builder.int_literal(42, span);
        let param2 = builder.optional_parameter(
            "y",
            Type::Primitive(PrimitiveType::I32),
            Mutability::Immutable,
            default_val,
            span,
        );
        let param3 = builder.out_parameter("result", Type::Primitive(PrimitiveType::I32), span);

        // Test function call with named arguments
        let callee = builder.variable("func", Type::Primitive(PrimitiveType::I32), span);
        let arg1 = builder.int_literal(1, span);
        let arg2 = builder.int_literal(2, span);
        let call = builder.call_named(
            callee,
            vec![("x", arg1), ("y", arg2)],
            Type::Primitive(PrimitiveType::I32),
            span,
        );

        // Test pattern matching
        let px_name = builder.intern("px");
        let py_name = builder.intern("py");
        let pattern = builder.struct_pattern(
            "Point",
            vec![
                (
                    "x",
                    typed_node(TypedPattern::immutable_var(px_name), Type::Never, span),
                ),
                (
                    "y",
                    typed_node(TypedPattern::immutable_var(py_name), Type::Never, span),
                ),
            ],
            span,
        );

        // Verify parameter kinds
        assert_eq!(param1.kind, crate::typed_ast::ParameterKind::Regular);
        assert_eq!(param2.kind, crate::typed_ast::ParameterKind::Optional);
        assert_eq!(param3.kind, crate::typed_ast::ParameterKind::Out);
        assert!(param2.default_value.is_some());

        // Verify call structure
        if let crate::typed_ast::TypedExpression::Call(call_data) = &call.node {
            assert_eq!(call_data.named_args.len(), 2);
        } else {
            panic!("Expected function call");
        }

        // Verify pattern structure
        if let crate::typed_ast::TypedPattern::Struct { fields, .. } = &pattern.node {
            assert_eq!(fields.len(), 2);
        } else {
            panic!("Expected struct pattern");
        }
    }
}
