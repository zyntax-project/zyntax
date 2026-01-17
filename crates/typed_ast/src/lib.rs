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

pub mod source;
pub mod error;
pub mod diagnostics;

pub mod type_registry;
pub mod typed_ast;
pub mod typed_builder;
pub mod ast_convert;
pub mod type_inference;
pub mod constraint_solver;
pub mod smt_solver;
pub mod type_checker;
pub mod advanced_analysis;
pub mod import_resolver;

// Universal Type System - supporting multiple language paradigms
// pub(crate) mod universal_type_system; // Internal use only - deprecated
pub mod multi_paradigm_checker;
pub mod nominal_type_checker;
pub mod structural_type_checker;
pub mod gradual_type_checker;
pub mod const_evaluator;
pub mod dependent_types;
pub mod linear_types;
pub mod effect_system;

#[cfg(test)]
pub mod test_dsl;

// Re-exports for convenience
pub use arena::{AstArena, InternedString, ArenaStatistics, MemoryUsage};



pub use source::{SourceFile, SourceMap, Span, Location};
pub use error::{AstError, ValidationError, ErrorReporter};
pub use diagnostics::{
    Diagnostic, DiagnosticLevel, DiagnosticCode, DiagnosticCollector,
    DiagnosticDisplay, ConsoleDiagnosticDisplay, DiagnosticBuilder,
    Annotation, AnnotationStyle, Suggestion, SuggestionApplicability,
    codes,
};

// Unified Type System exports (enhanced TypeRegistry with universal features)
pub use type_registry::{
    TypeRegistry, TypeDefinition, TypeKind, TypeConstraint,
    TypeParam, FieldDef, MethodSig, ConstructorSig, ParamDef, ParamInfo,
    TypeMetadata, Variance, Mutability, Visibility, Lifetime, LifetimeBound,
    TypeBound, TypeVar, TypeVarKind, TypeVarId,
    Type, TypeId, PrimitiveType,
    TraitDef, ImplDef, MethodImpl, AssociatedTypeDef,
    VariantDef, VariantFields,
    // Universal type system features
    NullabilityKind, AsyncKind, CallingConvention, ConstValue, ConstConstraint, ConstPredicate,
    ConstBinaryOp, ConstUnaryOp, ConstVarId, Kind,
};

// Operator traits feature exports
#[cfg(feature = "operator_traits")]
pub use type_registry::BuiltinTraitIds;

pub use typed_ast::{
    TypedNode, TypedProgram, TypedDeclaration, TypedFunction, TypedVariable,
    TypedStatement, TypedExpression, TypedLiteral, BinaryOp, UnaryOp,
    typed_node,
    // Additional types for parser generation
    TypedParameter, TypedLet, TypedLetPattern, TypedIf, TypedWhile, TypedBlock,
    TypedBinary, TypedUnary, TypedCall, TypedFieldAccess, TypedIndex,
    TypedFor, TypedMatch, TypedMatchExpr, TypedMatchArm, ParameterKind,
    // Lambda types
    TypedLambda, TypedLambdaBody, TypedLambdaParam,
    // Method types
    TypedMethodCall, TypedMethod, TypedMethodParam,
    // Range type
    TypedRange,
    // List comprehension, slice, and import modifier types
    TypedListComprehension, TypedSlice, TypedImportModifier,
    // Struct type
    TypedStructLiteral, TypedFieldInit,
    // Pattern types
    TypedPattern, TypedLiteralPattern, TypedFieldPattern,
    // Reference type
    TypedReference,
    // Cast type
    TypedCast,
    // If expression type
    TypedIfExpr,
    // Defer type
    TypedDefer,
    // Class/Struct/Enum types
    TypedClass, TypedEnum, TypedField, TypedVariant, TypedVariantFields, TypedTypeParam, TypedTypeAlias,
    // Import types
    TypedImport, TypedImportItem, TypedModule,
    // Extern types for external declarations
    TypedExtern, TypedExternClass, TypedExternStruct, TypedExternEnum,
    TypedExternEnumVariant, TypedExternTypeDef, TypedExternMethod, TypedExternProperty,
    // Trait implementation types
    TypedTraitImpl, TypedImplAssociatedType, TypedInterface,
    // Annotation types
    TypedAnnotation, TypedAnnotationArg, TypedAnnotationValue,
    // Algebraic effect types
    TypedEffect, TypedEffectOp, TypedEffectHandler, TypedEffectHandlerImpl,
};

pub use type_inference::{
    InferenceContext, InferenceOptions, InferenceError,
    Constraint,
};

pub use constraint_solver::{
    ConstraintSolver, TypeScheme, Substitution, SolverError,
    Constraint as SolverConstraint,
};

pub use type_checker::{
    TypeChecker, TypeCheckOptions, TypeError,
};

pub use typed_builder::TypedASTBuilder;

pub use advanced_analysis::{
    AnalysisContext, AnalysisResult, AnalysisError, AnalysisPhase,
    DataFlowGraph, ControlFlowGraph, OwnershipAnalysis, LifetimeAnalysis,
    OwnershipError, LifetimeError, ControlFlowError, DataFlowError,
};

// Note: UniversalType features have been merged into the unified Type enum above
// Note: universal_type_system is used internally by some type checkers but should not be used directly

pub use nominal_type_checker::{
    NominalTypeChecker, TraitDefinition, ImplDefinition,
    ResolvedMethod, MethodSource, NominalTypeError,
};

pub use structural_type_checker::{
    StructuralTypeChecker, StructuralCompatibility, StructuralError, StructuralMode,
    MethodSignature, FieldSignature, AdapterRequirement,
};

pub use gradual_type_checker::{
    GradualTypeChecker, RuntimeCheck, RuntimeCheckKind, FallbackBehavior,
    TypeEvidence, Evidence, EvidenceKind, ConfidenceLevel, GradualTypeError,
    BoundaryKind,
};


pub use const_evaluator::{
    ConstEvaluator, ConstEvalError, ConstEvalResult,
};

pub use dependent_types::{
    DependentType, RefinementPredicate, RefinementExpr, TypePath, DependentIndex,
    DependentTypeChecker, DependentTypeError, DependentTypeResult,
    TypeFamily, TypeFamilyParam, ParamKind, RefinementFunction,
    ComparisonOp, ArithmeticOp, UnaryOp as RefinementUnaryOp,
};

pub use linear_types::{
    LinearTypeChecker, LinearTypeError, LinearTypeResult,
    LinearityKind, BorrowKind, ResourceKind, ScopeKind,
    UsageInfo, ResourceInfo, BorrowInfo, LinearTypeInfo,
    ResourceTracker, BorrowChecker, LinearConstraint,
    CleanupBehavior, BorrowingRules, ResourceId, BorrowId,
};

pub use effect_system::{
    EffectSystem, EffectTypeInfo, EffectSignature, EffectSet, Effect,
    EffectKind, EffectIntensity, EffectVar, EffectVarId, EffectVarKind,
    EffectRegion, EffectRegionId, EffectHandler, EffectHandlerId,
    EffectConstraint, EffectError, EffectResult, EffectVariance,
    EffectTransformation, EffectRequirement, EffectCapability,
    CompositionOperator, HandlerType, EffectScopeKind,
    EffectInferenceContext, EffectId,
};

pub use import_resolver::{
    ImportResolver, ImportContext, ImportManager, ImportError,
    ResolvedImport, ExportedSymbol, SymbolKind,
    ChainedResolver, BuiltinResolver, ModuleArchitecture, ModuleSource,
    CompiledCacheFormat, CompiledModuleCache, EntryPointResolver,
};

// Note: universal_type_system is deprecated and only used internally by some type checkers



/// Quick way to create a new TypedAST builder
pub fn typed_builder() -> TypedASTBuilder {
    TypedASTBuilder::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_ast::{TypedPattern, BinaryOp};



    #[test]
    fn test_typed_builder_comprehensive() {
        let mut builder = typed_builder();
        let span = Span::new(0, 10);

        // Test enhanced parameter system
        let param1 = builder.parameter("x", Type::Primitive(PrimitiveType::I32), Mutability::Immutable, span);
        let default_val = builder.int_literal(42, span);
        let param2 = builder.optional_parameter(
            "y", 
            Type::Primitive(PrimitiveType::I32), 
            Mutability::Immutable,
            default_val, 
            span
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
                ("x", typed_node(TypedPattern::immutable_var(px_name), Type::Never, span)),
                ("y", typed_node(TypedPattern::immutable_var(py_name), Type::Never, span)),
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