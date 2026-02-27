//! # Zyntax Embed - Rust SDK for Embedding Zyntax JIT
//!
//! This crate provides ergonomic Rust APIs for embedding Zyntax as a JIT runtime,
//! enabling bidirectional conversion between Zyntax runtime values and native Rust types.
//!
//! ## Key Features
//!
//! - **Compiler Integration**: Compile and execute Zyntax code directly from Rust
//! - **Type-safe conversions**: `FromZyntax` and `IntoZyntax` traits for seamless value conversion
//! - **Async Support**: `ZyntaxPromise` for handling async operations with `.then()` and `.catch()`
//! - **Runtime value handling**: `ZyntaxValue` enum for working with dynamically-typed Zyntax values
//! - **String/Array interop**: Zero-copy wrappers for Zyntax's native formats
//! - **Hot Reloading**: Update functions at runtime without restarting
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use zyntax_embed::{ZyntaxRuntime, ZyntaxValue, FromZyntax};
//!
//! // Create a runtime and compile code
//! let mut runtime = ZyntaxRuntime::new()?;
//! runtime.compile_module(&hir_module)?;
//!
//! // Call functions with automatic type conversion
//! let result: i32 = runtime.call("add", &[10.into(), 20.into()])?;
//! assert_eq!(result, 30);
//!
//! // Async functions return Promises
//! let promise = runtime.call_async("fetch_data", &[url.into()])?;
//! let data: String = promise.await_result()?;
//! ```
//!
//! ## Memory Management
//!
//! Zyntax uses a specific memory format for its runtime values:
//! - **Strings**: Length-prefixed format `[i32 length][utf8_bytes...]`
//! - **Arrays**: Header format `[i32 capacity][i32 length][elements...]`
//!
//! This crate handles all memory conversion automatically, ensuring proper allocation
//! and deallocation when values cross the Rust/Zyntax boundary.
//!
//! ## Language Grammar Support
//!
//! Use `LanguageGrammar` to parse source code using ZynPEG grammars:
//!
//! ```rust,ignore
//! use zyntax_embed::LanguageGrammar;
//!
//! // Compile from .zyn grammar source
//! let grammar = LanguageGrammar::compile_zyn(include_str!("my_lang.zyn"))?;
//!
//! // Parse source code
//! let program = grammar.parse("fn main() { 42 }")?;
//! ```

mod array;
mod convert;
mod error;
mod grammar;
mod grammar2;
pub mod iterator;
mod runtime;
mod string;
mod value;

pub use array::ZyntaxArray;
pub use convert::{FromZyntax, IntoZyntax, TryFromZyntax, TryIntoZyntax};
pub use error::{ConversionError, ZyntaxError};
pub use grammar::{GrammarError, GrammarResult, LanguageGrammar};
pub use grammar2::{Grammar2, Grammar2Error, Grammar2Result};
pub use iterator::{
    IntoZrtlIterator, StdIteratorAdapter, ZrtlIterable, ZrtlIterator, ZrtlIteratorAdapter,
    ZrtlIteratorExt, ZrtlRangeIterator, ZyntaxArrayIterator, ZyntaxStringBytesIterator,
    ZyntaxStringCharsIterator, ZyntaxValueIterator,
};
pub use runtime::{
    // Async ABI types
    AsyncPollResult,
    BuiltinResolver,
    ChainedResolver,
    ExportedSymbol,
    ImportContext,
    ImportError,
    ImportManager,
    ImportResolverCallback,
    // Re-export import resolver types for advanced use cases
    ImportResolverTrait,
    ModuleArchitecture,
    NativeSignature,
    // Native calling interface
    NativeType,
    // Promise combinators (Promise.all, Promise.race, etc.)
    PromiseAll,
    PromiseAllSettled,
    PromiseAllState,
    PromiseRace,
    PromiseRaceState,
    PromiseState,
    ResolvedImport,
    RuntimeError,
    RuntimeEvent,
    RuntimeResult,
    SettledResult,
    SymbolKind,
    TieredRuntime,
    ZyntaxPromise,
    ZyntaxRuntime,
};
pub use string::ZyntaxString;
pub use value::ZyntaxValue;

// Re-export zyn_peg types for custom AST builders and advanced grammar use
pub use zyn_peg::runtime::{
    AstCommand, AstHostFunctions, CommandInterpreter, NodeHandle, RuleCommands, RuntimeValue,
    TypedAstBuilder, ZpegMetadata, ZpegModule,
};

// Re-export TypedProgram for users who parse to TypedAST
pub use zyntax_typed_ast::TypedProgram;

// Re-export tiered compilation types
pub use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig, TieredStatistics};

// Re-export core types from zyntax_compiler for convenience
pub use zyntax_compiler::zrtl::{
    DynamicValue,
    GenericTypeArgs,
    GenericValue,
    TypeCategory,
    TypeFlags,
    TypeId,
    TypeInfo,
    TypeMeta,
    TypeRegistry,
    TypeTag,
    ZrtlError,
    ZrtlInfo,
    // ZRTL plugin loading
    ZrtlPlugin,
    ZrtlRegistry,
    ZrtlSymbol,
    ZRTL_VERSION,
};

// Re-export compiler types needed for module compilation
pub use zyntax_compiler::{
    compile_to_hir, compile_to_jit, CompilationConfig, CompilerError, CompilerResult, HirFunction,
    HirModule,
};
