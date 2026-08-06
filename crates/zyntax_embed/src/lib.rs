#![allow(unused, dead_code, deprecated)]

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
mod effect_runtime;
mod error;
#[cfg(feature = "native")]
mod fiber;
mod grammar;
mod grammar2;
/// Cooperative-async future table. Browser-runtime parking layer
/// that breaks the spin-poll in `__zyntax_effect_resume` for SMs
/// that wait on host async ops (setTimeout / fetch / WebSocket).
/// Used by the Phase H+ scheduler in `crates/zyntax_wasm`; native
/// targets get the same surface so the per-bridge stdlib code
/// stays target-uniform.
pub mod host_futures;
#[cfg(feature = "native")]
mod import_chain;
/// BC-interpreter-backed execution engine. On native it's internal
/// scaffolding for [`runtime::ZyntaxRuntime`]; on wasm32 (where the
/// native runtime is gated off) it's the primary execution entry
/// point that the `zyntax_wasm` crate's wasm-bindgen shim wires
/// up. Holds the `HirInterpreter`, beadie's `TieredAdapter` (tier 0
/// only on wasm), and the FFI symbol table.
pub mod interp_runtime;
pub mod iterator;
/// Post-`compile_to_hir` krio passes: async-fn → poll-fn SM and
/// resumable-effect fn → poll-fn SM. Available on both native
/// and wasm targets (the native `ZyntaxRuntime::compile_typed_program`
/// path delegates to these; the wasm `run_impl` calls them
/// directly after `compile_to_hir`).
pub mod krio_lowering;
// `runtime` carries the full ZyntaxRuntime (Cranelift JIT, plugin
// loader, async executor). Native-only — the wasm-target entry point
// (separate `zyntax_wasm` crate, Phase F) wires the BC interpreter
// directly without dragging the native backend along.
#[cfg(feature = "native")]
mod runtime;
mod string;
mod value;

// Re-export the algebraic-effects runtime symbols + registration helper.
// Hosts that build a ZyntaxRuntime through the front-door
// `ZyntaxRuntime::new()` get them registered automatically; this is
// the seam for hosts that want to register them manually.
#[cfg(feature = "native")]
pub use effect_runtime::register_effect_runtime_symbols;
pub use effect_runtime::{
    __zyntax_async_set_timeout, __zyntax_effect_abort, __zyntax_effect_lookup_handler,
    __zyntax_effect_pop_handler, __zyntax_effect_push_handler, __zyntax_effect_resume,
    __zyntax_runtime_release_sm, __zyntax_runtime_release_sm_by_offset, __zyntax_runtime_retain_sm,
};
// Cooperative-async externs. Phase G plumbing — the browser shim
// in zyntax_wasm exports thin wrappers around these so JS-side
// host bridges can park / resolve / reject SMs.
pub use host_futures::{__zyntax_register_future, __zyntax_reject_future, __zyntax_resolve_future};

pub use array::ZyntaxArray;
// Re-export the BC interpreter so embedders that want a bare
// HirInterpreter without the beadie wrapper can grab it directly.
pub use convert::{FromZyntax, IntoZyntax, TryFromZyntax, TryIntoZyntax};
pub use error::{ConversionError, ZyntaxError};
#[cfg(feature = "native")]
pub use fiber::{ZyntaxFiber, ZyntaxFiberStep};
pub use grammar::{GrammarError, GrammarResult, LanguageGrammar};
pub use grammar2::{Grammar2, Grammar2Error, Grammar2Result};
pub use iterator::{
    IntoZrtlIterator, StdIteratorAdapter, ZrtlIterable, ZrtlIterator, ZrtlIteratorAdapter,
    ZrtlIteratorExt, ZrtlRangeIterator, ZyntaxArrayIterator, ZyntaxStringBytesIterator,
    ZyntaxStringCharsIterator, ZyntaxValueIterator,
};
/// Cooperative multi-task driver (native only — wasm is JS-event-loop driven).
#[cfg(all(feature = "native", not(target_arch = "wasm32")))]
pub use runtime::drive_tasks;
#[cfg(feature = "native")]
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
pub use zyntax_compiler::hir_interp::{HirInterpreter, InterpError, JitDispatch, ProfileSample};
pub use zyntax_compiler::reload::ReloadReport;

// Re-export zyn_peg types for custom AST builders and advanced grammar use
pub use zyn_peg::runtime::{
    AstCommand, AstHostFunctions, CommandInterpreter, NodeHandle, RuleCommands, RuntimeValue,
    TypedAstBuilder, ZpegMetadata, ZpegModule,
};

// Re-export TypedProgram for users who parse to TypedAST
pub use zyntax_typed_ast::TypedProgram;

// Re-export tiered compilation types
#[cfg(feature = "native")]
pub use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig, TieredStatistics};

// Re-export core types from zyntax_compiler for convenience
pub use zyntax_compiler::zrtl::{
    DynamicValue,
    GenericTypeArgs,
    GenericValue,
    RuntimeSymbolInfo,
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
    ZrtlSigFlags,
    ZrtlSymbol,
    ZrtlSymbolSig,
    ZRTL_VERSION,
};

// Re-export compiler types needed for module compilation
pub use zyntax_compiler::{
    compile_to_hir, CompilationConfig, CompilerError, CompilerResult, HirModule,
};
#[cfg(feature = "native")]
pub use zyntax_compiler::{compile_to_jit, HirFunction};
