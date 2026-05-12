//! # ZRTL - Zyntax Runtime Library
//!
//! This crate provides types and utilities for creating ZRTL plugins that can be
//! loaded by the Zyntax compiler at runtime. It is the Rust equivalent of `zrtl.h`.
//!
//! ## Features
//!
//! - **Type System**: Type tags, categories, and flags for dynamic typing
//! - **DynamicBox**: Runtime boxed values with type information
//! - **GenericBox**: Support for generic types like `Array<T>`, `Map<K,V>`
//! - **String/Array**: Inline format helpers matching Zyntax's memory layout
//! - **Plugin Support**: Macros for defining ZRTL plugins
//! - **Async Support**: Promise-based async runtime for Zyntax async functions
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use zrtl::{zrtl_plugin, TypeTag, DynamicBox};
//!
//! // Define exported functions
//! extern "C" fn add(a: i32, b: i32) -> i32 {
//!     a + b
//! }
//!
//! extern "C" fn multiply(a: i32, b: i32) -> i32 {
//!     a * b
//! }
//!
//! // Create the plugin
//! zrtl_plugin! {
//!     name: "math_runtime",
//!     symbols: [
//!         ("$Math$add", add),
//!         ("$Math$multiply", multiply),
//!     ]
//! }
//! ```
//!
//! ## Native Async Functions (for Language Frontends)
//!
//! When building a custom language with Zyntax that supports async/await,
//! you can expose native async functions that guest code can await:
//!
//! ```rust,ignore
//! use zrtl::prelude::*;
//! use zrtl_macros::zrtl_async;
//!
//! // This function can be awaited from any Zyntax-based language
//! #[zrtl_async("$IO$read_file")]
//! pub async fn read_file(path_ptr: *const u8) -> i64 {
//!     // Real async I/O - can use tokio, async-std, etc.
//!     let contents = async_read(path_ptr).await;
//!     contents.len() as i64
//! }
//! ```
//!
//! In your custom language:
//! ```text
//! // MyLang source code
//! async fn main() {
//!     let size = await read_file("data.txt");  // Calls native async!
//!     print(size);
//! }
//! ```
//!
//! The async support includes:
//! - `ZrtlPromise`: C ABI promise type for native async functions
//! - `PollResult`: Poll result enum (Pending/Ready/Failed) with i64 ABI
//! - `StateMachineHeader`: Base type for manual state machines
//! - `PromiseAll`, `PromiseRace`, `PromiseAllSettled`: Promise combinators
//!
//! ## Memory Formats
//!
//! Zyntax uses specific inline memory formats:
//!
//! - **Strings**: `[i32 length][utf8_bytes...]`
//! - **Arrays**: `[i32 capacity][i32 length][elements...]`
//!
//! Use the `string` and `array` modules to work with these formats.
//!
//! ## Type Tags
//!
//! Type tags are 32-bit packed identifiers:
//! ```text
//! [flags:8][type_id:16][category:8]
//! ```
//!
//! Use the `zrtl_tag!` macro for compile-time tag creation:
//!
//! ```rust
//! use zrtl::zrtl_tag;
//!
//! let int_tag = zrtl_tag!(i32);
//! let custom_tag = zrtl_tag!(Struct, 42);
//! ```

pub mod array;
pub mod async_support;
pub mod closure;
pub mod dynamic_box;
pub mod generic_box;
pub mod plugin;
pub mod string;
pub mod type_system;

// Re-export main types at crate root
pub use array::{ArrayConstPtr, ArrayIterator, ArrayPtr, OwnedArray};
pub use dynamic_box::{DropFn, DynamicBox};
pub use generic_box::{GenericBox, GenericTypeArgs, MAX_TYPE_ARGS};
pub use plugin::{
    StaticPlugin, TypeInfo, ZrtlInfo, ZrtlSigFlags, ZrtlSymbol, ZrtlSymbolEntry, ZrtlSymbolSig,
    ZrtlTyped, MAX_PARAMS, ZRTL_VERSION,
};
pub use string::{OwnedString, StringConstPtr, StringPtr, StringView};
pub use type_system::{PrimitiveSize, TypeCategory, TypeFlags, TypeTag};

// Re-export async types
pub use async_support::{
    next_task_id, noop_context, noop_waker, sleep, yield_once, AsyncState, FutureAdapter,
    PollResult, PromiseAll, PromiseAllSettled, PromiseError, PromiseRace, SettledResult,
    StateMachineHeader, Timer, YieldOnce, ZrtlPromise,
};

// Re-export closure types
pub use closure::{
    zrtl_closure_call, zrtl_closure_clone, zrtl_closure_free, zrtl_closure_from_fn,
    zrtl_closure_from_raw, zrtl_closure_from_raw_noenv, zrtl_closure_is_null, ClosureResult,
    RawClosureFn, ThreadEntry, ZrtlClosure, ZrtlOnceClosure,
};

// Re-export string functions
pub use string::{
    string_alloc_size, string_as_bytes, string_as_str, string_copy, string_data, string_empty,
    string_equals, string_free, string_length, string_new, STRING_HEADER_SIZE,
};

// Re-export array functions
pub use array::{
    array_alloc_size, array_as_slice, array_capacity, array_data, array_free, array_get,
    array_length, array_new, array_push, array_set, ARRAY_HEADER_BYTES, ARRAY_HEADER_SIZE,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::array::OwnedArray;
    pub use crate::async_support::{
        sleep, yield_once, AsyncState, PollResult, PromiseAll, PromiseAllSettled, PromiseError,
        PromiseRace, SettledResult, StateMachineHeader, ZrtlPromise,
    };
    pub use crate::closure::{ClosureResult, ThreadEntry, ZrtlClosure, ZrtlOnceClosure};
    pub use crate::dynamic_box::DynamicBox;
    pub use crate::generic_box::{GenericBox, GenericTypeArgs};
    pub use crate::plugin::{TypeInfo, ZrtlTyped};
    pub use crate::string::OwnedString;
    pub use crate::type_system::{TypeCategory, TypeFlags, TypeTag};
    pub use crate::zrtl_plugin;
    pub use crate::zrtl_symbol;
    pub use crate::zrtl_tag;
    // Test framework macros
    pub use crate::zrtl_assert;
    pub use crate::zrtl_assert_eq;
    pub use crate::zrtl_assert_err;
    pub use crate::zrtl_assert_ne;
    pub use crate::zrtl_assert_none;
    pub use crate::zrtl_assert_ok;
    pub use crate::zrtl_assert_some;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_tag_roundtrip() {
        let tag = TypeTag::new(
            TypeCategory::Struct,
            123,
            TypeFlags::NULLABLE | TypeFlags::BOXED,
        );

        assert_eq!(tag.category(), TypeCategory::Struct);
        assert_eq!(tag.type_id(), 123);
        assert!(tag.flags().is_nullable());
        assert!(tag.flags().is_boxed());
    }

    #[test]
    fn test_dynamic_box_primitives() {
        let b = DynamicBox::owned_i32(42);
        assert_eq!(b.as_i32(), Some(42));

        let b = DynamicBox::owned_f64(std::f64::consts::PI);
        assert_eq!(b.as_f64(), Some(std::f64::consts::PI));
    }

    #[test]
    fn test_owned_string() {
        let s = OwnedString::from("Hello, ZRTL!");
        assert_eq!(s.len(), 12);
        assert_eq!(s.as_str(), Some("Hello, ZRTL!"));
    }

    #[test]
    fn test_owned_array() {
        let mut arr: OwnedArray<i32> = OwnedArray::new().unwrap();
        arr.push(1);
        arr.push(2);
        arr.push(3);

        assert_eq!(arr.len(), 3);
        assert_eq!(arr.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_generic_box() {
        let args = GenericTypeArgs::array(TypeTag::I32);
        assert_eq!(args.get(0), Some(TypeTag::I32));
    }
}
