pub mod async_runtime; // Async runtime types (Future trait, Poll enum, Context)
pub mod hashmap; // HashMap<K,V> with hash functions
pub mod io;
pub mod iterator; // Iterator-style functions (vec_for_each)
pub mod memory;
/// Standard Library for Zyntax
///
/// This module contains the standard library implementation built using the HIR Builder API.
/// Instead of source-based compilation, we directly construct HIR for all stdlib types and functions.
///
/// This approach provides:
/// - No parser dependency
/// - Type-safe construction
/// - Fast compilation
/// - Version-controlled as Rust code
/// - Easy to test and maintain
pub mod option;
pub mod result;
pub mod string; // UTF-8 string type backed by vec_u8
pub mod vec; // Generic Vec<T> with monomorphization support
pub mod vec_f64; // Performance-optimized concrete vec_f64
pub mod vec_i32; // Performance-optimized concrete vec_i32
pub mod vec_u8; // Performance-optimized concrete vec_u8 (byte buffers) // I/O functions (printf, println, etc.)

use crate::hir::HirModule;
use crate::hir_builder::HirBuilder;
use zyntax_typed_ast::AstArena;

/// Builds the complete standard library as a HIR module
///
/// This function constructs all standard library types and functions
/// using the HIR Builder API.
///
/// # Returns
/// A complete HirModule containing the standard library
pub fn build_stdlib(arena: &mut AstArena) -> HirModule {
    let mut builder = HirBuilder::new("std", arena);

    // Build Option<T> type and methods
    option::build_option_type(&mut builder);

    // Build Result<T,E> type and methods
    result::build_result_type(&mut builder);

    // Build memory management functions
    memory::build_memory_functions(&mut builder);

    // Build Vec<T> generic dynamic array (flexible, works for any type)
    vec::build_vec_type(&mut builder);

    // Build concrete vec types (performance-optimized for specific types)
    vec_i32::build_vec_i32_type(&mut builder); // Integers
    vec_f64::build_vec_f64_type(&mut builder); // Floats
    vec_u8::build_vec_u8_type(&mut builder); // Byte buffers

    // Build String type (UTF-8 strings backed by vec_u8)
    string::build_string_type(&mut builder);

    // Build iterator-style functions
    iterator::build_iterator(&mut builder);

    // Build HashMap<K,V> with hash functions
    hashmap::build_hashmap(&mut builder);

    // Build async runtime types (Future trait, Poll enum, Context)
    async_runtime::build_async_runtime(&mut builder);

    // Build I/O functions (printf, println, etc.)
    io::build_io_functions(&mut builder);
    io::build_print_i32(&mut builder);

    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_builds() {
        let mut arena = AstArena::new();
        let stdlib = build_stdlib(&mut arena);

        // Should have functions from all modules
        assert!(!stdlib.functions.is_empty());

        // Module should be named "std"
        assert_eq!(arena.resolve_string(stdlib.name), Some("std"));
    }
}
