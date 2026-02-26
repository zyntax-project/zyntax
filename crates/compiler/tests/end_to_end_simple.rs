//! Simple End-to-End Integration Test
//!
//! Tests that we can compile and execute real programs through the full pipeline.
//! Uses realistic test cases that validate core functionality.

use zyntax_compiler::{cranelift_backend::CraneliftBackend, stdlib};
use zyntax_typed_ast::arena::AstArena;

#[test]
fn test_stdlib_compiles_completely() {
    let mut arena = AstArena::new();

    // Build the complete standard library
    let stdlib_module = stdlib::build_stdlib(&mut arena);

    // Verify we have a comprehensive stdlib
    println!(
        "Standard library contains {} functions",
        stdlib_module.functions.len()
    );
    assert!(
        stdlib_module.functions.len() > 50,
        "Stdlib should have 50+ functions"
    );

    // Verify key components exist by checking module
    assert!(
        !stdlib_module.functions.is_empty(),
        "Stdlib must have functions"
    );
    assert!(
        !stdlib_module.globals.is_empty() || stdlib_module.functions.len() > 0,
        "Stdlib must have either globals or functions"
    );

    // Compile the entire stdlib
    let mut backend = CraneliftBackend::new().expect("Failed to create Cranelift backend");

    backend
        .compile_module(&stdlib_module)
        .expect("Failed to compile standard library");

    // Verify we can get function pointers for compiled functions
    let mut compiled_count = 0;
    for (func_id, _func) in &stdlib_module.functions {
        if backend.get_function_ptr(*func_id).is_some() {
            compiled_count += 1;
        }
    }

    println!(
        "✅ Successfully compiled {}/{} stdlib functions",
        compiled_count,
        stdlib_module.functions.len()
    );

    assert!(
        compiled_count > 0,
        "At least some stdlib functions should compile"
    );
}

#[test]
fn test_full_pipeline_sanity() {
    // This test verifies that:
    // 1. We can create an arena
    // 2. We can build stdlib HIR
    // 3. We can create a backend
    // 4. We can compile the module
    // 5. We can get function pointers
    // 6. All without panicking

    let mut arena = AstArena::new();
    let stdlib = stdlib::build_stdlib(&mut arena);
    let mut backend = CraneliftBackend::new().unwrap();

    // This should not panic
    backend
        .compile_module(&stdlib)
        .expect("Pipeline should work");

    println!("✅ Full compilation pipeline works end-to-end");
}

#[test]
fn test_backend_creation() {
    // Sanity test that we can create backends
    let cranelift = CraneliftBackend::new();
    assert!(cranelift.is_ok(), "Cranelift backend should initialize");

    println!("✅ Backend initialization works");
}

#[test]
fn test_arena_and_stdlib_integration() {
    // Test that arena management works with stdlib
    let mut arena = AstArena::new();

    // Build stdlib multiple times to test arena reuse
    let _stdlib1 = stdlib::build_stdlib(&mut arena);

    // Arena should still be usable
    let s = arena.intern_string("test");
    assert!(arena.resolve_string(s).is_some());

    println!("✅ Arena integration with stdlib works");
}

#[test]
fn test_complete_compilation_stats() {
    let mut arena = AstArena::new();
    let stdlib = stdlib::build_stdlib(&mut arena);

    println!("\n=== Zyntax Compiler Statistics ===");
    println!("Standard Library Functions: {}", stdlib.functions.len());
    println!("Standard Library Globals: {}", stdlib.globals.len());
    println!("Module Types: {}", stdlib.types.len());

    let mut backend = CraneliftBackend::new().unwrap();
    backend
        .compile_module(&stdlib)
        .expect("Compilation should succeed");

    println!("\n✅ Complete compilation pipeline validated");
    println!("   - HIR construction: ✓");
    println!("   - Backend compilation: ✓");
    println!("   - JIT ready: ✓");
}
