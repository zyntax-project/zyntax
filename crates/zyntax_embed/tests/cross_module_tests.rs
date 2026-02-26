//! Cross-module function call tests for zyntax_embed
//!
//! These tests verify that functions can be exported from one module
//! and called from another module via extern declarations.

use zyntax_embed::{LanguageGrammar, NativeSignature, NativeType, RuntimeError, ZyntaxRuntime};

/// The Zig grammar source (embedded at compile time)
const ZIG_GRAMMAR_SOURCE: &str = include_str!("../../zyn_peg/grammars/zig.zyn");

/// Helper to load the Zig grammar for tests
fn load_zig_grammar() -> Result<LanguageGrammar, Box<dyn std::error::Error>> {
    Ok(LanguageGrammar::compile_zyn(ZIG_GRAMMAR_SOURCE)?)
}

#[test]
fn test_single_module_function_call() {
    // First verify that a single module works
    let grammar = match load_zig_grammar() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: could not load grammar: {}", e);
            return;
        }
    };

    let mut runtime = ZyntaxRuntime::new().expect("Failed to create runtime");
    runtime.register_grammar("zig", grammar);

    // Note: The zig.zyn grammar doesn't have a `pub` modifier, just `fn`
    let functions = runtime
        .load_module(
            "zig",
            r#"
fn add(a: i32, b: i32) i32 {
    return a + b;
}
    "#,
        )
        .expect("Failed to load module");

    // Debug: print all function names
    println!("Loaded functions: {:?}", functions);
    println!("All runtime functions: {:?}", runtime.functions());

    // All functions are tracked (may have multiple due to compiler internals)
    assert!(
        !functions.is_empty(),
        "Should have at least one function: got {:?}",
        functions
    );

    // Use native calling for JIT-compiled functions with signature
    let sig = NativeSignature::new(&[NativeType::I32, NativeType::I32], NativeType::I32);
    let result = runtime
        .call_function("add", &[10.into(), 32.into()], &sig)
        .expect("Failed to call add");
    assert_eq!(result.as_i32().unwrap(), 42);
}

#[test]
fn test_export_function_explicit() {
    let grammar = match load_zig_grammar() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: could not load grammar: {}", e);
            return;
        }
    };

    let mut runtime = ZyntaxRuntime::new().expect("Failed to create runtime");
    runtime.register_grammar("zig", grammar);

    // Load module without exports
    runtime
        .load_module(
            "zig",
            r#"
fn square(x: i32) i32 {
    return x * x;
}
    "#,
        )
        .expect("Failed to load module");

    // Verify no exports yet
    assert!(runtime.exported_symbols().is_empty());

    // Explicitly export the function
    runtime
        .export_function("square")
        .expect("Failed to export function");

    // Verify it's now exported
    let exports = runtime.exported_symbols();
    assert_eq!(exports.len(), 1);
    assert_eq!(exports[0].0, "square");
}

#[test]
fn test_load_module_with_exports() {
    let grammar = match load_zig_grammar() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: could not load grammar: {}", e);
            return;
        }
    };

    let mut runtime = ZyntaxRuntime::new().expect("Failed to create runtime");
    runtime.register_grammar("zig", grammar);

    // Load module with explicit exports
    runtime
        .load_module_with_exports(
            "zig",
            r#"
fn add(a: i32, b: i32) i32 {
    return a + b;
}
fn sub(a: i32, b: i32) i32 {
    return a - b;
}
    "#,
            &["add"],
        )
        .expect("Failed to load module");

    // Only "add" should be exported
    let exports = runtime.exported_symbols();
    assert_eq!(exports.len(), 1);
    assert_eq!(exports[0].0, "add");

    // But both functions should be callable using native calling
    let sig = NativeSignature::new(&[NativeType::I32, NativeType::I32], NativeType::I32);
    let add_result = runtime
        .call_function("add", &[10.into(), 5.into()], &sig)
        .expect("Failed to call add");
    assert_eq!(add_result.as_i32().unwrap(), 15);

    let sub_result = runtime
        .call_function("sub", &[10.into(), 5.into()], &sig)
        .expect("Failed to call sub");
    assert_eq!(sub_result.as_i32().unwrap(), 5);
}

#[test]
fn test_export_conflict_warning() {
    let grammar = match load_zig_grammar() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: could not load grammar: {}", e);
            return;
        }
    };

    let mut runtime = ZyntaxRuntime::new().expect("Failed to create runtime");
    runtime.register_grammar("zig", grammar);

    // Load and export a function
    runtime
        .load_module_with_exports(
            "zig",
            r#"
fn compute(x: i32) i32 {
    return x * 2;
}
    "#,
            &["compute"],
        )
        .expect("Failed to load first module");

    // Check conflict detection
    assert!(runtime.check_export_conflict("compute").is_some());
    assert!(runtime.check_export_conflict("nonexistent").is_none());
}

#[test]
fn test_export_nonexistent_function_error() {
    let grammar = match load_zig_grammar() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: could not load grammar: {}", e);
            return;
        }
    };

    let mut runtime = ZyntaxRuntime::new().expect("Failed to create runtime");
    runtime.register_grammar("zig", grammar);

    runtime
        .load_module(
            "zig",
            r#"
fn real_fn(x: i32) i32 {
    return x;
}
    "#,
        )
        .expect("Failed to load module");

    // Try to export a function that doesn't exist
    let result = runtime.export_function("fake_fn");
    assert!(result.is_err());

    if let Err(RuntimeError::FunctionNotFound(name)) = result {
        assert_eq!(name, "fake_fn");
    } else {
        panic!("Expected FunctionNotFound error");
    }
}

#[test]
fn test_cross_module_function_call() {
    let grammar = match load_zig_grammar() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping test: could not load grammar: {}", e);
            return;
        }
    };

    let mut runtime = ZyntaxRuntime::new().expect("Failed to create runtime");
    runtime.register_grammar("zig", grammar);

    // Module A: Export the 'add' function
    runtime
        .load_module_with_exports(
            "zig",
            r#"
fn add(a: i32, b: i32) i32 {
    return a + b;
}
    "#,
            &["add"],
        )
        .expect("Failed to load module A");

    // Verify the function is exported
    let exports = runtime.exported_symbols();
    println!("Exported symbols: {:?}", exports);
    assert!(!exports.is_empty(), "Should have exported 'add'");

    // Module B: Declare extern and use it
    runtime
        .load_module(
            "zig",
            r#"
extern fn add(a: i32, b: i32) i32;

fn double_add(a: i32, b: i32) i32 {
    return add(a, b) + add(a, b);
}
    "#,
        )
        .expect("Failed to load module B");

    // Call the function that uses the extern
    let sig = NativeSignature::new(&[NativeType::I32, NativeType::I32], NativeType::I32);
    let result = runtime
        .call_function("double_add", &[5.into(), 3.into()], &sig)
        .expect("Failed to call double_add");
    assert_eq!(result.as_i32().unwrap(), 16); // (5+3) + (5+3) = 16
}
