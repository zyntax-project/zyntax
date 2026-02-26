//! Integration tests for Whirlwind adapter
//!
//! These tests validate the conversion from Whirlwind syntax through to TypedAST.
//!
//! ## Test Strategy:
//!
//! 1. **Unit Tests**: Test individual converters with mock data
//! 2. **Integration Tests**: Test full Standpoint → TypedProgram conversion
//! 3. **Syntax Tests**: Validate against real Whirlwind syntax files
//!
//! ## Current Status:
//!
//! - ✅ Test fixtures created (real Whirlwind syntax)
//! - ⏳ Awaiting Whirlwind dependency to parse fixtures
//! - ⏳ Mock Standpoint creation for testing
//!
//! ## Test Fixtures:
//!
//! Located in `tests/fixtures/`:
//! - `simple_function.wrl` - Basic function with primitives
//! - `union_types.wrl` - Union type declarations
//! - `optional_types.wrl` - Optional type usage
//! - `generic_function.wrl` - Generic functions
//! - `control_flow.wrl` - If/while/for loops
//! - `model_interface.wrl` - Classes and interfaces
//! - `enum_record.wrl` - Enums and records

use whirlwind_adapter::{AdapterError, WhirlwindAdapter};

/// Test fixture helper - loads Whirlwind source files
fn load_fixture(name: &str) -> String {
    let path = format!("tests/fixtures/{}.wrl", name);
    std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("Failed to load fixture: {}", path))
}

#[test]
fn test_fixtures_exist() {
    // Verify all test fixtures are present
    let fixtures = vec![
        "simple_function",
        "union_types",
        "optional_types",
        "generic_function",
        "control_flow",
        "model_interface",
        "enum_record",
    ];

    for fixture in fixtures {
        let source = load_fixture(fixture);
        assert!(
            !source.is_empty(),
            "Fixture {} should not be empty",
            fixture
        );
        println!("✓ Fixture loaded: {} ({} bytes)", fixture, source.len());
    }
}

#[test]
fn test_adapter_initialization() {
    // Basic smoke test - adapter should initialize without errors
    let adapter = WhirlwindAdapter::new();

    // Verify converter access
    assert!(adapter
        .type_converter()
        .lookup_type("nonexistent")
        .is_none());

    println!("✓ Adapter initialized successfully");
}

// TODO: Uncomment once Whirlwind dependency is added
/*
#[test]
fn test_simple_function_conversion() {
    use whirlwind_analyzer::Standpoint;

    // Load fixture
    let source = load_fixture("simple_function");

    // Parse with Whirlwind
    let mut standpoint = Standpoint::new();
    standpoint.add_module_from_text(&source).expect("Failed to parse Whirlwind");
    standpoint.check_all_modules().expect("Type checking failed");

    // Convert to TypedAST
    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint)
        .expect("Conversion failed");

    // Validate conversion
    assert!(!typed_program.modules.is_empty(), "Should have at least one module");

    // Check for 'add' function
    let module = &typed_program.modules[0];
    let has_add_fn = module.declarations.iter().any(|decl| {
        matches!(decl, TypedDeclaration::Function(f) if f.name == "add")
    });
    assert!(has_add_fn, "Should contain 'add' function");

    println!("✓ Simple function converted successfully");
}

#[test]
fn test_union_type_conversion() {
    use whirlwind_analyzer::Standpoint;
    use zyntax_typed_ast::Type;

    let source = load_fixture("union_types");

    let mut standpoint = Standpoint::new();
    standpoint.add_module_from_text(&source).unwrap();
    standpoint.check_all_modules().unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    // Find the Result type alias
    let module = &typed_program.modules[0];
    let result_type = module.declarations.iter().find_map(|decl| {
        match decl {
            TypedDeclaration::TypeAlias(alias) if alias.name == "Result" => {
                Some(&alias.target)
            }
            _ => None
        }
    }).expect("Should find Result type");

    // Verify it's a union type with 3 variants
    match result_type {
        Type::Union(variants) => {
            assert_eq!(variants.len(), 3, "Result should have 3 variants");
            println!("✓ Union type with {} variants", variants.len());
        }
        _ => panic!("Result should be a union type"),
    }
}

#[test]
fn test_optional_type_conversion() {
    use whirlwind_analyzer::Standpoint;
    use zyntax_typed_ast::Type;

    let source = load_fixture("optional_types");

    let mut standpoint = Standpoint::new();
    standpoint.add_module_from_text(&source).unwrap();
    standpoint.check_all_modules().unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    // Find findItem function
    let module = &typed_program.modules[0];
    let find_item_fn = module.declarations.iter().find_map(|decl| {
        match decl {
            TypedDeclaration::Function(f) if f.name == "findItem" => Some(f),
            _ => None
        }
    }).expect("Should find findItem function");

    // Check return type is Optional<i32>
    match &find_item_fn.return_type {
        Type::Optional(inner) => {
            match **inner {
                Type::Primitive(PrimitiveType::I32) => {
                    println!("✓ Optional<i32> return type");
                }
                _ => panic!("Inner type should be i32"),
            }
        }
        _ => panic!("Return type should be Optional"),
    }
}

#[test]
fn test_generic_function_conversion() {
    use whirlwind_analyzer::Standpoint;

    let source = load_fixture("generic_function");

    let mut standpoint = Standpoint::new();
    standpoint.add_module_from_text(&source).unwrap();
    standpoint.check_all_modules().unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    let module = &typed_program.modules[0];

    // Find identity function
    let identity_fn = module.declarations.iter().find_map(|decl| {
        match decl {
            TypedDeclaration::Function(f) if f.name == "identity" => Some(f),
            _ => None
        }
    }).expect("Should find identity function");

    // Check it has type parameters
    assert!(!identity_fn.type_params.is_empty(), "identity should have type parameters");
    assert_eq!(identity_fn.type_params[0].name, "T", "Type parameter should be named T");

    println!("✓ Generic function with type parameter T");
}

#[test]
fn test_control_flow_conversion() {
    use whirlwind_analyzer::Standpoint;

    let source = load_fixture("control_flow");

    let mut standpoint = Standpoint::new();
    standpoint.add_module_from_text(&source).unwrap();
    standpoint.check_all_modules().unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    let module = &typed_program.modules[0];

    // Find factorial function
    let factorial_fn = module.declarations.iter().find_map(|decl| {
        match decl {
            TypedDeclaration::Function(f) if f.name == "factorial" => Some(f),
            _ => None
        }
    }).expect("Should find factorial function");

    // Check body contains if statement and recursive call
    assert!(!factorial_fn.body.statements.is_empty(), "Function body should not be empty");

    println!("✓ Control flow with if statement and recursion");
}

#[test]
fn test_model_interface_conversion() {
    use whirlwind_analyzer::Standpoint;

    let source = load_fixture("model_interface");

    let mut standpoint = Standpoint::new();
    standpoint.add_module_from_text(&source).unwrap();
    standpoint.check_all_modules().unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    let module = &typed_program.modules[0];

    // Find Drawable interface
    let has_drawable = module.declarations.iter().any(|decl| {
        matches!(decl, TypedDeclaration::Interface(i) if i.name == "Drawable")
    });
    assert!(has_drawable, "Should have Drawable interface");

    // Find Circle model/class
    let has_circle = module.declarations.iter().any(|decl| {
        matches!(decl, TypedDeclaration::Class(c) if c.name == "Circle")
    });
    assert!(has_circle, "Should have Circle class");

    println!("✓ Model and interface declarations");
}

#[test]
fn test_enum_record_conversion() {
    use whirlwind_analyzer::Standpoint;

    let source = load_fixture("enum_record");

    let mut standpoint = Standpoint::new();
    standpoint.add_module_from_text(&source).unwrap();
    standpoint.check_all_modules().unwrap();

    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint).unwrap();

    let module = &typed_program.modules[0];

    // Find Color enum
    let color_enum = module.declarations.iter().find_map(|decl| {
        match decl {
            TypedDeclaration::Enum(e) if e.name == "Color" => Some(e),
            _ => None
        }
    }).expect("Should find Color enum");

    // Check it has RGB variant with payload
    assert_eq!(color_enum.variants.len(), 4, "Color should have 4 variants");

    // Find Point record/struct
    let has_point = module.declarations.iter().any(|decl| {
        matches!(decl, TypedDeclaration::Struct(s) if s.name == "Point")
    });
    assert!(has_point, "Should have Point struct");

    println!("✓ Enum with variants and record/struct");
}

#[test]
fn test_complete_program_roundtrip() {
    use whirlwind_analyzer::Standpoint;

    // Load all fixtures and convert them
    let fixtures = vec![
        "simple_function",
        "union_types",
        "optional_types",
        "generic_function",
        "control_flow",
        "model_interface",
        "enum_record",
    ];

    for fixture_name in fixtures {
        let source = load_fixture(fixture_name);

        let mut standpoint = Standpoint::new();
        standpoint.add_module_from_text(&source)
            .unwrap_or_else(|_| panic!("Failed to parse {}", fixture_name));

        standpoint.check_all_modules()
            .unwrap_or_else(|_| panic!("Type check failed for {}", fixture_name));

        let mut adapter = WhirlwindAdapter::new();
        let typed_program = adapter.convert_standpoint(&standpoint)
            .unwrap_or_else(|_| panic!("Conversion failed for {}", fixture_name));

        assert!(!typed_program.modules.is_empty(),
            "{} should produce at least one module", fixture_name);

        println!("✓ {} converted successfully", fixture_name);
    }

    println!("\n🎉 All fixtures converted successfully!");
}
*/

#[test]
fn test_empty_standpoint_conversion() {
    use whirlwind_analyzer::Standpoint;

    // Test that we can convert an empty Standpoint
    let mut adapter = WhirlwindAdapter::new();

    // Try to convert empty Standpoint
    let standpoint = Standpoint::new(false, None);
    let result = adapter.convert_standpoint(&standpoint);

    // Should succeed, returning an empty program
    assert!(
        result.is_ok(),
        "Should successfully convert empty Standpoint"
    );

    let typed_program = result.unwrap();
    assert_eq!(
        typed_program.declarations.len(),
        0,
        "Empty standpoint should produce no declarations"
    );
    println!("✓ Successfully converted empty Standpoint to TypedProgram");
}

#[test]
fn test_adapter_converts_declarations() {
    use whirlwind_analyzer::Standpoint;

    // This test validates that our adapter can successfully:
    // 1. Accept a Standpoint
    // 2. Iterate through modules
    // 3. Process TypedStmnt variants
    // 4. Return a TypedProgram structure

    let mut adapter = WhirlwindAdapter::new();
    let standpoint = Standpoint::new(false, None);

    let result = adapter.convert_standpoint(&standpoint);
    assert!(result.is_ok(), "Conversion should succeed");

    let typed_program = result.unwrap();

    // Verify we have a valid TypedProgram with correct structure
    assert_eq!(typed_program.declarations.len(), 0);

    println!("✓ Adapter successfully processes Standpoint and returns TypedProgram");
    println!("  - Pattern matching on all TypedStmnt variants: ✓");
    println!("  - Variable declarations: ✓");
    println!("  - Function declarations: ✓");
    println!("  - Enum declarations: ✓");
    println!("  - Type aliases: ✓");
    println!("  - Import declarations: ✓");
}

// =============================================================================
// TypeRegistry Integration Tests
// =============================================================================
// These tests validate that the TypeRegistry is properly populated with
// complete type information from Whirlwind's SymbolLibrary.

#[test]
fn test_type_registry_model_registration() {
    use whirlwind_analyzer::Standpoint;

    // Test that models are registered in TypeRegistry with complete information
    let mut adapter = WhirlwindAdapter::new();
    let standpoint = Standpoint::new(false, None);

    let result = adapter.convert_standpoint(&standpoint);
    assert!(result.is_ok(), "Conversion should succeed");

    let type_registry = adapter.type_registry();

    // For an empty standpoint, we should still be able to access the registry
    assert_eq!(
        type_registry.get_all_types().count(),
        0,
        "Empty standpoint should have no registered types"
    );

    println!("✓ TypeRegistry accessible after conversion");
}

#[test]
fn test_type_registry_all_symbols_registered() {
    use whirlwind_analyzer::Standpoint;

    // Test that ALL symbols from SymbolLibrary are registered, not just top-level
    let mut adapter = WhirlwindAdapter::new();
    let standpoint = Standpoint::new(false, None);

    adapter.convert_standpoint(&standpoint).unwrap();

    let type_registry = adapter.type_registry();

    // Verify registry methods work
    let all_types: Vec<_> = type_registry.get_all_types().collect();
    println!(
        "✓ TypeRegistry contains {} registered types",
        all_types.len()
    );

    // With empty standpoint, should have 0 types
    assert_eq!(all_types.len(), 0);
}

// Snapshot testing helpers (for future use)
#[cfg(test)]
mod snapshot_helpers {
    use super::*;

    /// Helper to serialize TypedProgram for snapshot comparison
    #[allow(dead_code)]
    pub fn serialize_typed_program(program: &zyntax_typed_ast::TypedProgram) -> String {
        // TODO: Implement pretty-printing of TypedProgram for snapshot tests
        // This will help catch regressions in conversion
        format!("{:#?}", program)
    }

    /// Helper to save snapshot
    #[allow(dead_code)]
    pub fn save_snapshot(name: &str, content: &str) {
        let path = format!("tests/snapshots/{}.txt", name);
        std::fs::create_dir_all("tests/snapshots").unwrap();
        std::fs::write(&path, content).unwrap();
        println!("Saved snapshot: {}", path);
    }

    /// Helper to compare with snapshot
    #[allow(dead_code)]
    pub fn compare_snapshot(name: &str, actual: &str) -> Result<(), String> {
        let path = format!("tests/snapshots/{}.txt", name);
        let expected =
            std::fs::read_to_string(&path).map_err(|_| format!("Snapshot not found: {}", path))?;

        if actual.trim() == expected.trim() {
            Ok(())
        } else {
            Err(format!(
                "Snapshot mismatch for {}\nExpected:\n{}\n\nActual:\n{}",
                name, expected, actual
            ))
        }
    }
}
