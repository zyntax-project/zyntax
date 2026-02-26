use std::fs;
use std::path::PathBuf;
/// Systematic fixture tests from simple to complex
/// Tests real Whirlwind features: functions, models, enums, interfaces, generics, etc.
use whirlwind_adapter::WhirlwindAdapter;
use whirlwind_analyzer::{Module, Standpoint};

/// Helper to load and convert a fixture file
fn test_fixture(fixture_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let fixture_path = format!("tests/fixtures/{}.wrl", fixture_name);
    let mut source = fs::read_to_string(&fixture_path)?;

    // Whirlwind requires module declaration
    if !source.starts_with("module") {
        source = format!("module Test;\n\n{}", source);
    }

    println!("\n=== Testing fixture: {} ===", fixture_name);
    println!("Source:\n{}\n", source);

    // Parse and analyze with Whirlwind
    let mut module = Module::from_text(&source);
    module.module_path = Some(PathBuf::from(format!("testing://{}.wrl", fixture_name)));
    let standpoint =
        Standpoint::build_from_module(module, false).ok_or("Failed to build standpoint")?;

    println!(
        "Symbols found: {}",
        standpoint.symbol_library.symbols().count()
    );

    println!("Diagnostics: {} (Note: UnknownValue errors for 'int', 'string', etc. are expected - Whirlwind core library not available)", standpoint.diagnostics.len());

    // Convert with adapter
    let mut adapter = WhirlwindAdapter::new();
    let typed_program = adapter.convert_standpoint(&standpoint)?;

    println!(
        "Declarations converted: {}",
        typed_program.declarations.len()
    );

    // Print summary of declarations
    for decl in &typed_program.declarations {
        match &decl.node {
            zyntax_typed_ast::typed_ast::TypedDeclaration::Function(f) => {
                if let Some(name) = adapter.arena().resolve_string(f.name) {
                    println!("  ✓ Function: {}", name);
                    if !f.params.is_empty() {
                        println!("      Parameters: {}", f.params.len());
                        for (i, param) in f.params.iter().enumerate() {
                            if let Some(pname) = adapter.arena().resolve_string(param.name) {
                                println!("        [{}] {} : {:?}", i, pname, param.ty);
                            } else {
                                println!("        [{}] <unnamed> : {:?}", i, param.ty);
                            }
                        }
                    }
                    println!("      Return type: {:?}", f.return_type);
                }
            }
            zyntax_typed_ast::typed_ast::TypedDeclaration::Class(c) => {
                if let Some(name) = adapter.arena().resolve_string(c.name) {
                    println!(
                        "  ✓ Class: {} ({} fields, {} methods, {} constructors)",
                        name,
                        c.fields.len(),
                        c.methods.len(),
                        c.constructors.len()
                    );

                    // Print fields
                    for field in &c.fields {
                        if let Some(fname) = adapter.arena().resolve_string(field.name) {
                            println!("      - Field: {} : {:?}", fname, field.ty);
                        }
                    }

                    // Print methods
                    for method in &c.methods {
                        if let Some(mname) = adapter.arena().resolve_string(method.name) {
                            println!("      - Method: {}() -> {:?}", mname, method.return_type);
                        }
                    }
                }
            }
            zyntax_typed_ast::typed_ast::TypedDeclaration::TypeAlias(ta) => {
                if let Some(name) = adapter.arena().resolve_string(ta.name) {
                    println!("  ✓ TypeAlias: {}", name);
                }
            }
            zyntax_typed_ast::typed_ast::TypedDeclaration::Enum(e) => {
                if let Some(name) = adapter.arena().resolve_string(e.name) {
                    println!("  ✓ Enum: {} ({} variants)", name, e.variants.len());
                }
            }
            zyntax_typed_ast::typed_ast::TypedDeclaration::Interface(i) => {
                if let Some(name) = adapter.arena().resolve_string(i.name) {
                    println!("  ✓ Interface: {} ({} methods)", name, i.methods.len());
                }
            }
            _ => {
                println!("  ✓ Other declaration");
            }
        }
    }

    Ok(())
}

// ============================================================================
// LEVEL 1: Simple constructs
// ============================================================================

#[test]
fn test_simple_function() {
    test_fixture("simple_function").expect("simple_function should convert");
}

#[test]
fn test_type_alias() {
    test_fixture("type_alias").expect("type_alias should convert");
}

#[test]
fn test_simple_enum() {
    test_fixture("simple_enum").expect("simple_enum should convert");
}

// ============================================================================
// LEVEL 2: Models and control flow
// ============================================================================

#[test]
fn test_simple_model() {
    test_fixture("simple_model").expect("simple_model should convert");
}

#[test]
fn test_control_flow() {
    test_fixture("control_flow").expect("control_flow should convert");
}

// ============================================================================
// LEVEL 3: Advanced types
// ============================================================================

#[test]
fn test_optional_types() {
    test_fixture("optional_types").expect("optional_types should convert");
}

#[test]
fn test_enum_record() {
    test_fixture("enum_record").expect("enum_record should convert");
}

#[test]
fn test_union_types() {
    test_fixture("union_types").expect("union_types should convert");
}

// ============================================================================
// LEVEL 4: Inheritance and interfaces
// ============================================================================

#[test]
fn test_model_interface() {
    test_fixture("model_interface").expect("model_interface should convert");
}

// ============================================================================
// LEVEL 5: Generics
// ============================================================================

#[test]
fn test_generic_function() {
    test_fixture("generic_function").expect("generic_function should convert");
}
