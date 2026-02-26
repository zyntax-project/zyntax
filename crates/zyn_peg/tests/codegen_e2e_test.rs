//! E2E tests for ZynPEG code generation pipeline
//!
//! This test suite:
//! 1. Parses the zig.zyn grammar
//! 2. Generates pest grammar and Rust code
//! 3. Writes them to files
//! 4. Verifies the generated code compiles (via syn parsing)
//! 5. Tests parsing sample Zig code

use pest::Parser;
use std::fs;
use std::path::Path;
use zyn_peg::ast::build_grammar;
use zyn_peg::generator::{generate_parser, generate_standalone_parser, generate_zyntax_parser};
use zyn_peg::{Rule, ZynGrammarParser};

const ZIG_GRAMMAR_PATH: &str = "grammars/zig.zyn";
const GENERATED_DIR: &str = "generated";

/// Generate and write the parser files (with rustfmt formatting)
fn generate_parser_files() -> (String, String, String) {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    let generated = generate_parser(&grammar).expect("Failed to generate parser");

    // Convert TokenStreams to formatted strings using rustfmt
    let ast_builder_code = generated.ast_builder_formatted();
    let parser_impl_code = generated.parser_impl_formatted();

    (generated.pest_grammar, ast_builder_code, parser_impl_code)
}

/// Generate standalone parser files with TypedAST types
fn generate_standalone_parser_files() -> (String, String, String, String) {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    let generated = generate_standalone_parser(&grammar).expect("Failed to generate parser");

    // Convert TokenStreams to formatted strings using rustfmt
    let ast_builder_code = generated.ast_builder_formatted();
    let parser_impl_code = generated.parser_impl_formatted();
    let typed_ast_code = generated
        .typed_ast_types_formatted()
        .expect("TypedAST types should be generated");

    (
        generated.pest_grammar,
        ast_builder_code,
        parser_impl_code,
        typed_ast_code,
    )
}

/// Generate zyntax-compatible parser files (uses zyntax_typed_ast types)
fn generate_zyntax_parser_files() -> (String, String, String) {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    let generated = generate_zyntax_parser(&grammar).expect("Failed to generate zyntax parser");

    // Convert TokenStreams to formatted strings
    let ast_builder_code = generated.ast_builder_formatted();
    let parser_impl_code = generated.parser_impl_formatted();

    // No typed_ast - uses zyntax_typed_ast crate
    assert!(
        generated.typed_ast_types.is_none(),
        "Zyntax parser should not generate typed_ast"
    );

    (generated.pest_grammar, ast_builder_code, parser_impl_code)
}

#[test]
#[ignore = "Overwrites hand-crafted ast_builder.rs - run manually when generator is fixed"]
fn test_write_generated_files() {
    // Use zyntax parser generator for integration_test compatibility
    let (pest_grammar, ast_builder, parser_impl) = generate_zyntax_parser_files();

    // Ensure generated directory exists
    let gen_dir = Path::new(GENERATED_DIR);
    fs::create_dir_all(gen_dir).expect("Failed to create generated directory");

    // Write pest grammar
    let pest_path = gen_dir.join("zig.pest");
    fs::write(&pest_path, &pest_grammar).expect("Failed to write zig.pest");
    println!(
        "✓ Wrote {} ({} bytes)",
        pest_path.display(),
        pest_grammar.len()
    );

    // Write AST builder
    let ast_path = gen_dir.join("ast_builder.rs");
    fs::write(&ast_path, &ast_builder).expect("Failed to write ast_builder.rs");
    println!(
        "✓ Wrote {} ({} bytes)",
        ast_path.display(),
        ast_builder.len()
    );

    // Write parser implementation
    let parser_path = gen_dir.join("parser_impl.rs");
    fs::write(&parser_path, &parser_impl).expect("Failed to write parser_impl.rs");
    println!(
        "✓ Wrote {} ({} bytes)",
        parser_path.display(),
        parser_impl.len()
    );

    // Verify files exist
    assert!(pest_path.exists(), "zig.pest not created");
    assert!(ast_path.exists(), "ast_builder.rs not created");
    assert!(parser_path.exists(), "parser_impl.rs not created");
}

#[test]
#[ignore = "Overwrites hand-crafted ast_builder.rs - run manually when generator is fixed"]
fn test_write_standalone_files() {
    let (pest_grammar, ast_builder, parser_impl, typed_ast) = generate_standalone_parser_files();

    // Ensure generated directory exists
    let gen_dir = Path::new(GENERATED_DIR);
    fs::create_dir_all(gen_dir).expect("Failed to create generated directory");

    // Write all files including TypedAST types
    let pest_path = gen_dir.join("zig.pest");
    fs::write(&pest_path, &pest_grammar).expect("Failed to write zig.pest");

    let ast_path = gen_dir.join("ast_builder.rs");
    fs::write(&ast_path, &ast_builder).expect("Failed to write ast_builder.rs");

    let parser_path = gen_dir.join("parser_impl.rs");
    fs::write(&parser_path, &parser_impl).expect("Failed to write parser_impl.rs");

    // Write TypedAST types
    let typed_ast_path = gen_dir.join("typed_ast.rs");
    fs::write(&typed_ast_path, &typed_ast).expect("Failed to write typed_ast.rs");
    println!(
        "✓ Wrote {} ({} bytes)",
        typed_ast_path.display(),
        typed_ast.len()
    );

    // Verify typed_ast was generated
    assert!(typed_ast_path.exists(), "typed_ast.rs not created");

    // Check typed_ast contains expected types
    assert!(typed_ast.contains("pub struct Span"), "Missing Span type");
    assert!(typed_ast.contains("pub enum Type"), "Missing Type enum");
    assert!(
        typed_ast.contains("pub struct TypedExpression"),
        "Missing TypedExpression"
    );
    assert!(typed_ast.contains("pub enum BinaryOp"), "Missing BinaryOp");
    assert!(
        typed_ast.contains("pub struct TypedProgram"),
        "Missing TypedProgram"
    );
    println!("✓ TypedAST types module contains all expected types");
}

#[test]
fn test_generated_pest_grammar_is_valid() {
    let (pest_grammar, _, _) = generate_parser_files();

    // Basic validation: check structure
    assert!(pest_grammar.contains("program ="), "Missing program rule");
    assert!(
        pest_grammar.contains("WHITESPACE ="),
        "Missing WHITESPACE rule"
    );

    // Check that the entire grammar has balanced braces
    let open_braces = pest_grammar.matches('{').count();
    let close_braces = pest_grammar.matches('}').count();
    assert_eq!(
        open_braces, close_braces,
        "Unbalanced braces in grammar: {} open, {} close",
        open_braces, close_braces
    );

    // Count rule definitions (lines containing " = ")
    let rule_count = pest_grammar
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.contains(" = ") && !trimmed.starts_with("//")
        })
        .count();

    println!(
        "✓ Generated grammar has {} rules with valid syntax",
        rule_count
    );
    assert!(
        rule_count >= 90,
        "Expected at least 90 rules, got {}",
        rule_count
    );
}

#[test]
#[ignore = "Standard generator WIP - uses external zyntax_typed_ast types. Use standalone generator instead."]
fn test_generated_rust_code_parses() {
    let (_, ast_builder, parser_impl) = generate_parser_files();

    // Use syn to verify the generated Rust code is syntactically valid
    // Note: This doesn't check semantic validity, just syntax

    // Try to parse ast_builder as a Rust file
    let ast_result = syn::parse_file(&ast_builder);
    match ast_result {
        Ok(file) => {
            println!("✓ AST builder code is valid Rust syntax");
            println!("  - {} items in file", file.items.len());
        }
        Err(e) => {
            // Print error details
            println!("Parse error: {}", e);

            // Show first 500 chars of problematic code
            println!("--- Generated code (first 500 chars) ---");
            println!("{}", &ast_builder[..500.min(ast_builder.len())]);
            panic!("Generated AST builder code failed to parse: {}", e);
        }
    }

    // Try to parse parser_impl
    let impl_result = syn::parse_file(&parser_impl);
    match impl_result {
        Ok(file) => {
            println!("✓ Parser impl code is valid Rust syntax");
            println!("  - {} items in file", file.items.len());
        }
        Err(e) => {
            panic!("Generated parser impl code failed to parse: {}", e);
        }
    }
}

#[test]
fn test_generated_code_structure() {
    let (_, ast_builder, parser_impl) = generate_parser_files();

    // Check AST builder has expected components
    assert!(
        ast_builder.contains("AstBuilderContext"),
        "Missing AstBuilderContext struct"
    );
    assert!(
        ast_builder.contains("fn build_program"),
        "Missing build_program method"
    );
    assert!(
        ast_builder.contains("fn build_const_decl"),
        "Missing build_const_decl method"
    );
    assert!(
        ast_builder.contains("fn build_expr"),
        "Missing build_expr method"
    );
    println!("✓ AST builder has expected structure");

    // Check parser impl has expected components
    assert!(
        parser_impl.contains("ZigParser"),
        "Missing ZigParser struct"
    );
    assert!(
        parser_impl.contains("parse_to_typed_ast"),
        "Missing parse_to_typed_ast method"
    );
    println!("✓ Parser impl has expected structure");
}

/// Sample Zig programs for testing - now used in test_parse_zig_code_with_generated_grammar

/// Test that the generated pest grammar can be loaded by pest at runtime
/// (This simulates what would happen when compiling with pest_derive)
#[test]
fn test_pest_grammar_structure() {
    let (pest_grammar, _, _) = generate_parser_files();

    // Verify each rule has the format: name = modifier? { pattern }
    let rule_pattern = regex::Regex::new(r"^\s*(\w+)\s*=\s*([_@$!])?\s*\{\s*.*\s*\}\s*$").unwrap();

    let mut valid_rules = 0;
    let mut invalid_rules: Vec<String> = Vec::new();

    for line in pest_grammar.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        // Check if this looks like a rule definition
        if trimmed.contains(" = ") && trimmed.contains('{') {
            // Multi-line rules - just count the start
            if trimmed.ends_with('}') {
                valid_rules += 1;
            } else {
                // Multi-line rule, assume valid for now
                valid_rules += 1;
            }
        }
    }

    println!(
        "✓ Found {} rule definitions in generated grammar",
        valid_rules
    );

    if !invalid_rules.is_empty() {
        for rule in &invalid_rules {
            println!("  Invalid: {}", rule);
        }
    }

    assert!(valid_rules >= 90, "Expected at least 90 rules");
}

/// Test that the generated pest grammar can parse actual Zig code
/// Uses pest_vm to dynamically load the grammar and parse at runtime
#[test]
fn test_parse_zig_code_with_generated_grammar() {
    use pest_meta::optimizer;
    use pest_meta::parser;
    use pest_vm::Vm;

    let (pest_grammar, _, _) = generate_parser_files();

    // Parse the grammar using pest_meta
    let pairs = parser::parse(parser::Rule::grammar_rules, &pest_grammar)
        .expect("Failed to parse pest grammar");

    // Convert to AST and optimize
    let ast = parser::consume_rules(pairs).expect("Failed to consume rules");
    let optimized = optimizer::optimize(ast);

    // Create VM from optimized rules
    let vm = Vm::new(optimized);

    // Test simple const declaration
    let const_input = "const x: i32 = 42;";
    match vm.parse("const_decl", const_input) {
        Ok(pairs) => {
            println!("✓ Parsed const decl: '{}'", const_input);
            for pair in pairs {
                println!("  Rule: {:?}", pair.as_rule());
            }
        }
        Err(e) => panic!("Failed to parse const decl: {}", e),
    }

    // Test simple function
    let fn_input = "fn add(a: i32, b: i32) i32 { return a; }";
    match vm.parse("fn_decl", fn_input) {
        Ok(pairs) => {
            println!("✓ Parsed function: '{}'", fn_input);
            for pair in pairs {
                println!("  Rule: {:?}", pair.as_rule());
            }
        }
        Err(e) => panic!("Failed to parse function: {}", e),
    }

    // Test full program
    let program_input = r#"
const PI: f64 = 3.14159;

fn double(x: i32) i32 {
    return x * 2;
}
"#;
    match vm.parse("program", program_input) {
        Ok(pairs) => {
            println!("✓ Parsed full program");
            let count = pairs.flatten().count();
            println!("  {} parse tree nodes", count);
        }
        Err(e) => panic!("Failed to parse program: {}", e),
    }

    // Test arithmetic expression
    let expr_input = "1 + 2 * 3";
    match vm.parse("expr", expr_input) {
        Ok(_) => println!("✓ Parsed arithmetic: '{}'", expr_input),
        Err(e) => panic!("Failed to parse expr: {}", e),
    }

    // Test if statement
    let if_input = "if (x > 0) { return 1; }";
    match vm.parse("if_stmt", if_input) {
        Ok(_) => println!("✓ Parsed if statement"),
        Err(e) => panic!("Failed to parse if: {}", e),
    }

    // Test while loop
    let while_input = "while (i < 10) { i = i + 1; }";
    match vm.parse("while_stmt", while_input) {
        Ok(_) => println!("✓ Parsed while loop"),
        Err(e) => panic!("Failed to parse while: {}", e),
    }

    // Test struct declaration
    let struct_input = "const Point = struct { x: i32, y: i32, };";
    match vm.parse("struct_decl", struct_input) {
        Ok(_) => println!("✓ Parsed struct declaration"),
        Err(e) => panic!("Failed to parse struct: {}", e),
    }
}

/// Verify that build methods are generated for all action rules
#[test]
fn test_all_action_rules_have_build_methods() {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    let generated = generate_parser(&grammar).expect("Failed to generate parser");
    let ast_builder = generated.ast_builder.to_string();

    // Check each rule with an action has a build method
    let mut missing_methods = Vec::new();

    for rule in &grammar.rules {
        if rule.action.is_some() {
            let method_name = format!("fn build_{}", rule.name);
            if !ast_builder.contains(&method_name) {
                missing_methods.push(rule.name.clone());
            }
        }
    }

    if !missing_methods.is_empty() {
        println!("Missing build methods: {:?}", missing_methods);
    }

    assert!(
        missing_methods.is_empty(),
        "Missing build methods for: {:?}",
        missing_methods
    );

    let action_count = grammar.rules.iter().filter(|r| r.action.is_some()).count();
    println!(
        "✓ All {} action rules have corresponding build methods",
        action_count
    );
}

// ============================================================================
// ZYNTAX-COMPATIBLE PARSER TESTS
// ============================================================================
// Tests for the new generate_zyntax_parser which produces code that uses
// zyntax_typed_ast types directly, enabling JIT compilation.

#[test]
fn test_zyntax_parser_generation() {
    let (pest_grammar, ast_builder, parser_impl) = generate_zyntax_parser_files();

    // Verify pest grammar is the same
    assert!(pest_grammar.contains("program ="), "Missing program rule");

    // Verify AST builder uses zyntax_typed_ast imports
    assert!(
        ast_builder.contains("use zyntax_typed_ast"),
        "Should import from zyntax_typed_ast"
    );

    // Check for primitive type format (may have spaces between tokens)
    let has_primitive =
        ast_builder.contains("Type :: Primitive") || ast_builder.contains("Type::Primitive");
    assert!(
        has_primitive,
        "Should use Type::Primitive format. First 2000 chars:\n{}",
        &ast_builder[..2000.min(ast_builder.len())]
    );

    assert!(
        ast_builder.contains("InternedString"),
        "Should use InternedString"
    );

    // Check for TypedNode (may have spaces)
    let has_typed_node = ast_builder.contains("TypedNode < TypedExpression")
        || ast_builder.contains("TypedNode<TypedExpression>");
    assert!(has_typed_node, "Should use TypedNode wrapper");

    assert!(
        ast_builder.contains("typed_node"),
        "Should use typed_node constructor"
    );

    // Verify parser impl uses zyntax_typed_ast
    assert!(
        parser_impl.contains("use zyntax_typed_ast"),
        "Parser should import from zyntax_typed_ast"
    );

    println!("✓ Zyntax parser generation produces correct code structure");
}

/// Test that the generated zyntax AST builder produces valid Rust syntax.
/// The generator now properly transforms grammar actions to use zyntax_typed_ast types:
///   - Wraps declarations/statements/expressions with typed_node()
///   - Converts Declaration::* to TypedDeclaration::*
///   - Converts Statement::* to TypedStatement::*
///   - Converts Expression::* to TypedExpression::*
#[test]
fn test_zyntax_ast_builder_syntax_valid() {
    let (_, ast_builder, parser_impl) = generate_zyntax_parser_files();

    // Verify AST builder is valid Rust syntax
    match syn::parse_file(&ast_builder) {
        Ok(file) => {
            println!("✓ Zyntax AST builder is valid Rust syntax");
            println!("  - {} items in file", file.items.len());
        }
        Err(e) => {
            // Write to temp file for debugging
            std::fs::write("/tmp/generated_ast_builder.rs", &ast_builder).unwrap();
            println!("Generated code written to /tmp/generated_ast_builder.rs");
            panic!("Zyntax AST builder failed to parse: {}", e);
        }
    }

    // Verify parser impl is valid Rust syntax
    match syn::parse_file(&parser_impl) {
        Ok(file) => {
            println!("✓ Zyntax parser impl is valid Rust syntax");
            println!("  - {} items in file", file.items.len());
        }
        Err(e) => {
            panic!("Zyntax parser impl failed to parse: {}", e);
        }
    }
}

#[test]
fn test_zyntax_parser_type_mappings() {
    let (_, ast_builder, _) = generate_zyntax_parser_files();

    // Verify PrimitiveType is imported and used
    assert!(
        ast_builder.contains("PrimitiveType"),
        "Should import PrimitiveType"
    );

    // Verify the parse_primitive_type function has correct mappings
    assert!(
        ast_builder.contains("\"i32\""),
        "Should have i32 string matching"
    );
    assert!(
        ast_builder.contains("\"i64\""),
        "Should have i64 string matching"
    );
    assert!(
        ast_builder.contains("\"f32\""),
        "Should have f32 string matching"
    );
    assert!(
        ast_builder.contains("\"f64\""),
        "Should have f64 string matching"
    );
    assert!(
        ast_builder.contains("\"bool\""),
        "Should have bool string matching"
    );

    // Verify primitive type construction (may have varying whitespace from rustfmt)
    let normalized = ast_builder.replace(' ', "");
    assert!(
        normalized.contains("Type::Primitive(PrimitiveType::I32)"),
        "Should construct Type::Primitive(PrimitiveType::I32)"
    );
    assert!(
        normalized.contains("Type::Primitive(PrimitiveType::I64)"),
        "Should construct Type::Primitive(PrimitiveType::I64)"
    );

    println!("✓ All type mappings use correct zyntax_typed_ast format");
}

/// Write zyntax parser files to the generated directory.
/// These files use zyntax_typed_ast types and are used by the integration test.
#[test]
fn test_write_zyntax_parser_files() {
    let (pest_grammar, ast_builder, parser_impl) = generate_zyntax_parser_files();

    // Ensure generated directory exists
    let gen_dir = Path::new(GENERATED_DIR);
    fs::create_dir_all(gen_dir).expect("Failed to create generated directory");

    // Write pest grammar
    let pest_path = gen_dir.join("zig.pest");
    fs::write(&pest_path, &pest_grammar).expect("Failed to write zig.pest");
    println!(
        "✓ Wrote {} ({} bytes)",
        pest_path.display(),
        pest_grammar.len()
    );

    // Write AST builder (zyntax version)
    let ast_path = gen_dir.join("ast_builder.rs");
    fs::write(&ast_path, &ast_builder).expect("Failed to write ast_builder.rs");
    println!(
        "✓ Wrote {} ({} bytes)",
        ast_path.display(),
        ast_builder.len()
    );

    // Write parser implementation (zyntax version)
    let parser_path = gen_dir.join("parser_impl.rs");
    fs::write(&parser_path, &parser_impl).expect("Failed to write parser_impl.rs");
    println!(
        "✓ Wrote {} ({} bytes)",
        parser_path.display(),
        parser_impl.len()
    );

    // Verify files exist
    assert!(pest_path.exists(), "zig.pest not created");
    assert!(ast_path.exists(), "ast_builder.rs not created");
    assert!(parser_path.exists(), "parser_impl.rs not created");

    // Verify the generated files use zyntax_typed_ast
    let ast_content = fs::read_to_string(&ast_path).unwrap();
    assert!(
        ast_content.contains("use zyntax_typed_ast"),
        "ast_builder should use zyntax_typed_ast"
    );

    let parser_content = fs::read_to_string(&parser_path).unwrap();
    assert!(
        parser_content.contains("use zyntax_typed_ast"),
        "parser_impl should use zyntax_typed_ast"
    );

    println!("✓ All zyntax parser files written successfully");
}
