//! Tests for parsing the zig.zyn grammar file

use pest::Parser;
use std::fs;
use zyn_peg::ast::build_grammar;
use zyn_peg::generator::generate_parser;
use zyn_peg::{Rule, ZynGrammarParser};

const ZIG_GRAMMAR_PATH: &str = "grammars/zig.zyn";

#[test]
fn test_parse_zig_zyn_grammar() {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    // Parse the grammar
    let result = ZynGrammarParser::parse(Rule::program, &grammar_content);

    match result {
        Ok(pairs) => {
            println!("✓ Successfully parsed zig.zyn grammar");

            // Build the grammar AST
            let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

            // Verify language info
            assert_eq!(grammar.language.name, "Zig");
            assert_eq!(grammar.language.version, "0.11");
            assert!(grammar
                .language
                .file_extensions
                .contains(&".zig".to_string()));
            println!(
                "✓ Language info: {} v{}",
                grammar.language.name, grammar.language.version
            );

            // Verify we have imports
            assert!(!grammar.imports.code.is_empty());
            println!("✓ Imports present");

            // Verify we have context vars
            assert!(!grammar.context.is_empty());
            println!("✓ Context: {} variables", grammar.context.len());
            for ctx in &grammar.context {
                println!("    - {}: {}", ctx.name, ctx.ty);
            }

            // Verify rules
            assert!(!grammar.rules.is_empty());
            println!("✓ Rules: {} defined", grammar.rules.len());

            // Count rules with actions
            let rules_with_actions = grammar.rules.iter().filter(|r| r.action.is_some()).count();
            println!("✓ Rules with action blocks: {}", rules_with_actions);

            // Print some rule names
            let rule_names: Vec<_> = grammar
                .rules
                .iter()
                .take(10)
                .map(|r| r.name.as_str())
                .collect();
            println!("✓ First 10 rules: {:?}", rule_names);

            // Verify type_helpers
            if !grammar.type_helpers.code.is_empty() {
                println!(
                    "✓ Type helpers present ({} chars)",
                    grammar.type_helpers.code.len()
                );
            }
        }
        Err(e) => {
            panic!("Failed to parse zig.zyn grammar:\n{}", e);
        }
    }
}

#[test]
fn test_zig_zyn_has_required_rules() {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    // Check for required rules
    let required_rules = vec![
        "program",
        "declaration",
        "fn_decl",
        "const_decl",
        "var_decl",
        "struct_decl",
        "type_expr",
        "primitive_type",
        "statement",
        "expr",
        "literal",
        "integer_literal",
        "identifier",
    ];

    let rule_names: std::collections::HashSet<_> =
        grammar.rules.iter().map(|r| r.name.as_str()).collect();

    for required in &required_rules {
        assert!(
            rule_names.contains(required),
            "Missing required rule: {}",
            required
        );
    }

    println!("✓ All {} required rules present", required_rules.len());
}

#[test]
fn test_zig_zyn_action_blocks() {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    // Rules that should have action blocks for TypedAST generation
    let expected_action_rules = vec![
        "program",
        "const_decl",
        "var_decl",
        "fn_decl",
        "integer_literal",
        "bool_literal",
        "return_stmt",
    ];

    for rule_name in &expected_action_rules {
        let rule = grammar.rules.iter().find(|r| r.name == *rule_name);
        assert!(rule.is_some(), "Missing rule: {}", rule_name);

        let rule = rule.unwrap();
        assert!(
            rule.action.is_some(),
            "Rule '{}' should have an action block",
            rule_name
        );

        let action = rule.action.as_ref().unwrap();
        println!(
            "✓ {} -> {} with {} fields",
            rule_name,
            action.return_type,
            action.fields.len()
        );
    }
}

#[test]
fn test_generate_pest_grammar_from_zig_zyn() {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    // Generate the parser code
    let generated = generate_parser(&grammar).expect("Failed to generate parser");

    // Verify pest grammar was generated
    assert!(!generated.pest_grammar.is_empty());
    println!(
        "✓ Generated pest grammar ({} chars)",
        generated.pest_grammar.len()
    );

    // Check that the pest grammar contains expected rules
    assert!(generated.pest_grammar.contains("program ="));
    assert!(generated.pest_grammar.contains("declaration ="));
    assert!(generated.pest_grammar.contains("fn_decl ="));
    assert!(generated.pest_grammar.contains("const_decl ="));
    assert!(generated.pest_grammar.contains("expr ="));
    println!("✓ Generated grammar contains core rules");

    // Check modifiers are applied correctly
    assert!(generated.pest_grammar.contains("identifier = @{"));
    assert!(generated.pest_grammar.contains("WHITESPACE = _{"));
    println!("✓ Rule modifiers correctly applied");

    // Print first 500 chars of generated grammar
    println!("--- Generated Pest Grammar (first 500 chars) ---");
    println!(
        "{}",
        &generated.pest_grammar[..500.min(generated.pest_grammar.len())]
    );
    println!("---");

    // Verify AST builder was generated
    let ast_builder_str = generated.ast_builder.to_string();
    assert!(!ast_builder_str.is_empty());
    println!("✓ Generated AST builder ({} chars)", ast_builder_str.len());

    // Verify parser impl was generated
    let parser_impl_str = generated.parser_impl.to_string();
    assert!(!parser_impl_str.is_empty());
    assert!(parser_impl_str.contains("ZigParser"));
    println!("✓ Generated parser implementation");

    // Show sample of AST builder code
    println!("\n--- Generated AST Builder (sample) ---");
    let ast_sample = &ast_builder_str[..2000.min(ast_builder_str.len())];
    println!("{}", ast_sample);
    println!("---");
}

#[test]
fn test_generated_build_methods() {
    let grammar_content =
        fs::read_to_string(ZIG_GRAMMAR_PATH).expect("Failed to read zig.zyn grammar file");

    let pairs =
        ZynGrammarParser::parse(Rule::program, &grammar_content).expect("Failed to parse grammar");
    let grammar = build_grammar(pairs).expect("Failed to build grammar AST");

    let generated = generate_parser(&grammar).expect("Failed to generate parser");
    let ast_builder_str = generated.ast_builder.to_string();

    // Check that build methods are generated for rules with actions
    assert!(
        ast_builder_str.contains("fn build_program"),
        "Missing build_program method"
    );
    assert!(
        ast_builder_str.contains("fn build_const_decl"),
        "Missing build_const_decl method"
    );
    assert!(
        ast_builder_str.contains("fn build_integer_literal"),
        "Missing build_integer_literal method"
    );
    println!("✓ Build methods generated for action rules");

    // Check that capture references are transformed to child_<binding>_<rulename> variables
    assert!(
        ast_builder_str.contains("child_name_identifier")
            || ast_builder_str.contains("child_decl_declaration"),
        "Capture refs should use child_<binding>_<rulename> pattern"
    );
    println!("✓ Capture references transformed correctly");

    // Check raw code actions are embedded
    // Rules like primitive_type use match expressions
    assert!(
        ast_builder_str.contains("fn build_primitive_type"),
        "Missing build_primitive_type method"
    );
    println!("✓ Raw code actions embedded");
}
