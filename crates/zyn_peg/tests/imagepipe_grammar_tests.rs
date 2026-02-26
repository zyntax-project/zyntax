//! Comprehensive ImagePipe Grammar Tests for ZynPEG 2.0
//!
//! Tests the ZynPEG 2.0 parser against the ImagePipe DSL grammar.
//! ImagePipe v2.0 uses:
//! - Named bindings (name:identifier syntax)
//! - Direct TypedAST construction (TypedStatement::Let, etc.)
//! - No legacy JSON actions
//!
//! These tests validate:
//! 1. Grammar parsing into GrammarIR
//! 2. Pattern structure validation for each rule
//! 3. Action structure (Construct, not LegacyJson)
//! 4. Using GrammarInterpreter to parse actual ImagePipe source files

use zyn_peg::grammar::{parse_grammar, ActionIR, CharClass, PatternIR, RuleModifier};
use zyn_peg::runtime2::{GrammarInterpreter, ParseResult, ParsedValue, ParserState};
use zyntax_typed_ast::type_registry::TypeRegistry;
use zyntax_typed_ast::{TypedASTBuilder, TypedDeclaration};

const IMAGEPIPE_GRAMMAR: &str = include_str!("../../../examples/imagepipe/imagepipe.zyn");
const VINTAGE_IMGPIPE: &str = include_str!("../../../examples/imagepipe/samples/vintage.imgpipe");

// =============================================================================
// Grammar Parsing Tests
// =============================================================================

#[test]
fn test_imagepipe_grammar_parses_successfully() {
    let grammar =
        parse_grammar(IMAGEPIPE_GRAMMAR).expect("ImagePipe grammar should parse successfully");

    // Validate metadata
    assert_eq!(
        grammar.metadata.name, "ImagePipe",
        "Language name should be ImagePipe"
    );
    assert_eq!(grammar.metadata.version, "2.0", "Version should be 2.0");
    assert_eq!(
        grammar.metadata.file_extensions,
        vec![".imgpipe", ".ip"],
        "File extensions should include .imgpipe and .ip"
    );
    assert_eq!(
        grammar.metadata.entry_point,
        Some("run_pipeline".to_string()),
        "Entry point should be run_pipeline"
    );
}

#[test]
fn test_imagepipe_builtin_mappings() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();

    // Test image I/O builtins
    assert_eq!(
        grammar.builtins.functions.get("image_load"),
        Some(&"$Image$load".to_string()),
        "image_load should map to $Image$load"
    );
    assert_eq!(
        grammar.builtins.functions.get("image_save"),
        Some(&"$Image$save".to_string()),
        "image_save should map to $Image$save"
    );

    // Test transformation builtins
    assert_eq!(
        grammar.builtins.functions.get("image_resize"),
        Some(&"$Image$resize".to_string()),
    );
    assert_eq!(
        grammar.builtins.functions.get("image_rotate90"),
        Some(&"$Image$rotate90".to_string()),
    );

    // Test filter builtins
    assert_eq!(
        grammar.builtins.functions.get("image_blur"),
        Some(&"$Image$blur".to_string()),
    );
    assert_eq!(
        grammar.builtins.functions.get("println"),
        Some(&"$IO$println".to_string()),
    );

    // Should have 15 builtin functions
    assert_eq!(
        grammar.builtins.functions.len(),
        15,
        "Should have 15 builtin functions"
    );
}

#[test]
fn test_imagepipe_rule_count() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();

    // 26 rules total in v2.0:
    // program, statements, statement
    // load_stmt, save_stmt, print_stmt, resize_stmt, crop_stmt
    // rotate90_stmt, rotate180_stmt, rotate270_stmt
    // flip_h_stmt, flip_v_stmt
    // blur_stmt, brightness_stmt, contrast_stmt, grayscale_stmt, invert_stmt
    // string_literal, string_inner, integer, signed_integer, number, identifier
    // WHITESPACE, COMMENT
    assert_eq!(
        grammar.rules.len(),
        26,
        "ImagePipe grammar should have 26 rules"
    );
}

// =============================================================================
// Rule Structure Tests - Program and Statements
// =============================================================================

#[test]
fn test_program_rule_structure() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let program = grammar
        .rules
        .get("program")
        .expect("program rule should exist");

    // Pattern: SOI ~ stmts:statements ~ EOI
    match &program.pattern {
        PatternIR::Sequence(seq) => {
            assert_eq!(seq.len(), 3, "program should have 3 elements");
            println!("seq[0] = {:?}", &seq[0]);
            assert!(matches!(&seq[0], PatternIR::StartOfInput));
            assert!(matches!(&seq[1], PatternIR::RuleRef { rule_name, binding }
                if rule_name == "statements" && binding.as_ref().map(|s| s.as_str()) == Some("stmts")));
            assert!(matches!(&seq[2], PatternIR::EndOfInput));
        }
        _ => panic!("program pattern should be Sequence"),
    }

    // Action: -> stmts (PassThrough)
    match &program.action {
        Some(ActionIR::PassThrough { binding }) => {
            assert_eq!(
                binding, "stmts",
                "program should pass through stmts binding"
            );
        }
        _ => panic!("program action should be PassThrough"),
    }
}

#[test]
fn test_statements_rule_structure() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let statements = grammar
        .rules
        .get("statements")
        .expect("statements rule should exist");

    // Pattern: stmt:statement*
    match &statements.pattern {
        PatternIR::Repeat {
            pattern, min, max, ..
        } => {
            assert_eq!(*min, 0, "statements should allow zero statements");
            assert!(
                max.is_none(),
                "statements should allow unlimited statements"
            );
            match pattern.as_ref() {
                PatternIR::RuleRef { rule_name, binding } => {
                    assert_eq!(rule_name, "statement");
                    assert_eq!(binding.as_ref().map(|s| s.as_str()), Some("stmt"));
                }
                _ => panic!("statements pattern should contain RuleRef to statement"),
            }
        }
        _ => panic!("statements pattern should be Repeat"),
    }

    // Action should be Construct (TypedProgram)
    match &statements.action {
        Some(ActionIR::Construct { type_path, fields }) => {
            assert_eq!(type_path, "TypedProgram");
            assert!(!fields.is_empty(), "statements action should have fields");
        }
        _ => panic!("statements action should be Construct"),
    }
}

#[test]
fn test_statement_rule_is_choice() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let statement = grammar
        .rules
        .get("statement")
        .expect("statement rule should exist");

    match &statement.pattern {
        PatternIR::Choice(choices) => {
            assert_eq!(choices.len(), 15, "statement should have 15 alternatives");

            // Verify key alternatives exist
            let choice_names: Vec<_> = choices
                .iter()
                .filter_map(|c| match c {
                    PatternIR::RuleRef { rule_name, .. } => Some(rule_name.as_str()),
                    _ => None,
                })
                .collect();

            assert!(choice_names.contains(&"load_stmt"), "should have load_stmt");
            assert!(choice_names.contains(&"save_stmt"), "should have save_stmt");
            assert!(
                choice_names.contains(&"print_stmt"),
                "should have print_stmt"
            );
            assert!(
                choice_names.contains(&"resize_stmt"),
                "should have resize_stmt"
            );
            assert!(choice_names.contains(&"blur_stmt"), "should have blur_stmt");
            assert!(
                choice_names.contains(&"rotate90_stmt"),
                "should have rotate90_stmt"
            );
        }
        _ => panic!("statement pattern should be Choice"),
    }

    // statement has no action (it's a passthrough choice)
    assert!(
        statement.action.is_none(),
        "statement should have no explicit action"
    );
}

// =============================================================================
// Rule Structure Tests - Statement Rules with Construct Actions
// =============================================================================

#[test]
fn test_load_stmt_rule_structure() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let load_stmt = grammar
        .rules
        .get("load_stmt")
        .expect("load_stmt rule should exist");

    // Pattern: "load" ~ path:string_literal ~ "as" ~ name:identifier
    match &load_stmt.pattern {
        PatternIR::Sequence(seq) => {
            assert_eq!(seq.len(), 4, "load_stmt should have 4 elements");
            assert!(matches!(&seq[0], PatternIR::Literal(s) if s == "load"));
            assert!(matches!(&seq[1], PatternIR::RuleRef { rule_name, binding }
                if rule_name == "string_literal" && binding.as_ref().map(|s| s.as_str()) == Some("path")));
            assert!(matches!(&seq[2], PatternIR::Literal(s) if s == "as"));
            assert!(matches!(&seq[3], PatternIR::RuleRef { rule_name, binding }
                if rule_name == "identifier" && binding.as_ref().map(|s| s.as_str()) == Some("name")));
        }
        _ => panic!("load_stmt pattern should be Sequence"),
    }

    // Action: TypedStatement::Let { ... }
    match &load_stmt.action {
        Some(ActionIR::Construct { type_path, fields }) => {
            assert_eq!(type_path, "TypedStatement::Let");
            let field_names: Vec<_> = fields.iter().map(|(n, _)| n.as_str()).collect();
            assert!(field_names.contains(&"name"), "should have name field");
            assert!(
                field_names.contains(&"initializer"),
                "should have initializer field"
            );
            assert!(
                field_names.contains(&"is_mutable"),
                "should have is_mutable field"
            );
        }
        _ => panic!("load_stmt action should be Construct"),
    }
}

#[test]
fn test_save_stmt_rule_structure() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let save_stmt = grammar
        .rules
        .get("save_stmt")
        .expect("save_stmt rule should exist");

    // Pattern: "save" ~ img:identifier ~ "as" ~ path:string_literal
    match &save_stmt.pattern {
        PatternIR::Sequence(seq) => {
            assert_eq!(seq.len(), 4, "save_stmt should have 4 elements");
            assert!(matches!(&seq[0], PatternIR::Literal(s) if s == "save"));
            assert!(matches!(&seq[1], PatternIR::RuleRef { binding, .. }
                if binding.as_ref().map(|s| s.as_str()) == Some("img")));
        }
        _ => panic!("save_stmt pattern should be Sequence"),
    }

    // Action: TypedStatement::Expression { expr: ... }
    match &save_stmt.action {
        Some(ActionIR::Construct { type_path, .. }) => {
            assert_eq!(type_path, "TypedStatement::Expression");
        }
        _ => panic!("save_stmt action should be Construct"),
    }
}

#[test]
fn test_print_stmt_rule_structure() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let print_stmt = grammar
        .rules
        .get("print_stmt")
        .expect("print_stmt rule should exist");

    // Pattern: "print" ~ msg:string_literal
    match &print_stmt.pattern {
        PatternIR::Sequence(seq) => {
            assert_eq!(seq.len(), 2, "print_stmt should have 2 elements");
            assert!(matches!(&seq[0], PatternIR::Literal(s) if s == "print"));
        }
        _ => panic!("print_stmt pattern should be Sequence"),
    }

    // Action: TypedStatement::Expression
    match &print_stmt.action {
        Some(ActionIR::Construct { type_path, .. }) => {
            assert_eq!(type_path, "TypedStatement::Expression");
        }
        _ => panic!("print_stmt action should be Construct"),
    }
}

#[test]
fn test_resize_stmt_rule_structure() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let resize_stmt = grammar
        .rules
        .get("resize_stmt")
        .expect("resize_stmt rule should exist");

    // Pattern: "resize" ~ img:identifier ~ "to" ~ w:integer ~ "x" ~ h:integer
    match &resize_stmt.pattern {
        PatternIR::Sequence(seq) => {
            assert_eq!(seq.len(), 6, "resize_stmt should have 6 elements");
            assert!(matches!(&seq[0], PatternIR::Literal(s) if s == "resize"));
            assert!(matches!(&seq[2], PatternIR::Literal(s) if s == "to"));
            assert!(matches!(&seq[4], PatternIR::Literal(s) if s == "x"));
        }
        _ => panic!("resize_stmt pattern should be Sequence"),
    }

    // Action: TypedStatement::Assignment
    match &resize_stmt.action {
        Some(ActionIR::Construct { type_path, .. }) => {
            assert_eq!(type_path, "TypedStatement::Assignment");
        }
        _ => panic!("resize_stmt action should be Construct"),
    }
}

#[test]
fn test_blur_stmt_uses_number() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let blur_stmt = grammar
        .rules
        .get("blur_stmt")
        .expect("blur_stmt rule should exist");

    // Pattern should include number for sigma
    match &blur_stmt.pattern {
        PatternIR::Sequence(seq) => {
            // "blur" ~ img:identifier ~ "by" ~ sigma:number
            let has_number = seq.iter().any(
                |p| matches!(p, PatternIR::RuleRef { rule_name, .. } if rule_name == "number"),
            );
            assert!(has_number, "blur_stmt should use number rule for sigma");
        }
        _ => panic!("blur_stmt pattern should be Sequence"),
    }
}

#[test]
fn test_brightness_stmt_uses_signed_integer() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let brightness_stmt = grammar
        .rules
        .get("brightness_stmt")
        .expect("brightness_stmt rule should exist");

    match &brightness_stmt.pattern {
        PatternIR::Sequence(seq) => {
            let has_signed_int = seq.iter().any(|p|
                matches!(p, PatternIR::RuleRef { rule_name, .. } if rule_name == "signed_integer"));
            assert!(
                has_signed_int,
                "brightness_stmt should use signed_integer for amount"
            );
        }
        _ => panic!("brightness_stmt pattern should be Sequence"),
    }
}

#[test]
fn test_grayscale_stmt_unary() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let grayscale_stmt = grammar
        .rules
        .get("grayscale_stmt")
        .expect("grayscale_stmt rule should exist");

    // Pattern: "grayscale" ~ img:identifier (just 2 elements)
    match &grayscale_stmt.pattern {
        PatternIR::Sequence(seq) => {
            assert_eq!(seq.len(), 2, "grayscale_stmt should have only 2 elements");
        }
        _ => panic!("grayscale_stmt pattern should be Sequence"),
    }
}

// =============================================================================
// Terminal Rule Tests
// =============================================================================

#[test]
fn test_string_literal_is_atomic() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let string_literal = grammar
        .rules
        .get("string_literal")
        .expect("string_literal rule should exist");

    assert_eq!(
        string_literal.modifier,
        Some(RuleModifier::Atomic),
        "string_literal should be atomic"
    );

    // Action: TypedExpression::StringLiteral
    match &string_literal.action {
        Some(ActionIR::Construct { type_path, .. }) => {
            assert_eq!(type_path, "TypedExpression::StringLiteral");
        }
        _ => panic!("string_literal action should be Construct"),
    }
}

#[test]
fn test_integer_is_atomic() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let integer = grammar
        .rules
        .get("integer")
        .expect("integer rule should exist");

    assert_eq!(
        integer.modifier,
        Some(RuleModifier::Atomic),
        "integer should be atomic"
    );

    // Action: TypedExpression::IntLiteral
    match &integer.action {
        Some(ActionIR::Construct { type_path, .. }) => {
            assert_eq!(type_path, "TypedExpression::IntLiteral");
        }
        _ => panic!("integer action should be Construct"),
    }
}

#[test]
fn test_number_is_atomic() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let number = grammar
        .rules
        .get("number")
        .expect("number rule should exist");

    assert_eq!(
        number.modifier,
        Some(RuleModifier::Atomic),
        "number should be atomic"
    );

    // Action: TypedExpression::FloatLiteral
    match &number.action {
        Some(ActionIR::Construct { type_path, .. }) => {
            assert_eq!(type_path, "TypedExpression::FloatLiteral");
        }
        _ => panic!("number action should be Construct"),
    }
}

#[test]
fn test_identifier_is_atomic() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let identifier = grammar
        .rules
        .get("identifier")
        .expect("identifier rule should exist");

    assert_eq!(
        identifier.modifier,
        Some(RuleModifier::Atomic),
        "identifier should be atomic"
    );

    // identifier has no action (returns text)
    assert!(
        identifier.action.is_none(),
        "identifier should have no action"
    );
}

#[test]
fn test_whitespace_is_silent() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let whitespace = grammar
        .rules
        .get("WHITESPACE")
        .expect("WHITESPACE rule should exist");

    assert_eq!(
        whitespace.modifier,
        Some(RuleModifier::Silent),
        "WHITESPACE should be silent"
    );
}

#[test]
fn test_comment_is_silent() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let comment = grammar
        .rules
        .get("COMMENT")
        .expect("COMMENT rule should exist");

    assert_eq!(
        comment.modifier,
        Some(RuleModifier::Silent),
        "COMMENT should be silent"
    );
}

// =============================================================================
// GrammarInterpreter Tests - Parse Individual Statements
// =============================================================================

#[test]
fn test_interpreter_parse_identifier() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("my_image_var", &mut builder, &mut registry);

    let result = interp.parse_rule("identifier", &mut state);
    assert!(result.is_success(), "Should parse identifier");
    match result {
        ParseResult::Success(ParsedValue::Text(s), _) => {
            assert_eq!(s, "my_image_var");
        }
        _ => panic!("identifier should return Text"),
    }
}

#[test]
fn test_interpreter_parse_integer() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("12345", &mut builder, &mut registry);

    let result = interp.parse_rule("integer", &mut state);
    assert!(result.is_success(), "Should parse integer");
}

#[test]
fn test_interpreter_parse_string_literal() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("\"hello world\"", &mut builder, &mut registry);

    let result = interp.parse_rule("string_literal", &mut state);
    assert!(result.is_success(), "Should parse string literal");
}

#[test]
fn test_interpreter_parse_number() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("3.14159", &mut builder, &mut registry);

    let result = interp.parse_rule("number", &mut state);
    assert!(result.is_success(), "Should parse decimal number");
}

#[test]
fn test_interpreter_parse_signed_integer() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    // Test negative
    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("-50", &mut builder, &mut registry);
    assert!(
        interp.parse_rule("signed_integer", &mut state).is_success(),
        "Should parse negative"
    );

    // Test positive with sign
    let mut state2 = ParserState::new("+30", &mut builder, &mut registry);
    assert!(
        interp
            .parse_rule("signed_integer", &mut state2)
            .is_success(),
        "Should parse positive with sign"
    );
}

#[test]
fn test_interpreter_parse_load_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("load \"image.jpg\" as photo", &mut builder, &mut registry);

    let result = interp.parse_rule("load_stmt", &mut state);
    assert!(result.is_success(), "Should parse load statement");
}

#[test]
fn test_interpreter_parse_save_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("save photo as \"output.png\"", &mut builder, &mut registry);

    let result = interp.parse_rule("save_stmt", &mut state);
    assert!(result.is_success(), "Should parse save statement");
}

#[test]
fn test_interpreter_parse_print_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("print \"Hello!\"", &mut builder, &mut registry);

    let result = interp.parse_rule("print_stmt", &mut state);
    assert!(result.is_success(), "Should parse print statement");
}

#[test]
fn test_interpreter_parse_resize_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("resize photo to 800x600", &mut builder, &mut registry);

    let result = interp.parse_rule("resize_stmt", &mut state);
    assert!(result.is_success(), "Should parse resize statement");
}

#[test]
fn test_interpreter_parse_blur_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("blur photo by 2.5", &mut builder, &mut registry);

    let result = interp.parse_rule("blur_stmt", &mut state);
    assert!(result.is_success(), "Should parse blur statement");
}

#[test]
fn test_interpreter_parse_brightness_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("brighten photo by -30", &mut builder, &mut registry);

    let result = interp.parse_rule("brightness_stmt", &mut state);
    assert!(result.is_success(), "Should parse brightness statement");
}

#[test]
fn test_interpreter_parse_contrast_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("contrast photo by 1.5", &mut builder, &mut registry);

    let result = interp.parse_rule("contrast_stmt", &mut state);
    assert!(result.is_success(), "Should parse contrast statement");
}

#[test]
fn test_interpreter_parse_grayscale_stmt() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("grayscale photo", &mut builder, &mut registry);

    let result = interp.parse_rule("grayscale_stmt", &mut state);
    assert!(result.is_success(), "Should parse grayscale statement");
}

// =============================================================================
// GrammarInterpreter Tests - Parse vintage.imgpipe
// =============================================================================

#[test]
fn test_interpreter_parse_vintage_imgpipe_statements() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new(VINTAGE_IMGPIPE, &mut builder, &mut registry);

    // Parse using the 'statements' rule
    // Note: The statements rule now constructs a TypedProgram with a run_pipeline function
    let result = interp.parse_rule("statements", &mut state);

    match result {
        ParseResult::Success(ParsedValue::Program(program), _) => {
            // The statements are wrapped in a run_pipeline function
            // vintage.imgpipe has 7 statements: load, resize, contrast, brighten, blur, save, print
            assert_eq!(
                program.declarations.len(),
                1,
                "Should have 1 declaration (run_pipeline function)"
            );
            if let TypedDeclaration::Function(func) = &program.declarations[0].node {
                if let Some(ref body) = func.body {
                    // Print each statement for debugging
                    println!("Found {} statements:", body.statements.len());
                    for (i, stmt) in body.statements.iter().enumerate() {
                        println!("  [{i}] {:?}", stmt.node);
                    }
                    assert_eq!(
                        body.statements.len(),
                        7,
                        "vintage.imgpipe should have 7 statements, got {}",
                        body.statements.len()
                    );
                } else {
                    panic!("Expected function body");
                }
            } else {
                panic!("Expected function declaration");
            }
        }
        ParseResult::Success(other, _) => {
            panic!("Expected Program, got {:?}", other);
        }
        ParseResult::Failure(e) => {
            panic!(
                "Failed to parse vintage.imgpipe: line {} col {}: {:?}",
                e.line, e.column, e.expected
            );
        }
    }
}

#[test]
fn test_interpreter_parse_vintage_imgpipe_program() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new(VINTAGE_IMGPIPE, &mut builder, &mut registry);

    // Parse using the 'program' rule (SOI ~ statements ~ EOI)
    let result = interp.parse_rule("program", &mut state);

    match &result {
        ParseResult::Success(_, _) => {}
        ParseResult::Failure(e) => {
            let remaining = if state.pos() < VINTAGE_IMGPIPE.len() {
                &VINTAGE_IMGPIPE
                    [state.pos()..std::cmp::min(state.pos() + 50, VINTAGE_IMGPIPE.len())]
            } else {
                ""
            };
            panic!(
                "Failed to parse vintage.imgpipe as program at line {} col {} (pos {}): expected {:?}\nRemaining input: {:?}",
                e.line, e.column, state.pos(), e.expected, remaining
            );
        }
    }

    assert!(
        result.is_success(),
        "Should successfully parse entire vintage.imgpipe as program"
    );

    // Should have consumed entire input
    assert!(
        state.is_eof(),
        "Should be at end of file after parsing program"
    );
}

#[test]
fn test_interpreter_parse_individual_vintage_statements() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    // Test each statement from vintage.imgpipe individually
    let statements = vec![
        ("load_stmt", "load \"PXL_20251101_145444778.jpg\" as photo"),
        ("resize_stmt", "resize photo to 800x1200"),
        ("contrast_stmt", "contrast photo by -50"),
        ("brightness_stmt", "brighten photo by -30"),
        ("blur_stmt", "blur photo by 3"),
        ("save_stmt", "save photo as \"vintage_rotunda.png\""),
        ("print_stmt", "print \"Vintage effect applied!\""),
    ];

    for (rule_name, input) in statements {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new(input, &mut builder, &mut registry);

        let result = interp.parse_rule(rule_name, &mut state);
        assert!(
            result.is_success(),
            "Failed to parse '{}' with rule '{}': {:?}",
            input,
            rule_name,
            result
        );
    }
}

// =============================================================================
// TypedAST Generation Tests - Print the AST structure
// =============================================================================

/// This test demonstrates the TypedAST action structure for each statement type.
/// The grammar defines direct TypedAST construction via ActionIR::Construct.
#[test]
fn test_print_typed_ast_actions() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();

    println!("\n=== TypedAST Action Definitions ===\n");

    // Print the action for each statement rule
    let stmt_rules = vec![
        "load_stmt",
        "save_stmt",
        "print_stmt",
        "resize_stmt",
        "crop_stmt",
        "rotate90_stmt",
        "rotate180_stmt",
        "rotate270_stmt",
        "flip_h_stmt",
        "flip_v_stmt",
        "blur_stmt",
        "brightness_stmt",
        "contrast_stmt",
        "grayscale_stmt",
        "invert_stmt",
    ];

    for rule_name in stmt_rules {
        let rule = grammar.rules.get(rule_name).expect(rule_name);
        println!("--- {} ---", rule_name);

        if let Some(action) = &rule.action {
            match action {
                ActionIR::Construct { type_path, fields } => {
                    println!("  Type: {}", type_path);
                    println!("  Fields:");
                    for (field_name, expr) in fields {
                        println!("    {}: {:?}", field_name, expr);
                    }
                }
                ActionIR::PassThrough { binding } => {
                    println!("  PassThrough: {}", binding);
                }
                _ => println!("  Other action: {:?}", action),
            }
        } else {
            println!("  (no action)");
        }
        println!();
    }
}

/// This test parses vintage.imgpipe and prints the bindings captured for each statement.
/// This shows what values would be used to construct TypedAST nodes.
#[test]
fn test_print_parsed_bindings() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    println!("\n=== Parsed Statement Bindings from vintage.imgpipe ===\n");

    let statements = vec![
        ("load_stmt", "load \"PXL_20251101_145444778.jpg\" as photo"),
        ("resize_stmt", "resize photo to 800x1200"),
        ("contrast_stmt", "contrast photo by -50"),
        ("brightness_stmt", "brighten photo by -30"),
        ("blur_stmt", "blur photo by 3"),
        ("save_stmt", "save photo as \"vintage_rotunda.png\""),
        ("print_stmt", "print \"Vintage effect applied!\""),
    ];

    for (rule_name, input) in &statements {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new(input, &mut builder, &mut registry);

        println!("Input: {}", input);
        println!("Rule: {}", rule_name);

        let result = interp.parse_rule(rule_name, &mut state);
        match result {
            ParseResult::Success(value, _) => {
                println!("Parsed value: {:?}", value);
            }
            ParseResult::Failure(e) => {
                println!("Failed: {:?}", e);
            }
        }
        println!();
    }
}

/// Demonstrates that the grammar generates TypedStatement and TypedExpression constructs.
#[test]
fn test_action_types_are_typed_ast() {
    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();

    // Check that statement rules produce TypedStatement constructs
    for rule_name in &["load_stmt", "save_stmt", "resize_stmt", "blur_stmt"] {
        let rule = grammar.rules.get(*rule_name).expect(rule_name);
        match &rule.action {
            Some(ActionIR::Construct { type_path, .. }) => {
                assert!(
                    type_path.starts_with("TypedStatement::"),
                    "{} should produce TypedStatement, got {}",
                    rule_name,
                    type_path
                );
            }
            _ => panic!("{} should have Construct action", rule_name),
        }
    }

    // Check that terminal rules produce TypedExpression constructs
    for rule_name in &["string_literal", "integer", "number"] {
        let rule = grammar.rules.get(*rule_name).expect(rule_name);
        match &rule.action {
            Some(ActionIR::Construct { type_path, .. }) => {
                assert!(
                    type_path.starts_with("TypedExpression::"),
                    "{} should produce TypedExpression, got {}",
                    rule_name,
                    type_path
                );
            }
            _ => panic!("{} should have Construct action", rule_name),
        }
    }

    println!("\nAll action types verified as TypedAST constructs!");
}

// =============================================================================
// TypedAST Construction Tests - Full Pipeline Demonstration
// =============================================================================

/// This test demonstrates the complete TypedAST construction by the interpreter.
/// It parses vintage.imgpipe and prints the resulting TypedProgram structure.
#[test]
fn test_print_constructed_typed_ast() {
    use zyntax_typed_ast::{TypedExpression, TypedStatement};

    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new(VINTAGE_IMGPIPE, &mut builder, &mut registry);

    println!("\n=== Constructed TypedAST from vintage.imgpipe ===\n");

    let result = interp.parse_rule("program", &mut state);

    match result {
        ParseResult::Success(ParsedValue::Program(program), _) => {
            println!(
                "TypedProgram created with {} declaration(s)\n",
                program.declarations.len()
            );

            for (i, decl) in program.declarations.iter().enumerate() {
                println!("Declaration [{}]:", i);
                if let TypedDeclaration::Function(func) = &decl.node {
                    let func_name = builder.arena().resolve_string(func.name);
                    println!("  Function: {}", func_name.unwrap_or("<unknown>"));
                    println!("  Return type: {:?}", func.return_type);

                    if let Some(ref body) = func.body {
                        println!("  Body ({} statements):", body.statements.len());

                        for (j, stmt_node) in body.statements.iter().enumerate() {
                            print!("    [{j}] ");
                            match &stmt_node.node {
                                TypedStatement::Let(let_stmt) => {
                                    let var_name = builder.arena().resolve_string(let_stmt.name);
                                    println!(
                                        "Let {} : {:?}",
                                        var_name.unwrap_or("<unknown>"),
                                        let_stmt.ty
                                    );
                                }
                                TypedStatement::Expression(expr) => match &expr.node {
                                    TypedExpression::Call(call) => {
                                        if let TypedExpression::Variable(callee_name) =
                                            &call.callee.node
                                        {
                                            let name = builder.arena().resolve_string(*callee_name);
                                            println!(
                                                "Call {}(...) [{} args]",
                                                name.unwrap_or("<unknown>"),
                                                call.positional_args.len()
                                            );
                                        } else {
                                            println!(
                                                "Call <expr>(...) [{} args]",
                                                call.positional_args.len()
                                            );
                                        }
                                    }
                                    TypedExpression::Binary(bin) => {
                                        println!("Binary {:?} = ...", bin.op);
                                    }
                                    TypedExpression::Literal(lit) => {
                                        println!("Literal {:?}", lit);
                                    }
                                    other => {
                                        println!("{:?}", other);
                                    }
                                },
                                other => {
                                    println!("{:?}", other);
                                }
                            }
                        }
                    }
                }
            }
        }
        ParseResult::Success(other, _) => {
            panic!("Expected Program, got {:?}", other);
        }
        ParseResult::Failure(e) => {
            panic!("Failed to parse: {:?}", e);
        }
    }

    println!("\n=== TypedAST construction complete ===");
}

/// Test that individual statements produce the correct TypedStatement variant.
#[test]
fn test_statement_produces_typed_ast_nodes() {
    use zyntax_typed_ast::{TypedExpression, TypedStatement};

    let grammar = parse_grammar(IMAGEPIPE_GRAMMAR).unwrap();
    let interp = GrammarInterpreter::new(&grammar);

    // Test load statement produces Let
    {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new(r#"load "test.jpg" as img"#, &mut builder, &mut registry);

        let result = interp.parse_rule("load_stmt", &mut state);
        match result {
            ParseResult::Success(ParsedValue::Statement(stmt), _) => match &stmt.node {
                TypedStatement::Let(let_stmt) => {
                    let name = builder.arena().resolve_string(let_stmt.name);
                    assert_eq!(name, Some("img"));
                    assert!(let_stmt.initializer.is_some());
                }
                _ => panic!("Expected Let statement"),
            },
            _ => panic!("Expected Statement"),
        }
    }

    // Test print statement produces Expression(Call)
    {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new(r#"print "hello""#, &mut builder, &mut registry);

        let result = interp.parse_rule("print_stmt", &mut state);
        match result {
            ParseResult::Success(ParsedValue::Statement(stmt), _) => {
                match &stmt.node {
                    TypedStatement::Expression(expr) => {
                        match &expr.node {
                            TypedExpression::Call(call) => {
                                // The callee should be a Variable referencing "println"
                                if let TypedExpression::Variable(callee_name) = &call.callee.node {
                                    let name = builder.arena().resolve_string(*callee_name);
                                    assert_eq!(name, Some("println"));
                                }
                                // Should have 1 argument (the string literal)
                                assert_eq!(call.positional_args.len(), 1);
                            }
                            _ => panic!("Expected Call expression"),
                        }
                    }
                    _ => panic!("Expected Expression statement"),
                }
            }
            _ => panic!("Expected Statement"),
        }
    }

    println!("Statement TypedAST node tests passed!");
}
