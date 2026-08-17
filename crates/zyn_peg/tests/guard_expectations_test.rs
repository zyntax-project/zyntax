//! An alternative the guard skips still says what it wanted.
//!
//! Deciding an alternative on the byte ahead of it is what makes the
//! machine fast, but a skipped alternative never reaches the code that
//! records its expectation. The reader was then told only about the
//! alternatives that ran: where a grammar offered a list of types, the
//! message named whichever one happened to be tried last, or the class
//! some unrelated rule wanted.
//!
//! The contract these tests hold is stronger than "says something
//! useful": the message is the same whether or not the machine is on.
//! An optimisation that changes what a grammar author is told is a
//! different grammar as far as they are concerned.

use zyn_peg::grammar::parse_grammar;
use zyn_peg::runtime2::{GrammarInterpreter, ParseResult, ParserState};
use zyntax_typed_ast::{TypeRegistry, TypedASTBuilder};

/// What a failed parse says it expected, at the furthest position.
fn expectations(grammar_src: &str, rule: &str, input: &str) -> Vec<String> {
    let grammar = parse_grammar(grammar_src).expect("grammar parses");
    let interp = GrammarInterpreter::new(&grammar);
    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new(input, &mut builder, &mut registry);
    match interp.parse_rule(rule, &mut state) {
        ParseResult::Success(_, pos) => {
            panic!("expected a parse failure, but it succeeded at {pos}")
        }
        ParseResult::Failure(_) => state.furthest_error().expected,
    }
}

/// A choice between named rules, which is the shape a DSL uses for a
/// list of types and the shape the guard skips through.
const TYPES: &str = r#"
    decl = { "let" ~ name:ident ~ ":" ~ ty:type_name }
    type_name = { int_type | float_type | bool_type }
    int_type = { "int" }
    float_type = { "float" }
    bool_type = { "bool" }
    ident = @{ ASCII_ALPHA+ }
    WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
"#;

#[test]
fn a_skipped_alternative_still_reports_what_it_expected() {
    let expected = expectations(TYPES, "decl", "let x : 42");

    // Every alternative, not merely the last one tried. Before this,
    // the two the guard ruled out on the byte `4` contributed nothing.
    for want in ["'int'", "'float'", "'bool'"] {
        assert!(
            expected.iter().any(|e| e == want),
            "every alternative should be named; {want} missing from {expected:?}"
        );
    }
}

/// The message does not depend on which engine ran, so turning the
/// machine off to compare is a fair comparison.
#[test]
fn the_machine_and_the_interpreter_say_the_same_thing() {
    // `ZYNPEG_MACHINE` is read once per process, so this cannot flip
    // the engine from in here. It reads whichever the process has, and
    // both are pinned to the same expected list, which is what makes
    // the pair meaningful when CI runs the suite each way.
    let expected = expectations(TYPES, "decl", "let x : 42");
    assert_eq!(
        expected,
        vec![
            "'int'".to_string(),
            "'float'".to_string(),
            "'bool'".to_string()
        ],
        "the guard must not change what a failure reports"
    );
}

/// A rule that refers to itself is followed to a bounded depth rather
/// than forever.
#[test]
fn a_recursive_rule_does_not_hang_the_naming() {
    const RECURSIVE: &str = r#"
        expr = { list | atom }
        list = { "(" ~ expr ~ ")" }
        atom = { "x" }
        WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
    "#;

    let expected = expectations(RECURSIVE, "expr", "!");
    assert!(
        !expected.is_empty(),
        "a recursive grammar should still report something"
    );
}
