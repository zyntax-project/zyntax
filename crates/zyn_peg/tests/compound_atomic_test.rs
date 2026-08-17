//! `$` on a rule stops whitespace being skipped between its elements.
//!
//! The modifier parsed and then did nothing: both the interpreter and
//! the machine asked only whether a rule was `@`, so a rule marked `$`
//! skipped whitespace like any other. Nothing said so, and the grammar
//! quietly meant something else than it read.
//!
//! What it costs is visible in any interpolated string. `f"a{x} b"`
//! holds the parts of a string next to each other, and the space after
//! the closing brace belongs to the text that follows it. With the
//! skipping left on, that space was consumed between one part and the
//! next and never reached the string.
//!
//! `@` cannot serve here: it also replaces the rule's value with the
//! text it matched, which throws away the parts the rule exists to
//! collect.

use zyn_peg::grammar::parse_grammar;
use zyn_peg::runtime2::{GrammarInterpreter, ParseResult, ParserState};
use zyntax_typed_ast::{TypeRegistry, TypedASTBuilder};

/// How much of the input a rule consumed, or `None` if it did not match.
fn consumed(grammar_src: &str, rule: &str, input: &str) -> Option<usize> {
    let grammar = parse_grammar(grammar_src).expect("grammar parses");
    let interp = GrammarInterpreter::new(&grammar);
    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new(input, &mut builder, &mut registry);
    match interp.parse_rule(rule, &mut state) {
        ParseResult::Success(_, pos) => Some(pos),
        ParseResult::Failure(_) => None,
    }
}

/// A plain rule skips the space between its two pieces, so it matches
/// the whole of `a b`. The same rule marked `$` does not, so it stops
/// after `a`.
#[test]
fn compound_atomic_stops_whitespace_being_skipped() {
    const PLAIN: &str = r#"
        pair = { "a" ~ "b" }
        WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
    "#;
    const COMPOUND: &str = r#"
        pair = ${ "a" ~ "b" }
        WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
    "#;

    assert_eq!(
        consumed(PLAIN, "pair", "a b"),
        Some(3),
        "a plain rule skips the space and matches both pieces"
    );
    assert_eq!(
        consumed(COMPOUND, "pair", "a b"),
        None,
        "a `$` rule must not skip the space, so `b` is not there to match"
    );
    assert_eq!(
        consumed(COMPOUND, "pair", "ab"),
        Some(2),
        "with nothing between them a `$` rule matches both pieces"
    );
}

/// The text between the pieces reaches the rule that wants it rather
/// than being skipped past. This is the shape an interpolated string
/// has: a part, then whatever text follows it.
#[test]
fn compound_atomic_leaves_the_space_for_the_next_rule() {
    const GRAMMAR: &str = r#"
        run = ${ "a" ~ rest }
        rest = @{ (!"\"" ~ ANY)+ }
        WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
    "#;

    // `rest` takes everything after `a`, and under `$` that includes
    // the space. Skipping it would have handed `rest` only `tail`.
    assert_eq!(
        consumed(GRAMMAR, "run", "a tail"),
        Some(6),
        "the space belongs to `rest`, not to the gap between elements"
    );
}
