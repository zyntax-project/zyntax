//! Parser Combinators for ZynPEG 2.0
//!
//! This module provides combinator functions for building parsers:
//! - Sequence, choice, repetition
//! - Lookahead (positive and negative)
//! - Optional parsing
//! - Built-in character classes

use super::state::{ParseFailure, ParseResult, ParsedValue, ParserState};

/// Parse a literal string
pub fn literal<'a>(state: &mut ParserState<'a>, s: &str) -> ParseResult<()> {
    state.match_literal(s)
}

/// Parse any single character
pub fn any_char<'a>(state: &mut ParserState<'a>) -> ParseResult<char> {
    match state.peek_char() {
        Some(c) => {
            state.advance();
            ParseResult::Success(c, state.pos())
        }
        None => state.fail("any character"),
    }
}

/// Parse an ASCII digit
pub fn ascii_digit<'a>(state: &mut ParserState<'a>) -> ParseResult<char> {
    state.match_char(|c| c.is_ascii_digit(), "digit")
}

/// Parse an ASCII letter
pub fn ascii_alpha<'a>(state: &mut ParserState<'a>) -> ParseResult<char> {
    state.match_char(|c| c.is_ascii_alphabetic(), "letter")
}

/// Parse an ASCII alphanumeric character
pub fn ascii_alphanumeric<'a>(state: &mut ParserState<'a>) -> ParseResult<char> {
    state.match_char(|c| c.is_ascii_alphanumeric(), "alphanumeric")
}

/// Parse an ASCII hex digit
pub fn ascii_hex_digit<'a>(state: &mut ParserState<'a>) -> ParseResult<char> {
    state.match_char(|c| c.is_ascii_hexdigit(), "hex digit")
}

/// Parse a newline
pub fn newline<'a>(state: &mut ParserState<'a>) -> ParseResult<()> {
    if state.check("\r\n") {
        state.advance();
        state.advance();
        ParseResult::Success((), state.pos())
    } else if let Some(c) = state.peek_char() {
        if c == '\n' || c == '\r' {
            state.advance();
            ParseResult::Success((), state.pos())
        } else {
            state.fail("newline")
        }
    } else {
        state.fail("newline")
    }
}

/// Parse a character in a range
pub fn char_range<'a>(state: &mut ParserState<'a>, start: char, end: char) -> ParseResult<char> {
    state.match_char(
        |c| c >= start && c <= end,
        &format!("'{}'..'{}'", start, end),
    )
}

/// Parse a specific character
pub fn char_exact<'a>(state: &mut ParserState<'a>, expected: char) -> ParseResult<char> {
    state.match_char(|c| c == expected, &format!("'{}'", expected))
}

/// Parse start of input
pub fn soi<'a>(state: &mut ParserState<'a>) -> ParseResult<()> {
    if state.pos() == 0 {
        ParseResult::Success((), 0)
    } else {
        state.fail("start of input")
    }
}

/// Parse end of input
pub fn eoi<'a>(state: &mut ParserState<'a>) -> ParseResult<()> {
    if state.is_eof() {
        ParseResult::Success((), state.pos())
    } else {
        state.fail("end of input")
    }
}

/// Run a parser optionally (always succeeds)
pub fn optional<'a, T, F>(state: &mut ParserState<'a>, parser: F) -> ParseResult<Option<T>>
where
    F: FnOnce(&mut ParserState<'a>) -> ParseResult<T>,
{
    let start_pos = state.pos();
    let saved_bindings = state.save_bindings();

    match parser(state) {
        ParseResult::Success(v, pos) => ParseResult::Success(Some(v), pos),
        ParseResult::Failure(_) => {
            state.set_pos(start_pos);
            state.restore_bindings(saved_bindings);
            ParseResult::Success(None, start_pos)
        }
    }
}

/// Run a parser zero or more times
pub fn zero_or_more<'a, T, F>(state: &mut ParserState<'a>, mut parser: F) -> ParseResult<Vec<T>>
where
    F: FnMut(&mut ParserState<'a>) -> ParseResult<T>,
{
    let mut results = Vec::new();

    loop {
        let start_pos = state.pos();
        let saved_bindings = state.save_bindings();

        match parser(state) {
            ParseResult::Success(v, _) => {
                results.push(v);
            }
            ParseResult::Failure(_) => {
                state.set_pos(start_pos);
                state.restore_bindings(saved_bindings);
                break;
            }
        }
    }

    ParseResult::Success(results, state.pos())
}

/// Run a parser one or more times
pub fn one_or_more<'a, T, F>(state: &mut ParserState<'a>, mut parser: F) -> ParseResult<Vec<T>>
where
    F: FnMut(&mut ParserState<'a>) -> ParseResult<T>,
{
    // First must succeed
    let first = match parser(state) {
        ParseResult::Success(v, _) => v,
        ParseResult::Failure(e) => return ParseResult::Failure(e),
    };

    let mut results = vec![first];

    // Rest are optional
    loop {
        let start_pos = state.pos();
        let saved_bindings = state.save_bindings();

        match parser(state) {
            ParseResult::Success(v, _) => {
                results.push(v);
            }
            ParseResult::Failure(_) => {
                state.set_pos(start_pos);
                state.restore_bindings(saved_bindings);
                break;
            }
        }
    }

    ParseResult::Success(results, state.pos())
}

/// Parse items separated by a delimiter
pub fn sep_by<'a, T, D, FT, FD>(
    state: &mut ParserState<'a>,
    mut item: FT,
    mut delimiter: FD,
) -> ParseResult<Vec<T>>
where
    FT: FnMut(&mut ParserState<'a>) -> ParseResult<T>,
    FD: FnMut(&mut ParserState<'a>) -> ParseResult<D>,
{
    let mut results = Vec::new();

    // Try first item
    let start_pos = state.pos();
    let saved_bindings = state.save_bindings();

    match item(state) {
        ParseResult::Success(v, _) => {
            results.push(v);
        }
        ParseResult::Failure(_) => {
            state.set_pos(start_pos);
            state.restore_bindings(saved_bindings);
            return ParseResult::Success(results, start_pos);
        }
    }

    // Try delimiter + item repeatedly
    loop {
        let start_pos = state.pos();
        let saved_bindings = state.save_bindings();

        match delimiter(state) {
            ParseResult::Success(_, _) => {}
            ParseResult::Failure(_) => {
                state.set_pos(start_pos);
                state.restore_bindings(saved_bindings);
                break;
            }
        }

        match item(state) {
            ParseResult::Success(v, _) => {
                results.push(v);
            }
            ParseResult::Failure(_) => {
                state.set_pos(start_pos);
                state.restore_bindings(saved_bindings);
                break;
            }
        }
    }

    ParseResult::Success(results, state.pos())
}

/// Positive lookahead (succeeds if parser succeeds, doesn't consume)
pub fn positive_lookahead<'a, T, F>(state: &mut ParserState<'a>, parser: F) -> ParseResult<()>
where
    F: FnOnce(&mut ParserState<'a>) -> ParseResult<T>,
{
    let start_pos = state.pos();
    let saved_bindings = state.save_bindings();

    match parser(state) {
        ParseResult::Success(_, _) => {
            state.set_pos(start_pos);
            state.restore_bindings(saved_bindings);
            ParseResult::Success((), start_pos)
        }
        ParseResult::Failure(e) => {
            state.set_pos(start_pos);
            state.restore_bindings(saved_bindings);
            ParseResult::Failure(e)
        }
    }
}

/// Negative lookahead (succeeds if parser fails, doesn't consume)
pub fn negative_lookahead<'a, T, F>(state: &mut ParserState<'a>, parser: F) -> ParseResult<()>
where
    F: FnOnce(&mut ParserState<'a>) -> ParseResult<T>,
{
    let start_pos = state.pos();
    let saved_bindings = state.save_bindings();

    match parser(state) {
        ParseResult::Success(_, _) => {
            state.set_pos(start_pos);
            state.restore_bindings(saved_bindings);
            state.fail("not to match")
        }
        ParseResult::Failure(_) => {
            state.set_pos(start_pos);
            state.restore_bindings(saved_bindings);
            ParseResult::Success((), start_pos)
        }
    }
}

/// Try first parser, if it fails try second (ordered choice)
pub fn choice<'a, T, F1, F2>(state: &mut ParserState<'a>, first: F1, second: F2) -> ParseResult<T>
where
    F1: FnOnce(&mut ParserState<'a>) -> ParseResult<T>,
    F2: FnOnce(&mut ParserState<'a>) -> ParseResult<T>,
{
    let start_pos = state.pos();
    let saved_bindings = state.save_bindings();

    match first(state) {
        ParseResult::Success(v, pos) => ParseResult::Success(v, pos),
        ParseResult::Failure(e1) => {
            state.set_pos(start_pos);
            state.restore_bindings(saved_bindings);

            match second(state) {
                ParseResult::Success(v, pos) => ParseResult::Success(v, pos),
                ParseResult::Failure(e2) => ParseResult::Failure(e1.merge(e2)),
            }
        }
    }
}

/// Try multiple parsers in order (ordered choice)
pub fn choice_n<'a, T>(
    state: &mut ParserState<'a>,
    parsers: Vec<Box<dyn FnOnce(&mut ParserState<'a>) -> ParseResult<T> + 'a>>,
) -> ParseResult<T> {
    let start_pos = state.pos();
    let saved_bindings = state.save_bindings();
    let mut last_error: Option<ParseFailure> = None;

    for parser in parsers {
        state.set_pos(start_pos);
        state.restore_bindings(saved_bindings.clone());

        match parser(state) {
            ParseResult::Success(v, pos) => return ParseResult::Success(v, pos),
            ParseResult::Failure(e) => {
                last_error = Some(match last_error {
                    Some(prev) => prev.merge(e),
                    None => e,
                });
            }
        }
    }

    ParseResult::Failure(
        last_error.unwrap_or_else(|| {
            ParseFailure::new("choice", start_pos, state.line(), state.column())
        }),
    )
}

/// Run parsers in sequence, returning all results
pub fn sequence2<'a, T1, T2, F1, F2>(
    state: &mut ParserState<'a>,
    p1: F1,
    p2: F2,
) -> ParseResult<(T1, T2)>
where
    F1: FnOnce(&mut ParserState<'a>) -> ParseResult<T1>,
    F2: FnOnce(&mut ParserState<'a>) -> ParseResult<T2>,
{
    let v1 = match p1(state) {
        ParseResult::Success(v, _) => v,
        ParseResult::Failure(e) => return ParseResult::Failure(e),
    };

    let v2 = match p2(state) {
        ParseResult::Success(v, pos) => return ParseResult::Success((v1, v), pos),
        ParseResult::Failure(e) => return ParseResult::Failure(e),
    };
}

/// Run parsers in sequence, returning all results
pub fn sequence3<'a, T1, T2, T3, F1, F2, F3>(
    state: &mut ParserState<'a>,
    p1: F1,
    p2: F2,
    p3: F3,
) -> ParseResult<(T1, T2, T3)>
where
    F1: FnOnce(&mut ParserState<'a>) -> ParseResult<T1>,
    F2: FnOnce(&mut ParserState<'a>) -> ParseResult<T2>,
    F3: FnOnce(&mut ParserState<'a>) -> ParseResult<T3>,
{
    let v1 = match p1(state) {
        ParseResult::Success(v, _) => v,
        ParseResult::Failure(e) => return ParseResult::Failure(e),
    };

    let v2 = match p2(state) {
        ParseResult::Success(v, _) => v,
        ParseResult::Failure(e) => return ParseResult::Failure(e),
    };

    match p3(state) {
        ParseResult::Success(v3, pos) => ParseResult::Success((v1, v2, v3), pos),
        ParseResult::Failure(e) => ParseResult::Failure(e),
    }
}

/// Parse whitespace (wrapper for state.skip_ws)
pub fn ws<'a>(state: &mut ParserState<'a>) -> ParseResult<()> {
    state.skip_ws();
    ParseResult::Success((), state.pos())
}

/// Parse and capture text between two parsers
pub fn between<'a, O, C, T, FO, FC, FT>(
    state: &mut ParserState<'a>,
    open: FO,
    close: FC,
    content: FT,
) -> ParseResult<T>
where
    FO: FnOnce(&mut ParserState<'a>) -> ParseResult<O>,
    FC: FnOnce(&mut ParserState<'a>) -> ParseResult<C>,
    FT: FnOnce(&mut ParserState<'a>) -> ParseResult<T>,
{
    match open(state) {
        ParseResult::Success(_, _) => {}
        ParseResult::Failure(e) => return ParseResult::Failure(e),
    }

    let inner = match content(state) {
        ParseResult::Success(v, _) => v,
        ParseResult::Failure(e) => return ParseResult::Failure(e),
    };

    match close(state) {
        ParseResult::Success(_, pos) => ParseResult::Success(inner, pos),
        ParseResult::Failure(e) => ParseResult::Failure(e),
    }
}

/// Capture the matched text of a parser
pub fn capture<'a, T, F>(state: &mut ParserState<'a>, parser: F) -> ParseResult<String>
where
    F: FnOnce(&mut ParserState<'a>) -> ParseResult<T>,
{
    let start = state.pos();

    match parser(state) {
        ParseResult::Success(_, pos) => {
            let text = state.slice(start, pos).to_string();
            ParseResult::Success(text, pos)
        }
        ParseResult::Failure(e) => ParseResult::Failure(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::type_registry::TypeRegistry;
    use zyntax_typed_ast::TypedASTBuilder;

    fn make_state<'a>(
        input: &'a str,
        builder: &'a mut TypedASTBuilder,
        registry: &'a mut TypeRegistry,
    ) -> ParserState<'a> {
        ParserState::new(input, builder, registry)
    }

    #[test]
    fn test_literal() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = make_state("hello world", &mut builder, &mut registry);

        assert!(literal(&mut state, "hello").is_success());
        assert_eq!(state.pos(), 5);
    }

    #[test]
    fn test_optional() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = make_state("hello", &mut builder, &mut registry);

        let result = optional(&mut state, |s| literal(s, "world"));
        assert!(result.is_success());
        assert_eq!(state.pos(), 0); // Didn't consume

        let result = optional(&mut state, |s| literal(s, "hello"));
        assert!(result.is_success());
        assert_eq!(state.pos(), 5); // Did consume
    }

    #[test]
    fn test_zero_or_more() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = make_state("aaa", &mut builder, &mut registry);

        let result = zero_or_more(&mut state, |s| char_exact(s, 'a'));
        match result {
            ParseResult::Success(chars, _) => {
                assert_eq!(chars.len(), 3);
            }
            _ => panic!("Expected success"),
        }
    }

    #[test]
    fn test_one_or_more() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = make_state("aaa", &mut builder, &mut registry);

        let result = one_or_more(&mut state, |s| char_exact(s, 'a'));
        assert!(result.is_success());

        let mut state2 = make_state("bbb", &mut builder, &mut registry);
        let result2 = one_or_more(&mut state2, |s| char_exact(s, 'a'));
        assert!(!result2.is_success());
    }

    #[test]
    fn test_choice() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = make_state("hello", &mut builder, &mut registry);

        let result = choice(&mut state, |s| literal(s, "world"), |s| literal(s, "hello"));
        assert!(result.is_success());
    }

    #[test]
    fn test_capture() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = make_state("12345abc", &mut builder, &mut registry);

        let result = capture(&mut state, |s| one_or_more(s, ascii_digit));
        match result {
            ParseResult::Success(text, _) => {
                assert_eq!(text, "12345");
            }
            _ => panic!("Expected success"),
        }
    }
}
