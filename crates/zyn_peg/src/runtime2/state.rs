//! Parser State for ZynPEG 2.0
//!
//! ParserState manages the parsing context including:
//! - Input position tracking (with line/column for error messages)
//! - AstArena integration for string interning
//! - Type registry for type construction
//! - Packrat memoization cache

use super::memo::{MemoCache, MemoEntry, MemoKey};
use std::collections::HashMap;
use std::rc::Rc;
use zyntax_typed_ast::typed_ast::TypedCatch;
use zyntax_typed_ast::{
    type_registry::{PrimitiveType, Type, TypeRegistry},
    InternedString, Span, TypedASTBuilder,
};

/// Result of a parse attempt
#[derive(Debug, Clone)]
pub enum ParseResult<T> {
    /// Parse succeeded with value
    Success(T, usize), // (value, new_position)
    /// Parse failed (can backtrack)
    Failure(ParseFailure),
}

/// Information about a parse failure
#[derive(Debug, Clone)]
pub struct ParseFailure {
    pub expected: Vec<String>,
    pub pos: usize,
    pub line: usize,
    pub column: usize,
}

impl ParseFailure {
    /// A failure at a position, carrying no description.
    ///
    /// What the parser expected is reported from the state's furthest
    /// failure, which is the only one a reader sees. Every alternative
    /// a PEG tries produces a failure, so describing each one allocated
    /// a string per attempt that nothing ever read.
    pub fn new(_expected: &str, pos: usize, line: usize, column: usize) -> Self {
        ParseFailure {
            expected: Vec::new(),
            pos,
            line,
            column,
        }
    }

    pub fn merge(mut self, other: ParseFailure) -> Self {
        // Keep the failure that got furthest
        if other.pos > self.pos {
            other
        } else if other.pos == self.pos {
            self.expected.extend(other.expected);
            self
        } else {
            self
        }
    }
}

impl<T> ParseResult<T> {
    pub fn is_success(&self) -> bool {
        matches!(self, ParseResult::Success(_, _))
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> ParseResult<U> {
        match self {
            ParseResult::Success(v, pos) => ParseResult::Success(f(v), pos),
            ParseResult::Failure(e) => ParseResult::Failure(e),
        }
    }

    pub fn and_then<U, F: FnOnce(T, usize) -> ParseResult<U>>(self, f: F) -> ParseResult<U> {
        match self {
            ParseResult::Success(v, pos) => f(v, pos),
            ParseResult::Failure(e) => ParseResult::Failure(e),
        }
    }
}

/// Parser state with all context needed for parsing
pub struct ParserState<'a> {
    /// Input source code
    input: &'a str,
    /// Current byte position in input
    pos: usize,
    /// Current line number (1-based)
    line: usize,
    /// Current column number (1-based)
    column: usize,
    /// Line start positions for fast line lookup
    line_starts: Vec<usize>,
    /// AST builder for constructing nodes
    builder: &'a mut TypedASTBuilder,
    /// Type registry for type lookups
    type_registry: &'a mut TypeRegistry,
    /// Packrat memoization cache
    memo: MemoCache,
    /// Stack of local bindings for actions
    bindings: HashMap<String, ParsedValue>,
    /// What each change to `bindings` replaced, so an alternative that
    /// fails can be undone by rolling back to a mark instead of
    /// restoring a copy of the whole map.
    binding_undo: Vec<(String, Option<ParsedValue>)>,
    /// Furthest position reached (for error reporting)
    furthest_pos: usize,
    /// Expected items at furthest position
    furthest_expected: Vec<String>,
}

/// A parsed value that can be bound to a name
#[derive(Debug, Clone)]
pub enum ParsedValue {
    /// No value (from literals or failed optionals)
    None,
    /// String value (from identifiers, string literals)
    Text(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
    /// A span in the source
    Span(Span),
    /// Interned string
    Interned(InternedString),
    /// A type
    Type(Type),
    /// A list of values
    List(Vec<ParsedValue>),
    /// An optional value
    Optional(Option<Box<ParsedValue>>),
    /// A generic node handle (for complex AST nodes)
    Node(NodeHandle),
    // The node variants are shared rather than owned. Every rule's
    // value is cloned into the memo at the position it started, and a
    // hit clones it back out; owning them made each of those copy a
    // whole subtree. Sharing makes the copy a reference count, and a
    // consumer that needs the node to itself takes it back with
    // `own`, which only copies when something else still holds it.
    /// A TypedStatement AST node
    Statement(Rc<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedStatement>>),
    /// A TypedExpression AST node
    Expression(Rc<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedExpression>>),
    /// A TypedDeclaration AST node
    Declaration(Rc<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedDeclaration>>),
    /// A TypedProgram AST node
    ///
    /// Owned, not shared: a program is made once at the end of a parse
    /// and never copied on the way there, so sharing it would only put
    /// a reference count between a caller and the tree it asked for.
    Program(Box<zyntax_typed_ast::TypedProgram>),
    /// A TypedBlock AST node
    Block(Rc<zyntax_typed_ast::TypedBlock>),
    /// A field initialization (name -> value)
    FieldInit {
        name: InternedString,
        value: Box<ParsedValue>,
    },
    /// A function/method parameter
    Parameter(zyntax_typed_ast::TypedParameter),
    /// A TypedLiteral value
    Literal(zyntax_typed_ast::TypedLiteral),
    /// An enum variant
    Variant(zyntax_typed_ast::TypedVariant),
    /// A struct/class field
    Field(zyntax_typed_ast::TypedField),
    /// A postfix suffix for fold operations (field access, method call, etc.)
    Suffix {
        kind: String,
        fields: std::collections::HashMap<String, Box<ParsedValue>>,
    },
    /// A TypedPattern AST node
    Pattern(Box<zyntax_typed_ast::TypedNode<zyntax_typed_ast::TypedPattern>>),
    /// A TypedAnnotation (e.g., @deprecated, @inline)
    Annotation(zyntax_typed_ast::TypedAnnotation),
    /// A TypedAnnotationArg (positional or named)
    AnnotationArg(zyntax_typed_ast::TypedAnnotationArg),
    /// A TypedAnnotationValue (string, int, bool, identifier, list)
    AnnotationValue(zyntax_typed_ast::TypedAnnotationValue),
    /// A TypedEffectOp (effect operation declaration)
    EffectOp(zyntax_typed_ast::TypedEffectOp),
    /// A TypedEffectHandlerImpl (handler operation implementation)
    EffectHandlerImpl(zyntax_typed_ast::TypedEffectHandlerImpl),
    /// A lambda parameter
    LambdaParam(zyntax_typed_ast::TypedLambdaParam),
    /// A match arm (pattern -> body)
    MatchArm(zyntax_typed_ast::TypedMatchArm),
    /// A catch clause for try/catch statements
    Catch(TypedCatch),
}

/// Take a shared value back as an owned one.
///
/// Copies only when the value is still held elsewhere, which for a
/// node the memo remembers is the case exactly when a copy is needed.
pub fn own<T: Clone>(shared: Rc<T>) -> T {
    Rc::try_unwrap(shared).unwrap_or_else(|shared| (*shared).clone())
}

/// Handle to an AST node (opaque, managed by builder)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeHandle(pub usize);

impl<'a> ParserState<'a> {
    /// Create a new parser state
    pub fn new(
        input: &'a str,
        builder: &'a mut TypedASTBuilder,
        type_registry: &'a mut TypeRegistry,
    ) -> Self {
        // Pre-compute line starts for fast position lookups
        let mut line_starts = vec![0];
        for (i, c) in input.char_indices() {
            if c == '\n' {
                line_starts.push(i + 1);
            }
        }

        ParserState {
            input,
            pos: 0,
            line: 1,
            column: 1,
            line_starts,
            builder,
            type_registry,
            memo: MemoCache::new(),
            bindings: HashMap::new(),
            binding_undo: Vec::new(),
            furthest_pos: 0,
            furthest_expected: Vec::new(),
        }
    }

    // =========================================================================
    // Position Management
    // =========================================================================

    /// Get current position
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Set position (for backtracking)
    pub fn set_pos(&mut self, pos: usize) {
        if pos != self.pos {
            self.pos = pos;
            self.update_line_column();
        }
    }

    /// Get current line number
    pub fn line(&self) -> usize {
        self.line
    }

    /// Get current column number
    pub fn column(&self) -> usize {
        self.column
    }

    /// Check if at end of input
    pub fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    /// Get remaining input
    pub fn remaining(&self) -> &str {
        &self.input[self.pos..]
    }

    /// Length of the whole input, for reporting work against its size.
    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    /// Get a slice of input
    pub fn slice(&self, start: usize, end: usize) -> &str {
        &self.input[start..end]
    }

    /// Update line/column after position change
    fn update_line_column(&mut self) {
        // Binary search for the line
        let line_idx = self
            .line_starts
            .partition_point(|&start| start <= self.pos)
            .saturating_sub(1);

        self.line = line_idx + 1;
        self.column = self.pos - self.line_starts[line_idx] + 1;
    }

    /// Create a span from start to current position
    pub fn span_from(&self, start: usize) -> Span {
        Span::new(start, self.pos)
    }

    // =========================================================================
    // Character/String Matching
    // =========================================================================

    /// Peek at the byte at the current position.
    ///
    /// A byte rather than a character: what reads it only asks whether
    /// a pattern could begin here, and every byte that begins a
    /// multi-byte character is outside the ASCII sets it tests.
    pub fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.pos).copied()
    }

    /// Peek at current character
    pub fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    /// Peek at character at offset
    pub fn peek_char_at(&self, offset: usize) -> Option<char> {
        self.input[self.pos..].chars().nth(offset)
    }

    /// Advance by one character
    pub fn advance(&mut self) -> Option<char> {
        let c = self.peek_char()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    /// Advance by n bytes (must be valid UTF-8 boundary)
    pub fn advance_by(&mut self, n: usize) {
        for _ in 0..n {
            self.advance();
        }
    }

    /// Check if input starts with string at current position
    pub fn check(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }

    /// Match a literal string
    pub fn match_literal(&mut self, s: &str) -> ParseResult<()> {
        if self.check(s) {
            let start = self.pos;
            for _ in s.chars() {
                self.advance();
            }
            ParseResult::Success((), self.pos)
        } else {
            self.fail(&format!("'{}'", s))
        }
    }

    /// Match a character satisfying a predicate
    pub fn match_char<F: Fn(char) -> bool>(&mut self, pred: F, desc: &str) -> ParseResult<char> {
        match self.peek_char() {
            Some(c) if pred(c) => {
                self.advance();
                ParseResult::Success(c, self.pos)
            }
            _ => self.fail(desc),
        }
    }

    /// Match zero or more characters satisfying a predicate
    pub fn match_while<F: Fn(char) -> bool>(&mut self, pred: F) -> String {
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if pred(c) {
                self.advance();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }

    // =========================================================================
    // Error Handling
    // =========================================================================

    /// Record what was expected here without failing.
    ///
    /// An alternative decided against on the byte ahead of it never
    /// runs, so it never reports what it wanted, and the reader is left
    /// with whichever alternatives did run. This puts it back where a
    /// failure would have.
    pub fn note_expected(&mut self, expected: &str) {
        const MOST_EXPECTED: usize = 8;
        if self.pos > self.furthest_pos {
            self.furthest_pos = self.pos;
            self.furthest_expected.clear();
            self.furthest_expected.push(expected.to_string());
        } else if self.pos == self.furthest_pos
            && self.furthest_expected.len() < MOST_EXPECTED
            && !self.furthest_expected.iter().any(|e| e == expected)
        {
            self.furthest_expected.push(expected.to_string());
        }
    }

    /// Create a failure result
    pub fn fail<T>(&mut self, expected: &str) -> ParseResult<T> {
        // Track furthest failure for error reporting.
        //
        // A PEG fails constantly at the position it has reached, since
        // that is where every alternative is being tried, and each one
        // used to leave an owned string behind. Only the first few say
        // anything a reader can act on, so the rest are dropped.
        const MOST_EXPECTED: usize = 8;
        if self.pos > self.furthest_pos {
            self.furthest_pos = self.pos;
            self.furthest_expected.clear();
            self.furthest_expected.push(expected.to_string());
        } else if self.pos == self.furthest_pos && self.furthest_expected.len() < MOST_EXPECTED {
            self.furthest_expected.push(expected.to_string());
        }

        ParseResult::Failure(ParseFailure::new(
            expected,
            self.pos,
            self.line,
            self.column,
        ))
    }

    /// Get the furthest error information
    pub fn furthest_error(&self) -> ParseFailure {
        ParseFailure {
            expected: self.furthest_expected.clone(),
            pos: self.furthest_pos,
            line: self.line_for_pos(self.furthest_pos),
            column: self.column_for_pos(self.furthest_pos),
        }
    }

    fn line_for_pos(&self, pos: usize) -> usize {
        self.line_starts
            .partition_point(|&start| start <= pos)
            .max(1)
    }

    fn column_for_pos(&self, pos: usize) -> usize {
        let line_idx = self
            .line_starts
            .partition_point(|&start| start <= pos)
            .saturating_sub(1);
        pos - self.line_starts[line_idx] + 1
    }

    // =========================================================================
    // Memoization
    // =========================================================================

    /// Check memo cache for a rule at current position
    pub fn check_memo(&self, rule_id: usize) -> Option<&MemoEntry> {
        self.memo.get(MemoKey {
            pos: self.pos,
            rule_id,
        })
    }

    /// Store result in memo cache at current position
    pub fn store_memo(&mut self, rule_id: usize, entry: MemoEntry) {
        self.memo.insert(
            MemoKey {
                pos: self.pos,
                rule_id,
            },
            entry,
        );
    }

    /// Store result in memo cache at a specific position (use start_pos after rule completes)
    pub fn store_memo_at(&mut self, pos: usize, rule_id: usize, entry: MemoEntry) {
        self.memo.insert(MemoKey { pos, rule_id }, entry);
    }

    // =========================================================================
    // Binding Management
    // =========================================================================

    /// Set a binding
    pub fn set_binding(&mut self, name: &str, value: ParsedValue) {
        let previous = self.bindings.insert(name.to_string(), value);
        self.binding_undo.push((name.to_string(), previous));
    }

    /// Get a binding
    pub fn get_binding(&self, name: &str) -> Option<&ParsedValue> {
        self.bindings.get(name)
    }

    /// Clear all bindings (between rule invocations)
    ///
    /// Records what it removed, so a mark taken before the clear still
    /// rolls back to the bindings that were in place then.
    pub fn clear_bindings(&mut self) {
        for (name, value) in self.bindings.drain() {
            self.binding_undo.push((name, Some(value)));
        }
    }

    /// Mark the current bindings, to roll back to after a failed
    /// alternative.
    ///
    /// Returns a position in the undo log rather than a copy of the
    /// bindings. A PEG takes one of these per alternative tried and per
    /// repetition step, and copying the map made that cost the size of
    /// everything bound so far rather than the size of what the
    /// alternative changed.
    pub fn save_bindings(&self) -> usize {
        self.binding_undo.len()
    }

    /// Roll back to a mark from [`Self::save_bindings`].
    ///
    /// Undoing in reverse order restores the value each change
    /// replaced, so a name written several times ends at what it held
    /// when the mark was taken. Restoring twice from one mark is
    /// harmless: the second call finds nothing left to undo.
    pub fn restore_bindings(&mut self, mark: usize) {
        while self.binding_undo.len() > mark {
            let (name, previous) = match self.binding_undo.pop() {
                Some(entry) => entry,
                None => break,
            };
            match previous {
                Some(value) => {
                    self.bindings.insert(name, value);
                }
                None => {
                    self.bindings.remove(&name);
                }
            }
        }
    }

    // =========================================================================
    // AST Construction Helpers
    // =========================================================================

    /// Intern a string
    pub fn intern(&mut self, s: &str) -> InternedString {
        self.builder.intern(s)
    }

    /// Get text from a ParsedValue
    pub fn value_to_text(&self, value: &ParsedValue) -> Option<String> {
        match value {
            ParsedValue::Text(s) => Some(s.clone()),
            ParsedValue::Interned(s) => s.resolve_global().map(|s| s.to_string()),
            _ => None,
        }
    }

    /// Get a type by name
    pub fn get_type(&mut self, name: &str) -> Type {
        // Check for built-in types first
        match name {
            "i8" => Type::Primitive(PrimitiveType::I8),
            "i16" => Type::Primitive(PrimitiveType::I16),
            "i32" => Type::Primitive(PrimitiveType::I32),
            "i64" => Type::Primitive(PrimitiveType::I64),
            "u8" => Type::Primitive(PrimitiveType::U8),
            "u16" => Type::Primitive(PrimitiveType::U16),
            "u32" => Type::Primitive(PrimitiveType::U32),
            "u64" => Type::Primitive(PrimitiveType::U64),
            "f32" => Type::Primitive(PrimitiveType::F32),
            "f64" => Type::Primitive(PrimitiveType::F64),
            "bool" => Type::Primitive(PrimitiveType::Bool),
            "char" => Type::Primitive(PrimitiveType::Char),
            "str" | "String" => Type::Primitive(PrimitiveType::String),
            "unit" | "()" => Type::Primitive(PrimitiveType::Unit),
            "void" | "never" => Type::Never,
            _ => {
                // Look up in type registry or create unresolved
                let interned = self.builder.intern(name);
                Type::Unresolved(interned)
            }
        }
    }

    /// Access the builder
    pub fn builder(&mut self) -> &mut TypedASTBuilder {
        self.builder
    }

    /// Access the type registry
    pub fn type_registry(&mut self) -> &mut TypeRegistry {
        self.type_registry
    }
}

// =========================================================================
// Whitespace Handling
// =========================================================================

impl<'a> ParserState<'a> {
    /// Skip whitespace and comments
    pub fn skip_ws(&mut self) {
        loop {
            // Skip whitespace
            while let Some(c) = self.peek_char() {
                if c.is_ascii_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }

            // Skip line comments
            if self.check("//") {
                while let Some(c) = self.peek_char() {
                    self.advance();
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }

            // Skip block comments
            if self.check("/*") {
                self.advance();
                self.advance();
                let mut depth = 1;
                while depth > 0 && !self.is_eof() {
                    if self.check("/*") {
                        depth += 1;
                        self.advance();
                        self.advance();
                    } else if self.check("*/") {
                        depth -= 1;
                        self.advance();
                        self.advance();
                    } else {
                        self.advance();
                    }
                }
                continue;
            }

            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyntax_typed_ast::type_registry::TypeRegistry;
    use zyntax_typed_ast::TypedASTBuilder;

    #[test]
    fn test_position_tracking() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("hello\nworld", &mut builder, &mut registry);

        assert_eq!(state.line(), 1);
        assert_eq!(state.column(), 1);

        // Advance through "hello"
        for _ in 0..5 {
            state.advance();
        }
        assert_eq!(state.line(), 1);
        assert_eq!(state.column(), 6);

        // Advance past newline
        state.advance();
        assert_eq!(state.line(), 2);
        assert_eq!(state.column(), 1);
    }

    #[test]
    fn test_literal_matching() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("hello world", &mut builder, &mut registry);

        assert!(state.match_literal("hello").is_success());
        assert_eq!(state.pos(), 5);

        // Whitespace not automatically skipped
        assert!(!state.match_literal("world").is_success());

        state.skip_ws();
        assert!(state.match_literal("world").is_success());
    }

    #[test]
    fn test_bindings() {
        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("test", &mut builder, &mut registry);

        state.set_binding("name", ParsedValue::Text("foo".to_string()));
        assert!(matches!(
            state.get_binding("name"),
            Some(ParsedValue::Text(s)) if s == "foo"
        ));
    }
}
