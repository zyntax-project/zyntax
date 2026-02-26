//! Parser for ZynPEG 2.0 Grammar Syntax
//!
//! This module implements a hand-written recursive descent parser for .zyn grammar files.
//! It parses the new syntax with named bindings and produces GrammarIR.
//!
//! # Grammar Syntax
//!
//! ```text
//! @language { name: "MyLang", version: "1.0" }
//!
//! @imports {
//!     use zyntax_typed_ast::*;
//! }
//!
//! // Rule with named bindings and action
//! fn_def = { "fn" ~ name:identifier ~ "(" ~ params:fn_params? ~ ")" ~ body:block }
//!   -> TypedDeclaration::Function {
//!       name: name.text,
//!       params: params.unwrap_or_default(),
//!       body: body,
//!   }
//!
//! // Atomic rule (no implicit whitespace)
//! identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
//!
//! // Silent rule (doesn't appear in AST)
//! WHITESPACE = _{ " " | "\t" | "\n" }
//! ```

use super::ir::*;
use std::collections::HashMap;

/// Parser state for grammar parsing
pub struct GrammarParser<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
    column: usize,
}

/// Parse error with location
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub pos: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for ParseError {}

pub type ParseResult<T> = Result<T, ParseError>;

impl<'a> GrammarParser<'a> {
    /// Create a new parser for the given input
    pub fn new(input: &'a str) -> Self {
        GrammarParser {
            input,
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    /// Parse a complete grammar file
    pub fn parse_grammar(&mut self) -> ParseResult<GrammarIR> {
        let mut grammar = GrammarIR::new();

        self.skip_ws_and_comments();

        while !self.is_eof() {
            if self.peek_char() == Some('@') {
                self.parse_directive(&mut grammar)?;
            } else if self.peek_identifier().is_some() {
                let rule = self.parse_rule()?;
                grammar.add_rule(rule);
            } else {
                self.skip_ws_and_comments();
                if !self.is_eof() {
                    return Err(self.error("Expected directive or rule definition"));
                }
            }
            self.skip_ws_and_comments();
        }

        Ok(grammar)
    }

    /// Parse a directive (@language, @imports, etc.)
    fn parse_directive(&mut self, grammar: &mut GrammarIR) -> ParseResult<()> {
        self.expect_char('@')?;
        let directive_name = self.parse_identifier()?;

        self.skip_ws();

        match directive_name.as_str() {
            "language" => {
                grammar.metadata = self.parse_language_directive()?;
            }
            "imports" => {
                let code = self.parse_braced_code()?;
                for line in code.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        grammar.imports.push(trimmed.to_string());
                    }
                }
            }
            "types" => {
                grammar.type_decls = self.parse_types_directive()?;
            }
            "builtin" => {
                grammar.builtins = self.parse_builtin_directive()?;
            }
            "context" | "type_helpers" | "error_messages" => {
                // Skip these directives for now (not needed for core functionality)
                let _ = self.parse_braced_code()?;
            }
            _ => {
                return Err(self.error(&format!("Unknown directive: @{}", directive_name)));
            }
        }

        Ok(())
    }

    /// Parse @language { name: "...", version: "..." }
    fn parse_language_directive(&mut self) -> ParseResult<GrammarMetadata> {
        self.expect_char('{')?;
        self.skip_ws_and_comments();

        let mut metadata = GrammarMetadata::default();

        while self.peek_char() != Some('}') {
            let field_name = self.parse_identifier()?;
            self.skip_ws();
            self.expect_char(':')?;
            self.skip_ws();

            match field_name.as_str() {
                "name" => {
                    metadata.name = self.parse_string_literal()?;
                }
                "version" => {
                    metadata.version = self.parse_string_literal()?;
                }
                "file_extensions" => {
                    metadata.file_extensions = self.parse_string_list()?;
                }
                "entry_point" => {
                    metadata.entry_point = Some(self.parse_string_literal()?);
                }
                _ => {
                    // Skip unknown fields
                    self.skip_to_next_field()?;
                }
            }

            self.skip_ws_and_comments();
            // Optional comma
            if self.peek_char() == Some(',') {
                self.advance();
            }
            self.skip_ws_and_comments();
        }

        self.expect_char('}')?;
        Ok(metadata)
    }

    /// Parse @types { opaque: [...], returns: { ... } }
    fn parse_types_directive(&mut self) -> ParseResult<TypeDeclarations> {
        self.expect_char('{')?;
        self.skip_ws_and_comments();

        let mut types = TypeDeclarations::default();

        while self.peek_char() != Some('}') {
            let field_name = self.parse_identifier()?;
            self.skip_ws();
            self.expect_char(':')?;
            self.skip_ws();

            match field_name.as_str() {
                "opaque" => {
                    types.opaque_types = self.parse_identifier_list()?;
                }
                "returns" => {
                    types.function_returns = self.parse_string_map()?;
                }
                _ => {
                    self.skip_to_next_field()?;
                }
            }

            self.skip_ws_and_comments();
            if self.peek_char() == Some(',') {
                self.advance();
            }
            self.skip_ws_and_comments();
        }

        self.expect_char('}')?;
        Ok(types)
    }

    /// Parse @builtin { func: "symbol", ... }
    fn parse_builtin_directive(&mut self) -> ParseResult<BuiltinMappings> {
        self.expect_char('{')?;
        self.skip_ws_and_comments();

        let mut builtins = BuiltinMappings::default();

        while self.peek_char() != Some('}') {
            let name = self.parse_builtin_name()?;
            self.skip_ws();
            self.expect_char(':')?;
            self.skip_ws();
            let symbol = self.parse_string_literal()?;

            // Check for method prefix (@) or operator prefix ($)
            if let Some(method_name) = name.strip_prefix('@') {
                builtins
                    .methods
                    .entry(method_name.to_string())
                    .or_insert_with(Vec::new)
                    .push(symbol);
            } else if let Some(op) = name.strip_prefix('$') {
                builtins
                    .operators
                    .entry(op.to_string())
                    .or_insert_with(Vec::new)
                    .push(symbol);
            } else {
                builtins.functions.insert(name, symbol);
            }

            self.skip_ws_and_comments();
            if self.peek_char() == Some(',') {
                self.advance();
            }
            self.skip_ws_and_comments();
        }

        self.expect_char('}')?;
        Ok(builtins)
    }

    /// Parse a grammar rule
    fn parse_rule(&mut self) -> ParseResult<RuleIR> {
        let name = self.parse_identifier()?;
        self.skip_ws();

        self.expect_char('=')?;
        self.skip_ws();

        // Check for modifier
        let modifier = self.parse_rule_modifier();
        self.skip_ws();

        self.expect_char('{')?;
        self.skip_ws_and_comments();

        let pattern = self.parse_pattern()?;

        self.skip_ws_and_comments();
        self.expect_char('}')?;

        self.skip_ws_and_comments();

        // Check for action block
        let (action, return_type) = if self.check_str("->") {
            self.advance_by(2);
            self.skip_ws();
            let (action, ret_type) = self.parse_action()?;
            (Some(action), Some(ret_type))
        } else {
            (None, None)
        };

        Ok(RuleIR {
            name,
            modifier,
            pattern,
            action,
            return_type,
        })
    }

    /// Parse rule modifier (@, _, $, !)
    fn parse_rule_modifier(&mut self) -> Option<RuleModifier> {
        match self.peek_char() {
            Some('@') => {
                self.advance();
                Some(RuleModifier::Atomic)
            }
            Some('_') => {
                self.advance();
                Some(RuleModifier::Silent)
            }
            Some('$') => {
                self.advance();
                Some(RuleModifier::Compound)
            }
            Some('!') => {
                self.advance();
                Some(RuleModifier::NonAtomic)
            }
            _ => None,
        }
    }

    /// Parse a pattern (alternation level)
    fn parse_pattern(&mut self) -> ParseResult<PatternIR> {
        let first = self.parse_sequence()?;

        self.skip_ws_and_comments();

        if self.peek_char() == Some('|') {
            let mut choices = vec![first];
            while self.peek_char() == Some('|') {
                self.advance();
                self.skip_ws_and_comments();
                choices.push(self.parse_sequence()?);
                self.skip_ws_and_comments();
            }
            Ok(PatternIR::Choice(choices))
        } else {
            Ok(first)
        }
    }

    /// Parse a sequence of patterns
    fn parse_sequence(&mut self) -> ParseResult<PatternIR> {
        let mut items = Vec::new();

        loop {
            self.skip_ws_and_comments();

            // Check for end of sequence
            match self.peek_char() {
                Some('}') | Some('|') | Some(')') | None => break,
                _ => {}
            }

            // Skip tilde separator
            if self.peek_char() == Some('~') {
                self.advance();
                self.skip_ws_and_comments();
            }

            let item = self.parse_postfix()?;
            items.push(item);
        }

        if items.is_empty() {
            Err(self.error("Expected at least one pattern element"))
        } else if items.len() == 1 {
            Ok(items.pop().unwrap())
        } else {
            Ok(PatternIR::Sequence(items))
        }
    }

    /// Parse postfix operators (?, *, +, {n,m})
    fn parse_postfix(&mut self) -> ParseResult<PatternIR> {
        let base = self.parse_prefix()?;
        self.skip_ws();

        match self.peek_char() {
            Some('?') => {
                self.advance();
                Ok(PatternIR::Optional(Box::new(base)))
            }
            Some('*') => {
                self.advance();
                Ok(PatternIR::Repeat {
                    pattern: Box::new(base),
                    min: 0,
                    max: None,
                    separator: None,
                })
            }
            Some('+') => {
                self.advance();
                Ok(PatternIR::Repeat {
                    pattern: Box::new(base),
                    min: 1,
                    max: None,
                    separator: None,
                })
            }
            _ => Ok(base),
        }
    }

    /// Parse prefix operators (&, !)
    fn parse_prefix(&mut self) -> ParseResult<PatternIR> {
        match self.peek_char() {
            Some('&') => {
                self.advance();
                self.skip_ws();
                let inner = self.parse_primary()?;
                Ok(PatternIR::PositiveLookahead(Box::new(inner)))
            }
            Some('!') if !self.check_str("!=") => {
                self.advance();
                self.skip_ws();
                let inner = self.parse_primary()?;
                Ok(PatternIR::NegativeLookahead(Box::new(inner)))
            }
            _ => self.parse_primary(),
        }
    }

    /// Parse primary pattern elements
    fn parse_primary(&mut self) -> ParseResult<PatternIR> {
        self.skip_ws_and_comments();

        match self.peek_char() {
            Some('"') => {
                let s = self.parse_string_literal()?;
                Ok(PatternIR::Literal(s))
            }
            Some('\'') => {
                let class = self.parse_char_class()?;
                Ok(PatternIR::CharClass(class))
            }
            Some('(') => {
                self.advance();
                self.skip_ws_and_comments();
                let inner = self.parse_pattern()?;
                self.skip_ws_and_comments();
                self.expect_char(')')?;
                Ok(inner)
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => self.parse_rule_ref_or_builtin(),
            _ => Err(self.error("Expected pattern element")),
        }
    }

    /// Parse rule reference (potentially with binding) or built-in
    fn parse_rule_ref_or_builtin(&mut self) -> ParseResult<PatternIR> {
        // Check for binding syntax: name:rule
        let first_ident = self.parse_identifier()?;
        self.skip_ws();

        if self.peek_char() == Some(':') && !self.check_str("::") {
            // This is a binding
            self.advance();
            self.skip_ws();
            let rule_name = self.parse_identifier()?;

            // Handle built-in names
            let pattern = self.make_pattern_for_name(&rule_name);

            if let PatternIR::RuleRef { rule_name: rn, .. } = pattern {
                Ok(PatternIR::RuleRef {
                    rule_name: rn,
                    binding: Some(first_ident),
                })
            } else {
                // Built-ins can't have bindings directly, wrap in a rule ref
                Ok(PatternIR::RuleRef {
                    rule_name,
                    binding: Some(first_ident),
                })
            }
        } else {
            // No binding, just a rule reference or built-in
            Ok(self.make_pattern_for_name(&first_ident))
        }
    }

    /// Convert a name to a pattern (handling built-ins)
    fn make_pattern_for_name(&self, name: &str) -> PatternIR {
        match name {
            "ANY" => PatternIR::Any,
            "SOI" => PatternIR::StartOfInput,
            "EOI" => PatternIR::EndOfInput,
            "WHITESPACE" => PatternIR::Whitespace,
            "ASCII_DIGIT" => PatternIR::CharClass(CharClass::Builtin("ASCII_DIGIT".to_string())),
            "ASCII_ALPHA" => PatternIR::CharClass(CharClass::Builtin("ASCII_ALPHA".to_string())),
            "ASCII_ALPHANUMERIC" => {
                PatternIR::CharClass(CharClass::Builtin("ASCII_ALPHANUMERIC".to_string()))
            }
            "ASCII_HEX_DIGIT" => {
                PatternIR::CharClass(CharClass::Builtin("ASCII_HEX_DIGIT".to_string()))
            }
            "NEWLINE" => PatternIR::CharClass(CharClass::Builtin("NEWLINE".to_string())),
            _ => PatternIR::RuleRef {
                rule_name: name.to_string(),
                binding: None,
            },
        }
    }

    /// Parse a character class: 'a'..'z' or 'a'
    fn parse_char_class(&mut self) -> ParseResult<CharClass> {
        self.expect_char('\'')?;
        let c = self.parse_char_in_literal()?;
        self.expect_char('\'')?;

        self.skip_ws();

        // Check for range
        if self.check_str("..") {
            self.advance_by(2);
            self.skip_ws();
            self.expect_char('\'')?;
            let c2 = self.parse_char_in_literal()?;
            self.expect_char('\'')?;
            Ok(CharClass::Range(c, c2))
        } else {
            Ok(CharClass::Single(c))
        }
    }

    /// Parse an action block
    fn parse_action(&mut self) -> ParseResult<(ActionIR, String)> {
        // Parse first identifier/type path
        let first_path = self.parse_type_path()?;
        self.skip_ws();

        // Check for function call: -> intern(name)
        if self.peek_char() == Some('(') {
            self.advance(); // consume '('
            self.skip_ws();

            // Parse arguments
            let mut args = Vec::new();
            while self.peek_char() != Some(')') {
                let arg = self.parse_expr()?;
                args.push(arg);
                self.skip_ws();
                if self.peek_char() == Some(',') {
                    self.advance();
                    self.skip_ws();
                }
            }
            self.expect_char(')')?;

            // Function call action
            return Ok((
                ActionIR::HelperCall {
                    function: first_path.clone(),
                    args,
                },
                "Any".to_string(),
            ));
        }

        // Check for simple pass-through: -> binding (no '{' follows)
        // If the first_path is a simple identifier (no ::) and no '{' follows,
        // it's a pass-through action
        if !self.check_str("{") {
            // Check if it's a simple identifier (no :: in path) - that's pass-through
            if !first_path.contains("::") {
                // No '::' means it's just a binding name, not a type path
                return Ok((
                    ActionIR::PassThrough {
                        binding: first_path.clone(),
                    },
                    first_path,
                ));
            }
            // If it has ::, then there should be a { with fields
            return Err(self.error(&format!("Expected '{{' after type path '{}'", first_path)));
        }

        let return_type = first_path;

        // Full action block
        self.expect_char('{')?;
        self.skip_ws_and_comments();

        // Check if this is legacy JSON syntax (starts with quoted string key)
        if self.peek_char() == Some('"') {
            // Legacy JSON action - capture everything until matching closing brace
            let json_content = self.parse_json_block_content()?;
            self.expect_char('}')?;
            return Ok((
                ActionIR::LegacyJson {
                    return_type: return_type.clone(),
                    json_content,
                },
                return_type,
            ));
        }

        let mut fields = Vec::new();

        while self.peek_char() != Some('}') {
            let field_name = self.parse_identifier()?;
            self.skip_ws();
            self.expect_char(':')?;
            self.skip_ws();

            let value = self.parse_expr()?;
            fields.push((field_name, value));

            self.skip_ws_and_comments();
            if self.peek_char() == Some(',') {
                self.advance();
            }
            self.skip_ws_and_comments();
        }

        self.expect_char('}')?;

        Ok((
            ActionIR::Construct {
                type_path: return_type.clone(),
                fields,
            },
            return_type,
        ))
    }

    /// Parse the content of a JSON action block (legacy format)
    /// This captures everything between { and the matching }
    fn parse_json_block_content(&mut self) -> ParseResult<String> {
        let start = self.pos;
        let mut depth = 0;

        loop {
            match self.peek_char() {
                Some('{') => {
                    depth += 1;
                    self.advance();
                }
                Some('}') => {
                    if depth == 0 {
                        // Found the closing brace of the outer block
                        let content = self.input[start..self.pos].to_string();
                        return Ok(content);
                    }
                    depth -= 1;
                    self.advance();
                }
                Some('"') => {
                    // Skip string literals (they might contain braces)
                    self.advance();
                    while let Some(c) = self.peek_char() {
                        if c == '"' {
                            self.advance();
                            break;
                        }
                        if c == '\\' {
                            self.advance();
                            self.advance(); // Skip escaped char
                        } else {
                            self.advance();
                        }
                    }
                }
                Some('[') => {
                    // Array - just advance
                    self.advance();
                }
                Some(']') => {
                    self.advance();
                }
                Some(_) => {
                    self.advance();
                }
                None => {
                    return Err(self.error("Unexpected EOF in JSON block"));
                }
            }
        }
    }

    /// Parse a type path like TypedExpression::Binary
    fn parse_type_path(&mut self) -> ParseResult<String> {
        let mut path = self.parse_identifier()?;

        while self.check_str("::") {
            path.push_str("::");
            self.advance_by(2);
            path.push_str(&self.parse_identifier()?);
        }

        Ok(path)
    }

    /// Parse an expression in action code
    fn parse_expr(&mut self) -> ParseResult<ExprIR> {
        self.parse_expr_or()
    }

    /// Parse || expressions
    fn parse_expr_or(&mut self) -> ParseResult<ExprIR> {
        let mut left = self.parse_expr_and()?;

        while self.check_str("||") {
            self.advance_by(2);
            self.skip_ws();
            let right = self.parse_expr_and()?;
            left = ExprIR::Binary {
                left: Box::new(left),
                op: "||".to_string(),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse && expressions
    fn parse_expr_and(&mut self) -> ParseResult<ExprIR> {
        let mut left = self.parse_expr_comparison()?;

        while self.check_str("&&") {
            self.advance_by(2);
            self.skip_ws();
            let right = self.parse_expr_comparison()?;
            left = ExprIR::Binary {
                left: Box::new(left),
                op: "&&".to_string(),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// Parse comparison expressions
    fn parse_expr_comparison(&mut self) -> ParseResult<ExprIR> {
        let left = self.parse_expr_primary()?;
        self.skip_ws();

        let op = if self.check_str("==") {
            self.advance_by(2);
            "=="
        } else if self.check_str("!=") {
            self.advance_by(2);
            "!="
        } else {
            return Ok(left);
        };

        self.skip_ws();
        let right = self.parse_expr_primary()?;

        Ok(ExprIR::Binary {
            left: Box::new(left),
            op: op.to_string(),
            right: Box::new(right),
        })
    }

    /// Parse primary expressions (with postfix operations)
    fn parse_expr_primary(&mut self) -> ParseResult<ExprIR> {
        let mut expr = self.parse_expr_atom()?;

        loop {
            self.skip_ws();
            match self.peek_char() {
                Some('.') if !self.check_str("..") => {
                    self.advance();
                    let name = self.parse_identifier()?;
                    self.skip_ws();

                    if self.peek_char() == Some('(') {
                        // Method call
                        self.advance();
                        let args = self.parse_arg_list()?;
                        self.expect_char(')')?;

                        // Handle special methods
                        expr = match name.as_str() {
                            "unwrap_or" => {
                                if args.len() != 1 {
                                    return Err(self.error("unwrap_or requires exactly 1 argument"));
                                }
                                ExprIR::UnwrapOr {
                                    optional: Box::new(expr),
                                    default: Box::new(args.into_iter().next().unwrap()),
                                }
                            }
                            "unwrap_or_default" => ExprIR::UnwrapOr {
                                optional: Box::new(expr),
                                default: Box::new(ExprIR::Default("".to_string())),
                            },
                            "is_some" => ExprIR::IsSome(Box::new(expr)),
                            "text" => ExprIR::Text(Box::new(expr)),
                            "span" => ExprIR::GetSpan(Box::new(expr)),
                            _ => ExprIR::MethodCall {
                                receiver: Box::new(expr),
                                method: name,
                                args,
                            },
                        };
                    } else {
                        // Field access
                        expr = ExprIR::FieldAccess {
                            base: Box::new(expr),
                            field: name,
                        };
                    }
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    /// Parse atomic expressions
    fn parse_expr_atom(&mut self) -> ParseResult<ExprIR> {
        self.skip_ws();

        match self.peek_char() {
            Some('"') => {
                let s = self.parse_string_literal()?;
                Ok(ExprIR::StringLit(s))
            }
            Some('[') => {
                self.advance();
                let items = self.parse_arg_list()?;
                self.expect_char(']')?;
                Ok(ExprIR::List(items))
            }
            Some(c) if c.is_ascii_digit() => {
                let n = self.parse_integer()?;
                Ok(ExprIR::IntLit(n))
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let name = self.parse_identifier()?;

                // Check for keywords
                match name.as_str() {
                    "true" => return Ok(ExprIR::BoolLit(true)),
                    "false" => return Ok(ExprIR::BoolLit(false)),
                    "None" => {
                        return Ok(ExprIR::EnumVariant {
                            type_name: "Option".to_string(),
                            variant: "None".to_string(),
                            value: None,
                        })
                    }
                    _ => {}
                }

                self.skip_ws();

                // Check for :: (enum variant or type path)
                if self.check_str("::") {
                    self.advance_by(2);
                    let variant = self.parse_identifier()?;
                    self.skip_ws();

                    // Check for value
                    if self.peek_char() == Some('(') {
                        self.advance();
                        let arg = self.parse_expr()?;
                        self.expect_char(')')?;
                        return Ok(ExprIR::EnumVariant {
                            type_name: name,
                            variant,
                            value: Some(Box::new(arg)),
                        });
                    } else if self.peek_char() == Some('{') {
                        // Struct-like enum variant
                        self.advance();
                        self.skip_ws_and_comments();
                        let mut fields = Vec::new();

                        while self.peek_char() != Some('}') {
                            let field_name = self.parse_identifier()?;
                            self.skip_ws();
                            self.expect_char(':')?;
                            self.skip_ws();
                            let value = self.parse_expr()?;
                            fields.push((field_name, value));

                            self.skip_ws_and_comments();
                            if self.peek_char() == Some(',') {
                                self.advance();
                            }
                            self.skip_ws_and_comments();
                        }
                        self.expect_char('}')?;

                        return Ok(ExprIR::StructLit {
                            type_name: format!("{}::{}", name, variant),
                            fields,
                        });
                    }

                    return Ok(ExprIR::EnumVariant {
                        type_name: name,
                        variant,
                        value: None,
                    });
                }

                // Check for function call
                if self.peek_char() == Some('(') {
                    self.advance();
                    let args = self.parse_arg_list()?;
                    self.expect_char(')')?;

                    // Handle special functions
                    return Ok(match name.as_str() {
                        "intern" => {
                            if args.len() != 1 {
                                return Err(self.error("intern requires exactly 1 argument"));
                            }
                            ExprIR::Intern(Box::new(args.into_iter().next().unwrap()))
                        }
                        "Some" => {
                            if args.len() != 1 {
                                return Err(self.error("Some requires exactly 1 argument"));
                            }
                            ExprIR::EnumVariant {
                                type_name: "Option".to_string(),
                                variant: "Some".to_string(),
                                value: Some(Box::new(args.into_iter().next().unwrap())),
                            }
                        }
                        _ => ExprIR::FunctionCall {
                            function: name,
                            args,
                        },
                    });
                }

                // Check for struct literal
                if self.peek_char() == Some('{') {
                    self.advance();
                    self.skip_ws_and_comments();
                    let mut fields = Vec::new();

                    while self.peek_char() != Some('}') {
                        let field_name = self.parse_identifier()?;
                        self.skip_ws();
                        self.expect_char(':')?;
                        self.skip_ws();
                        let value = self.parse_expr()?;
                        fields.push((field_name, value));

                        self.skip_ws_and_comments();
                        if self.peek_char() == Some(',') {
                            self.advance();
                        }
                        self.skip_ws_and_comments();
                    }
                    self.expect_char('}')?;

                    return Ok(ExprIR::StructLit {
                        type_name: name,
                        fields,
                    });
                }

                // Simple binding reference
                Ok(ExprIR::Binding(name))
            }
            Some('(') => {
                self.advance();
                self.skip_ws();
                let inner = self.parse_expr()?;
                self.skip_ws();
                self.expect_char(')')?;
                Ok(inner)
            }
            _ => Err(self.error("Expected expression")),
        }
    }

    /// Parse argument list (comma-separated expressions, allows trailing comma)
    fn parse_arg_list(&mut self) -> ParseResult<Vec<ExprIR>> {
        let mut args = Vec::new();
        self.skip_ws();

        if self.peek_char() == Some(')') || self.peek_char() == Some(']') {
            return Ok(args);
        }

        args.push(self.parse_expr()?);

        loop {
            self.skip_ws();
            if self.peek_char() == Some(',') {
                self.advance();
                self.skip_ws();
                // Check for trailing comma (next is ) or ])
                if self.peek_char() == Some(')') || self.peek_char() == Some(']') {
                    break;
                }
                args.push(self.parse_expr()?);
            } else {
                break;
            }
        }

        Ok(args)
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn is_eof(&self) -> bool {
        self.pos >= self.input.len()
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn peek_char_at(&self, offset: usize) -> Option<char> {
        self.input[self.pos..].chars().nth(offset)
    }

    fn advance(&mut self) {
        if let Some(c) = self.peek_char() {
            self.pos += c.len_utf8();
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
    }

    fn advance_by(&mut self, n: usize) {
        for _ in 0..n {
            self.advance();
        }
    }

    fn check_str(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }

    fn expect_char(&mut self, expected: char) -> ParseResult<()> {
        match self.peek_char() {
            Some(c) if c == expected => {
                self.advance();
                Ok(())
            }
            Some(c) => Err(self.error(&format!("Expected '{}', found '{}'", expected, c))),
            None => Err(self.error(&format!("Expected '{}', found EOF", expected))),
        }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            self.skip_ws();

            // Line comment
            if self.check_str("//") {
                while let Some(c) = self.peek_char() {
                    self.advance();
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }

            // Block comment
            if self.check_str("/*") {
                self.advance_by(2);
                let mut depth = 1;
                while depth > 0 && !self.is_eof() {
                    if self.check_str("/*") {
                        depth += 1;
                        self.advance_by(2);
                    } else if self.check_str("*/") {
                        depth -= 1;
                        self.advance_by(2);
                    } else {
                        self.advance();
                    }
                }
                continue;
            }

            break;
        }
    }

    fn peek_identifier(&self) -> Option<&str> {
        let start = self.pos;
        let rest = &self.input[start..];

        let first = rest.chars().next()?;
        if !first.is_ascii_alphabetic() && first != '_' {
            return None;
        }

        let len = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .map(|c| c.len_utf8())
            .sum();

        Some(&rest[..len])
    }

    fn parse_identifier(&mut self) -> ParseResult<String> {
        let start = self.pos;

        match self.peek_char() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                self.advance();
            }
            _ => return Err(self.error("Expected identifier")),
        }

        while let Some(c) = self.peek_char() {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        Ok(self.input[start..self.pos].to_string())
    }

    fn parse_builtin_name(&mut self) -> ParseResult<String> {
        // Builtin names can start with @ or $ for method/operator prefixes
        let mut name = String::new();

        if let Some(c) = self.peek_char() {
            if c == '@' || c == '$' {
                name.push(c);
                self.advance();
            }
        }

        // Rest is identifier-like or operator chars
        if name.starts_with('$') {
            // Operator: can include things like *, +, @, etc.
            while let Some(c) = self.peek_char() {
                if c.is_ascii_alphanumeric() || "*+-/<>=!@%^&|".contains(c) {
                    name.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
        } else {
            // Method or function name
            name.push_str(&self.parse_identifier()?);
        }

        Ok(name)
    }

    fn parse_string_literal(&mut self) -> ParseResult<String> {
        self.expect_char('"')?;
        let mut result = String::new();

        while let Some(c) = self.peek_char() {
            if c == '"' {
                self.advance();
                return Ok(result);
            } else if c == '\\' {
                self.advance();
                match self.peek_char() {
                    Some('n') => {
                        result.push('\n');
                        self.advance();
                    }
                    Some('t') => {
                        result.push('\t');
                        self.advance();
                    }
                    Some('r') => {
                        result.push('\r');
                        self.advance();
                    }
                    Some('\\') => {
                        result.push('\\');
                        self.advance();
                    }
                    Some('"') => {
                        result.push('"');
                        self.advance();
                    }
                    Some(c) => {
                        result.push(c);
                        self.advance();
                    }
                    None => return Err(self.error("Unexpected EOF in string")),
                }
            } else {
                result.push(c);
                self.advance();
            }
        }

        Err(self.error("Unterminated string literal"))
    }

    fn parse_char_in_literal(&mut self) -> ParseResult<char> {
        match self.peek_char() {
            Some('\\') => {
                self.advance();
                match self.peek_char() {
                    Some('n') => {
                        self.advance();
                        Ok('\n')
                    }
                    Some('t') => {
                        self.advance();
                        Ok('\t')
                    }
                    Some('r') => {
                        self.advance();
                        Ok('\r')
                    }
                    Some('\\') => {
                        self.advance();
                        Ok('\\')
                    }
                    Some('\'') => {
                        self.advance();
                        Ok('\'')
                    }
                    Some(c) => {
                        self.advance();
                        Ok(c)
                    }
                    None => Err(self.error("Unexpected EOF in character")),
                }
            }
            Some(c) => {
                self.advance();
                Ok(c)
            }
            None => Err(self.error("Unexpected EOF")),
        }
    }

    fn parse_integer(&mut self) -> ParseResult<i64> {
        let start = self.pos;

        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        let s = &self.input[start..self.pos];
        s.parse()
            .map_err(|_| self.error(&format!("Invalid integer: {}", s)))
    }

    fn parse_string_list(&mut self) -> ParseResult<Vec<String>> {
        self.expect_char('[')?;
        self.skip_ws_and_comments();

        let mut result = Vec::new();

        while self.peek_char() != Some(']') {
            result.push(self.parse_string_literal()?);
            self.skip_ws_and_comments();
            if self.peek_char() == Some(',') {
                self.advance();
            }
            self.skip_ws_and_comments();
        }

        self.expect_char(']')?;
        Ok(result)
    }

    fn parse_identifier_list(&mut self) -> ParseResult<Vec<String>> {
        self.expect_char('[')?;
        self.skip_ws_and_comments();

        let mut result = Vec::new();

        while self.peek_char() != Some(']') {
            // Identifiers might start with $ for opaque types
            let mut name = String::new();
            if self.peek_char() == Some('$') {
                name.push('$');
                self.advance();
            }
            name.push_str(&self.parse_identifier()?);
            result.push(name);

            self.skip_ws_and_comments();
            if self.peek_char() == Some(',') {
                self.advance();
            }
            self.skip_ws_and_comments();
        }

        self.expect_char(']')?;
        Ok(result)
    }

    fn parse_string_map(&mut self) -> ParseResult<HashMap<String, String>> {
        self.expect_char('{')?;
        self.skip_ws_and_comments();

        let mut result = HashMap::new();

        while self.peek_char() != Some('}') {
            let key = self.parse_identifier()?;
            self.skip_ws();
            self.expect_char(':')?;
            self.skip_ws();

            // Value can be string or identifier (for type names like $Tensor)
            let value = if self.peek_char() == Some('"') {
                self.parse_string_literal()?
            } else {
                let mut v = String::new();
                if self.peek_char() == Some('$') {
                    v.push('$');
                    self.advance();
                }
                v.push_str(&self.parse_identifier()?);
                v
            };

            result.insert(key, value);

            self.skip_ws_and_comments();
            if self.peek_char() == Some(',') {
                self.advance();
            }
            self.skip_ws_and_comments();
        }

        self.expect_char('}')?;
        Ok(result)
    }

    fn parse_braced_code(&mut self) -> ParseResult<String> {
        self.expect_char('{')?;
        let start = self.pos;
        let mut depth = 1;

        while depth > 0 && !self.is_eof() {
            match self.peek_char() {
                Some('{') => {
                    depth += 1;
                    self.advance();
                }
                Some('}') => {
                    depth -= 1;
                    if depth > 0 {
                        self.advance();
                    }
                }
                Some('"') => {
                    // Skip string literal
                    self.advance();
                    while let Some(c) = self.peek_char() {
                        if c == '"' {
                            self.advance();
                            break;
                        } else if c == '\\' {
                            self.advance();
                            self.advance();
                        } else {
                            self.advance();
                        }
                    }
                }
                _ => {
                    self.advance();
                }
            }
        }

        let code = self.input[start..self.pos].to_string();
        self.expect_char('}')?;
        Ok(code)
    }

    fn skip_to_next_field(&mut self) -> ParseResult<()> {
        // Skip until we find a comma or closing brace
        let mut depth = 0;
        while !self.is_eof() {
            match self.peek_char() {
                Some('{') | Some('[') | Some('(') => {
                    depth += 1;
                    self.advance();
                }
                Some('}') | Some(']') | Some(')') => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    self.advance();
                }
                Some(',') if depth == 0 => {
                    break;
                }
                Some('"') => {
                    // Skip string
                    let _ = self.parse_string_literal();
                }
                _ => {
                    self.advance();
                }
            }
        }
        Ok(())
    }

    fn error(&self, message: &str) -> ParseError {
        ParseError {
            message: message.to_string(),
            line: self.line,
            column: self.column,
            pos: self.pos,
        }
    }
}

/// Parse a grammar string into GrammarIR
pub fn parse_grammar(input: &str) -> ParseResult<GrammarIR> {
    let mut parser = GrammarParser::new(input);
    parser.parse_grammar()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_rule() {
        let input = r#"
            identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
        "#;

        let grammar = parse_grammar(input).unwrap();
        assert!(grammar.rules.contains_key("identifier"));

        let rule = grammar.rules.get("identifier").unwrap();
        assert_eq!(rule.name, "identifier");
        assert_eq!(rule.modifier, Some(RuleModifier::Atomic));
    }

    #[test]
    fn test_parse_rule_with_binding() {
        let input = r#"
            fn_def = { "fn" ~ name:identifier ~ "(" ~ ")" }
        "#;

        let grammar = parse_grammar(input).unwrap();
        let rule = grammar.rules.get("fn_def").unwrap();

        // Check that the pattern has the binding
        let bindings = GrammarIR::collect_bindings(&rule.pattern);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0], ("name".to_string(), "identifier".to_string()));
    }

    #[test]
    fn test_parse_rule_with_action() {
        let input = r#"
            number = @{ ASCII_DIGIT+ }
              -> TypedExpression::IntLiteral {
                  value: text.parse(),
              }
        "#;

        let grammar = parse_grammar(input).unwrap();
        let rule = grammar.rules.get("number").unwrap();

        assert!(rule.action.is_some());
        assert_eq!(
            rule.return_type,
            Some("TypedExpression::IntLiteral".to_string())
        );
    }

    #[test]
    fn test_parse_language_directive() {
        let input = r#"
            @language {
                name: "TestLang",
                version: "1.0",
            }
        "#;

        let grammar = parse_grammar(input).unwrap();
        assert_eq!(grammar.metadata.name, "TestLang");
        assert_eq!(grammar.metadata.version, "1.0");
    }

    #[test]
    fn test_parse_optional_pattern() {
        let input = r#"
            fn_def = { "fn" ~ name:identifier ~ params:param_list? }
        "#;

        let grammar = parse_grammar(input).unwrap();
        let rule = grammar.rules.get("fn_def").unwrap();

        let bindings = GrammarIR::collect_bindings(&rule.pattern);
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_parse_repeat_pattern() {
        let input = r#"
            items = { item* }
        "#;

        let grammar = parse_grammar(input).unwrap();
        let rule = grammar.rules.get("items").unwrap();

        match &rule.pattern {
            PatternIR::Repeat { min, max, .. } => {
                assert_eq!(*min, 0);
                assert!(max.is_none());
            }
            _ => panic!("Expected Repeat pattern"),
        }
    }

    #[test]
    fn test_parse_nested_struct_in_list() {
        let input = r#"
@language { name: "Test", version: "1.0" }

test_rule = { "test" }
  -> Foo::Bar {
      args: [X::Y { f: v }],
  }
"#;

        match parse_grammar(input) {
            Ok(grammar) => {
                println!("OK: {} rules", grammar.rules.len());
                let rule = grammar.rules.get("test_rule").unwrap();
                println!("Rule action: {:?}", rule.action);
            }
            Err(e) => {
                panic!("Failed to parse: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parse_box_new_call() {
        let input = r#"
@language { name: "Test", version: "1.0" }

test_rule = { "test" }
  -> Foo::Bar {
      callee: Box::new(X::Y { name: n }),
  }
"#;

        match parse_grammar(input) {
            Ok(grammar) => {
                println!("OK: {} rules", grammar.rules.len());
                let rule = grammar.rules.get("test_rule").unwrap();
                println!("Rule action: {:?}", rule.action);
            }
            Err(e) => {
                panic!("Failed to parse: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parse_nested_array_in_struct() {
        // Simplified test case - struct in array in struct
        let input = r#"
@language { name: "Test", version: "1.0" }

test = { "test" }
  -> A::B {
      outer: C::D {
          inner: [E::F { x: v }],
      },
  }
"#;

        match parse_grammar(input) {
            Ok(grammar) => {
                println!("OK: {} rules", grammar.rules.len());
                let rule = grammar.rules.get("test").unwrap();
                println!("Rule action: {:?}", rule.action);
            }
            Err(e) => {
                panic!("Failed to parse: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parse_array_with_function_call() {
        // Test array containing struct with function call (no trailing comma)
        let input = r#"
@language { name: "Test", version: "1.0" }

test = { "test" }
  -> A::B {
      args: [
          C::D { x: intern(v) }
      ],
  }
"#;

        match parse_grammar(input) {
            Ok(grammar) => {
                println!("OK: {} rules", grammar.rules.len());
                let rule = grammar.rules.get("test").unwrap();
                println!("Rule action: {:?}", rule.action);
            }
            Err(e) => {
                panic!("Failed to parse: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parse_array_with_trailing_comma() {
        // Test array with trailing comma
        let input = r#"
@language { name: "Test", version: "1.0" }

test = { "test" }
  -> A::B {
      args: [a, b,],
  }
"#;

        match parse_grammar(input) {
            Ok(grammar) => {
                println!("OK: {} rules", grammar.rules.len());
                let rule = grammar.rules.get("test").unwrap();
                println!("Rule action: {:?}", rule.action);
            }
            Err(e) => {
                panic!("Failed to parse trailing comma: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parse_imagepipe_grammar() {
        let input = include_str!("../../../../examples/imagepipe/imagepipe.zyn");

        match parse_grammar(input) {
            Ok(grammar) => {
                println!("Successfully parsed ImagePipe grammar!");
                println!(
                    "Language: {} v{}",
                    grammar.metadata.name, grammar.metadata.version
                );
                println!("Rules: {}", grammar.rules.len());
                for (name, _rule) in &grammar.rules {
                    println!("  - {}", name);
                }

                // Verify key metadata
                assert_eq!(grammar.metadata.name, "ImagePipe");
                assert_eq!(grammar.metadata.version, "2.0");

                // Verify key rules exist
                assert!(grammar.rules.contains_key("program"));
                assert!(grammar.rules.contains_key("statement"));
                assert!(grammar.rules.contains_key("load_stmt"));
                assert!(grammar.rules.contains_key("save_stmt"));
                assert!(grammar.rules.contains_key("identifier"));
            }
            Err(e) => {
                panic!("Failed to parse ImagePipe grammar: {:?}", e);
            }
        }
    }

    #[test]
    fn test_parse_zynml_grammar() {
        let input = include_str!("../../../zynml/ml.zyn");

        match parse_grammar(input) {
            Ok(grammar) => {
                println!("Successfully parsed ZynML grammar!");
                println!(
                    "Language: {} v{}",
                    grammar.metadata.name, grammar.metadata.version
                );
                println!("Rules: {}", grammar.rules.len());

                // Verify key metadata
                assert_eq!(grammar.metadata.name, "ZynML");
                assert_eq!(grammar.metadata.version, "1.0");

                // Verify key rules exist
                assert!(grammar.rules.contains_key("program"));
                assert!(grammar.rules.contains_key("fn_def"));
                assert!(grammar.rules.contains_key("expr"));
                assert!(grammar.rules.contains_key("statement"));
                assert!(grammar.rules.contains_key("identifier"));

                // Print a few rule names
                let mut rule_names: Vec<_> = grammar.rules.keys().collect();
                rule_names.sort();
                println!("Sample rules:");
                for name in rule_names.iter().take(20) {
                    println!("  - {}", name);
                }
            }
            Err(e) => {
                panic!("Failed to parse ZynML grammar: {:?}", e);
            }
        }
    }
}
