//! Parser Code Generator for ZynPEG 2.0
//!
//! Generates Rust parser methods from GrammarIR. Each grammar rule becomes
//! a `parse_<rule_name>` method that returns `ParseResult<T>`.
//!
//! # Generated Code Structure
//!
//! ```rust,ignore
//! impl<'a> GeneratedParser<'a> {
//!     fn parse_fn_def(&mut self) -> ParseResult<TypedDeclaration> {
//!         // Pattern matching code
//!         // Binding collection
//!         // Action execution
//!     }
//! }
//! ```

use crate::grammar::{CharClass, GrammarIR, PatternIR, RuleIR, RuleModifier};
use std::collections::HashSet;

/// Code generator for parser methods
pub struct ParserGenerator {
    /// Generated code buffer
    code: String,
    /// Indentation level
    indent: usize,
    /// Rule IDs for memoization
    rule_ids: HashSet<String>,
    /// Current rule being generated
    current_rule: Option<String>,
}

impl ParserGenerator {
    pub fn new() -> Self {
        ParserGenerator {
            code: String::new(),
            indent: 0,
            rule_ids: HashSet::new(),
            current_rule: None,
        }
    }

    /// Generate complete parser code from grammar
    pub fn generate(&mut self, grammar: &GrammarIR) -> String {
        self.code.clear();
        self.rule_ids.clear();

        // Collect all rule names for ID generation
        for name in grammar.rule_names() {
            self.rule_ids.insert(name.to_string());
        }

        // Generate header
        self.emit_header(grammar);

        // Generate parser struct
        self.emit_parser_struct(grammar);

        // Generate rule methods
        self.emit_impl_start();
        for (name, rule) in &grammar.rules {
            self.generate_rule(rule);
        }
        self.emit_impl_end();

        self.code.clone()
    }

    fn emit_header(&mut self, grammar: &GrammarIR) {
        self.line("//! Generated parser for ZynPEG 2.0");
        self.line("//!");
        self.line(&format!("//! Language: {}", grammar.metadata.name));
        self.line(&format!("//! Version: {}", grammar.metadata.version));
        self.line("");
        self.line("use crate::runtime2::{ParserState, ParseResult, ParsedValue};");
        self.line("use crate::runtime2::combinator::*;");
        self.line("use crate::runtime2::memo::{MemoKey, MemoEntry, RuleIdGenerator};");
        self.line("use zyntax_typed_ast::{");
        self.line("    TypedASTBuilder, Span, InternedString,");
        self.line("    type_registry::{Type, TypeRegistry, PrimitiveType},");
        self.line("};");
        self.line("");

        // Emit imports from grammar
        for import in &grammar.imports {
            self.line(import);
        }
        self.line("");
    }

    fn emit_parser_struct(&mut self, grammar: &GrammarIR) {
        self.line("/// Generated parser from ZynPEG grammar");
        self.line("pub struct GeneratedParser<'a> {");
        self.indent += 1;
        self.line("state: ParserState<'a>,");
        self.line("rule_ids: RuleIdGenerator,");
        self.indent -= 1;
        self.line("}");
        self.line("");
    }

    fn emit_impl_start(&mut self) {
        self.line("impl<'a> GeneratedParser<'a> {");
        self.indent += 1;

        // Constructor
        self.line("/// Create a new parser");
        self.line("pub fn new(");
        self.indent += 1;
        self.line("input: &'a str,");
        self.line("builder: &'a mut TypedASTBuilder,");
        self.line("type_registry: &'a mut TypeRegistry,");
        self.indent -= 1;
        self.line(") -> Self {");
        self.indent += 1;
        self.line("GeneratedParser {");
        self.indent += 1;
        self.line("state: ParserState::new(input, builder, type_registry),");
        self.line("rule_ids: RuleIdGenerator::new(),");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("");
    }

    fn emit_impl_end(&mut self) {
        self.indent -= 1;
        self.line("}");
    }

    /// Generate code for a single rule
    fn generate_rule(&mut self, rule: &RuleIR) {
        self.current_rule = Some(rule.name.clone());

        // Generate rule method
        let return_type = rule
            .return_type
            .clone()
            .unwrap_or_else(|| "ParsedValue".to_string());

        self.line(&format!("/// Parse rule: {}", rule.name));
        self.line(&format!(
            "pub fn parse_{}(&mut self) -> ParseResult<{}> {{",
            rule.name, return_type
        ));
        self.indent += 1;

        // Memoization check
        self.line(&format!(
            "let rule_id = self.rule_ids.get_id(\"{}\");",
            rule.name
        ));
        self.line("let start_pos = self.state.pos();");
        self.line("");
        self.line("// Check memoization cache");
        self.line("if let Some(entry) = self.state.check_memo(rule_id) {");
        self.indent += 1;
        self.line("return match entry {");
        self.indent += 1;
        self.line("MemoEntry::Success { value, end_pos } => {");
        self.indent += 1;
        self.line("self.state.set_pos(*end_pos);");
        self.line(&format!(
            "ParseResult::Success(value.clone().try_into().unwrap_or_default(), *end_pos)"
        ));
        self.indent -= 1;
        self.line("}");
        self.line("MemoEntry::Failure => self.state.fail(\"memoized failure\"),");
        self.line("MemoEntry::InProgress => self.state.fail(\"left recursion detected\"),");
        self.indent -= 1;
        self.line("};");
        self.indent -= 1;
        self.line("}");
        self.line("");

        // Mark as in progress (for left recursion detection)
        self.line("self.state.store_memo(rule_id, MemoEntry::InProgress);");
        self.line("self.state.clear_bindings();");
        self.line("");

        // Handle atomic rules (no whitespace skipping)
        if rule.modifier != Some(RuleModifier::Atomic) {
            self.line("self.state.skip_ws();");
        }

        // Generate pattern matching code
        self.line("let result = (|| {");
        self.indent += 1;
        self.generate_pattern(&rule.pattern, true);
        self.indent -= 1;
        self.line("})();");
        self.line("");

        // Store result in memo cache
        self.line("match &result {");
        self.indent += 1;
        self.line("ParseResult::Success(value, end_pos) => {");
        self.indent += 1;
        self.line("self.state.store_memo(rule_id, MemoEntry::Success {");
        self.indent += 1;
        self.line("value: ParsedValue::Text(format!(\"{:?}\", value)),"); // TODO: proper conversion
        self.line("end_pos: *end_pos,");
        self.indent -= 1;
        self.line("});");
        self.indent -= 1;
        self.line("}");
        self.line("ParseResult::Failure(_) => {");
        self.indent += 1;
        self.line("self.state.store_memo(rule_id, MemoEntry::Failure);");
        self.line("self.state.set_pos(start_pos);");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("");
        self.line("result");

        self.indent -= 1;
        self.line("}");
        self.line("");

        self.current_rule = None;
    }

    /// Generate pattern matching code
    fn generate_pattern(&mut self, pattern: &PatternIR, is_top_level: bool) {
        match pattern {
            PatternIR::Literal(s) => {
                self.line(&format!("match literal(&mut self.state, {:?}) {{", s));
                self.indent += 1;
                self.line(
                    "ParseResult::Success(_, pos) => ParseResult::Success(ParsedValue::None, pos),",
                );
                self.line("ParseResult::Failure(e) => return ParseResult::Failure(e),");
                self.indent -= 1;
                self.line("}");
            }

            PatternIR::CharClass(class) => {
                self.generate_char_class(class);
            }

            PatternIR::RuleRef { rule_name, binding } => {
                // Check for built-in rules
                let parse_call = match rule_name.as_str() {
                    "ASCII_DIGIT" => "ascii_digit(&mut self.state)".to_string(),
                    "ASCII_ALPHA" => "ascii_alpha(&mut self.state)".to_string(),
                    "ASCII_ALPHANUMERIC" => "ascii_alphanumeric(&mut self.state)".to_string(),
                    "ASCII_HEX_DIGIT" => "ascii_hex_digit(&mut self.state)".to_string(),
                    "NEWLINE" => "newline(&mut self.state)".to_string(),
                    "ANY" => "any_char(&mut self.state)".to_string(),
                    "SOI" => "soi(&mut self.state)".to_string(),
                    "EOI" => "eoi(&mut self.state)".to_string(),
                    _ => format!("self.parse_{}()", rule_name),
                };

                if let Some(bind_name) = binding {
                    self.line(&format!("let {} = match {} {{", bind_name, parse_call));
                    self.indent += 1;
                    self.line("ParseResult::Success(v, pos) => {");
                    self.indent += 1;
                    self.line(&format!("self.state.set_binding(\"{}\", ParsedValue::Node(crate::runtime2::state::NodeHandle(pos)));", bind_name));
                    self.line("v");
                    self.indent -= 1;
                    self.line("}");
                    self.line("ParseResult::Failure(e) => return ParseResult::Failure(e),");
                    self.indent -= 1;
                    self.line("};");
                } else {
                    self.line(&format!("match {} {{", parse_call));
                    self.indent += 1;
                    self.line("ParseResult::Success(_, _) => {}");
                    self.line("ParseResult::Failure(e) => return ParseResult::Failure(e),");
                    self.indent -= 1;
                    self.line("}");
                }
            }

            PatternIR::Sequence(patterns) => {
                for (i, p) in patterns.iter().enumerate() {
                    if i > 0 {
                        // Skip whitespace between elements (unless atomic)
                        self.line("self.state.skip_ws();");
                    }
                    self.generate_pattern(p, false);
                }
                if is_top_level {
                    self.line("ParseResult::Success(ParsedValue::None, self.state.pos())");
                }
            }

            PatternIR::Choice(choices) => {
                self.line("let choice_start = self.state.pos();");
                self.line("let saved_bindings = self.state.save_bindings();");

                for (i, choice) in choices.iter().enumerate() {
                    if i == 0 {
                        self.line("let choice_result = (|| {");
                    } else {
                        self.line("}).or_else(|_| {");
                        self.indent += 1;
                        self.line("self.state.set_pos(choice_start);");
                        self.line("self.state.restore_bindings(saved_bindings.clone());");
                    }
                    self.indent += 1;
                    self.generate_pattern(choice, true);
                    self.indent -= 1;
                }
                self.line("})();");
                self.line("choice_result");
            }

            PatternIR::Optional(inner) => {
                self.line("{");
                self.indent += 1;
                self.line("let opt_start = self.state.pos();");
                self.line("let saved = self.state.save_bindings();");
                self.line("match (|| {");
                self.indent += 1;
                self.generate_pattern(inner, true);
                self.indent -= 1;
                self.line("})() {");
                self.indent += 1;
                self.line("ParseResult::Success(v, pos) => ParseResult::Success(ParsedValue::Optional(Some(Box::new(v.into()))), pos),");
                self.line("ParseResult::Failure(_) => {");
                self.indent += 1;
                self.line("self.state.set_pos(opt_start);");
                self.line("self.state.restore_bindings(saved);");
                self.line("ParseResult::Success(ParsedValue::Optional(None), opt_start)");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
            }

            PatternIR::Repeat {
                pattern,
                min,
                max,
                separator,
            } => {
                self.line("{");
                self.indent += 1;
                self.line("let mut items = Vec::new();");
                self.line("loop {");
                self.indent += 1;
                self.line("let item_start = self.state.pos();");
                self.line("let saved = self.state.save_bindings();");

                // Handle separator
                if separator.is_some() && *min > 0 {
                    self.line("if !items.is_empty() {");
                    self.indent += 1;
                    self.line("self.state.skip_ws();");
                    // Generate separator pattern
                    if let Some(sep) = separator {
                        self.line("match (|| {");
                        self.indent += 1;
                        self.generate_pattern(sep, true);
                        self.indent -= 1;
                        self.line("})() {");
                        self.indent += 1;
                        self.line("ParseResult::Success(_, _) => {}");
                        self.line("ParseResult::Failure(_) => {");
                        self.indent += 1;
                        self.line("self.state.set_pos(item_start);");
                        self.line("self.state.restore_bindings(saved);");
                        self.line("break;");
                        self.indent -= 1;
                        self.line("}");
                        self.indent -= 1;
                        self.line("}");
                    }
                    self.line("self.state.skip_ws();");
                    self.indent -= 1;
                    self.line("}");
                }

                self.line("match (|| {");
                self.indent += 1;
                self.generate_pattern(pattern, true);
                self.indent -= 1;
                self.line("})() {");
                self.indent += 1;
                self.line("ParseResult::Success(v, _) => items.push(v),");
                self.line("ParseResult::Failure(_) => {");
                self.indent += 1;
                self.line("self.state.set_pos(item_start);");
                self.line("self.state.restore_bindings(saved);");
                self.line("break;");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");

                // Check max
                if let Some(max_count) = max {
                    self.line(&format!("if items.len() >= {} {{ break; }}", max_count));
                }

                self.indent -= 1;
                self.line("}");

                // Check min
                if *min > 0 {
                    self.line(&format!("if items.len() < {} {{", min));
                    self.indent += 1;
                    self.line(&format!(
                        "return self.state.fail(\"expected at least {} items\");",
                        min
                    ));
                    self.indent -= 1;
                    self.line("}");
                }

                self.line("ParseResult::Success(ParsedValue::List(items), self.state.pos())");
                self.indent -= 1;
                self.line("}");
            }

            PatternIR::PositiveLookahead(inner) => {
                self.line("{");
                self.indent += 1;
                self.line("let la_start = self.state.pos();");
                self.line("let saved = self.state.save_bindings();");
                self.line("let result = (|| {");
                self.indent += 1;
                self.generate_pattern(inner, true);
                self.indent -= 1;
                self.line("})();");
                self.line("self.state.set_pos(la_start);");
                self.line("self.state.restore_bindings(saved);");
                self.line("match result {");
                self.indent += 1;
                self.line("ParseResult::Success(_, _) => ParseResult::Success(ParsedValue::None, la_start),");
                self.line("ParseResult::Failure(e) => ParseResult::Failure(e),");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
            }

            PatternIR::NegativeLookahead(inner) => {
                self.line("{");
                self.indent += 1;
                self.line("let la_start = self.state.pos();");
                self.line("let saved = self.state.save_bindings();");
                self.line("let result = (|| {");
                self.indent += 1;
                self.generate_pattern(inner, true);
                self.indent -= 1;
                self.line("})();");
                self.line("self.state.set_pos(la_start);");
                self.line("self.state.restore_bindings(saved);");
                self.line("match result {");
                self.indent += 1;
                self.line("ParseResult::Success(_, _) => self.state.fail(\"negative lookahead matched\"),");
                self.line(
                    "ParseResult::Failure(_) => ParseResult::Success(ParsedValue::None, la_start),",
                );
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
            }

            PatternIR::Any => {
                self.line("any_char(&mut self.state).map(|c| ParsedValue::Text(c.to_string()))");
            }

            PatternIR::StartOfInput => {
                self.line("soi(&mut self.state).map(|_| ParsedValue::None)");
            }

            PatternIR::EndOfInput => {
                self.line("eoi(&mut self.state).map(|_| ParsedValue::None)");
            }

            PatternIR::Whitespace => {
                self.line("self.state.skip_ws();");
                self.line("ParseResult::Success(ParsedValue::None, self.state.pos())");
            }
        }
    }

    /// Generate character class matching code
    fn generate_char_class(&mut self, class: &CharClass) {
        match class {
            CharClass::Single(c) => {
                self.line(&format!(
                    "char_exact(&mut self.state, {:?}).map(|c| ParsedValue::Text(c.to_string()))",
                    c
                ));
            }
            CharClass::Range(start, end) => {
                self.line(&format!("char_range(&mut self.state, {:?}, {:?}).map(|c| ParsedValue::Text(c.to_string()))", start, end));
            }
            CharClass::Builtin(name) => {
                let func = match name.as_str() {
                    "ASCII_DIGIT" => "ascii_digit",
                    "ASCII_ALPHA" => "ascii_alpha",
                    "ASCII_ALPHANUMERIC" => "ascii_alphanumeric",
                    "ASCII_HEX_DIGIT" => "ascii_hex_digit",
                    "NEWLINE" => "newline",
                    _ => "any_char",
                };
                if func == "newline" {
                    self.line(&format!(
                        "{}(&mut self.state).map(|_| ParsedValue::None)",
                        func
                    ));
                } else {
                    self.line(&format!(
                        "{}(&mut self.state).map(|c| ParsedValue::Text(c.to_string()))",
                        func
                    ));
                }
            }
            CharClass::Union(classes) => {
                self.line("{");
                self.indent += 1;
                self.line("let union_start = self.state.pos();");
                for (i, c) in classes.iter().enumerate() {
                    if i == 0 {
                        self.line("let union_result = (|| {");
                    } else {
                        self.line("}).or_else(|_: ParsedValue| {");
                        self.indent += 1;
                        self.line("self.state.set_pos(union_start);");
                    }
                    self.indent += 1;
                    self.generate_char_class(c);
                    self.indent -= 1;
                }
                self.line("})();");
                self.line("union_result");
                self.indent -= 1;
                self.line("}");
            }
            CharClass::Negation(inner) => {
                self.line("{");
                self.indent += 1;
                self.line("let neg_start = self.state.pos();");
                self.line("let result = (|| {");
                self.indent += 1;
                self.generate_char_class(inner);
                self.indent -= 1;
                self.line("})();");
                self.line("match result {");
                self.indent += 1;
                self.line("ParseResult::Success(_, _) => {");
                self.indent += 1;
                self.line("self.state.set_pos(neg_start);");
                self.line("self.state.fail(\"negated class matched\")");
                self.indent -= 1;
                self.line("}");
                self.line("ParseResult::Failure(_) => {");
                self.indent += 1;
                self.line("self.state.set_pos(neg_start);");
                self.line("any_char(&mut self.state).map(|c| ParsedValue::Text(c.to_string()))");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");
            }
        }
    }

    // Helper methods for code generation

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.code.push_str("    ");
        }
        self.code.push_str(text);
        self.code.push('\n');
    }
}

impl Default for ParserGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parser::parse_grammar;

    #[test]
    fn test_generate_simple_rule() {
        let input = r#"
            @language { name: "Test", version: "1.0" }
            identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
        "#;

        let grammar = parse_grammar(input).unwrap();
        let mut gen = ParserGenerator::new();
        let code = gen.generate(&grammar);

        assert!(code.contains("fn parse_identifier"));
        assert!(code.contains("ascii_alpha"));
    }

    #[test]
    fn test_generate_rule_with_binding() {
        let input = r#"
            @language { name: "Test", version: "1.0" }
            fn_def = { "fn" ~ name:identifier }
            identifier = @{ ASCII_ALPHA+ }
        "#;

        let grammar = parse_grammar(input).unwrap();
        let mut gen = ParserGenerator::new();
        let code = gen.generate(&grammar);

        assert!(code.contains("fn parse_fn_def"));
        assert!(code.contains("let name ="));
    }
}
