//! AST types for parsed .zyn grammar files

use crate::{
    ActionBlock, ActionField, BuiltinMappings, ContextVar, Imports, LanguageInfo, Rule, RuleDef,
    RuleModifier, TypeDeclarations, TypeHelpers, ZynGrammar,
};
use pest::iterators::Pair;

/// Build a ZynGrammar from parsed pest pairs
pub fn build_grammar(pairs: pest::iterators::Pairs<Rule>) -> Result<ZynGrammar, String> {
    let mut grammar = ZynGrammar::default();

    for pair in pairs {
        match pair.as_rule() {
            Rule::program => {
                for inner in pair.into_inner() {
                    process_top_level(&mut grammar, inner)?;
                }
            }
            Rule::directive => {
                process_directive(&mut grammar, pair)?;
            }
            Rule::rule_def => {
                grammar.rules.push(build_rule_def(pair)?);
            }
            Rule::EOI => {}
            _ => {
                process_top_level(&mut grammar, pair)?;
            }
        }
    }

    Ok(grammar)
}

fn process_top_level(grammar: &mut ZynGrammar, pair: Pair<Rule>) -> Result<(), String> {
    match pair.as_rule() {
        Rule::directive => process_directive(grammar, pair)?,
        Rule::rule_def => grammar.rules.push(build_rule_def(pair)?),
        Rule::language_directive => grammar.language = build_language_info(pair)?,
        Rule::imports_directive => grammar.imports = build_imports(pair)?,
        Rule::context_directive => grammar.context = build_context(pair)?,
        Rule::type_helpers_directive => grammar.type_helpers = build_type_helpers(pair)?,
        Rule::builtin_directive => grammar.builtins = build_builtins(pair)?,
        Rule::types_directive => grammar.types = build_types(pair)?,
        Rule::EOI => {}
        _ => {}
    }
    Ok(())
}

fn process_directive(grammar: &mut ZynGrammar, pair: Pair<Rule>) -> Result<(), String> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::language_directive => grammar.language = build_language_info(inner)?,
            Rule::imports_directive => grammar.imports = build_imports(inner)?,
            Rule::context_directive => grammar.context = build_context(inner)?,
            Rule::type_helpers_directive => grammar.type_helpers = build_type_helpers(inner)?,
            Rule::builtin_directive => grammar.builtins = build_builtins(inner)?,
            Rule::types_directive => grammar.types = build_types(inner)?,
            Rule::error_messages_directive => {
                // TODO: Parse error messages
            }
            _ => {}
        }
    }
    Ok(())
}

fn build_language_info(pair: Pair<Rule>) -> Result<LanguageInfo, String> {
    let mut info = LanguageInfo::default();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::language_field {
            // Get the raw text of the field to determine type
            let field_text = inner.as_str();
            let field_content: Vec<_> = inner.into_inner().collect();

            if field_text.trim().starts_with("name") {
                // Find the string literal
                for item in &field_content {
                    if item.as_rule() == Rule::string_literal {
                        info.name = extract_string_value(item);
                        break;
                    }
                }
            } else if field_text.trim().starts_with("version") {
                for item in &field_content {
                    if item.as_rule() == Rule::string_literal {
                        info.version = extract_string_value(item);
                        break;
                    }
                }
            } else if field_text.trim().starts_with("file_extensions") {
                for item in &field_content {
                    if item.as_rule() == Rule::string_literal {
                        info.file_extensions.push(extract_string_value(item));
                    }
                }
            } else if field_text.trim().starts_with("entry_point") {
                for item in &field_content {
                    if item.as_rule() == Rule::string_literal {
                        info.entry_point = Some(extract_string_value(item));
                        break;
                    }
                }
            }
        }
    }

    Ok(info)
}

fn extract_string_value(pair: &Pair<Rule>) -> String {
    let s = pair.as_str();
    // Remove quotes
    if s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn build_imports(pair: Pair<Rule>) -> Result<Imports, String> {
    let mut imports = Imports::default();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::rust_code {
            imports.code = inner.as_str().trim().to_string();
        }
    }

    Ok(imports)
}

fn build_context(pair: Pair<Rule>) -> Result<Vec<ContextVar>, String> {
    let mut context = Vec::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::context_field {
            let mut parts = inner.into_inner();
            if let (Some(name), Some(ty)) = (parts.next(), parts.next()) {
                context.push(ContextVar {
                    name: name.as_str().to_string(),
                    ty: ty.as_str().to_string(),
                });
            }
        }
    }

    Ok(context)
}

fn build_type_helpers(pair: Pair<Rule>) -> Result<TypeHelpers, String> {
    let mut helpers = TypeHelpers::default();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::rust_code {
            helpers.code = inner.as_str().trim().to_string();
        }
    }

    Ok(helpers)
}

/// Parse @builtin { name: "symbol", ... } directive
///
/// Supports three types of mappings:
/// - Functions (default): `println: "$IO$println"` - direct function name mapping
/// - Methods (prefix @): `@sum: "tensor_sum"` - `x.sum()` becomes `tensor_sum(x)`
///   Multiple targets for same method: `@sum: "audio_sum"` adds to the list
/// - Operators (prefix $): `$*: "vec_dot"` - `x * y` becomes `vec_dot(x, y)`
///   Multiple targets for same operator: `$*: "matrix_mul"` adds to the list
fn build_builtins(pair: Pair<Rule>) -> Result<BuiltinMappings, String> {
    let mut builtins = BuiltinMappings::default();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::builtin_def {
            // Parse: identifier : string_literal
            let mut parts = inner.into_inner();
            if let Some(name_pair) = parts.next() {
                let name = name_pair.as_str().to_string();
                if let Some(symbol_pair) = parts.next() {
                    let symbol = extract_string_value(&symbol_pair);

                    // Check for method prefix (@) or operator prefix ($)
                    if let Some(method_name) = name.strip_prefix('@') {
                        // Method mapping: @sum -> tensor_sum means x.sum() -> tensor_sum(x)
                        // Multiple definitions accumulate into a list for type-based dispatch
                        builtins
                            .methods
                            .entry(method_name.to_string())
                            .or_insert_with(Vec::new)
                            .push(symbol);
                    } else if let Some(op) = name.strip_prefix('$') {
                        // Operator mapping: $* -> vec_dot means x * y -> vec_dot(x, y)
                        // Multiple definitions accumulate into a list for type-based dispatch
                        builtins
                            .operators
                            .entry(op.to_string())
                            .or_insert_with(Vec::new)
                            .push(symbol);
                    } else {
                        // Regular function mapping (single target)
                        builtins.functions.insert(name, symbol);
                    }
                }
            }
        }
    }

    Ok(builtins)
}

/// Parse @types { opaque: [$Tensor, $Audio], returns: { tensor: $Tensor } } directive
///
/// Declares:
/// - opaque: List of opaque type names that are pointer types backed by ZRTL plugins
/// - returns: Map of function name -> return type for proper type tracking
fn build_types(pair: Pair<Rule>) -> Result<TypeDeclarations, String> {
    let mut types = TypeDeclarations::default();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::types_def {
            let def_text = inner.as_str().trim();

            if def_text.starts_with("opaque") {
                // Parse opaque type list: opaque: [$Tensor, $Audio]
                for type_inner in inner.into_inner() {
                    if type_inner.as_rule() == Rule::type_name {
                        let type_name = type_inner.as_str().to_string();
                        types.opaque_types.push(type_name);
                    }
                }
            } else if def_text.starts_with("returns") {
                // Parse return type mapping: returns: { tensor: $Tensor, audio_load: $Audio }
                for return_inner in inner.into_inner() {
                    if return_inner.as_rule() == Rule::return_def {
                        let mut parts = return_inner.into_inner();
                        if let (Some(fn_name), Some(type_name)) = (parts.next(), parts.next()) {
                            let fn_name_str = fn_name.as_str().to_string();
                            let type_name_str = type_name.as_str().to_string();
                            types.function_returns.insert(fn_name_str, type_name_str);
                        }
                    }
                }
            }
        }
    }

    Ok(types)
}

fn build_rule_def(pair: Pair<Rule>) -> Result<RuleDef, String> {
    let mut name = String::new();
    let mut modifier = None;
    let mut pattern = String::new();
    let mut action = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::identifier => {
                if name.is_empty() {
                    name = inner.as_str().to_string();
                }
            }
            Rule::rule_modifier => {
                modifier = Some(match inner.as_str() {
                    "@" => RuleModifier::Atomic,
                    "_" => RuleModifier::Silent,
                    "$" => RuleModifier::Compound,
                    "!" => RuleModifier::NonAtomic,
                    _ => RuleModifier::Atomic,
                });
            }
            Rule::pattern => {
                pattern = inner.as_str().to_string();
            }
            Rule::action_block => {
                action = Some(build_action_block(inner)?);
            }
            _ => {}
        }
    }

    Ok(RuleDef {
        name,
        modifier,
        pattern,
        action,
    })
}

fn build_action_block(pair: Pair<Rule>) -> Result<ActionBlock, String> {
    let mut return_type = String::new();
    let mut fields = Vec::new();
    let mut raw_code = None;
    let mut json_commands = None;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::type_path => {
                // type_path = { rust_type ~ ("::" ~ identifier)? }
                // Extract the full type path including variant
                return_type = inner.as_str().trim().to_string();
            }
            Rule::rust_type => {
                // Legacy support: direct rust_type
                return_type = inner.as_str().trim().to_string();
            }
            Rule::identifier => {
                // Helper function call or passthrough: -> intern(binding) or -> binding
                if return_type.is_empty() {
                    return_type = inner.as_str().trim().to_string();
                }
            }
            Rule::action_body => {
                for field_pair in inner.into_inner() {
                    match field_pair.as_rule() {
                        Rule::json_action => {
                            // Build JSON commands string
                            json_commands = Some(build_json_commands(field_pair)?);
                        }
                        Rule::action_field => {
                            fields.push(build_action_field(field_pair)?);
                        }
                        Rule::action_code => {
                            raw_code = Some(field_pair.as_str().trim().to_string());
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(ActionBlock {
        return_type,
        fields,
        raw_code,
        json_commands,
    })
}

/// Build JSON commands string from a json_action parse tree node
fn build_json_commands(pair: Pair<Rule>) -> Result<String, String> {
    // Wrap the fields in a JSON object and return as string
    let mut json_str = String::from("{");
    let mut first = true;

    for field in pair.into_inner() {
        if field.as_rule() == Rule::json_field {
            if !first {
                json_str.push_str(", ");
            }
            first = false;

            let mut field_inner = field.into_inner();
            if let (Some(key), Some(value)) = (field_inner.next(), field_inner.next()) {
                // Key is already quoted string literal
                json_str.push_str(key.as_str());
                json_str.push_str(": ");
                // Value needs to be converted to JSON
                json_str.push_str(&json_value_to_string(value)?);
            }
        }
    }

    json_str.push('}');
    Ok(json_str)
}

/// Convert a json_value parse node to JSON string
fn json_value_to_string(pair: Pair<Rule>) -> Result<String, String> {
    match pair.as_rule() {
        Rule::string_literal | Rule::capture_ref_string => {
            // Already quoted
            Ok(pair.as_str().to_string())
        }
        Rule::json_number | Rule::json_bool | Rule::json_null => Ok(pair.as_str().to_string()),
        Rule::json_array => {
            let mut result = String::from("[");
            let mut first = true;
            for inner in pair.into_inner() {
                if !first {
                    result.push_str(", ");
                }
                first = false;
                result.push_str(&json_value_to_string(inner)?);
            }
            result.push(']');
            Ok(result)
        }
        Rule::json_object => {
            let mut result = String::from("{");
            let mut first = true;
            for field in pair.into_inner() {
                if field.as_rule() == Rule::json_field {
                    if !first {
                        result.push_str(", ");
                    }
                    first = false;

                    let mut field_inner = field.into_inner();
                    if let (Some(key), Some(value)) = (field_inner.next(), field_inner.next()) {
                        result.push_str(key.as_str());
                        result.push_str(": ");
                        result.push_str(&json_value_to_string(value)?);
                    }
                }
            }
            result.push('}');
            Ok(result)
        }
        Rule::json_value => {
            // Recurse into actual value
            if let Some(inner) = pair.into_inner().next() {
                json_value_to_string(inner)
            } else {
                Ok("null".to_string())
            }
        }
        _ => {
            // Fallback: use raw text
            Ok(pair.as_str().to_string())
        }
    }
}

fn build_action_field(pair: Pair<Rule>) -> Result<ActionField, String> {
    let mut parts = pair.into_inner();
    let name = parts
        .next()
        .ok_or("Missing field name")?
        .as_str()
        .to_string();
    let value = parts
        .next()
        .ok_or("Missing field value")?
        .as_str()
        .to_string();

    Ok(ActionField { name, value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ZynGrammarParser;
    use pest::Parser;

    #[test]
    fn test_build_language_info() {
        let input = r#"@language {
            name: "Calculator",
            version: "1.0",
        }"#;

        let pairs = ZynGrammarParser::parse(Rule::language_directive, input).unwrap();
        let info = build_language_info(pairs.into_iter().next().unwrap()).unwrap();

        assert_eq!(info.name, "Calculator");
        assert_eq!(info.version, "1.0");
    }

    #[test]
    fn test_build_rule_def() {
        let input = r#"number = @{ ASCII_DIGIT+ }"#;

        let pairs = ZynGrammarParser::parse(Rule::rule_def, input).unwrap();
        let rule = build_rule_def(pairs.into_iter().next().unwrap()).unwrap();

        assert_eq!(rule.name, "number");
        assert_eq!(rule.modifier, Some(RuleModifier::Atomic));
        assert!(rule.pattern.contains("ASCII_DIGIT"));
    }

    #[test]
    fn test_build_rule_with_action() {
        let input = r#"number = @{ ASCII_DIGIT+ }
          -> TypedExpression {
              expr: IntLiteral(parse_int($1)),
              ty: Type::I32,
          }"#;

        let pairs = ZynGrammarParser::parse(Rule::rule_def, input).unwrap();
        let rule = build_rule_def(pairs.into_iter().next().unwrap()).unwrap();

        assert_eq!(rule.name, "number");
        assert!(rule.action.is_some());

        let action = rule.action.unwrap();
        assert_eq!(action.return_type, "TypedExpression");
        // Grammar parses "field: value" syntax as action_field entries
        assert_eq!(action.fields.len(), 2);
        assert_eq!(action.fields[0].name, "expr");
        assert!(action.fields[0].value.contains("IntLiteral"));
        assert_eq!(action.fields[1].name, "ty");
        assert!(action.fields[1].value.contains("Type::I32"));
    }

    #[test]
    fn test_nested_braces_in_action() {
        let input = r#"test = { "test" }
  -> TestType {
      decl: ConstDecl {
          name: intern($2),
          ty: Type::I32,
      },
      visibility: Visibility::Private,
  }"#;

        let result = ZynGrammarParser::parse(Rule::rule_def, input);
        match result {
            Ok(pairs) => {
                let rule = build_rule_def(pairs.into_iter().next().unwrap()).unwrap();
                assert_eq!(rule.name, "test");
                assert!(rule.action.is_some());
                let action = rule.action.unwrap();
                assert_eq!(action.return_type, "TestType");
                // Grammar parses "field: value" syntax as action_field entries
                // even with nested braces in the value
                assert_eq!(action.fields.len(), 2);
                assert_eq!(action.fields[0].name, "decl");
                assert!(action.fields[0].value.contains("ConstDecl"));
                assert!(action.fields[0].value.contains("intern($2)"));
                assert_eq!(action.fields[1].name, "visibility");
                assert!(action.fields[1].value.contains("Visibility::Private"));
            }
            Err(e) => {
                panic!("Failed to parse: {}", e);
            }
        }
    }
}
