//! Runtime Interpreter for ZynPEG 2.0
//!
//! Executes GrammarIR patterns directly without code generation.
//! This is useful for:
//! - Testing grammars
//! - Embedded DSL use cases
//! - Rapid prototyping
//!
//! For production use, the code generator (`codegen::ParserGenerator`) should
//! be used instead as it produces more efficient compiled code.

use crate::grammar::{GrammarIR, RuleIR, PatternIR, ActionIR, ExprIR, CharClass, RuleModifier};
use super::state::{ParserState, ParseResult, ParsedValue, ParseFailure};
use std::collections::HashMap;
use log::{debug, trace};
use zyntax_typed_ast::{
    TypedNode, TypedStatement, TypedExpression, TypedLiteral, TypedBlock,
    TypedDeclaration, TypedFunction, TypedLet, TypedLetPattern, TypedCall, TypedMethodCall, TypedProgram,
    TypedIf, TypedWhile, TypedFor, TypedMatch, TypedMatchArm, TypedUnary, TypedFieldAccess, TypedIndex,
    TypedRange, TypedStructLiteral, TypedFieldInit, TypedPattern,
    TypedParameter, TypedVariant, TypedVariantFields, TypedTypeAlias, ParameterKind,
    TypedInterface, TypedExtern, TypedExternStruct, TypedTypeParam,
    TypedAnnotation, TypedAnnotationArg, TypedAnnotationValue,
    TypedLambda, TypedLambdaBody, TypedLambdaParam, TypedImportModifier, TypedPath,
    UnaryOp,
    typed_node, Span,
    type_registry::{Type, PrimitiveType, Mutability, Visibility, CallingConvention, NullabilityKind, ConstValue},
};

/// Runtime interpreter for GrammarIR
pub struct GrammarInterpreter<'g> {
    grammar: &'g GrammarIR,
    /// Rule ID counter for memoization
    rule_id_map: HashMap<String, usize>,
}

impl<'g> GrammarInterpreter<'g> {
    /// Create a new interpreter for the given grammar
    pub fn new(grammar: &'g GrammarIR) -> Self {
        let mut rule_id_map = HashMap::new();
        for (i, name) in grammar.rules.keys().enumerate() {
            rule_id_map.insert(name.clone(), i);
        }
        GrammarInterpreter { grammar, rule_id_map }
    }

    /// Parse input using a specific rule
    pub fn parse_rule<'a>(
        &self,
        rule_name: &str,
        state: &mut ParserState<'a>,
    ) -> ParseResult<ParsedValue> {
        let rule = match self.grammar.get_rule(rule_name) {
            Some(r) => r,
            None => return state.fail(&format!("unknown rule: {}", rule_name)),
        };

        self.execute_rule(rule, state)
    }

    /// Parse input from the entry rule
    pub fn parse<'a>(&self, state: &mut ParserState<'a>) -> ParseResult<ParsedValue> {
        let entry = &self.grammar.entry_rule;
        self.parse_rule(entry, state)
    }

    /// Execute a rule
    fn execute_rule<'a>(
        &self,
        rule: &RuleIR,
        state: &mut ParserState<'a>,
    ) -> ParseResult<ParsedValue> {
        let start_pos = state.pos();
        trace!("execute_rule: {} at pos {}", rule.name, start_pos);

        // Note: We don't skip whitespace here at the start of a rule.
        // Whitespace skipping happens between sequence elements (see execute_pattern).
        // This preserves the position for SOI matching.

        // Save parent bindings to prevent inner rules from overwriting them
        // Each rule has its own binding scope
        let saved_bindings = state.save_bindings();
        state.clear_bindings();

        // Execute pattern
        let result = self.execute_pattern(&rule.pattern, state, rule.modifier == Some(RuleModifier::Atomic));
        trace!("execute_rule: {} at pos {} -> {:?}", rule.name, start_pos,
               match &result { ParseResult::Success(_, p) => format!("ok@{}", p), ParseResult::Failure(f) => format!("fail: {:?}", f) });

        let final_result = match result {
            ParseResult::Success(value, pos) => {
                // For atomic rules, capture the text and bind it so text() can access it
                let parsed_value = if rule.modifier == Some(RuleModifier::Atomic) {
                    let text = state.slice(start_pos, pos).to_string();
                    // Bind the captured text to a special name for text() helper
                    state.set_binding("__text__", ParsedValue::Text(text.clone()));
                    ParsedValue::Text(text)
                } else {
                    value
                };

                // Execute action if present
                if let Some(ref action) = rule.action {
                    let span = Span::new(start_pos, pos);
                    match self.execute_action(action, state, span) {
                        Ok(result) => ParseResult::Success(result, pos),
                        Err(e) => {
                            trace!("execute_rule: {} action FAILED: {}", rule.name, e);
                            state.fail(&e)
                        }
                    }
                } else {
                    ParseResult::Success(parsed_value, pos)
                }
            }
            ParseResult::Failure(e) => {
                state.set_pos(start_pos);
                ParseResult::Failure(e)
            }
        };

        // Restore parent bindings after rule completes
        state.restore_bindings(saved_bindings);

        final_result
    }

    /// Execute an action to construct a TypedAST node
    fn execute_action<'a>(
        &self,
        action: &ActionIR,
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        match action {
            ActionIR::PassThrough { binding } => {
                // Simply return the bound value
                state.get_binding(binding)
                    .cloned()
                    .ok_or_else(|| format!("binding '{}' not found", binding))
            }

            ActionIR::Construct { type_path, fields } => {
                self.execute_construct(type_path, fields, state, span)
            }

            ActionIR::HelperCall { function, args } => {
                self.execute_helper_call(function, args, state, span)
            }

            ActionIR::Match { binding, cases } => {
                let value = state.get_binding(binding)
                    .ok_or_else(|| format!("binding '{}' not found", binding))?;

                let text = match value {
                    ParsedValue::Text(s) => s.clone(),
                    _ => return Err(format!("match binding '{}' is not text", binding)),
                };

                for (pattern, action) in cases {
                    if text == *pattern {
                        return self.execute_action(action, state, span);
                    }
                }
                Err(format!("no match case for '{}'", text))
            }

            ActionIR::Conditional { condition, then_action, else_action } => {
                let cond_val = self.eval_expr(condition, state)?;
                let is_true = match cond_val {
                    ParsedValue::Bool(b) => b,
                    ParsedValue::Optional(opt) => opt.is_some(),
                    _ => return Err("condition must evaluate to bool".to_string()),
                };

                if is_true {
                    self.execute_action(then_action, state, span)
                } else if let Some(else_act) = else_action {
                    self.execute_action(else_act, state, span)
                } else {
                    Ok(ParsedValue::None)
                }
            }

            ActionIR::LegacyJson { return_type, json_content } => {
                // Legacy JSON actions are not executed - they require codegen
                Err(format!(
                    "legacy JSON action for '{}' requires code generation, not runtime interpretation",
                    return_type
                ))
            }
        }
    }

    /// Execute a Construct action to build a TypedAST node
    fn execute_construct<'a>(
        &self,
        type_path: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        // Parse the type path (e.g., "TypedStatement::Let", "TypedExpression::Call")
        let parts: Vec<&str> = type_path.split("::").collect();

        match parts.as_slice() {
            ["TypedStatement", variant] => {
                self.construct_statement(variant, fields, state, span)
            }
            ["TypedExpression", variant] => {
                self.construct_expression(variant, fields, state, span)
            }
            ["TypedDeclaration", variant] => {
                self.construct_declaration(variant, fields, state, span)
            }
            ["TypedProgram"] => {
                self.construct_program(fields, state, span)
            }
            ["TypedBlock"] => {
                self.construct_block(fields, state, span)
            }
            ["Type", variant] => {
                self.construct_type(variant, fields, state, span)
            }
            ["Box", "new"] | ["Box"] => {
                // Box::new(expr) - just return the inner expression
                // Fields should have a single entry for the inner value
                if let Some((_, expr)) = fields.first() {
                    self.eval_expr(expr, state)
                } else {
                    Ok(ParsedValue::None)
                }
            }
            ["Some"] => {
                // Some(value) - wrap in Optional
                if let Some((_, expr)) = fields.first() {
                    let inner = self.eval_expr(expr, state)?;
                    Ok(ParsedValue::Optional(Some(Box::new(inner))))
                } else {
                    Ok(ParsedValue::Optional(None))
                }
            }
            ["TypedLiteral", variant] => {
                self.construct_literal(variant, fields, state)
            }
            ["TypedField"] => {
                self.construct_field(fields, state, span)
            }
            ["TypedVariant"] => {
                self.construct_variant(fields, state, span)
            }
            ["TypedParameter"] => {
                self.construct_parameter(fields, state, span)
            }
            ["TypedFieldInit"] => {
                self.construct_field_init(fields, state, span)
            }
            ["TypedMatchArm"] => {
                self.construct_match_arm(fields, state, span)
            }
            ["TypedPattern", variant] => {
                self.construct_pattern(variant, fields, state, span)
            }
            ["TypedAnnotation"] => {
                self.construct_annotation(fields, state, span)
            }
            ["TypedAnnotationArg", variant] => {
                self.construct_annotation_arg(variant, fields, state, span)
            }
            ["TypedAnnotationValue", variant] => {
                self.construct_annotation_value(variant, fields, state, span)
            }
            // Effect types
            ["TypedEffectOp"] => {
                self.construct_effect_op(fields, state, span)
            }
            ["TypedEffectHandlerImpl"] => {
                self.construct_effect_handler_impl(fields, state, span)
            }
            // Postfix suffix types for fold operations
            ["SuffixField"] | ["SuffixMethod"] | ["SuffixCall"] | ["SuffixIndex"] | ["SuffixSlice"] => {
                self.construct_suffix(type_path, fields, state)
            }
            // Lambda parameter
            ["TypedLambdaParam"] => {
                self.construct_lambda_param(fields, state, span)
            }
            _ => Err(format!("unknown type path: {}", type_path)),
        }
    }

    /// Construct a postfix suffix for fold operations
    fn construct_suffix<'a>(
        &self,
        type_path: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<ParsedValue, String> {
        let mut suffix_fields = std::collections::HashMap::new();
        for (name, expr) in fields {
            let value = self.eval_expr(expr, state)?;
            suffix_fields.insert(name.clone(), Box::new(value));
        }
        Ok(ParsedValue::Suffix {
            kind: type_path.to_string(),
            fields: suffix_fields,
        })
    }

    /// Construct a TypedBlock
    fn construct_block<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let statements = if let Some(expr) = self.get_field("statements", fields) {
            let val = self.eval_expr(expr, state)?;
            match val {
                ParsedValue::List(items) => {
                    items.into_iter()
                        .map(|item| self.parsed_value_to_stmt(item))
                        .collect::<Result<Vec<_>, _>>()?
                }
                ParsedValue::Statement(s) => vec![*s],
                ParsedValue::None => vec![],
                _ => vec![],
            }
        } else {
            vec![]
        };

        Ok(ParsedValue::Block(TypedBlock { statements, span }))
    }

    /// Construct a TypedStatement variant
    fn construct_statement<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let stmt = match variant {
            "Let" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let type_annotation = self.get_field_optional("type_annotation", fields, state)?;
                let initializer = self.get_field_optional_expr("initializer", fields, state)?;
                let is_mutable = self.get_field_as_bool("is_mutable", fields, state).unwrap_or(false);

                // Use Type::Any when no type annotation is provided - let the compiler infer from initializer
                let ty = type_annotation.unwrap_or(Type::Any);
                let mutability = if is_mutable { Mutability::Mutable } else { Mutability::Immutable };

                TypedStatement::Let(TypedLet {
                    name,
                    ty,
                    mutability,
                    initializer: initializer.map(Box::new),
                    span,
                })
            }
            "Expression" => {
                let expr = self.get_field_as_expr("expr", fields, state)?;
                TypedStatement::Expression(Box::new(expr))
            }
            "Assignment" => {
                // Assignment is typically represented as an Expression with Binary Assign
                let target = self.get_field_as_expr("target", fields, state)?;
                let value = self.get_field_as_expr("value", fields, state)?;

                let assign_expr = typed_node(
                    TypedExpression::Binary(zyntax_typed_ast::TypedBinary {
                        op: zyntax_typed_ast::BinaryOp::Assign,
                        left: Box::new(target),
                        right: Box::new(value),
                    }),
                    Type::Primitive(PrimitiveType::Unit),
                    span,
                );
                TypedStatement::Expression(Box::new(assign_expr))
            }
            "Return" => {
                let value = self.get_field_optional_expr("value", fields, state)?;
                TypedStatement::Return(value.map(Box::new))
            }
            "If" => {
                let condition = self.get_field_as_expr("condition", fields, state)?;
                let then_block = self.get_field_as_block("then_branch", fields, state)?;
                let else_block = self.get_field_optional_block("else_branch", fields, state)?;

                TypedStatement::If(TypedIf {
                    condition: Box::new(condition),
                    then_block,
                    else_block,
                    span,
                })
            }
            "While" => {
                let condition = self.get_field_as_expr("condition", fields, state)?;
                let body = self.get_field_as_block("body", fields, state)?;

                TypedStatement::While(TypedWhile {
                    condition: Box::new(condition),
                    body,
                    span,
                })
            }
            "For" => {
                let variable = self.get_field_as_interned("variable", fields, state)?;
                let iterable = self.get_field_as_expr("iterable", fields, state)?;
                let body = self.get_field_as_block("body", fields, state)?;

                // Create a binding pattern for the variable
                let pattern = typed_node(
                    TypedPattern::Identifier { name: variable, mutability: Mutability::Immutable },
                    Type::Any,
                    span,
                );

                TypedStatement::For(TypedFor {
                    pattern: Box::new(pattern),
                    iterator: Box::new(iterable),
                    body,
                })
            }
            "Break" => {
                let value = self.get_field_optional_expr("value", fields, state)?;
                TypedStatement::Break(value.map(Box::new))
            }
            "Continue" => {
                TypedStatement::Continue
            }
            "LetPattern" => {
                let pattern = self.get_field_as_pattern("pattern", fields, state)?;
                let initializer = self.get_field_as_expr("initializer", fields, state)?;

                TypedStatement::LetPattern(TypedLetPattern {
                    pattern: Box::new(pattern),
                    initializer: Box::new(initializer),
                    span,
                })
            }
            "Match" => {
                let scrutinee = self.get_field_as_expr("scrutinee", fields, state)?;
                let arms = self.get_field_as_match_arm_list("arms", fields, state)?;

                TypedStatement::Match(TypedMatch {
                    scrutinee: Box::new(scrutinee),
                    arms,
                })
            }
            _ => return Err(format!("unknown TypedStatement variant: {}", variant)),
        };

        Ok(ParsedValue::Statement(Box::new(typed_node(
            stmt,
            Type::Primitive(PrimitiveType::Unit),
            span,
        ))))
    }

    /// Construct a TypedExpression variant
    fn construct_expression<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let expr = match variant {
            "IntLiteral" => {
                let value = self.get_field_as_int("value", fields, state)?;
                TypedExpression::Literal(TypedLiteral::Integer(value as i128))
            }
            "FloatLiteral" => {
                let value = self.get_field_as_float("value", fields, state)?;
                TypedExpression::Literal(TypedLiteral::Float(value))
            }
            "StringLiteral" => {
                let value = self.get_field_as_string("value", fields, state)?;
                // Strip quotes from string literal
                let unquoted = value.trim_matches('"').to_string();
                let interned = state.intern(&unquoted);
                TypedExpression::Literal(TypedLiteral::String(interned))
            }
            "Variable" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                TypedExpression::Variable(name)
            }
            "Call" => {
                let callee = self.get_field_as_expr("callee", fields, state)?;
                let args = self.get_field_as_expr_list("args", fields, state)?;

                TypedExpression::Call(TypedCall {
                    callee: Box::new(callee),
                    positional_args: args,
                    named_args: vec![],
                    type_args: vec![],
                })
            }
            "Binary" => {
                let left = self.get_field_as_expr("left", fields, state)?;
                let right = self.get_field_as_expr("right", fields, state)?;
                let op = self.get_field_as_string("op", fields, state)?;

                let binary_op = self.string_to_binary_op(&op)?;
                TypedExpression::Binary(zyntax_typed_ast::TypedBinary {
                    op: binary_op,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            "Unary" => {
                let operand = self.get_field_as_expr("operand", fields, state)?;
                let op = self.get_field_as_string("op", fields, state)?;

                let unary_op = self.string_to_unary_op(&op)?;
                TypedExpression::Unary(TypedUnary {
                    op: unary_op,
                    operand: Box::new(operand),
                })
            }
            "Array" => {
                let elements = self.get_field_as_expr_list("elements", fields, state)?;
                TypedExpression::Array(elements)
            }
            "Index" => {
                let object = self.get_field_as_expr("object", fields, state)?;
                let index = self.get_field_as_expr("index", fields, state)?;

                TypedExpression::Index(TypedIndex {
                    object: Box::new(object),
                    index: Box::new(index),
                })
            }
            "Field" | "FieldAccess" => {
                let object = self.get_field_as_expr("object", fields, state)?;
                let field = self.get_field_as_interned("field", fields, state)?;

                TypedExpression::Field(TypedFieldAccess {
                    object: Box::new(object),
                    field,
                })
            }
            "Range" => {
                let start = self.get_field_optional_expr("start", fields, state)?;
                let end = self.get_field_optional_expr("end", fields, state)?;
                let inclusive = self.get_field_as_bool("inclusive", fields, state).unwrap_or(false);

                TypedExpression::Range(TypedRange {
                    start: start.map(Box::new),
                    end: end.map(Box::new),
                    inclusive,
                })
            }
            "Struct" | "StructLiteral" => {
                let type_name = self.get_field_as_interned("type_name", fields, state)?;
                let field_inits = self.get_field_as_field_init_list("fields", fields, state)?;

                TypedExpression::Struct(TypedStructLiteral {
                    name: type_name,
                    fields: field_inits,
                })
            }
            "BoolLiteral" => {
                let value = self.get_field_as_bool("value", fields, state)?;
                TypedExpression::Literal(TypedLiteral::Bool(value))
            }
            "Literal" => {
                // Generic literal: { value: TypedLiteral::Variant { ... } }
                let value = if let Some(expr) = self.get_field("value", fields) {
                    self.eval_expr(expr, state)?
                } else {
                    return Err("Literal requires a 'value' field".to_string());
                };
                match value {
                    ParsedValue::Literal(lit) => TypedExpression::Literal(lit),
                    ParsedValue::Int(i) => TypedExpression::Literal(TypedLiteral::Integer(i as i128)),
                    ParsedValue::Float(f) => TypedExpression::Literal(TypedLiteral::Float(f)),
                    ParsedValue::Bool(b) => TypedExpression::Literal(TypedLiteral::Bool(b)),
                    ParsedValue::Text(s) => {
                        let interned = state.intern(&s);
                        TypedExpression::Literal(TypedLiteral::String(interned))
                    }
                    _ => return Err(format!("Literal value must be a literal type, got: {:?}", value)),
                }
            }
            "Ternary" | "If" => {
                // Ternary expression: condition ? then_expr : else_expr
                let condition = self.get_field_as_expr("condition", fields, state)?;
                let then_expr = self.get_field_as_expr("then_expr", fields, state)?;
                let else_expr = self.get_field_as_expr("else_expr", fields, state)?;

                TypedExpression::If(zyntax_typed_ast::TypedIfExpr {
                    condition: Box::new(condition),
                    then_branch: Box::new(then_expr),
                    else_branch: Box::new(else_expr),
                })
            }
            "ListComprehension" => {
                // List comprehension: [expr for var in iter if cond]
                let output_expr = self.get_field_as_expr("output_expr", fields, state)?;
                let variable = self.get_field_as_interned("variable", fields, state)?;
                let iterator = self.get_field_as_expr("iterator", fields, state)?;
                let condition = self.get_field_optional_expr("condition", fields, state)?;

                TypedExpression::ListComprehension(zyntax_typed_ast::TypedListComprehension {
                    output_expr: Box::new(output_expr),
                    variable,
                    iterator: Box::new(iterator),
                    condition: condition.map(Box::new),
                })
            }
            "Slice" => {
                // Slice expression: arr[start:end:step]
                let object = self.get_field_as_expr("object", fields, state)?;
                let start = self.get_field_optional_expr("start", fields, state)?;
                let end = self.get_field_optional_expr("end", fields, state)?;
                let step = self.get_field_optional_expr("step", fields, state)?;

                TypedExpression::Slice(zyntax_typed_ast::TypedSlice {
                    object: Box::new(object),
                    start: start.map(Box::new),
                    end: end.map(Box::new),
                    step: step.map(Box::new),
                })
            }
            "Lambda" => {
                // Lambda expression: def(x): x * 2
                let params = self.get_field_as_lambda_param_list("params", fields, state)?;
                let body = self.get_field_as_expr("body", fields, state)?;

                TypedExpression::Lambda(TypedLambda {
                    params,
                    body: TypedLambdaBody::Expression(Box::new(body)),
                    captures: vec![],
                })
            }
            "ImportModifier" => {
                // Import modifier expression: import asset("image.jpg") as Image
                let loader = self.get_field_as_interned("loader", fields, state)?;
                let path = self.get_field_as_interned("path", fields, state)?;
                let target_type = self.get_field_as_interned("target_type", fields, state)?;

                TypedExpression::ImportModifier(TypedImportModifier {
                    loader,
                    path,
                    target_type,
                })
            }
            "Path" => {
                // Path expression: Type::method or module::function
                let segments = self.get_field_as_interned_list("segments", fields, state)?;

                TypedExpression::Path(TypedPath {
                    segments,
                })
            }
            "Array" => {
                // Array literal: [1, 2, 3]
                let elements = self.get_field_as_expr_list("elements", fields, state)?;

                TypedExpression::Array(elements)
            }
            "Tuple" => {
                // Tuple expression: (a, b, c)
                let elements = self.get_field_as_expr_list("elements", fields, state)?;
                TypedExpression::Tuple(elements)
            }
            _ => return Err(format!("unknown TypedExpression variant: {}", variant)),
        };

        // Determine the type based on the expression variant
        // Default to I64 for integers (64-bit system) and F64 for floats
        let ty = match &expr {
            TypedExpression::Literal(TypedLiteral::Integer(_)) => Type::Primitive(PrimitiveType::I64),
            TypedExpression::Literal(TypedLiteral::Float(_)) => Type::Primitive(PrimitiveType::F64),
            TypedExpression::Literal(TypedLiteral::String(_)) => Type::Primitive(PrimitiveType::String),
            TypedExpression::Literal(TypedLiteral::Bool(_)) => Type::Primitive(PrimitiveType::Bool),
            // Call and Variable expressions need type inference from callee/declaration
            // Use Type::Any to signal that lowering should infer the type
            TypedExpression::Call(_) => Type::Any,
            TypedExpression::Variable(_) => Type::Any,
            // Struct literal gets its type from the struct name - use Unresolved for compiler to resolve
            TypedExpression::Struct(lit) => Type::Unresolved(lit.name),
            _ => Type::Primitive(PrimitiveType::Unit),
        };

        Ok(ParsedValue::Expression(Box::new(typed_node(expr, ty, span))))
    }

    /// Construct a TypedDeclaration variant
    fn construct_declaration<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let decl = match variant {
            "Function" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let params = self.get_field_as_param_list("params", fields, state)?;
                let return_type = self.get_field_optional("return_type", fields, state)?
                    .unwrap_or(Type::Primitive(PrimitiveType::Unit));
                let body = self.get_field_optional_block("body", fields, state)?;

                TypedDeclaration::Function(TypedFunction {
                    name,
                    annotations: vec![],
                    effects: vec![],
                    type_params: vec![],
                    params,
                    return_type,
                    body,
                    visibility: Visibility::Public,
                    is_async: false,
                    is_pure: false,
                    is_external: false,
                    calling_convention: CallingConvention::Default,
                    link_name: None,
                })
            }
            "Variable" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let ty = self.get_field_optional("type_annotation", fields, state)?
                    .unwrap_or(Type::Any);
                let initializer = self.get_field_optional_expr("initializer", fields, state)?;
                let is_mutable = self.get_field_as_bool("is_mutable", fields, state).unwrap_or(false);

                TypedDeclaration::Variable(zyntax_typed_ast::TypedVariable {
                    name,
                    ty,
                    mutability: if is_mutable { Mutability::Mutable } else { Mutability::Immutable },
                    initializer: initializer.map(Box::new),
                    visibility: Visibility::Public,
                })
            }
            "TypeAlias" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let target = self.get_field_optional("target", fields, state)?
                    .unwrap_or(Type::Any);

                TypedDeclaration::TypeAlias(TypedTypeAlias {
                    name,
                    type_params: vec![],
                    target,
                    visibility: Visibility::Public,
                    span,
                })
            }
            "Import" => {
                let module_path = self.get_field_as_interned_list("path", fields, state)?;

                TypedDeclaration::Import(zyntax_typed_ast::TypedImport {
                    module_path,
                    items: vec![zyntax_typed_ast::TypedImportItem::Glob],
                    span,
                })
            }
            "Enum" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let variants = self.get_field_as_variant_list("variants", fields, state)?;

                TypedDeclaration::Enum(zyntax_typed_ast::TypedEnum {
                    name,
                    type_params: vec![],
                    variants,
                    visibility: Visibility::Public,
                    span,
                })
            }
            "Class" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let fields_list = self.get_field_as_field_list("fields", fields, state)?;

                TypedDeclaration::Class(zyntax_typed_ast::TypedClass {
                    name,
                    type_params: vec![],
                    extends: None,
                    implements: vec![],
                    fields: fields_list,
                    methods: vec![],
                    constructors: vec![],
                    visibility: Visibility::Public,
                    is_abstract: false,
                    is_final: false,
                    span,
                })
            }
            "Module" => {
                let name = self.get_field_as_interned("name", fields, state)?;

                TypedDeclaration::Module(zyntax_typed_ast::TypedModule {
                    name,
                    declarations: vec![],
                    visibility: Visibility::Public,
                    span,
                })
            }
            "Impl" => {
                let trait_name = self.get_field_as_interned("trait_name", fields, state)?;
                let for_type_name = self.get_field_as_interned("for_type", fields, state)?;
                let for_type = Type::Unresolved(for_type_name);

                // Get methods and associated types from impl_items
                let items = self.get_field_optional_decl_list("items", fields, state)?;
                let mut methods = vec![];
                let mut associated_types = vec![];

                for item in items {
                    match item.node {
                        TypedDeclaration::Function(func) => {
                            // Convert TypedFunction to TypedMethod
                            let self_name = state.intern("self");
                            let method_params: Vec<zyntax_typed_ast::TypedMethodParam> = func.params
                                .into_iter()
                                .map(|p| {
                                    let is_self = p.name == self_name;
                                    zyntax_typed_ast::TypedMethodParam {
                                        name: p.name,
                                        ty: p.ty,
                                        mutability: p.mutability,
                                        is_self,
                                        kind: p.kind,
                                        default_value: p.default_value,
                                        attributes: p.attributes,
                                        span: p.span,
                                    }
                                })
                                .collect();

                            methods.push(zyntax_typed_ast::TypedMethod {
                                name: func.name,
                                type_params: func.type_params,
                                params: method_params,
                                return_type: func.return_type,
                                body: func.body,
                                visibility: func.visibility,
                                is_static: false,
                                is_async: func.is_async,
                                is_override: false,
                                span: item.span,
                            });
                        }
                        TypedDeclaration::TypeAlias(alias) => {
                            associated_types.push(zyntax_typed_ast::TypedImplAssociatedType {
                                name: alias.name,
                                ty: alias.target,
                                span: alias.span,
                            });
                        }
                        _ => {
                            // Skip other items for now
                        }
                    }
                }

                TypedDeclaration::Impl(zyntax_typed_ast::TypedTraitImpl {
                    trait_name,
                    trait_type_args: vec![],
                    for_type,
                    methods,
                    associated_types,
                    span,
                })
            }
            "Interface" => {
                // Interface is used for trait definitions
                let name = self.get_field_as_interned("name", fields, state)?;
                // TODO: support type_params, methods, associated_types

                TypedDeclaration::Interface(zyntax_typed_ast::TypedInterface {
                    name,
                    type_params: vec![],
                    extends: vec![],
                    methods: vec![],
                    associated_types: vec![],
                    visibility: Visibility::Public,
                    span,
                })
            }
            "AnnotatedFunction" => {
                // An annotated function: merge annotations into the function
                let annotations = self.get_field_as_annotation_list("annotations", fields, state)?;
                let func_decl = self.get_field_as_declaration("function", fields, state)?;

                // Extract function and add annotations
                match func_decl.node {
                    TypedDeclaration::Function(mut func) => {
                        func.annotations = annotations;
                        TypedDeclaration::Function(func)
                    }
                    _ => return Err("AnnotatedFunction requires a Function declaration".to_string()),
                }
            }
            "Effect" => {
                // Algebraic effect declaration: effect Probabilistic { def sample<T>(): T }
                let name = self.get_field_as_interned("name", fields, state)?;
                // TODO: Parse type_params when generic effect support is needed
                let operations = self.get_field_as_effect_op_list("operations", fields, state)?;

                TypedDeclaration::Effect(zyntax_typed_ast::TypedEffect {
                    name,
                    type_params: vec![],
                    operations,
                    span,
                })
            }
            "EffectHandler" => {
                // Effect handler declaration: handler MCMC for Probabilistic { ... }
                let name = self.get_field_as_interned("name", fields, state)?;
                let effect_name = self.get_field_as_interned("effect_name", fields, state)?;
                // TODO: Parse type_params when generic handler support is needed
                let handlers = self.get_field_as_handler_impl_list("handlers", fields, state)?;

                TypedDeclaration::EffectHandler(zyntax_typed_ast::TypedEffectHandler {
                    name,
                    effect_name,
                    type_params: vec![],
                    fields: vec![],
                    handlers,
                    span,
                })
            }
            "ExternStruct" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let type_params = self.get_field_as_type_param_list("type_params", fields, state)?;

                // Runtime prefix is $TypeName to match ZRTL symbol convention
                let name_str = name.resolve_global().unwrap_or_default();
                let runtime_prefix = zyntax_typed_ast::InternedString::new_global(&format!("${}", name_str));

                TypedDeclaration::Extern(TypedExtern::Struct(TypedExternStruct {
                    name,
                    runtime_prefix,
                    type_params,
                }))
            }
            _ => return Err(format!("unknown TypedDeclaration variant: {}", variant)),
        };

        Ok(ParsedValue::Declaration(Box::new(typed_node(
            decl,
            Type::Never,
            span,
        ))))
    }

    /// Construct a TypedProgram
    fn construct_program<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let declarations = self.get_field_as_decl_list("declarations", fields, state)?;

        Ok(ParsedValue::Program(Box::new(TypedProgram {
            declarations,
            span,
            source_files: vec![],
            type_registry: zyntax_typed_ast::type_registry::TypeRegistry::new(),
        })))
    }

    /// Construct a Type variant
    fn construct_type<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        _span: Span,
    ) -> Result<ParsedValue, String> {
        let ty = match variant {
            "Unit" => Type::Primitive(PrimitiveType::Unit),
            "Named" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                Type::Unresolved(name)
            }
            "Primitive" => {
                // Parse primitive type from name
                let name = self.get_field_as_interned("name", fields, state)?;
                let name_str = name.resolve_global()
                    .ok_or_else(|| "cannot resolve interned string".to_string())?;
                // Handle special "type" keyword differently (it's a meta-type)
                if name_str == "type" {
                    return Ok(ParsedValue::Type(Type::Any));
                }
                let prim = match name_str.as_str() {
                    "i8" => PrimitiveType::I8,
                    "i16" => PrimitiveType::I16,
                    "i32" => PrimitiveType::I32,
                    "i64" => PrimitiveType::I64,
                    "u8" => PrimitiveType::U8,
                    "u16" => PrimitiveType::U16,
                    "u32" => PrimitiveType::U32,
                    "u64" => PrimitiveType::U64,
                    "f32" => PrimitiveType::F32,
                    "f64" => PrimitiveType::F64,
                    "bool" => PrimitiveType::Bool,
                    "void" => PrimitiveType::Unit,
                    _ => return Err(format!("unknown primitive type: {}", name_str)),
                };
                Type::Primitive(prim)
            }
            "Pointer" => {
                let pointee = self.get_field_as_type("pointee", fields, state)?;
                Type::Reference {
                    ty: Box::new(pointee),
                    mutability: Mutability::Immutable,
                    lifetime: None,
                    nullability: NullabilityKind::NonNull,
                }
            }
            "Optional" => {
                let inner = self.get_field_as_type("inner", fields, state)?;
                Type::Optional(Box::new(inner))
            }
            "Array" => {
                let element = self.get_field_as_type("element", fields, state)?;
                let size = if let Some(expr) = self.get_field("size", fields) {
                    match self.eval_expr(expr, state)? {
                        ParsedValue::Int(n) => Some(ConstValue::Int(n)),
                        ParsedValue::None => None,
                        ParsedValue::Optional(None) => None,
                        ParsedValue::Optional(Some(v)) => {
                            if let ParsedValue::Int(n) = *v {
                                Some(ConstValue::Int(n))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                Type::Array {
                    element_type: Box::new(element),
                    size,
                    nullability: NullabilityKind::NonNull,
                }
            }
            "ErrorUnion" => {
                let payload = self.get_field_as_type("payload", fields, state)?;
                // Error union with inferred error set
                Type::Result {
                    ok_type: Box::new(payload),
                    err_type: Box::new(Type::Any), // Error type is inferred
                }
            }
            "Tuple" => {
                let elements = self.get_field_as_type_list("elements", fields, state)?;
                Type::Tuple(elements)
            }
            "Function" => {
                // Function type: (params) => return_type
                let param_types = self.get_field_as_type_list_optional("params", fields, state)?
                    .unwrap_or_default();
                let return_type = self.get_field_as_type("return_type", fields, state)?;

                // Convert plain types to ParamInfo
                use zyntax_typed_ast::type_registry::ParamInfo;
                let params: Vec<ParamInfo> = param_types.into_iter().map(|ty| {
                    ParamInfo {
                        name: None,
                        ty,
                        is_optional: false,
                        is_varargs: false,
                        is_keyword_only: false,
                        is_positional_only: false,
                        is_out: false,
                        is_ref: false,
                        is_inout: false,
                    }
                }).collect();

                Type::Function {
                    params,
                    return_type: Box::new(return_type),
                    is_varargs: false,
                    has_named_params: false,
                    has_default_params: false,
                    async_kind: zyntax_typed_ast::type_registry::AsyncKind::Sync,
                    calling_convention: zyntax_typed_ast::type_registry::CallingConvention::Default,
                    nullability: zyntax_typed_ast::type_registry::NullabilityKind::NonNull,
                }
            }
            _ => return Err(format!("unknown Type variant: {}", variant)),
        };

        Ok(ParsedValue::Type(ty))
    }

    /// Construct a TypedLiteral variant
    fn construct_literal<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<ParsedValue, String> {
        let lit = match variant {
            "Int" | "Integer" => {
                let value = self.get_field_as_int("value", fields, state)?;
                TypedLiteral::Integer(value as i128)
            }
            "Float" => {
                let value = self.get_field_as_float("value", fields, state)?;
                TypedLiteral::Float(value)
            }
            "Bool" => {
                // Can be a bool value or a string "true"/"false"
                let value = if let Some(expr) = self.get_field("value", fields) {
                    match self.eval_expr(expr, state)? {
                        ParsedValue::Bool(b) => b,
                        ParsedValue::Text(s) => s == "true",
                        other => return Err(format!("Bool value must be bool or text, got: {:?}", other)),
                    }
                } else {
                    false
                };
                TypedLiteral::Bool(value)
            }
            "String" => {
                let value = self.get_field_as_string("value", fields, state)?;
                // Strip quotes from string literal if present
                let unquoted = value.trim_matches('"').to_string();
                let interned = state.intern(&unquoted);
                TypedLiteral::String(interned)
            }
            "Char" => {
                let value = self.get_field_as_string("value", fields, state)?;
                let c = value.chars().next().unwrap_or('\0');
                TypedLiteral::Char(c)
            }
            _ => return Err(format!("unknown TypedLiteral variant: {}", variant)),
        };

        Ok(ParsedValue::Literal(lit))
    }

    /// Construct a TypedField (struct/class field)
    fn construct_field<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let name = self.get_field_as_interned("name", fields, state)?;
        let ty = self.get_field_optional("ty", fields, state)?
            .unwrap_or(Type::Any);

        Ok(ParsedValue::Field(zyntax_typed_ast::TypedField {
            name,
            ty,
            initializer: None,
            visibility: Visibility::Public,
            mutability: Mutability::Immutable,
            is_static: false,
            span,
        }))
    }

    /// Construct a TypedVariant (enum variant)
    fn construct_variant<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let name = self.get_field_as_interned("name", fields, state)?;

        // Handle fields - could be Unit (no fields), Named, or Tuple
        let variant_fields = if let Some(expr) = self.get_field("fields", fields) {
            let val = self.eval_expr(expr, state)?;
            match val {
                ParsedValue::Text(s) if s == "Unit" || s == "TypedVariantFields::Unit" => TypedVariantFields::Unit,
                ParsedValue::None => TypedVariantFields::Unit,
                _ => TypedVariantFields::Unit, // TODO: handle named and tuple fields
            }
        } else {
            TypedVariantFields::Unit
        };

        Ok(ParsedValue::Variant(TypedVariant {
            name,
            fields: variant_fields,
            discriminant: None,
            span,
        }))
    }

    /// Construct a TypedFieldInit (field initialization in struct literal)
    fn construct_field_init<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        _span: Span,
    ) -> Result<ParsedValue, String> {
        let name = self.get_field_as_interned("name", fields, state)?;
        let value = self.get_field_as_expr("value", fields, state)?;

        Ok(ParsedValue::FieldInit {
            name,
            value: Box::new(ParsedValue::Expression(Box::new(value))),
        })
    }

    /// Construct a TypedMatchArm (match arm with pattern and body)
    fn construct_match_arm<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let pattern = self.get_field_as_pattern("pattern", fields, state)?;
        let body = self.get_field_as_block("body", fields, state)?;

        // Convert block to expression (Block expression)
        let body_expr = typed_node(
            TypedExpression::Block(body),
            Type::Any,
            span,
        );

        Ok(ParsedValue::MatchArm(TypedMatchArm {
            pattern: Box::new(pattern),
            guard: None,
            body: Box::new(body_expr),
        }))
    }

    /// Construct a TypedPattern variant (for destructuring)
    fn construct_pattern<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let pattern = match variant {
            "Identifier" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let is_mutable = self.get_field_as_bool("is_mutable", fields, state).unwrap_or(false);
                let mutability = if is_mutable { Mutability::Mutable } else { Mutability::Immutable };
                TypedPattern::Identifier { name, mutability }
            }
            "Wildcard" => {
                TypedPattern::Wildcard
            }
            "Tuple" => {
                let elements = self.get_field_as_pattern_list("elements", fields, state)?;
                TypedPattern::Tuple(elements)
            }
            "Constructor" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let fields_list = self.get_field_as_pattern_list("fields", fields, state)?;

                // For patterns like Some(x) or None(), convert to Constructor type
                let constructor_type = Type::Unresolved(name);
                let inner_pattern = if fields_list.is_empty() {
                    typed_node(TypedPattern::Wildcard, Type::Any, span)
                } else {
                    fields_list.into_iter().next().unwrap()
                };

                TypedPattern::Constructor {
                    constructor: constructor_type,
                    pattern: Box::new(inner_pattern),
                }
            }
            _ => return Err(format!("unknown TypedPattern variant: {}", variant)),
        };

        Ok(ParsedValue::Pattern(Box::new(typed_node(
            pattern,
            Type::Any,
            span,
        ))))
    }

    /// Construct a TypedAnnotation (e.g., @deprecated, @inline)
    fn construct_annotation<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let name = self.get_field_as_interned("name", fields, state)?;
        let args = self.get_field_as_annotation_arg_list("args", fields, state)?;

        Ok(ParsedValue::Annotation(TypedAnnotation {
            name,
            args,
            span,
        }))
    }

    /// Construct a TypedAnnotationArg (positional or named)
    fn construct_annotation_arg<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        _span: Span,
    ) -> Result<ParsedValue, String> {
        let arg = match variant {
            "Positional" => {
                let value = self.get_field_as_annotation_value("value", fields, state)?;
                TypedAnnotationArg::Positional(value)
            }
            "Named" => {
                let name = self.get_field_as_interned("name", fields, state)?;
                let value = self.get_field_as_annotation_value("value", fields, state)?;
                TypedAnnotationArg::Named { name, value }
            }
            _ => return Err(format!("unknown TypedAnnotationArg variant: {}", variant)),
        };

        Ok(ParsedValue::AnnotationArg(arg))
    }

    /// Construct a TypedAnnotationValue (string, int, bool, identifier, list)
    fn construct_annotation_value<'a>(
        &self,
        variant: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        _span: Span,
    ) -> Result<ParsedValue, String> {
        let value = match variant {
            "String" => {
                let s = self.get_field_as_interned("value", fields, state)?;
                TypedAnnotationValue::String(s)
            }
            "Integer" => {
                let n = self.get_field_as_int("value", fields, state)?;
                TypedAnnotationValue::Integer(n)
            }
            "Float" => {
                let f = self.get_field_as_float("value", fields, state)?;
                TypedAnnotationValue::Float(f)
            }
            "Bool" => {
                let b = self.get_field_as_bool("value", fields, state)?;
                TypedAnnotationValue::Bool(b)
            }
            "Identifier" => {
                let id = self.get_field_as_interned("value", fields, state)?;
                TypedAnnotationValue::Identifier(id)
            }
            "List" => {
                let items = self.get_field_as_annotation_value_list("items", fields, state)?;
                TypedAnnotationValue::List(items)
            }
            _ => return Err(format!("unknown TypedAnnotationValue variant: {}", variant)),
        };

        Ok(ParsedValue::AnnotationValue(value))
    }

    /// Construct a TypedEffectOp (effect operation declaration)
    fn construct_effect_op<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let name = self.get_field_as_interned("name", fields, state)?;
        // TODO: Parse type_params when generic effect op support is needed
        let params = self.get_field_as_param_list("params", fields, state)?;
        let return_type = self.get_field_optional("return_type", fields, state)?
            .unwrap_or(Type::Primitive(PrimitiveType::Unit));

        Ok(ParsedValue::EffectOp(zyntax_typed_ast::TypedEffectOp {
            name,
            type_params: vec![],
            params,
            return_type,
            span,
        }))
    }

    /// Construct a TypedEffectHandlerImpl (handler operation implementation)
    fn construct_effect_handler_impl<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let op_name = self.get_field_as_interned("op_name", fields, state)?;
        // TODO: Parse type_params when generic handler impl support is needed
        let params = self.get_field_as_param_list("params", fields, state)?;
        let return_type = self.get_field_optional("return_type", fields, state)?
            .unwrap_or(Type::Primitive(PrimitiveType::Unit));
        let body = self.get_field_optional_block("body", fields, state)?;

        Ok(ParsedValue::EffectHandlerImpl(zyntax_typed_ast::TypedEffectHandlerImpl {
            op_name,
            type_params: vec![],
            params,
            return_type,
            body,
            span,
        }))
    }

    /// Construct a TypedParameter (function parameter)
    fn construct_parameter<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        let name = self.get_field_as_interned("name", fields, state)?;
        let ty = self.get_field_optional("ty", fields, state)?
            .unwrap_or(Type::Any);

        // Check for kind
        let kind = if let Some(expr) = self.get_field("kind", fields) {
            match self.eval_expr(expr, state)? {
                ParsedValue::Text(s) if s.contains("KeywordOnly") => ParameterKind::KeywordOnly,
                ParsedValue::Text(s) if s.contains("Rest") => ParameterKind::Rest,
                _ => ParameterKind::Regular,
            }
        } else {
            ParameterKind::Regular
        };

        // Parse default_value if present
        let default_value = self.get_field_optional_expr("default_value", fields, state)?
            .map(|expr| Box::new(expr));

        Ok(ParsedValue::Parameter(TypedParameter {
            name,
            ty,
            mutability: Mutability::Immutable,
            kind,
            default_value,
            attributes: vec![],
            span,
        }))
    }

    /// Construct a TypedLambdaParam (lambda parameter)
    fn construct_lambda_param<'a>(
        &self,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
        _span: Span,
    ) -> Result<ParsedValue, String> {
        let name = self.get_field_as_interned("name", fields, state)?;
        let ty = self.get_field_optional("ty", fields, state)?;

        Ok(ParsedValue::LambdaParam(TypedLambdaParam {
            name,
            ty,
        }))
    }

    /// Get a field value as a Type
    fn get_field_as_type<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Type, String> {
        if let Some(expr) = self.get_field(name, fields) {
            match self.eval_expr(expr, state)? {
                ParsedValue::Type(ty) => Ok(ty),
                other => Err(format!("field '{}' is not a type: {:?}", name, other)),
            }
        } else {
            Err(format!("missing required type field: {}", name))
        }
    }

    fn get_field_as_type_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<Type>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;

        match val {
            ParsedValue::List(items) => {
                items.into_iter()
                    .map(|item| match item {
                        ParsedValue::Type(ty) => Ok(ty),
                        other => Err(format!("expected Type, got {:?}", other)),
                    })
                    .collect()
            }
            ParsedValue::Optional(None) | ParsedValue::None => Ok(vec![]),
            ParsedValue::Optional(Some(inner)) => {
                match *inner {
                    ParsedValue::List(items) => {
                        items.into_iter()
                            .map(|item| match item {
                                ParsedValue::Type(ty) => Ok(ty),
                                other => Err(format!("expected Type, got {:?}", other)),
                            })
                            .collect()
                    }
                    ParsedValue::Type(ty) => Ok(vec![ty]),
                    other => Err(format!("expected Type list, got {:?}", other)),
                }
            }
            ParsedValue::Type(ty) => Ok(vec![ty]),
            other => Err(format!("field '{}' is not a type list: {:?}", name, other)),
        }
    }

    /// Get an optional field as a list of types
    fn get_field_as_type_list_optional<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Option<Vec<Type>>, String> {
        let Some(expr) = self.get_field(name, fields) else {
            return Ok(None);
        };
        let val = self.eval_expr(expr, state)?;

        match val {
            ParsedValue::List(items) => {
                let types: Result<Vec<Type>, String> = items.into_iter()
                    .map(|item| match item {
                        ParsedValue::Type(ty) => Ok(ty),
                        other => Err(format!("expected Type, got {:?}", other)),
                    })
                    .collect();
                Ok(Some(types?))
            }
            ParsedValue::Optional(None) | ParsedValue::None => Ok(None),
            ParsedValue::Optional(Some(inner)) => {
                match *inner {
                    ParsedValue::List(items) => {
                        let types: Result<Vec<Type>, String> = items.into_iter()
                            .map(|item| match item {
                                ParsedValue::Type(ty) => Ok(ty),
                                other => Err(format!("expected Type, got {:?}", other)),
                            })
                            .collect();
                        Ok(Some(types?))
                    }
                    ParsedValue::Type(ty) => Ok(Some(vec![ty])),
                    other => Err(format!("expected Type list, got {:?}", other)),
                }
            }
            ParsedValue::Type(ty) => Ok(Some(vec![ty])),
            other => Err(format!("field '{}' is not a type list: {:?}", name, other)),
        }
    }

    /// Execute a helper function call
    fn execute_helper_call<'a>(
        &self,
        function: &str,
        args: &[ExprIR],
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        match function {
            "intern" => {
                if args.len() != 1 {
                    return Err("intern() requires exactly 1 argument".to_string());
                }
                let text = self.eval_expr_as_string(&args[0], state)?;
                let interned = state.intern(&text);
                Ok(ParsedValue::Interned(interned))
            }
            "parse_int" => {
                if args.len() != 1 {
                    return Err("parse_int() requires exactly 1 argument".to_string());
                }
                let text = self.eval_expr_as_string(&args[0], state)?;
                let value: i64 = text.parse()
                    .map_err(|_| format!("cannot parse '{}' as integer", text))?;
                Ok(ParsedValue::Int(value))
            }
            "parse_float" => {
                if args.len() != 1 {
                    return Err("parse_float() requires exactly 1 argument".to_string());
                }
                let text = self.eval_expr_as_string(&args[0], state)?;
                let value: f64 = text.parse()
                    .map_err(|_| format!("cannot parse '{}' as float", text))?;
                Ok(ParsedValue::Float(value))
            }
            "parse_bool" => {
                if args.len() != 1 {
                    return Err("parse_bool() requires exactly 1 argument".to_string());
                }
                let text = self.eval_expr_as_string(&args[0], state)?;
                let value = match text.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err(format!("cannot parse '{}' as boolean", text)),
                };
                Ok(ParsedValue::Bool(value))
            }
            "text" => {
                // Get the text of the current match (typically used in atomic rules)
                // The text is bound to __text__ for atomic rules
                if args.is_empty() {
                    // Return the captured text from atomic rule
                    Ok(state.get_binding("__text__")
                        .cloned()
                        .unwrap_or(ParsedValue::Text(String::new())))
                } else {
                    self.eval_expr(&args[0], state)
                }
            }
            "prepend_list" => {
                // prepend_list(first, rest) - prepend first to the list rest
                // Useful for comma-separated lists: first:expr ~ ("," ~ rest:expr)*
                // When the repetition matches zero times, rest won't be bound - treat as empty
                if args.len() != 2 {
                    return Err("prepend_list() requires exactly 2 arguments".to_string());
                }
                let first = self.eval_expr(&args[0], state)?;

                // Try to evaluate rest - if binding not found, treat as empty list
                let rest = match self.eval_expr(&args[1], state) {
                    Ok(v) => v,
                    Err(e) if e.contains("not found") => ParsedValue::List(vec![]),
                    Err(e) => return Err(e),
                };

                let mut result = vec![first];
                match rest {
                    ParsedValue::List(items) => result.extend(items),
                    ParsedValue::Optional(Some(inner)) => {
                        if let ParsedValue::List(items) = *inner {
                            result.extend(items);
                        } else {
                            result.push(*inner);
                        }
                    }
                    ParsedValue::Optional(None) => {} // Empty rest, just return [first]
                    ParsedValue::None => {} // Empty rest
                    other => result.push(other),
                }
                Ok(ParsedValue::List(result))
            }
            "concat_list" => {
                // concat_list(list1, list2) - concatenate two lists
                if args.len() != 2 {
                    return Err("concat_list() requires exactly 2 arguments".to_string());
                }
                let list1 = self.eval_expr(&args[0], state)?;
                let list2 = self.eval_expr(&args[1], state)?;

                let mut result = Vec::new();
                match list1 {
                    ParsedValue::List(items) => result.extend(items),
                    other => result.push(other),
                }
                match list2 {
                    ParsedValue::List(items) => result.extend(items),
                    other => result.push(other),
                }
                Ok(ParsedValue::List(result))
            }
            "make_pair" => {
                // make_pair(a, b) - create a two-element list [a, b]
                // Useful for building operator-operand pairs in binary expression parsing
                if args.len() != 2 {
                    return Err("make_pair() requires exactly 2 arguments".to_string());
                }
                let first = self.eval_expr(&args[0], state)?;
                let second = self.eval_expr(&args[1], state)?;
                Ok(ParsedValue::List(vec![first, second]))
            }
            "fold_left_ops" => {
                // fold_left_ops(first, rest) - fold binary operations with left associativity
                // first: the first operand expression
                // rest: list of [op, operand, op, operand, ...] pairs or [[op, operand], [op, operand], ...]
                // Returns nested Binary expressions folded left-to-right
                // e.g., fold_left_ops(a, [["+", b], ["-", c]]) -> Binary(Binary(a, +, b), -, c)
                if args.len() != 2 {
                    return Err("fold_left_ops() requires exactly 2 arguments (first, rest)".to_string());
                }
                let first = self.eval_expr(&args[0], state)?;
                let rest = self.eval_expr(&args[1], state)?;

                // Get the rest as a list
                let rest_list = match rest {
                    ParsedValue::List(items) => items,
                    ParsedValue::Optional(None) | ParsedValue::None => vec![],
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::List(items) => items,
                            other => vec![other],
                        }
                    }
                    other => vec![other],
                };

                // If no rest, return first as-is
                if rest_list.is_empty() {
                    return Ok(first);
                }

                // Flatten nested lists from the repeat pattern
                let mut flat_items: Vec<ParsedValue> = Vec::new();
                for item in rest_list {
                    match item {
                        ParsedValue::List(inner) => {
                            flat_items.extend(inner);
                        }
                        other => flat_items.push(other),
                    }
                }

                // Now we have [op, operand, op, operand, ...]
                // Fold left: accumulator op operand -> new accumulator
                let mut acc = first;
                let mut i = 0;
                while i + 1 < flat_items.len() {
                    let op_val = &flat_items[i];
                    let operand_val = flat_items[i + 1].clone();

                    // Get operator string
                    let op_str = match op_val {
                        ParsedValue::Text(s) => s.clone(),
                        ParsedValue::Interned(s) => s.resolve_global().unwrap_or_else(|| "+".to_string()),
                        other => {
                            log::warn!("[fold_left_ops] Unexpected operator value: {:?}", other);
                            "+".to_string()
                        }
                    };

                    // Convert accumulator to expression
                    let left_expr = self.parsed_value_to_expr(acc, state)?;

                    // Convert operand to expression
                    let right_expr = self.parsed_value_to_expr(operand_val, state)?;

                    // Create Binary expression
                    let binary_op = self.string_to_binary_op(&op_str)?;
                    acc = ParsedValue::Expression(Box::new(typed_node(
                        TypedExpression::Binary(zyntax_typed_ast::TypedBinary {
                            op: binary_op,
                            left: Box::new(left_expr),
                            right: Box::new(right_expr),
                        }),
                        Type::Unknown,
                        span,
                    )));

                    i += 2;
                }

                Ok(acc)
            }
            "fold_postfix" => {
                // fold_postfix(base, suffixes) - fold postfix operations into nested expressions
                // base: the base expression (TypedExpression)
                // suffixes: list of suffix operations, each with a "kind" field
                // Returns the folded TypedExpression
                if args.len() != 2 {
                    return Err("fold_postfix() requires exactly 2 arguments (base, suffixes)".to_string());
                }
                let base = self.eval_expr(&args[0], state)?;
                let suffixes = self.eval_expr(&args[1], state)?;

                // Get the suffixes as a list
                let suffix_list = match suffixes {
                    ParsedValue::List(items) => items,
                    ParsedValue::Optional(None) | ParsedValue::None => vec![],
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::List(items) => items,
                            other => vec![other],
                        }
                    }
                    other => vec![other],
                };

                // If no suffixes, return base as-is
                if suffix_list.is_empty() {
                    return Ok(base);
                }

                // Fold left over suffixes
                let mut acc = base;
                for suffix in suffix_list {
                    acc = self.apply_postfix_suffix(acc, suffix, state, span)?;
                }

                Ok(acc)
            }
            "fold_concat" => {
                // fold_concat(parts) - fold string parts into nested concat calls
                // e.g., fold_concat(["a", "b", "c"]) -> concat(concat("a", "b"), "c")
                // Used by f-string desugaring
                if args.len() != 1 {
                    return Err("fold_concat() requires exactly 1 argument (parts list)".to_string());
                }
                let parts = self.eval_expr(&args[0], state)?;

                // Get the parts as a list
                let parts_list = match parts {
                    ParsedValue::List(items) => items,
                    ParsedValue::Optional(None) | ParsedValue::None => vec![],
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::List(items) => items,
                            other => vec![other],
                        }
                    }
                    other => vec![other],
                };

                // Handle edge cases
                if parts_list.is_empty() {
                    // Return empty string
                    return Ok(ParsedValue::Expression(Box::new(typed_node(
                        TypedExpression::Literal(zyntax_typed_ast::TypedLiteral::String(
                            state.intern(""),
                        )),
                        Type::Primitive(zyntax_typed_ast::PrimitiveType::String),
                        span,
                    ))));
                }

                if parts_list.len() == 1 {
                    // Just return the single part as an expression
                    return self.parsed_value_to_expr(parts_list.into_iter().next().unwrap(), state)
                        .map(|e| ParsedValue::Expression(Box::new(e)));
                }

                // Fold left: concat(concat(a, b), c)
                let concat_name = state.intern("concat");
                let mut iter = parts_list.into_iter();
                let first = iter.next().unwrap();
                let mut acc = self.parsed_value_to_expr(first, state)?;

                for part in iter {
                    let part_expr = self.parsed_value_to_expr(part, state)?;
                    // Create concat(acc, part)
                    acc = typed_node(
                        TypedExpression::Call(TypedCall {
                            callee: Box::new(typed_node(
                                TypedExpression::Variable(concat_name),
                                Type::Unknown,
                                span,
                            )),
                            positional_args: vec![acc, part_expr],
                            named_args: vec![],
                            type_args: vec![],
                        }),
                        Type::Primitive(zyntax_typed_ast::PrimitiveType::String),
                        span,
                    );
                }

                Ok(ParsedValue::Expression(Box::new(acc)))
            }
            _ => Err(format!("unknown helper function: {}", function)),
        }
    }

    /// Apply a postfix suffix to an accumulator expression
    /// The suffix should be a Suffix value with a kind like "SuffixField", "SuffixMethod", etc.
    fn apply_postfix_suffix<'a>(
        &self,
        acc: ParsedValue,
        suffix: ParsedValue,
        state: &mut ParserState<'a>,
        span: Span,
    ) -> Result<ParsedValue, String> {
        // The suffix should be a Suffix variant with kind and fields
        match suffix {
            ParsedValue::Suffix { kind, fields } => {
                match kind.as_str() {
                    "SuffixField" => {
                        // Field access: acc.field
                        let field = fields.get("field")
                            .ok_or("SuffixField missing 'field'")?;

                        // Convert acc to TypedExpression
                        let acc_expr = self.parsed_value_to_expr(acc, state)?;

                        // Get the field name
                        let field_name = match field.as_ref() {
                            ParsedValue::Interned(s) => *s,
                            ParsedValue::Text(s) => state.intern(s),
                            other => return Err(format!("SuffixField field must be interned, got {:?}", other)),
                        };

                        Ok(ParsedValue::Expression(Box::new(typed_node(
                            TypedExpression::Field(TypedFieldAccess {
                                object: Box::new(acc_expr),
                                field: field_name,
                            }),
                            Type::Unknown,
                            span,
                        ))))
                    }
                    "SuffixMethod" => {
                        // Method call: acc.method(args)
                        let method = fields.get("method")
                            .ok_or("SuffixMethod missing 'method'")?;
                        let args_val = fields.get("args");

                        // Convert acc to TypedExpression
                        let acc_expr = self.parsed_value_to_expr(acc, state)?;

                        // Get the method name
                        let method_name = match method.as_ref() {
                            ParsedValue::Interned(s) => *s,
                            ParsedValue::Text(s) => state.intern(s),
                            other => return Err(format!("SuffixMethod method must be interned, got {:?}", other)),
                        };

                        // Get args as expression list
                        let args = match args_val {
                            Some(expr_val) => {
                                self.parsed_value_list_to_expr_vec(expr_val.as_ref().clone(), state)?
                            }
                            None => vec![],
                        };

                        Ok(ParsedValue::Expression(Box::new(typed_node(
                            TypedExpression::MethodCall(TypedMethodCall {
                                receiver: Box::new(acc_expr),
                                method: method_name,
                                type_args: vec![],
                                positional_args: args,
                                named_args: vec![],
                            }),
                            Type::Unknown,
                            span,
                        ))))
                    }
                    "SuffixCall" => {
                        // Function call: acc(args) - this treats acc as the callee
                        let args_val = fields.get("args");

                        // Convert acc to TypedExpression
                        let acc_expr = self.parsed_value_to_expr(acc, state)?;

                        // Get args as expression list
                        let args = match args_val {
                            Some(expr_val) => {
                                self.parsed_value_list_to_expr_vec(expr_val.as_ref().clone(), state)?
                            }
                            None => vec![],
                        };

                        Ok(ParsedValue::Expression(Box::new(typed_node(
                            TypedExpression::Call(TypedCall {
                                callee: Box::new(acc_expr),
                                positional_args: args,
                                named_args: vec![],
                                type_args: vec![],
                            }),
                            Type::Unknown,
                            span,
                        ))))
                    }
                    "SuffixIndex" => {
                        // Index access: acc[index]
                        let index = fields.get("index")
                            .ok_or("SuffixIndex missing 'index'")?;

                        // Convert acc to TypedExpression
                        let acc_expr = self.parsed_value_to_expr(acc, state)?;

                        // Convert index to TypedExpression
                        let index_expr = self.parsed_value_to_expr(index.as_ref().clone(), state)?;

                        Ok(ParsedValue::Expression(Box::new(typed_node(
                            TypedExpression::Index(TypedIndex {
                                object: Box::new(acc_expr),
                                index: Box::new(index_expr),
                            }),
                            Type::Unknown,
                            span,
                        ))))
                    }
                    "SuffixSlice" => {
                        // Slice access: acc[start:end:step]
                        let start_val = fields.get("start");
                        let end_val = fields.get("end");
                        let step_val = fields.get("step");

                        // Convert acc to TypedExpression
                        let acc_expr = self.parsed_value_to_expr(acc, state)?;

                        // Convert optional slice components
                        let start = match start_val {
                            Some(v) => match v.as_ref() {
                                ParsedValue::None | ParsedValue::Optional(None) => None,
                                other => Some(Box::new(self.parsed_value_to_expr(other.clone(), state)?)),
                            }
                            None => None,
                        };
                        let end = match end_val {
                            Some(v) => match v.as_ref() {
                                ParsedValue::None | ParsedValue::Optional(None) => None,
                                other => Some(Box::new(self.parsed_value_to_expr(other.clone(), state)?)),
                            }
                            None => None,
                        };
                        let step = match step_val {
                            Some(v) => match v.as_ref() {
                                ParsedValue::None | ParsedValue::Optional(None) => None,
                                other => Some(Box::new(self.parsed_value_to_expr(other.clone(), state)?)),
                            }
                            None => None,
                        };

                        Ok(ParsedValue::Expression(Box::new(typed_node(
                            TypedExpression::Slice(zyntax_typed_ast::TypedSlice {
                                object: Box::new(acc_expr),
                                start,
                                end,
                                step,
                            }),
                            Type::Unknown,
                            span,
                        ))))
                    }
                    other => Err(format!("Unknown postfix suffix type: {}", other)),
                }
            }
            other => Err(format!("Postfix suffix must be a Suffix, got {:?}", other)),
        }
    }

    /// Convert a ParsedValue list directly to a Vec of TypedExpression
    fn parsed_value_list_to_expr_vec<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedNode<TypedExpression>>, String> {
        match val {
            ParsedValue::List(items) => {
                items.into_iter()
                    .map(|item| self.parsed_value_to_expr(item, state))
                    .collect()
            }
            ParsedValue::Optional(None) | ParsedValue::None => Ok(vec![]),
            ParsedValue::Optional(Some(inner)) => {
                match *inner {
                    ParsedValue::List(items) => {
                        items.into_iter()
                            .map(|item| self.parsed_value_to_expr(item, state))
                            .collect()
                    }
                    other => Ok(vec![self.parsed_value_to_expr(other, state)?]),
                }
            }
            other => Ok(vec![self.parsed_value_to_expr(other, state)?]),
        }
    }

    /// Evaluate an expression IR to a ParsedValue
    fn eval_expr<'a>(
        &self,
        expr: &ExprIR,
        state: &mut ParserState<'a>,
    ) -> Result<ParsedValue, String> {
        match expr {
            ExprIR::Binding(name) => {
                // Handle special binding names
                if name == "text" {
                    // 'text' refers to the captured text from an atomic rule (bound to __text__)
                    return state.get_binding("__text__")
                        .cloned()
                        .ok_or_else(|| "binding 'text' not found (not in atomic rule?)".to_string());
                }
                state.get_binding(name)
                    .cloned()
                    .ok_or_else(|| format!("binding '{}' not found", name))
            }
            ExprIR::StringLit(s) => Ok(ParsedValue::Text(s.clone())),
            ExprIR::IntLit(i) => Ok(ParsedValue::Int(*i)),
            ExprIR::BoolLit(b) => Ok(ParsedValue::Bool(*b)),
            ExprIR::List(items) => {
                let values: Result<Vec<_>, _> = items.iter()
                    .map(|item| self.eval_expr(item, state))
                    .collect();
                Ok(ParsedValue::List(values?))
            }
            ExprIR::FunctionCall { function, args } => {
                let span = Span::new(0, 0); // Dummy span for helper calls
                self.execute_helper_call(function, args, state, span)
            }
            ExprIR::MethodCall { receiver, method, args } => {
                let recv_val = self.eval_expr(receiver, state)?;
                match method.as_str() {
                    "unwrap_or_default" | "unwrap_or" => {
                        match recv_val {
                            ParsedValue::Optional(Some(inner)) => Ok(*inner),
                            ParsedValue::Optional(None) => {
                                if args.is_empty() {
                                    Ok(ParsedValue::None)
                                } else {
                                    self.eval_expr(&args[0], state)
                                }
                            }
                            other => Ok(other),
                        }
                    }
                    "is_some" => {
                        match recv_val {
                            ParsedValue::Optional(opt) => Ok(ParsedValue::Bool(opt.is_some())),
                            _ => Ok(ParsedValue::Bool(true)),
                        }
                    }
                    _ => Err(format!("unknown method: {}", method)),
                }
            }
            ExprIR::FieldAccess { base, field } => {
                let base_val = self.eval_expr(base, state)?;
                // For now, just return the field name as we don't have struct fields
                Err(format!("field access not supported: {}", field))
            }
            ExprIR::Intern(inner) => {
                let text = self.eval_expr_as_string(inner, state)?;
                let interned = state.intern(&text);
                Ok(ParsedValue::Interned(interned))
            }
            ExprIR::Text(inner) => {
                // Get text representation of a value
                let val = self.eval_expr(inner, state)?;
                match val {
                    ParsedValue::Text(s) => Ok(ParsedValue::Text(s)),
                    ParsedValue::Int(i) => Ok(ParsedValue::Text(i.to_string())),
                    ParsedValue::Float(f) => Ok(ParsedValue::Text(f.to_string())),
                    ParsedValue::Bool(b) => Ok(ParsedValue::Text(b.to_string())),
                    _ => Err("cannot convert to text".to_string()),
                }
            }
            ExprIR::IsSome(inner) => {
                let val = self.eval_expr(inner, state)?;
                match val {
                    ParsedValue::Optional(opt) => Ok(ParsedValue::Bool(opt.is_some())),
                    _ => Ok(ParsedValue::Bool(true)),
                }
            }
            ExprIR::StructLit { type_name, fields } => {
                // Construct a TypedAST node based on type_name
                let span = Span::new(0, 0);
                self.execute_construct(type_name, fields, state, span)
            }
            ExprIR::EnumVariant { type_name, variant, value } => {
                // Handle enum variants like Type::Named, Some(x), None, Box::new(x)
                let span = Span::new(0, 0);

                if let Some(val_expr) = value {
                    // Variant with value
                    match (type_name.as_str(), variant.as_str()) {
                        ("Option" | "Some", _) | (_, "Some") => {
                            let inner = self.eval_expr(val_expr, state)?;
                            Ok(ParsedValue::Optional(Some(Box::new(inner))))
                        }
                        ("Box", "new") => {
                            // Box::new(expr) - just evaluate the inner expression
                            // The Box is transparent at the value level
                            self.eval_expr(val_expr, state)
                        }
                        _ => {
                            // For other variants, try constructing them
                            let full_path = format!("{}::{}", type_name, variant);
                            self.execute_construct(&full_path, &[], state, span)
                        }
                    }
                } else {
                    // Variant without value
                    match (type_name.as_str(), variant.as_str()) {
                        (_, "None") => Ok(ParsedValue::Optional(None)),
                        ("TypedVariantFields", "Unit") | (_, "Unit") if type_name == "TypedVariantFields" => {
                            // Return a marker text that construct_variant can recognize
                            Ok(ParsedValue::Text("TypedVariantFields::Unit".to_string()))
                        }
                        ("ParameterKind", "Regular") => {
                            Ok(ParsedValue::Text("ParameterKind::Regular".to_string()))
                        }
                        ("ParameterKind", "KeywordOnly") => {
                            Ok(ParsedValue::Text("ParameterKind::KeywordOnly".to_string()))
                        }
                        ("ParameterKind", "Rest") => {
                            Ok(ParsedValue::Text("ParameterKind::Rest".to_string()))
                        }
                        ("Mutability", "Mutable") => {
                            Ok(ParsedValue::Text("Mutability::Mutable".to_string()))
                        }
                        ("Mutability", "Immutable") => {
                            Ok(ParsedValue::Text("Mutability::Immutable".to_string()))
                        }
                        _ => {
                            let full_path = format!("{}::{}", type_name, variant);
                            self.execute_construct(&full_path, &[], state, span)
                        }
                    }
                }
            }
            ExprIR::Default(type_name) => {
                // Return default value for a type
                match type_name.as_str() {
                    "Vec" | "[]" => Ok(ParsedValue::List(vec![])),
                    "String" => Ok(ParsedValue::Text(String::new())),
                    "i64" | "i32" => Ok(ParsedValue::Int(0)),
                    "f64" | "f32" => Ok(ParsedValue::Float(0.0)),
                    "bool" => Ok(ParsedValue::Bool(false)),
                    _ => Ok(ParsedValue::None),
                }
            }
            ExprIR::UnwrapOr { optional, default } => {
                let opt_val = self.eval_expr(optional, state)?;
                match opt_val {
                    ParsedValue::Optional(Some(inner)) => Ok(*inner),
                    ParsedValue::Optional(None) => self.eval_expr(default, state),
                    other => Ok(other),
                }
            }
            ExprIR::MapOption { optional, param, body } => {
                let opt_val = self.eval_expr(optional, state)?;
                match opt_val {
                    ParsedValue::Optional(Some(inner)) => {
                        // Temporarily bind the parameter
                        let old_binding = state.get_binding(param).cloned();
                        state.set_binding(param, *inner);
                        let result = self.eval_expr(body, state)?;
                        // Restore old binding
                        if let Some(old) = old_binding {
                            state.set_binding(param, old);
                        }
                        Ok(ParsedValue::Optional(Some(Box::new(result))))
                    }
                    ParsedValue::Optional(None) => Ok(ParsedValue::Optional(None)),
                    _ => Err("map_option requires Optional value".to_string()),
                }
            }
            ExprIR::Cast { expr, target_type } => {
                // For now, just evaluate the expression
                self.eval_expr(expr, state)
            }
            ExprIR::GetSpan(_) => {
                // Return a dummy span for now
                Ok(ParsedValue::Span(Span::new(0, 0)))
            }
            ExprIR::Binary { left, op, right } => {
                let l = self.eval_expr(left, state)?;
                let r = self.eval_expr(right, state)?;

                // Handle basic binary operations
                match (l, op.as_str(), r) {
                    (ParsedValue::Int(a), "+", ParsedValue::Int(b)) => Ok(ParsedValue::Int(a + b)),
                    (ParsedValue::Int(a), "-", ParsedValue::Int(b)) => Ok(ParsedValue::Int(a - b)),
                    (ParsedValue::Int(a), "*", ParsedValue::Int(b)) => Ok(ParsedValue::Int(a * b)),
                    (ParsedValue::Int(a), "/", ParsedValue::Int(b)) => Ok(ParsedValue::Int(a / b)),
                    (ParsedValue::Int(a), "==", ParsedValue::Int(b)) => Ok(ParsedValue::Bool(a == b)),
                    (ParsedValue::Int(a), "!=", ParsedValue::Int(b)) => Ok(ParsedValue::Bool(a != b)),
                    (ParsedValue::Int(a), "<", ParsedValue::Int(b)) => Ok(ParsedValue::Bool(a < b)),
                    (ParsedValue::Int(a), ">", ParsedValue::Int(b)) => Ok(ParsedValue::Bool(a > b)),
                    (ParsedValue::Bool(a), "&&", ParsedValue::Bool(b)) => Ok(ParsedValue::Bool(a && b)),
                    (ParsedValue::Bool(a), "||", ParsedValue::Bool(b)) => Ok(ParsedValue::Bool(a || b)),
                    _ => Err(format!("unsupported binary operation: {}", op)),
                }
            }
        }
    }

    /// Evaluate expression as string
    fn eval_expr_as_string<'a>(
        &self,
        expr: &ExprIR,
        state: &mut ParserState<'a>,
    ) -> Result<String, String> {
        let val = self.eval_expr(expr, state)?;
        match val {
            ParsedValue::Text(s) => Ok(s),
            ParsedValue::Int(i) => Ok(i.to_string()),
            ParsedValue::Float(f) => Ok(f.to_string()),
            ParsedValue::Bool(b) => Ok(b.to_string()),
            _ => Err("expected string value".to_string()),
        }
    }

    // =========================================================================
    // Field extraction helpers
    // =========================================================================

    fn get_field<'b>(&self, name: &str, fields: &'b [(String, ExprIR)]) -> Option<&'b ExprIR> {
        fields.iter().find(|(n, _)| n == name).map(|(_, e)| e)
    }

    fn get_field_as_interned<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<zyntax_typed_ast::InternedString, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        match val {
            ParsedValue::Interned(s) => Ok(s),
            ParsedValue::Text(s) => Ok(state.intern(&s)),
            _ => Err(format!("field '{}' is not a string/interned", name)),
        }
    }

    fn get_field_as_string<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<String, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        self.eval_expr_as_string(expr, state)
    }

    fn get_field_as_int<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<i64, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        match val {
            ParsedValue::Int(i) => Ok(i),
            ParsedValue::Text(s) => s.parse().map_err(|_| format!("cannot parse '{}' as int", s)),
            _ => Err(format!("field '{}' is not an integer", name)),
        }
    }

    fn get_field_as_float<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<f64, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        match val {
            ParsedValue::Float(f) => Ok(f),
            ParsedValue::Int(i) => Ok(i as f64),
            ParsedValue::Text(s) => s.parse().map_err(|_| format!("cannot parse '{}' as float", s)),
            _ => Err(format!("field '{}' is not a float", name)),
        }
    }

    fn get_field_as_bool<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<bool, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        match val {
            ParsedValue::Bool(b) => Ok(b),
            _ => Err(format!("field '{}' is not a boolean", name)),
        }
    }

    fn get_field_optional<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Option<Type>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::Optional(None) => Ok(None),
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::Type(t) => Ok(Some(t)),
                            _ => Ok(None),
                        }
                    }
                    ParsedValue::Type(t) => Ok(Some(t)),
                    ParsedValue::None => Ok(None),
                    _ => Ok(None),
                }
            }
            None => Ok(None),
        }
    }

    fn get_field_as_expr<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<TypedNode<TypedExpression>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        self.parsed_value_to_expr(val, state)
    }

    fn get_field_optional_expr<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Option<TypedNode<TypedExpression>>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::Optional(None) => Ok(None),
                    ParsedValue::Optional(Some(inner)) => {
                        let e = self.parsed_value_to_expr(*inner, state)?;
                        Ok(Some(e))
                    }
                    ParsedValue::None => Ok(None),
                    other => {
                        let e = self.parsed_value_to_expr(other, state)?;
                        Ok(Some(e))
                    }
                }
            }
            None => Ok(None),
        }
    }

    fn get_field_as_expr_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedNode<TypedExpression>>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;

        match val {
            ParsedValue::List(items) => {
                items.into_iter()
                    .map(|item| self.parsed_value_to_expr(item, state))
                    .collect()
            }
            // Handle optional fields that are None (e.g., empty argument lists)
            ParsedValue::Optional(None) | ParsedValue::None => Ok(vec![]),
            // Optional with a value - unwrap and process
            ParsedValue::Optional(Some(inner)) => {
                match *inner {
                    ParsedValue::List(items) => {
                        items.into_iter()
                            .map(|item| self.parsed_value_to_expr(item, state))
                            .collect()
                    }
                    other => {
                        let e = self.parsed_value_to_expr(other, state)?;
                        Ok(vec![e])
                    }
                }
            }
            other => {
                // Single expression becomes a list of one
                let e = self.parsed_value_to_expr(other, state)?;
                Ok(vec![e])
            }
        }
    }

    fn get_field_optional_block<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Option<TypedBlock>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::Optional(None) => Ok(None),
                    ParsedValue::Optional(Some(inner)) => {
                        let block = self.parsed_value_to_block(*inner, state)?;
                        Ok(Some(block))
                    }
                    ParsedValue::None => Ok(None),
                    ParsedValue::Block(b) => Ok(Some(b)),
                    other => {
                        let block = self.parsed_value_to_block(other, state)?;
                        Ok(Some(block))
                    }
                }
            }
            None => Ok(None),
        }
    }

    fn get_field_as_block<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<TypedBlock, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        self.parsed_value_to_block(val, state)
    }

    fn get_field_as_pattern<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<TypedNode<TypedPattern>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        self.parsed_value_to_pattern(val, state)
    }

    fn get_field_as_pattern_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedNode<TypedPattern>>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;

        match val {
            ParsedValue::List(items) => {
                items.into_iter()
                    .map(|item| self.parsed_value_to_pattern(item, state))
                    .collect()
            }
            ParsedValue::Optional(None) | ParsedValue::None => Ok(vec![]),
            ParsedValue::Optional(Some(inner)) => {
                match *inner {
                    ParsedValue::List(items) => {
                        items.into_iter()
                            .map(|item| self.parsed_value_to_pattern(item, state))
                            .collect()
                    }
                    other => {
                        let p = self.parsed_value_to_pattern(other, state)?;
                        Ok(vec![p])
                    }
                }
            }
            other => {
                let p = self.parsed_value_to_pattern(other, state)?;
                Ok(vec![p])
            }
        }
    }

    fn get_field_as_field_init_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedFieldInit>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let field_init = self.parsed_value_to_field_init(item, state)?;
                            result.push(field_init);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    other => {
                        // Single item - try to convert
                        let field_init = self.parsed_value_to_field_init(other, state)?;
                        Ok(vec![field_init])
                    }
                }
            }
            None => Ok(vec![]),
        }
    }

    fn parsed_value_to_field_init<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<TypedFieldInit, String> {
        match val {
            ParsedValue::FieldInit { name, value } => {
                let expr = self.parsed_value_to_expr(*value, state)?;
                Ok(TypedFieldInit {
                    name,
                    value: Box::new(expr),
                })
            }
            _ => Err("cannot convert value to field init".to_string()),
        }
    }

    fn get_field_as_match_arm_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedMatchArm>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;

        match val {
            ParsedValue::List(items) => {
                items.into_iter()
                    .map(|item| self.parsed_value_to_match_arm(item))
                    .collect()
            }
            ParsedValue::MatchArm(arm) => Ok(vec![arm]),
            _ => Err("cannot convert value to match arm list".to_string()),
        }
    }

    fn parsed_value_to_match_arm(&self, val: ParsedValue) -> Result<TypedMatchArm, String> {
        match val {
            ParsedValue::MatchArm(arm) => Ok(arm),
            _ => Err("cannot convert value to match arm".to_string()),
        }
    }

    fn get_field_as_decl_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedNode<TypedDeclaration>>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;

        match val {
            ParsedValue::List(items) => {
                items.into_iter()
                    .map(|item| self.parsed_value_to_decl(item))
                    .collect()
            }
            ParsedValue::Declaration(decl) => Ok(vec![*decl]),
            _ => Err(format!("field '{}' is not a declaration list", name)),
        }
    }

    /// Get a field as a list of function parameters
    fn get_field_as_param_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedParameter>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let param = self.parsed_value_to_param(item, state)?;
                            result.push(param);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::List(items) => {
                                let mut result = Vec::new();
                                for item in items {
                                    let param = self.parsed_value_to_param(item, state)?;
                                    result.push(param);
                                }
                                Ok(result)
                            }
                            other => {
                                let param = self.parsed_value_to_param(other, state)?;
                                Ok(vec![param])
                            }
                        }
                    }
                    ParsedValue::Parameter(p) => Ok(vec![p]),
                    other => {
                        // Single item - try to convert
                        let param = self.parsed_value_to_param(other, state)?;
                        Ok(vec![param])
                    }
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Convert ParsedValue to TypedParameter
    fn parsed_value_to_param<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<TypedParameter, String> {
        match val {
            ParsedValue::Parameter(p) => Ok(p),
            ParsedValue::FieldInit { name, value } => {
                // Convert FieldInit to a parameter (name: Type format)
                let ty = match *value {
                    ParsedValue::Type(t) => t,
                    _ => Type::Any,
                };
                Ok(TypedParameter {
                    name,
                    ty,
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::new(0, 0),
                })
            }
            ParsedValue::Interned(name) => {
                // Just a name, no type annotation
                Ok(TypedParameter {
                    name,
                    ty: Type::Any,
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::new(0, 0),
                })
            }
            ParsedValue::Text(name) => {
                let interned = state.intern(&name);
                Ok(TypedParameter {
                    name: interned,
                    ty: Type::Any,
                    mutability: Mutability::Immutable,
                    kind: ParameterKind::Regular,
                    default_value: None,
                    attributes: vec![],
                    span: Span::new(0, 0),
                })
            }
            _ => Err("cannot convert value to parameter".to_string()),
        }
    }

    /// Get a field as a list of interned strings
    fn get_field_as_interned_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<zyntax_typed_ast::InternedString>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let interned = self.parsed_value_to_interned(item, state)?;
                            result.push(interned);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::Optional(Some(inner)) => {
                        let interned = self.parsed_value_to_interned(*inner, state)?;
                        Ok(vec![interned])
                    }
                    ParsedValue::Interned(i) => Ok(vec![i]),
                    ParsedValue::Text(s) => Ok(vec![state.intern(&s)]),
                    _ => Err(format!("field '{}' is not an interned string list", name)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Convert ParsedValue to InternedString
    fn parsed_value_to_interned<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<zyntax_typed_ast::InternedString, String> {
        match val {
            ParsedValue::Interned(i) => Ok(i),
            ParsedValue::Text(s) => Ok(state.intern(&s)),
            _ => Err("cannot convert value to interned string".to_string()),
        }
    }

    /// Get a field as a list of enum variants
    fn get_field_as_variant_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedVariant>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let variant = self.parsed_value_to_variant(item, state)?;
                            result.push(variant);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::List(items) => {
                                let mut result = Vec::new();
                                for item in items {
                                    let variant = self.parsed_value_to_variant(item, state)?;
                                    result.push(variant);
                                }
                                Ok(result)
                            }
                            other => {
                                let variant = self.parsed_value_to_variant(other, state)?;
                                Ok(vec![variant])
                            }
                        }
                    }
                    ParsedValue::Variant(v) => Ok(vec![v]),
                    other => {
                        let variant = self.parsed_value_to_variant(other, state)?;
                        Ok(vec![variant])
                    }
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Convert ParsedValue to TypedVariant
    fn parsed_value_to_variant<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<TypedVariant, String> {
        match val {
            ParsedValue::Variant(v) => Ok(v),
            ParsedValue::Interned(name) => {
                // Simple unit variant (just a name)
                Ok(TypedVariant {
                    name,
                    fields: TypedVariantFields::Unit,
                    discriminant: None,
                    span: Span::new(0, 0),
                })
            }
            ParsedValue::Text(name) => {
                let interned = state.intern(&name);
                Ok(TypedVariant {
                    name: interned,
                    fields: TypedVariantFields::Unit,
                    discriminant: None,
                    span: Span::new(0, 0),
                })
            }
            _ => Err("cannot convert value to variant".to_string()),
        }
    }

    /// Get a field as a list of struct fields
    fn get_field_as_field_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<zyntax_typed_ast::TypedField>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let field = self.parsed_value_to_field(item, state)?;
                            result.push(field);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::List(items) => {
                                let mut result = Vec::new();
                                for item in items {
                                    let field = self.parsed_value_to_field(item, state)?;
                                    result.push(field);
                                }
                                Ok(result)
                            }
                            other => {
                                let field = self.parsed_value_to_field(other, state)?;
                                Ok(vec![field])
                            }
                        }
                    }
                    ParsedValue::Field(f) => Ok(vec![f]),
                    other => {
                        let field = self.parsed_value_to_field(other, state)?;
                        Ok(vec![field])
                    }
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Get a field as a list of declarations (for impl items)
    fn get_field_optional_decl_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedNode<TypedDeclaration>>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            if let Ok(decl) = self.parsed_value_to_decl(item) {
                                result.push(decl);
                            }
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::Optional(Some(inner)) => {
                        match *inner {
                            ParsedValue::List(items) => {
                                let mut result = Vec::new();
                                for item in items {
                                    if let Ok(decl) = self.parsed_value_to_decl(item) {
                                        result.push(decl);
                                    }
                                }
                                Ok(result)
                            }
                            other => {
                                if let Ok(decl) = self.parsed_value_to_decl(other) {
                                    Ok(vec![decl])
                                } else {
                                    Ok(vec![])
                                }
                            }
                        }
                    }
                    ParsedValue::Declaration(d) => Ok(vec![*d]),
                    other => {
                        if let Ok(decl) = self.parsed_value_to_decl(other) {
                            Ok(vec![decl])
                        } else {
                            Ok(vec![])
                        }
                    }
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Convert ParsedValue to TypedField
    fn parsed_value_to_field<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<zyntax_typed_ast::TypedField, String> {
        match val {
            ParsedValue::Field(f) => Ok(f),
            _ => Err(format!("cannot convert value to field: {:?}", val)),
        }
    }

    /// Convert ParsedValue to TypedExpression
    fn parsed_value_to_expr<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<TypedNode<TypedExpression>, String> {
        let span = Span::new(0, 0);
        match val {
            ParsedValue::Expression(e) => Ok(*e),
            ParsedValue::Int(i) => Ok(typed_node(
                TypedExpression::Literal(TypedLiteral::Integer(i as i128)),
                Type::Primitive(PrimitiveType::I32),
                span,
            )),
            ParsedValue::Float(f) => Ok(typed_node(
                TypedExpression::Literal(TypedLiteral::Float(f)),
                Type::Primitive(PrimitiveType::F32),
                span,
            )),
            ParsedValue::Text(s) => {
                // Check if it looks like a string literal (quoted)
                if s.starts_with('"') && s.ends_with('"') {
                    let unquoted = s.trim_matches('"').to_string();
                    let interned = state.intern(&unquoted);
                    Ok(typed_node(
                        TypedExpression::Literal(TypedLiteral::String(interned)),
                        Type::Primitive(PrimitiveType::String),
                        span,
                    ))
                } else {
                    // Treat as variable reference
                    let interned = state.intern(&s);
                    Ok(typed_node(
                        TypedExpression::Variable(interned),
                        Type::Primitive(PrimitiveType::Unit), // Unknown type
                        span,
                    ))
                }
            }
            ParsedValue::Bool(b) => Ok(typed_node(
                TypedExpression::Literal(TypedLiteral::Bool(b)),
                Type::Primitive(PrimitiveType::Bool),
                span,
            )),
            ParsedValue::Interned(s) => Ok(typed_node(
                TypedExpression::Variable(s),
                Type::Primitive(PrimitiveType::Unit),
                span,
            )),
            _ => Err("cannot convert value to expression".to_string()),
        }
    }

    /// Convert ParsedValue to TypedBlock
    fn parsed_value_to_block<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<TypedBlock, String> {
        match val {
            ParsedValue::Block(b) => Ok(b),
            ParsedValue::List(items) => {
                let stmts: Result<Vec<_>, _> = items.into_iter()
                    .map(|item| self.parsed_value_to_stmt(item))
                    .collect();
                Ok(TypedBlock {
                    statements: stmts?,
                    span: Span::new(0, 0),
                })
            }
            ParsedValue::Statement(s) => Ok(TypedBlock {
                statements: vec![*s],
                span: Span::new(0, 0),
            }),
            _ => Err("cannot convert value to block".to_string()),
        }
    }

    /// Convert ParsedValue to TypedStatement
    fn parsed_value_to_stmt(&self, val: ParsedValue) -> Result<TypedNode<TypedStatement>, String> {
        match val {
            ParsedValue::Statement(s) => Ok(*s),
            _ => Err("cannot convert value to statement".to_string()),
        }
    }

    /// Convert ParsedValue to TypedPattern
    fn parsed_value_to_pattern<'a>(&self, val: ParsedValue, _state: &mut ParserState<'a>) -> Result<TypedNode<TypedPattern>, String> {
        match val {
            ParsedValue::Pattern(p) => Ok(*p),
            _ => Err("cannot convert value to pattern".to_string()),
        }
    }

    /// Convert ParsedValue to TypedDeclaration
    fn parsed_value_to_decl(&self, val: ParsedValue) -> Result<TypedNode<TypedDeclaration>, String> {
        match val {
            ParsedValue::Declaration(d) => Ok(*d),
            ParsedValue::Statement(s) => {
                // Try to extract declaration from expression statement
                log::warn!("Got Statement when expecting Declaration, trying to convert. Statement: {:?}", s.node);
                Err("cannot convert value to declaration".to_string())
            }
            other => {
                log::error!("Cannot convert value to declaration. Got: {:?}", std::mem::discriminant(&other));
                Err("cannot convert value to declaration".to_string())
            }
        }
    }

    /// Get a field as a TypedDeclaration
    fn get_field_as_declaration<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<TypedNode<TypedDeclaration>, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        self.parsed_value_to_decl(val)
    }

    /// Get a field as a list of TypedAnnotation
    fn get_field_as_annotation_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedAnnotation>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let ann = self.parsed_value_to_annotation(item)?;
                            result.push(ann);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::Annotation(a) => Ok(vec![a]),
                    other => Err(format!("field '{}' is not an annotation list: {:?}", name, other)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Get a field as a list of TypedAnnotationArg
    fn get_field_as_annotation_arg_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedAnnotationArg>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let arg = self.parsed_value_to_annotation_arg(item)?;
                            result.push(arg);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::AnnotationArg(a) => Ok(vec![a]),
                    other => Err(format!("field '{}' is not an annotation arg list: {:?}", name, other)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Get a field as a TypedAnnotationValue
    fn get_field_as_annotation_value<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<TypedAnnotationValue, String> {
        let expr = self.get_field(name, fields)
            .ok_or_else(|| format!("missing field: {}", name))?;
        let val = self.eval_expr(expr, state)?;
        self.parsed_value_to_annotation_value(val)
    }

    /// Get a field as a list of TypedAnnotationValue
    fn get_field_as_annotation_value_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedAnnotationValue>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let v = self.parsed_value_to_annotation_value(item)?;
                            result.push(v);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::AnnotationValue(v) => Ok(vec![v]),
                    other => Err(format!("field '{}' is not an annotation value list: {:?}", name, other)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Convert ParsedValue to TypedAnnotation
    fn parsed_value_to_annotation(&self, val: ParsedValue) -> Result<TypedAnnotation, String> {
        match val {
            ParsedValue::Annotation(a) => Ok(a),
            _ => Err(format!("cannot convert value to annotation: {:?}", val)),
        }
    }

    /// Convert ParsedValue to TypedAnnotationArg
    fn parsed_value_to_annotation_arg(&self, val: ParsedValue) -> Result<TypedAnnotationArg, String> {
        match val {
            ParsedValue::AnnotationArg(a) => Ok(a),
            _ => Err(format!("cannot convert value to annotation arg: {:?}", val)),
        }
    }

    /// Convert ParsedValue to TypedAnnotationValue
    fn parsed_value_to_annotation_value(&self, val: ParsedValue) -> Result<TypedAnnotationValue, String> {
        match val {
            ParsedValue::AnnotationValue(v) => Ok(v),
            // Allow direct conversion from primitives
            ParsedValue::Int(i) => Ok(TypedAnnotationValue::Integer(i)),
            ParsedValue::Bool(b) => Ok(TypedAnnotationValue::Bool(b)),
            ParsedValue::Float(f) => Ok(TypedAnnotationValue::Float(f)),
            ParsedValue::Text(s) => Ok(TypedAnnotationValue::String(zyntax_typed_ast::InternedString::new_global(&s))),
            ParsedValue::Interned(s) => Ok(TypedAnnotationValue::Identifier(s)),
            _ => Err(format!("cannot convert value to annotation value: {:?}", val)),
        }
    }

    /// Get a field as a list of TypedEffectOp
    fn get_field_as_effect_op_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<zyntax_typed_ast::TypedEffectOp>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let op = self.parsed_value_to_effect_op(item)?;
                            result.push(op);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::EffectOp(op) => Ok(vec![op]),
                    other => Err(format!("field '{}' is not an effect op list: {:?}", name, other)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Get a field as a list of TypedEffectHandlerImpl
    fn get_field_as_handler_impl_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<zyntax_typed_ast::TypedEffectHandlerImpl>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let impl_ = self.parsed_value_to_handler_impl(item)?;
                            result.push(impl_);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::EffectHandlerImpl(impl_) => Ok(vec![impl_]),
                    other => Err(format!("field '{}' is not a handler impl list: {:?}", name, other)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Get a field as a list of TypedTypeParam
    fn get_field_as_type_param_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedTypeParam>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let interned = self.parsed_value_to_interned(item, state)?;
                            result.push(TypedTypeParam {
                                name: interned,
                                bounds: vec![],
                                default: None,
                                span: Span::default(),
                            });
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::Interned(i) => Ok(vec![TypedTypeParam {
                        name: i,
                        bounds: vec![],
                        default: None,
                        span: Span::default(),
                    }]),
                    ParsedValue::Text(s) => Ok(vec![TypedTypeParam {
                        name: state.intern(&s),
                        bounds: vec![],
                        default: None,
                        span: Span::default(),
                    }]),
                    _ => Err(format!("field '{}' is not a type param list", name)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Get a field as a list of lambda parameters
    fn get_field_as_lambda_param_list<'a>(
        &self,
        name: &str,
        fields: &[(String, ExprIR)],
        state: &mut ParserState<'a>,
    ) -> Result<Vec<TypedLambdaParam>, String> {
        match self.get_field(name, fields) {
            Some(expr) => {
                let val = self.eval_expr(expr, state)?;
                match val {
                    ParsedValue::List(items) => {
                        let mut result = Vec::new();
                        for item in items {
                            let param = self.parsed_value_to_lambda_param(item, state)?;
                            result.push(param);
                        }
                        Ok(result)
                    }
                    ParsedValue::None => Ok(vec![]),
                    ParsedValue::Optional(None) => Ok(vec![]),
                    ParsedValue::LambdaParam(p) => Ok(vec![p]),
                    ParsedValue::Interned(i) => Ok(vec![TypedLambdaParam {
                        name: i,
                        ty: None,
                    }]),
                    ParsedValue::Text(s) => Ok(vec![TypedLambdaParam {
                        name: state.intern(&s),
                        ty: None,
                    }]),
                    _ => Err(format!("field '{}' is not a lambda param list: {:?}", name, val)),
                }
            }
            None => Ok(vec![]),
        }
    }

    /// Convert ParsedValue to TypedLambdaParam
    fn parsed_value_to_lambda_param<'a>(
        &self,
        val: ParsedValue,
        state: &mut ParserState<'a>,
    ) -> Result<TypedLambdaParam, String> {
        match val {
            ParsedValue::LambdaParam(p) => Ok(p),
            ParsedValue::Interned(i) => Ok(TypedLambdaParam {
                name: i,
                ty: None,
            }),
            ParsedValue::Text(s) => Ok(TypedLambdaParam {
                name: state.intern(&s),
                ty: None,
            }),
            _ => Err(format!("cannot convert value to lambda param: {:?}", val)),
        }
    }

    /// Convert ParsedValue to TypedEffectOp
    fn parsed_value_to_effect_op(&self, val: ParsedValue) -> Result<zyntax_typed_ast::TypedEffectOp, String> {
        match val {
            ParsedValue::EffectOp(op) => Ok(op),
            _ => Err(format!("cannot convert value to effect op: {:?}", val)),
        }
    }

    /// Convert ParsedValue to TypedEffectHandlerImpl
    fn parsed_value_to_handler_impl(&self, val: ParsedValue) -> Result<zyntax_typed_ast::TypedEffectHandlerImpl, String> {
        match val {
            ParsedValue::EffectHandlerImpl(impl_) => Ok(impl_),
            _ => Err(format!("cannot convert value to handler impl: {:?}", val)),
        }
    }

    /// Convert string to BinaryOp
    fn string_to_binary_op(&self, op: &str) -> Result<zyntax_typed_ast::BinaryOp, String> {
        match op {
            "+" | "Add" => Ok(zyntax_typed_ast::BinaryOp::Add),
            "-" | "Sub" => Ok(zyntax_typed_ast::BinaryOp::Sub),
            "*" | "Mul" => Ok(zyntax_typed_ast::BinaryOp::Mul),
            "/" | "Div" => Ok(zyntax_typed_ast::BinaryOp::Div),
            "%" | "Rem" => Ok(zyntax_typed_ast::BinaryOp::Rem),
            "==" | "Eq" => Ok(zyntax_typed_ast::BinaryOp::Eq),
            "!=" | "Ne" => Ok(zyntax_typed_ast::BinaryOp::Ne),
            "<" | "Lt" => Ok(zyntax_typed_ast::BinaryOp::Lt),
            "<=" | "Le" => Ok(zyntax_typed_ast::BinaryOp::Le),
            ">" | "Gt" => Ok(zyntax_typed_ast::BinaryOp::Gt),
            ">=" | "Ge" => Ok(zyntax_typed_ast::BinaryOp::Ge),
            "&&" | "And" => Ok(zyntax_typed_ast::BinaryOp::And),
            "||" | "Or" => Ok(zyntax_typed_ast::BinaryOp::Or),
            "=" | "Assign" => Ok(zyntax_typed_ast::BinaryOp::Assign),
            "??" | "Orelse" => Ok(zyntax_typed_ast::BinaryOp::Orelse),
            "catch" | "Catch" => Ok(zyntax_typed_ast::BinaryOp::Catch),
            "&" | "BitAnd" => Ok(zyntax_typed_ast::BinaryOp::BitAnd),
            "|" | "BitOr" => Ok(zyntax_typed_ast::BinaryOp::BitOr),
            "^" | "BitXor" => Ok(zyntax_typed_ast::BinaryOp::BitXor),
            "<<" | "Shl" => Ok(zyntax_typed_ast::BinaryOp::Shl),
            ">>" | "Shr" => Ok(zyntax_typed_ast::BinaryOp::Shr),
            _ => Err(format!("unknown binary operator: {}", op)),
        }
    }

    /// Convert string to UnaryOp
    fn string_to_unary_op(&self, op: &str) -> Result<UnaryOp, String> {
        match op {
            "-" | "Minus" | "Neg" => Ok(UnaryOp::Minus),
            "!" | "Not" => Ok(UnaryOp::Not),
            "+" | "Plus" => Ok(UnaryOp::Plus),
            "~" | "BitNot" => Ok(UnaryOp::BitNot),
            _ => Err(format!("unknown unary operator: {}", op)),
        }
    }

    /// Execute a pattern
    fn execute_pattern<'a>(
        &self,
        pattern: &PatternIR,
        state: &mut ParserState<'a>,
        atomic: bool,
    ) -> ParseResult<ParsedValue> {
        match pattern {
            PatternIR::Literal(s) => {
                match state.match_literal(s) {
                    ParseResult::Success(_, pos) => ParseResult::Success(ParsedValue::None, pos),
                    ParseResult::Failure(e) => ParseResult::Failure(e),
                }
            }

            PatternIR::CharClass(class) => {
                self.execute_char_class(class, state)
            }

            PatternIR::RuleRef { rule_name, binding } => {
                // Check for built-in patterns first
                let result = match rule_name.as_str() {
                    "ASCII_DIGIT" => {
                        state.match_char(|c| c.is_ascii_digit(), "digit")
                            .map(|c| ParsedValue::Text(c.to_string()))
                    }
                    "ASCII_ALPHA" => {
                        state.match_char(|c| c.is_ascii_alphabetic(), "letter")
                            .map(|c| ParsedValue::Text(c.to_string()))
                    }
                    "ASCII_ALPHANUMERIC" => {
                        state.match_char(|c| c.is_ascii_alphanumeric(), "alphanumeric")
                            .map(|c| ParsedValue::Text(c.to_string()))
                    }
                    "ASCII_HEX_DIGIT" => {
                        state.match_char(|c| c.is_ascii_hexdigit(), "hex digit")
                            .map(|c| ParsedValue::Text(c.to_string()))
                    }
                    "ANY" => {
                        match state.peek_char() {
                            Some(c) => {
                                state.advance();
                                ParseResult::Success(ParsedValue::Text(c.to_string()), state.pos())
                            }
                            None => state.fail("any character"),
                        }
                    }
                    "SOI" => {
                        if state.pos() == 0 {
                            ParseResult::Success(ParsedValue::None, 0)
                        } else {
                            state.fail("start of input")
                        }
                    }
                    "EOI" => {
                        if state.is_eof() {
                            ParseResult::Success(ParsedValue::None, state.pos())
                        } else {
                            state.fail("end of input")
                        }
                    }
                    _ => {
                        // Look up rule in grammar
                        match self.grammar.get_rule(rule_name) {
                            Some(rule) => self.execute_rule(rule, state),
                            None => state.fail(&format!("unknown rule: {}", rule_name)),
                        }
                    }
                };

                // Store binding if specified
                if let Some(bind_name) = binding {
                    if let ParseResult::Success(ref value, _) = result {
                        state.set_binding(bind_name, value.clone());
                    }
                }

                result
            }

            PatternIR::Sequence(patterns) => {
                let start_pos = state.pos();
                let mut last_value = ParsedValue::None;

                for (i, p) in patterns.iter().enumerate() {
                    // Skip whitespace between elements (unless atomic)
                    if !atomic && i > 0 {
                        state.skip_ws();
                    }

                    match self.execute_pattern(p, state, atomic) {
                        ParseResult::Success(v, _) => {
                            last_value = v;
                        }
                        ParseResult::Failure(e) => {
                            state.set_pos(start_pos);
                            return ParseResult::Failure(e);
                        }
                    }
                }

                ParseResult::Success(last_value, state.pos())
            }

            PatternIR::Choice(choices) => {
                let start_pos = state.pos();
                let saved_bindings = state.save_bindings();
                let mut last_error: Option<ParseFailure> = None;

                for choice in choices {
                    state.set_pos(start_pos);
                    state.restore_bindings(saved_bindings.clone());

                    match self.execute_pattern(choice, state, atomic) {
                        ParseResult::Success(v, pos) => {
                            return ParseResult::Success(v, pos);
                        }
                        ParseResult::Failure(e) => {
                            last_error = Some(match last_error {
                                Some(prev) => prev.merge(e),
                                None => e,
                            });
                        }
                    }
                }

                state.set_pos(start_pos);
                state.restore_bindings(saved_bindings);
                ParseResult::Failure(last_error.unwrap_or_else(|| {
                    ParseFailure::new("choice", start_pos, state.line(), state.column())
                }))
            }

            PatternIR::Optional(inner) => {
                let start_pos = state.pos();
                let saved_bindings = state.save_bindings();

                // Extract binding name from inner pattern if it's a bound RuleRef
                let inner_binding = match inner.as_ref() {
                    PatternIR::RuleRef { binding: Some(name), .. } => Some(name.clone()),
                    _ => None,
                };

                match self.execute_pattern(inner, state, atomic) {
                    ParseResult::Success(v, pos) => {
                        ParseResult::Success(ParsedValue::Optional(Some(Box::new(v))), pos)
                    }
                    ParseResult::Failure(_) => {
                        state.set_pos(start_pos);
                        state.restore_bindings(saved_bindings);
                        // If there was a binding on the inner pattern, set it to Optional(None)
                        // so the action can access it
                        if let Some(bind_name) = inner_binding {
                            state.set_binding(&bind_name, ParsedValue::Optional(None));
                        }
                        ParseResult::Success(ParsedValue::Optional(None), start_pos)
                    }
                }
            }

            PatternIR::Repeat { pattern, min, max, separator } => {
                let mut items = Vec::new();

                // Extract binding name from inner pattern if it's a bound RuleRef
                let inner_binding = match pattern.as_ref() {
                    PatternIR::RuleRef { binding: Some(name), .. } => Some(name.clone()),
                    _ => None,
                };

                // Track accumulated values for the inner binding
                let mut accumulated_bindings: HashMap<String, Vec<ParsedValue>> = HashMap::new();

                loop {
                    // Skip whitespace before each item (including the first)
                    if !atomic {
                        state.skip_ws();
                    }

                    let item_start = state.pos();
                    let saved_bindings = state.save_bindings();

                    // Handle separator (after first item)
                    if !items.is_empty() {
                        if let Some(sep) = separator {
                            match self.execute_pattern(sep, state, atomic) {
                                ParseResult::Success(_, _) => {
                                    if !atomic {
                                        state.skip_ws();
                                    }
                                }
                                ParseResult::Failure(_) => {
                                    state.set_pos(item_start);
                                    state.restore_bindings(saved_bindings);
                                    break;
                                }
                            }
                        }
                    }

                    // Try to match item
                    match self.execute_pattern(pattern, state, atomic) {
                        ParseResult::Success(v, _) => {
                            items.push(v.clone());

                            // If there's an inner binding, accumulate its value
                            if let Some(ref bind_name) = inner_binding {
                                accumulated_bindings
                                    .entry(bind_name.clone())
                                    .or_default()
                                    .push(v);
                            }
                        }
                        ParseResult::Failure(_) => {
                            state.set_pos(item_start);
                            state.restore_bindings(saved_bindings);
                            break;
                        }
                    }

                    // Check max
                    if let Some(max_count) = max {
                        if items.len() >= *max_count {
                            break;
                        }
                    }
                }

                // Check min
                if items.len() < *min {
                    return state.fail(&format!("expected at least {} items, got {}", min, items.len()));
                }

                // Set accumulated bindings as lists
                // Even if empty, we need to set the binding so actions can reference it
                if let Some(ref bind_name) = inner_binding {
                    let values = accumulated_bindings
                        .remove(bind_name)
                        .unwrap_or_default();
                    state.set_binding(bind_name, ParsedValue::List(values));
                }
                // Set any remaining bindings (shouldn't normally happen, but for safety)
                for (name, values) in accumulated_bindings {
                    state.set_binding(&name, ParsedValue::List(values));
                }

                ParseResult::Success(ParsedValue::List(items), state.pos())
            }

            PatternIR::PositiveLookahead(inner) => {
                let start_pos = state.pos();
                let saved_bindings = state.save_bindings();

                let result = self.execute_pattern(inner, state, atomic);

                state.set_pos(start_pos);
                state.restore_bindings(saved_bindings);

                match result {
                    ParseResult::Success(_, _) => ParseResult::Success(ParsedValue::None, start_pos),
                    ParseResult::Failure(e) => ParseResult::Failure(e),
                }
            }

            PatternIR::NegativeLookahead(inner) => {
                let start_pos = state.pos();
                let saved_bindings = state.save_bindings();

                let result = self.execute_pattern(inner, state, atomic);

                state.set_pos(start_pos);
                state.restore_bindings(saved_bindings);

                match result {
                    ParseResult::Success(_, _) => state.fail("negative lookahead matched"),
                    ParseResult::Failure(_) => ParseResult::Success(ParsedValue::None, start_pos),
                }
            }

            PatternIR::Any => {
                match state.peek_char() {
                    Some(c) => {
                        state.advance();
                        ParseResult::Success(ParsedValue::Text(c.to_string()), state.pos())
                    }
                    None => state.fail("any character"),
                }
            }

            PatternIR::StartOfInput => {
                if state.pos() == 0 {
                    ParseResult::Success(ParsedValue::None, 0)
                } else {
                    state.fail("start of input")
                }
            }

            PatternIR::EndOfInput => {
                if state.is_eof() {
                    ParseResult::Success(ParsedValue::None, state.pos())
                } else {
                    state.fail("end of input")
                }
            }

            PatternIR::Whitespace => {
                state.skip_ws();
                ParseResult::Success(ParsedValue::None, state.pos())
            }
        }
    }

    /// Execute a character class pattern
    fn execute_char_class<'a>(
        &self,
        class: &CharClass,
        state: &mut ParserState<'a>,
    ) -> ParseResult<ParsedValue> {
        match class {
            CharClass::Single(expected) => {
                match state.peek_char() {
                    Some(c) if c == *expected => {
                        state.advance();
                        ParseResult::Success(ParsedValue::Text(c.to_string()), state.pos())
                    }
                    _ => state.fail(&format!("'{}'", expected)),
                }
            }

            CharClass::Range(start, end) => {
                match state.peek_char() {
                    Some(c) if c >= *start && c <= *end => {
                        state.advance();
                        ParseResult::Success(ParsedValue::Text(c.to_string()), state.pos())
                    }
                    _ => state.fail(&format!("'{}'..'{}'", start, end)),
                }
            }

            CharClass::Builtin(name) => {
                let pred: Box<dyn Fn(char) -> bool> = match name.as_str() {
                    "ASCII_DIGIT" => Box::new(|c: char| c.is_ascii_digit()),
                    "ASCII_ALPHA" => Box::new(|c: char| c.is_ascii_alphabetic()),
                    "ASCII_ALPHANUMERIC" => Box::new(|c: char| c.is_ascii_alphanumeric()),
                    "ASCII_HEX_DIGIT" => Box::new(|c: char| c.is_ascii_hexdigit()),
                    "NEWLINE" => Box::new(|c: char| c == '\n' || c == '\r'),
                    _ => return state.fail(&format!("unknown char class: {}", name)),
                };

                match state.peek_char() {
                    Some(c) if pred(c) => {
                        state.advance();
                        ParseResult::Success(ParsedValue::Text(c.to_string()), state.pos())
                    }
                    _ => state.fail(name),
                }
            }

            CharClass::Union(classes) => {
                let start_pos = state.pos();

                for class in classes {
                    state.set_pos(start_pos);
                    match self.execute_char_class(class, state) {
                        ParseResult::Success(v, pos) => {
                            return ParseResult::Success(v, pos);
                        }
                        ParseResult::Failure(_) => continue,
                    }
                }

                state.set_pos(start_pos);
                state.fail("character class union")
            }

            CharClass::Negation(inner) => {
                let start_pos = state.pos();

                match self.execute_char_class(inner, state) {
                    ParseResult::Success(_, _) => {
                        state.set_pos(start_pos);
                        state.fail("negated character class matched")
                    }
                    ParseResult::Failure(_) => {
                        state.set_pos(start_pos);
                        // Match any character instead
                        match state.peek_char() {
                            Some(c) => {
                                state.advance();
                                ParseResult::Success(ParsedValue::Text(c.to_string()), state.pos())
                            }
                            None => state.fail("any character (negation)"),
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parser::parse_grammar;
    use zyntax_typed_ast::TypedASTBuilder;
    use zyntax_typed_ast::type_registry::TypeRegistry;

    #[test]
    fn test_interpret_literal() {
        let grammar_src = r#"
            @language { name: "Test", version: "1.0" }
            hello = { "hello" }
        "#;
        let grammar = parse_grammar(grammar_src).unwrap();
        let interp = GrammarInterpreter::new(&grammar);

        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("hello world", &mut builder, &mut registry);

        let result = interp.parse_rule("hello", &mut state);
        assert!(result.is_success());
        assert_eq!(state.pos(), 5);
    }

    #[test]
    fn test_interpret_sequence() {
        let grammar_src = r#"
            @language { name: "Test", version: "1.0" }
            greeting = { "hello" ~ "world" }
        "#;
        let grammar = parse_grammar(grammar_src).unwrap();
        let interp = GrammarInterpreter::new(&grammar);

        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("hello world", &mut builder, &mut registry);

        let result = interp.parse_rule("greeting", &mut state);
        assert!(result.is_success());
    }

    #[test]
    fn test_interpret_choice() {
        let grammar_src = r#"
            @language { name: "Test", version: "1.0" }
            word = { "hello" | "world" }
        "#;
        let grammar = parse_grammar(grammar_src).unwrap();
        let interp = GrammarInterpreter::new(&grammar);

        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();

        let mut state1 = ParserState::new("hello", &mut builder, &mut registry);
        assert!(interp.parse_rule("word", &mut state1).is_success());

        let mut state2 = ParserState::new("world", &mut builder, &mut registry);
        assert!(interp.parse_rule("word", &mut state2).is_success());
    }

    #[test]
    fn test_interpret_repeat() {
        let grammar_src = r#"
            @language { name: "Test", version: "1.0" }
            digits = @{ ASCII_DIGIT+ }
        "#;
        let grammar = parse_grammar(grammar_src).unwrap();
        let interp = GrammarInterpreter::new(&grammar);

        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("12345abc", &mut builder, &mut registry);

        let result = interp.parse_rule("digits", &mut state);
        match result {
            ParseResult::Success(ParsedValue::Text(s), _) => {
                assert_eq!(s, "12345");
            }
            _ => panic!("Expected text result"),
        }
    }

    #[test]
    fn test_interpret_optional() {
        let grammar_src = r#"
            @language { name: "Test", version: "1.0" }
            maybe_hello = { "hello"? ~ "world" }
        "#;
        let grammar = parse_grammar(grammar_src).unwrap();
        let interp = GrammarInterpreter::new(&grammar);

        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();

        // With optional
        let mut state1 = ParserState::new("hello world", &mut builder, &mut registry);
        assert!(interp.parse_rule("maybe_hello", &mut state1).is_success());

        // Without optional
        let mut state2 = ParserState::new("world", &mut builder, &mut registry);
        assert!(interp.parse_rule("maybe_hello", &mut state2).is_success());
    }

    #[test]
    fn test_interpret_identifier() {
        let grammar_src = r#"
            @language { name: "Test", version: "1.0" }
            identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
        "#;
        let grammar = parse_grammar(grammar_src).unwrap();
        let interp = GrammarInterpreter::new(&grammar);

        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("my_var_123 rest", &mut builder, &mut registry);

        let result = interp.parse_rule("identifier", &mut state);
        match result {
            ParseResult::Success(ParsedValue::Text(s), _) => {
                assert_eq!(s, "my_var_123");
            }
            _ => panic!("Expected text result"),
        }
    }

    #[test]
    fn test_interpret_rule_reference() {
        let grammar_src = r#"
            @language { name: "Test", version: "1.0" }
            greeting = { "hello" ~ name }
            name = @{ ASCII_ALPHA+ }
        "#;
        let grammar = parse_grammar(grammar_src).unwrap();
        let interp = GrammarInterpreter::new(&grammar);

        let mut builder = TypedASTBuilder::new();
        let mut registry = TypeRegistry::new();
        let mut state = ParserState::new("hello world", &mut builder, &mut registry);

        let result = interp.parse_rule("greeting", &mut state);
        assert!(result.is_success());
    }
}
