//! Statement-list flatten regression tests.
//!
//! When a single grammar rule emits multiple statements as a list
//! (`concat_list(...)` / `prepend_list(...)`), the parent block's
//! statement collector must flatten the nested list rather than
//! handing it to `parsed_value_to_stmt` verbatim. Without the
//! flatten, the parent fails with "cannot convert value to
//! statement" because a `ParsedValue::List(...)` is not a
//! statement.
//!
//! The flatten lives in two spots:
//! - `construct_block` (the in-action `TypedBlock { statements:
//!   stmts }` builder)
//! - `parsed_value_to_block` (the value-to-block coercion used
//!   when a function body field receives a list directly)
//!
//! These tests pin both paths against the multi-statement-from-
//! one-rule shape that DSLs like Blinc's `slot <name> { ... }`
//! rely on (the slot emits `[open_marker, body..., close_marker]`
//! from one statement-position alternate).

use zyn_peg::grammar::parse_grammar;
use zyn_peg::runtime2::{GrammarInterpreter, ParseResult, ParsedValue, ParserState};
use zyntax_typed_ast::type_registry::TypeRegistry;
use zyntax_typed_ast::{TypedASTBuilder, TypedExpression, TypedStatement};

/// `TypedBlock { statements: stmts }` where one of the items in
/// `stmts` is itself a `List<Statement>` from a sibling rule.
/// The flatten in `construct_block` should expand the nested list
/// in place.
#[test]
fn construct_block_flattens_nested_statement_list() {
    let grammar_src = r#"
        @language { name: "FlattenTest", version: "1.0" }

        program = { SOI ~ b:block ~ EOI }
          -> b

        // A statement-position rule whose action returns a List
        // of two statements from one source token.
        pair_stmt = { "pair" }
          -> concat_list(
              [TypedStatement::Expression {
                  expr: TypedExpression::Variable { name: intern("a") },
              }],
              [TypedStatement::Expression {
                  expr: TypedExpression::Variable { name: intern("b") },
              }],
          )

        single_stmt = { name:identifier }
          -> TypedStatement::Expression {
              expr: TypedExpression::Variable { name: intern(name) },
          }

        stmt = { pair_stmt | single_stmt }

        block = { "{" ~ stmts:stmt* ~ "}" }
          -> TypedBlock { statements: stmts }

        identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }

        WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
    "#;

    let grammar = parse_grammar(grammar_src).expect("grammar parses");
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    // `{ x pair y }` — should produce statements [x, a, b, y]
    // after flattening the inner pair.
    let mut state = ParserState::new("{ x pair y }", &mut builder, &mut registry);

    let result = interp.parse_rule("program", &mut state);
    let ParseResult::Success(value, _pos) = result else {
        panic!("expected parse success, got {result:?}");
    };

    let ParsedValue::Block(block) = value else {
        panic!("expected ParsedValue::Block, got {value:?}");
    };

    assert_eq!(
        block.statements.len(),
        4,
        "expected 4 statements after flatten (x, a, b, y), got {}: {:?}",
        block.statements.len(),
        block
            .statements
            .iter()
            .map(|s| match &s.node {
                TypedStatement::Expression(e) => match &e.node {
                    TypedExpression::Variable(name) => name.resolve_global(),
                    _ => None,
                },
                _ => None,
            })
            .collect::<Vec<_>>(),
    );

    // Spot-check ordering: source `x pair y` flattens to `[x, a, b, y]`.
    let names: Vec<String> = block
        .statements
        .iter()
        .filter_map(|s| {
            let TypedStatement::Expression(e) = &s.node else {
                return None;
            };
            let TypedExpression::Variable(name) = &e.node else {
                return None;
            };
            name.resolve_global().map(|s| s.to_string())
        })
        .collect();
    assert_eq!(names, vec!["x", "a", "b", "y"]);
}

/// `parsed_value_to_block` is the alternate path — used when a
/// function `body:` field receives a `List<Statement>` directly
/// (no explicit `TypedBlock { ... }` wrapper). The same flatten
/// must apply or the nested-list shape would slip through.
#[test]
fn function_body_field_flattens_nested_statement_list() {
    let grammar_src = r#"
        @language { name: "FunctionBodyFlattenTest", version: "1.0" }

        program = { SOI ~ f:function_decl ~ EOI }
          -> f

        pair_stmt = { "pair" }
          -> concat_list(
              [TypedStatement::Expression {
                  expr: TypedExpression::Variable { name: intern("a") },
              }],
              [TypedStatement::Expression {
                  expr: TypedExpression::Variable { name: intern("b") },
              }],
          )

        single_stmt = { name:identifier }
          -> TypedStatement::Expression {
              expr: TypedExpression::Variable { name: intern(name) },
          }

        stmt = { pair_stmt | single_stmt }

        // Hand a bare `stmts:stmt*` (a `List<Statement>`) to the
        // function `body:` field — no explicit `TypedBlock { ... }`
        // wrapper. The `parsed_value_to_block` path is what gets
        // exercised here.
        function_decl = { "fn" ~ name:identifier ~ "(" ~ ")" ~ "{" ~ stmts:stmt* ~ "}" }
          -> TypedDeclaration::Function {
              name: intern(name),
              params: [],
              return_type: Type::Unit,
              body: Some(TypedBlock { statements: stmts }),
          }

        identifier = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }

        WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
    "#;

    let grammar = parse_grammar(grammar_src).expect("grammar parses");
    let interp = GrammarInterpreter::new(&grammar);

    let mut builder = TypedASTBuilder::new();
    let mut registry = TypeRegistry::new();
    let mut state = ParserState::new("fn foo() { x pair y }", &mut builder, &mut registry);

    let result = interp.parse_rule("program", &mut state);
    let ParseResult::Success(value, _) = result else {
        panic!("expected parse success, got {result:?}");
    };

    // Drill into the function body and count statements.
    let ParsedValue::Declaration(decl) = value else {
        panic!("expected Declaration, got {value:?}");
    };
    let zyntax_typed_ast::TypedDeclaration::Function(func) = decl.node else {
        panic!("expected Function declaration");
    };
    let body = func.body.expect("function should have a body");
    assert_eq!(
        body.statements.len(),
        4,
        "expected 4 statements after flatten in function body, got {}",
        body.statements.len()
    );
}
