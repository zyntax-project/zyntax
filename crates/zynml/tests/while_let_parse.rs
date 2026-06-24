//! Parse-shape test for `while let PATTERN = expr { body }`.
//!
//! The grammar desugars at parse time to a plain
//! `while true { match expr { case PATTERN { body } case _ { break } } }`
//! shape so no new typed-AST variant is introduced. These tests
//! verify the synthesized AST has exactly that structure.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_typed_ast::typed_ast::{
    TypedDeclaration, TypedExpression, TypedLiteral, TypedPattern, TypedStatement,
};

fn parse(src: &str) -> zyntax_typed_ast::TypedProgram {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    grammar
        .parse_with_filename(src, "<while_let_parse>")
        .expect("source should parse")
}

/// Return the first `While` statement found inside the first
/// function's body — that's where the parse-time desugar lands.
fn first_while<'a>(
    program: &'a zyntax_typed_ast::TypedProgram,
) -> &'a zyntax_typed_ast::typed_ast::TypedWhile {
    for decl in &program.declarations {
        if let TypedDeclaration::Function(f) = &decl.node {
            if let Some(body) = &f.body {
                for stmt in &body.statements {
                    if let TypedStatement::While(w) = &stmt.node {
                        return w;
                    }
                }
            }
        }
    }
    panic!("no While statement found in first function");
}

#[test]
fn while_let_some_desugars_to_while_true_match() {
    let program = parse(
        r#"
        def consume() {
            while let Some(x) = source {
                println(x)
            }
        }
        "#,
    );

    let w = first_while(&program);

    // Condition is the literal `true`.
    match &w.condition.node {
        TypedExpression::Literal(TypedLiteral::Bool(true)) => {}
        other => panic!("expected `true` condition, got {other:?}"),
    }

    // Body is a single Expression(Match) statement.
    assert_eq!(
        w.body.statements.len(),
        1,
        "expected exactly one synthesized statement in body"
    );

    let match_expr = match &w.body.statements[0].node {
        TypedStatement::Expression(e) => match &e.node {
            TypedExpression::Match(m) => m,
            other => panic!("expected Match expr, got {other:?}"),
        },
        other => panic!("expected Expression stmt, got {other:?}"),
    };

    // Two arms: the user pattern, then a wildcard `break`.
    assert_eq!(match_expr.arms.len(), 2, "expected exactly two match arms");
    assert!(
        !matches!(match_expr.arms[0].pattern.node, TypedPattern::Wildcard),
        "first arm should be the user pattern, not wildcard"
    );
    assert!(
        matches!(match_expr.arms[1].pattern.node, TypedPattern::Wildcard),
        "second arm should be the wildcard break"
    );

    // Wildcard arm body is a `Block { Break }`.
    let break_body = match &match_expr.arms[1].body.node {
        TypedExpression::Block(b) => b,
        other => panic!("expected Block body, got {other:?}"),
    };
    assert_eq!(break_body.statements.len(), 1);
    assert!(matches!(
        break_body.statements[0].node,
        TypedStatement::Break(None)
    ));
}

#[test]
fn regular_while_still_parses() {
    let program = parse(
        r#"
        def loop_forever() {
            while true {
                break
            }
        }
        "#,
    );
    let w = first_while(&program);
    assert!(matches!(
        w.condition.node,
        TypedExpression::Literal(TypedLiteral::Bool(true))
    ));
}
