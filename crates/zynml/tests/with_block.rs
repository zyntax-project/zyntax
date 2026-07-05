//! Phase 1 of the fiber×effect×async plan: `with H { }` handler-scoping
//! block. This first slice covers parsing → typed AST. Lowering (emit
//! push_handler/pop_handler on every exit edge) and runtime dynamic
//! dispatch land in the next slices.

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedStatement};

/// The `with` body executes and its mutations flow out of the scope.
/// For the single-handler case this is behaviourally complete today
/// (static handler dispatch already routes performs); regional
/// multi-handler selection is a later slice.
#[test]
fn with_block_body_executes() {
    use zynml::ZynML;
    let mut rt = ZynML::new().expect("rt");
    rt.load_source(
        r#"
        def main(): i64 {
            let mut sum: i64 = 0
            with SomeHandler {
                sum = sum + 10
                sum = sum + 20
            }
            return sum
        }
        "#,
    )
    .expect("load");
    let result: i64 = rt.call_with_result("main").expect("call");
    assert_eq!(result, 30, "with-body mutations propagate out of the scope");
}

/// `with H { ... }` parses into a `TypedStatement::With` carrying the
/// handler name and the scoped body.
#[test]
fn with_block_parses_to_with_statement() {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(
            r#"
            def main(): i64 {
                with StderrLog {
                    let x = 1
                    let y = 2
                }
                return 0
            }
            "#,
            "<with_block>",
        )
        .expect("`with H { }` should parse");

    // Find main and its body; assert a With statement is present with
    // the right handler name and a non-empty body.
    let main = program
        .declarations
        .iter()
        .find_map(|d| match &d.node {
            TypedDeclaration::Function(f) if f.name.resolve_global().as_deref() == Some("main") => {
                Some(f)
            }
            _ => None,
        })
        .expect("main function");
    let body = main.body.as_ref().expect("main has a body");

    let with = body
        .statements
        .iter()
        .find_map(|s| match &s.node {
            TypedStatement::With(w) => Some(w),
            _ => None,
        })
        .expect("body contains a `with` statement");

    assert_eq!(with.handlers.len(), 1, "single handler");
    assert_eq!(
        with.handlers[0].name.resolve_global().as_deref(),
        Some("StderrLog"),
        "handler name preserved"
    );
    assert_eq!(
        with.body.statements.len(),
        2,
        "the two let bindings are in the scoped body"
    );
}
