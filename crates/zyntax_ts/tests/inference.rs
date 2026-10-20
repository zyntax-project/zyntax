//! Function dependencies resolve to concrete native types without duplicate bodies.

use std::collections::HashSet;
use zyntax_typed_ast::typed_ast::TypedDeclaration;
use zyntax_typed_ast::{PrimitiveType, Type};

#[test]
fn shared_forward_dependencies_have_one_concrete_body_each() {
    let mut source = String::new();
    for i in 0..256 {
        let body = if i == 255 {
            "x + 1".to_string()
        } else {
            format!("f{}(x) + f{}(x + 1)", i + 1, i + 1)
        };
        source.push_str(&format!("function f{i}(x: number) {{ return {body}; }}\n"));
    }
    let program = zyntax_ts::parse_program(&source, "dependencies.ts", Default::default()).unwrap();
    let mut names = HashSet::new();
    for declaration in &program.declarations {
        if let TypedDeclaration::Function(f) = &declaration.node {
            let name = f.name.resolve_global().unwrap();
            if name.starts_with("ts$m$dependencies_2ets$fn$") {
                assert!(names.insert(name), "duplicate function body");
                assert_eq!(f.return_type, Type::Primitive(PrimitiveType::F64));
            }
        }
    }
    assert_eq!(names.len(), 256);
}

#[test]
fn annotated_return_anchors_a_recursive_group_in_either_source_order() {
    let inferred = "function odd(n: number) { return !even(n); }";
    let annotated =
        "function even(n: number): boolean { if (n == 0) return true; return odd(n - 1); }";
    for source in [
        format!("{inferred}{annotated}"),
        format!("{annotated}{inferred}"),
    ] {
        let program =
            zyntax_ts::parse_program(&source, "recursion.ts", Default::default()).unwrap();
        for declaration in &program.declarations {
            if let TypedDeclaration::Function(f) = &declaration.node {
                if f.name
                    .resolve_global()
                    .unwrap()
                    .starts_with("ts$m$recursion_2ets$fn$")
                {
                    assert_eq!(f.return_type, Type::Primitive(PrimitiveType::Bool));
                }
            }
        }
    }
}
