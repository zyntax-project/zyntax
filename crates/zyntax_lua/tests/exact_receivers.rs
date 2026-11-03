use std::path::Path;
use std::process::Command;
use zyntax_typed_ast::TypedDeclaration;

#[test]
fn each_exact_call_has_only_its_own_method_target() {
    let source = include_str!("../conformance/metatables/exact_receiver_happy.lua");
    let program = zyntax_lua::parse_program(source, "exact.lua").unwrap();
    let functions: Vec<_> = program
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            TypedDeclaration::Function(f) => Some(f),
            _ => None,
        })
        .collect();
    let chunk = functions
        .iter()
        .find(|f| f.name.resolve_global().as_deref() == Some("lua$chunk"))
        .unwrap();
    let body = format!("{:?}", chunk.body);
    for class in ["A", "B"] {
        let method = functions
            .iter()
            .find(|f| {
                f.name.resolve_global().is_some_and(|n| {
                    n.starts_with(&format!("lua${class}$get$")) && !n.ends_with("$fn")
                })
            })
            .unwrap();
        assert_eq!(
            body.matches(&format!("{:?}", method.name)).count(),
            1,
            "{class}.get should appear only at its own receiver's call"
        );
    }
}

fn traced(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("conformance/metatables")
        .join(format!("exact_receiver_{name}.lua"));
    let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
        .arg(&path)
        .env("ZYNTAX_TRACE_TYPES", "1")
        .env("ZYNTAX_DISABLE_WARM_UP", "1")
        .output()
        .expect("zylua starts");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{name}: {stderr}");
    assert_eq!(
        output.stdout,
        std::fs::read(path.with_extension("expected")).unwrap(),
        "{name}: {stderr}"
    );
    stderr.into_owned()
}

#[test]
fn fresh_locals_keep_distinct_classes_on_a_shared_shape() {
    let trace = traced("happy");
    let class = |name: &str| {
        trace.lines().find_map(|line| {
            line.strip_prefix(&format!("[types] exact {name}#"))
                .and_then(|s| s.split_once(": ").map(|(_, class)| class))
        })
    };
    assert!(class("probe").is_some(), "{trace}");
    assert!(class("other").is_some(), "{trace}");
    assert_ne!(class("probe"), class("other"), "{trace}");
}

#[test]
fn uncertain_receivers_keep_general_dispatch() {
    for name in [
        "alias",
        "parameter",
        "capture",
        "captured_constructor",
        "method",
        "debug",
        "getmetatable",
        "reassigned",
        "mixed_return",
        "published",
        "conditional",
        "loop",
        "absent",
        "replaced",
        "call_constructor",
        "shadowed",
        "newindex",
    ] {
        let trace = traced(name);
        assert!(!trace.contains("[types] exact probe#"), "{name}: {trace}");
    }
    traced("index_function");
    traced("chained");
    traced("own_method");
}
