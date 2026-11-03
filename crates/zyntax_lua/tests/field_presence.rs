use std::path::{Path, PathBuf};
use std::process::Command;
use zyntax_typed_ast::{InternedString, TypedDeclaration};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("conformance/metatables")
        .join(format!("field_presence_{name}.lua"))
}

fn advance(name: &str) -> String {
    let source = std::fs::read_to_string(fixture(name)).unwrap();
    let program = zyntax_lua::parse_program(&source, "fields.lua").unwrap();
    program
        .declarations
        .iter()
        .find_map(|d| match &d.node {
            TypedDeclaration::Function(f)
                if f.name.resolve_global().is_some_and(|name| {
                    name.starts_with("lua$Full$advance$") && !name.ends_with("$fn")
                }) =>
            {
                Some(format!("{:?}", f.body))
            }
            _ => None,
        })
        .expect("advance is lowered")
}

#[test]
fn initialized_receivers_use_typed_arithmetic() {
    for name in ["complete", "shadow", "nested", "float"] {
        let body = advance(name);
        if name == "complete" {
            assert!(!body.contains(&format!("{:?}", InternedString::new_global("zl_index_nil"))));
        }
        for helper in ["zl_arith", "zl_le", "zl_index_key", "zl_table_index_key"] {
            assert!(
                !body.contains(&format!("{:?}", InternedString::new_global(helper))),
                "{name} retains {helper}"
            );
        }
    }
}

#[test]
fn partial_and_indirect_callers_keep_dynamic_arithmetic() {
    let helper = format!("{:?}", InternedString::new_global("zl_arith"));
    for name in [
        "partial",
        "callback",
        "nil_store",
        "argument_effect",
        "reassigned",
    ] {
        assert!(advance(name).contains(&helper), "{name} lost its checks");
    }
}

#[test]
fn field_presence_outputs_and_errors_match_lua() {
    for name in [
        "complete",
        "nested",
        "float",
        "debug_alias",
        "partial",
        "nil_store",
        "argument_effect",
        "callback",
        "dot_nil",
        "dynamic_store",
        "reassigned",
        "setter",
        "weak",
        "shadow",
    ] {
        let path = fixture(name);
        for llvm in ["0", "1"] {
            let output = Command::new(env!("CARGO_BIN_EXE_zylua"))
                .arg(&path)
                .env("ZYLUA_LLVM", llvm)
                .output()
                .expect("zylua starts");
            assert!(
                output.status.success(),
                "{name}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                output.stdout,
                std::fs::read(path.with_extension("expected")).unwrap(),
                "{name}, LLVM={llvm}"
            );
        }
    }
}
