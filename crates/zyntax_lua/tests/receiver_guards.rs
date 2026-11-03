use std::path::{Path, PathBuf};
use std::process::Command;
use zyntax_typed_ast::{InternedString, TypedDeclaration};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("conformance/metatables")
        .join(format!("receiver_guard_{name}.lua"))
}

fn probe_body(name: &str) -> String {
    let source = std::fs::read_to_string(fixture(name)).unwrap();
    let program = zyntax_lua::parse_program(&source, "guards.lua").unwrap();
    program
        .declarations
        .iter()
        .find_map(|d| match &d.node {
            TypedDeclaration::Function(f)
                if f.name.resolve_global().is_some_and(|name| {
                    name.starts_with("lua$probe$") && !name.ends_with("$fn")
                }) =>
            {
                Some(format!("{:?}", f.body))
            }
            _ => None,
        })
        .expect("probe is lowered")
}

#[test]
fn fresh_self_return_chains_need_no_nil_guards() {
    let guard = format!("{:?}", InternedString::new_global("zl_index_nil"));
    for name in ["self", "branches"] {
        assert!(
            !probe_body(name).contains(&guard),
            "{name} retains a nil guard"
        );
    }
}

#[test]
fn unproved_results_keep_nil_guards() {
    let guard = format!("{:?}", InternedString::new_global("zl_index_nil"));
    for name in [
        "nil",
        "fallthrough",
        "reassigned",
        "shadowed",
        "other",
        "field_call",
        "early_return",
    ] {
        assert!(
            probe_body(name).contains(&guard),
            "{name} lost its nil guard"
        );
    }
}

#[test]
fn receiver_chain_outputs_and_errors_match_lua() {
    for name in [
        "self",
        "branches",
        "nil",
        "fallthrough",
        "reassigned",
        "shadowed",
        "other",
        "captured",
        "field_call",
        "remetatable",
        "early_return",
        "error",
        "constructor_error",
        "method_deleted",
        "index_handler",
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
