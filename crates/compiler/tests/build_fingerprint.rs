//! Build identity follows serialized types and lowering dependencies, independent of checkout path.

#[path = "../build_fingerprint.rs"]
mod build_fingerprint;

use build_fingerprint::{BUILD_INPUTS, SOURCE_ROOTS, fingerprint};
use std::path::Path;

fn write(root: &Path, path: impl AsRef<Path>, contents: &str) {
    let path = root.join(path);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

fn workspace(root: &Path) {
    for file in BUILD_INPUTS {
        write(root, file, "build input");
    }
    for source in SOURCE_ROOTS {
        write(root, Path::new(source).join("Cargo.toml"), "manifest");
        write(root, Path::new(source).join("src/lib.rs"), "source");
    }
}

#[test]
fn shared_type_and_codec_changes_always_change_the_identity() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    workspace(root);
    let required = [
        "crates/compiler/src/hir.rs",
        "crates/compiler/src/bytecode.rs",
        "crates/typed_ast/src/source.rs",
        "crates/typed_ast/src/arena.rs",
        "crates/typed_ast/src/type_registry.rs",
        "crates/typed_ast/src/checker_interface.rs",
        "crates/zyntax_embed/src/compiled_artifact.rs",
        "sdk/zrtl/src/dynamic_box.rs",
        "crates/passes/normalization/src/lib.rs",
    ];
    for path in required {
        write(root, path, "serialized or lowering input");
    }
    let original = fingerprint(root).unwrap();
    for path in required {
        write(root, path, "changed source");
        assert_ne!(fingerprint(root).unwrap(), original, "{path}");
        write(root, path, "serialized or lowering input");
        assert_eq!(fingerprint(root).unwrap(), original);
    }
    write(
        root,
        "crates/typed_ast/src/nested/ownership.rs",
        "new serialized type",
    );
    assert_ne!(fingerprint(root).unwrap(), original);
    std::fs::remove_file(root.join("crates/typed_ast/src/nested/ownership.rs")).unwrap();
    assert_eq!(fingerprint(root).unwrap(), original);
    for file in BUILD_INPUTS {
        write(root, file, "changed build input");
        assert_ne!(fingerprint(root).unwrap(), original, "{file}");
        write(root, file, "build input");
    }
    std::fs::remove_file(root.join("Cargo.lock")).unwrap();
    let unlocked = fingerprint(root).unwrap();
    assert_ne!(
        unlocked, original,
        "an absent lockfile is part of the identity"
    );
    write(root, "Cargo.lock", "build input");
    std::fs::remove_file(root.join("crates/compiler/build.rs")).unwrap();
    assert!(
        fingerprint(root).is_err(),
        "missing inputs must not silently produce a partial identity"
    );
}

#[test]
fn equivalent_relocated_workspaces_have_the_same_identity() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    workspace(left.path());
    workspace(right.path());
    assert_eq!(
        fingerprint(left.path()).unwrap(),
        fingerprint(right.path()).unwrap()
    );
    let hex: String = zyntax_compiler::BUILD_ID_BYTES
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(hex, zyntax_compiler::BUILD_ID);
}

#[test]
fn snapshot_generators_watch_every_fingerprinted_source_root() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let expected = build_fingerprint::watched_inputs(&root);
    let generated = include_str!(concat!(env!("OUT_DIR"), "/build_inputs.txt"));
    for required in [
        "crates/compiler/src",
        "crates/typed_ast/src",
        "crates/zyntax_embed/src",
        "sdk/zrtl/src",
        "Cargo.lock",
    ] {
        assert!(
            generated
                .lines()
                .any(|path| Path::new(path) == root.join(required)),
            "{required}"
        );
    }
    assert_eq!(
        generated
            .lines()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        expected
    );
    for input in expected {
        assert!(input.exists(), "{}", input.display());
    }
}
