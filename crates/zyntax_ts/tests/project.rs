use std::path::Path;
use std::process::Command;

use serde_json::Value;
use zyntax_ts::{Options, ProjectError, load_project};

fn write(root: &Path, file: &str, source: &str) {
    let path = root.join(file);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, source).unwrap();
}

fn check(root: &Path, source: &str) -> (i32, Value) {
    write(root, "entry.ts", source);
    let output = Command::new(env!("CARGO_BIN_EXE_zynode"))
        .args(["check", "--diagnostics=json"])
        .arg(root.join("entry.ts"))
        .output()
        .unwrap();
    (
        output.status.code().unwrap(),
        serde_json::from_slice(&output.stderr).unwrap_or(Value::Null),
    )
}

#[test]
fn relative_resolution_uses_shared_candidates_and_canonical_identities() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "library.ts",
        "export function f(n:number):number{return n+1;}",
    );
    write(
        root,
        "library/index.ts",
        "export function wrong():number{return 999;}",
    );
    write(root, "index/index.ts", "export type Point={x:number};");
    #[cfg(unix)]
    std::os::unix::fs::symlink(root.join("library.ts"), root.join("link.ts")).unwrap();
    let mut source = String::from(
        "import {f as a} from './library'; import {f as b} from './library.js';
         import {f as c} from './library.ts'; import type {Point} from './index';
         import {f as normalized} from './missing/../library';
         const p:Point={x:1}; console.log(a(p.x)+b(2)+c(3));",
    );
    #[cfg(unix)]
    source.push_str("import {f as d} from './link'; console.log(d(4));");
    write(root, "entry.ts", &source);
    let project = load_project(root.join("entry.ts"), Options::default()).unwrap();
    assert_eq!(
        project.module_count(),
        3,
        "alternate paths share a checked module"
    );
    let (entry, resolver) = project.into_parts();
    let imports: Vec<_> = entry
        .declarations
        .iter()
        .filter_map(|d| {
            if let zyntax_typed_ast::TypedDeclaration::Import(i) = &d.node {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(imports.len(), 1);
    let name = imports[0].module_path[0].resolve_global().unwrap();
    let dependency = resolver(&name).unwrap().unwrap();
    assert_eq!(dependency.module_name(), name);
    assert!(
        resolver(&name).unwrap().is_none(),
        "bodies move into the linker once"
    );
    let types = root.join("index/index.ts").canonicalize().unwrap();
    assert!(
        resolver(&types.to_string_lossy()).unwrap().is_none(),
        "type-only bodies are dropped"
    );
}

#[test]
fn diagnostics_point_to_the_dependency_source_and_import_spans() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "bad.ts",
        "export function f():number{return 'wrong';}",
    );
    let (code, error) = check(root, "import {f} from './bad'; console.log(f());");
    assert_eq!(code, 1, "{error}");
    assert_eq!(error["kind"], "Type");
    assert_eq!(
        error["file"],
        root.join("bad.ts")
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
    );
    assert!(error["start"].as_u64().unwrap() > 0);
    let project = load_project(root.join("entry.ts"), Options::default())
        .err()
        .unwrap();
    let ProjectError::Diagnostic {
        file,
        source_code,
        error,
    } = project
    else {
        panic!("{project}")
    };
    let rendered = error.render(file.to_str().unwrap(), &source_code, false);
    assert!(rendered.contains("bad.ts"));
    assert!(rendered.contains("wrong"));
    let (code, error) = check(root, "import {f} from './missing';");
    assert_eq!(code, 1);
    assert_eq!(error["kind"], "Type");
    assert_eq!(error["start"], 16);
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("cannot resolve module")
    );
}

#[test]
fn unsupported_module_semantics_remain_distinct_from_type_errors() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "cycle.ts",
        "import './entry'; export function f():number{return 1;}",
    );
    for (source, message) in [
        ("import './cycle';", "cyclic module initialization"),
        (
            "import {x} from 'package';",
            "package and absolute module resolution",
        ),
    ] {
        let (code, error) = check(root, source);
        assert_eq!(code, 3, "{error}");
        assert_eq!(error["kind"], "Unsupported");
        assert!(
            error["message"].as_str().unwrap().contains(message),
            "{error}"
        );
        assert!(error["end"].as_u64().unwrap() > error["start"].as_u64().unwrap());
    }
    for file in [
        "only.tsx",
        "only.d.ts",
        "only.js",
        "only.mts",
        "only.cts",
        "only.mjs",
        "only.cjs",
        "only.json",
    ] {
        write(root, file, "export function f():number{return 1;}");
        let (code, error) = check(root, &format!("import {{f}} from './{file}'; f();"));
        assert_eq!(code, 3, "{file}: {error}");
        assert_eq!(error["kind"], "Unsupported");
    }
}

#[test]
fn deep_dependency_chains_report_a_bounded_capability_gap() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    for n in 0..128 {
        write(
            root,
            &format!("m{n}.ts"),
            &format!("import './m{}';", n + 1),
        );
    }
    write(root, "m128.ts", "export {};");
    let (code, error) = check(root, "import './m0';");
    assert_eq!(code, 3, "{error}");
    assert_eq!(error["kind"], "Unsupported");
    assert!(error["message"].as_str().unwrap().contains("depth exceeds"));
}
