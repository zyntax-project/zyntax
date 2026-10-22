use std::sync::Arc;
use zyntax_embed::{CompiledImport, Snapshot, SnapshotBuilder};
use zyntax_ts::{
    ErrorKind, ModuleDependency, ModuleId, Options, parse_module, parse_module_with_dependencies,
};
use zyntax_typed_ast::{InternedString, TypedDeclaration, TypedProgram};

fn module(
    name: &str,
    source: &str,
    dependencies: &[(&str, &TypedProgram)],
) -> Result<TypedProgram, zyntax_ts::Error> {
    let dependencies: Vec<_> = dependencies
        .iter()
        .map(|(specifier, program)| ModuleDependency { specifier, program })
        .collect();
    parse_module_with_dependencies(
        source,
        name,
        ModuleId::new(name),
        Options::default(),
        &dependencies,
    )
}
fn name(s: &str) -> InternedString {
    InternedString::new_global(s)
}
fn library() -> TypedProgram {
    parse_module("export type Point={readonly x:number}; export function id<T>(x:T):T{return x;} export function alias(xs:number[]):number[]{return xs;} export {id as default};", "a", ModuleId::new("a"), Options::default()).unwrap()
}

#[test]
fn bindings_keep_their_origin_through_named_local_star_artifact_and_snapshot_paths() {
    let a = library();
    let b = module(
        "b",
        "export {id as renamed,alias} from './a';export type {Point} from './a';",
        &[("./a", &a)],
    )
    .unwrap();
    let artifact = CompiledImport::decode(
        &CompiledImport::new("typescript", "b", b.clone())
            .encode()
            .unwrap(),
    )
    .unwrap();
    let snapshot = Snapshot::load_owned(
        SnapshotBuilder::new("typescript")
            .module("b", b.clone())
            .unwrap()
            .encode()
            .unwrap(),
    )
    .unwrap();
    let saved = snapshot.module_interface("b").unwrap().unwrap();
    for dependency in [&b, artifact.program(), saved.program()] {
        let c = module("c", "import type {Point as Local} from './b';export type {Local as Point};export * from './b';", &[("./b", dependency)]).unwrap();
        let interface = c.checker_interface.as_ref().unwrap();
        let original = dependency.checker_interface.as_ref().unwrap();
        assert_eq!(
            interface.exports[&name("Point")].origin,
            a.checker_interface.as_ref().unwrap().exports[&name("Point")].origin
        );
        assert_eq!(
            interface.exports[&name("renamed")].origin,
            ModuleId::new("a").function_name("id")
        );
        assert!(Arc::ptr_eq(
            &interface.functions[&name("renamed")],
            &original.functions[&name("renamed")]
        ));
        assert_eq!(interface.functions[&name("renamed")].type_params.len(), 1);
        assert_eq!(
            interface.functions[&name("alias")]
                .array_result
                .unwrap()
                .parameter,
            Some(0)
        );
        let view = interface.types[&name("Point")].view.unwrap();
        assert!(interface.views[&view].field(name("x")).unwrap().readonly);
    }
    let all = module("all", "export * from './a';", &[("./a", &a)]).unwrap();
    assert!(
        !all.checker_interface
            .as_ref()
            .unwrap()
            .exports
            .contains_key(&name("default"))
    );
}

#[test]
fn star_conflicts_compare_original_bindings_and_explicit_exports_take_precedence() {
    let a = module("a", "export type X=number;", &[]).unwrap();
    let b = module("b", "export type X=number;", &[]).unwrap();
    let forwarded = module("forwarded", "export {X} from './a';", &[("./a", &a)]).unwrap();
    module(
        "same",
        "export * from './a';export * from './b';",
        &[("./a", &a), ("./b", &forwarded)],
    )
    .unwrap();
    let error = module(
        "different",
        "export * from './a';export * from './b';",
        &[("./a", &a), ("./b", &b)],
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Type);
    assert!(error.message.contains("conflicting star export"));
    module(
        "explicit",
        "export * from './a';export * from './b';export {X} from './a';",
        &[("./a", &a), ("./b", &b)],
    )
    .unwrap();
    module(
        "local",
        "export * from './a';export * from './b';export type X=boolean;",
        &[("./a", &a), ("./b", &b)],
    )
    .unwrap();
    let alias = module(
        "alias",
        "import type {X as Y} from './a'; export type X=Y;",
        &[("./a", &a)],
    )
    .unwrap();
    assert_eq!(
        module(
            "different-alias",
            "export * from './a';export * from './b';",
            &[("./a", &a), ("./b", &alias)]
        )
        .unwrap_err()
        .kind,
        ErrorKind::Type
    );
}

#[test]
fn type_only_function_exports_propagate_and_value_paths_upgrade_star_exports() {
    let a = library();
    let b = module("b", "export type * from './a';", &[("./a", &a)]).unwrap();
    assert!(b.checker_interface.as_ref().unwrap().exports[&name("id")].type_only);
    assert!(
        !b.declarations
            .iter()
            .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
    );
    assert_eq!(
        module("bad", "import {id} from './b';id(1);", &[("./b", &b)])
            .unwrap_err()
            .kind,
        ErrorKind::Type
    );
    module(
        "types",
        "import type {id} from './b';export type {id};",
        &[("./b", &b)],
    )
    .unwrap();
    for (source, dependency) in [
        ("import {id} from './b';export {id};", &b),
        ("import {id} from './b';export type {id};", &a),
    ] {
        let forwarded = module("local-type", source, &[("./b", dependency)]).unwrap();
        assert!(forwarded.checker_interface.as_ref().unwrap().exports[&name("id")].type_only);
        assert!(
            !forwarded
                .declarations
                .iter()
                .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
        );
    }
    for source in [
        "export type * from './a';export * from './a';",
        "export * from './a';export type * from './a';",
    ] {
        let program = module("upgraded", source, &[("./a", &a)]).unwrap();
        assert!(!program.checker_interface.as_ref().unwrap().exports[&name("id")].type_only);
        module(
            "value",
            "import {id} from './b';id(1);",
            &[("./b", &program)],
        )
        .unwrap();
    }
    let explicit = module(
        "explicit",
        "export type {id} from './a';export * from './a';",
        &[("./a", &a)],
    )
    .unwrap();
    assert!(explicit.checker_interface.as_ref().unwrap().exports[&name("id")].type_only);
}

#[test]
fn missing_names_and_namespace_exports_have_distinct_diagnostics() {
    let a = library();
    assert_eq!(
        module("missing", "export {missing} from './a';", &[("./a", &a)])
            .unwrap_err()
            .kind,
        ErrorKind::Type
    );
    assert_eq!(
        module("namespace", "export * as ns from './a';", &[("./a", &a)])
            .unwrap_err()
            .kind,
        ErrorKind::Unsupported
    );
    let types = module("types", "export type console={x:number};", &[]).unwrap();
    assert_eq!(
        module(
            "bad-console",
            "import type {console} from './types';console.log(1);",
            &[("./types", &types)]
        )
        .unwrap_err()
        .kind,
        ErrorKind::Type
    );
}
