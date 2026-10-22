use zyntax_ts::{
    ErrorKind, ModuleDependency, ModuleId, Options, parse_module, parse_module_with_dependencies,
};
use zyntax_typed_ast::{TypedDeclaration, TypedProgram};

fn library() -> TypedProgram {
    parse_module("export type Read={readonly x:number};export function used(n:number):number{return n+1;}export function unused<T>(xs:T[]):T[]{return xs;}export type {used as typeOnly};", "library.ts", ModuleId::new("library"), Options::default()).unwrap()
}

fn client(source: &str, library: &TypedProgram) -> Result<TypedProgram, zyntax_ts::Error> {
    parse_module_with_dependencies(
        source,
        "client.ts",
        ModuleId::new("client"),
        Options::default(),
        &[ModuleDependency {
            specifier: "./library",
            program: library,
        }],
    )
}

#[test]
fn only_referenced_namespace_functions_request_native_bodies() {
    let library = library();
    let source = client(
        "import * as lib from './library';const f=lib.used;console.log(f(3));",
        &library,
    )
    .unwrap();
    let imports: Vec<_> = source
        .declarations
        .iter()
        .filter_map(|d| {
            if let TypedDeclaration::Import(i) = &d.node {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(imports.len(), 1);
    assert_eq!(imports[0].items.len(), 1);
    assert!(
        matches!(&imports[0].items[0], zyntax_typed_ast::TypedImportItem::Named{name,..} if *name==ModuleId::new("library").function_name("used"))
    );
    for source in [
        "import * as lib from './library';",
        "import type * as lib from './library';const p:lib.Read={x:3};",
        "import * as lib from './library';const p:lib.Read={x:3};",
    ] {
        let program = client(source, &library).unwrap();
        assert!(
            !program
                .declarations
                .iter()
                .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
        );
    }
}

#[test]
fn namespace_diagnostics_distinguish_invalid_types_values_writes_and_object_gaps() {
    let library = library();
    for source in [
        "import * as lib from './library';lib.missing(1);",
        "import * as lib from './library';const p:lib.Missing={x:3};",
        "import * as lib from './library';const p:lib.used=3;",
        "import * as lib from './library';lib.Read(1);",
        "import * as lib from './library';lib.typeOnly(1);",
        "import type * as lib from './library';lib.used(1);",
        "import * as lib from './library';lib.used=(n:number):number=>n;",
        "import * as lib from './library';lib['used']=(n:number):number=>n;",
        "import * as lib from './library';lib.used++;",
        "import * as lib from './library';const p:lib.Read={x:3};p.x=4;",
    ] {
        let error = client(source, &library).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Type, "{source}: {error}");
        assert!(error.end > error.start);
    }
    for source in [
        "import * as lib from './library';const object=lib;",
        "import * as lib from './library';export {lib};",
        "import * as lib from './library';export default lib;",
        "import * as lib from './library';const key='used';lib[key](1);",
    ] {
        assert_eq!(
            client(source, &library).unwrap_err().kind,
            ErrorKind::Unsupported,
            "{source}"
        );
    }
}

#[test]
fn generic_namespace_results_preserve_readonly_policy() {
    let library = parse_module(
        "export type Read={readonly x:number};export function id<T>(x:T):T{return x;}",
        "library.ts",
        ModuleId::new("library"),
        Options::default(),
    )
    .unwrap();
    let source = "import * as lib from './library';type Write={x:number};const p:lib.Read={x:3};const q:Write=lib.id(p);q.x=4;";
    client(source, &library).unwrap();
    let checked = parse_module_with_dependencies(
        source,
        "client.ts",
        ModuleId::new("client"),
        Options {
            soundness: zyntax_ts::Soundness::Checked,
        },
        &[ModuleDependency {
            specifier: "./library",
            program: &library,
        }],
    )
    .unwrap_err();
    assert_eq!(checked.kind, ErrorKind::Soundness);
}
