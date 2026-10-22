//! Import checking reads portable interfaces, without library source or executable bodies.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirType, Intrinsic};
use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{CompiledImport, Snapshot, SnapshotBuilder, TieredRuntime, ZyntaxValue};
use zyntax_ts::{
    ErrorKind, ModuleDependency, ModuleId, Options, Soundness, parse_module,
    parse_module_with_dependencies,
};
use zyntax_typed_ast::typed_ast::{TypedDeclaration, TypedImportItem};
use zyntax_typed_ast::{InternedString, TypedProgram};

static TEST: Mutex<()> = Mutex::new(());
const LIBRARY: &str = r#"
    type Private = {x:number;y:number};
    export type Read = {readonly x:number};
    export function make(n:number):Read {const p:Private={x:n,y:4};return p;}
    export default function keep<T>(x:T):T{return x;}
    export {keep};
    export function alias(xs:number[]):number[]{return xs;}
    export function numbers(n:number):number[]{return [n,n+2];}
    function second<T>(xs:T[],fallback:T):T{if(xs.length>1)return xs[1];return fallback;}
    export {second as pick};
    function secret(n:number):number{return n*100;}
"#;
const CLIENT: &str = r#"
    import type {Read as ImportedRead} from "./library";
    import keep, {make as build,alias,numbers} from "./library";
    import {pick} from "./library";
    export function score(n:number):number {
        const p:ImportedRead=build(n);const q=keep(p);
        const xs=numbers(q.x);const ys=alias(xs);ys[1]=ys[1]+3;
        return pick(xs,0)+p.x;
    }
    score(1);
"#;
const NAMESPACE_CLIENT: &str = r#"
    import * as lib from "./library";
    export function score(n:number):number {
        const p:lib.Read=lib.make(n);const q=lib.default(p);
        const xs=lib.numbers(q.x);const ys=lib['alias'](xs);ys[1]=ys[1]+3;
        return lib.pick(xs,0)+p.x;
    }
    score(1);
"#;

fn library() -> TypedProgram {
    parse_module(
        LIBRARY,
        "library.ts",
        ModuleId::new("library"),
        Options::default(),
    )
    .unwrap()
}

fn client(
    source: &str,
    dependency: &TypedProgram,
    options: Options,
) -> Result<TypedProgram, zyntax_ts::Error> {
    parse_module_with_dependencies(
        source,
        "client.ts",
        ModuleId::new("client"),
        options,
        &[ModuleDependency {
            specifier: "./library",
            program: dependency,
        }],
    )
}

fn snapshot(program: TypedProgram) -> Arc<Snapshot> {
    let hir = zyntax_embed::lower_for_snapshot_releasing(
        "library",
        program.clone(),
        Default::default(),
        vec![],
        true,
        None,
    )
    .unwrap();
    Arc::new(
        Snapshot::load_owned(
            SnapshotBuilder::new("typescript")
                .module_lowered("library", program, &hir)
                .unwrap()
                .encode()
                .unwrap(),
        )
        .unwrap(),
    )
}

#[test]
fn source_and_snapshot_interfaces_preserve_private_views_generics_and_array_provenance() {
    let source = library();
    let artifact = CompiledImport::decode(
        &CompiledImport::new("typescript", "library", source.clone())
            .encode()
            .unwrap(),
    )
    .unwrap();
    let snapshot = snapshot(source.clone());
    let compact = snapshot.module_interface("library").unwrap().unwrap();
    assert!(
        !compact
            .program()
            .declarations
            .iter()
            .any(|d| matches!(d.node, TypedDeclaration::Function(_)))
    );
    for dependency in [&source, artifact.program(), compact.program()] {
        let interface = dependency.checker_interface.as_ref().unwrap();
        let name = |name| InternedString::new_global(name);
        assert!(!interface.types.contains_key(&name("Private")));
        assert!(!interface.functions.contains_key(&name("secret")));
        assert!(!interface.functions.contains_key(&name("second")));
        assert_eq!(
            interface.functions[&name("pick")].native,
            ModuleId::new("library").function_name("second")
        );
        assert_eq!(interface.functions[&name("keep")].type_params.len(), 1);
        assert_eq!(
            interface.functions[&name("alias")]
                .array_result
                .unwrap()
                .parameter,
            Some(0)
        );
        assert_eq!(
            interface.functions[&name("numbers")]
                .array_result
                .unwrap()
                .minimum,
            2
        );
        let view = interface.types[&name("Read")].view.unwrap();
        assert!(interface.views[&view].field(name("x")).unwrap().readonly);
        let root = client(CLIENT, dependency, Options::default()).unwrap();
        let imports: Vec<_> = root
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
        assert_eq!(
            imports.len(),
            1,
            "multiple source imports combine their requested native signatures"
        );
        assert_eq!(imports[0].items.len(), 5);
        assert!(imports[0].items.contains(&TypedImportItem::Named {
            name: ModuleId::new("library").function_name("second"),
            alias: None
        }));
        let mixed = client(
            "import './library'; import {make} from './library'; make(1);",
            dependency,
            Options::default(),
        )
        .unwrap();
        let imports: Vec<_> = mixed
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
        assert_eq!(
            imports[0].items.len(),
            1,
            "a side-effect import keeps named signature selection"
        );
        let reexport = client(
            "import {keep} from './library'; export {keep as forward};",
            dependency,
            Options::default(),
        )
        .unwrap();
        assert!(Arc::ptr_eq(
            &interface.functions[&name("keep")],
            &reexport.checker_interface.as_ref().unwrap().functions[&name("forward")]
        ));
    }
}

#[test]
fn imports_preserve_readonly_policy_and_diagnose_inaccessible_or_invalid_names() {
    let library = library();
    let namespaces = parse_module(
        "export type Both={x:number}; export function Both(p:Both):number{return p.x+1;}",
        "dual.ts",
        ModuleId::new("dual"),
        Options::default(),
    )
    .unwrap();
    client(
        "import {Both} from './library'; const p:Both={x:2}; console.log(Both(p));",
        &namespaces,
        Options::default(),
    )
    .unwrap();
    client(
        "import type {Both} from './library'; const p:Both={x:2};",
        &namespaces,
        Options::default(),
    )
    .unwrap();
    assert_eq!(
        client(
            "import type {Both} from './library'; const p:Both={x:2}; Both(p);",
            &namespaces,
            Options::default()
        )
        .unwrap_err()
        .kind,
        ErrorKind::Type
    );
    for (source, kind) in [
        (
            "import {make} from './library'; const p=make(1); p.x=2;",
            ErrorKind::Type,
        ),
        (
            "import {secret} from './library'; console.log(secret(1));",
            ErrorKind::Type,
        ),
        ("import type {Private} from './library';", ErrorKind::Type),
        ("import {missing} from './library';", ErrorKind::Type),
        ("import {make} from './missing';", ErrorKind::Type),
        (
            "import * as lib from './library'; const namespace=lib;",
            ErrorKind::Unsupported,
        ),
    ] {
        let error = client(source, &library, Options::default()).unwrap_err();
        assert_eq!(error.kind, kind, "{error}");
        assert!(error.end > error.start);
    }
    let weaken = "import {make} from './library'; type Write={x:number}; const p=make(1); const q:Write=p; q.x=2;";
    client(weaken, &library, Options::default()).unwrap();
    assert_eq!(
        client(
            weaken,
            &library,
            Options {
                soundness: Soundness::Checked,
                ..Default::default()
            }
        )
        .unwrap_err()
        .kind,
        ErrorKind::Soundness
    );
    client(
        CLIENT,
        &library,
        Options {
            soundness: Soundness::Checked,
            ..Default::default()
        },
    )
    .unwrap();
}

fn execute(snapshot_mode: bool, reexport: bool, namespace: bool) {
    let source = library();
    let image = snapshot(source.clone());
    let compact = image.module_interface("library").unwrap().unwrap();
    let dependency = if snapshot_mode {
        compact.program()
    } else {
        &source
    };
    let barrel = parse_module_with_dependencies(
        "export * from './original'; export {default} from './original'; export {keep} from './original';",
        "barrel.ts",
        ModuleId::new("barrel"),
        Options::default(),
        &[ModuleDependency {
            specifier: "./original",
            program: dependency,
        }],
    )
    .unwrap();
    let program = client(
        if namespace { NAMESPACE_CLIENT } else { CLIENT },
        if reexport { &barrel } else { dependency },
        Options::default(),
    )
    .unwrap();
    let mut runtime = TieredRuntime::new(TieredConfig {
        baseline_threshold: u32::MAX,
        enable_osr: false,
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    })
    .unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    if reexport {
        runtime.add_compiled_import_resolver(Box::new(move |name| {
            Ok((name == "barrel").then(|| CompiledImport::new("typescript", name, barrel.clone())))
        }));
    }
    if snapshot_mode {
        runtime.install_snapshot(image).unwrap();
    } else {
        runtime.add_compiled_import_resolver(Box::new(move |name| {
            Ok(
                (name == "library")
                    .then(|| CompiledImport::new("typescript", name, source.clone())),
            )
        }));
    }
    let score = ModuleId::new("client")
        .function_name("score")
        .resolve_global()
        .unwrap();
    runtime.declare_entry_points([score.as_str()]);
    runtime.enter_only_through_entry_points();
    let mut hir = runtime.lower_to_hir(program.clone()).unwrap();
    zyntax_compiler::drop_insert::run_module(&mut hir);
    let body = hir
        .functions
        .values()
        .find(|f| f.name.resolve_global().as_deref() == Some(&score))
        .unwrap();
    assert!(body.blocks.values().flat_map(|b| &b.instructions).any(
        |i| matches!(i,HirInstruction::GetElementPtr{ty:HirType::Ptr(t),..} if **t==HirType::F64)
    ));
    assert_eq!(
        body.blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter(|i| matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                    ..
                }
            ))
            .count(),
        3,
        "the object, array header and array buffer each have one owner"
    );
    let names = runtime.compile_typed_program(program).unwrap();
    let natives: Vec<_> = names
        .into_iter()
        .filter(|n| n.starts_with("ts$m$"))
        .collect();
    let check = |runtime: &mut TieredRuntime| {
        for n in [-4.0, 0.0, 3.25] {
            assert_eq!(
                runtime.call_raw(&score, &[ZyntaxValue::Float(n)]).unwrap(),
                ZyntaxValue::Float(n * 2.0 + 5.0)
            );
        }
    };
    assert_eq!(runtime.statistics().baseline_functions, 0);
    check(&mut runtime);
    runtime
        .optimize_function(&score, OptimizationTier::Baseline)
        .unwrap();
    check(&mut runtime);
    for name in &natives {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        runtime
            .optimize_function(&score, OptimizationTier::Optimized)
            .unwrap();
        wait(&runtime, 1);
        check(&mut runtime);
        for name in &natives {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, natives.len());
        check(&mut runtime);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}

#[cfg(feature = "llvm-backend")]
fn wait(runtime: &TieredRuntime, count: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while runtime.statistics().optimized_functions < count {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn source_interfaces_run_with_native_layouts_and_array_aliases_in_each_tier() {
    let _test = TEST.lock().unwrap();
    execute(false, false, false);
}

#[test]
fn snapshot_interfaces_run_with_native_layouts_and_array_aliases_in_each_tier() {
    let _test = TEST.lock().unwrap();
    execute(true, false, false);
}

#[test]
fn source_reexports_preserve_native_geps_release_sites_and_each_promotion() {
    let _test = TEST.lock().unwrap();
    execute(false, true, false);
}

#[test]
fn snapshot_reexports_preserve_native_geps_release_sites_and_each_promotion() {
    let _test = TEST.lock().unwrap();
    execute(true, true, false);
}

#[test]
fn namespace_calls_reuse_native_layouts_release_sites_and_promotions_through_source_and_snapshots()
{
    let _test = TEST.lock().unwrap();
    for snapshot in [false, true] {
        for reexport in [false, true] {
            execute(snapshot, reexport, true);
        }
    }
}

#[test]
fn type_only_and_elided_imports_do_not_evaluate_dependencies() {
    let _test = TEST.lock().unwrap();
    let library = library();
    let requests = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&requests);
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.add_compiled_import_resolver(Box::new(move |_| {
        seen.fetch_add(1, Ordering::Relaxed);
        Err("type-only imports must not load code".into())
    }));
    for source in [
        "import type {Read} from './library'; const p:Read={x:2}; console.log(p.x);",
        "import {make} from './library'; console.log(2);",
        "import {} from './library'; console.log(2);",
    ] {
        let program = client(source, &library, Options::default()).unwrap();
        assert!(
            !program
                .declarations
                .iter()
                .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
        );
        runtime.compile_typed_program(program).unwrap();
    }
    assert_eq!(requests.load(Ordering::Relaxed), 0);
    let program = client("import './library';", &library, Options::default()).unwrap();
    assert!(
        program
            .declarations
            .iter()
            .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
    );
    let script = parse_module(
        "console.log(3);",
        "script.ts",
        ModuleId::new("script"),
        Options::default(),
    )
    .unwrap();
    let side_effect = client("import './library';", &script, Options::default()).unwrap();
    assert!(
        side_effect
            .declarations
            .iter()
            .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
    );
    assert_eq!(
        client("import {} from './library';", &script, Options::default())
            .unwrap_err()
            .kind,
        ErrorKind::Type
    );
    runtime.stop();
}

#[test]
fn a_bare_import_evaluates_the_snapshot_initializer() {
    let _test = TEST.lock().unwrap();
    let program = parse_module(
        "export type Read={readonly x:number}; const started=1;",
        "library.ts",
        ModuleId::new("library"),
        Options::default(),
    )
    .unwrap();
    let mut hir = zyntax_embed::lower_for_snapshot_releasing(
        "library",
        program.clone(),
        Default::default(),
        vec![],
        true,
        None,
    )
    .unwrap();
    let init = hir.functions.get_mut(&hir.initializers[0]).unwrap();
    // Make evaluation observable without adding a host global or a runtime shim.
    init.blocks.get_mut(&init.entry_block).unwrap().terminator =
        zyntax_compiler::hir::HirTerminator::Unreachable;
    let image = Arc::new(
        Snapshot::load_owned(
            SnapshotBuilder::new("typescript")
                .module_lowered("library", program, &hir)
                .unwrap()
                .encode()
                .unwrap(),
        )
        .unwrap(),
    );
    let interface = image.module_interface("library").unwrap().unwrap();
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.install_snapshot(image).unwrap();
    runtime.enter_only_through_entry_points();
    let types = client(
        "import type {Read} from './library'; const p:Read={x:1};",
        interface.program(),
        Options::default(),
    )
    .unwrap();
    runtime.compile_typed_program(types).unwrap();
    let side_effect = client(
        "import './library';",
        interface.program(),
        Options::default(),
    )
    .unwrap();
    let error = runtime.compile_typed_program(side_effect).unwrap_err();
    assert!(error.to_string().contains("unreachable"), "{error}");
    runtime.stop();
}
