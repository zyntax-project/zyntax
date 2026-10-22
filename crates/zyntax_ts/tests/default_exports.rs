use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{HirCallable, HirInstruction, HirType, Intrinsic};
use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{CompiledImport, Snapshot, SnapshotBuilder, TieredRuntime, ZyntaxValue};
use zyntax_ts::{
    ErrorKind, ModuleDependency, ModuleId, Options, parse_module, parse_module_with_dependencies,
};
use zyntax_typed_ast::{InternedString, TypedDeclaration, TypedProgram};

static RUNTIME: Mutex<()> = Mutex::new(());

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

fn name(value: &str) -> InternedString {
    InternedString::new_global(value)
}

#[test]
fn default_aliases_share_native_signatures_and_type_only_provenance() {
    let source = module(
        "lib",
        "function id<T>(x:T):T{return x;} export default id; export {id};",
        &[],
    )
    .unwrap();
    let interface = source.checker_interface.as_ref().unwrap();
    assert!(Arc::ptr_eq(
        &interface.functions[&name("default")],
        &interface.functions[&name("id")]
    ));
    let forwarded = module(
        "forwarded",
        "import identity from './lib'; export default identity;",
        &[("./lib", &source)],
    )
    .unwrap();
    assert!(
        forwarded
            .declarations
            .iter()
            .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
    );
    assert!(Arc::ptr_eq(
        &interface.functions[&name("default")],
        &forwarded.checker_interface.as_ref().unwrap().functions[&name("default")]
    ));
    for statement in [
        "import type identity from './lib'; export type {identity as default};",
        "import identity from './lib'; export type {identity as default};",
        "import type identity from './lib'; export default identity;",
    ] {
        let types = module("types", statement, &[("./lib", &source)]).unwrap();
        assert!(types.checker_interface.as_ref().unwrap().exports[&name("default")].type_only);
        assert!(
            !types
                .declarations
                .iter()
                .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
        );
        assert_eq!(
            module(
                "bad",
                "import f from './types'; f(1);",
                &[("./types", &types)]
            )
            .unwrap_err()
            .kind,
            ErrorKind::Type
        );
    }
    let unused = module(
        "unused",
        "import identity from './lib'; console.log(1);",
        &[("./lib", &source)],
    )
    .unwrap();
    assert!(
        !unused
            .declarations
            .iter()
            .any(|d| matches!(d.node, TypedDeclaration::Import(_)))
    );
    assert_eq!(
        module(
            "unresolved",
            "export default function(){return missing();}",
            &[]
        )
        .unwrap_err()
        .kind,
        ErrorKind::Type
    );
    for source in [
        "export default 1;",
        "const f=(x:number):number=>x; export default f;",
        "export default class C {}",
    ] {
        assert_eq!(
            module("unsupported", source, &[]).unwrap_err().kind,
            ErrorKind::Unsupported
        );
    }
}

#[test]
fn anonymous_default_generics_keep_array_geps_and_aliases_through_snapshots_and_native_tiers() {
    let _guard = RUNTIME.lock().unwrap();
    let library = parse_module(
        "export default function<T>(xs:T[]):T[]{return xs;}",
        "lib.ts",
        ModuleId::new("lib"),
        Options::default(),
    )
    .unwrap();
    let hir = zyntax_embed::lower_for_snapshot_releasing(
        "lib",
        library.clone(),
        Default::default(),
        vec![],
        true,
        None,
    )
    .unwrap();
    let snapshot = Arc::new(
        Snapshot::load_owned(
            SnapshotBuilder::new("typescript")
                .module_lowered("lib", library.clone(), &hir)
                .unwrap()
                .encode()
                .unwrap(),
        )
        .unwrap(),
    );
    let compact = snapshot.module_interface("lib").unwrap().unwrap();
    for dependency in [&library, compact.program()] {
        let signature = &dependency.checker_interface.as_ref().unwrap().functions[&name("default")];
        assert_eq!(signature.type_params.len(), 1);
        assert_eq!(signature.array_result.unwrap().parameter, Some(0));
    }
    for from_snapshot in [false, true] {
        let root = module("root", "import keep from './lib'; export function score(n:number):number{const a=[2,8];const b=keep(a);b[1]=b[1]+n;return a[1];}", &[("./lib", if from_snapshot {compact.program()} else {&library})]).unwrap();
        let mut runtime = TieredRuntime::new(TieredConfig {
            baseline_threshold: u32::MAX,
            enable_osr: false,
            #[cfg(feature = "llvm-backend")]
            tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
            ..Default::default()
        })
        .unwrap();
        zyntax_ts::register_runtime(&mut runtime).unwrap();
        if from_snapshot {
            runtime.install_snapshot(Arc::clone(&snapshot)).unwrap();
        } else {
            let library = library.clone();
            runtime.add_compiled_import_resolver(Box::new(move |name| {
                Ok((name == "lib")
                    .then(|| CompiledImport::new("typescript", name, library.clone())))
            }));
        }
        let score = ModuleId::new("root")
            .function_name("score")
            .resolve_global()
            .unwrap();
        runtime.declare_entry_points([score.as_str()]);
        runtime.enter_only_through_entry_points();
        let mut hir = runtime.lower_to_hir(root.clone()).unwrap();
        zyntax_compiler::drop_insert::run_module(&mut hir);
        let body = hir
            .functions
            .values()
            .find(|f| f.name.resolve_global().as_deref() == Some(&score))
            .unwrap();
        assert!(body.blocks.values().flat_map(|b| &b.instructions).any(|i| matches!(i, HirInstruction::GetElementPtr{ty:HirType::Ptr(t),..} if **t==HirType::F64)));
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
            2,
            "array buffer and header each have one owner"
        );
        let functions = runtime.compile_typed_program(root).unwrap();
        let functions: Vec<_> = functions
            .iter()
            .filter(|n| n.starts_with("ts$m$"))
            .collect();
        let check = |runtime: &mut TieredRuntime| {
            for n in [-5.0, 0.0, 3.25] {
                assert_eq!(
                    runtime.call_raw(&score, &[ZyntaxValue::Float(n)]).unwrap(),
                    ZyntaxValue::Float(8.0 + n)
                );
            }
        };
        assert_eq!(runtime.statistics().baseline_functions, 0);
        check(&mut runtime);
        runtime
            .optimize_function(&score, OptimizationTier::Baseline)
            .unwrap();
        check(&mut runtime);
        for function in &functions {
            runtime
                .optimize_function(function, OptimizationTier::Baseline)
                .unwrap();
        }
        check(&mut runtime);
        #[cfg(feature = "llvm-backend")]
        {
            runtime
                .optimize_function(&score, OptimizationTier::Optimized)
                .unwrap();
            let wait = |runtime: &TieredRuntime, expected| {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                while runtime.statistics().optimized_functions < expected {
                    assert!(std::time::Instant::now() < deadline, "LLVM did not install");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            };
            wait(&runtime, 1);
            check(&mut runtime);
            for function in &functions {
                runtime
                    .optimize_function(function, OptimizationTier::Optimized)
                    .unwrap();
            }
            wait(&runtime, functions.len());
            check(&mut runtime);
        }
        assert!(!zyntax_compiler::collector::is_enabled());
        runtime.stop();
    }
}
