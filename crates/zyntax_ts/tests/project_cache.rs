use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use zyntax_ts::{CacheStatus, ErrorKind, Options, ProjectError, Soundness, load_project_cached};

// Tier callbacks are process-wide; in-process runtime fixtures run serially.
static RUNTIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn write(root: &Path, file: &str, source: &str) {
    let path = root.join(file);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, source).unwrap();
}

fn load(root: &Path) -> Result<(zyntax_ts::SourceProject, CacheStatus), ProjectError> {
    load_project_cached(
        root.join("entry.ts"),
        Options::default(),
        root.join("cache"),
    )
}

fn cached_file(root: &Path) -> PathBuf {
    std::fs::read_dir(root.join("cache"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().unwrap() == "zsnap")
        .unwrap()
}

fn rewrite_manifest(root: &Path, change: impl FnOnce(&mut Value)) {
    let path = cached_file(root);
    let bytes = std::fs::read(&path).unwrap();
    let length = u32::from_le_bytes(bytes[9..13].try_into().unwrap()) as usize;
    let mut manifest: Value = ciborium::from_reader(&bytes[13..13 + length]).unwrap();
    change(&mut manifest["cache"]["key"]);
    let mut encoded = Vec::new();
    ciborium::into_writer(&manifest, &mut encoded).unwrap();
    let mut replacement = bytes[..9].to_vec();
    replacement.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
    replacement.extend_from_slice(&encoded);
    replacement.extend_from_slice(&bytes[13 + length..]);
    std::fs::write(path, replacement).unwrap();
}

#[test]
fn checked_graph_round_trips_and_moves_only_runtime_dependencies() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "types.ts", "export type Point={readonly x:number};");
    write(root, "lib.ts", "export function id<T>(x:T):T{return x;}");
    write(root, "barrel.ts", "export {id} from './lib';");
    write(
        root,
        "entry.ts",
        "import type {Point} from './types'; import {id} from './barrel';
         const p:Point={x:4}; console.log(id(p.x));",
    );
    let (cold, status) = load(root).unwrap();
    assert_eq!(status, CacheStatus::Stored);
    let (warm, status) = load(root).unwrap();
    assert_eq!(status, CacheStatus::Hit);
    assert_eq!(warm.module_count(), 4);
    assert_eq!(warm.entry().source_files, cold.entry().source_files);
    assert_eq!(
        serde_json::to_value(warm.entry()).unwrap(),
        serde_json::to_value(cold.entry()).unwrap(),
    );
    let (_, resolver) = warm.into_parts();
    let barrel = root.join("barrel.ts").canonicalize().unwrap();
    let barrel = resolver(barrel.to_str().unwrap()).unwrap().unwrap();
    assert!(barrel.program().checker_interface.is_some());
    let library = root.join("lib.ts").canonicalize().unwrap();
    let library = resolver(library.to_str().unwrap()).unwrap().unwrap();
    assert!(library.program().declarations.iter().any(|d| matches!(
        &d.node, zyntax_typed_ast::TypedDeclaration::Function(f) if f.body.is_some()
    )));
    let types = root.join("types.ts").canonicalize().unwrap();
    assert!(resolver(types.to_str().unwrap()).unwrap().is_none());
    // A different spelling of the entry shares the canonical cache identity.
    let (_, status) = load_project_cached(
        root.join("./entry.ts"),
        Options::default(),
        root.join("cache"),
    )
    .unwrap();
    assert_eq!(status, CacheStatus::Hit);
}

#[test]
fn transitive_and_type_only_edits_recheck_source_and_preserve_diagnostics() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "types.ts", "export type Point=number;");
    write(root, "lib.ts", "export function f():number{return 1;}");
    write(root, "barrel.ts", "export {f} from './lib';");
    write(
        root,
        "entry.ts",
        "import type {Point} from './types'; import {f} from './barrel'; const p:Point=f(); console.log(p);",
    );
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    let file = root.join("lib.ts");
    let modified = std::fs::metadata(&file).unwrap().modified().unwrap();
    write(root, "lib.ts", "export function f():number{return 2;}");
    std::fs::File::open(file)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(modified))
        .unwrap();
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    write(root, "types.ts", "export type Point=boolean;");
    let error = load(root).err().unwrap();
    let ProjectError::Diagnostic {
        file,
        source_code,
        error,
    } = error
    else {
        panic!("{error}")
    };
    assert_eq!(error.kind, ErrorKind::Type);
    assert!(file.ends_with("entry.ts"));
    assert!(source_code.contains("p:Point"));
    write(root, "types.ts", "export type Point=number;");
    assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    std::fs::remove_file(root.join("lib.ts")).unwrap();
    let ProjectError::Diagnostic { file, error, .. } = load(root).err().unwrap() else {
        panic!()
    };
    assert!(file.ends_with("barrel.ts"));
    assert_eq!(error.kind, ErrorKind::Type);
}

#[test]
fn resolution_changes_and_symlink_retargeting_invalidate_the_graph() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "library/index.ts",
        "export function f():number{return 1;}",
    );
    write(
        root,
        "entry.ts",
        "import {f} from './library'; console.log(f());",
    );
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    write(root, "library.ts", "export function f():number{return 2;}");
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    #[cfg(unix)]
    {
        std::fs::remove_file(root.join("library.ts")).unwrap();
        write(root, "first.ts", "export function f():number{return 3;}");
        write(root, "second.ts", "export function f():number{return 4;}");
        std::os::unix::fs::symlink(root.join("first.ts"), root.join("library.ts")).unwrap();
        assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
        assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
        std::fs::remove_file(root.join("library.ts")).unwrap();
        std::os::unix::fs::symlink(root.join("second.ts"), root.join("library.ts")).unwrap();
        assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    }
}

#[test]
fn corrupt_and_incompatible_entries_are_replaced_atomically() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "entry.ts", "console.log(3);");
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    for field in ["build", "options", "entry"] {
        rewrite_manifest(root, |manifest| manifest[field] = Value::from("obsolete"));
        assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
        assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    }
    let path = cached_file(root);
    let good = std::fs::read(&path).unwrap();
    for bytes in [
        vec![],
        good[..12].to_vec(),
        good[..good.len() - 1].to_vec(),
        {
            let mut corrupt = good.clone();
            *corrupt.last_mut().unwrap() ^= 1;
            corrupt
        },
        {
            let mut corrupt = good.clone();
            corrupt[9..13].copy_from_slice(&u32::MAX.to_le_bytes());
            corrupt
        },
    ] {
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
        assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    }
    assert_eq!(std::fs::read_dir(root.join("cache")).unwrap().count(), 1);
}

#[test]
fn soundness_options_and_unavailable_cache_never_bypass_checking() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "entry.ts",
        "type Read={readonly x:number}; type Write={x:number}; const r:Read={x:1}; const w:Write=r; console.log(w.x);",
    );
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    let error = load_project_cached(
        root.join("entry.ts"),
        Options {
            soundness: Soundness::Checked,
            ..Default::default()
        },
        root.join("cache"),
    )
    .err()
    .unwrap();
    let ProjectError::Diagnostic { error, .. } = error else {
        panic!()
    };
    assert_eq!(error.kind, ErrorKind::Soundness);
    write(root, "not-a-directory", "file");
    let (_, status) = load_project_cached(
        root.join("entry.ts"),
        Options::default(),
        root.join("not-a-directory/cache"),
    )
    .unwrap();
    assert_eq!(status, CacheStatus::Unavailable);
}

#[test]
fn checked_cache_upgrades_to_one_lowered_project_and_validates_all_sources() {
    let _runtime = RUNTIME.lock().unwrap();
    use zyntax_embed::{Snapshot, TieredConfig, TieredRuntime};

    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "types.ts", "export type Scalar=number;");
    write(root, "lib.ts", "export function id<T>(x:T):T{return x;}");
    write(
        root,
        "entry.ts",
        "import type {Scalar} from './types'; import {id} from './lib'; export function keep<T>(x:T):T{return id(x);} const n:Scalar=2; console.log(keep(n));",
    );
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    let prepare = |runtime: &mut TieredRuntime| {
        zyntax_ts::load_project_for_runtime(
            root.join("entry.ts"),
            Options::default(),
            root.join("cache"),
            runtime,
        )
    };
    let (project, status) = prepare(&mut runtime).unwrap();
    assert_eq!(status, CacheStatus::Stored);
    assert_eq!(project.module_count(), 3);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    assert_eq!(runtime.statistics().optimized_functions, 0);
    let entry = root.join("entry.ts").canonicalize().unwrap();
    let snapshot = Snapshot::load_owned(std::fs::read(cached_file(root)).unwrap()).unwrap();
    assert_eq!(snapshot.module_names().count(), 1);
    assert_eq!(snapshot.cached_sources().unwrap().len(), 3);
    assert!(snapshot.carries_hir(entry.to_str().unwrap()));
    let import = snapshot.module(entry.to_str().unwrap()).unwrap().unwrap();
    assert!(import.program().checker_interface.is_some());
    assert!(
        import
            .program()
            .declarations
            .iter()
            .all(|d| !matches!(&d.node,
        zyntax_typed_ast::TypedDeclaration::Function(f) if f.body.is_some()))
    );
    let name = zyntax_ts::ModuleId::new(entry.to_str().unwrap())
        .function_name("keep")
        .resolve_global()
        .unwrap();
    assert!(
        !import
            .hir()
            .unwrap()
            .by_name(&name)
            .unwrap()
            .signature
            .type_params
            .is_empty()
    );
    assert_eq!(prepare(&mut runtime).unwrap().1, CacheStatus::Hit);
    assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    write(root, "types.ts", "export type Scalar=boolean;");
    let ProjectError::Diagnostic { error, .. } = prepare(&mut runtime).err().unwrap() else {
        panic!()
    };
    assert_eq!(error.kind, ErrorKind::Type);
    write(root, "types.ts", "export type Scalar=number;");
    assert_eq!(prepare(&mut runtime).unwrap().1, CacheStatus::Hit);
    write(
        root,
        "lib.ts",
        "export function id<T>(x:T):T{console.log('changed');return x;}",
    );
    assert_eq!(prepare(&mut runtime).unwrap().1, CacheStatus::Stored);
    assert_eq!(std::fs::read_dir(root.join("cache")).unwrap().count(), 1);
    let snapshot = std::sync::Arc::new(
        Snapshot::load_owned(std::fs::read(cached_file(root)).unwrap()).unwrap(),
    );
    runtime
        .install_snapshot(std::sync::Arc::clone(&snapshot))
        .unwrap();
    let interface = snapshot
        .module_interface(entry.to_str().unwrap())
        .unwrap()
        .unwrap();
    let client_id = zyntax_ts::ModuleId::new("cached-client");
    let client = zyntax_ts::parse_module_with_dependencies(
        "import {keep} from 'saved'; export function client():boolean{return keep(true);}",
        "client.ts",
        client_id,
        Options::default(),
        &[zyntax_ts::ModuleDependency {
            specifier: "saved",
            program: interface.program(),
        }],
    )
    .unwrap();
    let name = client_id.function_name("client").resolve_global().unwrap();
    runtime.declare_entry_points([name.as_str()]);
    runtime.enter_only_through_entry_points();
    runtime.compile_typed_program(client).unwrap();
    assert_eq!(
        runtime.call_raw(&name, &[]).unwrap(),
        zyntax_embed::ZyntaxValue::Bool(true)
    );
    runtime
        .optimize_function(
            &name,
            zyntax_compiler::tiered_backend::OptimizationTier::Baseline,
        )
        .unwrap();
    assert_eq!(
        runtime.call_raw(&name, &[]).unwrap(),
        zyntax_embed::ZyntaxValue::Bool(true)
    );
    runtime.stop();
}

#[test]
fn an_unwritable_lowered_cache_still_returns_executable_hir() {
    let _runtime = RUNTIME.lock().unwrap();
    use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};
    for background in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        write(
            root,
            "entry.ts",
            "export function answer():number{return 42;}",
        );
        write(root, "not-a-directory", "file");
        let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
        let (project, status) = if background {
            let (project, status, worker) = zyntax_ts::load_project_for_runtime_background(
                root.join("entry.ts"),
                Options::default(),
                root.join("not-a-directory/cache"),
                &mut runtime,
            )
            .unwrap();
            assert!(worker.is_none());
            (project, status)
        } else {
            zyntax_ts::load_project_for_runtime(
                root.join("entry.ts"),
                Options::default(),
                root.join("not-a-directory/cache"),
                &mut runtime,
            )
            .unwrap()
        };
        assert_eq!(status, CacheStatus::Unavailable);
        let answer = zyntax_ts::ModuleId::new(
            root.join("entry.ts")
                .canonicalize()
                .unwrap()
                .to_str()
                .unwrap(),
        )
        .function_name("answer")
        .resolve_global()
        .unwrap();
        let (program, resolver) = project.into_parts();
        runtime.add_compiled_import_resolver(resolver);
        runtime.declare_entry_points([answer.as_str()]);
        runtime.enter_only_through_entry_points();
        runtime.compile_typed_program(program).unwrap();
        assert_eq!(runtime.statistics().baseline_functions, 0);
        assert_eq!(
            runtime.call_raw(&answer, &[]).unwrap(),
            ZyntaxValue::Float(42.0)
        );
        runtime.stop();
    }
}

#[test]
fn background_preparation_uses_owned_hir_and_keeps_interpreter_first_execution() {
    let _runtime = RUNTIME.lock().unwrap();
    use zyntax_compiler::tiered_backend::OptimizationTier;
    use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "lib.ts", "export function keep<T>(x:T):T{return x;}");
    write(
        root,
        "entry.ts",
        "import {keep} from './lib'; export function score(n:number):number{const xs=[n,n+2];const ys=keep(xs);ys[1]+=3;return xs[0]+ys[1];}",
    );
    let config = TieredConfig {
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    let (project, status, worker) = zyntax_ts::load_project_for_runtime_background(
        root.join("entry.ts"),
        Options::default(),
        root.join("cache"),
        &mut runtime,
    )
    .unwrap();
    assert_eq!(status, CacheStatus::Stored);
    let worker = worker.expect("the bounded worker is available");
    let entry = root.join("entry.ts").canonicalize().unwrap();
    let score = zyntax_ts::ModuleId::new(entry.to_str().unwrap())
        .function_name("score")
        .resolve_global()
        .unwrap();
    let (program, resolver) = project.into_parts();
    runtime.add_compiled_import_resolver(resolver);
    runtime.declare_entry_points([score.as_str()]);
    runtime.enter_only_through_entry_points();
    runtime.compile_typed_program(program).unwrap();
    assert_eq!(runtime.statistics().baseline_functions, 0);
    assert_eq!(runtime.statistics().optimized_functions, 0);
    let check = |runtime: &mut TieredRuntime| {
        for n in [-4.0, 0.0, 3.25] {
            assert_eq!(
                runtime.call_raw(&score, &[ZyntaxValue::Float(n)]).unwrap(),
                ZyntaxValue::Float(n * 2.0 + 5.0)
            );
        }
    };
    check(&mut runtime);
    runtime
        .optimize_function(&score, OptimizationTier::Baseline)
        .unwrap();
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        runtime
            .optimize_function(&score, OptimizationTier::Optimized)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while runtime.statistics().optimized_functions == 0 {
            assert!(std::time::Instant::now() < deadline, "LLVM did not install");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check(&mut runtime);
    }
    runtime.stop();
    drop(runtime);
    assert!(worker.finish().unwrap());
    let snapshot =
        zyntax_embed::Snapshot::load_owned(std::fs::read(cached_file(root)).unwrap()).unwrap();
    let import = snapshot.module(entry.to_str().unwrap()).unwrap().unwrap();
    assert!(
        import
            .hir()
            .unwrap()
            .by_name(&score)
            .unwrap()
            .attributes
            .optimized
    );
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    let (_, status, worker) = zyntax_ts::load_project_for_runtime_background(
        &entry,
        Options::default(),
        root.join("cache"),
        &mut runtime,
    )
    .unwrap();
    assert_eq!(status, CacheStatus::Hit);
    assert!(worker.is_none());
    runtime.stop();
}

#[test]
fn cancelled_snapshot_optimization_leaves_input_untouched() {
    let _runtime = RUNTIME.lock().unwrap();
    let mut runtime = zyntax_embed::TieredRuntime::new(Default::default()).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    let program = zyntax_ts::parse_program(
        "export function f(n:number):number{const xs=[n,n+2];return xs[1];}",
        "cancel.ts",
        Default::default(),
    )
    .unwrap();
    let (_, mut hir) = runtime.lower_typed_program_snapshot(program, None).unwrap();
    assert!(
        hir.functions
            .values()
            .filter(|f| !f.is_external && f.signature.type_params.is_empty())
            .all(|f| f.attributes.release_facts.is_some() && !f.attributes.optimized)
    );
    let format = zyntax_compiler::bytecode::Format::Split;
    let before = zyntax_compiler::bytecode::serialize_module(&hir, format).unwrap();
    let stopped = std::sync::atomic::AtomicBool::new(true);
    assert!(
        !zyntax_embed::optimize_snapshot_hir("cancel", &mut hir, true, None, Some(&stopped))
            .unwrap()
    );
    assert_eq!(
        zyntax_compiler::bytecode::serialize_module(&hir, format).unwrap(),
        before
    );
    runtime.stop();
}

#[test]
fn closed_cache_keys_preserve_host_roots_and_do_not_replace_open_projects() {
    let _runtime = RUNTIME.lock().unwrap();
    use zyntax_embed::{TieredConfig, TieredRuntime, ZyntaxValue};
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "entry.ts",
        "export function first():number{return 41;} export function second():number{return 99;}",
    );
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
    let entry = root.join("entry.ts").canonicalize().unwrap();
    let id = zyntax_ts::ModuleId::new(entry.to_str().unwrap());
    for (selected, omitted, answer) in [("first", "second", 41.0), ("second", "first", 99.0)] {
        let name = id.function_name(selected).resolve_global().unwrap();
        let missing = id.function_name(omitted).resolve_global().unwrap();
        let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
        runtime.declare_entry_points([name.as_str()]);
        runtime.enter_only_through_entry_points();
        let (project, status, worker) = zyntax_ts::load_project_for_runtime_background(
            &entry,
            Options::default(),
            root.join("cache"),
            &mut runtime,
        )
        .unwrap();
        assert_eq!(
            status,
            CacheStatus::Stored,
            "each root set gets a separate HIR artifact"
        );
        assert!(project.entry().declarations.iter().all(|decl| !matches!(
            &decl.node, zyntax_typed_ast::TypedDeclaration::Function(f) if f.name.resolve_global().as_deref() == Some(missing.as_str())
        )), "closed metadata omits unreached declarations as well as their HIR");
        let (program, resolver) = project.into_parts();
        runtime.add_compiled_import_resolver(resolver);
        runtime.compile_typed_program(program).unwrap();
        assert_eq!(runtime.statistics().baseline_functions, 0);
        assert_eq!(
            runtime.call_raw(&name, &[]).unwrap(),
            ZyntaxValue::Float(answer)
        );
        assert!(runtime.call_raw(&missing, &[]).is_err());
        runtime.stop();
        assert!(
            worker.is_none(),
            "small closed preparations complete inline"
        );
    }
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    let (project, status) = zyntax_ts::load_project_for_runtime(
        &entry,
        Options::default(),
        root.join("cache"),
        &mut runtime,
    )
    .unwrap();
    assert_eq!(
        status,
        CacheStatus::Stored,
        "the open AST entry remains available for a full-library upgrade"
    );
    let (program, resolver) = project.into_parts();
    runtime.add_compiled_import_resolver(resolver);
    runtime.compile_typed_program(program).unwrap();
    for (name, value) in [("first", 41.0), ("second", 99.0)] {
        assert_eq!(
            runtime
                .call_raw(&id.function_name(name).resolve_global().unwrap(), &[])
                .unwrap(),
            ZyntaxValue::Float(value)
        );
    }
    runtime.stop();
    assert_eq!(std::fs::read_dir(root.join("cache")).unwrap().count(), 3);
}

#[test]
fn large_closed_graphs_prepare_on_a_worker_and_survive_runtime_drop() {
    let _runtime = RUNTIME.lock().unwrap();
    use zyntax_embed::{TieredConfig, TieredRuntime};
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut source = String::new();
    for i in 0..129 {
        source.push_str(&format!(
            "function f{i}(x:number):number{{return f{}(x)+1;}}\n",
            i + 1
        ));
    }
    source.push_str("function f129(x:number):number{return x;} console.log(f0(0));");
    write(root, "entry.ts", &source);
    let mut runtime = TieredRuntime::new(TieredConfig::default()).unwrap();
    runtime.enter_only_through_entry_points();
    let (project, status, worker) = zyntax_ts::load_project_for_runtime_background(
        root.join("entry.ts"),
        Options::default(),
        root.join("cache"),
        &mut runtime,
    )
    .unwrap();
    assert_eq!(status, CacheStatus::Stored);
    let worker = worker.expect("the large reached graph uses the worker");
    // The worker keeps no interpreter, backend or plugin lifetime from this runtime.
    runtime.stop();
    drop(runtime);
    assert!(worker.finish().unwrap());
    assert_eq!(project.module_count(), 1);
}

#[test]
fn cached_cli_preserves_initializers_arrays_and_generic_calls_across_tiers() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "lib.ts",
        "console.log('library'); export default function<T>(x:T):T{return x;} export function array():number[]{return [2,4,8];}",
    );
    write(
        root,
        "barrel.ts",
        "export * from './lib'; export {default} from './lib';",
    );
    write(
        root,
        "entry.ts",
        "import id,* as lib from './barrel'; const a=lib.array(); console.log(id(a[1])); console.log(lib.default('text')); console.log('entry');",
    );
    let mut profiles = vec![
        vec!["--baseline-threshold", "2000000000"],
        vec!["--baseline-threshold", "1", "--no-osr"],
    ];
    if cfg!(feature = "llvm-backend") {
        profiles.push(vec!["--llvm", "--baseline-threshold", "1"]);
    }
    for (profile_index, profile) in profiles.into_iter().enumerate() {
        for _ in 0..2 {
            let output = Command::new(env!("CARGO_BIN_EXE_zynode"))
                .arg("run")
                .args(&profile)
                .arg("--cache-dir")
                .arg(root.join("cache"))
                .arg(root.join("entry.ts"))
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                String::from_utf8(output.stdout).unwrap(),
                "library\n4\ntext\nentry\n"
            );
        }
        assert_eq!(
            load(root).unwrap().1,
            if profile_index == 0 {
                CacheStatus::Stored
            } else {
                CacheStatus::Hit
            }
        );
    }
    write(
        root,
        "lib.ts",
        "console.log('library'); export default function id<T>(x:T):T{return x;} export function array():number[]{return [2,6,8];}",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_zynode"))
        .arg("run")
        .arg("--cache-dir")
        .arg(root.join("cache"))
        .arg(root.join("entry.ts"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "library\n6\ntext\nentry\n"
    );
}

#[test]
fn warm_snapshots_preserve_native_layouts_release_sites_and_real_promotions() {
    let _runtime = RUNTIME.lock().unwrap();
    use zyntax_compiler::hir::{HirCallable, HirInstruction, HirTerminator, HirType, Intrinsic};
    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};

    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(root, "lib.ts", "type Private={x:number;y:number}; export type Read={readonly x:number};
        export function make(n:number):Read{const p:Private={x:n,y:4};return p;}
        export function keep<T>(x:T):T{return x;}
        export function alias(xs:number[]):number[]{return xs;}
        export function numbers(n:number):number[]{return [n,n+2];}
        function pick<T>(xs:T[],fallback:T):T{if(xs.length>1)return xs[1];return fallback;} export {pick};");
    write(root, "barrel.ts", "export * from './lib';");
    write(
        root,
        "entry.ts",
        "import type {Read} from './barrel'; import {make,keep,alias,numbers,pick} from './barrel';
        export function score(n:number):number{const p:Read=make(n);const q=keep(p);
        const xs=numbers(q.x);const ys=alias(xs);ys[1]=ys[1]+3;return pick(xs,0)+p.x;}",
    );
    let mut preparation = TieredRuntime::new(TieredConfig::default()).unwrap();
    assert_eq!(
        zyntax_ts::load_project_for_runtime(
            root.join("entry.ts"),
            Options::default(),
            root.join("cache"),
            &mut preparation
        )
        .unwrap()
        .1,
        CacheStatus::Stored
    );
    assert_eq!(preparation.statistics().baseline_functions, 0);
    assert_eq!(preparation.statistics().optimized_functions, 0);
    preparation.stop();
    let score = zyntax_ts::ModuleId::new(
        root.join("entry.ts")
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap(),
    )
    .function_name("score")
    .resolve_global()
    .unwrap();
    for inspect in [true, false] {
        let (project, status) = load(root).unwrap();
        assert_eq!(status, CacheStatus::Hit);
        let (program, resolver) = project.into_parts();
        let mut runtime = TieredRuntime::new(TieredConfig {
            baseline_threshold: u32::MAX,
            enable_osr: false,
            #[cfg(feature = "llvm-backend")]
            tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
            ..Default::default()
        })
        .unwrap();
        zyntax_ts::register_runtime(&mut runtime).unwrap();
        runtime.add_compiled_import_resolver(resolver);
        runtime.declare_entry_points([score.as_str()]);
        runtime.enter_only_through_entry_points();
        if inspect {
            let snapshot =
                zyntax_embed::Snapshot::load_owned(std::fs::read(cached_file(root)).unwrap())
                    .unwrap();
            let entry = root.join("entry.ts").canonicalize().unwrap();
            let import = snapshot.module(entry.to_str().unwrap()).unwrap().unwrap();
            let hir = import.hir().unwrap();
            let signature = hir.by_name(&score).unwrap();
            assert!(signature.attributes.optimized);
            assert!(signature.blocks.is_empty());
            let body = hir.function(signature.id).unwrap();
            let instructions: Vec<_> = body.blocks.values().flat_map(|b| &b.instructions).collect();
            assert!(instructions.iter().any(|i|matches!(i,HirInstruction::GetElementPtr{ty:HirType::Ptr(t),..} if **t==HirType::F64)));
            let mut paths = vec![(body.entry_block, 0, 0)];
            let mut returns = 0;
            while let Some((id, releases, depth)) = paths.pop() {
                assert!(depth < body.blocks.len(), "unexpected cycle in score");
                let block = &body.blocks[&id];
                let releases = releases
                    + block
                        .instructions
                        .iter()
                        .filter(|i| {
                            matches!(
                                i,
                                HirInstruction::Call {
                                    callee: HirCallable::Intrinsic(Intrinsic::Free),
                                    ..
                                }
                            )
                        })
                        .count();
                match block.terminator {
                    HirTerminator::Branch { target } => paths.push((target, releases, depth + 1)),
                    HirTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        paths.push((true_target, releases, depth + 1));
                        paths.push((false_target, releases, depth + 1));
                    }
                    HirTerminator::Return { .. } => {
                        assert_eq!(
                            releases, 3,
                            "object, array header and array buffer released once per path"
                        );
                        returns += 1;
                    }
                    _ => panic!("unexpected score terminator"),
                }
            }
            assert!(returns > 0);
            runtime.stop();
            continue;
        }
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
        assert_eq!(runtime.statistics().optimized_functions, 0);
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
            for name in &natives {
                runtime
                    .optimize_function(name, OptimizationTier::Optimized)
                    .unwrap();
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            while runtime.statistics().optimized_functions < natives.len() {
                assert!(std::time::Instant::now() < deadline, "LLVM did not install");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            check(&mut runtime);
        }
        assert!(!zyntax_compiler::collector::is_enabled());
        runtime.stop();
    }
}
