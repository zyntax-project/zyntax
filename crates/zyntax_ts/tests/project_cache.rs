use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use zyntax_ts::{CacheStatus, ErrorKind, Options, ProjectError, Soundness, load_project_cached};

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
fn cached_cli_preserves_initializers_arrays_and_generic_calls_across_tiers() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    write(
        root,
        "lib.ts",
        "console.log('library'); export function id<T>(x:T):T{return x;} export function array():number[]{return [2,4,8];}",
    );
    write(root, "barrel.ts", "export * from './lib';");
    write(
        root,
        "entry.ts",
        "import {id,array} from './barrel'; const a=array(); console.log(id(a[1])); console.log(id('text')); console.log('entry');",
    );
    let mut profiles = vec![
        vec!["--baseline-threshold", "2000000000"],
        vec!["--baseline-threshold", "1", "--no-osr"],
    ];
    if cfg!(feature = "llvm-backend") {
        profiles.push(vec!["--llvm", "--baseline-threshold", "1"]);
    }
    for profile in profiles {
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
        assert_eq!(load(root).unwrap().1, CacheStatus::Hit);
    }
    write(
        root,
        "lib.ts",
        "console.log('library'); export function id<T>(x:T):T{return x;} export function array():number[]{return [2,6,8];}",
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
    use zyntax_compiler::hir::{HirCallable, HirInstruction, HirType, Intrinsic};
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
    assert_eq!(load(root).unwrap().1, CacheStatus::Stored);
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
            let mut hir = runtime.lower_to_hir(program).unwrap();
            zyntax_compiler::drop_insert::run_module(&mut hir);
            let body = hir
                .functions
                .values()
                .find(|f| f.name.resolve_global().as_deref() == Some(&score))
                .unwrap();
            let instructions: Vec<_> = body.blocks.values().flat_map(|b| &b.instructions).collect();
            assert!(instructions.iter().any(|i|matches!(i,HirInstruction::GetElementPtr{ty:HirType::Ptr(t),..} if **t==HirType::F64)));
            assert_eq!(
                instructions
                    .iter()
                    .filter(|i| matches!(
                        i,
                        HirInstruction::Call {
                            callee: HirCallable::Intrinsic(Intrinsic::Free),
                            ..
                        }
                    ))
                    .count(),
                3
            );
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
