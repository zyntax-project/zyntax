//! Dense iteration keeps native element addresses and shared array ownership across tiers.

use std::sync::{Arc, Mutex};
use zyntax_compiler::hir::{CastOp, HirCallable, HirInstruction, HirModule, HirType, Intrinsic};
use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
use zyntax_typed_ast::{InternedString, arena::AstArena};

const SOURCE: &str = r#"
function score(n:number):number {
    let sum=0;for(const x of [n,n+1,n+2])sum+=x;return sum;
}
function zeroed(n:number):number {
    const xs=new Array<number>(n).fill(0);let sum=0;
    for(const x of xs)sum+=x;return sum+xs.length;
}
function booleans(n:number):number {
    const xs=new Array<boolean>(n).fill(false);let sum=0;
    for(const x of xs)if(x)sum++;return sum+xs.length;
}
function filled(n:number):number {
    const xs=new Array<number>(3).fill(n);let sum=0;
    for(const x of xs)sum+=x;return sum;
}
function nested(n:number):number {
    const row=[n];const rows=new Array<number[]>(3).fill(row);
    for(const entry of rows)entry.push(1);
    let sum=0;for(const entry of rows)for(const x of entry)sum+=x;
    return sum+row.length;
}
function live(n:number):number {
    let xs=[n,1];let sum=0;let steps=0;
    for(const x of xs){sum+=x;steps++;if(steps===1){for(let i=0;i<32;i++)xs.push(i);xs=[100];}}
    return sum;
}
function control(n:number):number {
    let sum=0;for(const x of [1,2,3,4,5]){if(x===2)continue;if(x===5)break;sum+=x;}
    return sum+n;
}
function tick(log:number[],n:number):number {log.push(n);return n;}
function order(n:number):number {
    const log:number[]=[];
    const full=new Array<number>(tick(log,3)).fill(tick(log,n));
    const empty=new Array<number>(0).fill(tick(log,7));
    return log[0]*1000+log[1]*100+log[2]*10+full[0]+empty.length;
}
function firstRow(rows:number[][]):number[] {
    for(const row of rows)return row;return [];
}
function escaped(n:number):number {
    const row=firstRow([[n],[7]]);row.push(2);return row[0]+row[1];
}
function unexpectedFill():number {console.log("unexpected fill");return 1;}
function guarded(n:number):number {
    const xs=new Array<number>(n).fill(unexpectedFill());return xs.length;
}
"#;

#[test]
fn array_iteration_uses_integer_offsets_and_typed_loads() {
    let mut program = zyntax_ts::parse_program(SOURCE, "iteration.ts", Default::default()).unwrap();
    let mut context = LoweringContext::new(
        InternedString::new_global("iteration"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let module = context.lower_program(&mut program).unwrap();
    let score = module
        .functions
        .values()
        .find(|f| f.name.resolve_global().unwrap().ends_with("$fn$score"))
        .unwrap();
    let instructions: Vec<_> = score
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .collect();
    assert!(instructions.iter().any(|i| matches!(i, HirInstruction::GetElementPtr {ty:HirType::Ptr(element),..} if **element==HirType::F64)));
    assert!(instructions.iter().any(|i| matches!(
        i,
        HirInstruction::Binary {
            ty: HirType::I64,
            ..
        }
    )));
    assert!(!instructions.iter().any(|i| matches!(
        i,
        HirInstruction::Cast {
            op: CastOp::FpToSi | CastOp::FpToUi,
            ..
        }
    )));
    let mut isolated = HirModule::new(InternedString::new_global("iteration_score"));
    isolated.functions.insert(score.id, score.clone());
    isolated.automatic_release = true;
    zyntax_compiler::drop_insert::run_module(&mut isolated);
    let frees = isolated.functions[&score.id]
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
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
    let allocations = isolated.functions[&score.id]
        .blocks
        .values()
        .flat_map(|b| &b.instructions)
        .filter(|i| {
            matches!(
                i,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc),
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        allocations, 2,
        "array iteration must not allocate an iterator object"
    );
    assert_eq!(
        frees, 2,
        "the iterator's temporary owns one header and one buffer"
    );
    zyntax_compiler::run_interp_safe_opts(&mut isolated);
    let mut backend = zyntax_compiler::cranelift_backend::CraneliftBackend::new().unwrap();
    backend.set_capture_ir(true);
    backend.compile_module(&isolated).unwrap();
    let (clif, _) = backend.take_captured_ir().unwrap();
    assert!(clif.contains("load.f64"), "{clif}");
    assert!(!clif.contains("fcvt_to_sint"), "{clif}");
}

#[test]
fn array_iteration_and_shared_fill_survive_actual_tier_promotion() {
    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};

    let program = zyntax_ts::parse_program(SOURCE, "iteration.ts", Default::default()).unwrap();
    let config = TieredConfig {
        baseline_threshold: u32::MAX,
        enable_osr: false,
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    let names: Vec<_> = runtime
        .compile_typed_program(program)
        .unwrap()
        .into_iter()
        .filter(|n| n.starts_with("ts$m$iteration_2ets$fn$"))
        .collect();
    let check = |runtime: &mut TieredRuntime| {
        for n in [0.0, 1.0, 3.0, 7.0] {
            for (name, expected) in [
                ("score", 3.0 * n + 3.0),
                ("zeroed", n),
                ("booleans", n),
                ("filled", 3.0 * n),
                ("nested", 3.0 * n + 13.0),
                ("live", n + 497.0),
                ("control", n + 8.0),
                ("order", 3070.0 + 101.0 * n),
                ("escaped", n + 2.0),
            ] {
                assert_eq!(
                    runtime
                        .call_raw(
                            &format!("ts$m$iteration_2ets$fn${name}"),
                            &[ZyntaxValue::Float(n)]
                        )
                        .unwrap(),
                    ZyntaxValue::Float(expected),
                    "{name}({n})"
                );
            }
        }
    };
    check(&mut runtime);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    for name in &names {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    assert_eq!(runtime.statistics().baseline_functions, names.len());
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        runtime
            .optimize_function("ts$m$iteration_2ets$fn$tick", OptimizationTier::Optimized)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while runtime.statistics().optimized_functions < 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "array helper LLVM promotion did not install"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check(&mut runtime);
        for name in &names {
            if name.ends_with("$fn$tick") {
                continue;
            }
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while runtime.statistics().optimized_functions < names.len() {
            assert!(
                std::time::Instant::now() < deadline,
                "array iteration LLVM promotion did not install"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check(&mut runtime);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}

#[test]
fn invalid_lengths_trap_after_actual_tier_promotion() {
    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};
    const CHILD: &str = "ZYNTAX_TS_FILL_TRAP_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        let config = TieredConfig {
            baseline_threshold: u32::MAX,
            enable_osr: false,
            #[cfg(feature = "llvm-backend")]
            tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
            ..Default::default()
        };
        let mut runtime = TieredRuntime::new(config).unwrap();
        zyntax_ts::register_runtime(&mut runtime).unwrap();
        let program = zyntax_ts::parse_program(SOURCE, "iteration.ts", Default::default()).unwrap();
        runtime.compile_typed_program(program).unwrap();
        let name = "ts$m$iteration_2ets$fn$guarded";
        if mode != "interpreter" {
            runtime
                .optimize_function(name, OptimizationTier::Baseline)
                .unwrap();
            assert_eq!(runtime.statistics().baseline_functions, 1);
        }
        if mode == "llvm" {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while runtime.statistics().optimized_functions < 1 {
                assert!(
                    std::time::Instant::now() < deadline,
                    "length guard LLVM promotion did not install"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        eprintln!("length guard tier ready: {mode}");
        let value = std::env::var("ZYNTAX_TS_FILL_BAD_LENGTH")
            .unwrap()
            .parse::<f64>()
            .unwrap();
        runtime
            .call_raw(name, &[ZyntaxValue::Float(value)])
            .unwrap();
        panic!("invalid length returned");
    }
    let mut modes = vec!["interpreter", "baseline"];
    if cfg!(feature = "llvm-backend") {
        modes.push("llvm");
    }
    for mode in modes {
        for value in ["-1", "1.5", "NaN", "inf", "-inf", "4294967296", "1e300"] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "invalid_lengths_trap_after_actual_tier_promotion",
                    "--nocapture",
                ])
                .env(CHILD, mode)
                .env("ZYNTAX_TS_FILL_BAD_LENGTH", value)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while child.try_wait().unwrap().is_none() {
                if std::time::Instant::now() > deadline {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("length guard did not finish: {mode} {value}");
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            let output = child.wait_with_output().unwrap();
            let error = String::from_utf8_lossy(&output.stderr);
            assert_eq!(output.status.code(), Some(1), "{mode} {value}: {error}");
            assert!(
                error.contains(&format!("length guard tier ready: {mode}")),
                "{error}"
            );
            assert!(
                error.contains("RangeError: Invalid array length"),
                "{error}"
            );
            assert!(!String::from_utf8_lossy(&output.stdout).contains("unexpected fill"));
        }
    }
}
