//! Scalar optionals use the shared union ABI without managed allocations.

use zyntax_ts::{ErrorKind, Options};

const SOURCE: &str = r#"
    function make(n: number): number | undefined { if (n > 0) return n; }
    function echo(n: number | undefined): number | undefined { return n; }
    function partial(n: number) { if (n > 0) return n + 2; }
    function read(n: number | undefined) { if (n === undefined) return -10; return n + 1; }
    function flag(n: number): boolean | undefined { if (n > 0) return false; }
    function flagScore(b: boolean | undefined) { if (b === undefined) return -1; return b ? 2 : 3; }
    function absent(): undefined {}
    function absenceScore() { return absent() === undefined ? 1 : 0; }
    function nested(n: number): number | undefined {
        const m = make(n);
        return m !== undefined ? (m > 1 ? m : undefined) : undefined;
    }
    function nestedScore(n: number) { return read(nested(n)); }
    function branches(n: number): number | undefined { return n > 0 ? n : -n; }
    function branchScore(n: number) { return read(branches(n)); }
    function changes(n: number | undefined) {
        if (n === undefined) return -10;
        n += 2; const old = n++; return n + old;
    }
    function changeScore(n: number) { return changes(make(n)); }
    function score(n: number) { return read(echo(make(n))) + read(partial(n)) + flagScore(flag(n)); }
    function loop(n: number) {
        let value: number | undefined = make(n);
        let sum = 0;
        while (value !== undefined) { sum += value; value = undefined; }
        return sum;
    }
"#;

#[test]
fn optionals_keep_tagged_storage_without_heap_boxes() {
    use std::sync::{Arc, Mutex};
    use zyntax_compiler::hir::{HirCallable, HirInstruction, HirType, Intrinsic};
    use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
    use zyntax_typed_ast::{InternedString, arena::AstArena};
    let mut program = zyntax_ts::parse_program(SOURCE, "optionals.ts", Options::default()).unwrap();
    let mut context = LoweringContext::new(
        InternedString::new_global("optionals"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let mut module = context.lower_program(&mut program).unwrap();
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    module.automatic_release = true;
    zyntax_compiler::drop_insert::run_module(&mut module);
    let mut creates = 0;
    let mut extracts = 0;
    for function in module.functions.values() {
        let name = function.name.resolve_global().unwrap();
        if [
            "ts$fn$make",
            "ts$fn$echo",
            "ts$fn$partial",
            "ts$fn$flag",
            "ts$fn$absent",
            "ts$fn$nested",
            "ts$fn$branches",
        ]
        .contains(&name.as_str())
        {
            assert!(
                matches!(function.signature.returns.as_slice(), [HirType::Union(_)]),
                "{name}"
            );
        }
        for instruction in function.blocks.values().flat_map(|b| &b.instructions) {
            match instruction {
                HirInstruction::CreateUnion { .. } => creates += 1,
                HirInstruction::ExtractUnionValue { .. } => extracts += 1,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc | Intrinsic::Free),
                    ..
                } => panic!("scalar union allocated: {name}"),
                HirInstruction::Call {
                    callee: HirCallable::Symbol(symbol),
                    ..
                } => assert!(
                    !symbol.contains("Dynamic") && !symbol.contains("Box"),
                    "{symbol}"
                ),
                _ => {}
            }
        }
    }
    assert!(creates > 0 && extracts > 0);
}

#[test]
fn optional_calls_agree_during_mixed_native_promotions() {
    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};
    let program = zyntax_ts::parse_program(SOURCE, "tiers.ts", Options::default()).unwrap();
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
        .filter(|name| name.starts_with("ts$fn$"))
        .collect();
    let check = |runtime: &mut TieredRuntime| {
        assert_eq!(
            runtime.call_raw("ts$fn$absenceScore", &[]).unwrap(),
            ZyntaxValue::Float(1.0)
        );
        for (input, expected) in [(3.0, 13.0), (0.0, -21.0), (-2.0, -21.0), (7.0, 21.0)] {
            assert_eq!(
                runtime
                    .call_raw("ts$fn$score", &[ZyntaxValue::Float(input)])
                    .unwrap(),
                ZyntaxValue::Float(expected)
            );
            assert_eq!(
                runtime
                    .call_raw("ts$fn$changeScore", &[ZyntaxValue::Float(input)])
                    .unwrap(),
                ZyntaxValue::Float(if input > 0.0 {
                    2.0 * input + 5.0
                } else {
                    -10.0
                })
            );
            assert_eq!(
                runtime
                    .call_raw("ts$fn$nestedScore", &[ZyntaxValue::Float(input)])
                    .unwrap(),
                ZyntaxValue::Float(if input > 1.0 { input + 1.0 } else { -10.0 })
            );
            assert_eq!(
                runtime
                    .call_raw("ts$fn$branchScore", &[ZyntaxValue::Float(input)])
                    .unwrap(),
                ZyntaxValue::Float(input.abs() + 1.0)
            );
            assert_eq!(
                runtime
                    .call_raw("ts$fn$loop", &[ZyntaxValue::Float(input)])
                    .unwrap(),
                ZyntaxValue::Float(if input > 0.0 { input } else { 0.0 })
            );
        }
    };
    check(&mut runtime);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    for name in ["ts$fn$make", "ts$fn$echo", "ts$fn$flag"] {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    for name in &names {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        let wait = |runtime: &TieredRuntime, installed| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
            while runtime.statistics().optimized_functions < installed {
                assert!(
                    std::time::Instant::now() < deadline,
                    "optional LLVM promotion did not install"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        for name in ["ts$fn$make", "ts$fn$echo", "ts$fn$flag"] {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, 3);
        check(&mut runtime);
        runtime
            .optimize_function("ts$fn$score", OptimizationTier::Optimized)
            .unwrap();
        wait(&runtime, 4);
        check(&mut runtime);
        for name in names.iter().filter(|name| {
            !["ts$fn$make", "ts$fn$echo", "ts$fn$flag", "ts$fn$score"].contains(&name.as_str())
        }) {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, names.len());
        check(&mut runtime);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}

#[test]
fn reassignment_and_control_flow_do_not_reuse_stale_narrowing() {
    for source in [
        "function f(n:number|undefined){if(n!==undefined){n=undefined;return n+1;}return 0;}",
        "function f(n:number|undefined,b:boolean){if(b)n=undefined;return n+1;}",
        "function f(n:number|undefined){while(n!==undefined){n=undefined;}return n+1;}",
        "function f(n:number|undefined){if(!n)return n+1;return 0;}",
        "function f(n:number|undefined){if(n===undefined)return n+1;return 0;}",
    ] {
        let error = zyntax_ts::parse_program(source, "flow.ts", Options::default()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Type, "{source}: {error}");
    }
}

#[test]
fn unproven_reference_and_callback_union_layouts_are_explicit() {
    for source in [
        "const value:string|undefined=undefined;",
        "type F=(n:number|undefined)=>number;",
        "function f(n:number|undefined){return n!;}",
        "type F=()=>number|undefined;",
    ] {
        let error = zyntax_ts::parse_program(source, "layout.ts", Options::default()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Unsupported, "{source}: {error}");
    }
}
