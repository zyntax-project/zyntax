//! Primitive alternatives retain tagged storage and native calls across tiers.
use zyntax_ts::{ErrorKind, Options};

const SOURCE: &str = r#"
function choose(n:number) { if(n>0)return n+0.5;return "cold"; }
function echo(x:string|number):number|string { return x; }
function wider(x:number|string):number|boolean|string { return x; }
function first<T>(x:T):T { return x; }
function read(x:number|string):number { if(typeof x==="string")return x==="cold"?-5:-20;return x*2; }
function three(n:number):number|boolean|string { if(n>0)return n;if(n===0)return false;return "cold"; }
function classify(x:string|boolean|number) {
    if(typeof x==="number")return x;
    if(typeof x==="boolean")return x?10:-10;
    return x==="cold"?3:-30;
}
function mutable(n:number) {
    let x:number|string="cold";
    while(n>0) { x=n;n-=1; }
    if(typeof x!=="string")return x;
    return -1;
}
function score(n:number) { return read(first(echo(choose(n))))+classify(wider(choose(n)))+classify(three(n))+mutable(n); }
function conditional(n:number) { const x=n>0?n:"cold";return read(x); }
function truth(n:number) { const x:number|string|boolean=three(n);return x?1:0; }
function equalNumber(n:number) { return echo(n)===n?1:0; }
function equalText() { return echo("cold")===echo("cold")?1:0; }
"#;

#[test]
fn union_calls_agree_before_and_after_each_native_promotion() {
    use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
    use zyntax_embed::{TieredRuntime, ZyntaxValue};
    let program = zyntax_ts::parse_program(SOURCE, "union.ts", Options::default()).unwrap();
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
        .filter(|n| n.starts_with("ts$fn$"))
        .collect();
    let check = |runtime: &mut TieredRuntime| {
        assert_eq!(
            runtime.call_raw("ts$fn$equalText", &[]).unwrap(),
            ZyntaxValue::Float(1.0)
        );
        for (n, expected) in [(0.0, 1.0), (-0.0, 1.0), (3.0, 1.0), (f64::NAN, 0.0)] {
            assert_eq!(
                runtime
                    .call_raw("ts$fn$equalNumber", &[ZyntaxValue::Float(n)])
                    .unwrap(),
                ZyntaxValue::Float(expected)
            );
        }
        for (n, expected) in [(2.0, 10.5), (0.0, -13.0), (-2.0, 0.0)] {
            assert_eq!(
                runtime
                    .call_raw("ts$fn$score", &[ZyntaxValue::Float(n)])
                    .unwrap(),
                ZyntaxValue::Float(expected),
                "score({n})"
            );
            assert_eq!(
                runtime
                    .call_raw("ts$fn$conditional", &[ZyntaxValue::Float(n)])
                    .unwrap(),
                ZyntaxValue::Float(if n > 0.0 { n * 2.0 } else { -5.0 })
            );
            assert_eq!(
                runtime
                    .call_raw("ts$fn$truth", &[ZyntaxValue::Float(n)])
                    .unwrap(),
                ZyntaxValue::Float(if n == 0.0 { 0.0 } else { 1.0 })
            );
        }
    };
    check(&mut runtime);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    let leaves = ["ts$fn$choose", "ts$fn$echo", "ts$fn$three"];
    for name in leaves {
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
        let wait = |runtime: &TieredRuntime, count| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            while runtime.statistics().optimized_functions < count {
                assert!(
                    std::time::Instant::now() < deadline,
                    "union promotion did not install"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        for name in leaves {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, leaves.len());
        check(&mut runtime);
        runtime
            .optimize_function("ts$fn$score", OptimizationTier::Optimized)
            .unwrap();
        wait(&runtime, leaves.len() + 1);
        check(&mut runtime);
        for name in names
            .iter()
            .filter(|n| !leaves.contains(&n.as_str()) && n.as_str() != "ts$fn$score")
        {
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
fn primitive_unions_do_not_allocate_or_box_payloads() {
    use std::sync::{Arc, Mutex};
    use zyntax_compiler::{
        hir::*,
        lowering::{AstLowering, LoweringConfig, LoweringContext},
    };
    use zyntax_typed_ast::{InternedString, arena::AstArena};
    let mut program = zyntax_ts::parse_program(SOURCE, "union.ts", Options::default()).unwrap();
    let mut lowering = LoweringContext::new(
        InternedString::new_global("unions"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let mut module = lowering.lower_program(&mut program).unwrap();
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    module.automatic_release = true;
    zyntax_compiler::drop_insert::run_module(&mut module);
    let mut creates = 0;
    let mut extracts = 0;
    for f in module.functions.values() {
        for i in f.blocks.values().flat_map(|b| &b.instructions) {
            match i {
                HirInstruction::CreateUnion { .. } => creates += 1,
                HirInstruction::ExtractUnionValue { .. } => extracts += 1,
                HirInstruction::Call {
                    callee: HirCallable::Intrinsic(Intrinsic::Malloc | Intrinsic::Free),
                    ..
                } => panic!("union heap allocation"),
                HirInstruction::Call {
                    callee: HirCallable::Symbol(s),
                    ..
                } => assert!(!s.contains("box_") && !s.contains("Dynamic"), "{s}"),
                _ => {}
            }
        }
    }
    assert!(creates > 0 && extracts > 0);
    let before = format!("{module:?}");
    zyntax_compiler::drop_insert::run_module(&mut module);
    assert_eq!(before, format!("{module:?}"));
}

#[test]
fn unsupported_reference_ownership_and_stale_narrowing_remain_explicit() {
    for source in [
        "function f(n:number){if(n>0)return n;return \"a\"+\"b\";}",
        "const s=\"a\"+\"b\";const x:number|string=s;",
        "function f(s:string):number|string{return s;}",
        "const s=\"a\"+\"b\";const x=true?1:s;",
    ] {
        let error =
            zyntax_ts::parse_program(source, "ownership.ts", Options::default()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Unsupported, "{source}: {error}");
    }
    for source in [
        "function f(x:number|string){if(typeof x===\"number\"){x=\"s\";return x*2;}return 0;}",
        "function f(x:number|string,b:boolean){if(typeof x===\"number\"){if(b)x=\"s\";return x*2;}return 0;}",
        "function f(x:number|string){if(typeof x===\"number\"&&(x=\"s\")==='s')return x*2;return 0;}",
    ] {
        let error = zyntax_ts::parse_program(source, "flow.ts", Options::default()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Type, "{source}: {error}");
    }
}
