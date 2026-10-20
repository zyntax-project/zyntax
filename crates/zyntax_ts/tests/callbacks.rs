//! Scalar callbacks retain their signature through references, calls and tier changes.

use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

const SOURCE: &str = r#"
    type Transform = (n: number) => number;
    function apply(f: Transform, x: number): number { return f(x); }
    function select(): Transform { return n => n + 0.5; }
    function boolApply(f: (b: boolean) => boolean, b: boolean): boolean { return f(b); }
    function mixedApply(f: (n: number, b: boolean) => number, n: number, b: boolean): number { return f(n, b); }
    function invoke(f: (n: number) => void, x: number): void { f(x); }
    function score(x: number): number { const f = select(); return apply(f, x); }
    function boolScore(b: boolean): boolean { const f = (b: boolean) => !b; return boolApply(f, b); }
    function mixedScore(x: number): number { return mixedApply((n, b) => b ? n + 0.25 : n - 1, x, true); }
    function voidScore(x: number): number { const f: (n: number) => void = n => n + 1; invoke(f, x); return x + 2; }
"#;

fn check(runtime: &mut TieredRuntime) {
    assert_eq!(
        runtime
            .call_raw("ts$m$callbacks_2ets$fn$score", &[ZyntaxValue::Float(1.5)])
            .unwrap(),
        ZyntaxValue::Float(2.0)
    );
    for (input, expected) in [(false, true), (true, false)] {
        assert_eq!(
            runtime
                .call_raw(
                    "ts$m$callbacks_2ets$fn$boolScore",
                    &[ZyntaxValue::Bool(input)]
                )
                .unwrap(),
            ZyntaxValue::Bool(expected)
        );
    }
    assert_eq!(
        runtime
            .call_raw(
                "ts$m$callbacks_2ets$fn$voidScore",
                &[ZyntaxValue::Float(4.0)]
            )
            .unwrap(),
        ZyntaxValue::Float(6.0)
    );
    assert_eq!(
        runtime
            .call_raw(
                "ts$m$callbacks_2ets$fn$mixedScore",
                &[ZyntaxValue::Float(2.0)]
            )
            .unwrap(),
        ZyntaxValue::Float(2.25)
    );
}

#[cfg(feature = "llvm-backend")]
fn wait_optimized(runtime: &TieredRuntime, count: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while runtime.statistics().optimized_functions < count {
        assert!(
            std::time::Instant::now() < deadline,
            "LLVM callbacks did not install"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}

#[test]
fn callbacks_agree_before_and_after_each_native_promotion() {
    let program = zyntax_ts::parse_program(SOURCE, "callbacks.ts", Default::default()).unwrap();
    let config = TieredConfig {
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    let names = runtime.compile_typed_program(program).unwrap();
    assert_eq!(runtime.statistics().baseline_functions, 0);
    check(&mut runtime);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    // Native callers can enter callback targets that have not been promoted.
    for name in [
        "ts$m$callbacks_2ets$fn$apply",
        "ts$m$callbacks_2ets$fn$boolApply",
        "ts$m$callbacks_2ets$fn$mixedApply",
        "ts$m$callbacks_2ets$fn$invoke",
    ] {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    let names: Vec<_> = names
        .into_iter()
        .filter(|name| {
            name.starts_with("ts$m$callbacks_2ets$fn$")
                || name.starts_with("ts$m$callbacks_2ets$arrow$")
        })
        .collect();
    for name in &names {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        runtime
            .optimize_function("ts$m$callbacks_2ets$fn$apply", OptimizationTier::Optimized)
            .unwrap();
        wait_optimized(&runtime, 1);
        check(&mut runtime);
        for name in names
            .iter()
            .filter(|name| *name != "ts$m$callbacks_2ets$fn$apply")
        {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait_optimized(&runtime, names.len());
        check(&mut runtime);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}

#[test]
fn lifted_functions_use_typed_code_references_without_heap_environments() {
    use std::sync::{Arc, Mutex};
    use zyntax_compiler::hir::{HirCallable, HirInstruction, HirType, Intrinsic};
    use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
    use zyntax_typed_ast::{InternedString, arena::AstArena};

    let mut program = zyntax_ts::parse_program(SOURCE, "callbacks.ts", Default::default()).unwrap();
    let mut context = LoweringContext::new(
        InternedString::new_global("callbacks"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let mut module = context.lower_program(&mut program).unwrap();
    module.automatic_release = true;
    zyntax_compiler::drop_insert::run_module(&mut module);
    let mut references = 0;
    for function in module.functions.values() {
        let name = function.name.resolve_global().unwrap();
        if name.starts_with("ts$m$callbacks_2ets$fn$")
            || name.starts_with("ts$m$callbacks_2ets$arrow$")
        {
            assert!(function.attributes.strict_fp, "{name}");
        }
        for instruction in function.blocks.values().flat_map(|b| &b.instructions) {
            assert!(!matches!(instruction, HirInstruction::CreateClosure { .. }));
            if let HirInstruction::Call { result, callee, .. } = instruction {
                assert!(!matches!(
                    callee,
                    HirCallable::Intrinsic(Intrinsic::Malloc | Intrinsic::Free)
                ));
                if matches!(callee, HirCallable::FuncRef(_)) {
                    references += 1;
                    assert!(matches!(
                        function.values[&result.unwrap()].ty,
                        HirType::Function(_)
                    ));
                }
            }
        }
    }
    assert!(references >= 4);
}

#[test]
fn captures_and_function_object_semantics_require_explicit_implementation() {
    for source in [
        "let x=1; const f=()=>x; x=2; console.log(f());",
        "function make(x:number):()=>number{return ()=>x;}",
        "const f=()=>x; const x=1; console.log(f());",
        "const f=(n:number):number=>n>0?f(n-1):0;",
        "const f=()=>1; console.log(f===f);",
        "const f=()=>1; console.log(f.length);",
        "const f=(p:{x:number})=>p.x;",
        "const f=(s:string)=>s;",
        "const f=()=>({x:1});",
        "const f=()=>\"owned\";",
        "const f:(n:number)=>number=()=>1;",
    ] {
        let error = zyntax_ts::parse_program(source, "pending.ts", Default::default()).unwrap_err();
        assert_eq!(
            error.kind,
            zyntax_ts::ErrorKind::Unsupported,
            "{source}: {error}"
        );
    }
}
