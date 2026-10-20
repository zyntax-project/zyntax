//! Property views retain concrete storage and share native alias ownership.

use zyntax_ts::{ErrorKind, Options, Soundness};

const SOURCE: &str = r#"
    interface X { x: number }
    interface Z { z: number }
    interface BZ { b: boolean; z: number }
    function borrow(p: X): X { p.x += 2; return p; }
    function make(x: number): X { const p = { x: x, z: 4 }; return p; }
    function identity<T>(p: T): T { return p; }
    function inferred(x: number) { const p = { a: true, z: x }; const q: Z = p; return q; }
    function padding(p: BZ): BZ { p.z += 2; p.b = false; return p; }
    function paddingScore(n: number) {
        const p = { b: true, c: true, z: n }; const q = padding(p);
        return p.z + q.z + (p.b ? 100 : 0) + (p.c ? 10 : 0);
    }
    function score(n: number) {
        const p = { x: n, z: 4 };
        const q: X = p;
        const r = borrow(p); r.x += 1;
        const g = identity(q); g.x += 1;
        const m = make(1);
        const w = inferred(n); const z: Z = w; z.z += 1;
        return p.x * 100 + r.x * 10 + g.x + m.x + z.z;
    }
"#;

#[test]
fn logical_views_hide_fields_in_reads_writes_calls_and_inference() {
    let prefix =
        "interface X{x:number} interface XY{x:number;y:number} const p:XY={x:1,y:2}; const q:X=p;";
    for tail in [
        "console.log(q.y);",
        "q.y=9;",
        "const r:XY=q;",
        "function wide(p:XY){} wide(q);",
        "function wide(q:X):XY{return q;}",
        "function id<T>(p:T):T{return p;} const r=id(q); console.log(r.y);",
        "const r=q; console.log(r.y);",
        "const r=(q); console.log(r.y);",
        "let r:X=p; const s=(r=q); console.log(s.y);",
        "function f(){const p:XY={x:1,y:2};const q:X=p;return q;} const r=f(); console.log(r.y);",
    ] {
        let source = format!("{prefix}{tail}");
        let error = zyntax_ts::parse_program(&source, "hidden.ts", Options::default()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Type, "{tail}: {error}");
    }
}

#[test]
fn fresh_objects_keep_excess_property_checks() {
    for source in [
        "interface X{x:number} const p:X={x:1,y:2};",
        "interface X{x:number} const p:X=({x:1,y:2});",
        "interface X{x:number} const p:X=({x:1,y:2})!;",
        "interface X{x:number} function f(p:X){} f({x:1,y:2});",
        "interface X{x:number} function f():X{return {x:1,y:2};}",
        "interface X{x:number} let p:X={x:1}; p={x:1,y:2};",
        "interface X{x:number} function id<T>(p:T):T{return p;} id<X>({x:1,y:2});",
    ] {
        let error = zyntax_ts::parse_program(source, "fresh.ts", Options::default()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Type, "{source}: {error}");
        assert!(error.message.contains("excess property"), "{error}");
    }
}

#[test]
fn unproven_layout_changes_are_explicit_gaps() {
    for source in [
        "interface Z{z:number} function f(p:Z){return p.z;} const p={a:true,z:2}; f(p);",
        "interface Z{z:number} function f():Z{const p={a:true,z:2};return p;}",
        "interface X{x:number} let p:X={x:1,y:2} as X; p={x:3};",
        "interface XY{x:number;y:number} const p={x:1}; const q=p as XY;",
        "interface Z{z:number} function id<T>(p:T):T{return p;} const p={a:true,z:2};const q:Z=p;id(q);",
        "interface X{x:number} const a={x:1,y:2};const b={x:1};const c=true?a:b;",
        "interface X{x:number} function f(b:boolean){const p={x:1,y:2};const q:X=p;if(b)return q;return p;}",
    ] {
        let error = zyntax_ts::parse_program(source, "layout.ts", Options::default()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Unsupported, "{source}: {error}");
    }
}

#[test]
fn readonly_policy_survives_width_aliases_and_assertions() {
    let source = "interface R{readonly x:number} interface W{x:number} const p={x:1,y:2}; const q:R=p; const r:W=q; r.x=3;";
    zyntax_ts::parse_program(source, "readonly.ts", Options::default()).unwrap();
    let error = zyntax_ts::parse_program(
        source,
        "readonly.ts",
        Options {
            soundness: Soundness::Checked,
        },
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Soundness);
}

#[test]
fn casts_preserve_one_allocation_and_one_release() {
    use std::sync::{Arc, Mutex};
    use zyntax_compiler::hir::{HirCallable, HirInstruction, Intrinsic};
    use zyntax_compiler::lowering::{AstLowering, LoweringConfig, LoweringContext};
    use zyntax_typed_ast::{InternedString, arena::AstArena};
    let source = format!(
        "{SOURCE} const p={{x:1,z:2}}; const q:X=p; const r=borrow(p); const s=identity(q); s.x=8; console.log(p.x,r.x);"
    );
    let mut program = zyntax_ts::parse_program(&source, "drops.ts", Options::default()).unwrap();
    let mut context = LoweringContext::new(
        InternedString::new_global("views"),
        Arc::new(program.type_registry.clone()),
        Arc::new(Mutex::new(AstArena::new())),
        LoweringConfig::default(),
    );
    let mut module = context.lower_program(&mut program).unwrap();
    zyntax_compiler::monomorphize_module(&mut module).unwrap();
    module.automatic_release = true;
    zyntax_compiler::drop_insert::run_module(&mut module);
    for function in module.functions.values() {
        let name = function.name.resolve_global().unwrap();
        let count = |intrinsic| {
            function.blocks.values().flat_map(|b| &b.instructions).filter(|i| {
                matches!(i, HirInstruction::Call { callee:HirCallable::Intrinsic(i), .. } if *i==intrinsic)
            }).count()
        };
        if module.initializers.contains(&function.id) {
            assert_eq!(count(Intrinsic::Malloc), 1, "no view wrapper or copy");
            assert_eq!(count(Intrinsic::Free), 1, "one shared owner");
        } else if name == "ts$fn$borrow" || name.starts_with("ts$fn$identity__mono_") {
            assert_eq!(count(Intrinsic::Malloc), 0);
            assert_eq!(count(Intrinsic::Free), 0, "borrowed storage stays borrowed");
        } else if name == "ts$fn$make" || name == "ts$fn$inferred" {
            assert_eq!(count(Intrinsic::Malloc), 1);
            assert_eq!(
                count(Intrinsic::Free),
                0,
                "fresh storage transfers to caller"
            );
        }
    }
}

#[test]
fn width_views_agree_before_and_after_native_promotions() {
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
    let check = |runtime: &mut TieredRuntime, input, expected| {
        assert_eq!(
            runtime
                .call_raw("ts$fn$score", &[ZyntaxValue::Float(input)])
                .unwrap(),
            ZyntaxValue::Float(expected)
        );
        assert_eq!(
            runtime
                .call_raw("ts$fn$paddingScore", &[ZyntaxValue::Float(input)])
                .unwrap(),
            ZyntaxValue::Float(input * 2.0 + 14.0)
        );
    };
    check(&mut runtime, 3.0, 782.0);
    assert_eq!(runtime.statistics().baseline_functions, 0);
    for name in ["ts$fn$borrow", "ts$fn$make"] {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime, 4.0, 894.0);
    for name in &names {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime, 5.0, 1006.0);
    #[cfg(feature = "llvm-backend")]
    {
        let wait = |runtime: &TieredRuntime, installed| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
            while runtime.statistics().optimized_functions < installed {
                assert!(
                    std::time::Instant::now() < deadline,
                    "LLVM views did not install"
                );
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        for name in ["ts$fn$borrow", "ts$fn$make"] {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, 2);
        check(&mut runtime, 4.0, 894.0);
        runtime
            .optimize_function("ts$fn$score", OptimizationTier::Optimized)
            .unwrap();
        wait(&runtime, 3);
        check(&mut runtime, 5.0, 1006.0);
        for name in names
            .iter()
            .filter(|name| !["ts$fn$borrow", "ts$fn$make", "ts$fn$score"].contains(&name.as_str()))
        {
            runtime
                .optimize_function(name, OptimizationTier::Optimized)
                .unwrap();
        }
        wait(&runtime, names.len());
        check(&mut runtime, 3.0, 782.0);
    }
    assert!(!zyntax_compiler::collector::is_enabled());
    runtime.stop();
}
