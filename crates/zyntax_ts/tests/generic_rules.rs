use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

#[test]
fn constrained_specializations_and_defaults_agree_across_actual_tiers() {
    let program = zyntax_ts::parse_program(
        r#"
        function add<T extends number = number>(x:T,y:T):number{return x+y;}
        function square<T extends number>(x:T):number{return x*x;}
        function forward<U extends number>(x:U):number{return add<U>(x,x);}
        function pick<T,U extends T=T>(x:T,y:U):U{return y;}
        function keep<T extends number>(xs:T[]):T[]{return xs;}
        function size<T=number>():number{return 0;}
        function score(n:number):number {
            const xs=[n,n+1];const ys=keep<number>(xs);ys[0]+=1;
            return add(n,2)+square(n)+forward(n)+pick<number>(n,3)+xs[0]+size();
        }
        "#,
        "rules.ts",
        Default::default(),
    )
    .unwrap();
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
        .filter(|n| n.starts_with("ts$m$rules_2ets$fn$"))
        .collect();
    let check = |runtime: &mut TieredRuntime| {
        for n in [0.0, 1.0, 3.5, -2.0] {
            assert_eq!(
                runtime
                    .call_raw("ts$m$rules_2ets$fn$score", &[ZyntaxValue::Float(n)])
                    .unwrap(),
                ZyntaxValue::Float(n * n + 4.0 * n + 6.0)
            );
        }
    };
    assert_eq!(runtime.statistics().baseline_functions, 0);
    check(&mut runtime);
    for name in &names {
        runtime
            .optimize_function(name, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        let add = names.iter().find(|n| n.contains("add__mono_")).unwrap();
        runtime
            .optimize_function(add, OptimizationTier::Optimized)
            .unwrap();
        let wait = |runtime: &TieredRuntime, count| {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while runtime.statistics().optimized_functions < count {
                assert!(std::time::Instant::now() < deadline, "LLVM did not install");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        };
        wait(&runtime, 1);
        check(&mut runtime);
        for name in &names {
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
fn unavailable_library_constraints_remain_gaps_and_local_types_shadow_them() {
    for name in ["Date", "Number"] {
        let source = format!("function keep<T extends {name}>(x:T):T{{return x;}}");
        assert_eq!(
            zyntax_ts::parse_program(&source, "rules.ts", Default::default())
                .unwrap_err()
                .kind,
            zyntax_ts::ErrorKind::Unsupported
        );
    }
    assert_eq!(
        zyntax_ts::parse_program(
            "function keep<T extends Missing>(x:T):T{return x;}",
            "rules.ts",
            Default::default()
        )
        .unwrap_err()
        .kind,
        zyntax_ts::ErrorKind::Type
    );
    zyntax_ts::parse_program(
        "type Date={x:number};function keep<T extends Date>(x:T):T{return x;}keep({x:1});",
        "rules.ts",
        Default::default(),
    )
    .unwrap();
}
