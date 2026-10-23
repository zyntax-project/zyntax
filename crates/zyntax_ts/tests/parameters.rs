//! Optional, default and rest parameters agree across tiers; unmodelled forms are gaps.

use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

#[test]
fn omitted_and_rest_arguments_agree_across_tiers() {
    let source = r#"
        function scale(x: number, by: number = 2, offset?: number): number {
            return x * by + (offset === undefined ? 0 : offset);
        }
        function sum(first: number, ...rest: number[]): number {
            let total = first;
            for (const value of rest) total += value;
            return total;
        }
        function run(n: number): number {
            return scale(n) + scale(n, 3) + scale(n, undefined, 1) + sum(n) + sum(n, 1, 2);
        }
    "#;
    let program = zyntax_ts::parse_program(source, "params.ts", Default::default()).unwrap();
    let config = TieredConfig {
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.compile_typed_program(program).unwrap();
    let functions = [
        "ts$m$params_2ets$fn$run",
        "ts$m$params_2ets$fn$scale",
        "ts$m$params_2ets$fn$sum",
    ];
    // 2n + 3n + (2n + 1) + n + (n + 3)
    let expected = |n: f64| ZyntaxValue::Float(9.0 * n + 4.0);
    let check = |runtime: &mut TieredRuntime, n: f64| {
        assert_eq!(
            runtime
                .call_raw(functions[0], &[ZyntaxValue::Float(n)])
                .unwrap(),
            expected(n)
        );
    };
    check(&mut runtime, 1.0);
    for function in functions {
        runtime
            .optimize_function(function, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime, 2.0);
    #[cfg(feature = "llvm-backend")]
    {
        for function in functions {
            runtime
                .optimize_function(function, OptimizationTier::Optimized)
                .unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while runtime.statistics().optimized_functions < 3 {
            assert!(std::time::Instant::now() < deadline, "LLVM did not install");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check(&mut runtime, 3.0);
    }
    runtime.stop();
}

#[test]
fn unmodelled_parameter_forms_are_explicit_gaps() {
    for (source, message) in [
        (
            "function f(s: string = 'x'): string { return s; }",
            "of this type",
        ),
        (
            "function f(s = 'x'): string { return s; }",
            "default parameter's type",
        ),
        (
            "function f(a?: number): number { return 1; }\nconst g = f;",
            "function values",
        ),
        (
            "function f(...xs: number[]): number { return 1; }\nconst g = f;",
            "function values",
        ),
        (
            "export function f(a: number = 1): number { return a; }",
            "exported functions",
        ),
        (
            "class C { constructor(public x?: number) {} }",
            "optional parameter properties",
        ),
        ("function f<T>(x?: T): number { return 1; }", "of this type"),
        (
            "function f(a?: boolean): void { a ??= true; }",
            "logical assignment",
        ),
    ] {
        let error = zyntax_ts::parse_program(source, "gaps.ts", Default::default()).unwrap_err();
        assert_eq!(
            error.kind,
            zyntax_ts::ErrorKind::Unsupported,
            "{source}: {error}"
        );
        assert!(
            error.message.contains(message),
            "{source}: {}",
            error.message
        );
    }
}
