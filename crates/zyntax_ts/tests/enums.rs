//! Enum members are native constants on every tier; unmodelled enum uses stay explicit gaps.

use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

#[test]
fn enum_values_agree_across_tiers() {
    let source = r#"
        enum Shape { Circle = 1, Square = 2, Triangle = 3 }
        const enum Scale { Unit = 1, Double = Unit * 2 }
        enum Name { Circle = "circle", Square = "square" }
        function area(kind: Shape, size: number): number {
            if (kind === Shape.Circle) { return 3 * size * size; }
            if (kind === Shape.Square) { return size * size * Scale.Double; }
            return size;
        }
        function named(kind: Shape): boolean {
            const label: Name = kind === Shape.Circle ? Name.Circle : Name.Square;
            return label === Name.Square;
        }
    "#;
    let program = zyntax_ts::parse_program(source, "enums.ts", Default::default()).unwrap();
    let config = TieredConfig {
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.compile_typed_program(program).unwrap();
    let area = "ts$m$enums_2ets$fn$area";
    let named = "ts$m$enums_2ets$fn$named";
    let check = |runtime: &mut TieredRuntime| {
        assert_eq!(
            runtime
                .call_raw(area, &[ZyntaxValue::Float(2.0), ZyntaxValue::Float(3.0)])
                .unwrap(),
            ZyntaxValue::Float(18.0)
        );
        assert_eq!(
            runtime
                .call_raw(area, &[ZyntaxValue::Float(1.0), ZyntaxValue::Float(2.0)])
                .unwrap(),
            ZyntaxValue::Float(12.0)
        );
        assert_eq!(
            runtime.call_raw(named, &[ZyntaxValue::Float(2.0)]).unwrap(),
            ZyntaxValue::Bool(true)
        );
    };
    check(&mut runtime);
    for function in [area, named] {
        runtime
            .optimize_function(function, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        for function in [area, named] {
            runtime
                .optimize_function(function, OptimizationTier::Optimized)
                .unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while runtime.statistics().optimized_functions < 2 {
            assert!(std::time::Instant::now() < deadline, "LLVM did not install");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        check(&mut runtime);
    }
    runtime.stop();
}

#[test]
fn unmodelled_enum_uses_are_explicit_gaps() {
    for (source, message) in [
        ("enum E { A }\nconst xs: E[] = [];", "arrays of enum values"),
        ("enum E { A }\nconst xs = [E.A];", "arrays of enum values"),
        ("enum E { A }\ntype P = { e: E };", "enum-typed fields"),
        ("enum E { A }\nclass C { e: E = E.A; }", "enum-typed fields"),
        (
            "enum E { A }\nconst f = (e: E): number => e;",
            "function values",
        ),
        (
            "enum E { A }\nconst p = { e: E.A };",
            "enum-valued object fields",
        ),
        ("enum E { A }\nconsole.log(E[0]);", "reverse mapping"),
        ("enum E { A }\nconst e = E;", "enum objects as values"),
        (
            "enum E { A = 'a'.length }\nconsole.log(E.A);",
            "computed enum members",
        ),
        (
            "enum E { A = 1, B = 'b' }\nconsole.log(E.A);",
            "mixed numeric and string",
        ),
        (
            "declare enum E { A = 1 }\nconsole.log(E.A);",
            "ambient enums",
        ),
        ("enum E { A }\nenum E { B = 1 }", "declaration merging"),
        (
            "enum E { A }\nexport function f(e: E): E { return e; }",
            "exported declarations",
        ),
        (
            "enum E { A }\nlet e: E | undefined;",
            "reference-bearing unions",
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
