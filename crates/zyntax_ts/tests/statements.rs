//! `switch`, `do`/`while` and labeled jumps agree across tiers.

use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

#[test]
fn jumps_agree_across_tiers() {
    let source = r#"
        enum Step { Skip, Add, Stop }
        function classify(i: number): Step {
            switch (i % 4) {
                case 0: return Step.Skip;
                case 3: return Step.Stop;
                default: return Step.Add;
            }
        }
        function run(limit: number): number {
            let total = 0;
            let i = 0;
            outer: do {
                i++;
                switch (classify(i)) {
                    case Step.Skip: continue outer;
                    case Step.Stop:
                        if (i > limit) break outer;
                        break;
                    case Step.Add:
                        for (let j = 0; j < 3; j++) {
                            if (j === 2) continue outer;
                            total += j + i;
                        }
                }
                total += 100;
            } while (i < 50);
            return total;
        }
    "#;
    let program = zyntax_ts::parse_program(source, "jumps.ts", Default::default()).unwrap();
    let config = TieredConfig {
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime.compile_typed_program(program).unwrap();
    let run = "ts$m$jumps_2ets$fn$run";
    let classify = "ts$m$jumps_2ets$fn$classify";
    let expected = {
        let (mut total, mut i) = (0.0, 0.0);
        'outer: loop {
            i += 1.0;
            match i % 4.0 {
                0.0 => {}
                3.0 => {
                    if i > 20.0 {
                        break 'outer;
                    }
                    total += 100.0;
                }
                _ => {
                    total += i + (i + 1.0);
                }
            }
            if i >= 50.0 {
                break;
            }
        }
        total
    };
    let check = |runtime: &mut TieredRuntime| {
        assert_eq!(
            runtime.call_raw(run, &[ZyntaxValue::Float(20.0)]).unwrap(),
            ZyntaxValue::Float(expected)
        );
    };
    check(&mut runtime);
    for function in [run, classify] {
        runtime
            .optimize_function(function, OptimizationTier::Baseline)
            .unwrap();
    }
    check(&mut runtime);
    #[cfg(feature = "llvm-backend")]
    {
        for function in [run, classify] {
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
fn exhaustive_enum_switch_traps_on_other_numbers() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("exhaustive.ts");
    std::fs::write(
        &file,
        "enum Op { Add, Sub }\n\
         function apply(op: Op, a: number, b: number): number {\n\
         switch (op) { case Op.Add: return a + b; case Op.Sub: return a - b; }\n\
         }\n\
         const n: number = 7;\n\
         const op: Op = n;\n\
         console.log(apply(Op.Sub, 5, 2));\n\
         console.log(apply(op, 1, 2));\n",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_zynode"))
        .arg("run")
        .arg(&file)
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&output.stdout), "3\n");
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("no case matches"));
}

#[test]
fn unmodelled_switches_are_explicit_gaps() {
    for (source, message) in [
        (
            "function f(n: number): number { switch (n) { case 0: let x = 1; case 1: return x; } return 0; }",
            "sharing block-scoped declarations",
        ),
        (
            "function f(n: number | undefined): void { switch (n) { case 1: break; } }",
            "switch over this type",
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
