//! Module bindings read and written by functions share one storage across tiers.

use zyntax_compiler::tiered_backend::{OptimizationTier, TieredConfig};
use zyntax_embed::{TieredRuntime, ZyntaxValue};

fn runtime() -> TieredRuntime {
    let config = TieredConfig {
        #[cfg(feature = "llvm-backend")]
        tier2_backend: zyntax_compiler::tiered_backend::Tier2Backend::LLVM,
        ..Default::default()
    };
    let mut runtime = TieredRuntime::new(config).unwrap();
    zyntax_ts::register_runtime(&mut runtime).unwrap();
    runtime
}

fn check(source: &str) -> zyntax_ts::Error {
    zyntax_ts::parse_program(source, "globals.ts", Default::default()).unwrap_err()
}

#[test]
fn module_bindings_persist_across_promotion() {
    let source = r#"
        class Tally { hits: number = 0; }
        let total = 0;
        let enabled = true;
        const tally = new Tally();
        function add(x: number): number {
            if (enabled) { total += x; }
            tally.hits++;
            return total * 100 + tally.hits;
        }
        function disable(): void { enabled = false; }
    "#;
    let program = zyntax_ts::parse_program(source, "globals.ts", Default::default()).unwrap();
    let mut runtime = runtime();
    runtime.compile_typed_program(program).unwrap();
    let add = "ts$m$globals_2ets$fn$add";
    let call = |runtime: &mut TieredRuntime, x: f64| {
        runtime.call_raw(add, &[ZyntaxValue::Float(x)]).unwrap()
    };
    assert_eq!(call(&mut runtime, 2.0), ZyntaxValue::Float(201.0));
    runtime
        .optimize_function(add, OptimizationTier::Baseline)
        .unwrap();
    assert_eq!(call(&mut runtime, 3.0), ZyntaxValue::Float(502.0));
    #[cfg(feature = "llvm-backend")]
    {
        runtime
            .optimize_function(add, OptimizationTier::Optimized)
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while runtime.statistics().optimized_functions < 1 {
            assert!(std::time::Instant::now() < deadline, "LLVM did not install");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(call(&mut runtime, 4.0), ZyntaxValue::Float(903.0));
    }
    runtime
        .call_raw("ts$m$globals_2ets$fn$disable", &[])
        .unwrap();
    let hits = if cfg!(feature = "llvm-backend") {
        4.0
    } else {
        3.0
    };
    let total = if cfg!(feature = "llvm-backend") {
        9.0
    } else {
        5.0
    };
    assert_eq!(
        call(&mut runtime, 10.0),
        ZyntaxValue::Float(total * 100.0 + hits)
    );
    runtime.stop();
}

#[test]
fn module_binding_errors_and_gaps() {
    for (source, kind, message) in [
        (
            "console.log(n);\nlet n = 1;\nfunction f(): number { return n; }",
            zyntax_ts::ErrorKind::Type,
            "used before initialization",
        ),
        (
            "const n = 1;\nfunction f(): void { n = 2; }",
            zyntax_ts::ErrorKind::Type,
            "cannot assign to const",
        ),
        (
            "declare var n: number;\nconsole.log(n);",
            zyntax_ts::ErrorKind::Unsupported,
            "ambient variables",
        ),
        (
            "declare let n: number;\nfunction f(): number { return n; }",
            zyntax_ts::ErrorKind::Unsupported,
            "ambient variables",
        ),
        (
            "let n;\nfunction f(): number { return n; }",
            zyntax_ts::ErrorKind::Unsupported,
            "without a type annotation",
        ),
        (
            "let n: number;\nfunction f(): number { return n; }\nconsole.log(n);",
            zyntax_ts::ErrorKind::Type,
            "used before being assigned",
        ),
        (
            "let s = 'a';\nfunction f(x: string): void { s = s + x; }",
            zyntax_ts::ErrorKind::Unsupported,
            "static lifetime",
        ),
        (
            "let n: number | undefined = 1;\nfunction f(): number | undefined { return n; }",
            zyntax_ts::ErrorKind::Unsupported,
            "this type",
        ),
    ] {
        let error = check(source);
        assert_eq!(error.kind, kind, "{source}");
        assert!(
            error.message.contains(message),
            "{source}: {}",
            error.message
        );
    }
}

fn run(source: &str) -> std::process::Output {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("globals.ts");
    std::fs::write(&file, source).unwrap();
    std::process::Command::new(env!("CARGO_BIN_EXE_zynode"))
        .arg("run")
        .arg(&file)
        .output()
        .unwrap()
}

#[test]
fn early_module_binding_access_traps() {
    for (source, printed, error) in [
        (
            "function f(): number { return late * 2; }\nconsole.log('start');\nconsole.log(f());\nlet late = 1;",
            "start\n",
            "ReferenceError: Cannot access 'late' before initialization",
        ),
        (
            "function f(): void { late = 2; }\nconsole.log('start');\nf();\nlet late = 1;",
            "start\n",
            "ReferenceError: Cannot access 'late' before initialization",
        ),
        (
            "function f(): number { return count + 1; }\nconsole.log('start');\nconsole.log(f());\nvar count = 4;",
            "start\n",
            "'count' is read before it is assigned",
        ),
        (
            "var level: number;\nfunction f(): number { return level * 2; }\nconsole.log('start');\nconsole.log(f());",
            "start\n",
            "'level' is read before it is assigned",
        ),
    ] {
        let output = run(source);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(error), "{source}: {stderr}");
        assert_eq!(String::from_utf8_lossy(&output.stdout), printed, "{source}");
        assert_eq!(output.status.code(), Some(1), "{source}");
    }
}
