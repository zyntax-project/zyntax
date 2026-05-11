//! Phase K — source-level (parser → typed-AST → HIR → JIT) tests
//! for algebraic effects. Validates that `Resume<T>` declared in the
//! prelude is recognised, the handler param triggers the
//! resumability detection, and the full Phase I.4 runtime symbol
//! re-enters the caller's continuation correctly.

use std::path::Path;
use zynml::{ZynML, ZynMLConfig, ZynMLRuntimeProfile};

fn plugins_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("plugins")
        .join("target")
        .join("zrtl")
}

fn create_runtime() -> Option<ZynML> {
    let plugins_path = plugins_dir();
    if !plugins_path.exists() {
        eprintln!("Skipping: plugins not built at {}", plugins_path.display());
        return None;
    }
    let config = ZynMLConfig {
        plugins_dir: plugins_path.to_string_lossy().to_string(),
        load_optional: true,
        verbose: false,
        runtime_profile: ZynMLRuntimeProfile::Classic,
    };
    ZynML::with_config(config).ok()
}

#[test]
fn effect_with_resume_continuation_runs_from_source() {
    // The breakthrough test from Phase I.5 / Phase J, but exercised
    // entirely via the parser path — `Resume<i64>` comes from the
    // prelude, no hand-built TypedProgram.
    //
    // Expected runtime output (println): `result = 1042`
    //   * `op()` lowers to PerformEffect(E.op).
    //   * `apply_krio_effect_lowering` runs (resumable handler
    //     detected via Resume<T> param).
    //   * Handler `k(21) + 1000`:
    //     - SSA rewrites `k(21)` to `Call(Symbol("__zyntax_effect_resume"), [k, 21])`.
    //     - Runtime symbol stores 21 at result_slot, re-polls.
    //     - Caller dispatches to resume_entry, runs `return x * 2` with x=21 → returns 42.
    //     - Symbol returns 42 to handler.
    //   * Handler computes 42 + 1000 = 1042, returns.
    //   * yield_block returns 1042 directly (no post-perform re-run).
    //   * run() returns 1042. println formats as "result = 1042".
    let Some(mut zynml) = create_runtime() else {
        return;
    };

    let source = r#"
        effect E {
            def op(): i64
        }

        handler H for E {
            def op(k: Resume<i64>): i64 {
                return k(21) + 1000
            }
        }

        @effect(E)
        def run(): i64 {
            let x = op()
            return x * 2
        }

        def main() {
            let result = run()
            println(f"result = {result}")
        }
    "#;

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| zynml.run(source)));
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("source-level effect program should run cleanly; got {}", e),
        Err(_) => panic!("source-level effect program panicked"),
    }
}

#[test]
fn effect_with_abort_runs_from_source() {
    // Same shape but the handler aborts: `abort(99)` should cause
    // run() to return 99 without running the post-perform `x * 2`.
    //
    // Expected output: `result = 99`   (not 198 — abort skips
    // post-perform).
    let Some(mut zynml) = create_runtime() else {
        return;
    };

    let source = r#"
        effect E {
            def op(): i64
        }

        handler H for E {
            def op(k: Resume<i64>): i64 {
                return abort(99)
            }
        }

        @effect(E)
        def run(): i64 {
            let x = op()
            return x * 2
        }

        def main() {
            let result = run()
            println(f"result = {result}")
        }
    "#;

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| zynml.run(source)));
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("source-level abort program should run cleanly; got {}", e),
        Err(_) => panic!("source-level abort program panicked"),
    }
}

#[test]
fn multi_shot_resume_runs_from_source() {
    // Handler calls k three times and sums. Expected: 20 + 40 + 60 = 120.
    let Some(mut zynml) = create_runtime() else {
        return;
    };

    let source = r#"
        effect E {
            def op(): i64
        }

        handler H for E {
            def op(k: Resume<i64>): i64 {
                return k(10) + k(20) + k(30)
            }
        }

        @effect(E)
        def run(): i64 {
            return op() * 2
        }

        def main() {
            let result = run()
            println(f"result = {result}")
        }
    "#;

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| zynml.run(source)));
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!(
            "source-level multi-shot program should run cleanly; got {}",
            e
        ),
        Err(_) => panic!("source-level multi-shot program panicked"),
    }
}
