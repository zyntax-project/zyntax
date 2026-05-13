//! Tier-up integration test through `ZyntaxRuntime`: BC interpreter →
//! Cranelift JIT, beadie-orchestrated.
//!
//! Verifies the ladder closes via the single runtime:
//!   1. Parse ZynML source → TypedAST → HIR via `ZyntaxRuntime`.
//!   2. Configure a low promotion threshold and install the tier
//!      ladder via `install_interp_jit_with`.
//!   3. Repeated calls drive beadie's broker; Cranelift compiles the
//!      function; subsequent calls dispatch the JIT'd code.

use std::time::{Duration, Instant};

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::{ZyntaxRuntime, ZyntaxValue};

fn compile_with_jit(source: &str, threshold: u32) -> ZyntaxRuntime {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<test>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program).expect("compile");

    // Install the interp's beadie tier-up ladder with a custom warm
    // threshold so the test can control when promotion fires.
    let mut cfg = TieredConfig::default();
    cfg.profile_config.warm_threshold = threshold as u64;
    rt.install_interp_jit_with(cfg).expect("install interp JIT");
    rt
}

fn poll_until(deadline: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cond()
}

fn run_with_large_stack(test_body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("interp-jit-tier-up-large-stack".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(test_body)
        .expect("spawn large-stack test thread")
        .join()
        .expect("large-stack test thread panicked");
}

#[test]
fn tier_up_to_cranelift_preserves_correctness() {
    let rt = compile_with_jit(
        r#"
        def answer(): i64 {
            return 42
        }
        "#,
        2,
    );

    for _ in 0..8 {
        let result = rt.call_function_raw("answer", vec![]).unwrap();
        assert_eq!(
            result.as_i64(),
            Some(42),
            "tier-up broke correctness: got {:?}",
            result
        );
    }
}

#[test]
fn tier_up_to_cranelift_runs_arithmetic() {
    let rt = compile_with_jit(
        r#"
        def add(a: i64, b: i64): i64 {
            return a + b
        }
        "#,
        2,
    );

    let cases = [
        (10i64, 32i64, 42i64),
        (100, 200, 300),
        (-1, 1, 0),
        (0, 0, 0),
        (7, 35, 42),
    ];
    for (a, b, expected) in cases.iter().cycle().take(15) {
        let result = rt
            .call_function_raw("add", vec![ZyntaxValue::Int(*a), ZyntaxValue::Int(*b)])
            .unwrap();
        assert_eq!(
            result.as_i64(),
            Some(*expected),
            "tier-up broke arithmetic: add({}, {})",
            a,
            b
        );
    }
}

#[test]
fn tier_up_actually_fires_with_low_threshold() {
    // threshold=1: the FIRST call crosses promotion; the broker
    // compiles; we poll for `compiled()` becoming Some.
    let rt = compile_with_jit(
        r#"
        def trivial(): i64 { return 7 }
        "#,
        1,
    );

    for _ in 0..3 {
        let r = rt.call_function_raw("trivial", vec![]).unwrap();
        assert_eq!(r.as_i64(), Some(7));
    }

    let func_ids = rt.interp_registered_function_ids();
    assert!(!func_ids.is_empty(), "no functions registered");

    let any_compiled = poll_until(Duration::from_millis(1000), || {
        func_ids.iter().any(|fid| rt.interp_function_compiled(*fid))
    });
    assert!(
        any_compiled,
        "no function tiered up to Cranelift within 1s — broker didn't fire"
    );
}

#[test]
fn tier_up_recursive_fibonacci_crosses_threshold() {
    run_with_large_stack(|| {
        // Recursive fib explodes the call tree, easily crossing the
        // threshold mid-recursion. Beadie's broker fires; subsequent
        // recursive calls dispatch the JIT'd code.
        let rt = compile_with_jit(
            r#"
            def fib(n: i64): i64 {
                if n < 2 {
                    return n
                }
                return fib(n - 1) + fib(n - 2)
            }
            "#,
            100,
        );

        let r = rt
            .call_function_raw("fib", vec![ZyntaxValue::Int(20)])
            .unwrap();
        assert_eq!(
            r.as_i64(),
            Some(6765),
            "fib(20) tier-up broke value: {:?}",
            r
        );

        // Second call: most dispatches go through the JIT now.
        let r = rt
            .call_function_raw("fib", vec![ZyntaxValue::Int(20)])
            .unwrap();
        assert_eq!(
            r.as_i64(),
            Some(6765),
            "fib(20) (warm) broke value: {:?}",
            r
        );

        // Verify the bead reports compiled.
        let func_ids = rt.interp_registered_function_ids();
        assert!(!func_ids.is_empty());
        let any_compiled = poll_until(Duration::from_millis(1000), || {
            func_ids.iter().any(|fid| rt.interp_function_compiled(*fid))
        });
        assert!(
            any_compiled,
            "fib never tiered up to Cranelift — broker didn't fire"
        );

        // Heavier load: confirm correctness under recursive JIT dispatch.
        let r = rt
            .call_function_raw("fib", vec![ZyntaxValue::Int(25)])
            .unwrap();
        assert_eq!(r.as_i64(), Some(75025), "fib(25) broke value: {:?}", r);
    });
}

#[test]
fn tier_up_branching_preserves_correctness() {
    let rt = compile_with_jit(
        r#"
        def max(a: i64, b: i64): i64 {
            if a > b {
                return a
            } else {
                return b
            }
        }
        "#,
        2,
    );

    for _ in 0..10 {
        let r = rt
            .call_function_raw("max", vec![ZyntaxValue::Int(10), ZyntaxValue::Int(20)])
            .unwrap();
        assert_eq!(r.as_i64(), Some(20), "got {:?}", r);
    }
    for _ in 0..10 {
        let r = rt
            .call_function_raw("max", vec![ZyntaxValue::Int(99), ZyntaxValue::Int(7)])
            .unwrap();
        assert_eq!(r.as_i64(), Some(99), "got {:?}", r);
    }
}
