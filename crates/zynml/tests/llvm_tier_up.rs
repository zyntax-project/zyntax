//! LLVM-tier-up test — only runs when built with `--features llvm-backend`.
//!
//! Drives a function past tier 1 (Cranelift) into tier 2 (LLVM) and
//! calls it AFTER the LLVM swap. If the LLVM-emitted fn ptr is
//! dispatched through `call_extern_symbol`'s i64-funneled transmute
//! correctly, the call returns the right value. Otherwise we see
//! SIGSEGV / wrong-value, surfacing the ABI mismatch.

#![cfg(feature = "llvm-backend")]

use std::time::Duration;

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::tiered_backend::TieredConfig;
use zyntax_embed::ZyntaxRuntime;

fn compile_with_jit(source: &str, warm: u32, hot: u32) -> ZyntaxRuntime {
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("ZynML grammar should compile");
    let program = grammar
        .parse_with_filename(source, "<test>")
        .expect("source should parse");
    let mut rt = ZyntaxRuntime::new().expect("runtime");
    rt.compile_typed_program(program).expect("compile");

    let mut cfg = TieredConfig::default();
    cfg.profile_config.warm_threshold = warm as u64;
    cfg.profile_config.hot_threshold = hot as u64;
    rt.install_interp_jit_with(cfg).expect("install JIT");
    rt
}

fn poll_until(deadline: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cond()
}

#[test]
fn tier_up_to_llvm_dispatches_correctly() {
    // Simplest function: no args, primitive return. Drive to LLVM
    // tier and call afterwards.
    let rt = compile_with_jit(
        r#"
        def trivial(): i64 { return 7 }
        "#,
        1, // warm
        1, // hot — fires immediately after gen=1
    );

    // Call repeatedly to drive both tier promotions. The tick callback
    // re-routes to the freshest compiled pointer each invocation; once
    // beadie's PromotionBroker calls `swap_compiled` with the LLVM
    // pointer the bead's generation bumps to 1 and the next call
    // dispatches into LLVM-emitted code.
    //
    // Beadie indexing reminder: `install_compiled` (Tier 0) leaves
    // generation at 0, so gen=0 means "running Cranelift", gen=1 means
    // "running LLVM" (after the PromotionBroker swap).
    for _ in 0..30 {
        let r = rt.call_function_raw("trivial", vec![]).unwrap();
        assert_eq!(r.as_i64(), Some(7), "value broken: {:?}", r);
        std::thread::sleep(Duration::from_millis(20));
    }

    let func_ids = rt.interp_registered_function_ids();
    assert!(!func_ids.is_empty());

    let reached_llvm = poll_until(Duration::from_millis(3000), || {
        func_ids
            .iter()
            .any(|fid| rt.interp_function_generation(*fid) >= 1)
    });
    let max_gen: u64 = func_ids
        .iter()
        .map(|fid| rt.interp_function_generation(*fid))
        .max()
        .unwrap_or(0);
    assert!(
        reached_llvm,
        "LLVM tier never fired (max gen = {})",
        max_gen
    );

    // Call AFTER LLVM tier-up. This is the dispatch that previously
    // SIGSEGV'd on Apple Silicon — broker-thread MCJIT pointers were
    // not callable from the main thread. Now LLVM is compiled
    // synchronously on the dispatching thread.
    let r = rt.call_function_raw("trivial", vec![]).unwrap();
    assert_eq!(
        r.as_i64(),
        Some(7),
        "LLVM-tier dispatch broke value: {:?}",
        r
    );
}

/// A variable assigned a wider literal than it was declared still
/// reaches the LLVM tier.
///
/// `t` is declared `f32` and the clamp assigns `-30.0`, an `f64`
/// literal. An annotated binding narrows its initializer, but an
/// assignment is the same declared boundary and did not, so the join
/// after the `if` carried one incoming of each width. LLVM rejects a
/// phi whose incomings disagree with its result type, and the whole
/// module failed verification: not a wrong answer, a tier that stopped
/// installing and silently left every function on Cranelift.
///
/// Cranelift coerces branch arguments on the way into a phi and is
/// unaffected, which is why this only ever showed on the LLVM lane.
#[test]
fn a_clamp_assigning_a_wide_literal_reaches_llvm() {
    let rt = compile_with_jit(
        r#"
        def clamped(x: f32): i64 {
            let mut t: f32 = x
            if t < -30.0 { t = -30.0 }
            return (t * 100.0) as i64
        }
        def main(): i64 { return clamped(-40.0) }
        "#,
        1,
        1,
    );

    for _ in 0..30 {
        let r = rt.call_function_raw("main", vec![]).unwrap();
        assert_eq!(r.as_i64(), Some(-3000), "value broken: {:?}", r);
        std::thread::sleep(Duration::from_millis(20));
    }

    let func_ids = rt.interp_registered_function_ids();
    assert!(!func_ids.is_empty());
    let reached_llvm = poll_until(Duration::from_millis(3000), || {
        func_ids
            .iter()
            .any(|fid| rt.interp_function_generation(*fid) >= 1)
    });
    assert!(
        reached_llvm,
        "LLVM tier never fired: the module failed verification"
    );

    let r = rt.call_function_raw("main", vec![]).unwrap();
    assert_eq!(r.as_i64(), Some(-3000), "LLVM-tier dispatch broke value");
}
