//! End-to-end smoke test for the `zyntax_wasm` shim.
//!
//! Runs natively (the wasm-bindgen exports compile down to plain
//! Rust functions when the target isn't `wasm32`), so `cargo test`
//! exercises the same parse → lower → interpret pipeline that the
//! browser will call into. Same shape as `wren_lift/wasm`'s native
//! smoke coverage.
//!
//! When `wasm-pack test --headless --chrome` lands, this same body
//! reruns under a real browser to verify wasm-bindgen's binding
//! generation hasn't drifted from the Rust signatures.

use zyntax_wasm::{_zyntax_run_async, run, version, ErrorKind};

#[test]
fn version_is_self_describing() {
    let v = version();
    assert!(v.contains("zyntax_wasm"), "version string: {}", v);
}

#[test]
fn run_trivial_main_returns_value() {
    let result = run("def main(): i64 { return 42 }");
    assert!(
        result.ok(),
        "trivial main() should run cleanly. output={}",
        result.output()
    );
    assert_eq!(result.output(), "42");
    assert!(matches!(result.error_kind(), ErrorKind::None));
}

#[test]
fn run_arithmetic_main() {
    let result = run("def main(): i64 { return 6 * 7 }");
    assert!(result.ok(), "output={}", result.output());
    assert_eq!(result.output(), "42");
}

#[test]
fn parse_error_classifies_as_compile_error() {
    // Missing closing brace.
    let result = run("def main(): i64 { return 7");
    assert!(!result.ok());
    assert!(matches!(result.error_kind(), ErrorKind::CompileError));
}

// ----- Phase H: cooperative-async entry --------------------------

#[test]
fn run_async_sync_program_returns_same_runresult_as_run() {
    // For sync programs `_zyntax_run_async` is exactly `run` plus
    // the `js_complete_task` callback (no-op on native). The
    // returned RunResult shape stays identical so the JS Promise
    // wrapper has something to inspect even before the callback
    // fires.
    let r = _zyntax_run_async("def main(): i64 { return 42 }", 1);
    assert!(r.ok(), "output={}", r.output());
    assert_eq!(r.output(), "42");
    assert!(matches!(r.error_kind(), ErrorKind::None));
}

#[test]
fn run_async_compile_error_bubbles_through_runresult() {
    // Failure path: no callback firing; the Promise wrapper sees
    // `ok=false` on the returned RunResult and resolves with that.
    let r = _zyntax_run_async("def main(): i64 { return 7", 2);
    assert!(!r.ok());
    assert!(matches!(r.error_kind(), ErrorKind::CompileError));
}
