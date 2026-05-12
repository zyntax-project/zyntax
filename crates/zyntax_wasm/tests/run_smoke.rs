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

use zyntax_wasm::{run, version, ErrorKind};

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
