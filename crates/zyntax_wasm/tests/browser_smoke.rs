//! Browser smoke tests for the Zyntax/ZynML wasm runtime.
//!
//! Run via:
//!
//! ```bash
//! wasm-pack test --headless --chrome crates/zyntax_wasm
//! ```
//!
//! These tests exercise the wasm-bindgen export surface end-to-end
//! under a real Chrome engine — so wasm-bindgen marshalling, the
//! `RunResult` getters, the Phase G `_zyntax_resolve_future` /
//! `_zyntax_reject_future` exports, and the Phase H Promise-routed
//! `_zyntax_run_async` are all validated in the same execution
//! environment that production hosts will run.
//!
//! The `.github/workflows/ci.yml` `wasm-headless-chrome` job runs
//! this whole file every PR.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;
use zyntax_wasm::{
    _zyntax_reject_future, _zyntax_resolve_future, _zyntax_run_async, run, version, ErrorKind,
    RunResult,
};

// JS-side host shims are normally installed by `web/zynml.mjs::
// installJitHost()` before the wasm module instantiates. The
// wasm-bindgen-test runner loads the wasm module directly without
// going through zynml.mjs, so the wasm code's `globalThis.
// _zyntax_jit_install` and friends are undefined and the first
// call into them throws.
//
// Install no-op stubs at the start of any test that exercises the
// interpreter (run / run_async). Tests that only touch the
// FutureTable exports don't need this.
fn install_host_stubs() {
    let global = js_sys::global();
    // _zyntax_jit_install: returns 0xFFFFFFFF to signal "JIT not
    // available, keep the function in BC." `_zyntax_jit_install`
    // is the only one called eagerly via the compile hook; the
    // others (_zyntax_jit_call_*, _zyntax_complete_task) are only
    // hit on tier-up which doesn't happen in these short tests.
    let install_fn = Closure::wrap(Box::new(|_bytes: js_sys::Uint8Array| -> u32 { u32::MAX })
        as Box<dyn Fn(js_sys::Uint8Array) -> u32>);
    let _ = js_sys::Reflect::set(
        &global,
        &JsValue::from_str("_zyntax_jit_install"),
        install_fn.as_ref().unchecked_ref(),
    );
    install_fn.forget();

    // _zyntax_complete_task stub — Phase H's wasm side calls this
    // when a sync task completes. We don't need to observe the
    // call here (the test reads the returned RunResult directly),
    // so a no-op is fine.
    let complete_fn = Closure::wrap(
        Box::new(|_task_id: i64, _value: i64, _ok: u32| {}) as Box<dyn Fn(i64, i64, u32)>
    );
    let _ = js_sys::Reflect::set(
        &global,
        &JsValue::from_str("_zyntax_complete_task"),
        complete_fn.as_ref().unchecked_ref(),
    );
    complete_fn.forget();
}

// No `wasm_bindgen_test_configure!` — default is "wasm context"
// (Node when run via `wasm-pack test --node`, browser when run
// via `wasm-pack test --headless --chrome`). Neither test body
// uses DOM APIs, so both targets work with the same code. CI's
// `wasm-headless-chrome` job runs the browser variant; Node mode
// is the local-dev shortcut for environments where Chrome's
// headless driver dies (macOS quarantine / codesigning issues).

// ----- Smoke: module loads + version is reported -----------------

#[wasm_bindgen_test]
fn version_string_includes_crate_name() {
    let v = version();
    assert!(v.contains("zyntax_wasm"), "version string: {}", v);
}

// ----- run() — synchronous compile + interpret -------------------

#[wasm_bindgen_test]
fn run_trivial_main_returns_value() {
    install_host_stubs();
    let r: RunResult = run("def main(): i64 { return 42 }");
    assert!(r.ok(), "output={}", r.output());
    assert_eq!(r.output(), "42");
    assert!(matches!(r.error_kind(), ErrorKind::None));
}

#[wasm_bindgen_test]
fn run_arithmetic_main() {
    install_host_stubs();
    let r = run("def main(): i64 { return 6 * 7 }");
    assert!(r.ok(), "output={}", r.output());
    assert_eq!(r.output(), "42");
}

#[wasm_bindgen_test]
fn parse_error_classifies_as_compile_error() {
    install_host_stubs();
    let r = run("def main(): i64 { return 7");
    assert!(!r.ok());
    assert!(matches!(r.error_kind(), ErrorKind::CompileError));
}

// ----- _zyntax_run_async — Phase H Promise-routed entry ----------

#[wasm_bindgen_test]
fn run_async_sync_program_returns_runresult() {
    install_host_stubs();
    // Sync programs: `_zyntax_run_async` runs `run_impl` inline,
    // fires `js_complete_task` (no-op JS shim in test context),
    // and returns the RunResult. Equivalent to `run` for these.
    let r = _zyntax_run_async("def main(): i64 { return 42 }", 1);
    assert!(r.ok(), "output={}", r.output());
    assert_eq!(r.output(), "42");
}

#[wasm_bindgen_test]
fn run_async_compile_error_returns_ok_false() {
    install_host_stubs();
    let r = _zyntax_run_async("def main(): i64 { return 7", 2);
    assert!(!r.ok());
    assert!(matches!(r.error_kind(), ErrorKind::CompileError));
}

// ----- _zyntax_resolve_future / _zyntax_reject_future -----------
//
// Phase G plumbing. With no parked futures, `_zyntax_resolve_future`
// returns the i32 outcome code for `UnknownHandle` (= 2). We don't
// exercise the full register → resolve → SM-advance round-trip here
// (that needs a krio-emitted SM, which is Phase I.1 work); these
// tests just prove the exports are reachable and return the right
// outcome code on unknown handles.

#[wasm_bindgen_test]
fn resolve_future_unknown_handle_returns_outcome_2() {
    // ResolveOutcome::UnknownHandle.as_i32() = 2 — see
    // `crates/zyntax_embed/src/host_futures.rs`.
    let rc = _zyntax_resolve_future(999_999_999, 0);
    assert_eq!(rc, 2);
}

#[wasm_bindgen_test]
fn reject_future_unknown_handle_returns_outcome_2() {
    let rc = _zyntax_reject_future(999_999_998, "test failure");
    assert_eq!(rc, 2);
}
