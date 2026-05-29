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

// ----- Parked-task path (Phase J) ----------------------------
//
// End-to-end browser test: an async ZynML program with `await
// sleep(100)` parks the SM, the JS-side setTimeout fires after
// ~100ms, `_zyntax_resolve_future` drives the SM to Ready, and
// `_zyntax_complete_task` resolves a JS Promise the test awaits.
//
// Verifies wall-clock ≥ 90ms — proves the parking actually
// happened, not that the SM synchronously collapsed (which would
// finish in ~0ms).

/// Install the cooperative-async JS host shims:
///   * `_zyntax_complete_task(task_id, value, ok)` — captures the
///     resolved value into a JS Map keyed by task_id; per-test
///     promise resolvers (registered before `_zyntax_run_async`)
///     read from there.
///   * `_zyntax_call_host_async_set_timeout(handle, ms)` — invokes
///     real `setTimeout` then calls `_zyntax_resolve_future(handle,
///     0)` so the parked SM advances and the Promise resolves.
fn install_async_host_shims() {
    let global = js_sys::global();

    // Per-task resolver registry. The test awaits a JS Promise
    // whose resolver is keyed by task_id; complete_task looks it up
    // and resolves with the value.
    let init_registry = js_sys::eval(
        r#"
        if (!globalThis.__zw_test_task_resolvers) {
            globalThis.__zw_test_task_resolvers = new Map();
        }
        "#,
    );
    assert!(init_registry.is_ok());

    // _zyntax_complete_task: look up the registered resolver and
    // resolve it with `value`. Tests that don't register a resolver
    // beforehand are not affected.
    let complete_fn = Closure::wrap(Box::new(|task_id: i64, value: i64, _ok: u32| {
        let _ = js_sys::eval(&format!(
            r#"
            (function() {{
                console.log("[test] _zyntax_complete_task called task_id={}, value={}");
                const r = globalThis.__zw_test_task_resolvers.get({});
                if (r) {{
                    globalThis.__zw_test_task_resolvers.delete({});
                    r({});
                }}
            }})()
            "#,
            task_id, value, task_id, task_id, value
        ));
    }) as Box<dyn Fn(i64, i64, u32)>);
    let _ = js_sys::Reflect::set(
        &global,
        &JsValue::from_str("_zyntax_complete_task"),
        complete_fn.as_ref().unchecked_ref(),
    );
    complete_fn.forget();

    // _zyntax_call_host_async_set_timeout: fire setTimeout, then
    // resolve the future on the wasm side. Wire via JS string so we
    // don't need a Rust-side closure that takes a wasm extern
    // reference.
    let install_timeout = js_sys::eval(
        r#"
        globalThis._zyntax_call_host_async_set_timeout = function(handle, ms) {
            console.log("[test] _zyntax_call_host_async_set_timeout handle=" + handle + ", ms=" + ms);
            const delay = typeof ms === 'bigint' ? Number(ms) : ms;
            setTimeout(function() {
                try {
                    const h = typeof handle === 'bigint' ? handle : BigInt(handle);
                    console.log("[test] setTimeout fired h=" + h);
                    const outcome = globalThis.__zw_test_wasm_resolve_future(h, 0n);
                    console.log("[test] resolve_future outcome=" + outcome);
                } catch (e) {
                    console.log("[test] setTimeout cb threw: " + e);
                }
            }, delay);
        };
        "#,
    );
    assert!(install_timeout.is_ok());
}

/// Bind a JS-side helper that calls the wasm `_zyntax_resolve_future`
/// export. The setTimeout shim above can't reference the wasm
/// binding directly because it runs through `eval`; this helper
/// closes over the actual binding via a wasm-bindgen Closure.
fn install_resolve_future_helper() {
    let global = js_sys::global();
    let helper = Closure::wrap(Box::new(|handle: i64, value: i64| -> i32 {
        _zyntax_resolve_future(handle, value)
    }) as Box<dyn Fn(i64, i64) -> i32>);
    let _ = js_sys::Reflect::set(
        &global,
        &JsValue::from_str("__zw_test_wasm_resolve_future"),
        helper.as_ref().unchecked_ref(),
    );
    helper.forget();
}

#[wasm_bindgen_test]
async fn run_async_arithmetic_then_await_returns_42() {
    install_host_stubs();
    install_async_host_shims();
    install_resolve_future_helper();

    // Reproducer for the `arithmetic-then-await` browser demo hanging.
    // Sync setup (3 lets), then a single `await sleep`, then return.
    // `c = a * b` must survive the park as a captures-lift saved slot.
    let source = "async def main(): i64 {\n    let a = 7\n    let b = 6\n    let c = a * b\n    await sleep(80)\n    return c\n}\n";
    let task_id: i64 = 142;

    let register_resolver = js_sys::eval(&format!(
        r#"
        (function() {{
            return new Promise(function(resolve) {{
                globalThis.__zw_test_task_resolvers.set({}, resolve);
            }});
        }})()
        "#,
        task_id
    ))
    .expect("eval should produce a Promise");
    let promise: js_sys::Promise = register_resolver.dyn_into().expect("Promise");

    let initial = _zyntax_run_async(source, task_id);
    assert!(initial.ok(), "compile/start: output={}", initial.output());

    let timeout_promise: js_sys::Promise = js_sys::eval(
        r#"
        new Promise(function(_, reject) {
            setTimeout(function() {
                reject(new Error('5s timeout — complete_task never fired'));
            }, 5000);
        })
        "#,
    )
    .expect("eval timeout")
    .dyn_into()
    .expect("Promise");
    let race = js_sys::Promise::race(&js_sys::Array::of2(&promise, &timeout_promise));
    let resolved = wasm_bindgen_futures::JsFuture::from(race)
        .await
        .expect("Promise should resolve cleanly");
    let value = resolved.as_f64().expect("resolver delivered a number");
    assert!((value - 42.0).abs() < 0.5, "expected 42, got {value}",);
}

#[wasm_bindgen_test]
async fn run_async_with_sleep_yields_and_returns_42() {
    install_host_stubs();
    install_async_host_shims();
    install_resolve_future_helper();

    let source = "async def main(): i64 {\n    await sleep(100)\n    return 42\n}\n";
    let task_id: i64 = 42;

    // Build a JS Promise whose resolver is keyed in
    // `__zw_test_task_resolvers` by this task_id.
    let register_resolver = js_sys::eval(&format!(
        r#"
        (function() {{
            return new Promise(function(resolve) {{
                globalThis.__zw_test_task_resolvers.set({}, resolve);
            }});
        }})()
        "#,
        task_id
    ))
    .expect("eval should produce a Promise");
    let promise: js_sys::Promise = register_resolver.dyn_into().expect("Promise");

    // Start the parked task. RunResult.ok should be true; for async
    // programs the value is delivered later via complete_task →
    // promise resolution.
    let start_ms = js_sys::Date::now();
    let initial = _zyntax_run_async(source, task_id);
    assert!(
        initial.ok(),
        "run_async should compile + start the task, output={}, kind={:?}",
        initial.output(),
        initial.error_kind()
    );

    // Wait for complete_task to resolve the JS Promise. Race against
    // a 5s timeout so a hung test fails fast (the underlying
    // wasm-bindgen-test runner's default is much longer and
    // suppresses console output on hang).
    let timeout_promise: js_sys::Promise = js_sys::eval(
        r#"
        new Promise(function(_, reject) {
            setTimeout(function() {
                reject(new Error('5s timeout — complete_task never fired'));
            }, 5000);
        })
        "#,
    )
    .expect("eval timeout")
    .dyn_into()
    .expect("Promise");
    let race = js_sys::Promise::race(&js_sys::Array::of2(&promise, &timeout_promise));
    let resolved = wasm_bindgen_futures::JsFuture::from(race)
        .await
        .expect("Promise should resolve cleanly");
    let elapsed_ms = js_sys::Date::now() - start_ms;

    // The resolver was called with the task's i64 value — should be
    // 42 from `return 42`.
    let value = resolved.as_f64().expect("resolver delivered a number");
    assert!(
        (value - 42.0).abs() < 0.5,
        "expected 42, got {value} (initial RunResult output={:?})",
        initial.output()
    );

    // Wall-clock proves the SM actually parked. setTimeout(100) +
    // wasm-bindgen-test scheduling overhead → expect ≥ 90ms.
    assert!(
        elapsed_ms >= 90.0,
        "expected wall-clock ≥ 90ms (cooperative parking + setTimeout); got {elapsed_ms}ms"
    );
}
