//! wasm-bindgen entry shim for the Zyntax/ZynML interpreter.
//!
//! Modeled directly on `wren_lift/wasm/src/lib.rs`. Two things are
//! exported to JS / wasi hosts:
//!
//!   * `version()` — returns a build-identifying string, useful as a
//!     smoke test that the wasm module loaded at all.
//!   * `run(source: &str) -> RunResult` — parses the source as a
//!     ZynML module, lowers it to HIR, drives the BC interpreter,
//!     and returns the captured result plus an `ok` flag.
//!
//! No JIT, no plugin loading from disk, no fs / sockets — this is
//! the minimal browser demo surface. The wasm-encoder hot-function
//! tier-1 JIT (Phase E) will plug in here later, alongside the
//! static plugin registration scaffolding below.

#![cfg_attr(not(target_arch = "wasm32"), allow(unused))]

use wasm_bindgen::prelude::*;

use zyntax_compiler::hir_interp::InterpError;
use zyntax_compiler::wasm_backend::WasmBackend;
use zyntax_embed::interp_runtime::InterpRuntime;
use zyntax_embed::{Grammar2, ZyntaxString, ZyntaxValue};

// ---------------------------------------------------------------------------
// ZynML grammar — included at build time so the wasm module is self-contained.
// ---------------------------------------------------------------------------
//
// The `zynml` crate also `include_str!`s these, but the wasm shim
// doesn't depend on `zynml` (which itself pulls in the full native
// runtime + CLI deps). Including the grammar files directly here
// keeps the wasm dep tree minimal: zyntax_embed (parse-only) + the
// SDK + wasm-bindgen, nothing else.
const ZYNML_GRAMMAR: &str = include_str!("../../zynml/ml.zyn");

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a single `run` call. Exported to JS as a structural
/// object via wasm-bindgen's getter convention.
#[wasm_bindgen]
pub struct RunResult {
    output: String,
    ok: bool,
    error_kind: ErrorKind,
}

/// Classification of run outcomes, mirroring `wren_lift`'s shape so
/// host UIs can branch on the same categories.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum ErrorKind {
    /// `ok == true`. No error.
    None,
    /// Grammar / parse / lowering failed before the interpreter ran.
    CompileError,
    /// The interpreter started executing and then failed.
    RuntimeError,
}

#[wasm_bindgen]
impl RunResult {
    #[wasm_bindgen(getter)]
    pub fn output(&self) -> String {
        self.output.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn ok(&self) -> bool {
        self.ok
    }

    #[wasm_bindgen(getter, js_name = errorKind)]
    pub fn error_kind(&self) -> ErrorKind {
        self.error_kind
    }
}

// ---------------------------------------------------------------------------
// Init helpers
// ---------------------------------------------------------------------------

/// Install the panic hook so unhandled Rust panics surface as
/// `console.error` with a stack trace instead of an opaque
/// "unreachable executed" wasm trap. Called automatically by the
/// `#[wasm_bindgen(start)]` shim below; hosts that load the wasm
/// module by other means can call this themselves.
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
    }
}

#[wasm_bindgen(start)]
fn on_load() {
    init_panic_hook();
}

/// Build identifier — useful as a smoke test that the module loaded
/// and exposes the expected exports.
#[wasm_bindgen]
pub fn version() -> String {
    format!(
        "zyntax_wasm {} (interpreter-only; wasm32 build)",
        env!("CARGO_PKG_VERSION"),
    )
}

// ---------------------------------------------------------------------------
// Wasm-JIT hooks (Phase E.6 + E.7)
// ---------------------------------------------------------------------------
//
// The interpreter (zyntax_compiler::hir_interp) keeps every cold call
// in bytecode. When a function's call count crosses the hot
// threshold the interpreter calls our `compile_hook` with the
// `HirFunction`; we run `WasmBackend::compile_function`, ship the
// emitted wasm bytes to JS via `_zyntax_jit_install`, and store the
// returned funcref table index as the function's JIT handle. On
// every subsequent call the interpreter routes through
// `_zyntax_jit_call_*` (one extern per arity) instead of running BC.
//
// JS holds the funcref table off-wasm because:
//   * wasm32 has no addressable function pointers; only table indices.
//   * `WebAssembly.compile(bytes)` lives on the JS side anyway, so
//     it makes sense for JS to own the table the resulting Instance's
//     exports live in.
//
// The host page provides three globals (see `web/zynml.mjs`):
//   * `_zyntax_jit_install(bytes_ptr, bytes_len) -> u32` — install
//     a freshly-emitted wasm module, returns its table index.
//   * `_zyntax_jit_call_0_i64(handle) -> i64` — zero-arg dispatch
//     for `entry` exports that return i64.
//
// (Higher-arity dispatch externs land in Phase E.6.1 once `WasmBackend`
// emits multi-arg call sites.)

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    /// JS-provided shim:
    ///
    /// ```js
    /// globalThis._zyntax_jit_install = (bytes) => {
    ///   const mod  = new WebAssembly.Module(bytes);
    ///   const inst = new WebAssembly.Instance(mod, { /* imports */ });
    ///   return jitRegistry.push(inst.exports.entry) - 1;
    /// };
    /// ```
    ///
    /// `bytes` arrives as a `Uint8Array` — wasm-bindgen marshalls
    /// the `&[u8]` through linear memory automatically. Returns
    /// `u32::MAX` (0xFFFFFFFF) to signal failure; we drop that
    /// handle and keep the function in BC.
    #[wasm_bindgen(js_namespace = globalThis, js_name = _zyntax_jit_install)]
    fn js_jit_install(bytes: &[u8]) -> u32;

    /// Zero-arg / i64-return dispatch shim:
    ///
    /// ```js
    /// globalThis._zyntax_jit_call_0_i64 = (handle) =>
    ///   jitRegistry[handle]();
    /// ```
    ///
    /// JIT'd `entry` exports return wasm `i64` which wasm-bindgen
    /// surfaces as a JS `BigInt`. The shim returns it directly;
    /// wasm-bindgen marshalls back to Rust `i64` on the boundary.
    #[wasm_bindgen(js_namespace = globalThis, js_name = _zyntax_jit_call_0_i64)]
    fn js_jit_call_0_i64(handle: u32) -> i64;
}

// Non-wasm32 stubs so the crate's `cargo test` (native target) can
// still build. The native test pipeline doesn't exercise the JIT
// hooks — wasm-pack + Node.js does (see `test/node_smoke.mjs`).
#[cfg(not(target_arch = "wasm32"))]
fn js_jit_install(_bytes: &[u8]) -> u32 {
    u32::MAX
}

#[cfg(not(target_arch = "wasm32"))]
fn js_jit_call_0_i64(_handle: u32) -> i64 {
    0
}

/// Sentinel returned from `_zyntax_jit_install` on JS-side failure.
/// Matches the docstring on the extern above.
const JIT_INSTALL_FAILED: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Native-symbol bridge for JIT'd extern calls (Phase E.5)
// ---------------------------------------------------------------------------
//
// JIT'd modules import `(extern "<name>@<arity>")` for every ZRTL
// symbol they call. The host page builds an `importObject` for the
// instantiation by parsing the wasm module's imports list; each
// dispatcher calls back into this wasm via one of the
// `_zyntax_call_extern_<N>` exports below. We thread the active
// runtime's symbol table through a `RefCell` so the exports can
// resolve `name` to a function pointer without owning the runtime.
//
// SAFETY: the function pointers come from statically-linked plugin
// crates on wasm32 (Phase C), so they're real wasm function
// references that wasm-bindgen's transmute can call directly.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Snapshot of the active `InterpRuntime`'s symbol table for
    /// the duration of a `run()` call. Populated by `run_impl`
    /// before dispatch and cleared on exit. Keyed by raw symbol
    /// name (without the `@<arity>` suffix that the wasm import
    /// carries — the JS dispatcher strips it before calling our
    /// exports).
    static ACTIVE_SYMBOLS: RefCell<HashMap<String, *const u8>> =
        RefCell::new(HashMap::new());
}

fn lookup_active_symbol(name: &str) -> Option<*const u8> {
    ACTIVE_SYMBOLS.with(|s| s.borrow().get(name).copied())
}

/// Dispatcher for zero-arg ZRTL externs. JS-side
/// `_zyntax_call_extern_0(name)` from a JIT'd module's import
/// shim calls into this. Returns `0` when the symbol isn't found
/// (defensive — the JIT compile gate only fires when the symbol
/// is in the snapshot, but a stale handle after runtime tear-down
/// would otherwise be UB).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_extern_0(name: &str) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn() -> i64 = unsafe { core::mem::transmute(ptr) };
    f()
}

/// One-i64-arg dispatcher.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_extern_1(name: &str, a0: i64) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64) -> i64 = unsafe { core::mem::transmute(ptr) };
    f(a0)
}

/// Two-i64-arg dispatcher.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_extern_2(name: &str, a0: i64, a1: i64) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64, i64) -> i64 = unsafe { core::mem::transmute(ptr) };
    f(a0, a1)
}

/// Three-i64-arg dispatcher.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_extern_3(name: &str, a0: i64, a1: i64, a2: i64) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64, i64, i64) -> i64 = unsafe { core::mem::transmute(ptr) };
    f(a0, a1, a2)
}

/// Four-i64-arg dispatcher.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_extern_4(name: &str, a0: i64, a1: i64, a2: i64, a3: i64) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(ptr) };
    f(a0, a1, a2, a3)
}

/// Five-i64-arg dispatcher.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_extern_5(name: &str, a0: i64, a1: i64, a2: i64, a3: i64, a4: i64) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = unsafe { core::mem::transmute(ptr) };
    f(a0, a1, a2, a3, a4)
}

/// Six-i64-arg dispatcher.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_extern_6(
    name: &str,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
        unsafe { core::mem::transmute(ptr) };
    f(a0, a1, a2, a3, a4, a5)
}

/// Seven-i64-arg dispatcher.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_extern_7(
    name: &str,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 =
        unsafe { core::mem::transmute(ptr) };
    f(a0, a1, a2, a3, a4, a5, a6)
}

/// Eight-i64-arg dispatcher. Covers `zrtl_tensor::matmul` and the
/// remaining double-handful-of-args plugin entry points we ship today
/// (zrtl_simd, zrtl_audio mixing); higher arities currently bail at
/// `makeExternDispatcher` (zynml.mjs) with a clear error rather than
/// silently miscalling.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_extern_8(
    name: &str,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    a7: i64,
) -> i64 {
    let Some(ptr) = lookup_active_symbol(name) else {
        return 0;
    };
    let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
        unsafe { core::mem::transmute(ptr) };
    f(a0, a1, a2, a3, a4, a5, a6, a7)
}

// Non-wasm32 stubs so the crate's native cargo build is happy.
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_extern_0(_name: &str) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_extern_1(_name: &str, _a0: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_extern_2(_name: &str, _a0: i64, _a1: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_extern_3(_name: &str, _a0: i64, _a1: i64, _a2: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_extern_4(_name: &str, _a0: i64, _a1: i64, _a2: i64, _a3: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_extern_5(_name: &str, _a0: i64, _a1: i64, _a2: i64, _a3: i64, _a4: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_extern_6(
    _name: &str,
    _a0: i64,
    _a1: i64,
    _a2: i64,
    _a3: i64,
    _a4: i64,
    _a5: i64,
) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_extern_7(
    _name: &str,
    _a0: i64,
    _a1: i64,
    _a2: i64,
    _a3: i64,
    _a4: i64,
    _a5: i64,
    _a6: i64,
) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_extern_8(
    _name: &str,
    _a0: i64,
    _a1: i64,
    _a2: i64,
    _a3: i64,
    _a4: i64,
    _a5: i64,
    _a6: i64,
    _a7: i64,
) -> i64 {
    0
}

// ---------------------------------------------------------------------------
// Internal-call bridge for cross-function JIT calls
// ---------------------------------------------------------------------------
//
// `HirCallable::Function(id)` calls inside JIT'd wasm modules emit
// imports under the `internal` module: `internal.<hex_id>@<arity>`
// (see `WasmBackend::scan_imports`). The JS dispatcher
// (`makeInternalDispatcher` in zynml.mjs) routes those imports
// through the matching `_zyntax_call_internal_<N>` export below.
//
// Each export looks up the hex id in `ACTIVE_INTERNAL_FNS` (populated
// by `run_impl` from the runtime's module) to recover the function's
// name, then reaches the active `InterpRuntime` through
// `ACTIVE_RUNTIME` and calls `call_function`. The interp may dispatch
// the inner call to a sibling JIT'd module (recursive wasm call), to
// the BC interpreter, or to its tier-up infrastructure — whichever is
// current for that function.
//
// SAFETY: `ACTIVE_RUNTIME` carries a raw `*mut InterpRuntime` that's
// also borrowed mutably by the outer `call_function` call. Single-
// threaded wasm makes this re-entry sound (the outer borrow is parked
// in a paused wasm call frame while the inner dispatch runs through
// the same runtime). The unsafety is contained here; the contract
// callers see is the standard "every export resolves through the
// active runtime."

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// Pointer to the runtime that's currently executing a `run()`
    /// call. Populated by `run_impl` before dispatch, cleared on
    /// exit. Null between calls.
    static ACTIVE_RUNTIME: RefCell<*mut InterpRuntime> =
        const { RefCell::new(core::ptr::null_mut()) };

    /// Map of hex-encoded HirId → function name for every function in
    /// the active runtime's module. Populated by `run_impl` so the
    /// `_zyntax_call_internal_<N>` exports can resolve a `Function(id)`
    /// import back to the name `InterpRuntime::call_function` accepts.
    static ACTIVE_INTERNAL_FNS: RefCell<HashMap<String, String>> =
        RefCell::new(HashMap::new());
}

#[cfg(target_arch = "wasm32")]
fn call_internal_by_hex(hex_id: &str, args: Vec<zyntax_compiler::value::ZyntaxValue>) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    let name = match ACTIVE_INTERNAL_FNS.with(|m| m.borrow().get(hex_id).cloned()) {
        Some(n) => n,
        None => return 0,
    };
    let rt_ptr = ACTIVE_RUNTIME.with(|r| *r.borrow());
    if rt_ptr.is_null() {
        return 0;
    }
    // SAFETY: see the module-level comment above. The outer caller's
    // `&mut` to the runtime is parked in a wasm call frame; we re-
    // enter via this *mut for the duration of the inner call only.
    let rt = unsafe { &mut *rt_ptr };
    match rt.call_function(&name, args) {
        Ok(ZyntaxValue::Int(i)) => i,
        Ok(_) => 0,
        Err(_) => 0,
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_internal_0(hex_id: &str) -> i64 {
    call_internal_by_hex(hex_id, vec![])
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_internal_1(hex_id: &str, a0: i64) -> i64 {
    call_internal_by_hex(hex_id, vec![zyntax_compiler::value::ZyntaxValue::Int(a0)])
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_internal_2(hex_id: &str, a0: i64, a1: i64) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    call_internal_by_hex(hex_id, vec![ZyntaxValue::Int(a0), ZyntaxValue::Int(a1)])
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_internal_3(hex_id: &str, a0: i64, a1: i64, a2: i64) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    call_internal_by_hex(
        hex_id,
        vec![
            ZyntaxValue::Int(a0),
            ZyntaxValue::Int(a1),
            ZyntaxValue::Int(a2),
        ],
    )
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_internal_4(hex_id: &str, a0: i64, a1: i64, a2: i64, a3: i64) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    call_internal_by_hex(
        hex_id,
        vec![
            ZyntaxValue::Int(a0),
            ZyntaxValue::Int(a1),
            ZyntaxValue::Int(a2),
            ZyntaxValue::Int(a3),
        ],
    )
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn _zyntax_call_internal_5(hex_id: &str, a0: i64, a1: i64, a2: i64, a3: i64, a4: i64) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    call_internal_by_hex(
        hex_id,
        vec![
            ZyntaxValue::Int(a0),
            ZyntaxValue::Int(a1),
            ZyntaxValue::Int(a2),
            ZyntaxValue::Int(a3),
            ZyntaxValue::Int(a4),
        ],
    )
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_internal_6(
    hex_id: &str,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    call_internal_by_hex(
        hex_id,
        vec![
            ZyntaxValue::Int(a0),
            ZyntaxValue::Int(a1),
            ZyntaxValue::Int(a2),
            ZyntaxValue::Int(a3),
            ZyntaxValue::Int(a4),
            ZyntaxValue::Int(a5),
        ],
    )
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_internal_7(
    hex_id: &str,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    call_internal_by_hex(
        hex_id,
        vec![
            ZyntaxValue::Int(a0),
            ZyntaxValue::Int(a1),
            ZyntaxValue::Int(a2),
            ZyntaxValue::Int(a3),
            ZyntaxValue::Int(a4),
            ZyntaxValue::Int(a5),
            ZyntaxValue::Int(a6),
        ],
    )
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_internal_8(
    hex_id: &str,
    a0: i64,
    a1: i64,
    a2: i64,
    a3: i64,
    a4: i64,
    a5: i64,
    a6: i64,
    a7: i64,
) -> i64 {
    use zyntax_compiler::value::ZyntaxValue;
    call_internal_by_hex(
        hex_id,
        vec![
            ZyntaxValue::Int(a0),
            ZyntaxValue::Int(a1),
            ZyntaxValue::Int(a2),
            ZyntaxValue::Int(a3),
            ZyntaxValue::Int(a4),
            ZyntaxValue::Int(a5),
            ZyntaxValue::Int(a6),
            ZyntaxValue::Int(a7),
        ],
    )
}

// Non-wasm32 stubs so the crate's native cargo build is happy.
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_internal_0(_hex_id: &str) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_internal_1(_hex_id: &str, _a0: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_internal_2(_hex_id: &str, _a0: i64, _a1: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_internal_3(_hex_id: &str, _a0: i64, _a1: i64, _a2: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_internal_4(_hex_id: &str, _a0: i64, _a1: i64, _a2: i64, _a3: i64) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
pub fn _zyntax_call_internal_5(
    _hex_id: &str,
    _a0: i64,
    _a1: i64,
    _a2: i64,
    _a3: i64,
    _a4: i64,
) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_internal_6(
    _hex_id: &str,
    _a0: i64,
    _a1: i64,
    _a2: i64,
    _a3: i64,
    _a4: i64,
    _a5: i64,
) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_internal_7(
    _hex_id: &str,
    _a0: i64,
    _a1: i64,
    _a2: i64,
    _a3: i64,
    _a4: i64,
    _a5: i64,
    _a6: i64,
) -> i64 {
    0
}
#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn _zyntax_call_internal_8(
    _hex_id: &str,
    _a0: i64,
    _a1: i64,
    _a2: i64,
    _a3: i64,
    _a4: i64,
    _a5: i64,
    _a6: i64,
    _a7: i64,
) -> i64 {
    0
}

/// Install the wasm-JIT compile + dispatch hooks on an
/// `InterpRuntime`. Called from `run_impl` once per fresh runtime.
///
/// The compile hook:
///   * runs `WasmBackend::compile_function`,
///   * hands the bytes to JS, gets a handle back,
///   * returns `Some(handle)` on success and `None` on failure
///     (unsupported HIR shape, JS-side install failure, function
///     signature outside our zero-arg-i64-return demo coverage).
///
/// The dispatch hook unboxes the args to match the function's
/// signature and calls the matching `js_jit_call_*` extern. For now
/// only `() -> i64` is supported; anything else bails out as
/// `InterpError::Host(...)`.
fn register_wasm_jit_hooks(rt: &mut InterpRuntime) {
    let backend = WasmBackend::new();
    let compile_hook = Box::new(
        move |func: &zyntax_compiler::hir::HirFunction| -> Option<u32> {
            // Only zero-arg / single-i64-return for now. Anything else
            // stays BC — matches the WasmBackend coverage gates.
            if !func.signature.params.is_empty() {
                return None;
            }
            if func.signature.returns.len() != 1
                || !matches!(
                    func.signature.returns[0],
                    zyntax_compiler::hir::HirType::I64
                )
            {
                return None;
            }
            let module = backend.compile_function(func).ok()?;
            let handle = js_jit_install(&module.bytes);
            if handle == JIT_INSTALL_FAILED {
                None
            } else {
                Some(handle)
            }
        },
    );

    let dispatch_hook = Box::new(
        |handle: u32, args: &[ZyntaxValue]| -> Result<ZyntaxValue, InterpError> {
            if !args.is_empty() {
                return Err(InterpError::Host(
                    "wasm-JIT dispatch: only zero-arg functions supported in this slice"
                        .to_string(),
                ));
            }
            let raw = js_jit_call_0_i64(handle);
            Ok(ZyntaxValue::Int(raw))
        },
    );

    rt.install_wasm_jit_hooks(compile_hook, dispatch_hook);
}

// ---------------------------------------------------------------------------
// Static plugin registration
// ---------------------------------------------------------------------------
//
// Plugins are statically linked into the wasm module as path deps
// (see Cargo.toml — Phase C wires `zrtl::StaticPlugin` and
// `InterpRuntime::register_static_plugin`). For now no plugins are
// pinned here; once we add `zrtl_io` etc. as wasm deps, this is the
// seam:
//
//     fn register_static_plugins(rt: &mut InterpRuntime) {
//         rt.register_static_plugin(zrtl_io::static_plugin());
//         rt.register_static_plugin(zrtl_string::static_plugin());
//         // ...
//     }
fn register_static_plugins(rt: &mut InterpRuntime) {
    // Intentionally empty for the initial demo: a pure-arithmetic
    // `def main(): i64 { ... }` exercises the parse → lower →
    // interpret pipeline without needing any ZRTL symbols.

    // Phase E.5 end-to-end smoke test extern. `__zw_test_double` is
    // a host-provided extern that doubles its single i64 argument.
    // Used by `test/node_smoke.mjs` to verify the JIT'd module →
    // JS dispatcher → `_zyntax_call_extern_1` → ACTIVE_SYMBOLS →
    // transmute path actually executes a host call correctly.
    // Always registered (cheap; just adds one entry to the symbol
    // table) so the smoke test doesn't need a feature flag.
    rt.register_symbol("__zw_test_double", __zw_test_double as *const u8, 1);
}

/// Test-only extern: doubles its argument. Lives outside the
/// `register_static_plugins` body so its address survives across
/// runtime tear-downs and can safely be transmuted to
/// `extern "C" fn(i64) -> i64` inside `_zyntax_call_extern_1`.
extern "C" fn __zw_test_double(x: i64) -> i64 {
    x.wrapping_mul(2)
}

// ---------------------------------------------------------------------------
// run() — the demo entry point
// ---------------------------------------------------------------------------

/// Compile + interpret a ZynML source string.
///
/// Returns a `RunResult` whose `output` field carries the
/// stringified return value of `main()` (or the error message on
/// failure). `ok` is `true` iff `main()` returned normally.
///
/// The host page is expected to call this once per run — each
/// invocation builds a fresh `InterpRuntime` so state doesn't bleed
/// between scripts.
#[wasm_bindgen]
pub fn run(source: &str) -> RunResult {
    run_impl(source)
}

fn run_impl(source: &str) -> RunResult {
    // 1. Parse the source through the ZynML grammar.
    let grammar = match Grammar2::from_source(ZYNML_GRAMMAR) {
        Ok(g) => g,
        Err(e) => return compile_err(format!("ZynML grammar failed to load: {e}")),
    };
    let mut program = match grammar.parse_with_filename(source, "<run>") {
        Ok(p) => p,
        Err(e) => return compile_err(format!("parse error: {e}")),
    };

    // 2. Lower the TypedProgram to HIR. We share the program's own
    //    TypeRegistry — that's how `compile_to_hir` wants its second
    //    argument shaped.
    let type_registry = std::sync::Arc::new(program.type_registry.clone());
    let mut config = zyntax_compiler::CompilationConfig::default();
    // The interpreter consumes HIR directly, so disable HIR-level
    // optimisations that target the Cranelift consumer.
    config.opt_level = 0;
    // Inject extern aliases the wasm-only test surface needs. Each
    // entry goes into `LoweringConfig.builtins` → `extern_link_names`
    // BEFORE the source's `collect_declarations` runs. SSA's Call
    // arm checks `extern_link_names` and emits `HirCallable::Symbol`
    // — which is the only callee shape `WasmBackend` currently
    // emits to a wasm `(import "extern" "<name>@<arity>" …)`. The
    // matching host symbol is registered in `register_static_plugins`.
    config.builtins.insert(
        "__zw_test_double".to_string(),
        "__zw_test_double".to_string(),
    );
    let hir_module = match zyntax_compiler::compile_to_hir(&mut program, type_registry, config) {
        Ok(m) => m,
        Err(e) => return compile_err(format!("HIR lowering failed: {e}")),
    };

    // 3. Spin up an interpreter, register any statically-linked
    //    plugins, wire the wasm-JIT tier-up hooks, and dispatch
    //    `main`.
    let mut rt = InterpRuntime::new();
    register_static_plugins(&mut rt);
    register_wasm_jit_hooks(&mut rt);
    rt.compile_module(hir_module);

    // Mirror the runtime's FFI symbol table into the thread-local
    // store so JIT'd modules' extern calls (Phase E.5) can resolve
    // names through `_zyntax_call_extern_*` without holding a
    // reference to the runtime.
    let symbol_snapshot = rt.symbol_table_snapshot();
    ACTIVE_SYMBOLS.with(|s| {
        let mut map = s.borrow_mut();
        map.clear();
        for (name, ptr, _arity) in symbol_snapshot {
            map.insert(name, ptr);
        }
    });

    // Mirror the HIR module's function table — hex-id → name — so
    // JIT'd modules' internal calls (HirCallable::Function) can
    // resolve `internal.<hex_id>@<arity>` imports back to the
    // function name `InterpRuntime::call_function` accepts. Only
    // wasm32 builds populate this; the native build's
    // ACTIVE_INTERNAL_FNS is gated out.
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(module) = rt.module() {
            let snapshot: Vec<(String, String)> = module
                .functions
                .iter()
                .filter_map(|(id, f)| {
                    let name = f.name.resolve_global()?;
                    Some((id.to_hex(), name))
                })
                .collect();
            ACTIVE_INTERNAL_FNS.with(|m| {
                let mut map = m.borrow_mut();
                map.clear();
                for (hex, name) in snapshot {
                    map.insert(hex, name);
                }
            });
        }
        // Park a *mut to the runtime so internal-call dispatchers
        // can re-enter `call_function`. SAFETY: see the module-level
        // comment on ACTIVE_RUNTIME — single-threaded wasm makes the
        // re-entry sound.
        let rt_ptr: *mut InterpRuntime = &mut rt;
        ACTIVE_RUNTIME.with(|r| *r.borrow_mut() = rt_ptr);
    }

    let result = match rt.call_function("main", vec![]) {
        Ok(v) => RunResult {
            output: format_value(&v),
            ok: true,
            error_kind: ErrorKind::None,
        },
        Err(e) => runtime_err(format!("runtime error: {e}")),
    };

    // Clear the active-symbol thread-local so a subsequent `run()`
    // with a different runtime doesn't see stale pointers. Pointer
    // staleness here would manifest as silent miscompile / UB
    // (transmuting an old plugin's fn ptr through the new
    // runtime's wasm-JIT'd module).
    ACTIVE_SYMBOLS.with(|s| s.borrow_mut().clear());
    #[cfg(target_arch = "wasm32")]
    {
        ACTIVE_INTERNAL_FNS.with(|m| m.borrow_mut().clear());
        ACTIVE_RUNTIME.with(|r| *r.borrow_mut() = core::ptr::null_mut());
    }

    result
}

fn compile_err(msg: impl Into<String>) -> RunResult {
    RunResult {
        output: msg.into(),
        ok: false,
        error_kind: ErrorKind::CompileError,
    }
}

fn runtime_err(msg: impl Into<String>) -> RunResult {
    RunResult {
        output: msg.into(),
        ok: false,
        error_kind: ErrorKind::RuntimeError,
    }
}

/// Render a `ZyntaxValue` for the `output` field. The browser demo
/// only inspects primitive returns from `main()`; richer types
/// (structs, arrays) just get their `{:?}` form until a proper
/// `Display` adapter is wired through.
fn format_value(v: &ZyntaxValue) -> String {
    match v {
        ZyntaxValue::Int(n) => n.to_string(),
        ZyntaxValue::UInt(n) => n.to_string(),
        ZyntaxValue::Float(n) => n.to_string(),
        ZyntaxValue::Bool(b) => b.to_string(),
        ZyntaxValue::String(s) => s.as_str().to_string(),
        ZyntaxValue::Null => "null".to_string(),
        _ => format!("{:?}", v),
    }
}

// Marker so `ZyntaxString` is visible to rust-analyzer in unused-
// import diagnostics on non-wasm builds.
#[allow(dead_code)]
fn _zs_marker(_s: ZyntaxString) {}
