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
fn register_static_plugins(_rt: &mut InterpRuntime) {
    // Intentionally empty for the initial demo: a pure-arithmetic
    // `def main(): i64 { ... }` exercises the parse → lower →
    // interpret pipeline without needing any ZRTL symbols.
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
    let hir_module = match zyntax_compiler::compile_to_hir(&mut program, type_registry, config) {
        Ok(m) => m,
        Err(e) => return compile_err(format!("HIR lowering failed: {e}")),
    };

    // 3. Spin up an interpreter, register any statically-linked
    //    plugins, and dispatch `main`.
    let mut rt = InterpRuntime::new();
    register_static_plugins(&mut rt);
    rt.compile_module(hir_module);

    match rt.call_function("main", vec![]) {
        Ok(v) => RunResult {
            output: format_value(&v),
            ok: true,
            error_kind: ErrorKind::None,
        },
        Err(e) => runtime_err(format!("runtime error: {e}")),
    }
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
