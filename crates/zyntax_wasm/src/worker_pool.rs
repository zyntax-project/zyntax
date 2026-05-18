//! Browser threading model for zyntax_wasm. This module is
//! documentation-only — the actual code lives in `web/worker.js`
//! and `web/zynml.mjs::createZyntax`.
//!
//! ## Why no Rust code here
//!
//! The shipped browser story is a single Web Worker hosting the
//! BC interpreter. `createZyntax({ mode: "worker" })` in
//! `web/zynml.mjs` spawns the Worker (`web/worker.js`), which
//! loads its own copy of the wasm module and runs the interpreter
//! off the UI thread. The page communicates via `postMessage`
//! (`{cmd: "run", id, source}` → `{cmd: "result", id, output, ok,
//! errorKind}`). All of that is JS — the Rust side stays
//! single-threaded and lets the host pick whether to run the wasm
//! on the UI thread or in a Worker.
//!
//! Properties of the shipped path:
//!   - stable Rust toolchain;
//!   - no `SharedArrayBuffer`, no cross-origin isolation required;
//!   - UI stays responsive while a ZynML program runs to
//!     completion, even on long compute loops;
//!   - one Worker, not a pool — the wasm module is single-threaded
//!     and that's fine for the BC interpreter today.
//!
//! This mirrors `wren_lift/wasm/web/worker.js`, which has shaken
//! out the same model in production. wren_lift also has no
//! Rust-side worker code — for the same reason.
//!
//! ## Future: SAB-backed Worker pool
//!
//! A future rung would let the host expose its `WebAssembly.Memory`
//! (backed by `SharedArrayBuffer`) to N Workers and distribute
//! hot algebraic-effects poll fns across them via
//! `wasm-bindgen-rayon`. That rung does need:
//!
//!   - **Nightly Rust** for the `atomics` / `bulk-memory` /
//!     `mutable-globals` target features. Pinned in a
//!     `crates/zyntax_wasm/rust-toolchain.toml` when adopted.
//!   - `RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,
//!     +mutable-globals"` plus `wasm-pack build --target web
//!     --release -- -Z build-std=std,panic_abort`.
//!   - Cross-origin isolation (`COOP: same-origin` + `COEP:
//!     require-corp`). The vendored `coi-serviceworker.js`
//!     handles this without server-side header control.
//!
//! When that rung lands, the existing single-Worker path stays —
//! the SAB pool is a strict superset and falls back if the
//! environment can't provide isolation. The Rust-side init for
//! the pool will live here, behind whatever feature flag the
//! threading work introduces.

/// Whether the SAB-backed worker-pool rung is compiled in. Always
/// `false` today — the JS-side single-Worker mode covers every
/// browser test we have and requires no Rust-side feature flag.
pub const fn worker_pool_compiled_in() -> bool {
    false
}
