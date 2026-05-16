//! Worker-mode runtime status. The interesting code lives in
//! `web/worker.js` and `web/zynml.mjs::createZyntax`; this module
//! exists to document the threading model and expose compile-time
//! probes for embedders.
//!
//! ## What "worker mode" means in zyntax_wasm
//!
//! There are two threading rungs the runtime can sit on in a
//! browser. The first is shipped and stable; the second is opt-in
//! and gated on a build-pipeline upgrade.
//!
//! ### Rung 1 — single-Worker offload (default, stable, shipped)
//!
//! `createZyntax({ mode: "worker" })` in `web/zynml.mjs` spawns a
//! Web Worker hosting `web/worker.js`, which loads its own copy of
//! the wasm module and runs the BC interpreter off the UI thread.
//! The page communicates via `postMessage` (`{cmd: "run", id,
//! source}` → `{cmd: "result", id, output, ok, errorKind}`).
//!
//! Properties:
//!   - stable Rust toolchain, no `RUSTFLAGS` gymnastics;
//!   - no `SharedArrayBuffer`, no cross-origin isolation required;
//!   - UI stays responsive while a ZynML program runs to
//!     completion, even on long compute loops;
//!   - one Worker, not a pool — the wasm module is single-threaded
//!     and that's fine for the BC interpreter today.
//!
//! This is the model `wren_lift/wasm/web/worker.js` ships with;
//! it covers every test we have and is what the headless-Chrome
//! CI smoke exercises.
//!
//! ### Rung 2 — SAB-backed Worker pool (opt-in, future)
//!
//! `mode: "worker", shared: true` (not yet exposed) would ask the
//! Worker to expose its `WebAssembly.Memory` (backed by
//! `SharedArrayBuffer`) to the page, and a future
//! `wasm-bindgen-rayon`-driven scheduler would distribute hot
//! algebraic-effects poll fns across N Workers. This is the rung
//! that needs:
//!
//!   - **Nightly Rust** for the `atomics` / `bulk-memory` /
//!     `mutable-globals` target features. Pinned in a
//!     `crates/zyntax_wasm/rust-toolchain.toml` when adopted.
//!   - `RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,
//!     +mutable-globals"` plus `wasm-pack build --target web
//!     --release -- -Z build-std=std,panic_abort`.
//!   - The page MUST be cross-origin isolated (`COOP: same-origin`
//!     + `COEP: require-corp`). The vendored `coi-serviceworker.js`
//!     handles this without server-side header control — the
//!     service worker re-writes responses to add the headers, and
//!     `Zyntax.isolated()` reports the resulting state.
//!
//! When this rung lands, the existing single-Worker path stays —
//! `shared: true` is a strict superset and falls back if the
//! environment can't provide isolation.
//!
//! ## Compile-time probes
//!
//! The `worker_pool` cargo feature is reserved for the rung-2
//! build (it'll flip on the SAB-backed `wasm-bindgen-rayon` glue
//! when that lands). Today it's purely a marker — enabling it
//! `compile_error!`s loudly so accidental flips don't ship a
//! half-built runtime.
//!
//! Embedders can branch on `worker_pool_compiled_in()` at compile
//! time to decide whether the threaded path is available. The
//! single-Worker rung is always available in browser contexts
//! regardless of this flag.

#[cfg(feature = "worker_pool")]
compile_error!(
    "worker_pool feature is reserved for the SAB-backed wasm-bindgen-rayon \
     build. The single-Worker offload path is implemented in JS (see \
     crates/zyntax_wasm/web/worker.js + createZyntax in zynml.mjs) and is \
     always available without this feature. Flip the compile_error! to the \
     real wasm_bindgen_rayon::init_thread_pool call once the threaded-wasm \
     toolchain is wired up (nightly Rust + RUSTFLAGS in build.sh)."
);

/// Whether the SAB-backed worker-pool rung has been compiled in.
/// Always `false` on the default build. Independent of whether the
/// JS-side single-Worker mode is in use — that mode requires no
/// Rust-side feature.
pub const fn worker_pool_compiled_in() -> bool {
    #[cfg(feature = "worker_pool")]
    {
        true
    }
    #[cfg(not(feature = "worker_pool"))]
    {
        false
    }
}
