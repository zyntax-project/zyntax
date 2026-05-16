//! Worker-pool runtime for browser environments. Gated behind the
//! `worker_pool` cargo feature (not yet enabled by default).
//!
//! The shape Phase F asks for: a cross-origin-isolated page can spawn
//! a pool of Web Workers each running a copy of the wasm module, and
//! the main thread's `Zyntax.call_async` distributes algebraic-effects
//! state-machine poll fns across them. Workers communicate with the
//! main thread through `SharedArrayBuffer` + `Atomics.wait`/`notify`.
//!
//! ## Why this is a stub today
//!
//! Threaded wasm in the rustc/cargo toolchain still requires:
//!
//!   - **Nightly Rust** for the `atomics` target feature
//!     (`-C target-feature=+atomics,+bulk-memory,+mutable-globals`).
//!   - `wasm-bindgen-rayon` (and its `init_thread_pool` JS shim) or
//!     an equivalent custom Worker bootstrap.
//!   - A build pipeline that runs `wasm-pack build --target web` with
//!     `RUSTFLAGS` and a custom rust-toolchain so the wasm module
//!     advertises the right capabilities to the browser.
//!   - The page itself MUST be cross-origin isolated
//!     (`Cross-Origin-Opener-Policy: same-origin` +
//!     `Cross-Origin-Embedder-Policy: require-corp`) for the
//!     browser to instantiate the threaded module — without those
//!     headers `SharedArrayBuffer` is unavailable and the threaded
//!     wasm import resolution fails.
//!
//! Until the threading toolchain stabilises on stable Rust, this
//! crate's default build stays single-threaded. The single-threaded
//! path covers every test we have today; only Phase F's async-effect
//! runtime needs the pool.
//!
//! ## What the API looks like when enabled
//!
//! ```ignore
//! #[cfg(feature = "worker_pool")]
//! pub async fn init_worker_pool(n_workers: usize) -> Result<(), JsValue> {
//!     wasm_bindgen_rayon::init_thread_pool(n_workers).await
//! }
//! ```
//!
//! Hosts gate the call on `crossOriginIsolated`:
//!
//! ```js
//! if (Zyntax.isolated()) {
//!     await initWorkerPool(navigator.hardwareConcurrency);
//! }
//! ```
//!
//! See `docs/wasm-deployment.md` for the header recipes and the
//! Cargo feature naming convention.

#[cfg(feature = "worker_pool")]
compile_error!(
    "worker_pool feature is a structural stub — flip the `compile_error!` \
     to the real wasm-bindgen-rayon init once the threaded-wasm toolchain \
     is wired (nightly Rust + RUSTFLAGS in build.sh + jsconfig in \
     examples/web). Until then this feature is intentionally inert."
);

/// Whether the worker-pool feature has been compiled in. Always
/// `false` on the default build; the docs/wasm-deployment.md doc
/// references this so embedder code can branch deterministically.
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
