//! Pluggable backend for first-class fibers.
//!
//! `FiberCfg` is the contract every fiber backend implements. The
//! contract is C-ABI-shaped so the same vtable serves the BC interp's
//! typed FFI marshalling, Cranelift's direct calls, and LLVM's
//! external symbol resolution. The runtime symbol stubs in [`crate::zrtl`]
//! delegate every fiber op to the currently-installed backend; one
//! global install slot, set once at runtime construction.
//!
//! ## Available backends
//!
//! * **`krio_adapter::fiber::KrioFiberBackend`** — stackful, native,
//!   default. Wraps krio-fiber's per-fiber `mmap`-allocated stacks +
//!   per-arch context switch.
//! * *(future)* a wasm stack-switching backend once the proposal
//!   stabilises on wasmtime / browsers.
//! * *(future)* a stackless-emulation backend for wasm targets without
//!   stack switching — runs a whole-program transform at lower time
//!   rather than at fiber-creation time.
//!
//! Frontends never see the backend choice — the HIR's fiber ops lower
//! to `Call::Symbol("krio_fiber_*")` and the symbol resolution does the
//! rest.
//!
//! ## Install / replace
//!
//! [`install_fiber_backend`] is set-once. Calling it a second time
//! returns `false`; reset via the unsafe [`force_replace_fiber_backend`]
//! (test-only helper that's also gated by an env var to avoid
//! production foot-gun risk).

use crate::zrtl::FiberRepr;
use std::sync::OnceLock;

/// C-ABI-shaped contract for fiber primitives. Each method maps 1-1
/// with a `krio_fiber_*` symbol the HIR lowering pass emits.
///
/// Implementors must be `Send + Sync` — the backend itself is shared
/// across threads. Per-fiber state stays thread-local; the caller is
/// expected to resume each `FiberRepr` on the same thread that created
/// it (matches the constraint stackful fiber runtimes universally
/// have).
pub trait FiberCfg: Send + Sync {
    /// Construct a paused fiber from a closure handle and stack size
    /// hint. The closure handle's encoding is backend-defined; today
    /// it's an `extern "C" fn()` pointer (no captured environment).
    /// When the frontend gains closure-with-captures support, the
    /// handle becomes a richer struct pointer.
    ///
    /// Returns an opaque `*mut FiberRepr` handle the runtime threads
    /// through subsequent ops.
    ///
    /// # Safety
    /// `closure` must be a valid pointer the backend knows how to
    /// invoke. `stack_size` is a byte hint; backends may round up to
    /// page size.
    unsafe fn fiber_new(&self, closure: *mut u8, stack_size: i64) -> *mut FiberRepr;

    /// Drive the fiber until its next yield or completion. Returns a
    /// packed `FiberStep` encoding:
    /// * bits 0..2 — tag: 0 = Yielded, 1 = Done, 2 = Errored
    /// * bits 2..64 — payload (the yielded u64 value when tag = Yielded)
    ///
    /// # Safety
    /// `fiber` must be a handle returned by `fiber_new` (or a prior
    /// `fiber_resume*`) that hasn't been moved between threads.
    unsafe fn fiber_resume(&self, fiber: *mut FiberRepr) -> i64;

    /// Resume the fiber, delivering `value` as the result of its
    /// currently-blocked `FiberYield`. Same packed return shape as
    /// [`Self::fiber_resume`].
    unsafe fn fiber_resume_with(&self, fiber: *mut FiberRepr, value: i64) -> i64;

    /// Suspend the enclosing fiber. Only valid from inside a fiber
    /// body — the backend looks up the current fiber via a thread-
    /// local and yields it with `value`.
    fn fiber_yield(&self, value: i64);

    /// Symmetric switch — abandon the caller and transfer to `target`
    /// with `value`. Returns the value the current fiber receives when
    /// somebody later transfers back to it. Packed return shape
    /// matches [`Self::fiber_resume`].
    ///
    /// # Safety
    /// `target` must be a valid fiber handle.
    unsafe fn fiber_transfer(&self, target: *mut FiberRepr, value: i64) -> i64;

    /// Cooperative cancel — set the flag the fiber polls via
    /// `should_yield_early`. The fiber is responsible for checking
    /// and exiting cleanly.
    ///
    /// # Safety
    /// `fiber` must be a valid handle.
    unsafe fn fiber_cancel(&self, fiber: *mut FiberRepr);
}

/// The FiberStep tag returned by `fiber_resume*` in the low bits.
pub const FIBER_STEP_YIELDED: i64 = 0;
pub const FIBER_STEP_DONE: i64 = 1;
pub const FIBER_STEP_ERRORED: i64 = 2;

/// Bit shift between the tag and the packed payload.
pub const FIBER_STEP_TAG_BITS: u32 = 2;

/// Pack a (tag, payload) pair into a single `i64`. Used by backends
/// implementing [`FiberCfg::fiber_resume`].
#[inline]
pub fn pack_fiber_step(tag: i64, payload: i64) -> i64 {
    debug_assert!(tag >= 0 && tag <= FIBER_STEP_ERRORED, "invalid step tag");
    (payload << FIBER_STEP_TAG_BITS) | tag
}

/// Unpack a `FiberStep` value into `(tag, payload)`. Useful for tests
/// and for the future HIR-level decoder.
#[inline]
pub fn unpack_fiber_step(step: i64) -> (i64, i64) {
    let tag = step & ((1 << FIBER_STEP_TAG_BITS) - 1);
    let payload = step >> FIBER_STEP_TAG_BITS;
    (tag, payload)
}

static FIBER_BACKEND: OnceLock<Box<dyn FiberCfg>> = OnceLock::new();

/// Install `backend` as the global fiber backend. Returns `true` on
/// success, `false` if a backend was already installed. Set-once
/// semantics: a freshly-constructed runtime can install its preferred
/// backend, and subsequent calls are silent no-ops (so test harnesses
/// running in parallel don't fight over the slot).
pub fn install_fiber_backend(backend: Box<dyn FiberCfg>) -> bool {
    FIBER_BACKEND.set(backend).is_ok()
}

/// Look up the installed backend. Returns `None` if nothing has been
/// installed yet — the runtime symbol stubs in `zrtl` then panic with
/// a clear "no backend installed" message rather than silently
/// running broken code.
pub fn installed_fiber_backend() -> Option<&'static dyn FiberCfg> {
    FIBER_BACKEND.get().map(|b| b.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        for tag in [FIBER_STEP_YIELDED, FIBER_STEP_DONE, FIBER_STEP_ERRORED] {
            for payload in [0i64, 1, 42, -1, -7, i64::MAX >> FIBER_STEP_TAG_BITS] {
                let packed = pack_fiber_step(tag, payload);
                let (t, p) = unpack_fiber_step(packed);
                assert_eq!(t, tag, "tag round-trip failed at payload={payload}");
                assert_eq!(p, payload, "payload round-trip failed at tag={tag}");
            }
        }
    }
}
