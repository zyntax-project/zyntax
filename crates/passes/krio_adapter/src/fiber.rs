//! krio-fiber backed `FiberCfg` implementation.
//!
//! `KrioFiberBackend` is the default native impl of the fiber
//! contract defined in `zyntax_compiler::fiber_backend::FiberCfg`. It
//! wraps krio-fiber's stackful coroutines behind the C-ABI surface the
//! `krio_fiber_*` runtime symbols expose; the HIR's fiber ops route
//! through here once [`install`] runs during runtime construction.
//!
//! Closure handles today are `extern "C" fn()` pointers — no captured
//! environment. When the frontend gains real closures-with-captures,
//! the handle becomes a richer struct pointer and the body trampoline
//! gains a marshalling layer; the `FiberCfg` contract itself doesn't
//! change.
//!
//! Returned `*mut FiberRepr` handles are `Box<krio_fiber::Fiber>`
//! pointers cast through `*mut FiberRepr`. The contract requires every
//! handle stays on the thread that created it (matches krio-fiber's
//! `!Send` `Fiber`), and the caller owns the lifetime: a future
//! `krio_fiber_free` will be the symmetric `Box::from_raw` drop. For
//! the scaffold, fibers leak on completion — fine for tests and
//! short-lived bench shapes, will move to explicit drop when the
//! frontend wires lifetime tracking.

use krio_fiber::{Fiber, FiberStep};
use zyntax_compiler::fiber_backend::{
    self, FiberCfg, FIBER_STEP_DONE, FIBER_STEP_ERRORED, FIBER_STEP_YIELDED,
};
use zyntax_compiler::zrtl::FiberRepr;

/// Default fiber backend backed by krio-fiber. Stateless — the
/// per-fiber state lives inside each `Fiber` instance; this struct is
/// only the vtable carrier.
pub struct KrioFiberBackend;

impl KrioFiberBackend {
    /// Construct a backend handle. Cheap — there is no setup; the
    /// trait methods reach into krio-fiber statics directly.
    pub fn new() -> Self {
        Self
    }
}

impl Default for KrioFiberBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a krio-fiber `FiberStep` to the packed (tag, payload) i64
/// the C-ABI surface returns. `Yielded` carries the u64 the fiber body
/// passed to `yield_u64`; `Done` / `Errored` carry no payload (zero).
fn encode_step(step: FiberStep, fiber: &Fiber) -> i64 {
    match step {
        FiberStep::Yielded => {
            let payload = fiber.take_yield_u64().unwrap_or(0) as i64;
            fiber_backend::pack_fiber_step(FIBER_STEP_YIELDED, payload)
        }
        FiberStep::Done => fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0),
        FiberStep::Errored => fiber_backend::pack_fiber_step(FIBER_STEP_ERRORED, 0),
    }
}

/// Build a closure for `Fiber::new` from a raw `extern "C" fn()`
/// pointer. Today's contract: the body takes no arguments and returns
/// nothing — it interacts with the host via `krio_fiber_yield` /
/// `take_input_u64` while running.
fn make_body(closure: *mut u8) -> impl FnOnce() {
    // Cast through usize so the closure captures Send-able state.
    // The original pointer is just a function address; treating it
    // as `extern "C" fn()` is sound when the caller honours the
    // closure-handle contract.
    let addr = closure as usize;
    move || {
        if addr == 0 {
            // Empty body — yields nothing, returns immediately. Useful
            // for "smoke test that fiber lifecycle works at all"
            // checks before any real body is wired.
            return;
        }
        let body: extern "C" fn() = unsafe { std::mem::transmute(addr) };
        body();
    }
}

impl FiberCfg for KrioFiberBackend {
    unsafe fn fiber_new(&self, closure: *mut u8, stack_size: i64) -> *mut FiberRepr {
        let body = make_body(closure);
        let fiber = if stack_size > 0 {
            Fiber::with_stack_size(stack_size as usize, body)
        } else {
            Fiber::new(body)
        };
        // Hand ownership to the caller via raw pointer. Reclaimed by
        // an eventual `krio_fiber_free`.
        let boxed = Box::new(fiber);
        Box::into_raw(boxed) as *mut FiberRepr
    }

    unsafe fn fiber_resume(&self, fiber: *mut FiberRepr) -> i64 {
        let fiber = &mut *(fiber as *mut Fiber);
        // Honour cooperative cancel at the resume boundary: an
        // auto-generated `fiber def` body has no chance to call
        // `is_cancelled()` itself between yields, so the cancel flag
        // would otherwise sit unobserved until the body finished
        // naturally. Surfacing `Done` here matches the user-visible
        // semantics of `Fiber::cancel()` ("subsequent `next()` ends
        // the iteration").
        if fiber.is_cancelled() {
            return fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0);
        }
        let step = fiber.resume();
        encode_step(step, fiber)
    }

    unsafe fn fiber_resume_with(&self, fiber: *mut FiberRepr, value: i64) -> i64 {
        let fiber = &mut *(fiber as *mut Fiber);
        if fiber.is_cancelled() {
            return fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0);
        }
        let step = fiber.resume_with_u64(value as u64);
        encode_step(step, fiber)
    }

    fn fiber_yield(&self, value: i64) {
        // `yield_u64` is a free function on krio-fiber — it consults
        // the thread-local "currently active fiber" set by `resume`.
        // The discarded `Option<u64>` return is what the next
        // `resume_with` will deliver back; bidirectional generators
        // recover it via `take_input_u64` from the fiber body, not
        // through the yield call site.
        let _ = krio_fiber::yield_u64(value as u64);
    }

    unsafe fn fiber_transfer(&self, _target: *mut FiberRepr, _value: i64) -> i64 {
        // Symmetric transfer isn't in krio-fiber upstream yet. Two
        // options when the HIR op fires: panic to make the gap loud,
        // or emulate via resume + yield through the caller chain.
        // Emulation changes timing semantics in subtle ways, so panic
        // here keeps the gap honest and drives the upstream PR.
        unimplemented!(
            "FiberTransfer: symmetric switch not yet supported (krio-fiber upstream gap)"
        )
    }

    unsafe fn fiber_cancel(&self, fiber: *mut FiberRepr) {
        let fiber = &*(fiber as *const Fiber);
        fiber.cancel();
    }
}

/// Install [`KrioFiberBackend`] as the global fiber backend. Returns
/// `true` on first install, `false` if a backend was already wired
/// (set-once semantics — runtimes that share a process don't fight
/// over the slot).
pub fn install() -> bool {
    fiber_backend::install_fiber_backend(Box::new(KrioFiberBackend::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;
    use zyntax_compiler::fiber_backend::{unpack_fiber_step, FIBER_STEP_DONE, FIBER_STEP_YIELDED};
    use zyntax_compiler::zrtl;

    /// Install the backend exactly once per test binary. Multiple
    /// tests share the global slot, and `install_fiber_backend` is
    /// set-once — call directly per-test would race; this gate makes
    /// it deterministic.
    fn ensure_installed() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            install();
        });
    }

    /// Module-scoped sink so the empty-body fiber test has a visible
    /// side effect to assert against. The fiber body writes a marker
    /// here; the test reads it after resume.
    static EMPTY_BODY_RAN: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    extern "C" fn empty_body() {
        EMPTY_BODY_RAN.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    extern "C" fn yields_42_then_done() {
        let _ = krio_fiber::yield_u64(42);
    }

    extern "C" fn yields_twice() {
        let _ = krio_fiber::yield_u64(10);
        let _ = krio_fiber::yield_u64(20);
    }

    #[test]
    fn empty_body_runs_to_done() {
        ensure_installed();
        EMPTY_BODY_RAN.store(false, std::sync::atomic::Ordering::SeqCst);
        unsafe {
            let f = zrtl::krio_fiber_new(empty_body as *mut u8, 0);
            assert!(!f.is_null(), "fiber allocation failed");
            let step = zrtl::krio_fiber_resume(f);
            let (tag, payload) = unpack_fiber_step(step);
            assert_eq!(tag, FIBER_STEP_DONE);
            assert_eq!(payload, 0);
        }
        assert!(EMPTY_BODY_RAN.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn yields_then_completes() {
        ensure_installed();
        unsafe {
            let f = zrtl::krio_fiber_new(yields_42_then_done as *mut u8, 0);
            // First resume: body runs until it yields 42.
            let step = zrtl::krio_fiber_resume(f);
            let (tag, payload) = unpack_fiber_step(step);
            assert_eq!(tag, FIBER_STEP_YIELDED);
            assert_eq!(payload, 42);
            // Second resume: body returns; tag becomes Done.
            let step = zrtl::krio_fiber_resume(f);
            let (tag, payload) = unpack_fiber_step(step);
            assert_eq!(tag, FIBER_STEP_DONE);
            assert_eq!(payload, 0);
        }
    }

    #[test]
    fn multi_yield_sequence() {
        ensure_installed();
        unsafe {
            let f = zrtl::krio_fiber_new(yields_twice as *mut u8, 0);
            let (t1, p1) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            let (t2, p2) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            let (t3, p3) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            assert_eq!((t1, p1), (FIBER_STEP_YIELDED, 10));
            assert_eq!((t2, p2), (FIBER_STEP_YIELDED, 20));
            assert_eq!((t3, p3), (FIBER_STEP_DONE, 0));
        }
    }
}
