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
//! `!Send` `Fiber`), and the caller owns the lifetime. `fiber_free`
//! is the symmetric `Box::from_raw` drop (unmaps the stack + guard
//! page + clears the per-fiber error slot); the frontend emits it at
//! the scope exit of a fiber that isn't moved out. Fibers created in
//! conditional/inner scopes, or received via a call return, aren't
//! dropped yet (a later liveness-based pass) — they leak but never
//! double-free.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};

use krio_fiber::{Fiber, FiberStep};
use zyntax_compiler::fiber_backend::{
    self, FiberCfg, FIBER_STEP_DONE, FIBER_STEP_ERRORED, FIBER_STEP_YIELDED,
};
use zyntax_compiler::zrtl::FiberRepr;

/// `(variant_tag, payload)` stashed by [`Self::fiber_abort_with`].
/// The tag matches the prelude's `FiberError` enum (1 = Message,
/// 2 = Custom). Payload is a pointer for both — `*const u8` to a
/// `String` for Message, `*const u8` to a `DynamicBox` for Custom.
type AbortInfo = (i64, i64);

thread_local! {
    /// Payload latched by `fiber_abort_with` before yielding the
    /// active fiber. Promoted to the per-fiber error map by
    /// `encode_step` when it surfaces an `Errored` step.
    ///
    /// We can't unwind through the Cranelift / LLVM-generated frame
    /// above the abort call cleanly (DWARF info varies by platform
    /// / opt level), so the abort is implemented as a *deferred*
    /// one: stash payload, yield, then have the resume-side encoder
    /// swap `Yielded` for `Errored`. The fiber stays suspended at
    /// the abort point.
    static ABORT_PAYLOAD: Cell<Option<AbortInfo>> = const { Cell::new(None) };

    /// Scalar returned by `yield_u64` when the host resumes the active fiber.
    /// The yield runtime call must return `void` to generated HIR today, so the
    /// backend retains the value until `krio_fiber_take_input` consumes it.
    static RESUME_INPUT: Cell<Option<u64>> = const { Cell::new(None) };

    /// Per-fiber error slot. Populated by `encode_step` when it
    /// surfaces Errored, read+cleared by `fiber_take_error` from
    /// the caller's side after iteration ends. Keyed by the fiber
    /// handle's raw pointer cast to `usize` (the handle stays
    /// thread-local so this map is safely accessible without
    /// synchronisation).
    static ERROR_MAP: RefCell<HashMap<usize, AbortInfo>> = RefCell::new(HashMap::new());

    /// Task that owns fibers created right now. The cooperative executor
    /// stamps this (via `set_fiber_task_id`) before driving a task, so a
    /// fiber the task creates can be freed if that task is later cancelled
    /// — a cancelled task never reaches its normal scope-exit `FiberDrop`.
    /// 0 = no owning task (plain, non-async fibers).
    static FIBER_TASK_ID: Cell<i64> = const { Cell::new(0) };

    /// Live fibers grouped by owning task, for cancel-time cleanup. Added
    /// on `fiber_new` under the current `FIBER_TASK_ID`, removed on
    /// `fiber_free` (the normal scope-exit drop). `free_task_fibers` frees
    /// whatever a cancelled task left behind. A fiber that a task RETURNS
    /// only escapes on a completing (non-cancelled) path, so this never
    /// frees a live escapee.
    static TASK_FIBERS: RefCell<HashMap<i64, HashSet<usize>>> =
        RefCell::new(HashMap::new());
}

/// Stamp the task that owns fibers created from now on (see `TASK_FIBERS`).
pub fn set_fiber_task_id(id: i64) {
    FIBER_TASK_ID.with(|c| c.set(id));
}

/// Number of live fibers currently attributed to `task_id`. For tests /
/// diagnostics — after a task completes or is cancelled this should be 0.
pub fn task_fiber_count(task_id: i64) -> usize {
    TASK_FIBERS.with(|m| m.borrow().get(&task_id).map(|s| s.len()).unwrap_or(0))
}

/// Free every fiber still owned by `task_id`. Used when the task is
/// cancelled and won't run its own scope-exit `FiberDrop`s: unmaps each
/// fiber's stack + guard page and clears its per-fiber error slot.
pub fn free_task_fibers(task_id: i64) {
    let ptrs: Vec<usize> = TASK_FIBERS.with(|m| {
        m.borrow_mut()
            .remove(&task_id)
            .map(|s| s.into_iter().collect())
            .unwrap_or_default()
    });
    let backend = KrioFiberBackend;
    for ptr in ptrs {
        unsafe {
            backend.fiber_free(ptr as *mut FiberRepr);
        }
    }
}

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
/// the C-ABI surface returns.
///
/// On Errored, the `(variant, payload)` from `ABORT_PAYLOAD` (or
/// zeros for a non-abort panic) gets stashed in `ERROR_MAP` keyed
/// by the fiber handle so the caller's later `fiber_take_error`
/// can return a typed `FiberError`.
///
/// * `Yielded` — if `ABORT_PAYLOAD` is latched, the yield was
///   actually a `Fiber.abort(err)`; surface `Errored` and write
///   the variant/payload to `ERROR_MAP`. Otherwise the payload
///   is the u64 the body passed to `yield_u64`.
/// * `Done` — fiber returned normally; payload 0.
/// * `Errored` — krio-fiber-side panic catch fired (user-code
///   panic, not `abort_with`); payload 0; map gets a `(0, 0)`
///   placeholder so `f.error()` returns `None` (variant=0 is the
///   absent marker).
fn encode_step(step: FiberStep, fiber: &Fiber) -> i64 {
    let key = fiber as *const Fiber as usize;
    if let Some(info) = ABORT_PAYLOAD.with(|cell| cell.take()) {
        // Drain whatever yield value the body pushed (the yield
        // we used to suspend the body — not the payload of record).
        let _ = fiber.take_yield_u64();
        ERROR_MAP.with(|m| m.borrow_mut().insert(key, info));
        // The packed step's i64 payload carries the variant tag in
        // its low bits so callers that don't bother with
        // `take_error` still see *something* indicative; the
        // canonical retrieval is `take_error` though.
        return fiber_backend::pack_fiber_step(FIBER_STEP_ERRORED, info.0);
    }
    match step {
        FiberStep::Yielded => {
            let payload = fiber.take_yield_u64().unwrap_or(0) as i64;
            fiber_backend::pack_fiber_step(FIBER_STEP_YIELDED, payload)
        }
        FiberStep::Done => fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0),
        FiberStep::Errored => {
            // User-code panic, not abort_with — no payload to
            // record. Still mark the slot so a later
            // `fiber_take_error` doesn't return stale data from
            // an old fiber that reused the address.
            ERROR_MAP.with(|m| m.borrow_mut().insert(key, (0, 0)));
            fiber_backend::pack_fiber_step(FIBER_STEP_ERRORED, 0)
        }
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
        let ptr = Box::into_raw(boxed) as *mut FiberRepr;
        // Record the fiber under the task driving right now so a cancel of
        // that task can free it.
        let task = FIBER_TASK_ID.with(|c| c.get());
        TASK_FIBERS.with(|m| {
            m.borrow_mut().entry(task).or_default().insert(ptr as usize);
        });
        ptr
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
        // A finished fiber stays finished: resuming past exhaustion is an
        // ordinary thing for a consumer loop to do, and it means Done —
        // matching `next()` returning None forever, not a panic.
        if matches!(
            fiber.state(),
            krio_fiber::FiberState::Done | krio_fiber::FiberState::Errored
        ) {
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
        if matches!(
            fiber.state(),
            krio_fiber::FiberState::Done | krio_fiber::FiberState::Errored
        ) {
            return fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0);
        }
        let step = fiber.resume_with_u64(value as u64);
        encode_step(step, fiber)
    }

    fn fiber_yield(&self, value: i64) {
        // `yield_u64` is a free function on krio-fiber — it consults
        // the thread-local "currently active fiber" set by `resume`.
        // Its `Option<u64>` return is what the matching `resume_with`
        // delivered. Generated HIR currently models FiberYield as a void
        // instruction, so retain that return until the explicit receiving
        // primitive reads it.
        let input = krio_fiber::yield_u64(value as u64);
        RESUME_INPUT.with(|slot| slot.set(input));
    }

    fn fiber_take_input(&self) -> i64 {
        RESUME_INPUT.with(|slot| slot.take().map_or(0, |value| value as i64))
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

    unsafe fn fiber_free(&self, fiber: *mut FiberRepr) {
        if fiber.is_null() {
            return;
        }
        // Clear any per-fiber error latched under this handle before
        // the address is reclaimed — otherwise a future fiber reusing
        // the same address could inherit a stale error.
        let key = fiber as usize;
        ERROR_MAP.with(|m| {
            m.borrow_mut().remove(&key);
        });
        // Drop the fiber from its owning task's set so a later
        // `free_task_fibers` can't double-free it.
        TASK_FIBERS.with(|m| {
            for set in m.borrow_mut().values_mut() {
                set.remove(&key);
            }
        });
        // Reclaim ownership handed out by `fiber_new`'s `Box::into_raw`.
        // Dropping the box unmaps the fiber's stack + guard page.
        drop(Box::from_raw(fiber as *mut Fiber));
    }

    fn fiber_abort_with(&self, variant: i64, payload: i64) {
        // Latch the (variant, payload), then yield. The caller-
        // side resume (via `encode_step`) sees the latch, writes
        // it to the per-fiber `ERROR_MAP`, and surfaces an
        // `Errored` step. We can't unwind through Cranelift /
        // LLVM-emitted frames cleanly enough across platforms to
        // do a panic-based immediate abort, so the abort is
        // *deferred*: the fiber body suspends at the abort point.
        // The typical `while let Some(x) = f.next()` pattern then
        // ends because Errored decodes to None.
        ABORT_PAYLOAD.with(|cell| cell.set(Some((variant, payload))));
        let _ = krio_fiber::yield_u64(payload as u64);
    }

    unsafe fn fiber_take_error(&self, fiber: *mut FiberRepr) -> i64 {
        let key = fiber as usize;
        let info = ERROR_MAP.with(|m| m.borrow_mut().remove(&key));
        match info {
            // No entry, or the sentinel for "panicked but no
            // `abort_with` was called" — caller sees `None`.
            None | Some((0, _)) => 0,
            Some((variant, payload)) => {
                // Packed shape: bit 0 = present, bits 1-2 = variant,
                // bits 3-63 = payload. The variant fits in 2 bits
                // (we only ever use 1 or 2). The payload's high
                // bits get sign-extended on the SSA-side shr.
                let present = 1i64;
                let v_bits = (variant & 0b11) << 1;
                let p_bits = payload << 3;
                present | v_bits | p_bits
            }
        }
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

    extern "C" fn receives_resume_value() {
        unsafe { zrtl::krio_fiber_yield(7) };
        let input = unsafe { zrtl::krio_fiber_take_input() };
        unsafe { zrtl::krio_fiber_yield(input) };
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

    #[test]
    fn bidirectional_resume_value_is_visible_inside_the_fiber() {
        ensure_installed();
        unsafe {
            let f = zrtl::krio_fiber_new(receives_resume_value as *mut u8, 0);
            let (first_tag, first_payload) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            assert_eq!((first_tag, first_payload), (FIBER_STEP_YIELDED, 7));
            let (second_tag, second_payload) =
                unpack_fiber_step(zrtl::krio_fiber_resume_with(f, 99));
            assert_eq!((second_tag, second_payload), (FIBER_STEP_YIELDED, 99));
            zrtl::krio_fiber_free(f);
        }
    }
}
