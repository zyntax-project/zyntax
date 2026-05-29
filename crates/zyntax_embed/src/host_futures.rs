//! Cooperative-async future table for the browser runtime.
//!
//! ## Why this exists
//!
//! On native, krio_adapter-produced state machines drive themselves:
//! `__zyntax_effect_resume` spin-polls until Ready, the os has other
//! threads doing I/O, and everything makes progress. In a browser the
//! wasm module is the only thread that exists — any spin-poll
//! freezes the page until the loop reaches Ready. A ZynML program
//! that does `await fetch(url)` can't make progress because the JS
//! event loop never gets a chance to run the fetch callback that
//! would deliver the result.
//!
//! This module is the parking layer that breaks the spin-poll. When
//! a state machine reaches an `await __builtin_host_*` site, the
//! krio_adapter-emitted code calls [`register_future`] to stash the
//! SM's resume info in a thread-local table, calls the host bridge
//! (which kicks off a JS Promise and returns immediately), and
//! returns `0` (Pending) from the poll fn. Control flows back to the
//! wasm entry point, which returns to JS — the event loop now gets
//! to run, the JS Promise settles, and the host-side shim calls
//! [`resolve_future`] back into wasm to advance the SM by one step.
//!
//! Per-poll discipline: every call to [`resolve_future`] invokes the
//! parked SM's poll fn **exactly once**. If the SM reaches Ready,
//! the value bubbles up to the top-level task and resolves the JS
//! Promise. If the SM yields again (synchronously calls
//! `register_future` for the next host op during the poll), it gets
//! re-parked under the new handle. That single-poll-per-resolve
//! discipline is what enforces cooperative yielding — the wasm
//! thread never holds the event loop hostage.
//!
//! ## Native parity
//!
//! Native targets get the same surface — host bridges like
//! `__zyntax_async_set_timeout` simply spawn an std::thread that
//! does the work synchronously and calls [`resolve_future`] when
//! done. The native InterpRuntime continues to drive the SM
//! synchronously; `register_future` returns a handle but on native
//! the caller can choose to ignore parking and spin-poll instead
//! (the existing `__zyntax_effect_resume` behavior). This keeps the
//! ZynML stdlib body identical across targets — only the per-bridge
//! native implementation differs.
//!
//! ## Lifetime
//!
//! Parked SM pointers MUST outlive the future-table entry. The
//! host's top-level task scheduler (`crates/zyntax_wasm/src/lib.rs`
//! Phase H) holds the InterpRuntime + HIR module + SM allocation
//! alive in a `RuntimeHolder` keyed by `task_id`. When a future
//! resolves and the SM reaches Ready, the host invokes
//! [`complete_task`], which the scheduler uses to drop its
//! RuntimeHolder entry.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

/// Layout of a parked SM's resume info. Mirrors the field set of
/// the Resume<T> struct in `effect_runtime.rs:226–232`, plus the
/// `task_id` linkage to the top-level Promise resolver.
///
/// `result_slot_offset` and `refcount_offset` are byte offsets into
/// `state_machine_ptr` (pre-multiplied by 8 to match Cranelift's
/// `Store` instruction shape). `next_state` is the dispatcher
/// state-id the caller's Switch should route to on the next poll
/// — written as a full i64 to the state slot at offset 0 (writing
/// only the low u32 left garbage in the upper bytes on Windows; see
/// the long comment in `effect_runtime.rs:265–284`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ParkedFuture {
    pub poll_fn_ptr: *const u8,
    pub state_machine_ptr: *mut u8,
    pub result_slot_offset: i64,
    pub next_state: i64,
    pub refcount_offset: i64,
    /// The top-level task this future feeds into. When the SM reaches
    /// Ready, `complete_task(task_id, value)` notifies the JS Promise
    /// resolver registry.
    pub task_id: i64,
}

// SAFETY: ParkedFuture holds raw pointers but the FutureTable is
// thread-local — pointers never cross threads. Single-threaded
// wasm makes this trivially sound; on native the host-bridge thread
// that calls `resolve_future` does so by re-entering through the
// main thread's runtime, so the table itself stays single-threaded.
unsafe impl Send for ParkedFuture {}

/// Result of advancing a parked SM by one poll.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveOutcome {
    /// The SM yielded again during the poll. It must have called
    /// `register_future` synchronously for the next host op, so it's
    /// now parked under a fresh handle.
    ReParked,
    /// The SM reached Ready. The value is the program-level result.
    /// The caller (typically the wasm-bindgen export shim) is
    /// expected to forward this to `complete_task`.
    Ready(i64),
    /// The provided handle wasn't in the future table. Caused by a
    /// double-resolve, a stale callback after the SM unwound, or a
    /// handle from a different task that's already been cleaned up.
    UnknownHandle,
}

impl ResolveOutcome {
    /// Stable i32 encoding for wasm-bindgen exports.
    ///
    ///   0 = ReParked, 1 = Ready, 2 = UnknownHandle
    ///
    /// `Ready` callers extract the value via a side channel
    /// ([`complete_task`]); the wasm export only forwards the
    /// outcome code. JS-side code keys off this for the rare case
    /// where it wants to distinguish "still pending" from "done."
    pub fn as_i32(self) -> i32 {
        match self {
            ResolveOutcome::ReParked => 0,
            ResolveOutcome::Ready(_) => 1,
            ResolveOutcome::UnknownHandle => 2,
        }
    }
}

/// Signature for the wasm32-only poll dispatcher. Takes the closure
/// handle (i64 hash from `HirId::to_handle_hash`) and the SM ptr;
/// returns the poll fn's i64 result (0 = Pending, non-zero = Ready
/// value). The dispatcher implementation lives in `zyntax_wasm`
/// (which owns the InterpRuntime ↔ `ACTIVE_CLOSURE_FNS` mapping)
/// and is installed via [`set_wasm_poll_dispatcher`] at runtime
/// startup.
pub type WasmPollDispatcher = Box<dyn Fn(i64, *mut u8) -> i64>;

thread_local! {
    /// All currently-parked futures. Keyed by the i64 handle returned
    /// from [`register_future`]; cleared on [`resolve_future`] (one
    /// way or another — the entry is removed before the poll runs so
    /// a re-park inside the poll fn doesn't collide with itself).
    static FUTURE_TABLE: RefCell<HashMap<i64, ParkedFuture>> =
        RefCell::new(HashMap::new());

    /// Wasm32-only poll fn dispatcher. See [`WasmPollDispatcher`] and
    /// [`set_wasm_poll_dispatcher`]. Native targets ignore this slot
    /// entirely — they call the poll fn via raw fn-ptr transmute.
    static WASM_POLL_DISPATCH: RefCell<Option<WasmPollDispatcher>> =
        const { RefCell::new(None) };

    /// Monotonic handle counter. Starts at 1 so `0` can stay a
    /// "no handle" sentinel for the krio_adapter sentinel emitter.
    static NEXT_HANDLE: Cell<i64> = const { Cell::new(1) };

    /// Callback invoked when a future resolves and the parked SM
    /// reaches Ready. Set by the top-level scheduler (Phase H);
    /// `None` means "no scheduler attached, drop the Ready value on
    /// the floor" — which is the right behavior for the native unit
    /// tests that only exercise the FutureTable itself.
    static COMPLETE_TASK_CALLBACK: RefCell<Option<CompleteTaskFn>> =
        const { RefCell::new(None) };
}

/// Signature for the top-level task completion callback. Receives
/// the task_id (matched to the JS-side Promise resolver) and the
/// final value. The scheduler is expected to drop its
/// RuntimeHolder entry for the task_id before returning so the
/// InterpRuntime + SM allocation are freed.
pub type CompleteTaskFn = Box<dyn Fn(i64, i64)>;

/// Install the top-level task completion callback. Replaces any
/// previously-installed callback; intended to be called once at
/// runtime startup by the scheduler module.
pub fn set_complete_task_callback(cb: CompleteTaskFn) {
    COMPLETE_TASK_CALLBACK.with(|slot| {
        *slot.borrow_mut() = Some(cb);
    });
}

/// Install the wasm32 poll dispatcher. The dispatcher takes a
/// closure handle (i64) + SM ptr and returns the poll fn's i64
/// result. Wasm32 hosts (zyntax_wasm) install this at runtime
/// startup so [`resolve_future`] can drive parked SMs via the
/// indirect-call dispatcher rather than the unavailable raw fn-ptr
/// transmute. No-op on native (which uses the transmute path).
pub fn set_wasm_poll_dispatcher(dispatch: WasmPollDispatcher) {
    WASM_POLL_DISPATCH.with(|slot| {
        *slot.borrow_mut() = Some(dispatch);
    });
}

/// Clear the completion callback. Idempotent. Mostly useful for
/// test teardown so a panicked test on the same thread doesn't
/// leave the next test's table entries firing into stale state.
pub fn clear_complete_task_callback() {
    COMPLETE_TASK_CALLBACK.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

/// Notify the scheduler that the SM behind `task_id` has reached
/// Ready with `value`. Invoked by [`resolve_future`] when the
/// post-resume poll returns non-zero; intentionally exposed so
/// host bridges that complete synchronously (no parking step at
/// all) can take the same code path.
pub fn complete_task(task_id: i64, value: i64) {
    COMPLETE_TASK_CALLBACK.with(|slot| {
        if let Some(cb) = slot.borrow().as_ref() {
            cb(task_id, value);
        }
    });
}

/// Park `f` in the future table and return a fresh handle. The
/// handle is what the host-side bridge shim ultimately threads to
/// [`resolve_future`] when its JS Promise settles.
///
/// The handle is non-zero. Sequential calls produce increasing
/// values; the counter is per-thread and wraps after 2^63 calls
/// (which never happens in practice). Reuse of a handle within
/// the same wrap-around window can only happen after the entry
/// is removed via `resolve_future` / `reject_future`.
pub fn register_future(f: ParkedFuture) -> i64 {
    let handle = NEXT_HANDLE.with(|c| {
        let h = c.get();
        c.set(h.wrapping_add(1));
        h
    });
    FUTURE_TABLE.with(|t| {
        t.borrow_mut().insert(handle, f);
    });
    handle
}

/// Advance the SM parked under `handle` by one poll, with `value`
/// written into its result slot. Returns [`ResolveOutcome::Ready`]
/// with the SM's final value if the poll reaches Ready,
/// [`ResolveOutcome::ReParked`] if the SM yielded again (i.e.
/// synchronously called `register_future` for the next host op),
/// or [`ResolveOutcome::UnknownHandle`] if the handle is unknown.
///
/// On Ready, the parked future's `task_id` is forwarded to the
/// installed completion callback (if any). The caller doesn't need
/// to dispatch separately.
///
/// # Safety
///
/// The caller is responsible for ensuring the parked SM's pointer
/// is still valid — i.e. the SM allocation hasn't been freed by a
/// concurrent unwinding caller. The top-level scheduler enforces
/// this via the RuntimeHolder lifetime.
pub fn resolve_future(handle: i64, value: i64) -> ResolveOutcome {
    let parked = FUTURE_TABLE.with(|t| t.borrow_mut().remove(&handle));
    let Some(parked) = parked else {
        return ResolveOutcome::UnknownHandle;
    };
    // Write `value` into the SM's result slot, set the dispatcher
    // state to next_state, then poll once. Same dance as
    // __zyntax_effect_resume but without the loop — single poll
    // per resolve is the cooperative-yield invariant.
    unsafe {
        let result_slot = parked
            .state_machine_ptr
            .add(parked.result_slot_offset as usize) as *mut i64;
        *result_slot = value;
        *(parked.state_machine_ptr as *mut i64) = parked.next_state;
        let rc = poll_parked_sm(&parked);
        if rc != 0 {
            // SM reached Ready. Forward the value to the scheduler
            // and return it through the outcome so the wasm export
            // shim can also pass it back to JS if it wants.
            complete_task(parked.task_id, rc);
            ResolveOutcome::Ready(rc)
        } else {
            // SM yielded again. It must have synchronously called
            // `register_future` for the next host op; that new
            // handle is now in the table. Nothing else to do here.
            ResolveOutcome::ReParked
        }
    }
}

/// Drive the SM's poll fn once. The native and wasm32 paths differ
/// because wasm32 has no addressable function pointers — Phase I.2's
/// `CreateClosure` emits an i64 hash handle, not a real fn ptr, and
/// you can't `transmute` it to `extern "C" fn(...) -> i64`.
///
/// Native: raw transmute + direct call. Same shape as the existing
/// `__zyntax_effect_resume` spin-poll (effect_runtime.rs:312).
///
/// Wasm32: `poll_fn_ptr` is a closure handle. Route through the
/// JS-side indirect-call dispatcher (installed by
/// `zyntax_wasm::install_wasm_poll_dispatcher` at startup) which
/// looks the handle up in `ACTIVE_CLOSURE_FNS` and re-enters
/// `InterpRuntime::call_function`. Same re-entry path Phase I.3
/// uses for ZynML-level indirect calls — the InterpRuntime is held
/// in `ACTIVE_RUNTIME` as a `*mut` for exactly this reason, so the
/// recursive call doesn't double-borrow the thread-local.
///
/// # Safety
///
/// Caller must hold `parked.state_machine_ptr` live for the
/// duration of the call (the top-level scheduler's
/// `RUNTIME_HOLDER`).
unsafe fn poll_parked_sm(parked: &ParkedFuture) -> i64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let poll_fn: extern "C" fn(*mut u8) -> i64 = core::mem::transmute(parked.poll_fn_ptr);
        poll_fn(parked.state_machine_ptr)
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Treat `poll_fn_ptr` as an i64 closure handle (Phase I.2
        // emits `CreateClosure` which lowers to `HirId::to_handle_hash`
        // in the wasm backend / interp runtime; the raw bits live in
        // `poll_fn_ptr` because that's where the Resume struct stores
        // them and host_futures has been ABI-compatible with that all
        // along).
        let handle = parked.poll_fn_ptr as i64;
        WASM_POLL_DISPATCH.with(|slot| {
            if let Some(dispatch) = slot.borrow().as_ref() {
                dispatch(handle, parked.state_machine_ptr)
            } else {
                // No dispatcher installed: park stays dead. Logged at
                // the JS layer via console.warn after the
                // ResolveOutcome::ReParked is observed. Returning 0
                // (Pending) keeps the SM alive in case a late dispatch
                // install lands before the next resolve attempt.
                0
            }
        })
    }
}

/// Reject the future parked under `handle`. Today this is wired
/// through the same code path as a value resolve with a sentinel
/// `-1` value; richer error propagation (panic, exception, typed
/// error variant) is the follow-up when ZynML grows a `Result`-
/// returning bridge API. The `msg` parameter is accepted for
/// future compatibility but currently discarded — JS-side fetch
/// failures, etc., should also log via `console.error` so the
/// developer sees the reason.
pub fn reject_future(handle: i64, _msg: &str) -> ResolveOutcome {
    // Placeholder behavior: resolve with -1 so the program at
    // least makes progress instead of hanging. Replace with real
    // exception propagation when the language has a story for it.
    resolve_future(handle, -1)
}

/// Test-only: peek at the number of parked futures. Lets unit
/// tests assert "the table was cleaned up after the SM completed"
/// without exposing the internal structure.
#[doc(hidden)]
pub fn parked_count() -> usize {
    FUTURE_TABLE.with(|t| t.borrow().len())
}

/// Test-only: clear the future table. For test teardown so a
/// panicked earlier test doesn't poison the next one running on
/// the same thread.
#[doc(hidden)]
pub fn clear_for_tests() {
    FUTURE_TABLE.with(|t| t.borrow_mut().clear());
    NEXT_HANDLE.with(|c| c.set(1));
    clear_complete_task_callback();
}

// -- Externs invoked from krio_adapter-emitted SM code ----------
//
// These are the C ABI symbols the wasm-emitting backend (and
// Cranelift on native) emit calls to at the `await __builtin_host_*`
// sites. The SM passes its own offsets in; the runtime returns
// the handle.

/// C ABI wrapper for [`register_future`]. Receives the parked SM's
/// info inline so the emitted code doesn't have to construct a
/// `ParkedFuture` struct on the wasm side — every arg comes through
/// as an i64 (or pointer-sized) value compatible with the krio_adapter
/// emit shape.
#[no_mangle]
pub extern "C" fn __zyntax_register_future(
    poll_fn_ptr: *const u8,
    state_machine_ptr: *mut u8,
    result_slot_offset: i64,
    next_state: i64,
    refcount_offset: i64,
    task_id: i64,
) -> i64 {
    register_future(ParkedFuture {
        poll_fn_ptr,
        state_machine_ptr,
        result_slot_offset,
        next_state,
        refcount_offset,
        task_id,
    })
}

/// C ABI wrapper for [`resolve_future`]. Returns the i32 outcome
/// code so callers that re-enter through the C boundary (e.g. the
/// wasm-bindgen export, or a native test harness) can branch
/// without re-parsing the enum.
#[no_mangle]
pub extern "C" fn __zyntax_resolve_future(handle: i64, value: i64) -> i32 {
    resolve_future(handle, value).as_i32()
}

/// C ABI wrapper for [`reject_future`]. `msg_ptr` + `msg_len`
/// describe a UTF-8 string in linear memory (wasm) or any byte
/// buffer (native). The msg is accepted for forward compatibility;
/// today only its presence matters, not its content.
///
/// # Safety
///
/// `msg_ptr` + `msg_len` must describe a valid UTF-8 byte slice
/// owned by the caller for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn __zyntax_reject_future(
    handle: i64,
    msg_ptr: *const u8,
    msg_len: usize,
) -> i32 {
    let msg = if msg_ptr.is_null() || msg_len == 0 {
        ""
    } else {
        let slice = core::slice::from_raw_parts(msg_ptr, msg_len);
        core::str::from_utf8(slice).unwrap_or("<invalid utf-8>")
    };
    reject_future(handle, msg).as_i32()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // A hand-rolled SM that walks states 0 → 1 → 2. State slot at
    // offset 0; result slot at offset 8; total size 16 bytes.
    // poll_fn returns Pending (0) for states 0 and 1, Ready
    // (the loaded result value) for state 2.
    //
    // Layout:
    //   [0..8 ]  state  (i64)
    //   [8..16]  result (i64)
    extern "C" fn three_state_poll(sm: *mut u8) -> i64 {
        unsafe {
            let state = *(sm as *const i64);
            match state {
                0 => 0, // Pending
                1 => 0, // Pending
                2 => *((sm as *const u8).add(8) as *const i64),
                _ => i64::MIN, // bug if we get here
            }
        }
    }

    #[test]
    fn register_and_resolve_walks_to_ready() {
        clear_for_tests();
        // SM allocation owned by the test — pretend the scheduler
        // holds it alive.
        let sm: Arc<Mutex<[i64; 2]>> = Arc::new(Mutex::new([0, 0]));
        let sm_ptr = sm.lock().unwrap().as_mut_ptr() as *mut u8;

        // Park at state 1 with result-slot offset 8.
        let h0 = register_future(ParkedFuture {
            poll_fn_ptr: three_state_poll as *const u8,
            state_machine_ptr: sm_ptr,
            result_slot_offset: 8,
            next_state: 1,
            refcount_offset: 0,
            task_id: 99,
        });
        assert!(h0 >= 1);
        assert_eq!(parked_count(), 1);

        // First resolve: advances state 1 → poll returns Pending
        // → caller is supposed to have re-parked, but our test
        // poll fn doesn't (it just returns 0). So outcome is
        // ReParked but the table is empty afterward.
        let r = resolve_future(h0, 11);
        assert_eq!(r, ResolveOutcome::ReParked);
        assert_eq!(parked_count(), 0);

        // Park again at state 2, set result to 42 via the resolve
        // (next_state=2 means "advance to state 2 then poll, which
        // reads result from slot 8").
        let h1 = register_future(ParkedFuture {
            poll_fn_ptr: three_state_poll as *const u8,
            state_machine_ptr: sm_ptr,
            result_slot_offset: 8,
            next_state: 2,
            refcount_offset: 0,
            task_id: 99,
        });
        let r = resolve_future(h1, 42);
        assert_eq!(r, ResolveOutcome::Ready(42));
        assert_eq!(parked_count(), 0);
    }

    #[test]
    fn unknown_handle_is_not_an_error() {
        clear_for_tests();
        let r = resolve_future(9999, 0);
        assert_eq!(r, ResolveOutcome::UnknownHandle);
    }

    #[test]
    fn complete_task_callback_fires_on_ready() {
        clear_for_tests();
        let received: Arc<Mutex<Vec<(i64, i64)>>> = Arc::new(Mutex::new(Vec::new()));
        let received_cb = received.clone();
        set_complete_task_callback(Box::new(move |task_id, value| {
            received_cb.lock().unwrap().push((task_id, value));
        }));

        let mut sm = [2i64, 0];
        let sm_ptr = sm.as_mut_ptr() as *mut u8;
        let h = register_future(ParkedFuture {
            poll_fn_ptr: three_state_poll as *const u8,
            state_machine_ptr: sm_ptr,
            result_slot_offset: 8,
            next_state: 2,
            refcount_offset: 0,
            task_id: 7,
        });
        let r = resolve_future(h, 123);
        assert_eq!(r, ResolveOutcome::Ready(123));
        let captured = received.lock().unwrap().clone();
        assert_eq!(captured, vec![(7, 123)]);
        clear_for_tests();
    }

    #[test]
    fn reject_resolves_with_sentinel() {
        clear_for_tests();
        let mut sm = [2i64, 0];
        let sm_ptr = sm.as_mut_ptr() as *mut u8;
        let h = register_future(ParkedFuture {
            poll_fn_ptr: three_state_poll as *const u8,
            state_machine_ptr: sm_ptr,
            result_slot_offset: 8,
            next_state: 2,
            refcount_offset: 0,
            task_id: 0,
        });
        let r = reject_future(h, "fetch failed");
        // -1 sentinel — temporary; replace with Result-shaped
        // propagation when ZynML grows the language story.
        assert_eq!(r, ResolveOutcome::Ready(-1));
    }
}
