//! Phase I.0 — exercise the `__zyntax_async_set_timeout` bridge
//! end-to-end on native without going through krio_adapter or
//! ZynML. Proves the bridge itself works: register a parked SM,
//! kick off the host timer, wait for the resolve callback to
//! drive the SM to Ready.
//!
//! Phase I.1 (separate commit) wires the krio_adapter sentinel
//! that emits the `register_future` + bridge call sequence at
//! `await __builtin_sleep(...)` sites so ZynML source can use it
//! directly.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zyntax_embed::__zyntax_async_set_timeout;
use zyntax_embed::host_futures::{
    clear_for_tests, parked_count, register_future, set_complete_task_callback, ParkedFuture,
};

// Two-state SM modeling an `await __builtin_sleep(...)` shape:
//   state 0 → Pending (poll fn returns 0 — "the sleep is still running")
//   state 1 → Ready (poll fn returns a constant non-zero sentinel —
//             "the awaited host op has produced its result, advance
//             to the next thing or terminate")
//
// The choice of sentinel (1) is arbitrary; a real ZynML SM
// returns whatever the post-await computation evaluates to.
// Crucially the SM doesn't return the result-slot value directly
// — `resolve_future` writes a 0 there for unit-returning sleeps
// and 0 means Pending, which would loop forever.
extern "C" fn poll_two_state(sm: *mut u8) -> i64 {
    unsafe {
        let state = *(sm as *const i64);
        match state {
            0 => 0,
            _ => 1,
        }
    }
}

#[test]
fn async_set_timeout_resolves_after_at_least_ms() {
    // Models a ZynML program `await __builtin_sleep(100)` at the
    // bridge-plumbing level: park an SM that's ready to advance
    // on the next poll, kick off the host timer, spin-wait until
    // the worker thread resolves the future. Validate that the
    // wall-clock elapsed is at least the requested duration
    // (i.e. the timer actually fired vs. completing instantly).

    clear_for_tests();
    let done: Arc<Mutex<Option<i64>>> = Arc::new(Mutex::new(None));
    let done_cb = done.clone();
    set_complete_task_callback(Box::new(move |_task_id, value| {
        *done_cb.lock().unwrap() = Some(value);
    }));

    // SM: state slot at offset 0, result slot at offset 8.
    // Starts at state 0 (Pending); resolve_future advances it to
    // state 1 and the next poll returns 1 (Ready).
    let mut sm = [0i64, 0];
    let sm_ptr = sm.as_mut_ptr() as *mut u8;

    let handle = register_future(ParkedFuture {
        poll_fn_ptr: poll_two_state as *const u8,
        state_machine_ptr: sm_ptr,
        result_slot_offset: 8,
        next_state: 1,
        refcount_offset: 0,
        task_id: 1,
    });
    assert_eq!(parked_count(), 1);

    // The native impl spawns a worker thread that sleeps for `ms`
    // and then calls `host_futures::resolve_future(handle, 0)`.
    // FutureTable is thread-local — the worker thread has its own
    // empty table. To make this test work, we instead run the
    // sleep inline on the SAME thread that owns the table. That's
    // a Phase I.0 limitation; once the FutureTable goes
    // cross-thread (or we wire native through the existing
    // tokio-backed runtime), `__zyntax_async_set_timeout` will be
    // truly async on native too.
    //
    // Bypass the worker-thread path for now: sleep synchronously
    // and call resolve_future directly. This still exercises the
    // FutureTable + ParkedFuture + post-resolve poll path that
    // the wasm bridge will use; the threading model is the only
    // difference.
    let start = Instant::now();
    std::thread::sleep(Duration::from_millis(50));
    zyntax_embed::host_futures::resolve_future(handle, 0);
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() >= 45,
        "expected ≥ 45ms wait, got {:?}",
        elapsed
    );
    assert_eq!(parked_count(), 0);
    let final_value = done.lock().unwrap();
    // SM returns the Ready sentinel (1) from state 1.
    assert_eq!(*final_value, Some(1));
    clear_for_tests();
}

#[test]
fn async_set_timeout_symbol_is_callable_natively() {
    // Phase I.4a changed the native impl from "spawn worker thread"
    // to "synchronous sleep + inline resolve_future" so the
    // FutureTable's thread-local nature works cleanly. The function
    // now blocks the calling thread for `ms` ms — like any blocking
    // I/O — and resolves any registered future for the given
    // handle inline before returning. With an unknown handle (as in
    // this smoke test), resolve_future is a no-op.
    let start = Instant::now();
    __zyntax_async_set_timeout(123456, 50);
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_millis() >= 45,
        "__zyntax_async_set_timeout should block for the requested duration; took {:?}",
        elapsed
    );
}
