//! Integration tests for the cooperative-async FutureTable.
//!
//! Phase G — exercises `register_future` / `resolve_future` /
//! `reject_future` against hand-rolled state machines without any
//! krio_adapter or wasm in the loop. These tests prove the parking
//! ABI works in isolation; the SM-emitting glue lands in Phase I.

use std::sync::{Arc, Mutex};

use zyntax_embed::host_futures::{
    clear_for_tests, parked_count, register_future, reject_future, resolve_future,
    set_complete_task_callback, ParkedFuture, ResolveOutcome,
};

// SM layout used by these tests:
//
//   [0..8 ]  state    (i64) — dispatcher reads, krio_adapter writes
//   [8..16]  result   (i64) — caller writes via resolve_future
//
// poll_fn semantics:
//   state 0 → returns 0 (Pending, "waiting on first host op")
//   state 1 → returns 0 (Pending, "waiting on second host op")
//   state 2 → returns the result-slot value (Ready)
extern "C" fn three_state_poll(sm: *mut u8) -> i64 {
    unsafe {
        let state = *(sm as *const i64);
        match state {
            0 | 1 => 0,
            2 => *((sm as *const u8).add(8) as *const i64),
            _ => i64::MIN,
        }
    }
}

#[test]
fn end_to_end_two_step_async_chain() {
    // Walks state 0 → 1 (via first resolve) → 2 (via second resolve)
    // → Ready(42). Models a ZynML program like:
    //
    //   async def main(): i64 {
    //       await first_thing()
    //       return await second_thing()
    //   }
    //
    // The compiler emits an SM with three states; the JS host
    // bridges resolve each await site in turn.
    clear_for_tests();

    let sm: Arc<Mutex<[i64; 2]>> = Arc::new(Mutex::new([0, 0]));
    let sm_ptr = sm.lock().unwrap().as_mut_ptr() as *mut u8;

    // Park at state 1 to simulate the first await's continuation.
    let h0 = register_future(ParkedFuture {
        poll_fn_ptr: three_state_poll as *const u8,
        state_machine_ptr: sm_ptr,
        result_slot_offset: 8,
        next_state: 1,
        refcount_offset: 0,
        task_id: 1,
    });
    assert_eq!(parked_count(), 1);

    // First resolve: advance to state 1, poll returns Pending. In a
    // real flow the SM's poll fn would have called register_future
    // synchronously for the next host op; our hand-rolled SM
    // doesn't, so we re-park manually below.
    let r = resolve_future(h0, 0);
    assert_eq!(r, ResolveOutcome::ReParked);
    assert_eq!(parked_count(), 0);

    // Park again at state 2 (the SM's final state).
    let h1 = register_future(ParkedFuture {
        poll_fn_ptr: three_state_poll as *const u8,
        state_machine_ptr: sm_ptr,
        result_slot_offset: 8,
        next_state: 2,
        refcount_offset: 0,
        task_id: 1,
    });

    // Second resolve: writes 42 to slot, advances state to 2, poll
    // returns 42 (Ready). FutureTable cleans up.
    let r = resolve_future(h1, 42);
    assert_eq!(r, ResolveOutcome::Ready(42));
    assert_eq!(parked_count(), 0);
    clear_for_tests();
}

#[test]
fn unknown_handle_does_not_panic() {
    // JS-side could double-resolve a future or fire a callback
    // after the SM has already unwound. The table treats it as
    // "shrug, ignore" — no panic, no UB.
    clear_for_tests();
    let r = resolve_future(123_456_789, 1);
    assert_eq!(r, ResolveOutcome::UnknownHandle);
}

#[test]
fn complete_task_callback_receives_value() {
    // The top-level scheduler (Phase H) installs a callback on
    // startup so it knows when a future-driven SM reaches Ready —
    // that's the signal to resolve the JS Promise.
    clear_for_tests();
    let captured: Arc<Mutex<Vec<(i64, i64)>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_cb = captured.clone();
    set_complete_task_callback(Box::new(move |task_id, value| {
        captured_cb.lock().unwrap().push((task_id, value));
    }));

    let mut sm = [2i64, 0];
    let sm_ptr = sm.as_mut_ptr() as *mut u8;
    let h = register_future(ParkedFuture {
        poll_fn_ptr: three_state_poll as *const u8,
        state_machine_ptr: sm_ptr,
        result_slot_offset: 8,
        next_state: 2,
        refcount_offset: 0,
        task_id: 42,
    });
    let r = resolve_future(h, 1337);
    assert_eq!(r, ResolveOutcome::Ready(1337));

    let log = captured.lock().unwrap().clone();
    assert_eq!(log, vec![(42, 1337)]);
    clear_for_tests();
}

#[test]
fn reject_resolves_with_sentinel_for_now() {
    // Until ZynML grows a Result-returning bridge API, rejection
    // resolves the future with -1. Programs can detect "fetch
    // failed" by checking for the sentinel; eventually this
    // becomes a typed exception.
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
    let r = reject_future(h, "simulated fetch failure");
    assert_eq!(r, ResolveOutcome::Ready(-1));
    clear_for_tests();
}

#[test]
fn handles_are_unique_across_concurrent_parks() {
    // Two awaits in the same poll round (e.g. spawn two host ops
    // before yielding) must produce distinct handles so each can
    // be resolved independently.
    clear_for_tests();
    let mut sm = [0i64, 0];
    let sm_ptr = sm.as_mut_ptr() as *mut u8;

    let a = register_future(ParkedFuture {
        poll_fn_ptr: three_state_poll as *const u8,
        state_machine_ptr: sm_ptr,
        result_slot_offset: 8,
        next_state: 1,
        refcount_offset: 0,
        task_id: 0,
    });
    let b = register_future(ParkedFuture {
        poll_fn_ptr: three_state_poll as *const u8,
        state_machine_ptr: sm_ptr,
        result_slot_offset: 8,
        next_state: 1,
        refcount_offset: 0,
        task_id: 0,
    });
    assert_ne!(a, b);
    assert_eq!(parked_count(), 2);

    // Resolving in either order works.
    assert_eq!(resolve_future(b, 0), ResolveOutcome::ReParked);
    assert_eq!(parked_count(), 1);
    assert_eq!(resolve_future(a, 0), ResolveOutcome::ReParked);
    assert_eq!(parked_count(), 0);
    clear_for_tests();
}
