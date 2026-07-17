//! Cooperative resume: when a handler resumes a continuation (`k(v)`) that
//! then reaches a real suspension point — an `await` inside the `@effect`
//! fn — the caller's poll fn returns Pending and parks on a timer. The
//! resume runtime (`__zyntax_effect_resume`) must drive that timer and
//! re-poll rather than busy-spinning the poll fn to its LOOP_BUDGET panic.
//!
//! This is the runtime substrate for parking (async) handlers. It's tested
//! here at the host-ABI level with a synthetic poll fn because the codegen
//! that composes a resumable-effect SM with an async-await SM in one
//! function is a separate (larger) piece — until it lands, no ZynML program
//! reaches this arm, but the drive-on-Pending contract is exercised and
//! locked down here.

#![cfg(not(target_arch = "wasm32"))]

use zyntax_embed::__zyntax_effect_resume;
use zyntax_embed::host_futures::{register_future, schedule_timer, ParkedFuture};

/// A dummy parked SM whose poll fn is immediately Ready — stands in for
/// whatever future the driven timer resolves. `resolve_future` writes into
/// its result slot and polls once; returning non-zero marks it Ready.
extern "C" fn dummy_ready_poll(_sm: *mut u8) -> i64 {
    99
}

/// The resumed caller's poll fn: Pending (0) while a timer is still
/// pending — i.e. it "reached an await and parked" — then Ready (7) once
/// the timer has been drained. Mirrors a continuation that awaits once.
extern "C" fn caller_poll(_sm: *mut u8) -> i64 {
    if zyntax_embed::host_futures::has_pending_timers() {
        0
    } else {
        7
    }
}

// The Resume struct layout `__zyntax_effect_resume` reads (repr(C), the
// first three fields it touches: poll fn, SM ptr, result-slot offset,
// then next_state, refcount offset).
#[repr(C)]
struct ResumeAbi {
    poll_fn_ptr: *const u8,
    state_machine_ptr: *mut u8,
    result_slot_offset: i64,
    next_state: i64,
    refcount_offset: i64,
}

#[test]
fn resume_drives_pending_timer_instead_of_spinning() {
    // A future the timer resolves when it fires (deadline = now).
    let mut dummy_sm = [0i64; 4];
    let handle = register_future(ParkedFuture {
        poll_fn_ptr: dummy_ready_poll as *const u8,
        state_machine_ptr: dummy_sm.as_mut_ptr() as *mut u8,
        result_slot_offset: 8,
        next_state: 1,
        refcount_offset: 24,
        task_id: 0,
    });
    schedule_timer(handle, 0);

    // The resumed caller's SM buffer. result_slot_offset=8 → the resume
    // value lands at index 1; state slot is index 0.
    let mut caller_sm = [0i64; 4];
    let resume = ResumeAbi {
        poll_fn_ptr: caller_poll as *const u8,
        state_machine_ptr: caller_sm.as_mut_ptr() as *mut u8,
        result_slot_offset: 8,
        next_state: 1,
        refcount_offset: 24,
    };

    // First poll parks (timer pending → 0); the resume runtime drives the
    // timer (draining it) and re-polls → Ready(7). Without the cooperative
    // drive this would spin 100_000 iterations and panic.
    let out = __zyntax_effect_resume(&resume as *const ResumeAbi as *mut u8, 5);
    assert_eq!(
        out, 7,
        "resume must drive the pending timer and re-poll to Ready"
    );
    assert!(
        !zyntax_embed::host_futures::has_pending_timers(),
        "the driven timer must have been drained"
    );
}
