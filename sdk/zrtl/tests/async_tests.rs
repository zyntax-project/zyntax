//! Tests for ZRTL async support
//!
//! These tests verify the async primitives work correctly.

use std::future::Future;
use std::pin::Pin;
use std::task::Poll;
use std::time::Duration;
use zrtl::async_support::*;

// ============================================================================
// PollResult Tests
// ============================================================================

#[test]
fn test_poll_result_pending_abi() {
    let result = PollResult::Pending;
    assert_eq!(result.to_abi(), 0);
    assert!(result.is_pending());
    assert!(!result.is_ready());
    assert!(!result.is_failed());
}

#[test]
fn test_poll_result_ready_abi() {
    let result = PollResult::Ready(42);
    assert_eq!(result.to_abi(), 42);
    assert!(result.is_ready());
    assert!(!result.is_pending());
    assert!(!result.is_failed());
}

#[test]
fn test_poll_result_failed_abi() {
    let result = PollResult::Failed(-1);
    assert_eq!(result.to_abi(), -1);
    assert!(result.is_failed());
    assert!(!result.is_pending());
    assert!(!result.is_ready());
}

#[test]
fn test_poll_result_roundtrip() {
    assert_eq!(PollResult::from_abi(0), PollResult::Pending);
    assert_eq!(PollResult::from_abi(100), PollResult::Ready(100));
    assert_eq!(PollResult::from_abi(-5), PollResult::Failed(-5));
}

// ============================================================================
// StateMachineHeader Tests
// ============================================================================

#[test]
fn test_state_machine_header_new() {
    let header = StateMachineHeader::new();
    assert_eq!(header.state, 0);
    assert_eq!(header.async_state(), AsyncState::Initial);
    assert!(!header.async_state().is_finished());
}

#[test]
fn test_state_machine_header_advance() {
    let mut header = StateMachineHeader::new();
    header.advance();
    assert_eq!(header.state, 1);
    header.advance();
    assert_eq!(header.state, 2);
}

#[test]
fn test_state_machine_header_completed() {
    let mut header = StateMachineHeader::new();
    header.set_completed();
    assert!(header.async_state().is_finished());
    assert_eq!(header.async_state(), AsyncState::Completed);
}

#[test]
fn test_state_machine_header_failed() {
    let mut header = StateMachineHeader::new();
    header.set_failed();
    assert!(header.async_state().is_finished());
    assert_eq!(header.async_state(), AsyncState::Failed);
}

// ============================================================================
// Waker Tests
// ============================================================================

#[test]
fn test_noop_waker_does_not_panic() {
    let waker = noop_waker();
    waker.wake_by_ref();
    let cloned = waker.clone();
    cloned.wake();
}

#[test]
fn test_noop_context_can_be_used() {
    let mut cx = noop_context();

    // Create a simple future that completes immediately
    let mut future = std::future::ready(42);
    let pinned = Pin::new(&mut future);

    let result = pinned.poll(&mut cx);
    assert!(matches!(result, Poll::Ready(42)));
}

// ============================================================================
// YieldOnce Tests
// ============================================================================

#[test]
fn test_yield_once_behavior() {
    let mut future = yield_once();
    let mut cx = noop_context();

    // First poll should be pending
    let pinned = Pin::new(&mut future);
    assert!(pinned.poll(&mut cx).is_pending());

    // Second poll should be ready
    let pinned = Pin::new(&mut future);
    assert!(pinned.poll(&mut cx).is_ready());
}

#[test]
fn test_yield_once_multiple_creates() {
    let mut cx = noop_context();

    // Each new YieldOnce should start pending
    for _ in 0..5 {
        let mut future = yield_once();
        let pinned = Pin::new(&mut future);
        assert!(pinned.poll(&mut cx).is_pending());
    }
}

// ============================================================================
// Timer Tests
// ============================================================================

#[test]
fn test_timer_immediate() {
    let mut timer = Timer::after(Duration::ZERO);
    let mut cx = noop_context();

    // Zero duration should be ready immediately
    let pinned = Pin::new(&mut timer);
    assert!(pinned.poll(&mut cx).is_ready());
}

#[test]
fn test_timer_short_duration() {
    let mut timer = Timer::after(Duration::from_millis(1));
    let mut cx = noop_context();

    // First poll may be pending
    let pinned = Pin::new(&mut timer);
    let first = pinned.poll(&mut cx);

    if first.is_pending() {
        // Wait and poll again
        std::thread::sleep(Duration::from_millis(5));
        let pinned = Pin::new(&mut timer);
        assert!(pinned.poll(&mut cx).is_ready());
    }
}

// ============================================================================
// FutureAdapter Tests
// ============================================================================

#[test]
fn test_future_adapter_immediate() {
    let future = std::future::ready(42i32);
    let mut adapter = FutureAdapter::new(future);

    let result = adapter.poll();
    assert!(matches!(result, Poll::Ready(&42)));
}

#[test]
fn test_future_adapter_yielding() {
    // Create a future that yields once then completes
    async fn yielding_future() -> i32 {
        yield_once().await;
        100
    }

    let mut adapter = FutureAdapter::new(yielding_future());

    // First poll should be pending
    assert!(adapter.poll().is_pending());

    // Second poll should be ready
    let result = adapter.poll();
    assert!(matches!(result, Poll::Ready(&100)));
}

#[test]
fn test_future_adapter_take_result() {
    async fn compute() -> i32 {
        42
    }

    let mut adapter = FutureAdapter::new(compute());
    let _ = adapter.poll(); // Complete the future

    let result = adapter.take_result();
    assert_eq!(result, Some(42));
}

// ============================================================================
// ZrtlPromise Tests
// ============================================================================

// Mock poll function for testing
extern "C" fn mock_pending_poll(_state: *mut u8) -> i64 {
    0 // Pending
}

extern "C" fn mock_ready_poll(_state: *mut u8) -> i64 {
    42 // Ready with value 42
}

extern "C" fn mock_failed_poll(_state: *mut u8) -> i64 {
    -1 // Failed with code -1
}

#[repr(C)]
struct MockStateMachine {
    counter: i32,
    target: i32,
}

extern "C" fn counting_poll(state: *mut u8) -> i64 {
    unsafe {
        let sm = &mut *(state as *mut MockStateMachine);
        if sm.counter >= sm.target {
            sm.counter as i64 // Ready
        } else {
            sm.counter += 1;
            0 // Pending
        }
    }
}

#[test]
fn test_promise_poll_pending() {
    let mut state = [0u8; 16];
    let mut promise = unsafe { ZrtlPromise::new(state.as_mut_ptr(), mock_pending_poll) };

    let result = unsafe { promise.poll() };
    assert_eq!(result, PollResult::Pending);
}

#[test]
fn test_promise_poll_ready() {
    let mut state = [0u8; 16];
    let mut promise = unsafe { ZrtlPromise::new(state.as_mut_ptr(), mock_ready_poll) };

    let result = unsafe { promise.poll() };
    assert_eq!(result, PollResult::Ready(42));
}

#[test]
fn test_promise_poll_failed() {
    let mut state = [0u8; 16];
    let mut promise = unsafe { ZrtlPromise::new(state.as_mut_ptr(), mock_failed_poll) };

    let result = unsafe { promise.poll() };
    assert_eq!(result, PollResult::Failed(-1));
}

#[test]
fn test_promise_block_on() {
    let mut state = [0u8; 16];
    let mut promise = unsafe { ZrtlPromise::new(state.as_mut_ptr(), mock_ready_poll) };

    let result = promise.block_on();
    assert_eq!(result, Ok(42));
}

#[test]
fn test_promise_counting() {
    let mut state = MockStateMachine {
        counter: 0,
        target: 3,
    };
    let mut promise = unsafe { ZrtlPromise::new(&mut state as *mut _ as *mut u8, counting_poll) };

    // Should need 4 polls (0->1, 1->2, 2->3, then ready at 3)
    assert_eq!(unsafe { promise.poll() }, PollResult::Pending);
    assert_eq!(state.counter, 1);

    assert_eq!(unsafe { promise.poll() }, PollResult::Pending);
    assert_eq!(state.counter, 2);

    assert_eq!(unsafe { promise.poll() }, PollResult::Pending);
    assert_eq!(state.counter, 3);

    assert_eq!(unsafe { promise.poll() }, PollResult::Ready(3));
}

#[test]
fn test_promise_block_on_timeout() {
    let mut state = [0u8; 16];
    let mut promise = unsafe { ZrtlPromise::new(state.as_mut_ptr(), mock_ready_poll) };

    let result = promise.block_on_timeout(Duration::from_secs(1));
    assert_eq!(result, Ok(42));
}

#[test]
fn test_promise_timeout_on_pending() {
    let mut state = [0u8; 16];
    let mut promise = unsafe { ZrtlPromise::new(state.as_mut_ptr(), mock_pending_poll) };

    let result = promise.block_on_timeout(Duration::from_millis(10));
    assert!(matches!(result, Err(PromiseError::Timeout)));
}

// ============================================================================
// PromiseAll Tests
// ============================================================================

#[test]
fn test_promise_all_empty() {
    let mut all = PromiseAll::new(vec![]);
    let result = all.poll();
    assert_eq!(result, PollResult::Ready(0));
}

#[test]
fn test_promise_all_single() {
    let mut state = [0u8; 16];
    let promises = vec![unsafe { ZrtlPromise::new(state.as_mut_ptr(), mock_ready_poll) }];

    let mut all = PromiseAll::new(promises);
    let result = all.poll();
    assert_eq!(result, PollResult::Ready(1));

    let results = all.results();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0], Some(42));
}

#[test]
fn test_promise_all_multiple() {
    let mut state1 = [0u8; 16];
    let mut state2 = [0u8; 16];

    let promises = vec![
        unsafe { ZrtlPromise::new(state1.as_mut_ptr(), mock_ready_poll) },
        unsafe { ZrtlPromise::new(state2.as_mut_ptr(), mock_ready_poll) },
    ];

    let mut all = PromiseAll::new(promises);
    let result = all.poll();
    assert_eq!(result, PollResult::Ready(2));

    let results = all.results();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], Some(42));
    assert_eq!(results[1], Some(42));
}

#[test]
fn test_promise_all_fail_fast() {
    let mut state1 = [0u8; 16];
    let mut state2 = [0u8; 16];

    let promises = vec![
        unsafe { ZrtlPromise::new(state1.as_mut_ptr(), mock_failed_poll) },
        unsafe { ZrtlPromise::new(state2.as_mut_ptr(), mock_ready_poll) },
    ];

    let mut all = PromiseAll::new(promises);
    let result = all.poll();
    assert_eq!(result, PollResult::Failed(-1));
}

// ============================================================================
// PromiseRace Tests
// ============================================================================

#[test]
fn test_promise_race_first_wins() {
    let mut state1 = [0u8; 16];
    let mut state2 = [0u8; 16];

    let promises = vec![
        unsafe { ZrtlPromise::new(state1.as_mut_ptr(), mock_ready_poll) },
        unsafe { ZrtlPromise::new(state2.as_mut_ptr(), mock_pending_poll) },
    ];

    let mut race = PromiseRace::new(promises);
    let result = race.poll();
    assert_eq!(result, PollResult::Ready(42));

    let winner = race.winner();
    assert_eq!(winner, Some((0, 42)));
}

#[test]
fn test_promise_race_second_wins() {
    let mut state1 = [0u8; 16];
    let mut state2 = [0u8; 16];

    let promises = vec![
        unsafe { ZrtlPromise::new(state1.as_mut_ptr(), mock_pending_poll) },
        unsafe { ZrtlPromise::new(state2.as_mut_ptr(), mock_ready_poll) },
    ];

    let mut race = PromiseRace::new(promises);
    let result = race.poll();
    assert_eq!(result, PollResult::Ready(42));

    let winner = race.winner();
    assert_eq!(winner, Some((1, 42)));
}

// ============================================================================
// PromiseAllSettled Tests
// ============================================================================

#[test]
fn test_promise_all_settled() {
    let mut state1 = [0u8; 16];
    let mut state2 = [0u8; 16];

    let promises = vec![
        unsafe { ZrtlPromise::new(state1.as_mut_ptr(), mock_ready_poll) },
        unsafe { ZrtlPromise::new(state2.as_mut_ptr(), mock_failed_poll) },
    ];

    let mut settled = PromiseAllSettled::new(promises);
    let done = settled.poll();
    assert!(done); // All settled in one poll since mock functions complete immediately

    let results = settled.results();
    assert_eq!(results.len(), 2);
    assert!(matches!(results[0], Some(SettledResult::Fulfilled(42))));
    assert!(matches!(results[1], Some(SettledResult::Rejected(-1))));
}

#[test]
fn test_promise_all_settled_block_on() {
    let mut state1 = [0u8; 16];
    let mut state2 = [0u8; 16];

    let promises = vec![
        unsafe { ZrtlPromise::new(state1.as_mut_ptr(), mock_ready_poll) },
        unsafe { ZrtlPromise::new(state2.as_mut_ptr(), mock_failed_poll) },
    ];

    let mut settled = PromiseAllSettled::new(promises);
    let results = settled.block_on();

    assert_eq!(results.len(), 2);
    assert!(matches!(results[0], SettledResult::Fulfilled(42)));
    assert!(matches!(results[1], SettledResult::Rejected(-1)));
}

// ============================================================================
// Task ID Tests
// ============================================================================

#[test]
fn test_next_task_id_increments() {
    let id1 = next_task_id();
    let id2 = next_task_id();
    let id3 = next_task_id();

    assert!(id2 > id1);
    assert!(id3 > id2);
}

// ============================================================================
// PromiseError Tests
// ============================================================================

#[test]
fn test_promise_error_display() {
    assert_eq!(format!("{}", PromiseError::Timeout), "Promise timed out");
    assert_eq!(
        format!("{}", PromiseError::Failed(-1)),
        "Promise failed with code -1"
    );
    assert_eq!(
        format!("{}", PromiseError::InvalidPromise),
        "Invalid promise"
    );
}
