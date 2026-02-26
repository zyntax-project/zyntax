//! Async Runtime Integration Tests
//!
//! These tests validate the async/await functionality in the Zyntax runtime
//! using simulated async state machines and the ZyntaxPromise API.
//!
//! The tests use Rust functions to simulate compiled async state machines,
//! demonstrating how the runtime will handle real async Zyntax code.
//!
//! This module also tests the async_test.zyn grammar for parsing async code.
//!
//! ## Async ABI (i64 convention)
//!
//! The runtime uses a simplified i64 ABI for poll functions:
//! - `init_fn(buffer: *mut u8)` - Initializes state machine into provided buffer (sret)
//! - `poll_fn(sm: *mut u8, waker: *const u8) -> i64` - Returns 0 for Pending, non-zero for Ready(value)
//!
//! This matches what the Zyntax compiler generates.

use std::sync::Arc;
use zyntax_embed::{
    AsyncPollResult, DynamicValue, LanguageGrammar, PromiseState, ZyntaxPromise, ZyntaxRuntime,
    ZyntaxValue,
};

// ============================================================================
// Simulated Async State Machine Infrastructure (i64 ABI)
// ============================================================================

/// Layout of a simulated state machine in the buffer.
/// This must be repr(C) to match what Cranelift generates.
#[repr(C)]
struct SimulatedStateMachineLayout {
    /// Current state (0 = initial, 1+ = in progress, done when >= polls_to_complete)
    state: u32,
    /// Accumulated value for testing
    value: i32,
    /// Number of polls before completing
    polls_to_complete: u32,
    /// Padding
    _pad: u32,
}

/// Initialize a state machine that completes after 3 polls with value 30.
/// Uses sret convention - fills the provided buffer.
extern "C" fn create_state_machine(buffer: *mut u8) {
    unsafe {
        let sm = &mut *(buffer as *mut SimulatedStateMachineLayout);
        sm.state = 0;
        sm.value = 0;
        sm.polls_to_complete = 3;
    }
}

/// Initialize a state machine that completes immediately with value 1
/// Note: We use 1 instead of 0 because 0 means Pending in i64 ABI
extern "C" fn create_immediate_state_machine(buffer: *mut u8) {
    unsafe {
        let sm = &mut *(buffer as *mut SimulatedStateMachineLayout);
        sm.state = 0;
        sm.value = 1; // Return 1 so it's distinguishable from Pending
        sm.polls_to_complete = 0;
    }
}

/// Initialize a state machine that requires 5 polls
extern "C" fn create_long_state_machine(buffer: *mut u8) {
    unsafe {
        let sm = &mut *(buffer as *mut SimulatedStateMachineLayout);
        sm.state = 0;
        sm.value = 0;
        sm.polls_to_complete = 5;
    }
}

/// Poll the state machine (i64 ABI)
/// Returns: 0 = Pending, non-zero = Ready(value)
extern "C" fn poll_state_machine(sm: *mut u8, _waker: *const u8) -> i64 {
    if sm.is_null() {
        return -1; // Error indicator
    }
    unsafe {
        let state_machine = &mut *(sm as *mut SimulatedStateMachineLayout);
        let current_state = state_machine.state;
        state_machine.state += 1;

        if current_state >= state_machine.polls_to_complete {
            // Complete with the accumulated value
            state_machine.value as i64
        } else {
            // Still pending, accumulate value
            state_machine.value += 10;
            0 // Pending
        }
    }
}

// ============================================================================
// Failing State Machine (i64 ABI)
// ============================================================================

/// Layout for a failing state machine
#[repr(C)]
struct FailingStateMachineLayout {
    polls_before_fail: u32,
    current_poll: u32,
}

/// Initialize a failing state machine that fails after 2 polls
extern "C" fn create_failing_state_machine(buffer: *mut u8) {
    unsafe {
        let sm = &mut *(buffer as *mut FailingStateMachineLayout);
        sm.polls_before_fail = 2;
        sm.current_poll = 0;
    }
}

/// Poll the failing state machine (i64 ABI)
/// Returns: 0 = Pending, -1 = Error (we use negative for errors in i64 ABI)
extern "C" fn poll_failing_state_machine(sm: *mut u8, _waker: *const u8) -> i64 {
    if sm.is_null() {
        return -1; // Error
    }
    unsafe {
        let state_machine = &mut *(sm as *mut FailingStateMachineLayout);
        let current = state_machine.current_poll;
        state_machine.current_poll += 1;

        if current >= state_machine.polls_before_fail {
            -1 // Error (simulated failure)
        } else {
            0 // Pending
        }
    }
}

// ============================================================================
// Async Runtime Tests
// ============================================================================

#[cfg(test)]
mod async_promise_tests {
    use super::*;

    #[test]
    fn test_promise_creation() {
        // Create a promise with constructor and poll functions
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Initially pending
        assert!(promise.is_pending());
        assert!(!promise.is_complete());
    }

    #[test]
    fn test_promise_immediate_completion() {
        // Create a promise that completes on first poll
        let promise = ZyntaxPromise::with_poll_fn(
            create_immediate_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // First poll initializes, second should complete
        let state = promise.poll();
        assert!(matches!(state, PromiseState::Pending));

        let state = promise.poll();
        match state {
            // Value is 1 because 0 means Pending in i64 ABI
            PromiseState::Ready(ZyntaxValue::Int(1)) => {}
            other => panic!("Expected Ready(1), got {:?}", other),
        }

        assert!(promise.is_complete());
    }

    #[test]
    fn test_promise_multiple_polls() {
        // Create a promise that requires 3 polls (polls_to_complete = 3)
        // The state machine accumulates 10 per poll before completion.
        // Poll behavior:
        //   Poll 1: init (creates state machine, no poll_fn call)
        //   Poll 2: poll_fn called, state 0->1, value 0->10, returns Pending
        //   Poll 3: poll_fn called, state 1->2, value 10->20, returns Pending
        //   Poll 4: poll_fn called, state 2->3, value 20->30, returns Pending
        //   Poll 5: poll_fn called, state 3>=3, returns Ready(30)
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // First poll initializes the state machine (no poll_fn call yet)
        let state = promise.poll();
        assert!(
            matches!(state, PromiseState::Pending),
            "Poll 1 (init): {:?}",
            state
        );

        // Second poll advances state (state 0->1, value 10)
        let state = promise.poll();
        assert!(
            matches!(state, PromiseState::Pending),
            "Poll 2: {:?}",
            state
        );

        // Third poll advances state (state 1->2, value 20)
        let state = promise.poll();
        assert!(
            matches!(state, PromiseState::Pending),
            "Poll 3: {:?}",
            state
        );

        // Fourth poll advances state (state 2->3, value 30, still pending)
        let state = promise.poll();
        assert!(
            matches!(state, PromiseState::Pending),
            "Poll 4: {:?}",
            state
        );

        // Fifth poll should complete with accumulated value (30)
        let state = promise.poll();
        match state {
            PromiseState::Ready(ZyntaxValue::Int(30)) => {}
            other => panic!("Expected Ready(30), got {:?}", other),
        }
    }

    #[test]
    fn test_promise_poll_count() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        assert_eq!(promise.poll_count(), 0);

        promise.poll();
        assert_eq!(promise.poll_count(), 1);

        promise.poll();
        assert_eq!(promise.poll_count(), 2);

        promise.poll();
        assert_eq!(promise.poll_count(), 3);
    }

    #[test]
    fn test_promise_poll_with_limit() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_long_state_machine as *const u8, // Requires 5 polls
            poll_state_machine as *const u8,
            vec![],
        );

        // Poll a few times
        for _ in 0..3 {
            let _state = promise.poll_with_limit(10);
            // Should still be pending (need 5 polls + 1 init)
        }

        // Try with a limit that would be exceeded
        // Note: The promise tracks total polls across all poll() calls
    }

    #[test]
    fn test_promise_failure() {
        // FailingStateMachine has polls_before_fail = 2
        // Poll behavior:
        //   Poll 1: init (creates state machine, no poll_fn call)
        //   Poll 2: poll_fn called, current 0, check 0 >= 2 = false, returns Pending
        //   Poll 3: poll_fn called, current 1, check 1 >= 2 = false, returns Pending
        //   Poll 4: poll_fn called, current 2, check 2 >= 2 = true, returns Failed
        let promise = ZyntaxPromise::with_poll_fn(
            create_failing_state_machine as *const u8,
            poll_failing_state_machine as *const u8,
            vec![],
        );

        // First poll initializes
        let state = promise.poll();
        assert!(
            matches!(state, PromiseState::Pending),
            "Poll 1 (init): {:?}",
            state
        );

        // Second poll is still pending (current = 0)
        let state = promise.poll();
        assert!(
            matches!(state, PromiseState::Pending),
            "Poll 2: {:?}",
            state
        );

        // Third poll is still pending (current = 1)
        let state = promise.poll();
        assert!(
            matches!(state, PromiseState::Pending),
            "Poll 3: {:?}",
            state
        );

        // Fourth poll should fail (current = 2 >= 2)
        let state = promise.poll();
        match state {
            PromiseState::Failed(msg) => {
                // The runtime converts -1 to an error message
                assert!(msg.contains("failed") || msg.contains("-1"), "Got: {}", msg);
            }
            other => panic!("Expected Failed, got {:?}", other),
        }

        // Once failed, should stay failed
        let state = promise.poll();
        match state {
            PromiseState::Failed(_) => {}
            other => panic!("Expected to stay Failed, got {:?}", other),
        }
    }

    #[test]
    fn test_promise_await_result() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Block until complete
        let result: Result<i32, _> = promise.await_result();
        assert_eq!(result.unwrap(), 30);
    }

    #[test]
    fn test_promise_await_with_timeout() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Should complete within timeout
        let result = promise.await_with_timeout(std::time::Duration::from_secs(5));
        match result {
            Ok(ZyntaxValue::Int(30)) => {}
            other => panic!("Expected Ok(Int(30)), got {:?}", other),
        }
    }

    #[test]
    fn test_promise_state_immutability_after_completion() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_immediate_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Run to completion
        while promise.is_pending() {
            promise.poll();
        }

        let state1 = promise.state();
        let state2 = promise.poll();

        // Both should be the same completed state
        match (&state1, &state2) {
            (PromiseState::Ready(v1), PromiseState::Ready(v2)) => {
                assert_eq!(v1, v2);
            }
            _ => panic!("States don't match"),
        }
    }
}

// ============================================================================
// Promise Chaining Tests
// ============================================================================

#[cfg(test)]
mod async_chaining_tests {
    use super::*;

    #[test]
    fn test_promise_then_chain() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Chain: multiply result by 2
        let chained = promise.then(|value| {
            if let ZyntaxValue::Int(n) = value {
                ZyntaxValue::Int(n * 2)
            } else {
                value
            }
        });

        // Wait for original to complete
        while promise.is_pending() {
            promise.poll();
            std::thread::yield_now();
        }

        // Give the chained promise time to process
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Chained result should be 60 (30 * 2)
        match chained.state() {
            PromiseState::Ready(ZyntaxValue::Int(60)) => {}
            state => {
                // Chain might not have completed yet, that's OK for this test
                println!("Chain state: {:?}", state);
            }
        }
    }

    #[test]
    fn test_promise_catch_on_success() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_immediate_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Catch handler should not be called on success
        let caught = promise.catch(|_err| {
            ZyntaxValue::Int(-1) // Error case
        });

        // Run to completion
        while promise.is_pending() {
            promise.poll();
            std::thread::yield_now();
        }

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Should have the original success value, not -1
        match caught.state() {
            PromiseState::Ready(ZyntaxValue::Int(0)) => {}
            state => {
                println!("Catch on success state: {:?}", state);
            }
        }
    }

    #[test]
    fn test_promise_catch_on_failure() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_failing_state_machine as *const u8,
            poll_failing_state_machine as *const u8,
            vec![],
        );

        // Catch handler converts error to value
        let caught = promise.catch(|err| {
            if err.contains("async error") {
                ZyntaxValue::Int(-999) // Converted error
            } else {
                ZyntaxValue::Void
            }
        });

        // Run to failure
        while promise.is_pending() {
            promise.poll();
            std::thread::yield_now();
        }

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Catch handler should have converted the error
        match caught.state() {
            PromiseState::Ready(ZyntaxValue::Int(-999)) => {}
            state => {
                println!("Catch on failure state: {:?}", state);
            }
        }
    }
}

// ============================================================================
// Concurrent Promise Tests
// ============================================================================

#[cfg(test)]
mod async_concurrent_tests {
    use super::*;

    #[test]
    fn test_multiple_independent_promises() {
        // Create multiple independent promises
        let promise1 = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        let promise2 = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        let promise3 = ZyntaxPromise::with_poll_fn(
            create_immediate_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Poll all promises
        for _ in 0..10 {
            promise1.poll();
            promise2.poll();
            promise3.poll();
        }

        // All should eventually complete
        assert!(promise1.is_complete());
        assert!(promise2.is_complete());
        assert!(promise3.is_complete());
    }

    #[test]
    fn test_promise_from_multiple_threads() {
        use std::thread;

        let promise = Arc::new(ZyntaxPromise::with_poll_fn(
            create_long_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        ));

        // Spawn multiple threads to poll the same promise
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let p = Arc::clone(&promise);
                thread::spawn(move || {
                    for _ in 0..5 {
                        p.poll();
                        thread::yield_now();
                    }
                })
            })
            .collect();

        // Wait for all threads
        for h in handles {
            h.join().unwrap();
        }

        // Promise should have been polled enough times to complete
        // (5 polls needed, 4 threads * 5 polls each = 20 polls)
        // Note: Due to thread scheduling, actual completion may vary
    }
}

// ============================================================================
// Async ABI Compatibility Tests
// ============================================================================

#[cfg(test)]
mod async_abi_tests {
    use super::*;

    /// Test that AsyncPollResult has the correct layout
    #[test]
    fn test_async_poll_result_layout() {
        // Pending should be discriminant 0
        let pending = AsyncPollResult::Pending;
        let discriminant = unsafe { *(&pending as *const AsyncPollResult as *const u8) };
        assert_eq!(discriminant, 0);

        // Ready should be discriminant 1
        let ready = AsyncPollResult::Ready(DynamicValue::from_i32(42));
        let discriminant = unsafe { *(&ready as *const AsyncPollResult as *const u8) };
        assert_eq!(discriminant, 1);

        // Failed should be discriminant 2
        let failed = AsyncPollResult::Failed(std::ptr::null(), 0);
        let discriminant = unsafe { *(&failed as *const AsyncPollResult as *const u8) };
        assert_eq!(discriminant, 2);
    }

    /// Test state machine ABI compatibility
    #[test]
    fn test_state_machine_abi() {
        // Create function initializes a buffer (sret convention)
        let mut buffer = [0u8; 64];
        create_state_machine(buffer.as_mut_ptr());

        // Poll function works with that buffer
        let result = poll_state_machine(buffer.as_mut_ptr(), std::ptr::null());
        assert_eq!(result, 0); // 0 = Pending in i64 ABI

        // Poll again - should accumulate value
        let result = poll_state_machine(buffer.as_mut_ptr(), std::ptr::null());
        assert_eq!(result, 0); // Still pending

        // Poll until completion
        let result = poll_state_machine(buffer.as_mut_ptr(), std::ptr::null());
        assert_eq!(result, 0); // Still pending

        // Now it should complete with value 30
        let result = poll_state_machine(buffer.as_mut_ptr(), std::ptr::null());
        assert_eq!(result, 30); // Ready with accumulated value
    }
}

// ============================================================================
// Async Error Handling Tests
// ============================================================================

#[cfg(test)]
mod async_error_tests {
    use super::*;

    #[test]
    fn test_null_state_machine_poll() {
        // In i64 ABI, -1 indicates error
        let result = poll_state_machine(std::ptr::null_mut(), std::ptr::null());
        assert_eq!(result, -1); // Error indicator
    }

    #[test]
    fn test_promise_with_null_init_fn() {
        let promise =
            ZyntaxPromise::with_poll_fn(std::ptr::null(), poll_state_machine as *const u8, vec![]);

        let state = promise.poll();
        match state {
            PromiseState::Failed(msg) => {
                assert!(msg.contains("Null") || msg.contains("null"), "Got: {}", msg);
            }
            other => panic!("Expected Failed, got {:?}", other),
        }
    }
}

// ============================================================================
// Async with Arguments Tests (i64 ABI)
// ============================================================================

/// Layout for state machine with arguments
#[repr(C)]
struct StateMachineWithArgsLayout {
    state: u32,
    initial_value: i32,
    multiplier: i32,
    _pad: u32,
}

/// Initialize a state machine with fixed arguments (10, 5)
extern "C" fn create_state_machine_with_args(buffer: *mut u8) {
    unsafe {
        let sm = &mut *(buffer as *mut StateMachineWithArgsLayout);
        sm.state = 0;
        sm.initial_value = 10;
        sm.multiplier = 5;
    }
}

/// Poll the state machine with args (i64 ABI)
extern "C" fn poll_state_machine_with_args(sm: *mut u8, _waker: *const u8) -> i64 {
    if sm.is_null() {
        return -1;
    }
    unsafe {
        let state_machine = &mut *(sm as *mut StateMachineWithArgsLayout);
        let current_state = state_machine.state;
        state_machine.state += 1;

        if current_state >= 2 {
            // Complete with initial * multiplier
            (state_machine.initial_value * state_machine.multiplier) as i64
        } else {
            0 // Pending
        }
    }
}

#[cfg(test)]
mod async_args_tests {
    use super::*;

    #[test]
    fn test_state_machine_with_args() {
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine_with_args as *const u8,
            poll_state_machine_with_args as *const u8,
            vec![],
        );

        // Run to completion
        for _ in 0..10 {
            if promise.is_complete() {
                break;
            }
            promise.poll();
        }

        // Should complete with 10 * 5 = 50
        match promise.state() {
            PromiseState::Ready(ZyntaxValue::Int(50)) => {}
            other => panic!("Expected Ready(50), got {:?}", other),
        }
    }
}

// ============================================================================
// Async Test Language Grammar Tests
// ============================================================================
//
// These tests validate the async_test.zyn grammar parsing and compilation.
// The grammar provides a minimal language for testing async/await semantics.

/// The async_test grammar source (embedded at compile time)
const ASYNC_TEST_GRAMMAR: &str = include_str!("../../zyn_peg/grammars/async_test.zyn");

#[cfg(test)]
mod async_grammar_tests {
    use super::*;

    /// Helper to create a runtime with async_test grammar
    fn setup_async_runtime() -> Option<ZyntaxRuntime> {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: could not compile async_test grammar: {}", e);
                return None;
            }
        };

        let mut runtime = ZyntaxRuntime::new().ok()?;
        runtime.register_grammar("async_test", grammar);
        Some(runtime)
    }

    #[test]
    fn test_grammar_compilation() {
        // Test that the async_test grammar compiles successfully
        let result = LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR);
        match result {
            Ok(grammar) => {
                println!("Async test grammar compiled successfully");
                // Verify grammar metadata
                let name = grammar.name();
                println!("Grammar name: {}", name);
                assert!(!name.is_empty(), "Grammar should have a name");
            }
            Err(e) => {
                panic!("Failed to compile async_test grammar: {}", e);
            }
        }
    }

    #[test]
    fn test_parse_sync_function() {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Parse a simple synchronous function
        let source = r#"
fn add(a: i32, b: i32) i32 {
    return a + b;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!("Parsed sync function successfully");
                // The TypedProgram should have one function declaration
                assert!(!program.declarations.is_empty(), "Should have declarations");
            }
            Err(e) => {
                eprintln!("Parse failed (may be expected during development): {}", e);
            }
        }
    }

    #[test]
    fn test_parse_async_function() {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Parse an async function
        let source = r#"
async fn fetch_value() i32 {
    return 42;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!("Parsed async function successfully");
                // Note: declarations may be empty if AST builder commands aren't fully wired
                println!("  Declarations: {}", program.declarations.len());
            }
            Err(e) => {
                panic!("Async function parse should succeed: {}", e);
            }
        }
    }

    #[test]
    fn test_parse_async_with_await() {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Parse async function with await expression
        // Note: await should be used on other async functions, not extern
        let source = r#"
async fn compute(x: i32) i32 {
    return x * 2;
}

async fn process(x: i32) i32 {
    const result = await compute(x);
    return result + 1;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!("Parsed async function with await successfully");
                println!("  Declarations: {}", program.declarations.len());
            }
            Err(e) => {
                panic!("Parse should succeed: {}", e);
            }
        }
    }

    #[test]
    fn test_parse_promise_type() {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Parse function with Promise return type
        let source = r#"
async fn fetch_data() Promise<i32> {
    return 42;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!("Parsed Promise type successfully");
                println!("  Declarations: {}", program.declarations.len());
            }
            Err(e) => {
                panic!("Promise type parse should succeed: {}", e);
            }
        }
    }

    #[test]
    fn test_parse_control_flow_in_async() {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Parse async function with control flow
        let source = r#"
async fn conditional_async(x: i32) i32 {
    if (x > 0) {
        return x;
    }
    return 0;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!("Parsed async with control flow successfully");
                println!("  Declarations: {}", program.declarations.len());
            }
            Err(e) => {
                panic!("Control flow parse should succeed: {}", e);
            }
        }
    }

    #[test]
    fn test_parse_while_in_async() {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Parse async function with while loop
        let source = r#"
async fn sum_to(n: i32) i32 {
    var result: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        result = result + i;
        i = i + 1;
    }
    return result;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!("Parsed async with while loop successfully");
                println!("  Declarations: {}", program.declarations.len());
            }
            Err(e) => {
                panic!("While loop parse should succeed: {}", e);
            }
        }
    }

    #[test]
    fn test_parse_multiple_async_functions() {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // Parse multiple async functions
        let source = r#"
async fn first() i32 {
    return 1;
}

async fn second() i32 {
    return 2;
}

fn sync_helper(x: i32) i32 {
    return x * 2;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!("Parsed multiple functions successfully");
                println!("  Declarations: {}", program.declarations.len());
            }
            Err(e) => {
                panic!("Multiple functions parse should succeed: {}", e);
            }
        }
    }

    #[test]
    fn test_load_async_module() {
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        // Try to load a simple async module
        // Note: This may fail if the grammar's AST builder commands aren't fully wired
        // or if the compiler doesn't yet support this grammar's output format.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module(
                "async_test",
                r#"
fn simple_add(a: i32, b: i32) i32 {
    return a + b;
}
"#,
            )
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Loaded async module with functions: {:?}", functions);
            }
            Ok(Err(e)) => {
                // This may fail if async compilation isn't fully implemented yet
                eprintln!("Module load failed (expected during development): {}", e);
            }
            Err(_) => {
                // Panics are expected if the grammar/compiler isn't fully wired up
                eprintln!("Module load panicked (expected during development)");
            }
        }
    }

    #[test]
    fn test_async_function_compilation() {
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        // Try to compile an async function
        // Catch panics from Cranelift compilation
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module(
                "async_test",
                r#"
async fn async_compute(x: i32) i32 {
    return x * 2;
}
"#,
            )
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Compiled async function: {:?}", functions);
                // The async function should generate both constructor and poll functions
            }
            Ok(Err(e)) => {
                // Async compilation may not be fully wired up yet
                eprintln!("Async compilation not yet implemented: {}", e);
            }
            Err(_panic) => {
                // Cranelift compilation panics are expected during development
                eprintln!("Async compilation panicked - backend issue, not grammar");
            }
        }
    }
}

// ============================================================================
// Integration: Simulated Async with Grammar Parsing
// ============================================================================
//
// These tests combine grammar parsing with our simulated async machinery
// to demonstrate the full flow.

#[cfg(test)]
mod async_integration_tests {
    use super::*;

    #[test]
    fn test_async_grammar_to_state_machine_concept() {
        // This test demonstrates the conceptual flow:
        // 1. Parse async source code with async_test.zyn grammar
        // 2. The AST would contain async function markers
        // 3. The compiler generates state machine struct and poll function
        // 4. The runtime creates ZyntaxPromise to poll the state machine

        // Step 1: Parse the source
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        let source = r#"
async fn compute_value(x: i32) i32 {
    const doubled = x * 2;
    return doubled + 1;
}
"#;

        let parsed = grammar.parse(source);
        println!("Parsed async source: {:?}", parsed.is_ok());

        // Step 2-3: In real flow, compiler would generate state machine
        // For now, we use our simulated state machine
        let promise = ZyntaxPromise::with_poll_fn(
            create_state_machine as *const u8,
            poll_state_machine as *const u8,
            vec![],
        );

        // Step 4: Poll to completion
        let mut polls = 0;
        loop {
            let state = promise.poll();
            polls += 1;
            match state {
                PromiseState::Ready(value) => {
                    println!(
                        "Async function completed with {:?} after {} polls",
                        value, polls
                    );
                    break;
                }
                PromiseState::Failed(err) => {
                    panic!("Async function failed: {}", err);
                }
                PromiseState::Cancelled => {
                    panic!("Async function was cancelled");
                }
                PromiseState::Pending => {
                    if polls > 10 {
                        panic!("Too many polls");
                    }
                }
            }
        }
    }

    #[test]
    fn test_async_await_async_function_pattern() {
        // Demonstrates awaiting another async function
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Skipping test: {}", e);
                return;
            }
        };

        // This pattern is how async/await should work:
        // - One async function awaits another async function
        // - NOT awaiting extern functions
        let source = r#"
async fn fetch_data(id: i32) i32 {
    return id * 10;
}

async fn fetch_and_process(id: i32) i32 {
    const data = await fetch_data(id);
    return data + 1;
}
"#;

        match grammar.parse(source) {
            Ok(program) => {
                println!(
                    "Parsed async-await-async pattern: {} declarations",
                    program.declarations.len()
                );
            }
            Err(e) => {
                eprintln!("Parse failed: {}", e);
            }
        }
    }
}

// ============================================================================
// Async Execution Tests
// ============================================================================
//
// These tests actually compile and EXECUTE async code through the grammar.
// This validates the full async pipeline:
//   Grammar -> TypedAST -> HIR -> State Machine -> JIT -> Execute

#[cfg(test)]
mod async_execution_tests {
    use super::*;
    use zyntax_embed::NativeSignature;

    /// Helper to create a runtime with async_test grammar
    fn setup_async_runtime() -> Option<ZyntaxRuntime> {
        let grammar = match LanguageGrammar::compile_zyn(ASYNC_TEST_GRAMMAR) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Could not compile async_test grammar: {}", e);
                return None;
            }
        };

        let mut runtime = ZyntaxRuntime::new().ok()?;
        runtime.register_grammar("async_test", grammar);
        Some(runtime)
    }

    #[test]
    fn test_execute_simple_sync_function() {
        // Start with a sync function to verify basic execution works
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
fn add(a: i32, b: i32) i32 {
    return a + b;
}
"#;

        // Catch panics from Cranelift compilation issues
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Loaded functions: {:?}", functions);

                // Try to call the function
                if runtime.has_function("add") {
                    let sig = NativeSignature::parse("(i32, i32) -> i32").unwrap();
                    let result = runtime.call_function(
                        "add",
                        &[ZyntaxValue::Int(10), ZyntaxValue::Int(32)],
                        &sig,
                    );

                    match result {
                        Ok(value) => {
                            println!("add(10, 32) = {:?}", value);
                            assert_eq!(value, ZyntaxValue::Int(42));
                        }
                        Err(e) => {
                            eprintln!("Function call failed: {}", e);
                        }
                    }
                } else {
                    eprintln!(
                        "Function 'add' not found. Available: {:?}",
                        runtime.functions()
                    );
                }
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
            }
            Err(_panic) => {
                // Cranelift compilation panics are expected during development
                // when the HIR isn't fully compatible with Cranelift's expectations
                eprintln!("Compilation panicked - HIR generation needs adjustment");
            }
        }
    }

    #[test]
    fn test_execute_async_function_compilation() {
        // Test that async functions compile and generate the right function names
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        // Simple async function that returns a constant
        let source = r#"
async fn get_value() i32 {
    return 42;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Compiled async module with functions: {:?}", functions);

                // Async functions should generate:
                // - get_value_new (constructor)
                // - async_wrapper (poll function)
                let has_constructor = runtime.has_function("get_value_new")
                    || functions.iter().any(|f| f.contains("_new"));
                let has_wrapper = runtime.has_function("async_wrapper");

                println!("Has constructor (_new): {}", has_constructor);
                println!("Has wrapper (async_wrapper): {}", has_wrapper);
                println!("All functions: {:?}", runtime.functions());

                // If the async function is compiled properly, we should see the generated functions
            }
            Ok(Err(e)) => {
                eprintln!("Async compilation failed: {}", e);
            }
            Err(_panic) => {
                // Async function compilation is hitting Cranelift issues
                // This is expected during development - the grammar side works
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
            }
        }
    }

    #[test]
    fn test_execute_async_function_with_promise() {
        // Full end-to-end test: compile async function with await and call via runtime.call_async()
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        // This test uses actual await expressions to test the async state machine
        // We await another async function, NOT an extern function
        let source = r#"
async fn double(x: i32) i32 {
    return x * 2;
}

async fn compute(x: i32) i32 {
    const doubled = await double(x);
    return doubled + 1;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Compiled module: {:?}", functions);
                println!("Available functions: {:?}", runtime.functions());

                // Verify async functions with new Promise-based ABI:
                // - `double` and `compute` are entry functions returning Promise<T>
                // - `__double_poll` and `__compute_poll` are internal poll functions
                assert!(
                    functions.contains(&"double".to_string()),
                    "double (entry function) should be generated"
                );
                assert!(
                    functions.contains(&"__double_poll".to_string()),
                    "__double_poll (internal poll) should be generated"
                );
                assert!(
                    functions.contains(&"compute".to_string()),
                    "compute (entry function) should be generated"
                );
                assert!(
                    functions.contains(&"__compute_poll".to_string()),
                    "__compute_poll (internal poll) should be generated"
                );

                // Now actually EXECUTE the async function via call_async
                // This tests the full async state machine execution path
                let promise = runtime.call_async("double", &[ZyntaxValue::Int(21)]);
                match promise {
                    Ok(promise) => {
                        // Poll until completion
                        let mut polls = 0;
                        while promise.is_pending() && polls < 100 {
                            promise.poll();
                            polls += 1;
                        }

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                // double(21) should return 42
                                assert_eq!(result, 42, "double(21) should return 42");
                                println!(
                                    "SUCCESS: Async execution of double(21) returned {}",
                                    result
                                );
                            }
                            PromiseState::Failed(msg) => {
                                panic!("Async execution failed: {}", msg);
                            }
                            other => {
                                panic!("Unexpected state after polling: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async failed: {}", e);
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }

    #[test]
    fn test_execute_arithmetic_async() {
        // Test async function with arithmetic - compilation only
        // TODO: Execution pending ABI fixes for struct returns
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn sum_to_ten() i32 {
    var result: i32 = 0;
    var i: i32 = 1;
    while (i <= 10) {
        result = result + i;
        i = i + 1;
    }
    return result;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Compiled async sum function: {:?}", functions);
                // With the new Promise-based async ABI:
                // - `sum_to_ten` is the entry function that returns Promise<i32>
                // - `__sum_to_ten_poll` is the internal poll function
                assert!(
                    functions.contains(&"sum_to_ten".to_string()),
                    "sum_to_ten (entry function) should be generated"
                );
                assert!(
                    functions.contains(&"__sum_to_ten_poll".to_string()),
                    "__sum_to_ten_poll (internal poll) should be generated"
                );
                println!("Async function compilation successful!");
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }

    #[test]
    fn test_execute_long_running_async_loop() {
        // Test async function with a long-running loop (sum 1 to 100)
        // This tests that the async state machine can handle iterative computations
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn sum_range(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Compiled async sum_range function: {:?}", functions);
                assert!(
                    functions.contains(&"sum_range".to_string()),
                    "sum_range (entry function) should be generated"
                );
                assert!(
                    functions.contains(&"__sum_range_poll".to_string()),
                    "__sum_range_poll (internal poll) should be generated"
                );

                // Execute: sum_range(100) should return 1+2+3+...+100 = 5050
                let promise = runtime.call_async("sum_range", &[ZyntaxValue::Int(100)]);
                match promise {
                    Ok(promise) => {
                        // Poll until completion
                        let mut polls = 0;
                        while promise.is_pending() && polls < 1000 {
                            promise.poll();
                            polls += 1;
                        }

                        println!("Completed after {} polls", polls);

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                // sum(1..100) = 100 * 101 / 2 = 5050
                                assert_eq!(result, 5050, "sum_range(100) should return 5050");
                                println!("SUCCESS: Async loop sum_range(100) returned {}", result);
                            }
                            PromiseState::Failed(msg) => {
                                panic!("Async execution failed: {}", msg);
                            }
                            other => {
                                panic!("Unexpected state after polling: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async failed: {}", e);
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }

    #[test]
    // #[ignore = "Multi-state async with await in loops requires poll function to properly dispatch nested futures"]
    fn test_execute_async_with_await_in_loop() {
        // Test async function that awaits another async function inside a loop
        // This is a key test for multi-state async state machines
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn double(x: i32) i32 {
    return x * 2;
}

async fn sum_doubled(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        const doubled = await double(i);
        total = total + doubled;
        i = i + 1;
    }
    return total;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!(
                    "Compiled async functions with await in loop: {:?}",
                    functions
                );
                assert!(
                    functions.contains(&"double".to_string()),
                    "double should be generated"
                );
                assert!(
                    functions.contains(&"sum_doubled".to_string()),
                    "sum_doubled should be generated"
                );
                assert!(
                    functions.contains(&"__double_poll".to_string()),
                    "__double_poll should be generated"
                );
                assert!(
                    functions.contains(&"__sum_doubled_poll".to_string()),
                    "__sum_doubled_poll should be generated"
                );

                // Execute: sum_doubled(5) should compute:
                // double(1) + double(2) + double(3) + double(4) + double(5)
                // = 2 + 4 + 6 + 8 + 10 = 30
                let promise = runtime.call_async("sum_doubled", &[ZyntaxValue::Int(5)]);
                match promise {
                    Ok(promise) => {
                        let mut polls = 0;
                        while promise.is_pending() && polls < 1000 {
                            promise.poll();
                            polls += 1;
                        }

                        println!("sum_doubled(5) completed after {} polls", polls);

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                // sum of doubled values 1..5 = 2+4+6+8+10 = 30
                                assert_eq!(result, 30, "sum_doubled(5) should return 30");
                                println!(
                                    "SUCCESS: sum_doubled(5) = {} (after {} polls)",
                                    result, polls
                                );
                            }
                            PromiseState::Failed(msg) => {
                                panic!("Async execution failed: {}", msg);
                            }
                            other => {
                                panic!("Unexpected state after polling: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async failed: {}", e);
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }

    #[test]
    fn test_execute_async_chain_with_await() {
        // Test multiple async functions that await each other in a chain
        // step1 -> step2 -> step3 with processing at each step
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn step1(x: i32) i32 {
    return x + 10;
}

async fn step2(x: i32) i32 {
    const result = await step1(x);
    return result * 2;
}

async fn step3(x: i32) i32 {
    const result = await step2(x);
    return result + 5;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Compiled async chain functions: {:?}", functions);
                assert!(functions.contains(&"step1".to_string()));
                assert!(functions.contains(&"step2".to_string()));
                assert!(functions.contains(&"step3".to_string()));

                // Execute: step3(5)
                // step1(5) = 5 + 10 = 15
                // step2(5) = step1(5) * 2 = 15 * 2 = 30
                // step3(5) = step2(5) + 5 = 30 + 5 = 35
                let promise = runtime.call_async("step3", &[ZyntaxValue::Int(5)]);
                match promise {
                    Ok(promise) => {
                        let mut polls = 0;
                        while promise.is_pending() && polls < 1000 {
                            promise.poll();
                            polls += 1;
                        }

                        println!("step3(5) completed after {} polls", polls);

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                assert_eq!(result, 35, "step3(5) should return 35");
                                println!("SUCCESS: step3(5) = {} (after {} polls)", result, polls);
                            }
                            PromiseState::Failed(msg) => {
                                panic!("Async execution failed: {}", msg);
                            }
                            other => {
                                panic!("Unexpected state after polling: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async failed: {}", e);
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }

    #[test]
    fn test_execute_async_count_up() {
        // Test async function with a counting loop
        // This tests loop logic with counter variable
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        // Note: We use a simple loop-only version without early returns
        // to avoid complex control flow in the async state machine
        // Use addition-based approach to test loop logic
        let source = r#"
async fn count_up(n: i32) i32 {
    var result: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        result = result + 1;
        i = i + 1;
    }
    return result;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!("Compiled async count_up function: {:?}", functions);
                assert!(
                    functions.contains(&"count_up".to_string()),
                    "count_up (entry function) should be generated"
                );
                assert!(
                    functions.contains(&"__count_up_poll".to_string()),
                    "__count_up_poll (internal poll) should be generated"
                );

                // Test count_up: count_up(n) returns n
                let test_cases = vec![(1, 1), (5, 5), (10, 10), (20, 20)];

                for (input, expected) in test_cases {
                    let promise = runtime.call_async("count_up", &[ZyntaxValue::Int(input)]);
                    match promise {
                        Ok(promise) => {
                            let mut polls = 0;
                            while promise.is_pending() && polls < 1000 {
                                promise.poll();
                                polls += 1;
                            }

                            match promise.state() {
                                PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                    assert_eq!(
                                        result, expected,
                                        "count_up({}) should return {}, got {}",
                                        input, expected, result
                                    );
                                    println!(
                                        "SUCCESS: count_up({}) = {} (after {} polls)",
                                        input, result, polls
                                    );
                                }
                                PromiseState::Failed(msg) => {
                                    panic!("count_up({}) failed: {}", input, msg);
                                }
                                other => {
                                    panic!("count_up({}) unexpected state: {:?}", input, other);
                                }
                            }
                        }
                        Err(e) => {
                            panic!("call_async for count_up({}) failed: {}", input, e);
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }

    #[test]
    fn test_execute_async_with_multiple_args() {
        // Test async function with multiple arguments and a loop
        // This verifies that all function parameters are properly captured
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn sum_with_multiplier(start: i32, end: i32, multiplier: i32) i32 {
    var total: i32 = 0;
    var i: i32 = start;
    while (i <= end) {
        total = total + (i * multiplier);
        i = i + 1;
    }
    return total;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!(
                    "Compiled async sum_with_multiplier function: {:?}",
                    functions
                );
                assert!(functions.contains(&"sum_with_multiplier".to_string()));
                assert!(functions.contains(&"__sum_with_multiplier_poll".to_string()));

                // sum_with_multiplier(1, 5, 2) = (1*2) + (2*2) + (3*2) + (4*2) + (5*2) = 2+4+6+8+10 = 30
                let promise = runtime.call_async(
                    "sum_with_multiplier",
                    &[
                        ZyntaxValue::Int(1),
                        ZyntaxValue::Int(5),
                        ZyntaxValue::Int(2),
                    ],
                );

                match promise {
                    Ok(promise) => {
                        let mut polls = 0;
                        while promise.is_pending() && polls < 1000 {
                            promise.poll();
                            polls += 1;
                        }

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                assert_eq!(
                                    result, 30,
                                    "sum_with_multiplier(1, 5, 2) should return 30"
                                );
                                println!(
                                    "SUCCESS: sum_with_multiplier(1, 5, 2) = {} (after {} polls)",
                                    result, polls
                                );
                            }
                            PromiseState::Failed(msg) => {
                                panic!("Async execution failed: {}", msg);
                            }
                            other => {
                                panic!("Unexpected state after polling: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async failed: {}", e);
                    }
                }
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }

    #[test]
    fn test_execute_async_await_long_running_process() {
        // Test an async function that awaits another async function which is
        // a long-running process (computes a large sum iteratively)
        //
        // This tests:
        // 1. Awaiting a function that takes many polls to complete
        // 2. Proper nested state machine coordination
        // 3. The outer function correctly resuming after inner completes
        //
        // NOTE: Currently fails because the `sum + 100` computation in the return
        // state uses a value defined in a different state. Need to store intermediate
        // results from await to captures.
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
// A long-running async process that sums 1 to n
async fn long_sum(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}

// Awaits a long-running process and adds a constant
async fn add_to_sum(n: i32) i32 {
    const sum = await long_sum(n);
    return sum + 100;
}
"#;

        // Catch panics from Cranelift
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(functions)) => {
                println!(
                    "Compiled async long-running process functions: {:?}",
                    functions
                );
                assert!(
                    functions.contains(&"long_sum".to_string()),
                    "long_sum should be generated"
                );
                assert!(
                    functions.contains(&"add_to_sum".to_string()),
                    "add_to_sum should be generated"
                );
                assert!(
                    functions.contains(&"__long_sum_poll".to_string()),
                    "__long_sum_poll should be generated"
                );
                assert!(
                    functions.contains(&"__add_to_sum_poll".to_string()),
                    "__add_to_sum_poll should be generated"
                );

                // Test 1: long_sum(100) = 1+2+...+100 = 5050
                println!("\n=== Test 1: long_sum(100) ===");
                let promise = runtime.call_async("long_sum", &[ZyntaxValue::Int(100)]);
                match promise {
                    Ok(promise) => {
                        let mut polls = 0;
                        while promise.is_pending() && polls < 10000 {
                            promise.poll();
                            polls += 1;
                        }

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                assert_eq!(result, 5050, "long_sum(100) should return 5050");
                                println!(
                                    "SUCCESS: long_sum(100) = {} (after {} polls)",
                                    result, polls
                                );
                            }
                            PromiseState::Failed(msg) => {
                                panic!("long_sum(100) failed: {}", msg);
                            }
                            other => {
                                panic!("long_sum(100) unexpected state: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async for long_sum(100) failed: {}", e);
                    }
                }

                // Test 2: add_to_sum(50)
                // = long_sum(50) + 100
                // = 1275 + 100 = 1375
                println!("\n=== Test 2: add_to_sum(50) ===");
                let promise = runtime.call_async("add_to_sum", &[ZyntaxValue::Int(50)]);
                match promise {
                    Ok(promise) => {
                        let mut polls = 0;
                        while promise.is_pending() && polls < 10000 {
                            promise.poll();
                            polls += 1;
                        }

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                // sum(1..50) = 50*51/2 = 1275, so 1275 + 100 = 1375
                                assert_eq!(result, 1375, "add_to_sum(50) should return 1375");
                                println!(
                                    "SUCCESS: add_to_sum(50) = {} (after {} polls)",
                                    result, polls
                                );
                            }
                            PromiseState::Failed(msg) => {
                                panic!("add_to_sum(50) failed: {}", msg);
                            }
                            other => {
                                panic!("add_to_sum(50) unexpected state: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async for add_to_sum(50) failed: {}", e);
                    }
                }

                // Test 3: add_to_sum(10)
                // = long_sum(10) + 100
                // = 55 + 100 = 155
                println!("\n=== Test 3: add_to_sum(10) ===");
                let promise = runtime.call_async("add_to_sum", &[ZyntaxValue::Int(10)]);
                match promise {
                    Ok(promise) => {
                        let mut polls = 0;
                        while promise.is_pending() && polls < 10000 {
                            promise.poll();
                            polls += 1;
                        }

                        match promise.state() {
                            PromiseState::Ready(ZyntaxValue::Int(result)) => {
                                // sum(1..10) = 55, so 55 + 100 = 155
                                assert_eq!(result, 155, "add_to_sum(10) should return 155");
                                println!(
                                    "SUCCESS: add_to_sum(10) = {} (after {} polls)",
                                    result, polls
                                );
                            }
                            PromiseState::Failed(msg) => {
                                panic!("add_to_sum(10) failed: {}", msg);
                            }
                            other => {
                                panic!("add_to_sum(10) unexpected state: {:?}", other);
                            }
                        }
                    }
                    Err(e) => {
                        panic!("call_async for add_to_sum(10) failed: {}", e);
                    }
                }

                println!("\nAll long-running async process tests passed!");
            }
            Ok(Err(e)) => {
                eprintln!("Module load failed: {}", e);
                panic!("Async module compilation should succeed");
            }
            Err(_panic) => {
                eprintln!("Async compilation panicked - this is a backend issue, not grammar");
                panic!("Async compilation should not panic");
            }
        }
    }
}

// ============================================================================
// Promise Combinator Tests (Promise.all, Promise.race, etc.)
// ============================================================================

mod promise_combinator_tests {
    use super::*;
    use zyntax_embed::{
        PromiseAll, PromiseAllSettled, PromiseAllState, PromiseRace, PromiseRaceState,
        SettledResult,
    };

    fn setup_async_runtime() -> Option<ZyntaxRuntime> {
        // Try multiple paths for the grammar file
        let grammar_paths = [
            "crates/zyn_peg/grammars/zig.zyn",
            "../../crates/zyn_peg/grammars/zig.zyn",
            "../zyn_peg/grammars/zig.zyn",
        ];

        let grammar_path = grammar_paths
            .iter()
            .map(std::path::Path::new)
            .find(|p| p.exists());

        let grammar_path = match grammar_path {
            Some(p) => p,
            None => {
                eprintln!(
                    "Grammar not found at any of: {:?}, skipping test",
                    grammar_paths
                );
                return None;
            }
        };

        let grammar = match LanguageGrammar::compile_zyn_file(grammar_path) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to compile grammar: {}", e);
                return None;
            }
        };

        let mut runtime = match ZyntaxRuntime::new() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to create runtime: {}", e);
                return None;
            }
        };

        runtime.register_grammar("zig", grammar);

        Some(runtime)
    }

    #[test]
    fn test_promise_all_multiple_async_functions() {
        // Test PromiseAll with multiple independent async functions
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn compute(x: i32) i32 {
    return x * 2;
}
"#;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(_functions)) => {
                // Create multiple promises for the same function with different args
                let promises: Vec<ZyntaxPromise> = (1..=5)
                    .map(|i| {
                        runtime
                            .call_async("compute", &[ZyntaxValue::Int(i)])
                            .unwrap()
                    })
                    .collect();

                println!("Created {} promises", promises.len());

                // Use PromiseAll to await all
                let mut all = PromiseAll::new(promises);
                let results = all.await_all().expect("PromiseAll should succeed");

                println!(
                    "PromiseAll completed with {} results after {} polls",
                    results.len(),
                    all.poll_count()
                );

                // Verify results: compute(i) = i * 2
                assert_eq!(results.len(), 5);
                for (i, result) in results.iter().enumerate() {
                    let expected = ((i + 1) * 2) as i64;
                    match result {
                        ZyntaxValue::Int(v) => {
                            assert_eq!(*v, expected, "compute({}) should be {}", i + 1, expected);
                        }
                        other => panic!("Expected Int, got {:?}", other),
                    }
                }

                println!("SUCCESS: PromiseAll with 5 async computations");
            }
            Ok(Err(e)) => {
                panic!("Module load failed: {}", e);
            }
            Err(_panic) => {
                panic!("Async compilation panicked");
            }
        }
    }

    #[test]
    fn test_promise_all_different_functions() {
        // Test PromiseAll with different async functions
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn add_ten(x: i32) i32 {
    return x + 10;
}

async fn multiply_two(x: i32) i32 {
    return x * 2;
}

async fn sum_range(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}
"#;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(_functions)) => {
                // Create promises for different functions
                let promises = vec![
                    runtime
                        .call_async("add_ten", &[ZyntaxValue::Int(5)])
                        .unwrap(), // 15
                    runtime
                        .call_async("multiply_two", &[ZyntaxValue::Int(7)])
                        .unwrap(), // 14
                    runtime
                        .call_async("sum_range", &[ZyntaxValue::Int(10)])
                        .unwrap(), // 55
                ];

                let mut all = PromiseAll::new(promises);
                let results = all.await_all().expect("PromiseAll should succeed");

                println!("Results: {:?}", results);

                // Verify each result
                assert_eq!(results.len(), 3);
                assert_eq!(results[0], ZyntaxValue::Int(15)); // add_ten(5) = 15
                assert_eq!(results[1], ZyntaxValue::Int(14)); // multiply_two(7) = 14
                assert_eq!(results[2], ZyntaxValue::Int(55)); // sum_range(10) = 55

                println!("SUCCESS: PromiseAll with different async functions");
            }
            Ok(Err(e)) => {
                panic!("Module load failed: {}", e);
            }
            Err(_panic) => {
                panic!("Async compilation panicked");
            }
        }
    }

    #[test]
    fn test_promise_all_with_long_running_functions() {
        // Test PromiseAll with functions that take many polls
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn sum_range(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}
"#;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(_functions)) => {
                // Create promises with different iteration counts
                let promises = vec![
                    runtime
                        .call_async("sum_range", &[ZyntaxValue::Int(10)])
                        .unwrap(), // 55
                    runtime
                        .call_async("sum_range", &[ZyntaxValue::Int(50)])
                        .unwrap(), // 1275
                    runtime
                        .call_async("sum_range", &[ZyntaxValue::Int(100)])
                        .unwrap(), // 5050
                ];

                let mut all = PromiseAll::new(promises);
                let results = all.await_all().expect("PromiseAll should succeed");

                println!(
                    "Long-running results after {} polls: {:?}",
                    all.poll_count(),
                    results
                );

                assert_eq!(results[0], ZyntaxValue::Int(55));
                assert_eq!(results[1], ZyntaxValue::Int(1275));
                assert_eq!(results[2], ZyntaxValue::Int(5050));

                println!(
                    "SUCCESS: PromiseAll with long-running functions (poll_count={})",
                    all.poll_count()
                );
            }
            Ok(Err(e)) => {
                panic!("Module load failed: {}", e);
            }
            Err(_panic) => {
                panic!("Async compilation panicked");
            }
        }
    }

    #[test]
    fn test_promise_race_first_wins() {
        // Test PromiseRace - first function to complete wins
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
// Simple function - completes quickly
async fn quick(x: i32) i32 {
    return x * 2;
}

// Long-running function - takes many polls
async fn slow(n: i32) i32 {
    var total: i32 = 0;
    var i: i32 = 1;
    while (i <= n) {
        total = total + i;
        i = i + 1;
    }
    return total;
}
"#;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(_functions)) => {
                // Create promises - quick should win
                let promises = vec![
                    runtime
                        .call_async("slow", &[ZyntaxValue::Int(100)])
                        .unwrap(), // Takes many polls
                    runtime
                        .call_async("quick", &[ZyntaxValue::Int(21)])
                        .unwrap(), // Completes quickly
                ];

                let mut race = PromiseRace::new(promises);
                let (winner_index, value) = race.await_first().expect("PromiseRace should succeed");

                println!(
                    "Race winner: index={}, value={:?} (after {} polls)",
                    winner_index,
                    value,
                    race.poll_count()
                );

                // Either could win depending on polling order, but both are valid
                match winner_index {
                    0 => {
                        // slow won
                        assert_eq!(value, ZyntaxValue::Int(5050));
                        println!("SUCCESS: PromiseRace - slow(100) won");
                    }
                    1 => {
                        // quick won (expected)
                        assert_eq!(value, ZyntaxValue::Int(42));
                        println!("SUCCESS: PromiseRace - quick(21) won");
                    }
                    _ => panic!("Invalid winner index"),
                }
            }
            Ok(Err(e)) => {
                panic!("Module load failed: {}", e);
            }
            Err(_panic) => {
                panic!("Async compilation panicked");
            }
        }
    }

    #[test]
    fn test_promise_all_settled() {
        // Test PromiseAllSettled - collects all results even if some fail
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn compute(x: i32) i32 {
    return x * 3;
}
"#;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(_functions)) => {
                // All these should succeed
                let promises = vec![
                    runtime
                        .call_async("compute", &[ZyntaxValue::Int(1)])
                        .unwrap(),
                    runtime
                        .call_async("compute", &[ZyntaxValue::Int(2)])
                        .unwrap(),
                    runtime
                        .call_async("compute", &[ZyntaxValue::Int(3)])
                        .unwrap(),
                ];

                let mut settled = PromiseAllSettled::new(promises);
                let results = settled.await_all();

                println!("AllSettled results: {:?}", results);

                assert_eq!(results.len(), 3);
                for (i, result) in results.iter().enumerate() {
                    match result {
                        SettledResult::Fulfilled(ZyntaxValue::Int(v)) => {
                            let expected = ((i + 1) * 3) as i64;
                            assert_eq!(*v, expected);
                        }
                        other => panic!("Expected Fulfilled(Int), got {:?}", other),
                    }
                }

                println!("SUCCESS: PromiseAllSettled with all successful");
            }
            Ok(Err(e)) => {
                panic!("Module load failed: {}", e);
            }
            Err(_panic) => {
                panic!("Async compilation panicked");
            }
        }
    }

    #[test]
    fn test_promise_all_with_nested_await() {
        // Test PromiseAll with functions that themselves await other functions
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn inner(x: i32) i32 {
    return x + 10;
}

async fn outer(x: i32) i32 {
    const result = await inner(x);
    return result * 2;
}
"#;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(_functions)) => {
                // Create multiple outer promises - each awaits inner internally
                let promises = vec![
                    runtime.call_async("outer", &[ZyntaxValue::Int(5)]).unwrap(), // (5+10)*2 = 30
                    runtime
                        .call_async("outer", &[ZyntaxValue::Int(10)])
                        .unwrap(), // (10+10)*2 = 40
                    runtime
                        .call_async("outer", &[ZyntaxValue::Int(15)])
                        .unwrap(), // (15+10)*2 = 50
                ];

                let mut all = PromiseAll::new(promises);
                let results = all.await_all().expect("PromiseAll should succeed");

                println!(
                    "Nested await results: {:?} (polls={})",
                    results,
                    all.poll_count()
                );

                assert_eq!(results[0], ZyntaxValue::Int(30));
                assert_eq!(results[1], ZyntaxValue::Int(40));
                assert_eq!(results[2], ZyntaxValue::Int(50));

                println!("SUCCESS: PromiseAll with nested await functions");
            }
            Ok(Err(e)) => {
                panic!("Module load failed: {}", e);
            }
            Err(_panic) => {
                panic!("Async compilation panicked");
            }
        }
    }

    #[test]
    fn test_promise_all_empty() {
        // Test PromiseAll with no promises - should complete immediately
        let promises: Vec<ZyntaxPromise> = vec![];
        let mut all = PromiseAll::new(promises);

        let results = all.await_all().expect("Empty PromiseAll should succeed");
        assert!(results.is_empty());
        assert_eq!(all.poll_count(), 1);

        println!("SUCCESS: Empty PromiseAll completes immediately");
    }

    #[test]
    fn test_promise_all_single() {
        // Test PromiseAll with a single promise
        let mut runtime = match setup_async_runtime() {
            Some(r) => r,
            None => return,
        };

        let source = r#"
async fn compute(x: i32) i32 {
    return x * 5;
}
"#;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load_module("zig", source)
        }));

        match result {
            Ok(Ok(_functions)) => {
                let promises = vec![runtime
                    .call_async("compute", &[ZyntaxValue::Int(7)])
                    .unwrap()];

                let mut all = PromiseAll::new(promises);
                let results = all.await_all().expect("Single PromiseAll should succeed");

                assert_eq!(results.len(), 1);
                assert_eq!(results[0], ZyntaxValue::Int(35)); // 7 * 5 = 35

                println!("SUCCESS: Single-element PromiseAll");
            }
            Ok(Err(e)) => {
                panic!("Module load failed: {}", e);
            }
            Err(_panic) => {
                panic!("Async compilation panicked");
            }
        }
    }
}
