//! Async/Await Support for ZRTL Plugins
//!
//! This module enables **language frontend developers** to expose native async
//! functions that can be `await`ed from their custom languages built with Zyntax.
//!
//! # Use Case
//!
//! When you're building a custom language with a `.zyn` grammar and you want your
//! language to support async/await for calling native Rust functions:
//!
//! ```text
//! // Your custom language (e.g., MyLang)
//! async fn main() {
//!     let result = await fetch_data("https://api.example.com");  // Native async!
//!     print(result);
//! }
//! ```
//!
//! The `fetch_data` function is implemented in Rust as a ZRTL plugin, and your
//! language's async/await compiles to Zyntax's state machine format.
//!
//! # Zyntax Async ABI
//!
//! Native async functions return a `ZrtlPromise` struct (16 bytes):
//!
//! ```text
//! struct ZrtlPromise {
//!     state_machine: *mut u8,       // Pointer to native state
//!     poll_fn: fn(*mut u8) -> i64,  // Poll function
//! }
//!
//! // Poll result convention (returned by poll_fn):
//! // 0 = Pending (keep polling)
//! // positive = Ready(value)
//! // negative = Failed(error_code)
//! ```
//!
//! # Creating Native Async Functions
//!
//! ## Using the `#[zrtl_async]` Macro (Recommended)
//!
//! ```rust,ignore
//! use zrtl::prelude::*;
//! use zrtl_macros::zrtl_async;
//!
//! // This becomes an awaitable function in any Zyntax-based language
//! #[zrtl_async("$IO$read_file")]
//! pub async fn read_file(path: *const u8) -> i64 {
//!     // Real async I/O here
//!     let contents = tokio::fs::read_to_string(path_str).await;
//!     // Return handle or length
//!     contents.len() as i64
//! }
//! ```
//!
//! ## Manual State Machine (For Fine-Grained Control)
//!
//! ```rust,ignore
//! use zrtl::prelude::*;
//!
//! // Define your state machine
//! #[repr(C)]
//! struct HttpRequestState {
//!     header: StateMachineHeader,
//!     url_ptr: *const u8,
//!     response_buffer: *mut u8,
//!     bytes_received: usize,
//! }
//!
//! // Implement the poll function
//! extern "C" fn http_poll(state: *mut u8) -> i64 {
//!     let sm = unsafe { &mut *(state as *mut HttpRequestState) };
//!
//!     // Check if I/O is ready
//!     if let Some(data) = check_io_ready(sm.url_ptr) {
//!         sm.bytes_received = data.len();
//!         PollResult::Ready(sm.bytes_received as i64).to_abi()
//!     } else {
//!         PollResult::Pending.to_abi()
//!     }
//! }
//!
//! // Export the async function
//! #[no_mangle]
//! pub extern "C" fn http_get(url: *const u8) -> *const ZrtlPromise {
//!     let state = Box::new(HttpRequestState {
//!         header: StateMachineHeader::new(),
//!         url_ptr: url,
//!         response_buffer: std::ptr::null_mut(),
//!         bytes_received: 0,
//!     });
//!
//!     let promise = Box::new(ZrtlPromise {
//!         state_machine: Box::into_raw(state) as *mut u8,
//!         poll_fn: http_poll,
//!     });
//!
//!     Box::into_raw(promise)
//! }
//! ```
//!
//! # Integration with Language Frontends
//!
//! When your `.zyn` grammar compiles `await expr`, the generated code:
//! 1. Calls the async function, receiving a `*ZrtlPromise`
//! 2. Polls the promise until it returns non-zero
//! 3. Extracts the value or error from the result
//!
//! The async state machine compiler in `zyntax_compiler::async_support` handles
//! the guest-side state machine generation. This module handles the native side.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::time::{Duration, Instant};

/// Current task ID counter for wakers
static NEXT_TASK_ID: AtomicUsize = AtomicUsize::new(1);

// ============================================================================
// Poll Result
// ============================================================================

/// Result of polling an async operation
///
/// This follows the Zyntax async ABI convention:
/// - `Pending`: Operation is not complete, poll again
/// - `Ready(value)`: Operation completed successfully with a value
/// - `Failed(code)`: Operation failed with an error code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollResult {
    /// Operation is still in progress
    Pending,
    /// Operation completed with a value
    Ready(i64),
    /// Operation failed with error code
    Failed(i32),
}

impl PollResult {
    /// Convert to the i64 ABI format
    ///
    /// - 0 = Pending
    /// - positive = Ready(value)
    /// - negative = Failed(code)
    pub fn to_abi(self) -> i64 {
        match self {
            PollResult::Pending => 0,
            PollResult::Ready(v) => v,
            PollResult::Failed(code) => code as i64,
        }
    }

    /// Create from the i64 ABI format
    pub fn from_abi(value: i64) -> Self {
        if value == 0 {
            PollResult::Pending
        } else if value > 0 {
            PollResult::Ready(value)
        } else {
            PollResult::Failed(value as i32)
        }
    }

    /// Check if the result is pending
    pub fn is_pending(&self) -> bool {
        matches!(self, PollResult::Pending)
    }

    /// Check if the result is ready
    pub fn is_ready(&self) -> bool {
        matches!(self, PollResult::Ready(_))
    }

    /// Check if the result is failed
    pub fn is_failed(&self) -> bool {
        matches!(self, PollResult::Failed(_))
    }
}

// ============================================================================
// Promise Types (C ABI Compatible)
// ============================================================================

/// Promise layout for Zyntax async ABI
///
/// This struct is passed by pointer to/from async functions.
/// It contains the state machine and poll function.
#[repr(C)]
pub struct ZrtlPromise {
    /// Pointer to the state machine
    pub state_machine: *mut u8,
    /// Poll function: `fn(*mut u8) -> i64`
    pub poll_fn: unsafe extern "C" fn(*mut u8) -> i64,
}

impl ZrtlPromise {
    /// Create a new promise with a state machine and poll function
    ///
    /// # Safety
    /// The poll function must be valid and the state machine must have
    /// the correct layout expected by the poll function.
    pub const unsafe fn new(
        state_machine: *mut u8,
        poll_fn: unsafe extern "C" fn(*mut u8) -> i64,
    ) -> Self {
        Self {
            state_machine,
            poll_fn,
        }
    }

    /// Poll the promise
    ///
    /// # Safety
    /// Must only be called while the state machine is valid.
    pub unsafe fn poll(&mut self) -> PollResult {
        let result = (self.poll_fn)(self.state_machine);
        PollResult::from_abi(result)
    }

    /// Block until the promise resolves
    ///
    /// This spins on the promise until it completes.
    /// For real async execution, use `poll()` in an event loop.
    pub fn block_on(&mut self) -> Result<i64, i32> {
        loop {
            let result = unsafe { self.poll() };
            match result {
                PollResult::Pending => {
                    // Could add a yield here for better CPU usage
                    std::hint::spin_loop();
                }
                PollResult::Ready(value) => return Ok(value),
                PollResult::Failed(code) => return Err(code),
            }
        }
    }

    /// Block with timeout
    pub fn block_on_timeout(&mut self, timeout: Duration) -> Result<i64, PromiseError> {
        let start = Instant::now();
        loop {
            let result = unsafe { self.poll() };
            match result {
                PollResult::Pending => {
                    if start.elapsed() > timeout {
                        return Err(PromiseError::Timeout);
                    }
                    std::hint::spin_loop();
                }
                PollResult::Ready(value) => return Ok(value),
                PollResult::Failed(code) => return Err(PromiseError::Failed(code)),
            }
        }
    }
}

// SAFETY: Promise contains only raw pointers that are thread-safe when used correctly
unsafe impl Send for ZrtlPromise {}
unsafe impl Sync for ZrtlPromise {}

/// Errors from promise operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromiseError {
    /// Operation timed out
    Timeout,
    /// Operation failed with error code
    Failed(i32),
    /// Promise was null or invalid
    InvalidPromise,
}

impl std::fmt::Display for PromiseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromiseError::Timeout => write!(f, "Promise timed out"),
            PromiseError::Failed(code) => write!(f, "Promise failed with code {}", code),
            PromiseError::InvalidPromise => write!(f, "Invalid promise"),
        }
    }
}

impl std::error::Error for PromiseError {}

// ============================================================================
// State Machine Support
// ============================================================================

/// State for an async state machine
///
/// This represents the execution state of an async function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AsyncState {
    /// Initial state, not started
    Initial = 0,
    /// Resumed after first await
    Resume1 = 1,
    /// Resumed after second await
    Resume2 = 2,
    /// And so on...
    Resume3 = 3,
    Resume4 = 4,
    Resume5 = 5,
    Resume6 = 6,
    Resume7 = 7,
    /// Completed successfully
    Completed = 0xFFFFFFFE,
    /// Failed with error
    Failed = 0xFFFFFFFF,
}

impl AsyncState {
    /// Check if the state machine has finished
    pub fn is_finished(self) -> bool {
        matches!(self, AsyncState::Completed | AsyncState::Failed)
    }
}

/// Header for state machine structs
///
/// All async state machines start with this header.
/// The rest of the struct contains captured locals.
#[repr(C)]
pub struct StateMachineHeader {
    /// Current state (which await point we're at)
    pub state: u32,
    /// Reserved for alignment
    pub _reserved: u32,
}

impl StateMachineHeader {
    /// Create a new header in initial state
    pub const fn new() -> Self {
        Self {
            state: AsyncState::Initial as u32,
            _reserved: 0,
        }
    }

    /// Get the current async state
    pub fn async_state(&self) -> AsyncState {
        // SAFETY: We only store valid AsyncState values
        match self.state {
            0 => AsyncState::Initial,
            1 => AsyncState::Resume1,
            2 => AsyncState::Resume2,
            3 => AsyncState::Resume3,
            4 => AsyncState::Resume4,
            5 => AsyncState::Resume5,
            6 => AsyncState::Resume6,
            7 => AsyncState::Resume7,
            0xFFFFFFFE => AsyncState::Completed,
            0xFFFFFFFF => AsyncState::Failed,
            // For states > 7, treat as resume states
            _ => unsafe { std::mem::transmute(self.state.min(7)) },
        }
    }

    /// Set the state to completed
    pub fn set_completed(&mut self) {
        self.state = AsyncState::Completed as u32;
    }

    /// Set the state to failed
    pub fn set_failed(&mut self) {
        self.state = AsyncState::Failed as u32;
    }

    /// Advance to next state
    pub fn advance(&mut self) {
        if self.state < 0xFFFFFFFE {
            self.state += 1;
        }
    }
}

// ============================================================================
// Waker Implementation
// ============================================================================

/// A simple waker that does nothing (for synchronous polling)
pub fn noop_waker() -> Waker {
    // SAFETY: The vtable functions are correct for a no-op waker
    unsafe { Waker::from_raw(noop_raw_waker()) }
}

fn noop_raw_waker() -> RawWaker {
    RawWaker::new(std::ptr::null(), &NOOP_WAKER_VTABLE)
}

const NOOP_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |_| noop_raw_waker(), // clone
    |_| {},               // wake
    |_| {},               // wake_by_ref
    |_| {},               // drop
);

/// Create a waker context for polling
pub fn noop_context() -> Context<'static> {
    // Create a static waker (safe because noop waker has no state)
    static WAKER: std::sync::OnceLock<Waker> = std::sync::OnceLock::new();
    let waker = WAKER.get_or_init(noop_waker);
    Context::from_waker(waker)
}

// ============================================================================
// Future Adapter
// ============================================================================

/// Wrapper to use a Rust Future as a ZRTL async function
///
/// This allows you to write async Rust code and expose it to Zyntax.
pub struct FutureAdapter<F, T>
where
    F: Future<Output = T>,
{
    future: Pin<Box<F>>,
    result: Option<T>,
}

impl<F, T> FutureAdapter<F, T>
where
    F: Future<Output = T>,
{
    /// Create a new adapter for a future
    pub fn new(future: F) -> Self {
        Self {
            future: Box::pin(future),
            result: None,
        }
    }

    /// Poll the future
    pub fn poll(&mut self) -> Poll<&T> {
        if self.result.is_some() {
            return Poll::Ready(self.result.as_ref().unwrap());
        }

        let mut cx = noop_context();
        match self.future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => {
                self.result = Some(value);
                Poll::Ready(self.result.as_ref().unwrap())
            }
            Poll::Pending => Poll::Pending,
        }
    }

    /// Take the result (consumes the adapter)
    pub fn take_result(self) -> Option<T> {
        self.result
    }
}

// ============================================================================
// Promise Combinators
// ============================================================================

/// Collection of promises for parallel execution (like Promise.all)
pub struct PromiseAll {
    promises: Vec<ZrtlPromise>,
    results: Vec<Option<i64>>,
    poll_count: usize,
}

impl PromiseAll {
    /// Create a new PromiseAll with a collection of promises
    pub fn new(promises: Vec<ZrtlPromise>) -> Self {
        let len = promises.len();
        Self {
            promises,
            results: vec![None; len],
            poll_count: 0,
        }
    }

    /// Poll all promises, returns Ready when all are complete
    pub fn poll(&mut self) -> PollResult {
        self.poll_count += 1;

        let mut all_ready = true;

        for (i, promise) in self.promises.iter_mut().enumerate() {
            if self.results[i].is_some() {
                continue; // Already resolved
            }

            let result = unsafe { promise.poll() };
            match result {
                PollResult::Pending => {
                    all_ready = false;
                }
                PollResult::Ready(value) => {
                    self.results[i] = Some(value);
                }
                PollResult::Failed(code) => {
                    // Fail fast on first error
                    return PollResult::Failed(code);
                }
            }
        }

        if all_ready {
            // Return the number of completed promises (or could pack results differently)
            PollResult::Ready(self.results.len() as i64)
        } else {
            PollResult::Pending
        }
    }

    /// Get the results (only valid after poll returns Ready)
    pub fn results(&self) -> &[Option<i64>] {
        &self.results
    }

    /// Block until all promises resolve
    pub fn block_on(&mut self) -> Result<Vec<i64>, i32> {
        loop {
            match self.poll() {
                PollResult::Pending => {
                    std::hint::spin_loop();
                }
                PollResult::Ready(_) => {
                    return Ok(self.results.iter().filter_map(|r| *r).collect());
                }
                PollResult::Failed(code) => {
                    return Err(code);
                }
            }
        }
    }

    /// Block with timeout
    pub fn block_on_timeout(&mut self, timeout: Duration) -> Result<Vec<i64>, PromiseError> {
        let start = Instant::now();
        loop {
            match self.poll() {
                PollResult::Pending => {
                    if start.elapsed() > timeout {
                        return Err(PromiseError::Timeout);
                    }
                    std::hint::spin_loop();
                }
                PollResult::Ready(_) => {
                    return Ok(self.results.iter().filter_map(|r| *r).collect());
                }
                PollResult::Failed(code) => {
                    return Err(PromiseError::Failed(code));
                }
            }
        }
    }
}

/// Race multiple promises (like Promise.race)
pub struct PromiseRace {
    promises: Vec<ZrtlPromise>,
    winner: Option<(usize, i64)>,
    poll_count: usize,
}

impl PromiseRace {
    /// Create a new PromiseRace
    pub fn new(promises: Vec<ZrtlPromise>) -> Self {
        Self {
            promises,
            winner: None,
            poll_count: 0,
        }
    }

    /// Poll all promises, returns Ready when first completes
    pub fn poll(&mut self) -> PollResult {
        if let Some((_, value)) = self.winner {
            return PollResult::Ready(value);
        }

        self.poll_count += 1;

        for (i, promise) in self.promises.iter_mut().enumerate() {
            let result = unsafe { promise.poll() };
            match result {
                PollResult::Pending => continue,
                PollResult::Ready(value) => {
                    self.winner = Some((i, value));
                    return PollResult::Ready(value);
                }
                PollResult::Failed(code) => {
                    return PollResult::Failed(code);
                }
            }
        }

        PollResult::Pending
    }

    /// Get the winner index and value
    pub fn winner(&self) -> Option<(usize, i64)> {
        self.winner
    }

    /// Block until first promise resolves
    pub fn block_on(&mut self) -> Result<(usize, i64), i32> {
        loop {
            match self.poll() {
                PollResult::Pending => {
                    std::hint::spin_loop();
                }
                PollResult::Ready(_) => {
                    return Ok(self.winner.unwrap());
                }
                PollResult::Failed(code) => {
                    return Err(code);
                }
            }
        }
    }
}

/// Result of a settled promise (for PromiseAllSettled)
#[derive(Debug, Clone)]
pub enum SettledResult {
    /// Promise fulfilled with value
    Fulfilled(i64),
    /// Promise rejected with error code
    Rejected(i32),
}

/// Wait for all promises regardless of success/failure (like Promise.allSettled)
pub struct PromiseAllSettled {
    promises: Vec<ZrtlPromise>,
    results: Vec<Option<SettledResult>>,
    poll_count: usize,
}

impl PromiseAllSettled {
    /// Create a new PromiseAllSettled
    pub fn new(promises: Vec<ZrtlPromise>) -> Self {
        let len = promises.len();
        Self {
            promises,
            results: vec![None; len],
            poll_count: 0,
        }
    }

    /// Poll all promises, returns Ready when all are settled
    pub fn poll(&mut self) -> bool {
        self.poll_count += 1;

        let mut all_settled = true;

        for (i, promise) in self.promises.iter_mut().enumerate() {
            if self.results[i].is_some() {
                continue;
            }

            let result = unsafe { promise.poll() };
            match result {
                PollResult::Pending => {
                    all_settled = false;
                }
                PollResult::Ready(value) => {
                    self.results[i] = Some(SettledResult::Fulfilled(value));
                }
                PollResult::Failed(code) => {
                    self.results[i] = Some(SettledResult::Rejected(code));
                }
            }
        }

        all_settled
    }

    /// Get the results (only valid after poll returns true)
    pub fn results(&self) -> &[Option<SettledResult>] {
        &self.results
    }

    /// Block until all promises settle
    pub fn block_on(&mut self) -> Vec<SettledResult> {
        while !self.poll() {
            std::hint::spin_loop();
        }
        self.results.iter().filter_map(|r| r.clone()).collect()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate a unique task ID
pub fn next_task_id() -> usize {
    NEXT_TASK_ID.fetch_add(1, Ordering::SeqCst)
}

/// A simple yielding future that completes after one poll
pub struct YieldOnce {
    yielded: bool,
}

impl YieldOnce {
    /// Create a new YieldOnce future
    pub const fn new() -> Self {
        Self { yielded: false }
    }
}

impl Future for YieldOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            Poll::Pending
        }
    }
}

/// Yield once (like tokio::task::yield_now but synchronous)
pub fn yield_once() -> YieldOnce {
    YieldOnce::new()
}

/// A timer future that completes after a duration
pub struct Timer {
    deadline: Instant,
}

impl Timer {
    /// Create a new timer that completes after the given duration
    pub fn after(duration: Duration) -> Self {
        Self {
            deadline: Instant::now() + duration,
        }
    }
}

impl Future for Timer {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if Instant::now() >= self.deadline {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// Create a timer future
pub fn sleep(duration: Duration) -> Timer {
    Timer::after(duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poll_result_abi() {
        assert_eq!(PollResult::Pending.to_abi(), 0);
        assert_eq!(PollResult::Ready(42).to_abi(), 42);
        assert_eq!(PollResult::Failed(-1).to_abi(), -1);

        assert_eq!(PollResult::from_abi(0), PollResult::Pending);
        assert_eq!(PollResult::from_abi(42), PollResult::Ready(42));
        assert_eq!(PollResult::from_abi(-1), PollResult::Failed(-1));
    }

    #[test]
    fn test_state_machine_header() {
        let mut header = StateMachineHeader::new();
        assert_eq!(header.async_state(), AsyncState::Initial);

        header.advance();
        assert_eq!(header.state, 1);

        header.set_completed();
        assert!(header.async_state().is_finished());
    }

    #[test]
    fn test_yield_once() {
        let mut future = yield_once();
        let mut pinned = Pin::new(&mut future);
        let mut cx = noop_context();

        // First poll returns Pending
        assert!(pinned.as_mut().poll(&mut cx).is_pending());

        // Second poll returns Ready
        assert!(pinned.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn test_noop_waker() {
        let waker = noop_waker();
        waker.wake_by_ref(); // Should not panic
        drop(waker);
    }
}
