//! Promises: what a compiled async function hands back.
//!
//! A promise owns the state machine a compiled `async fn` was lowered
//! to, drives it when polled, and carries the combinators built on top
//! of that: all, race, and all-settled.

use super::types::{NativeSignature, RuntimeError, RuntimeResult};
use super::{native_call::call_with_signature, ZyntaxRuntime};
use crate::convert::FromZyntax;
use crate::value::ZyntaxValue;
use std::sync::{Arc, Mutex};
use zyntax_compiler::zrtl::DynamicValue;

/// A promise representing an async operation
///
/// `ZyntaxPromise` wraps a Zyntax async function call and provides methods
/// to await or poll the result.
///
/// # States
///
/// - `Pending`: The operation is still in progress
/// - `Ready`: The operation completed successfully with a value
/// - `Failed`: The operation failed with an error
///
/// # Example
///
/// ```ignore
/// let promise = runtime.call_async("fetch", &[url.into()])?;
///
/// // Block until complete
/// let result: String = promise.await_result()?;
///
/// // Or poll manually
/// loop {
///     match promise.poll() {
///         PromiseState::Ready(value) => break,
///         PromiseState::Pending => std::thread::yield_now(),
///         PromiseState::Failed(err) => return Err(err),
///     }
/// }
/// ```
pub struct ZyntaxPromise {
    state: Arc<Mutex<PromiseInner>>,
}

/// Poll result from async state machine
///
/// This matches the Zyntax async ABI where poll functions return a discriminated union.
#[repr(C, u8)]
#[derive(Clone, Debug)]
pub enum AsyncPollResult {
    /// Still pending, needs more polls
    Pending = 0,
    /// Completed with a value (the DynamicValue)
    Ready(DynamicValue) = 1,
    /// Failed with an error message
    Failed(*const u8, usize) = 2, // (ptr, len) for error string
}

struct PromiseInner {
    /// Function pointer for creating the state machine
    init_fn: *const u8,
    /// Poll function pointer (once state machine is created)
    poll_fn: Option<*const u8>,
    /// Arguments to pass
    args: Vec<DynamicValue>,
    /// Current state
    state: PromiseState,
    /// State machine pointer (for Zyntax async functions)
    state_machine: Option<*mut u8>,
    /// Ready queue for waker integration
    ready_queue: Arc<Mutex<std::collections::VecDeque<usize>>>,
    /// Task ID for waker
    task_id: usize,
    /// Poll count for timeout detection
    poll_count: usize,
    /// Waker for Rust Future integration
    waker: Option<std::task::Waker>,
    /// Thread that ran this task. The tables that can still name the
    /// state machine are thread-local, so only this thread can tell
    /// whether the region is free to release.
    owner_thread: std::thread::ThreadId,
}

impl Drop for PromiseInner {
    fn drop(&mut self) {
        // The state machine is a `malloc` from the async entry function
        // and nothing else releases it. It can go once the task is
        // finished and no async table still names it: a parked timer, a
        // latched completion, or a handler/performer pairing all mean
        // something can still poll it.
        let Some(sm) = self.state_machine else {
            return;
        };
        if matches!(self.state, PromiseState::Pending) {
            return;
        }
        if std::thread::current().id() != self.owner_thread {
            return;
        }
        // A completion latched for this task is ours and nobody can read
        // it now, so clear it. Leaving it would keep the map growing and,
        // once the region below is freed, let a later allocation landing
        // on the same address look already-complete.
        let _ = crate::host_futures::take_sm_completion(sm);
        if crate::host_futures::sm_is_referenced(sm) {
            return;
        }
        // SAFETY: the task is finished, nothing on this thread can reach
        // the region, and it came from the entry function's `malloc`.
        unsafe { crate::effect_runtime::free_handler_state(sm) };
    }
}

// SAFETY: Promise state is protected by mutex
unsafe impl Send for PromiseInner {}
unsafe impl Sync for PromiseInner {}

/// The state of a promise
#[derive(Debug, Clone)]
pub enum PromiseState {
    /// The operation is still in progress
    Pending,
    /// The operation completed with a value
    Ready(ZyntaxValue),
    /// The operation failed with an error
    Failed(String),
    /// The operation was cancelled
    Cancelled,
}

/// Global task ID counter for promise wakers
static NEXT_TASK_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl ZyntaxPromise {
    /// Create a new promise for an async function call
    fn new(func_ptr: *const u8, args: Vec<DynamicValue>) -> Self {
        let task_id = NEXT_TASK_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self {
            state: Arc::new(Mutex::new(PromiseInner {
                init_fn: func_ptr,
                poll_fn: None,
                args,
                state: PromiseState::Pending,
                state_machine: None,
                ready_queue: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                task_id,
                poll_count: 0,
                waker: None,
                owner_thread: std::thread::current().id(),
            })),
        }
    }

    /// Create a promise from a function that returns `*Promise<T>`
    ///
    /// The new Promise-based async ABI:
    /// - `async fn foo(x: i32) -> i32` compiles to `fn foo(x: i32) -> *Promise<i32>`
    /// - Promise is a struct on the stack: `{state_machine: *mut u8, poll_fn: fn(*mut u8) -> i64}`
    /// - Calling the function allocates state machine and returns pointer to Promise
    ///
    /// Uses the provided signature to properly invoke the function with the correct
    /// number and types of arguments.
    pub unsafe fn from_async_call(
        func_ptr: *const u8,
        args: Vec<DynamicValue>,
        signature: &NativeSignature,
    ) -> Self {
        let task_id = NEXT_TASK_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Call the function to get the Promise pointer using signature-based dynamic dispatch
        // Promise layout at pointer: offset 0 = state_machine (8 bytes), offset 8 = poll_fn (8 bytes)
        let (state_machine, poll_fn) = unsafe {
            let promise_ptr: *const u8 = call_with_signature(func_ptr, &args, signature);

            if promise_ptr.is_null() {
                (std::ptr::null_mut(), std::ptr::null())
            } else {
                // Read the Promise struct from the pointer
                // Promise layout: {state_machine: *mut u8, poll_fn: fn(*mut u8) -> i64}
                let state_machine = *(promise_ptr as *const *mut u8);
                let poll_fn = *((promise_ptr as *const u8).offset(8) as *const *const u8);
                // The entry function mallocs this 16-byte struct purely to
                // hand back the pair. Both fields are copied out above and
                // the pointer goes no further, so it is released here
                // rather than lost. The state machine it named is a
                // separate allocation and is NOT released.
                crate::effect_runtime::free_handler_state(promise_ptr as *mut u8);
                (state_machine, poll_fn)
            }
        };

        Self {
            state: Arc::new(Mutex::new(PromiseInner {
                init_fn: func_ptr, // Keep for reference
                poll_fn: if poll_fn.is_null() {
                    None
                } else {
                    Some(poll_fn)
                },
                args,
                state: PromiseState::Pending,
                state_machine: if state_machine.is_null() {
                    None
                } else {
                    Some(state_machine)
                },
                ready_queue: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                task_id,
                poll_count: 0,
                waker: None,
                owner_thread: std::thread::current().id(),
            })),
        }
    }

    /// Create a new promise with separate constructor and poll functions
    ///
    /// This follows the legacy Zyntax async ABI where:
    /// - `init_fn`: `{fn}_new(params...) -> *mut StateMachine` - constructor
    /// - `poll_fn`: `async_wrapper(self: *StateMachine, cx: *Context) -> Poll` - poll function
    ///
    /// See `crates/compiler/src/async_support.rs` for the full async ABI.
    pub fn with_poll_fn(init_fn: *const u8, poll_fn: *const u8, args: Vec<DynamicValue>) -> Self {
        let task_id = NEXT_TASK_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self {
            state: Arc::new(Mutex::new(PromiseInner {
                init_fn,
                poll_fn: Some(poll_fn),
                args,
                state: PromiseState::Pending,
                state_machine: None,
                ready_queue: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                task_id,
                poll_count: 0,
                waker: None,
                owner_thread: std::thread::current().id(),
            })),
        }
    }

    /// Set the poll function for this promise
    ///
    /// Call this before polling if the promise was created with just an init function.
    pub fn set_poll_fn(&self, poll_fn: *const u8) {
        let mut inner = self.state.lock().unwrap();
        inner.poll_fn = Some(poll_fn);
    }

    /// Poll the promise for completion
    ///
    /// Returns the current state without blocking.
    ///
    /// # Async ABI
    ///
    /// Zyntax async functions follow this ABI:
    /// 1. `init_fn(args...) -> *mut StateMachine` - Creates the state machine
    /// 2. `poll_fn(state_machine: *mut u8, waker_data: *const u8) -> AsyncPollResult`
    ///
    /// The poll function advances the state machine until it yields or completes.
    pub fn poll(&self) -> PromiseState {
        let mut inner = self.state.lock().unwrap();

        // If already complete or cancelled, return the state
        match &inner.state {
            PromiseState::Ready(_) | PromiseState::Failed(_) | PromiseState::Cancelled => {
                return inner.state.clone();
            }
            PromiseState::Pending => {}
        }

        inner.poll_count += 1;

        // Try to advance the state machine
        if let Some(state_machine) = inner.state_machine {
            if let Some(poll_fn) = inner.poll_fn {
                unsafe {
                    // Call the poll function on the state machine
                    // New ABI: poll(state_machine: *mut u8) -> i64
                    // Return value: 0 = Pending, positive = Ready(value), negative = Failed

                    // Drive the state machine one step. Under the cooperative
                    // executor the host bridges park a timer and return, so
                    // this poll advances to the next suspension (Pending) or
                    // to completion (non-zero); it never runs the whole task
                    // inline. Completion of a parked task happens later via
                    // `resolve_future` in the executor's timer loop.
                    let f: extern "C" fn(*mut u8) -> i64 = std::mem::transmute(poll_fn);
                    let result = f(state_machine);

                    if result == 0 {
                        // Pending - state remains unchanged
                    } else if result < 0 {
                        // Negative value indicates failure
                        inner.state = PromiseState::Failed(format!(
                            "Async operation failed with code {}",
                            result
                        ));
                    } else {
                        // Ready with value
                        // For i64/i32 returns, the value is in the result directly
                        inner.state = PromiseState::Ready(ZyntaxValue::Int(result));
                    }
                }
            } else {
                // No poll function available, mark as complete with void
                // This handles sync functions wrapped as async
                inner.state = PromiseState::Ready(ZyntaxValue::Void);
            }
        } else {
            // Initialize the state machine on first poll
            // Initialize state machine on first poll (legacy path)
            unsafe {
                // The init function creates the state machine and returns a pointer to it.
                // ABI: init_fn(args...) -> *mut StateMachine
                //
                // For Zyntax async functions, the constructor takes the same parameters
                // as the original async function and returns a state machine struct.

                if inner.init_fn.is_null() {
                    inner.state = PromiseState::Failed("Null async function pointer".to_string());
                    return inner.state.clone();
                }

                // Call the init function with the provided arguments
                // The state machine struct is returned by value (as a struct), not as a pointer
                // For Cranelift, structs are returned via pointer in the first hidden argument
                // We'll allocate space and pass the pointer
                let state_machine: *mut u8 = match inner.args.len() {
                    0 => {
                        // No arguments - allocate state machine on stack and call init()
                        // Allocate a fixed-size buffer for the state machine (state: u32 + local_x: i32 = 8 bytes)
                        let buffer = Box::into_raw(Box::new([0u8; 64])) as *mut u8;
                        let f: extern "C" fn(*mut u8) = std::mem::transmute(inner.init_fn);
                        f(buffer);
                        buffer
                    }
                    1 => {
                        // Single argument (common case: async fn foo(x: i32) ...)
                        // Allocate space for state machine, pass it as first arg (sret), original arg as second
                        let arg0 = inner.args[0]
                            .get_i32()
                            .map(|i| i as i64)
                            .or_else(|| inner.args[0].get_i64())
                            .unwrap_or(0i64);
                        let buffer = Box::into_raw(Box::new([0u8; 64])) as *mut u8;
                        let f: extern "C" fn(*mut u8, i64) = std::mem::transmute(inner.init_fn);
                        f(buffer, arg0);
                        buffer
                    }
                    _ => {
                        // Multiple arguments not yet supported
                        inner.state = PromiseState::Failed(format!(
                            "Async functions with {} arguments not yet supported",
                            inner.args.len()
                        ));
                        return inner.state.clone();
                    }
                };

                if state_machine.is_null() {
                    inner.state =
                        PromiseState::Failed("Failed to create async state machine".to_string());
                    return inner.state.clone();
                }

                inner.state_machine = Some(state_machine);

                // The async ABI in Zyntax generates two functions:
                // 1. Constructor: `{fn}_new(params...) -> StateMachine` (init_fn)
                // 2. Poll: `{fn}_poll(self: *StateMachine, cx: *Context) -> i64`
                //    where 0 = Pending, non-zero = Ready(value)
            }
        }

        inner.state.clone()
    }

    /// Poll with a maximum number of iterations
    ///
    /// This is useful for avoiding infinite loops when the async function
    /// might be stuck or taking too long.
    pub fn poll_with_limit(&self, max_polls: usize) -> PromiseState {
        let inner = self.state.lock().unwrap();
        if inner.poll_count >= max_polls {
            drop(inner);
            let mut inner = self.state.lock().unwrap();
            inner.state = PromiseState::Failed(format!(
                "Async operation timed out after {} polls",
                max_polls
            ));
            return inner.state.clone();
        }
        drop(inner);
        self.poll()
    }

    /// Block until the promise completes and return the result
    pub fn await_result<T: FromZyntax>(&self) -> RuntimeResult<T> {
        loop {
            match self.poll() {
                PromiseState::Pending => {
                    // Yield to allow other work
                    std::thread::yield_now();
                }
                PromiseState::Ready(value) => {
                    return T::from_zyntax(value).map_err(RuntimeError::from);
                }
                PromiseState::Failed(err) => {
                    return Err(RuntimeError::Promise(err));
                }
                PromiseState::Cancelled => {
                    return Err(RuntimeError::Promise("Promise was cancelled".to_string()));
                }
            }
        }
    }

    /// Block until the promise completes and return the raw value.
    /// Cooperatively drives the shared timer queue (native); on wasm the
    /// JS event loop drives resolution, so it just spins on `poll()`.
    pub fn await_raw(&self) -> RuntimeResult<ZyntaxValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            drive_until(std::slice::from_ref(self), None, |ps| ps[0].is_complete());
        }
        #[cfg(target_arch = "wasm32")]
        loop {
            if !matches!(self.poll(), PromiseState::Pending) {
                break;
            }
            std::thread::yield_now();
        }
        match self.state() {
            PromiseState::Ready(value) => Ok(value),
            PromiseState::Failed(err) => Err(RuntimeError::Promise(err)),
            PromiseState::Cancelled => {
                Err(RuntimeError::Promise("Promise was cancelled".to_string()))
            }
            PromiseState::Pending => {
                Err(RuntimeError::Promise("Task did not complete".to_string()))
            }
        }
    }

    /// Check if the promise is complete
    pub fn is_complete(&self) -> bool {
        let inner = self.state.lock().unwrap();
        !matches!(inner.state, PromiseState::Pending)
    }

    /// Check if the promise is pending
    pub fn is_pending(&self) -> bool {
        let inner = self.state.lock().unwrap();
        matches!(inner.state, PromiseState::Pending)
    }

    /// Check if the promise was cancelled
    pub fn is_cancelled(&self) -> bool {
        let inner = self.state.lock().unwrap();
        matches!(inner.state, PromiseState::Cancelled)
    }

    /// Cancel the promise
    ///
    /// Returns `true` if the promise was successfully cancelled (was pending),
    /// `false` if the promise was already complete or cancelled.
    ///
    /// Once cancelled, any subsequent polls will return `PromiseState::Cancelled`.
    /// Any code waiting on this promise will receive a cancellation error.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let promise = runtime.call_async("slow_task", &[])?;
    ///
    /// // Cancel if taking too long
    /// std::thread::sleep(std::time::Duration::from_secs(1));
    /// if promise.is_pending() {
    ///     promise.cancel();
    /// }
    /// ```
    pub fn cancel(&self) -> bool {
        let mut inner = self.state.lock().unwrap();
        if matches!(inner.state, PromiseState::Pending) {
            inner.state = PromiseState::Cancelled;
            // Wake any waiting futures
            if let Some(waker) = inner.waker.take() {
                waker.wake();
            }
            true
        } else {
            false
        }
    }

    /// Get the current state without polling
    pub fn state(&self) -> PromiseState {
        self.state.lock().unwrap().state.clone()
    }

    /// Block until the promise completes with a timeout.
    ///
    /// Returns `Err` if the timeout is exceeded. On timeout the task is
    /// cancelled and its parked future/timer torn down so the executor
    /// stops driving it.
    pub fn await_with_timeout(&self, timeout: std::time::Duration) -> RuntimeResult<ZyntaxValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let deadline = web_time::Instant::now() + timeout;
            let completed = drive_until(std::slice::from_ref(self), Some(deadline), |ps| {
                ps[0].is_complete()
            });
            if !completed {
                self.cancel();
                crate::host_futures::deregister_task(0);
                return Err(RuntimeError::Promise(format!(
                    "Async operation timed out after {:?}",
                    timeout
                )));
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let start = web_time::Instant::now();
            loop {
                if start.elapsed() > timeout {
                    return Err(RuntimeError::Promise(format!(
                        "Async operation timed out after {:?}",
                        timeout
                    )));
                }
                if !matches!(self.poll(), PromiseState::Pending) {
                    break;
                }
                std::thread::yield_now();
            }
        }
        match self.state() {
            PromiseState::Ready(value) => Ok(value),
            PromiseState::Failed(err) => Err(RuntimeError::Promise(err)),
            PromiseState::Cancelled => {
                Err(RuntimeError::Promise("Promise was cancelled".to_string()))
            }
            PromiseState::Pending => {
                Err(RuntimeError::Promise("Task did not complete".to_string()))
            }
        }
    }

    /// Get the number of times this promise has been polled
    pub fn poll_count(&self) -> usize {
        self.state.lock().unwrap().poll_count
    }

    /// Chain another operation to run when this promise completes
    pub fn then<F>(&self, f: F) -> ZyntaxPromise
    where
        F: FnOnce(ZyntaxValue) -> ZyntaxValue + Send + 'static,
    {
        let source = self.state.clone();
        let task_id = NEXT_TASK_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Create a new promise that depends on this one
        let new_promise = ZyntaxPromise {
            state: Arc::new(Mutex::new(PromiseInner {
                init_fn: std::ptr::null(),
                poll_fn: None,
                args: vec![],
                state: PromiseState::Pending,
                state_machine: None,
                ready_queue: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                task_id,
                poll_count: 0,
                waker: None,
                owner_thread: std::thread::current().id(),
            })),
        };

        let target = new_promise.state.clone();

        // Spawn a thread to wait for completion and run the callback
        std::thread::spawn(move || loop {
            let source_state = source.lock().unwrap().state.clone();
            match source_state {
                PromiseState::Ready(value) => {
                    let result = f(value);
                    target.lock().unwrap().state = PromiseState::Ready(result);
                    break;
                }
                PromiseState::Failed(err) => {
                    target.lock().unwrap().state = PromiseState::Failed(err);
                    break;
                }
                PromiseState::Cancelled => {
                    target.lock().unwrap().state = PromiseState::Cancelled;
                    break;
                }
                PromiseState::Pending => {
                    std::thread::yield_now();
                }
            }
        });

        new_promise
    }

    /// Handle errors from this promise
    pub fn catch<F>(&self, f: F) -> ZyntaxPromise
    where
        F: FnOnce(String) -> ZyntaxValue + Send + 'static,
    {
        let source = self.state.clone();
        let task_id = NEXT_TASK_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let new_promise = ZyntaxPromise {
            state: Arc::new(Mutex::new(PromiseInner {
                init_fn: std::ptr::null(),
                poll_fn: None,
                args: vec![],
                state: PromiseState::Pending,
                state_machine: None,
                ready_queue: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                task_id,
                poll_count: 0,
                waker: None,
                owner_thread: std::thread::current().id(),
            })),
        };

        let target = new_promise.state.clone();

        std::thread::spawn(move || {
            loop {
                let source_state = source.lock().unwrap().state.clone();
                match source_state {
                    PromiseState::Ready(value) => {
                        target.lock().unwrap().state = PromiseState::Ready(value);
                        break;
                    }
                    PromiseState::Failed(err) => {
                        let result = f(err);
                        target.lock().unwrap().state = PromiseState::Ready(result);
                        break;
                    }
                    PromiseState::Cancelled => {
                        // For catch, treat cancellation as an error to recover from
                        let result = f("Promise was cancelled".to_string());
                        target.lock().unwrap().state = PromiseState::Ready(result);
                        break;
                    }
                    PromiseState::Pending => {
                        std::thread::yield_now();
                    }
                }
            }
        });

        new_promise
    }
}

impl Clone for ZyntaxPromise {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

/// Cooperatively drive `promises` (each stamped with its slice index as
/// task id), advancing them via the shared timer queue, until `done`
/// returns true or `deadline` (if any) passes. Returns true if `done` was
/// reached, false on timeout. A task that becomes `Cancelled` mid-drive
/// has its parked futures/timers torn down so the executor stops driving
/// its (now dead) state machine.
///
/// This is the single cooperative core under `await_raw`, `drive_tasks`,
/// the timeout variants, and the `Promise.all`/`Promise.race` combinators.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn drive_until(
    promises: &[ZyntaxPromise],
    deadline: Option<web_time::Instant>,
    mut done: impl FnMut(&[ZyntaxPromise]) -> bool,
) -> bool {
    use crate::host_futures::{
        deregister_task, drive_next_timer_with_task, next_timer_deadline, set_current_task_id,
        ResolveOutcome,
    };

    // Drive one task's poll, stamped with its id and bracketed by its
    // handler-stack segment (so a `with`-block it opens across an await
    // stays isolated from other tasks).
    let poll_task = |i: usize, p: &ZyntaxPromise| {
        set_current_task_id(i as i64);
        let baseline = crate::effect_runtime::task_handler_enter(i as i64);
        let _ = p.poll();
        crate::effect_runtime::task_handler_leave(i as i64, baseline);
    };

    // Mark any promise whose SM was completed by a NESTED resume (an async
    // handler's `k(v)` drove a parked performer to Ready from inside the
    // executor's drive of the handler). Its top-level poll returned Pending
    // and it has no timer of its own, so the executor would otherwise never
    // finish it — and must NOT re-poll it (the finished SM would re-enter).
    let harvest = |promises: &[ZyntaxPromise]| {
        for p in promises.iter() {
            let mut inner = p.state.lock().unwrap();
            if matches!(inner.state, PromiseState::Pending) {
                if let Some(sm) = inner.state_machine {
                    if let Some(v) = crate::host_futures::take_sm_completion(sm) {
                        inner.state = PromiseState::Ready(ZyntaxValue::Int(v));
                    }
                }
            }
        }
    };

    // Initial drive: poll each task once (parks its first `await` timer,
    // stamped with its index, or completes synchronously).
    for (i, p) in promises.iter().enumerate() {
        poll_task(i, p);
    }
    harvest(promises);

    loop {
        // Tear down the parking of any newly-cancelled tasks.
        for (i, p) in promises.iter().enumerate() {
            if p.is_cancelled() {
                deregister_task(i as i64);
            }
        }
        harvest(promises);
        if done(promises) {
            return true;
        }
        if let Some(dl) = deadline {
            if web_time::Instant::now() >= dl {
                return false;
            }
        }

        match next_timer_deadline() {
            Some(td) => {
                // If the timeout fires before the next timer, wait it out
                // and report timeout rather than driving past the deadline.
                if let Some(dl) = deadline {
                    if td > dl {
                        let now = web_time::Instant::now();
                        if dl > now {
                            std::thread::sleep(dl - now);
                        }
                        return false;
                    }
                }
                if let Some((task_id, ResolveOutcome::Ready(v))) = drive_next_timer_with_task() {
                    if let Some(p) = promises.get(task_id as usize) {
                        let mut inner = p.state.lock().unwrap();
                        if matches!(inner.state, PromiseState::Pending) {
                            inner.state = PromiseState::Ready(ZyntaxValue::Int(v));
                        }
                    }
                }
            }
            None => {
                // No parked timers, yet a task is unfinished. A real
                // async SM always parks a timer, so this only happens for
                // a task that advances purely by re-polling (e.g. a
                // multi-poll simulated SM in tests) or one genuinely stuck
                // waiting on an external event. Re-poll the pending ones
                // and yield — the loop's `done`/`deadline` checks terminate
                // it (matching the pre-cooperative spin behaviour).
                for (i, p) in promises.iter().enumerate() {
                    if p.is_pending() {
                        poll_task(i, p);
                    }
                }
                std::thread::yield_now();
            }
        }
    }
}

/// `Promise.all` completion predicate: done when any child failed
/// (fast-fail) or all children are complete.
fn promise_all_done(ps: &[ZyntaxPromise]) -> bool {
    ps.iter()
        .any(|p| matches!(p.state(), PromiseState::Failed(_)))
        || ps.iter().all(|p| p.is_complete())
}

/// Collect the values of a completed promise group, fast-failing on the
/// first failure / cancellation / unfinished task.
fn collect_all(ps: &[ZyntaxPromise]) -> RuntimeResult<Vec<ZyntaxValue>> {
    let mut values = Vec::with_capacity(ps.len());
    for p in ps {
        match p.state() {
            PromiseState::Ready(v) => values.push(v),
            PromiseState::Failed(e) => return Err(RuntimeError::Promise(e)),
            PromiseState::Cancelled => {
                return Err(RuntimeError::Promise("Promise was cancelled".to_string()))
            }
            PromiseState::Pending => {
                return Err(RuntimeError::Promise("Task did not complete".to_string()))
            }
        }
    }
    Ok(values)
}

/// Drive multiple async tasks cooperatively to completion, interleaving
/// them via the shared timer queue: while one task is parked on its
/// `await` timer, another advances. Returns each task's result in order.
#[cfg(not(target_arch = "wasm32"))]
pub fn drive_tasks(promises: &[ZyntaxPromise]) -> Vec<RuntimeResult<ZyntaxValue>> {
    drive_until(promises, None, |ps| ps.iter().all(|p| p.is_complete()));
    promises
        .iter()
        .map(|p| match p.state() {
            PromiseState::Ready(v) => Ok(v),
            PromiseState::Failed(e) => Err(RuntimeError::Promise(e)),
            PromiseState::Cancelled => {
                Err(RuntimeError::Promise("Promise was cancelled".to_string()))
            }
            PromiseState::Pending => {
                Err(RuntimeError::Promise("Task did not complete".to_string()))
            }
        })
        .collect()
}

// ============================================================================
// Promise Combinators (Promise.all, Promise.race, etc.)
// ============================================================================

/// Result of awaiting multiple promises in parallel
///
/// Similar to JavaScript's `Promise.all()`, this collects the results of
/// multiple async operations that run concurrently.
#[derive(Debug, Clone)]
pub enum PromiseAllState {
    /// All promises are still pending or some are in progress
    Pending,
    /// All promises completed successfully with their values (in order)
    AllReady(Vec<ZyntaxValue>),
    /// At least one promise failed (first failure encountered)
    Failed(String),
}

/// Await multiple promises in parallel, similar to JavaScript's `Promise.all()`
///
/// This polls all promises concurrently and resolves when ALL promises complete.
/// If any promise fails, the entire operation fails with the first error.
///
/// # Example
///
/// ```rust,ignore
/// use zyntax_embed::{ZyntaxRuntime, ZyntaxValue, PromiseAll};
///
/// // Create multiple async calls
/// let promises = vec![
///     runtime.call_async("compute", &[ZyntaxValue::Int(1)])?,
///     runtime.call_async("compute", &[ZyntaxValue::Int(2)])?,
///     runtime.call_async("compute", &[ZyntaxValue::Int(3)])?,
/// ];
///
/// // Wait for all to complete
/// let mut all = PromiseAll::new(promises);
/// let results = all.await_all()?;
/// // results = [result1, result2, result3]
/// ```
pub struct PromiseAll {
    promises: Vec<ZyntaxPromise>,
    poll_count: usize,
}

impl PromiseAll {
    /// Create a new PromiseAll from a vector of promises
    pub fn new(promises: Vec<ZyntaxPromise>) -> Self {
        Self {
            promises,
            poll_count: 0,
        }
    }

    /// Create a PromiseAll from an iterator of promises
    pub fn from_iter<I: IntoIterator<Item = ZyntaxPromise>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }

    /// Poll all promises once, advancing each state machine
    ///
    /// Returns the combined state of all promises.
    pub fn poll(&mut self) -> PromiseAllState {
        self.poll_count += 1;

        let mut all_ready = true;
        let mut results = Vec::with_capacity(self.promises.len());

        for promise in &self.promises {
            match promise.poll() {
                PromiseState::Pending => {
                    all_ready = false;
                    results.push(ZyntaxValue::Void); // Placeholder
                }
                PromiseState::Ready(value) => {
                    results.push(value);
                }
                PromiseState::Failed(err) => {
                    // Fast-fail on first error
                    return PromiseAllState::Failed(err);
                }
                PromiseState::Cancelled => {
                    // Fast-fail on cancellation
                    return PromiseAllState::Failed("Promise was cancelled".to_string());
                }
            }
        }

        if all_ready {
            PromiseAllState::AllReady(results)
        } else {
            PromiseAllState::Pending
        }
    }

    /// Poll with a maximum number of iterations per promise
    pub fn poll_with_limit(&mut self, max_polls: usize) -> PromiseAllState {
        if self.poll_count >= max_polls {
            return PromiseAllState::Failed(format!(
                "PromiseAll timed out after {} polls",
                max_polls
            ));
        }
        self.poll()
    }

    /// Block until all promises complete
    ///
    /// Returns all results in order, or the first error encountered.
    pub fn await_all(&mut self) -> RuntimeResult<Vec<ZyntaxValue>> {
        #[cfg(not(target_arch = "wasm32"))]
        drive_until(&self.promises, None, promise_all_done);
        #[cfg(target_arch = "wasm32")]
        loop {
            match self.poll() {
                PromiseAllState::Pending => std::thread::yield_now(),
                PromiseAllState::AllReady(values) => return Ok(values),
                PromiseAllState::Failed(err) => return Err(RuntimeError::Promise(err)),
            }
        }
        collect_all(&self.promises)
    }

    /// Block until all promises complete with a timeout (drives all
    /// children cooperatively; concurrent, not one-after-the-other).
    pub fn await_all_with_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> RuntimeResult<Vec<ZyntaxValue>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let deadline = web_time::Instant::now() + timeout;
            let done = drive_until(&self.promises, Some(deadline), promise_all_done);
            if !done {
                for (i, p) in self.promises.iter().enumerate() {
                    if p.cancel() {
                        crate::host_futures::deregister_task(i as i64);
                    }
                }
                return Err(RuntimeError::Promise(format!(
                    "PromiseAll timed out after {:?}",
                    timeout
                )));
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let start = web_time::Instant::now();
            loop {
                if start.elapsed() > timeout {
                    return Err(RuntimeError::Promise(format!(
                        "PromiseAll timed out after {:?}",
                        timeout
                    )));
                }
                match self.poll() {
                    PromiseAllState::Pending => std::thread::yield_now(),
                    PromiseAllState::AllReady(values) => return Ok(values),
                    PromiseAllState::Failed(err) => return Err(RuntimeError::Promise(err)),
                }
            }
        }
        collect_all(&self.promises)
    }

    /// Get the number of promises in this group
    pub fn len(&self) -> usize {
        self.promises.len()
    }

    /// Check if this group is empty
    pub fn is_empty(&self) -> bool {
        self.promises.is_empty()
    }

    /// Get the total poll count
    pub fn poll_count(&self) -> usize {
        self.poll_count
    }

    /// Check if all promises are complete (without polling)
    pub fn is_complete(&self) -> bool {
        self.promises.iter().all(|p| p.is_complete())
    }
}

/// Await the first promise to complete, similar to JavaScript's `Promise.race()`
///
/// This polls all promises concurrently and resolves as soon as ANY promise completes
/// (either successfully or with an error).
///
/// # Example
///
/// ```rust,ignore
/// use zyntax_embed::{ZyntaxRuntime, ZyntaxValue, PromiseRace};
///
/// let promises = vec![
///     runtime.call_async("slow_task", &[])?,
///     runtime.call_async("fast_task", &[])?,
/// ];
///
/// let mut race = PromiseRace::new(promises);
/// let (index, result) = race.await_first()?;
/// // index = index of the first promise to complete
/// // result = the value from that promise
/// ```
pub struct PromiseRace {
    promises: Vec<ZyntaxPromise>,
    poll_count: usize,
}

/// Result of a promise race
#[derive(Debug, Clone)]
pub enum PromiseRaceState {
    /// No promise has completed yet
    Pending,
    /// A promise completed successfully (index, value)
    Winner(usize, ZyntaxValue),
    /// A promise failed (index, error)
    Failed(usize, String),
}

impl PromiseRace {
    /// Create a new PromiseRace from a vector of promises
    pub fn new(promises: Vec<ZyntaxPromise>) -> Self {
        Self {
            promises,
            poll_count: 0,
        }
    }

    /// Poll all promises once, checking for the first completion
    pub fn poll(&mut self) -> PromiseRaceState {
        self.poll_count += 1;

        for (index, promise) in self.promises.iter().enumerate() {
            match promise.poll() {
                PromiseState::Ready(value) => {
                    return PromiseRaceState::Winner(index, value);
                }
                PromiseState::Failed(err) => {
                    return PromiseRaceState::Failed(index, err);
                }
                PromiseState::Cancelled => {
                    return PromiseRaceState::Failed(index, "Promise was cancelled".to_string());
                }
                PromiseState::Pending => {
                    // Continue checking other promises
                }
            }
        }

        PromiseRaceState::Pending
    }

    /// Block until any promise completes, then cancel the losers.
    ///
    /// Returns the index and value of the first promise to complete. The
    /// still-pending promises are cancelled and their parked futures/timers
    /// torn down so the executor stops driving them.
    pub fn await_first(&mut self) -> RuntimeResult<(usize, ZyntaxValue)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            drive_until(&self.promises, None, |ps| {
                ps.iter().any(|p| !p.is_pending())
            });
            // Cancel the losers (anything still pending) and free their
            // parking.
            for (i, p) in self.promises.iter().enumerate() {
                if p.cancel() {
                    crate::host_futures::deregister_task(i as i64);
                }
            }
            for (index, p) in self.promises.iter().enumerate() {
                match p.state() {
                    PromiseState::Ready(value) => return Ok((index, value)),
                    PromiseState::Failed(err) => {
                        return Err(RuntimeError::Promise(format!(
                            "Promise {index} failed: {err}"
                        )))
                    }
                    _ => {}
                }
            }
            Err(RuntimeError::Promise("No promise completed".to_string()))
        }
        #[cfg(target_arch = "wasm32")]
        loop {
            match self.poll() {
                PromiseRaceState::Pending => std::thread::yield_now(),
                PromiseRaceState::Winner(index, value) => return Ok((index, value)),
                PromiseRaceState::Failed(index, err) => {
                    return Err(RuntimeError::Promise(format!(
                        "Promise {index} failed: {err}"
                    )))
                }
            }
        }
    }

    /// Block until any promise completes with a timeout
    pub fn await_first_with_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> RuntimeResult<(usize, ZyntaxValue)> {
        let start = web_time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(RuntimeError::Promise(format!(
                    "PromiseRace timed out after {:?}",
                    timeout
                )));
            }
            match self.poll() {
                PromiseRaceState::Pending => {
                    std::thread::yield_now();
                }
                PromiseRaceState::Winner(index, value) => {
                    return Ok((index, value));
                }
                PromiseRaceState::Failed(index, err) => {
                    return Err(RuntimeError::Promise(format!(
                        "Promise {} failed: {}",
                        index, err
                    )));
                }
            }
        }
    }

    /// Get the number of promises in the race
    pub fn len(&self) -> usize {
        self.promises.len()
    }

    /// Check if the race is empty
    pub fn is_empty(&self) -> bool {
        self.promises.is_empty()
    }

    /// Get the total poll count
    pub fn poll_count(&self) -> usize {
        self.poll_count
    }
}

/// Await all promises, collecting both successes and failures
///
/// Similar to JavaScript's `Promise.allSettled()`, this waits for ALL promises
/// to complete regardless of success or failure.
///
/// # Example
///
/// ```rust,ignore
/// use zyntax_embed::{ZyntaxRuntime, ZyntaxValue, PromiseAllSettled, SettledResult};
///
/// let promises = vec![
///     runtime.call_async("might_fail", &[ZyntaxValue::Int(1)])?,
///     runtime.call_async("might_fail", &[ZyntaxValue::Int(2)])?,
/// ];
///
/// let mut settled = PromiseAllSettled::new(promises);
/// let results = settled.await_all();
/// for result in results {
///     match result {
///         SettledResult::Fulfilled(value) => println!("Success: {:?}", value),
///         SettledResult::Rejected(err) => println!("Failed: {}", err),
///     }
/// }
/// ```
pub struct PromiseAllSettled {
    promises: Vec<ZyntaxPromise>,
    poll_count: usize,
}

/// Result for a single promise in allSettled
#[derive(Debug, Clone)]
pub enum SettledResult {
    /// Promise completed successfully
    Fulfilled(ZyntaxValue),
    /// Promise failed with an error
    Rejected(String),
}

impl PromiseAllSettled {
    /// Create a new PromiseAllSettled from a vector of promises
    pub fn new(promises: Vec<ZyntaxPromise>) -> Self {
        Self {
            promises,
            poll_count: 0,
        }
    }

    /// Poll all promises once
    ///
    /// Returns None if any promise is still pending, or Some with all results.
    pub fn poll(&mut self) -> Option<Vec<SettledResult>> {
        self.poll_count += 1;

        let mut all_settled = true;
        let mut results = Vec::with_capacity(self.promises.len());

        for promise in &self.promises {
            match promise.poll() {
                PromiseState::Pending => {
                    all_settled = false;
                    results.push(SettledResult::Rejected("pending".to_string()));
                    // Placeholder
                }
                PromiseState::Ready(value) => {
                    results.push(SettledResult::Fulfilled(value));
                }
                PromiseState::Failed(err) => {
                    results.push(SettledResult::Rejected(err));
                }
                PromiseState::Cancelled => {
                    results.push(SettledResult::Rejected("Promise was cancelled".to_string()));
                }
            }
        }

        if all_settled {
            Some(results)
        } else {
            None
        }
    }

    /// Block until all promises settle (complete or fail)
    pub fn await_all(&mut self) -> Vec<SettledResult> {
        loop {
            if let Some(results) = self.poll() {
                return results;
            }
            std::thread::yield_now();
        }
    }

    /// Block until all promises settle with a timeout
    pub fn await_all_with_timeout(
        &mut self,
        timeout: std::time::Duration,
    ) -> RuntimeResult<Vec<SettledResult>> {
        let start = web_time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(RuntimeError::Promise(format!(
                    "PromiseAllSettled timed out after {:?}",
                    timeout
                )));
            }
            if let Some(results) = self.poll() {
                return Ok(results);
            }
            std::thread::yield_now();
        }
    }

    /// Get the number of promises
    pub fn len(&self) -> usize {
        self.promises.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.promises.is_empty()
    }

    /// Get the total poll count
    pub fn poll_count(&self) -> usize {
        self.poll_count
    }
}

// ============================================================================
// Variadic Function Calling Support
// ============================================================================

/// Implement Rust's Future trait for ZyntaxPromise
impl std::future::Future for ZyntaxPromise {
    type Output = RuntimeResult<ZyntaxValue>;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        // Store the waker for later notification
        {
            let mut inner = self.state.lock().unwrap();
            inner.waker = Some(cx.waker().clone());
        }

        // Poll the promise
        match ZyntaxPromise::poll(&self) {
            PromiseState::Ready(value) => std::task::Poll::Ready(Ok(value)),
            PromiseState::Failed(err) => std::task::Poll::Ready(Err(RuntimeError::Promise(err))),
            PromiseState::Cancelled => std::task::Poll::Ready(Err(RuntimeError::Promise(
                "Promise was cancelled".to_string(),
            ))),
            PromiseState::Pending => std::task::Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promise_state() {
        let promise = ZyntaxPromise {
            state: Arc::new(Mutex::new(PromiseInner {
                init_fn: std::ptr::null(),
                poll_fn: None,
                args: vec![],
                state: PromiseState::Ready(ZyntaxValue::Int(42)),
                state_machine: None,
                ready_queue: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                task_id: 0,
                poll_count: 0,
                waker: None,
                owner_thread: std::thread::current().id(),
            })),
        };

        assert!(promise.is_complete());
        assert!(!promise.is_pending());

        match promise.state() {
            PromiseState::Ready(ZyntaxValue::Int(42)) => {}
            _ => panic!("Expected Ready(42)"),
        }
    }

    #[test]
    fn test_promise_then() {
        let promise = ZyntaxPromise {
            state: Arc::new(Mutex::new(PromiseInner {
                init_fn: std::ptr::null(),
                poll_fn: None,
                args: vec![],
                state: PromiseState::Ready(ZyntaxValue::Int(10)),
                state_machine: None,
                ready_queue: Arc::new(Mutex::new(std::collections::VecDeque::new())),
                task_id: 0,
                poll_count: 0,
                waker: None,
                owner_thread: std::thread::current().id(),
            })),
        };

        let chained = promise.then(|v| {
            if let ZyntaxValue::Int(n) = v {
                ZyntaxValue::Int(n * 2)
            } else {
                v
            }
        });

        // Wait for the chain to complete
        std::thread::sleep(std::time::Duration::from_millis(50));

        match chained.state() {
            PromiseState::Ready(ZyntaxValue::Int(20)) => {}
            state => panic!("Expected Ready(20), got {:?}", state),
        }
    }
}
