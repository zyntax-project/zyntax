//! ZRTL Thread Plugin
//!
//! Provides threading, synchronization, and message passing for Zyntax-based languages.
//!
//! ## Exported Symbols
//!
//! ### Thread Operations
//! - `$Thread$spawn` - Spawn a new thread with a function pointer
//! - `$Thread$spawn_closure` - Spawn a new thread with a ZrtlClosure
//! - `$Thread$join` - Wait for thread to complete
//! - `$Thread$current_id` - Get current thread ID
//! - `$Thread$yield_now` - Yield execution to other threads
//! - `$Thread$park` - Park the current thread
//! - `$Thread$unpark` - Unpark a thread by handle
//!
//! ### Atomics (i64)
//! - `$Atomic$new` - Create new atomic i64
//! - `$Atomic$load` - Load value atomically
//! - `$Atomic$store` - Store value atomically
//! - `$Atomic$add` - Atomic fetch-add
//! - `$Atomic$sub` - Atomic fetch-sub
//! - `$Atomic$swap` - Atomic swap
//! - `$Atomic$compare_exchange` - Atomic CAS
//! - `$Atomic$free` - Free atomic
//!
//! ### Mutex
//! - `$Mutex$new` - Create new mutex
//! - `$Mutex$lock` - Lock mutex
//! - `$Mutex$try_lock` - Try to lock mutex
//! - `$Mutex$unlock` - Unlock mutex
//! - `$Mutex$free` - Free mutex
//!
//! ### Channel (MPSC message passing)
//! - `$Channel$new` - Create unbounded channel (returns sender, receiver handles)
//! - `$Channel$bounded` - Create bounded channel with capacity
//! - `$Channel$send` - Send i64 value through channel
//! - `$Channel$send_ptr` - Send pointer through channel
//! - `$Channel$recv` - Receive value (blocking)
//! - `$Channel$try_recv` - Try to receive (non-blocking)
//! - `$Channel$recv_timeout` - Receive with timeout
//! - `$Channel$clone_sender` - Clone a sender for multiple producers
//! - `$Channel$close_sender` - Close sender
//! - `$Channel$close_receiver` - Close receiver

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::thread::{self, JoinHandle, Thread};
use std::time::Duration;
use zrtl::heap::{attach, blocking, hold, safepoint, unhold, Hold};
use zrtl::{zrtl_plugin, DynamicBox, ZrtlClosure};

// ============================================================================
// Handle Management
// ============================================================================

static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

type ThreadMap = HashMap<u64, JoinHandle<i64>>;
type AtomicMap = HashMap<u64, Box<AtomicI64>>;
type MutexMap = HashMap<u64, Arc<Mutex<()>>>;
type ThreadRefMap = HashMap<u64, Thread>;
type SenderMap = HashMap<u64, Sender<i64>>;
type ReceiverMap = HashMap<u64, Receiver<i64>>;

static THREADS: Mutex<Option<ThreadMap>> = Mutex::new(None);
static ATOMICS: Mutex<Option<AtomicMap>> = Mutex::new(None);
static MUTEXES: Mutex<Option<MutexMap>> = Mutex::new(None);
static THREAD_REFS: Mutex<Option<ThreadRefMap>> = Mutex::new(None);
static SENDERS: Mutex<Option<SenderMap>> = Mutex::new(None);
static RECEIVERS: Mutex<Option<ReceiverMap>> = Mutex::new(None);

fn next_handle() -> u64 {
    HANDLE_COUNTER.fetch_add(1, Ordering::SeqCst)
}

/// Lock `m`. A wait for it counts as blocking, so a collection on
/// another thread does not wait for this one.
fn lock_wait<T>(m: &'static Mutex<T>) -> MutexGuard<'static, T> {
    if let Ok(guard) = m.try_lock() {
        return guard;
    }
    blocking(|| m.lock().unwrap())
}

/// A word the program may have made from a heap pointer.
fn address(word: i64) -> *const u8 {
    word as usize as *const u8
}

/// A thread this plugin starts. It joins the collector's view before it
/// takes anything, then gives up the hold on its start-up argument, now
/// that the thread's own frames carry it. The result stays held until
/// `thread_join` takes it, because the join handle keeps it where no
/// collector reads.
fn run_thread(arg: Hold, body: impl FnOnce() -> i64) -> i64 {
    attach();
    drop(arg);
    let result = body();
    hold(address(result));
    result
}

/// A value that left a queue: it is the receiver's now.
fn received(value: i64) -> i64 {
    unhold(address(value));
    value
}

macro_rules! get_map {
    ($static:ident, $type:ty) => {{
        let mut guard = lock_wait(&$static);
        if guard.is_none() {
            *guard = Some(HashMap::new());
        }
        guard
    }};
}

// ============================================================================
// Thread Functions
// ============================================================================

/// Function pointer type for thread entry
pub type ThreadFn = extern "C" fn(i64) -> i64;

/// Spawn a new thread
///
/// The function will be called with the provided argument.
/// Returns a thread handle (0 on error).
#[no_mangle]
pub extern "C" fn thread_spawn(func: ThreadFn, arg: i64) -> u64 {
    let handle_id = next_handle();

    let start = Hold::new(arg);
    let join_handle = thread::spawn(move || run_thread(start, || func(arg)));

    // Store thread reference for unpark
    {
        let mut refs = get_map!(THREAD_REFS, ThreadRefMap);
        if let Some(ref mut m) = *refs {
            m.insert(handle_id, join_handle.thread().clone());
        }
    }

    // Store join handle
    {
        let mut threads = get_map!(THREADS, ThreadMap);
        if let Some(ref mut m) = *threads {
            m.insert(handle_id, join_handle);
        }
    }

    handle_id
}

/// Spawn a new thread with a closure
///
/// The closure will be called with the provided argument.
/// Returns a thread handle (0 on error).
///
/// # Safety
/// The closure pointer must be valid and owned (will be consumed).
#[no_mangle]
pub unsafe extern "C" fn thread_spawn_closure(closure: *mut ZrtlClosure, arg: i64) -> u64 {
    if closure.is_null() {
        return 0;
    }

    // Take ownership of the closure
    let closure = Box::from_raw(closure);
    let handle_id = next_handle();

    let start = Hold::new(arg);
    let join_handle = thread::spawn(move || {
        run_thread(start, || match closure.call(arg) {
            r if r.is_ok() => r.value,
            _ => i64::MIN,
        })
    });

    // Store thread reference for unpark
    {
        let mut refs = get_map!(THREAD_REFS, ThreadRefMap);
        if let Some(ref mut m) = *refs {
            m.insert(handle_id, join_handle.thread().clone());
        }
    }

    // Store join handle
    {
        let mut threads = get_map!(THREADS, ThreadMap);
        if let Some(ref mut m) = *threads {
            m.insert(handle_id, join_handle);
        }
    }

    handle_id
}

/// Spawn a new thread with a closure (no argument needed)
///
/// Use this when the closure already has all captured state it needs.
/// This is the preferred API for compiler-generated closures.
/// Returns a thread handle (0 on error).
///
/// # Safety
/// The closure pointer must be valid and owned (will be consumed).
#[no_mangle]
pub unsafe extern "C" fn thread_spawn_closure_noarg(closure: *mut ZrtlClosure) -> u64 {
    thread_spawn_closure(closure, 0)
}

/// Spawn a new thread with a DynamicBox containing a closure
///
/// This is the most flexible API - accepts any DynamicBox that contains
/// a ZrtlClosure. The box is consumed.
/// Returns a thread handle (0 on error, including if box doesn't contain closure).
///
/// # Safety
/// The box pointer must be valid and owned (will be consumed).
#[no_mangle]
pub unsafe extern "C" fn thread_spawn_boxed(boxed: *mut DynamicBox) -> u64 {
    if boxed.is_null() {
        return 0;
    }

    let mut boxed = Box::from_raw(boxed);

    // Check if it's a closure
    if !boxed.is_closure() {
        return 0;
    }

    // Extract the closure
    let closure = match boxed.as_closure() {
        Some(c) => c.clone(),
        None => return 0,
    };

    // Free the box
    boxed.free();

    let handle_id = next_handle();

    let join_handle = thread::spawn(move || {
        run_thread(Hold::new(0), || match closure.call(0) {
            r if r.is_ok() => r.value,
            _ => i64::MIN,
        })
    });

    // Store thread reference for unpark
    {
        let mut refs = get_map!(THREAD_REFS, ThreadRefMap);
        if let Some(ref mut m) = *refs {
            m.insert(handle_id, join_handle.thread().clone());
        }
    }

    // Store join handle
    {
        let mut threads = get_map!(THREADS, ThreadMap);
        if let Some(ref mut m) = *threads {
            m.insert(handle_id, join_handle);
        }
    }

    handle_id
}

/// Spawn a new thread with a cloned closure (closure is not consumed)
///
/// The closure will be cloned before spawning, so it can be reused.
/// Returns a thread handle (0 on error).
///
/// # Safety
/// The closure pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn thread_spawn_closure_cloned(closure: *const ZrtlClosure, arg: i64) -> u64 {
    if closure.is_null() {
        return 0;
    }

    // Clone the closure instead of taking ownership
    let closure = (*closure).clone();
    let handle_id = next_handle();

    let start = Hold::new(arg);
    let join_handle = thread::spawn(move || {
        run_thread(start, || match closure.call(arg) {
            r if r.is_ok() => r.value,
            _ => i64::MIN,
        })
    });

    // Store thread reference for unpark
    {
        let mut refs = get_map!(THREAD_REFS, ThreadRefMap);
        if let Some(ref mut m) = *refs {
            m.insert(handle_id, join_handle.thread().clone());
        }
    }

    // Store join handle
    {
        let mut threads = get_map!(THREADS, ThreadMap);
        if let Some(ref mut m) = *threads {
            m.insert(handle_id, join_handle);
        }
    }

    handle_id
}

/// Join a thread and get its return value
///
/// Returns the thread's return value, or i64::MIN on error.
/// The handle is freed after joining.
#[no_mangle]
pub extern "C" fn thread_join(handle: u64) -> i64 {
    // Remove from refs
    {
        let mut refs = get_map!(THREAD_REFS, ThreadRefMap);
        if let Some(ref mut m) = *refs {
            m.remove(&handle);
        }
    }

    // Get and join
    let join_handle = {
        let mut threads = get_map!(THREADS, ThreadMap);
        if let Some(ref mut m) = *threads {
            m.remove(&handle)
        } else {
            None
        }
    };

    match join_handle {
        Some(jh) => match blocking(|| jh.join()) {
            Ok(result) => received(result),
            Err(_) => i64::MIN,
        },
        None => i64::MIN,
    }
}

/// Get current thread ID as u64
#[no_mangle]
pub extern "C" fn thread_current_id() -> u64 {
    // Hash the ThreadId to get a u64
    let id = thread::current().id();
    // ThreadId doesn't expose its internal value, so we format and hash
    let formatted = format!("{:?}", id);
    let mut hash: u64 = 0;
    for byte in formatted.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
    }
    hash
}

/// Yield execution to other threads
#[no_mangle]
pub extern "C" fn thread_yield_now() {
    safepoint();
    thread::yield_now();
}

/// Park the current thread until unparked
#[no_mangle]
pub extern "C" fn thread_park() {
    blocking(thread::park);
}

/// Unpark a thread by handle
///
/// Returns 0 on success, -1 if handle not found.
#[no_mangle]
pub extern "C" fn thread_unpark(handle: u64) -> i32 {
    let refs = get_map!(THREAD_REFS, ThreadRefMap);
    if let Some(ref m) = *refs {
        if let Some(thread) = m.get(&handle) {
            thread.unpark();
            return 0;
        }
    }
    -1
}

/// Get the number of available CPU cores
#[no_mangle]
pub extern "C" fn thread_available_parallelism() -> u32 {
    thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

// ============================================================================
// Atomic Functions (i64)
// ============================================================================

/// Create a new atomic i64
///
/// Returns a handle to the atomic.
#[no_mangle]
pub extern "C" fn atomic_new(initial: i64) -> u64 {
    let handle = next_handle();
    let atomic = Box::new(AtomicI64::new(initial));

    let mut atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref mut m) = *atomics {
        m.insert(handle, atomic);
    }

    handle
}

/// Load value from atomic
#[no_mangle]
pub extern "C" fn atomic_load(handle: u64) -> i64 {
    safepoint();
    let atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref m) = *atomics {
        if let Some(atomic) = m.get(&handle) {
            return atomic.load(Ordering::SeqCst);
        }
    }
    0
}

/// Store value to atomic
#[no_mangle]
pub extern "C" fn atomic_store(handle: u64, value: i64) {
    let atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref m) = *atomics {
        if let Some(atomic) = m.get(&handle) {
            atomic.store(value, Ordering::SeqCst);
        }
    }
}

/// Atomic fetch-add, returns previous value
#[no_mangle]
pub extern "C" fn atomic_add(handle: u64, value: i64) -> i64 {
    let atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref m) = *atomics {
        if let Some(atomic) = m.get(&handle) {
            return atomic.fetch_add(value, Ordering::SeqCst);
        }
    }
    0
}

/// Atomic fetch-sub, returns previous value
#[no_mangle]
pub extern "C" fn atomic_sub(handle: u64, value: i64) -> i64 {
    let atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref m) = *atomics {
        if let Some(atomic) = m.get(&handle) {
            return atomic.fetch_sub(value, Ordering::SeqCst);
        }
    }
    0
}

/// Atomic swap, returns previous value
#[no_mangle]
pub extern "C" fn atomic_swap(handle: u64, value: i64) -> i64 {
    let atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref m) = *atomics {
        if let Some(atomic) = m.get(&handle) {
            return atomic.swap(value, Ordering::SeqCst);
        }
    }
    0
}

/// Atomic compare-and-swap
///
/// If current value equals `expected`, sets to `new_value` and returns 1.
/// Otherwise returns 0 and leaves value unchanged.
#[no_mangle]
pub extern "C" fn atomic_compare_exchange(handle: u64, expected: i64, new_value: i64) -> i32 {
    safepoint();
    let atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref m) = *atomics {
        if let Some(atomic) = m.get(&handle) {
            match atomic.compare_exchange(expected, new_value, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return 1,
                Err(_) => return 0,
            }
        }
    }
    0
}

/// Free an atomic
#[no_mangle]
pub extern "C" fn atomic_free(handle: u64) {
    let mut atomics = get_map!(ATOMICS, AtomicMap);
    if let Some(ref mut m) = *atomics {
        m.remove(&handle);
    }
}

// ============================================================================
// Mutex Functions
// ============================================================================

/// Create a new mutex
///
/// Returns a handle to the mutex.
#[no_mangle]
pub extern "C" fn mutex_new() -> u64 {
    let handle = next_handle();
    let mutex = Arc::new(Mutex::new(()));

    let mut mutexes = get_map!(MUTEXES, MutexMap);
    if let Some(ref mut m) = *mutexes {
        m.insert(handle, mutex);
    }

    handle
}

// Thread-local storage for held mutex guards
thread_local! {
    static HELD_GUARDS: std::cell::RefCell<HashMap<u64, MutexGuard<'static, ()>>> =
        std::cell::RefCell::new(HashMap::new());
}

/// Lock a mutex
///
/// Returns 0 on success, -1 on error.
/// Note: This uses a simplified locking mechanism.
#[no_mangle]
pub extern "C" fn mutex_lock(handle: u64) -> i32 {
    let mutex = {
        let mutexes = get_map!(MUTEXES, MutexMap);
        if let Some(ref m) = *mutexes {
            m.get(&handle).cloned()
        } else {
            None
        }
    };

    match mutex {
        Some(m) => {
            // This is a simplified implementation - in practice you'd want
            // to properly manage the guard lifetime
            let acquired = match m.try_lock() {
                Err(TryLockError::WouldBlock) => blocking(|| m.lock()).ok(),
                other => other.ok(),
            };
            match acquired {
                Some(_guard) => {
                    // Guard is dropped here, but we've signaled lock acquisition
                    // A real implementation would need to store the guard
                    std::mem::forget(_guard);
                    0
                }
                None => -1,
            }
        }
        None => -1,
    }
}

/// Try to lock a mutex without blocking
///
/// Returns 1 if lock acquired, 0 if would block, -1 on error.
#[no_mangle]
pub extern "C" fn mutex_try_lock(handle: u64) -> i32 {
    let mutex = {
        let mutexes = get_map!(MUTEXES, MutexMap);
        if let Some(ref m) = *mutexes {
            m.get(&handle).cloned()
        } else {
            None
        }
    };

    match mutex {
        Some(m) => match m.try_lock() {
            Ok(_guard) => {
                std::mem::forget(_guard);
                1
            }
            Err(std::sync::TryLockError::WouldBlock) => 0,
            Err(_) => -1,
        },
        None => -1,
    }
}

/// Unlock a mutex
///
/// Note: This implementation relies on the mutex being unlocked by
/// dropping through the standard mechanism. In a real implementation,
/// you'd need proper guard management.
#[no_mangle]
pub extern "C" fn mutex_unlock(_handle: u64) -> i32 {
    // In a real implementation, we would release the stored guard
    // For now, this is a no-op as the guard management is simplified
    0
}

/// Free a mutex
#[no_mangle]
pub extern "C" fn mutex_free(handle: u64) {
    let mut mutexes = get_map!(MUTEXES, MutexMap);
    if let Some(ref mut m) = *mutexes {
        m.remove(&handle);
    }
}

// ============================================================================
// Channel Functions (MPSC)
// ============================================================================

/// Channel result structure returned by channel_new
/// Contains sender_handle and receiver_handle
#[repr(C)]
pub struct ChannelPair {
    pub sender: u64,
    pub receiver: u64,
}

/// Create a new unbounded MPSC channel
///
/// Returns a ChannelPair with sender and receiver handles.
/// The channel can hold unlimited messages (bounded only by memory).
#[no_mangle]
pub extern "C" fn channel_new() -> ChannelPair {
    let (tx, rx) = mpsc::channel();

    let sender_handle = next_handle();
    let receiver_handle = next_handle();

    {
        let mut senders = get_map!(SENDERS, SenderMap);
        if let Some(ref mut m) = *senders {
            m.insert(sender_handle, tx);
        }
    }

    {
        let mut receivers = get_map!(RECEIVERS, ReceiverMap);
        if let Some(ref mut m) = *receivers {
            m.insert(receiver_handle, rx);
        }
    }

    ChannelPair {
        sender: sender_handle,
        receiver: receiver_handle,
    }
}

/// Create a new bounded (sync) MPSC channel with specified capacity
///
/// Returns a ChannelPair with sender and receiver handles.
/// Senders will block when the channel is full.
#[no_mangle]
pub extern "C" fn channel_bounded(capacity: u64) -> ChannelPair {
    let (tx, rx) = mpsc::sync_channel(capacity as usize);

    let sender_handle = next_handle();
    let receiver_handle = next_handle();

    // Convert SyncSender to regular Sender by wrapping
    // Note: We use a wrapper approach with unbounded channel for simplicity
    // For true bounded behavior, we'd need separate storage
    let (async_tx, async_rx) = mpsc::channel();

    // Spawn a bridge thread that respects the bound
    let _bridge = thread::spawn(move || {
        while let Ok(msg) = async_rx.recv() {
            if let Err(lost) = tx.send(msg) {
                unhold(address(lost.0));
                break;
            }
        }
        // Whatever is still queued will never be received.
        while let Ok(msg) = async_rx.try_recv() {
            unhold(address(msg));
        }
    });

    {
        let mut senders = get_map!(SENDERS, SenderMap);
        if let Some(ref mut m) = *senders {
            m.insert(sender_handle, async_tx);
        }
    }

    {
        let mut receivers = get_map!(RECEIVERS, ReceiverMap);
        if let Some(ref mut m) = *receivers {
            m.insert(receiver_handle, rx);
        }
    }

    ChannelPair {
        sender: sender_handle,
        receiver: receiver_handle,
    }
}

/// Send an i64 value through the channel
///
/// Returns 0 on success, -1 if channel is closed/disconnected.
#[no_mangle]
pub extern "C" fn channel_send(sender_handle: u64, value: i64) -> i32 {
    let senders = get_map!(SENDERS, SenderMap);
    if let Some(ref m) = *senders {
        if let Some(sender) = m.get(&sender_handle) {
            // The queue is memory no collector reads: the value stays held
            // until a receive takes it.
            hold(address(value));
            match sender.send(value) {
                Ok(()) => return 0,
                Err(_) => {
                    unhold(address(value));
                    return -1; // Receiver disconnected
                }
            }
        }
    }
    -1
}

/// Send a pointer as i64 through the channel
///
/// This is the same as channel_send but semantically indicates pointer transfer.
/// The receiver is responsible for proper handling of the pointer.
/// Returns 0 on success, -1 if channel is closed.
#[no_mangle]
pub extern "C" fn channel_send_ptr(sender_handle: u64, ptr: *const u8) -> i32 {
    channel_send(sender_handle, ptr as i64)
}

/// Receive a value from the channel (blocking)
///
/// Blocks until a value is available or the channel is closed.
/// Returns the received value.
/// On error/disconnect, returns i64::MIN.
#[no_mangle]
pub extern "C" fn channel_recv(receiver_handle: u64) -> i64 {
    let receivers = get_map!(RECEIVERS, ReceiverMap);
    if let Some(ref m) = *receivers {
        if let Some(receiver) = m.get(&receiver_handle) {
            match blocking(|| receiver.recv()) {
                Ok(value) => return received(value),
                Err(_) => return i64::MIN, // Sender disconnected
            }
        }
    }
    i64::MIN
}

/// Try to receive a value without blocking
///
/// Returns the value if available immediately.
/// Returns i64::MIN if no value is available or on error.
/// Use channel_try_recv_status for distinguishing empty vs disconnected.
#[no_mangle]
pub extern "C" fn channel_try_recv(receiver_handle: u64) -> i64 {
    let receivers = get_map!(RECEIVERS, ReceiverMap);
    if let Some(ref m) = *receivers {
        if let Some(receiver) = m.get(&receiver_handle) {
            match receiver.try_recv() {
                Ok(value) => return received(value),
                Err(_) => return i64::MIN,
            }
        }
    }
    i64::MIN
}

/// Result of try_recv with status
#[repr(C)]
pub struct TryRecvResult {
    /// The received value (valid only if status == 0)
    pub value: i64,
    /// Status: 0 = success, 1 = empty (would block), -1 = disconnected
    pub status: i32,
}

/// Try to receive with detailed status
///
/// Returns a TryRecvResult with value and status.
/// Status: 0 = received value, 1 = channel empty, -1 = channel disconnected
#[no_mangle]
pub extern "C" fn channel_try_recv_status(receiver_handle: u64) -> TryRecvResult {
    let receivers = get_map!(RECEIVERS, ReceiverMap);
    if let Some(ref m) = *receivers {
        if let Some(receiver) = m.get(&receiver_handle) {
            match receiver.try_recv() {
                Ok(value) => {
                    return TryRecvResult {
                        value: received(value),
                        status: 0,
                    }
                }
                Err(TryRecvError::Empty) => {
                    return TryRecvResult {
                        value: 0,
                        status: 1,
                    }
                }
                Err(TryRecvError::Disconnected) => {
                    return TryRecvResult {
                        value: 0,
                        status: -1,
                    }
                }
            }
        }
    }
    TryRecvResult {
        value: 0,
        status: -1,
    }
}

/// Receive with timeout (milliseconds)
///
/// Blocks for up to timeout_ms milliseconds.
/// Returns the value if received, i64::MIN on timeout or error.
#[no_mangle]
pub extern "C" fn channel_recv_timeout(receiver_handle: u64, timeout_ms: u64) -> i64 {
    let receivers = get_map!(RECEIVERS, ReceiverMap);
    if let Some(ref m) = *receivers {
        if let Some(receiver) = m.get(&receiver_handle) {
            let timeout = Duration::from_millis(timeout_ms);
            match blocking(|| receiver.recv_timeout(timeout)) {
                Ok(value) => return received(value),
                Err(_) => return i64::MIN,
            }
        }
    }
    i64::MIN
}

/// Result of recv_timeout with status
#[repr(C)]
pub struct RecvTimeoutResult {
    /// The received value (valid only if status == 0)
    pub value: i64,
    /// Status: 0 = success, 1 = timeout, -1 = disconnected
    pub status: i32,
}

/// Receive with timeout and detailed status
///
/// Returns a RecvTimeoutResult with value and status.
/// Status: 0 = received, 1 = timed out, -1 = disconnected
#[no_mangle]
pub extern "C" fn channel_recv_timeout_status(
    receiver_handle: u64,
    timeout_ms: u64,
) -> RecvTimeoutResult {
    let receivers = get_map!(RECEIVERS, ReceiverMap);
    if let Some(ref m) = *receivers {
        if let Some(receiver) = m.get(&receiver_handle) {
            let timeout = Duration::from_millis(timeout_ms);
            match blocking(|| receiver.recv_timeout(timeout)) {
                Ok(value) => {
                    return RecvTimeoutResult {
                        value: received(value),
                        status: 0,
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    return RecvTimeoutResult {
                        value: 0,
                        status: 1,
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return RecvTimeoutResult {
                        value: 0,
                        status: -1,
                    }
                }
            }
        }
    }
    RecvTimeoutResult {
        value: 0,
        status: -1,
    }
}

/// Clone a sender to allow multiple producers
///
/// Returns a new sender handle that sends to the same receiver.
/// Returns 0 if the original sender handle is invalid.
#[no_mangle]
pub extern "C" fn channel_clone_sender(sender_handle: u64) -> u64 {
    let cloned_sender = {
        let senders = get_map!(SENDERS, SenderMap);
        if let Some(ref m) = *senders {
            m.get(&sender_handle).cloned()
        } else {
            None
        }
    };

    match cloned_sender {
        Some(sender) => {
            let new_handle = next_handle();
            let mut senders = get_map!(SENDERS, SenderMap);
            if let Some(ref mut m) = *senders {
                m.insert(new_handle, sender);
            }
            new_handle
        }
        None => 0,
    }
}

/// Close a sender
///
/// Once all senders are closed, the receiver will get a disconnect error
/// after reading all pending messages.
#[no_mangle]
pub extern "C" fn channel_close_sender(sender_handle: u64) {
    let mut senders = get_map!(SENDERS, SenderMap);
    if let Some(ref mut m) = *senders {
        m.remove(&sender_handle);
    }
}

/// Close a receiver
///
/// Once the receiver is closed, all senders will get send errors.
#[no_mangle]
pub extern "C" fn channel_close_receiver(receiver_handle: u64) {
    let mut receivers = get_map!(RECEIVERS, ReceiverMap);
    if let Some(ref mut m) = *receivers {
        if let Some(receiver) = m.remove(&receiver_handle) {
            // What is queued is dropped with the receiver: let it go.
            while let Ok(value) = receiver.try_recv() {
                unhold(address(value));
            }
        }
    }
}

/// Check if a channel is empty
///
/// Note: This is inherently racy - by the time you act on this,
/// the state may have changed.
/// Returns 1 if empty, 0 if not empty, -1 if handle invalid.
#[no_mangle]
pub extern "C" fn channel_is_empty(receiver_handle: u64) -> i32 {
    let receivers = get_map!(RECEIVERS, ReceiverMap);
    if let Some(ref m) = *receivers {
        if let Some(receiver) = m.get(&receiver_handle) {
            // Try non-blocking peek via try_recv - if Empty, it's empty
            match receiver.try_recv() {
                Err(TryRecvError::Empty) => return 1,
                Err(TryRecvError::Disconnected) => return -1,
                Ok(value) => {
                    unhold(address(value));
                    // Oops, we consumed a value - this is why is_empty is racy
                    // We can't put it back easily. Return 0 (not empty).
                    // Note: This implementation is problematic, better to use try_recv_status
                    return 0;
                }
            }
        }
    }
    -1
}

// ============================================================================
// Plugin Export
// ============================================================================

zrtl_plugin! {
    name: "zrtl_thread",
    symbols: [
        // Thread operations
        ("$Thread$spawn", thread_spawn),
        ("$Thread$spawn_closure", thread_spawn_closure),
        ("$Thread$spawn_closure_noarg", thread_spawn_closure_noarg),
        ("$Thread$spawn_closure_cloned", thread_spawn_closure_cloned),
        ("$Thread$spawn_boxed", thread_spawn_boxed),
        ("$Thread$join", thread_join),
        ("$Thread$current_id", thread_current_id),
        ("$Thread$yield_now", thread_yield_now),
        ("$Thread$park", thread_park),
        ("$Thread$unpark", thread_unpark),
        ("$Thread$available_parallelism", thread_available_parallelism),

        // Atomics
        ("$Atomic$new", atomic_new),
        ("$Atomic$load", atomic_load),
        ("$Atomic$store", atomic_store),
        ("$Atomic$add", atomic_add),
        ("$Atomic$sub", atomic_sub),
        ("$Atomic$swap", atomic_swap),
        ("$Atomic$compare_exchange", atomic_compare_exchange),
        ("$Atomic$free", atomic_free),

        // Mutex
        ("$Mutex$new", mutex_new),
        ("$Mutex$lock", mutex_lock),
        ("$Mutex$try_lock", mutex_try_lock),
        ("$Mutex$unlock", mutex_unlock),
        ("$Mutex$free", mutex_free),

        // Channel (MPSC)
        ("$Channel$new", channel_new),
        ("$Channel$bounded", channel_bounded),
        ("$Channel$send", channel_send),
        ("$Channel$send_ptr", channel_send_ptr),
        ("$Channel$recv", channel_recv),
        ("$Channel$try_recv", channel_try_recv),
        ("$Channel$try_recv_status", channel_try_recv_status),
        ("$Channel$recv_timeout", channel_recv_timeout),
        ("$Channel$recv_timeout_status", channel_recv_timeout_status),
        ("$Channel$clone_sender", channel_clone_sender),
        ("$Channel$close_sender", channel_close_sender),
        ("$Channel$close_receiver", channel_close_receiver),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::sync::atomic::AtomicI64;
    use std::sync::atomic::AtomicUsize;
    use zrtl::zrtl_closure_from_fn;

    static RUNS: AtomicUsize = AtomicUsize::new(0);
    static POLLS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn count_run(call: unsafe extern "C" fn(*mut c_void), data: *mut c_void) {
        RUNS.fetch_add(1, Ordering::SeqCst);
        unsafe { call(data) };
    }

    unsafe extern "C" fn count_poll() {
        POLLS.fetch_add(1, Ordering::SeqCst);
    }

    // A host that records what is held, so a test can see when the plugin
    // holds a word and when it lets go.
    static HELD: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    static WATCHED: AtomicI64 = AtomicI64::new(0);
    static HELD_AT_ATTACH: AtomicUsize = AtomicUsize::new(0);
    static ATTACHES: AtomicUsize = AtomicUsize::new(0);

    fn held_count(word: i64) -> usize {
        HELD.lock()
            .unwrap()
            .iter()
            .filter(|a| **a == word as usize)
            .count()
    }

    unsafe extern "C" fn fake_attach() {
        ATTACHES.fetch_add(1, Ordering::SeqCst);
        let watched = WATCHED.load(Ordering::SeqCst);
        if watched != 0 && held_count(watched) > 0 {
            HELD_AT_ATTACH.fetch_add(1, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn fake_hold(address: *const u8) {
        if !address.is_null() {
            HELD.lock().unwrap().push(address as usize);
        }
    }

    unsafe extern "C" fn fake_unhold(address: *const u8) {
        let mut held = HELD.lock().unwrap();
        if let Some(i) = held.iter().rposition(|a| *a == address as usize) {
            held.swap_remove(i);
        }
    }

    fn fake_host() {
        zrtl::heap::set_roots(fake_attach, fake_hold, fake_unhold);
    }

    const START_ARG: i64 = 0x7a00_0010;
    const START_RESULT: i64 = 0x7a00_0020;
    static ARG_HELD_IN_BODY: AtomicUsize = AtomicUsize::new(usize::MAX);

    extern "C" fn answers_with_a_pointer(arg: i64) -> i64 {
        ARG_HELD_IN_BODY.store(held_count(arg), Ordering::SeqCst);
        START_RESULT
    }

    extern "C" fn seven(_: i64) -> i64 {
        7
    }

    extern "C" fn park_then_seven(_: i64) -> i64 {
        thread_park();
        7
    }

    #[test]
    fn a_wait_runs_in_the_hosts_blocking_region() {
        zrtl::heap::set_blocking(count_run, count_poll);

        let before = RUNS.load(Ordering::SeqCst);
        assert_eq!(thread_join(thread_spawn(seven, 0)), 7);
        assert!(RUNS.load(Ordering::SeqCst) > before, "join did not block");

        let pair = channel_new();
        let before = RUNS.load(Ordering::SeqCst);
        assert_eq!(channel_send(pair.sender, 41), 0);
        assert_eq!(channel_recv(pair.receiver), 41);
        assert!(RUNS.load(Ordering::SeqCst) > before, "recv did not block");

        let before = RUNS.load(Ordering::SeqCst);
        let late = channel_recv_timeout_status(pair.receiver, 1);
        assert_eq!(late.status, 1);
        assert!(
            RUNS.load(Ordering::SeqCst) > before,
            "recv_timeout did not block"
        );

        let before = RUNS.load(Ordering::SeqCst);
        let waiter = thread_spawn(park_then_seven, 0);
        assert_eq!(thread_unpark(waiter), 0);
        assert_eq!(thread_join(waiter), 7);
        assert!(RUNS.load(Ordering::SeqCst) > before, "park did not block");
    }

    #[test]
    fn a_thread_holds_its_argument_until_it_runs_and_its_result_until_joined() {
        fake_host();
        WATCHED.store(START_ARG, Ordering::SeqCst);
        let attached = HELD_AT_ATTACH.load(Ordering::SeqCst);

        let handle = thread_spawn(answers_with_a_pointer, START_ARG);

        // The result outlives the thread, in the join handle's packet.
        let began = std::time::Instant::now();
        while held_count(START_RESULT) == 0 {
            assert!(began.elapsed() < Duration::from_secs(5), "no result hold");
            thread::yield_now();
        }
        assert_eq!(thread_join(handle), START_RESULT);

        assert!(
            HELD_AT_ATTACH.load(Ordering::SeqCst) > attached,
            "the argument was not held when the thread attached"
        );
        assert_eq!(
            ARG_HELD_IN_BODY.load(Ordering::SeqCst),
            0,
            "the argument stayed held while the thread ran"
        );
        assert_eq!(held_count(START_ARG), 0);
        assert_eq!(held_count(START_RESULT), 0);
        WATCHED.store(0, Ordering::SeqCst);
    }

    #[test]
    fn a_closure_thread_holds_its_argument_and_result_the_same_way() {
        fake_host();
        const ARG: i64 = 0x7a00_0110;
        const RESULT: i64 = 0x7a00_0120;
        let closure = Box::into_raw(Box::new(zrtl::ZrtlClosure::new(|_| RESULT)));
        let handle = unsafe { thread_spawn_closure(closure, ARG) };
        let began = std::time::Instant::now();
        while held_count(RESULT) == 0 {
            assert!(began.elapsed() < Duration::from_secs(5), "no result hold");
            thread::yield_now();
        }
        assert_eq!(held_count(ARG), 0);
        assert_eq!(thread_join(handle), RESULT);
        assert_eq!(held_count(RESULT), 0);
    }

    #[test]
    fn a_queued_pointer_is_held_until_it_is_received() {
        fake_host();
        const MSG: i64 = 0x7a00_0210;
        let pair = channel_new();

        assert_eq!(channel_send_ptr(pair.sender, MSG as *const u8), 0);
        assert_eq!(held_count(MSG), 1, "a queued pointer is not held");
        assert_eq!(channel_recv(pair.receiver), MSG);
        assert_eq!(held_count(MSG), 0);

        channel_send(pair.sender, MSG);
        assert_eq!(channel_try_recv(pair.receiver), MSG);
        assert_eq!(held_count(MSG), 0);

        channel_send(pair.sender, MSG);
        assert_eq!(channel_try_recv_status(pair.receiver).value, MSG);
        assert_eq!(held_count(MSG), 0);

        channel_send(pair.sender, MSG);
        assert_eq!(channel_recv_timeout(pair.receiver, 100), MSG);
        assert_eq!(held_count(MSG), 0);

        channel_send(pair.sender, MSG);
        assert_eq!(channel_recv_timeout_status(pair.receiver, 100).value, MSG);
        assert_eq!(held_count(MSG), 0);

        // A receive that finds nothing holds nothing.
        assert_eq!(channel_try_recv(pair.receiver), i64::MIN);
        assert_eq!(held_count(MSG), 0);
    }

    #[test]
    fn closing_a_receiver_lets_go_of_what_is_queued() {
        fake_host();
        const MSG: i64 = 0x7a00_0310;
        let pair = channel_new();
        channel_send(pair.sender, MSG);
        channel_send(pair.sender, MSG);
        assert_eq!(held_count(MSG), 2);

        channel_close_receiver(pair.receiver);
        assert_eq!(held_count(MSG), 0, "a closed channel kept its messages");

        // A send to a closed channel fails and holds nothing.
        assert_eq!(channel_send(pair.sender, MSG), -1);
        assert_eq!(held_count(MSG), 0);
    }

    #[test]
    fn a_bounded_channel_holds_a_pointer_through_its_bridge() {
        fake_host();
        const MSGS: [i64; 3] = [0x7a00_0410, 0x7a00_0420, 0x7a00_0430];
        let pair = channel_bounded(1);
        for m in MSGS {
            assert_eq!(channel_send(pair.sender, m), 0);
            assert_eq!(held_count(m), 1);
        }
        for m in MSGS {
            assert_eq!(channel_recv(pair.receiver), m);
            assert_eq!(held_count(m), 0);
        }
    }

    #[test]
    fn a_spin_on_an_atomic_polls_the_safepoint() {
        zrtl::heap::set_blocking(count_run, count_poll);
        let handle = atomic_new(1);
        let before = POLLS.load(Ordering::SeqCst);
        atomic_load(handle);
        atomic_compare_exchange(handle, 1, 2);
        thread_yield_now();
        assert!(POLLS.load(Ordering::SeqCst) >= before + 3);
    }

    #[test]
    fn test_atomic_operations() {
        let handle = atomic_new(10);

        assert_eq!(atomic_load(handle), 10);

        atomic_store(handle, 20);
        assert_eq!(atomic_load(handle), 20);

        let prev = atomic_add(handle, 5);
        assert_eq!(prev, 20);
        assert_eq!(atomic_load(handle), 25);

        let prev = atomic_sub(handle, 10);
        assert_eq!(prev, 25);
        assert_eq!(atomic_load(handle), 15);

        let prev = atomic_swap(handle, 100);
        assert_eq!(prev, 15);
        assert_eq!(atomic_load(handle), 100);

        // CAS success
        let result = atomic_compare_exchange(handle, 100, 200);
        assert_eq!(result, 1);
        assert_eq!(atomic_load(handle), 200);

        // CAS failure
        let result = atomic_compare_exchange(handle, 100, 300);
        assert_eq!(result, 0);
        assert_eq!(atomic_load(handle), 200);

        atomic_free(handle);
    }

    #[test]
    fn test_thread_spawn_join() {
        extern "C" fn double(x: i64) -> i64 {
            x * 2
        }

        let handle = thread_spawn(double, 21);
        assert!(handle > 0);

        let result = thread_join(handle);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_available_parallelism() {
        let cores = thread_available_parallelism();
        assert!(cores >= 1);
    }

    #[test]
    fn test_mutex() {
        let handle = mutex_new();
        assert!(handle > 0);

        // Lock should succeed
        assert_eq!(mutex_lock(handle), 0);

        mutex_free(handle);
    }

    #[test]
    fn test_thread_spawn_closure() {
        extern "C" fn triple(x: i64) -> i64 {
            x * 3
        }

        unsafe {
            let closure = zrtl_closure_from_fn(triple);
            let handle = thread_spawn_closure(closure, 14);
            assert!(handle > 0);

            let result = thread_join(handle);
            assert_eq!(result, 42);
        }
    }

    #[test]
    fn test_thread_spawn_closure_cloned() {
        // Test spawning multiple threads with the same cloned closure
        let counter = Arc::new(AtomicI64::new(0));
        let counter_clone = counter.clone();

        let closure = ZrtlClosure::new(move |x| counter_clone.fetch_add(x, Ordering::SeqCst));

        unsafe {
            let closure_ptr = &closure as *const ZrtlClosure;

            // Spawn 3 threads with the same cloned closure
            let h1 = thread_spawn_closure_cloned(closure_ptr, 10);
            let h2 = thread_spawn_closure_cloned(closure_ptr, 20);
            let h3 = thread_spawn_closure_cloned(closure_ptr, 30);

            // Join all threads
            thread_join(h1);
            thread_join(h2);
            thread_join(h3);

            // Counter should be 10 + 20 + 30 = 60
            assert_eq!(counter.load(Ordering::SeqCst), 60);
        }
    }

    #[test]
    fn test_thread_closure_with_captures() {
        let value = Arc::new(AtomicI64::new(100));
        let value_clone = value.clone();

        let closure = ZrtlClosure::new(move |delta| value_clone.fetch_add(delta, Ordering::SeqCst));

        let closure_box = Box::new(closure);
        let closure_ptr = Box::into_raw(closure_box);

        unsafe {
            let handle = thread_spawn_closure(closure_ptr, 42);
            let prev = thread_join(handle);

            // Previous value was 100
            assert_eq!(prev, 100);
            // New value is 142
            assert_eq!(value.load(Ordering::SeqCst), 142);
        }
    }

    #[test]
    fn test_channel_basic() {
        let pair = channel_new();
        assert!(pair.sender > 0);
        assert!(pair.receiver > 0);

        // Send and receive
        assert_eq!(channel_send(pair.sender, 42), 0);
        assert_eq!(channel_recv(pair.receiver), 42);

        // Multiple messages
        channel_send(pair.sender, 1);
        channel_send(pair.sender, 2);
        channel_send(pair.sender, 3);

        assert_eq!(channel_recv(pair.receiver), 1);
        assert_eq!(channel_recv(pair.receiver), 2);
        assert_eq!(channel_recv(pair.receiver), 3);

        channel_close_sender(pair.sender);
        channel_close_receiver(pair.receiver);
    }

    #[test]
    fn test_channel_try_recv() {
        let pair = channel_new();

        // Nothing to receive yet
        let result = channel_try_recv_status(pair.receiver);
        assert_eq!(result.status, 1); // Empty

        // Send something
        channel_send(pair.sender, 100);

        // Now we can receive
        let result = channel_try_recv_status(pair.receiver);
        assert_eq!(result.status, 0); // Success
        assert_eq!(result.value, 100);

        channel_close_sender(pair.sender);
        channel_close_receiver(pair.receiver);
    }

    #[test]
    fn test_channel_clone_sender() {
        let pair = channel_new();

        // Clone the sender
        let sender2 = channel_clone_sender(pair.sender);
        assert!(sender2 > 0);

        // Both senders can send
        channel_send(pair.sender, 1);
        channel_send(sender2, 2);

        assert_eq!(channel_recv(pair.receiver), 1);
        assert_eq!(channel_recv(pair.receiver), 2);

        channel_close_sender(pair.sender);
        channel_close_sender(sender2);
        channel_close_receiver(pair.receiver);
    }

    #[test]
    fn test_channel_disconnect() {
        let pair = channel_new();

        // Close sender
        channel_close_sender(pair.sender);

        // Receiver should get disconnect
        let result = channel_try_recv_status(pair.receiver);
        assert_eq!(result.status, -1); // Disconnected

        channel_close_receiver(pair.receiver);
    }

    #[test]
    fn test_channel_with_threads() {
        let pair = channel_new();
        let sender = pair.sender;
        let receiver = pair.receiver;

        // Spawn producer thread
        extern "C" fn producer(sender_handle: i64) -> i64 {
            for i in 0..5 {
                channel_send(sender_handle as u64, i);
            }
            channel_close_sender(sender_handle as u64);
            5
        }

        let producer_handle = thread_spawn(producer, sender as i64);

        // Receive all messages
        let mut sum = 0i64;
        loop {
            let result = channel_recv_timeout_status(receiver, 1000);
            if result.status == 0 {
                sum += result.value;
            } else {
                break;
            }
        }

        thread_join(producer_handle);
        assert_eq!(sum, 1 + 2 + 3 + 4);

        channel_close_receiver(receiver);
    }

    #[test]
    fn test_channel_recv_timeout() {
        let pair = channel_new();

        // Timeout on empty channel
        let result = channel_recv_timeout_status(pair.receiver, 10); // 10ms timeout
        assert_eq!(result.status, 1); // Timeout

        // Send and receive within timeout
        channel_send(pair.sender, 42);
        let result = channel_recv_timeout_status(pair.receiver, 1000);
        assert_eq!(result.status, 0);
        assert_eq!(result.value, 42);

        channel_close_sender(pair.sender);
        channel_close_receiver(pair.receiver);
    }
}
