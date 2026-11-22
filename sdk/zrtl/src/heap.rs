//! Where the strings and boxes this SDK makes are allocated.
//!
//! A host that manages the program's heap installs its own allocator
//! with [`set_allocator`], so that a string a plugin builds is a block
//! of the same heap as everything the compiled program makes, and
//! whatever reclaims that heap sees it. Until a host does, the system
//! allocator serves. A plugin loaded from a file is its own copy of
//! this module; the loader installs the allocator into it through the
//! `_zrtl_set_allocator` export the plugin macro emits.
//!
//! Install before the first allocation: a block from one allocator
//! must not be released through the other.
//!
//! A host whose collector runs while several threads allocate also
//! installs [`set_blocking`]. A plugin wraps each wait in [`blocking`]
//! and polls [`safepoint`] in each loop that waits without blocking, so
//! that a collection does not wait for a thread that cannot answer.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Allocate `size` bytes, sixteen-aligned, never null for a size the
/// system could serve.
pub type AllocFn = unsafe extern "C" fn(usize) -> *mut u8;
/// Release a block from the matching [`AllocFn`]; null is ignored.
pub type FreeFn = unsafe extern "C" fn(*mut u8);

/// Run `call(data)` with the calling thread counted as stopped.
pub type BlockingFn =
    unsafe extern "C" fn(call: unsafe extern "C" fn(*mut c_void), data: *mut c_void);
/// Stop here while a collection waits for the calling thread.
pub type PollFn = unsafe extern "C" fn();

static ALLOC: AtomicUsize = AtomicUsize::new(0);
static FREE: AtomicUsize = AtomicUsize::new(0);
static BLOCKING: AtomicUsize = AtomicUsize::new(0);
static POLL: AtomicUsize = AtomicUsize::new(0);

/// Route every string and box allocation through `alloc` and `free`.
pub fn set_allocator(alloc: AllocFn, free: FreeFn) {
    ALLOC.store(alloc as usize, Ordering::SeqCst);
    FREE.store(free as usize, Ordering::SeqCst);
}

/// Whether a host has installed an allocator.
pub fn has_allocator() -> bool {
    ALLOC.load(Ordering::Relaxed) != 0
}

/// Allocate `size` bytes at `align`, which is at most sixteen.
///
/// # Safety
/// As `std::alloc::alloc`: `size` must not be zero.
pub unsafe fn alloc(size: usize, align: usize) -> *mut u8 {
    let f = ALLOC.load(Ordering::Relaxed);
    if f != 0 {
        let alloc: AllocFn = std::mem::transmute(f);
        return alloc(size);
    }
    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(size, align))
}

/// Release a block from [`alloc`] of the same `size` and `align`.
///
/// # Safety
/// `ptr` must have come from [`alloc`] with these arguments and must
/// not be used afterwards.
pub unsafe fn free(ptr: *mut u8, size: usize, align: usize) {
    if ptr.is_null() {
        return;
    }
    let f = FREE.load(Ordering::Relaxed);
    if f != 0 {
        let free: FreeFn = std::mem::transmute(f);
        return free(ptr);
    }
    std::alloc::dealloc(
        ptr,
        std::alloc::Layout::from_size_align_unchecked(size, align),
    );
}

/// Install the host's blocking region and safepoint.
pub fn set_blocking(run: BlockingFn, poll: PollFn) {
    BLOCKING.store(run as usize, Ordering::SeqCst);
    POLL.store(poll as usize, Ordering::SeqCst);
}

/// Run `f`, which waits and touches no heap object, with the thread
/// counted as stopped, so a collection does not wait for it. A heap
/// pointer that only `f`'s own frames hold is not kept alive. Calls `f`
/// at once when no host installed a region.
pub fn blocking<F: FnOnce() -> R, R>(f: F) -> R {
    let run = BLOCKING.load(Ordering::Relaxed);
    if run == 0 {
        return f();
    }
    struct Slot<F, R> {
        f: Option<F>,
        out: Option<std::thread::Result<R>>,
    }
    unsafe extern "C" fn call<F: FnOnce() -> R, R>(data: *mut c_void) {
        // SAFETY: `data` is the `Slot` that `blocking` below owns for the
        // whole call.
        let slot = unsafe { &mut *(data as *mut Slot<F, R>) };
        if let Some(f) = slot.f.take() {
            slot.out = Some(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)));
        }
    }
    let mut slot = Slot {
        f: Some(f),
        out: None,
    };
    // SAFETY: `run` came from `set_blocking`.
    let run: BlockingFn = unsafe { std::mem::transmute(run) };
    unsafe { run(call::<F, R>, &mut slot as *mut Slot<F, R> as *mut c_void) };
    match slot.out.take() {
        Some(Ok(r)) => r,
        Some(Err(panic)) => std::panic::resume_unwind(panic),
        None => unreachable!("the host's blocking region did not call back"),
    }
}

/// Stop here while a collection waits for this thread. A loop that
/// neither allocates nor blocks calls it, or it holds a collection up.
#[inline]
pub fn safepoint() {
    let poll = POLL.load(Ordering::Relaxed);
    if poll != 0 {
        // SAFETY: `poll` came from `set_blocking`.
        let poll: PollFn = unsafe { std::mem::transmute(poll) };
        unsafe { poll() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    static RUNS: AtomicUsize = AtomicUsize::new(0);
    static POLLS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn run(call: unsafe extern "C" fn(*mut c_void), data: *mut c_void) {
        RUNS.fetch_add(1, Ordering::SeqCst);
        unsafe { call(data) };
    }

    unsafe extern "C" fn poll() {
        POLLS.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn a_wait_runs_through_the_hosts_region() {
        set_blocking(run, poll);
        let runs = RUNS.load(Ordering::SeqCst);
        let polls = POLLS.load(Ordering::SeqCst);
        let held = String::from("kept");
        let out = blocking(|| held.len() + 3);
        assert_eq!(out, 7);
        assert!(RUNS.load(Ordering::SeqCst) > runs);
        safepoint();
        assert!(POLLS.load(Ordering::SeqCst) > polls);
        assert_eq!(held, "kept");
    }

    #[test]
    fn a_panic_in_a_wait_reaches_the_caller() {
        set_blocking(run, poll);
        let caught = std::panic::catch_unwind(|| blocking(|| -> u32 { panic!("from the wait") }));
        assert!(caught.is_err());
    }
}
