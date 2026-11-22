//! The thread plugin hands heap pointers between threads while the heap
//! collects.
//!
//! A started thread gets a list through its argument and returns another
//! through its result. A producer queues lists in a channel. In each case
//! the only reference sits where no collector reads, a join handle or a
//! queue, while other threads allocate. A list that is freed early is
//! overwritten by the garbage that follows, so the check at the end fails.

#![cfg(unix)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use zrtl_thread::{
    channel_close_receiver, channel_new, channel_recv, channel_send_ptr, thread_join, thread_spawn,
};
use zyntax_compiler::pool_alloc::{
    zyntax_alloc, zyntax_attach, zyntax_blocking, zyntax_hold, zyntax_safepoint, zyntax_unhold,
};
use zyntax_compiler::{host_heap, immix_heap};

#[repr(C)]
struct Node {
    next: *mut Node,
    tag: usize,
}

const GARBAGE: usize = 0xdead;
const ARG_TAG: usize = 7;
const RESULT_TAG: usize = 9;
const QUEUED_TAG: usize = 1000;

unsafe fn chain(len: usize, tag: usize) -> *mut Node {
    let mut head: *mut Node = std::ptr::null_mut();
    for _ in 0..len {
        let node = zyntax_alloc(std::mem::size_of::<Node>()) as *mut Node;
        (*node).next = head;
        (*node).tag = tag;
        head = node;
    }
    head
}

unsafe fn intact(mut node: *mut Node, len: usize, tag: usize) -> bool {
    for _ in 0..len {
        if node.is_null() || (*node).tag != tag {
            return false;
        }
        node = (*node).next;
    }
    node.is_null()
}

/// Allocate and drop lists, so collections run and what they free is
/// reused for something else.
fn churn(lists: usize) {
    for _ in 0..lists {
        black_box(unsafe { chain(500, GARBAGE) });
    }
}

/// A wait the heap does not have to stop for.
fn wait(ms: u64) {
    zrtl::heap::blocking(|| std::thread::sleep(Duration::from_millis(ms)));
}

static BROKEN: AtomicUsize = AtomicUsize::new(0);

/// A started thread: its argument is a list that nothing else refers to.
extern "C" fn worker(arg: i64) -> i64 {
    // Wait without allocating, as a thread that only reads does. A
    // collection that does not know the thread would free the list.
    let began = Instant::now();
    while began.elapsed() < Duration::from_millis(30) {
        zrtl::heap::safepoint();
        std::hint::spin_loop();
    }
    unsafe {
        if !intact(arg as *mut Node, 100, ARG_TAG) {
            BROKEN.fetch_add(1, Ordering::Relaxed);
        }
        churn(20);
        if !intact(black_box(arg) as *mut Node, 100, ARG_TAG) {
            BROKEN.fetch_add(1, Ordering::Relaxed);
        }
        chain(60, RESULT_TAG) as i64
    }
}

#[inline(never)]
fn start_one() -> u64 {
    // SAFETY: a list of the program's heap, handed to the thread.
    let head = unsafe { chain(100, ARG_TAG) };
    thread_spawn(worker, head as i64)
}

#[inline(never)]
fn start_workers(n: usize) -> Vec<u64> {
    (0..n).map(|_| start_one()).collect()
}

/// A thread that queues lists and ends.
extern "C" fn producer(sender: i64) -> i64 {
    for i in 0..200 {
        let list = unsafe { chain(20, QUEUED_TAG + i) };
        channel_send_ptr(sender as u64, list as *const u8);
    }
    0
}

#[test]
fn pointers_handed_between_threads_survive_collections() {
    // SAFETY: the only test in this binary, so nothing else reads the
    // environment while it is set.
    unsafe {
        std::env::set_var("ZYNTAX_HOST_HEAP", "immix");
        std::env::set_var("ZYNTAX_IMMIX_THREADS", "1");
        std::env::set_var("ZYNTAX_IMMIX_FLOOR_KB", "256");
    }
    host_heap::install_from_env();
    assert!(immix_heap::active(), "the Immix heap was not installed");
    zrtl::heap::set_blocking(zyntax_blocking, zyntax_safepoint);
    zrtl::heap::set_roots(zyntax_attach, zyntax_hold, zyntax_unhold);
    assert_eq!(host_heap::hold_count(), 0);

    // Arguments and results, over several rounds.
    for _ in 0..8 {
        let handles = start_workers(16);
        // The threads start and run while this one collects.
        churn(400);
        // Once they are gone, only their join handles name the results.
        wait(100);
        churn(400);
        for handle in handles {
            let result = thread_join(handle);
            assert!(
                unsafe { intact(result as *mut Node, 60, RESULT_TAG) },
                "a thread's result was freed before the join"
            );
        }
    }
    assert_eq!(BROKEN.load(Ordering::Relaxed), 0, "a start-up argument");
    assert_eq!(host_heap::hold_count(), 0, "a hold was not released");

    // Messages that only a queue refers to.
    let pair = channel_new();
    let handle = thread_spawn(producer, pair.sender as i64);
    wait(100);
    churn(600);
    for i in 0..200 {
        let list = channel_recv(pair.receiver) as *mut Node;
        assert!(
            unsafe { intact(list, 20, QUEUED_TAG + i) },
            "message {i} was freed while it was queued"
        );
    }
    thread_join(handle);
    assert_eq!(host_heap::hold_count(), 0, "a received message stayed held");

    // A closed channel lets go of what it still queued.
    for i in 0..5 {
        channel_send_ptr(pair.sender, unsafe { chain(20, i) } as *const u8);
    }
    assert!(host_heap::hold_count() >= 5);
    channel_close_receiver(pair.receiver);
    assert_eq!(host_heap::hold_count(), 0, "a closed channel stayed held");

    assert!(immix_heap::stats().collections > 10, "no collections ran");
    assert_eq!(
        immix_heap::cancelled_stops(),
        0,
        "a started thread was waited for"
    );
}
