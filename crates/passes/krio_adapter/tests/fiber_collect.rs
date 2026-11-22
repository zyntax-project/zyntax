//! A worker thread's suspended fibers keep what they hold while another
//! thread collects.
//!
//! Each fiber builds a list that only its own stack refers to, yields, and
//! checks the list when it is resumed. The worker keeps no other
//! reference, and every collection starts on a different thread. A list
//! that is freed early is overwritten by the garbage that follows, so the
//! check after the resume fails.

#![cfg(unix)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use zyntax_compiler::fiber_backend::{FIBER_STEP_DONE, FIBER_STEP_YIELDED, unpack_fiber_step};
use zyntax_compiler::pool_alloc::zyntax_alloc;
use zyntax_compiler::{host_heap, immix_heap, zrtl};

#[repr(C)]
struct Node {
    next: *mut Node,
    tag: usize,
}

const LIST_TAG: usize = 7;
const GARBAGE: usize = 0xdead;
const FIBERS: usize = 8;
const LEN: usize = 100;

static INTACT: AtomicUsize = AtomicUsize::new(0);
static BROKEN: AtomicUsize = AtomicUsize::new(0);

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

/// A fiber body: the list lives only in this frame while the fiber is
/// suspended.
extern "C" fn body() {
    // SAFETY: the list is the program's heap, built and read here.
    unsafe {
        let head = chain(LEN, LIST_TAG);
        zrtl::krio_fiber_yield(1);
        if intact(black_box(head), LEN, LIST_TAG) {
            INTACT.fetch_add(1, Ordering::Relaxed);
        } else {
            BROKEN.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct Flags {
    ready: AtomicBool,
    resume: AtomicBool,
    polled: AtomicBool,
}

/// A thread of the program: it makes fibers, leaves them suspended and
/// waits, first in a blocking region and then at a safepoint.
fn worker(flags: Arc<Flags>) {
    immix_heap::attach();
    // SAFETY: handles are used on this thread only, and each is resumed
    // after `fiber_new` and before it is dropped.
    unsafe {
        let fibers: Vec<usize> = (0..FIBERS)
            .map(|_| zrtl::krio_fiber_new(body as *mut u8, 0) as usize)
            .collect();
        for &f in &fibers {
            let (tag, _) = unpack_fiber_step(zrtl::krio_fiber_resume(f as *mut _));
            assert_eq!(tag, FIBER_STEP_YIELDED);
        }
        flags.ready.store(true, Ordering::Release);
        while !flags.polled.load(Ordering::Acquire) {
            immix_heap::blocking(|| std::thread::sleep(Duration::from_millis(1)));
        }
        while !flags.resume.load(Ordering::Acquire) {
            immix_heap::safepoint();
            std::thread::sleep(Duration::from_millis(1));
        }
        for &f in &fibers {
            let (tag, _) = unpack_fiber_step(zrtl::krio_fiber_resume(f as *mut _));
            assert_eq!(tag, FIBER_STEP_DONE);
        }
    }
}

/// Allocate and drop lists until `n` more collections have run, so what
/// they free is reused for something else.
fn collect_until(n: usize) {
    let target = immix_heap::stats().collections + n;
    while immix_heap::stats().collections < target {
        black_box(unsafe { chain(500, GARBAGE) });
    }
}

#[test]
fn a_suspended_fiber_on_another_thread_survives_a_collection() {
    // SAFETY: the only test in this binary, so nothing else reads the
    // environment while it is set.
    unsafe {
        std::env::set_var("ZYNTAX_HOST_HEAP", "immix");
        std::env::set_var("ZYNTAX_IMMIX_THREADS", "1");
        std::env::set_var("ZYNTAX_IMMIX_FLOOR_KB", "256");
    }
    host_heap::install_from_env();
    assert!(immix_heap::active(), "the Immix heap was not installed");
    assert!(
        krio_adapter::fiber::install(),
        "a fiber backend was installed already"
    );

    let flags = Arc::new(Flags {
        ready: AtomicBool::new(false),
        resume: AtomicBool::new(false),
        polled: AtomicBool::new(false),
    });
    let handle = {
        let flags = flags.clone();
        std::thread::spawn(move || worker(flags))
    };
    while !flags.ready.load(Ordering::Acquire) {
        std::thread::yield_now();
    }

    // The worker waits in a blocking region.
    collect_until(20);
    flags.polled.store(true, Ordering::Release);
    // The worker waits at a safepoint.
    collect_until(20);
    flags.resume.store(true, Ordering::Release);
    handle.join().unwrap();

    assert_eq!(BROKEN.load(Ordering::Relaxed), 0, "a fiber's list changed");
    assert_eq!(INTACT.load(Ordering::Relaxed), FIBERS);
    assert_eq!(
        immix_heap::cancelled_stops(),
        0,
        "the worker was waited for"
    );
}
