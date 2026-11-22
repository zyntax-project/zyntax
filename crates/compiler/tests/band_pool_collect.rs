//! Collections run while the band pool's threads allocate.
//!
//! The dispatcher holds a list on its stack and waits for the bands.
//! Each band holds a list of its own and allocates garbage around it. A
//! collection started on any of those threads must stop the rest,
//! which are idle in the pool, running a band or waiting for the
//! dispatch, and must keep every list.

#![cfg(unix)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use zyntax_compiler::immix_heap;
use zyntax_compiler::pool_alloc::zyntax_alloc;
use zyntax_compiler::zrtl::{BandPool, zyntax_parallel_for};

#[repr(C)]
struct Node {
    next: *mut Node,
    tag: usize,
}

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

unsafe extern "C" fn band(lo: i64, hi: i64, env: *mut u8) {
    let broken = &*(env as *const AtomicUsize);
    for i in lo..hi {
        let keep = chain(50, i as usize);
        for _ in 0..200 {
            black_box(zyntax_alloc(256));
        }
        if !intact(black_box(keep), 50, i as usize) {
            broken.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[test]
fn every_list_survives_collections_started_on_any_pool_thread() {
    // SAFETY: the only test in this binary, so nothing else reads the
    // environment while it is set.
    unsafe {
        std::env::set_var("ZYNTAX_HOST_HEAP", "immix");
        std::env::set_var("ZYNTAX_IMMIX_THREADS", "1");
        std::env::set_var("ZYNTAX_IMMIX_FLOOR_KB", "256");
        std::env::set_var("ZYNTAX_PARALLEL_THREADS", "4");
    }
    zyntax_compiler::host_heap::install_from_env();
    assert!(immix_heap::active(), "the Immix heap was not installed");
    assert!(BandPool::shared().workers() > 1);

    let broken = AtomicUsize::new(0);
    let env = &broken as *const AtomicUsize as *mut u8;
    let held = unsafe { chain(100, 7) };
    for _ in 0..20 {
        unsafe { zyntax_parallel_for(0, 400, 1, band, env) };
        assert!(
            unsafe { intact(black_box(held), 100, 7) },
            "the dispatcher's list"
        );
    }

    assert_eq!(broken.load(Ordering::Relaxed), 0, "a band's list");
    assert!(immix_heap::stats().collections > 5, "no collections ran");
    assert_eq!(
        immix_heap::cancelled_stops(),
        0,
        "a pool thread was waited for"
    );
}
