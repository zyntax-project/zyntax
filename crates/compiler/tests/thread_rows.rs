//! Rows for the heap with several threads: four threads that build trees,
//! keep lists, pass lists through channels, and start and join other
//! threads. The threads come from the thread plugin.
//!
//! Run with `--ignored --nocapture`. `ONLY` names the row: `trees`,
//! `retain`, `pipeline` or `churn`. The row `retain1` is `retain` on one
//! thread. The test prints the row's checksum and its time.
//! `ZYNTAX_HOST_HEAP=immix` selects the Immix heap, and
//! `ZYNTAX_IMMIX_THREADS=0` stops it from collecting while several
//! threads run. A run with the default heap frees what it builds, as
//! that heap needs.

#![cfg(unix)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use zrtl_thread::{
    channel_close_sender, channel_new, channel_recv, channel_send_ptr, thread_join, thread_spawn,
};
use zyntax_compiler::host_heap;
use zyntax_compiler::pool_alloc::{
    zyntax_alloc, zyntax_attach, zyntax_blocking, zyntax_free, zyntax_hold, zyntax_safepoint,
    zyntax_unhold,
};

#[repr(C)]
struct Tree {
    left: *mut Tree,
    right: *mut Tree,
    value: usize,
}

#[repr(C)]
struct Node {
    next: *mut Node,
    tag: usize,
}

static FREE: AtomicBool = AtomicBool::new(false);

unsafe fn release(p: *mut u8) {
    if FREE.load(Ordering::Relaxed) {
        zyntax_free(p);
    }
}

unsafe fn build(depth: u32, value: usize) -> *mut Tree {
    let t = zyntax_alloc(std::mem::size_of::<Tree>()) as *mut Tree;
    (*t).value = value;
    if depth == 0 {
        (*t).left = std::ptr::null_mut();
        (*t).right = std::ptr::null_mut();
    } else {
        (*t).left = build(depth - 1, value);
        (*t).right = build(depth - 1, value);
    }
    t
}

/// The sum of the values, which frees the tree when the run frees.
unsafe fn check(t: *mut Tree) -> usize {
    if t.is_null() {
        return 0;
    }
    let sum = (*t).value + check((*t).left) + check((*t).right);
    release(t as *mut u8);
    sum
}

const THREADS: usize = 4;

/// Run `f` on four threads and add what they return.
fn on_four(f: extern "C" fn(i64) -> i64) -> i64 {
    let handles: Vec<u64> = (0..THREADS as i64).map(|i| thread_spawn(f, i)).collect();
    handles.into_iter().map(|h| thread_join(h)).sum()
}

/// A long-lived tree and many short ones on each thread.
extern "C" fn trees(index: i64) -> i64 {
    // SAFETY: trees of the program's heap, built and read here.
    unsafe {
        let long = build(15, index as usize + 1);
        let mut sum = 0;
        for i in 0..2000 {
            sum += check(build(10, i % 7 + 1));
        }
        sum += check(black_box(long));
        sum as i64
    }
}

/// A list that grows to many nodes while its thread allocates garbage.
extern "C" fn retain(_: i64) -> i64 {
    // SAFETY: nodes of the program's heap, built and read here.
    unsafe {
        let mut head: *mut Node = std::ptr::null_mut();
        for i in 0..300_000 {
            let node = zyntax_alloc(std::mem::size_of::<Node>()) as *mut Node;
            (*node).next = head;
            (*node).tag = i % 5 + 1;
            head = node;
            for _ in 0..8 {
                let garbage = zyntax_alloc(64);
                *garbage = 1;
                release(garbage);
            }
        }
        let mut sum = 0;
        let mut at = black_box(head);
        while !at.is_null() {
            sum += (*at).tag;
            at = (*at).next;
        }
        sum as i64
    }
}

const MESSAGES: usize = 20_000;

/// Queues lists of nodes in a channel, then closes it.
extern "C" fn produce(sender: i64) -> i64 {
    // SAFETY: nodes of the program's heap, built here.
    unsafe {
        for i in 0..MESSAGES {
            let mut head: *mut Node = std::ptr::null_mut();
            for _ in 0..20 {
                let node = zyntax_alloc(std::mem::size_of::<Node>()) as *mut Node;
                (*node).next = head;
                (*node).tag = i % 3 + 1;
                head = node;
            }
            channel_send_ptr(sender as u64, head as *const u8);
        }
    }
    channel_close_sender(sender as u64);
    0
}

/// Takes lists from a channel until it closes, and adds their tags.
extern "C" fn consume(receiver: i64) -> i64 {
    let mut sum = 0;
    loop {
        let message = channel_recv(receiver as u64);
        if message == i64::MIN {
            return sum as i64;
        }
        // SAFETY: a list a producer built.
        unsafe {
            let mut at = message as *mut Node;
            while !at.is_null() {
                sum += (*at).tag;
                let next = (*at).next;
                release(at as *mut u8);
                at = next;
            }
        }
    }
}

/// Two producers and two consumers.
fn pipeline() -> i64 {
    let pairs: Vec<_> = (0..THREADS / 2).map(|_| channel_new()).collect();
    let mut handles = Vec::new();
    for pair in &pairs {
        handles.push(thread_spawn(produce, pair.sender as i64));
        handles.push(thread_spawn(consume, pair.receiver as i64));
    }
    handles.into_iter().map(|h| thread_join(h)).sum()
}

extern "C" fn short_lived(index: i64) -> i64 {
    // SAFETY: a tree of the program's heap, built and read here.
    unsafe { check(build(8, index as usize + 1)) as i64 }
}

/// Four threads at a time, started and joined many times over.
fn churn() -> i64 {
    (0..500).map(|_| on_four(short_lived)).sum()
}

#[test]
#[ignore]
fn thread_rows() {
    let row = std::env::var("ONLY").unwrap_or_default();
    host_heap::install_from_env();
    FREE.store(
        std::env::var_os("ZYNTAX_HOST_HEAP").is_none(),
        Ordering::Relaxed,
    );
    zrtl::heap::set_blocking(zyntax_blocking, zyntax_safepoint);
    zrtl::heap::set_roots(zyntax_attach, zyntax_hold, zyntax_unhold);
    zyntax_compiler::immix_heap::attach();

    let began = Instant::now();
    let checksum = match row.as_str() {
        "trees" => on_four(trees),
        "retain" => on_four(retain),
        "retain1" => thread_join(thread_spawn(retain, 0)),
        "pipeline" => pipeline(),
        "churn" => churn(),
        _ => panic!("ONLY names a row: trees, retain, retain1, pipeline or churn"),
    };
    println!("{row} {checksum} {:.3}", began.elapsed().as_secs_f64());
}
