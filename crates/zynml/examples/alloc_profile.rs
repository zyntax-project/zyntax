//! Counts what one parse allocates.
//!
//! The parser's remaining cost is dominated by allocation: values are
//! cloned into the memo at every position, bindings are keyed by owned
//! strings, and a character class hands back a heap string per
//! character. Counting gives each of those a number to move rather
//! than an argument about which matters.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size(), Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNT.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size.saturating_sub(layout.size()), Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let grammar = zynml::Grammar2::from_source(zynml::ZYNML_GRAMMAR).expect("grammar");
    let source = zynml::ZYNML_STDLIB_PRELUDE;

    // Everything before this is grammar loading, which a real run does
    // once. Only the parses are counted.
    COUNT.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    let mut decls = 0usize;
    for _ in 0..iters {
        decls += grammar
            .parse_with_filename(source, "prelude.zynml")
            .expect("parse")
            .declarations
            .len();
    }
    let count = COUNT.load(Relaxed);
    let bytes = BYTES.load(Relaxed);

    eprintln!(
        "{iters} parses of {} bytes: {} allocations ({:.1} MB), {} per parse, {:.1} per source byte (decls={decls})",
        source.len(),
        count,
        bytes as f64 / (1024.0 * 1024.0),
        count / iters,
        count as f64 / (source.len() * iters) as f64,
    );
}
