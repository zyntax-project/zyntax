//! The collector over blocks above the pooled sizes, which live outside
//! the slabs and are reached through the registry.
//!
//! One test to a process: the collector belongs to the thread that
//! enabled it, and a second thread touching the pool turns it off.

#![cfg(not(target_arch = "wasm32"))]

use zyntax_compiler::collector;
use zyntax_compiler::pool_alloc::{zyntax_alloc, zyntax_free};

const LARGE: usize = 1 << 20;
const SENTINEL: usize = 0x5157_4E54_4158_0001;

static mut HOLD: usize = 0;

#[test]
fn a_reached_large_block_is_read_to_its_end_and_keeps_what_it_holds() {
    if std::env::var_os("ZYNTAX_DISABLE_GC").is_some() {
        return;
    }
    collector::enable();
    // SAFETY: blocks of the pool, written within their bounds; the
    // root range is a static that outlives the collection.
    unsafe {
        // A word past a request, left by the block's previous occupant,
        // names nothing: the chain it pointed at is reclaimed.
        let held = block_with_a_stale_tail();
        scrub_stack();
        collector::collect();
        assert_eq!(collector::stats().collections, 1);
        let live = collector::stats().live;
        assert!(
            live < CHAIN * 32 / 2,
            "the chain behind the stale tail word was kept: {live} bytes live"
        );
        assert_eq!(*(held as *const usize), SENTINEL);
        zyntax_free(held);

        let large = zyntax_alloc(LARGE);
        assert!(!large.is_null());
        std::ptr::write_bytes(large, 0, LARGE);
        let small = zyntax_alloc(32);
        assert!(!small.is_null());
        *(small as *mut usize) = SENTINEL;
        // The small block is held only by the large one, and the large
        // one only by the registered root.
        *(large as *mut usize) = small as usize;
        HOLD = large as usize;
        collector::add_root_range(std::ptr::addr_of!(HOLD) as *const u8, 8);

        collector::collect();

        let stats = collector::stats();
        assert_eq!(stats.collections, 2);
        assert!(
            stats.live >= LARGE + 32,
            "the large block and what it holds were not reached: {} bytes live",
            stats.live
        );
        assert_eq!(*(small as *const usize), SENTINEL);

        // Released by the program, the block is gone from the registry
        // and the next collection reaches nothing through the root.
        zyntax_free(large);
        HOLD = 0;
        collector::collect();
        assert_eq!(collector::stats().collections, 3);
        zyntax_free(small);
    }
    collector::disable();
}

/// Blocks in a chain that only a stale word names.
const CHAIN: usize = 4096;

/// Overwrite the stack below the caller, where the frames that built
/// the chain left its addresses; the collection reads that far down.
#[inline(never)]
fn scrub_stack() {
    let mut scratch = [0u8; 64 << 10];
    std::hint::black_box(&mut scratch);
}

/// A 24-byte request served from a block whose last word, past the
/// request, named a chain of [`CHAIN`] blocks. The chain's addresses
/// never reach the caller's frame.
#[inline(never)]
fn block_with_a_stale_tail() -> *mut u8 {
    // SAFETY: blocks of the pool, written within their slots.
    unsafe {
        let mut head = 0usize;
        for _ in 0..CHAIN {
            let node = zyntax_alloc(32);
            *(node as *mut usize) = head;
            head = node as usize;
        }
        let stale = zyntax_alloc(32);
        *(stale.add(24) as *mut usize) = head;
        zyntax_free(stale);
        let block = zyntax_alloc(24);
        assert_eq!(block, stale, "the freed block serves the request");
        *(block as *mut usize) = SENTINEL;
        std::hint::black_box(block)
    }
}
