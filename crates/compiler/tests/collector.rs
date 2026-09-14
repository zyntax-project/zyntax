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
        assert_eq!(stats.collections, 1);
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
        assert_eq!(collector::stats().collections, 2);
        zyntax_free(small);
    }
    collector::disable();
}
