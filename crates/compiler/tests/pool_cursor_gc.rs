//! Collection must exclude unused reservations and invalidate them before reuse.
#![cfg(not(target_arch = "wasm32"))]

use std::collections::HashSet;
use zyntax_compiler::collector;
use zyntax_compiler::pool_alloc::{zyntax_alloc, zyntax_free};

static mut ROOTS: [usize; 2] = [0; 2];

#[test]
fn unused_reserved_blocks_are_free_and_sweeping_cannot_allocate_them_twice() {
    if std::env::var_os("ZYNTAX_DISABLE_GC").is_some() {
        return;
    }
    collector::enable();
    unsafe {
        let held = zyntax_alloc(32);
        std::ptr::write_bytes(held, 0x35, 32);
        ROOTS = [held as usize, held.add(32) as usize];
        collector::add_root_range(std::ptr::addr_of!(ROOTS).cast(), size_of::<[usize; 2]>());

        // The second root names an unused reserved slot, not a live object.
        collector::collect();
        assert_eq!(collector::stats().collections, 1);
        assert_eq!(collector::stats().live, 32);
        ROOTS[1] = 0;

        // Consume the swept remainder and cross into another slab. A surviving
        // fresh cursor would hand out the same blocks as the swept list.
        let mut taken = vec![0usize; 3000];
        collector::add_root_range(taken.as_ptr().cast(), taken.len() * size_of::<usize>());
        let mut distinct = HashSet::from([held as usize]);
        for (i, address) in taken.iter_mut().enumerate() {
            let block = zyntax_alloc(24);
            assert!(
                distinct.insert(block as usize),
                "block {i} was allocated twice"
            );
            // The class padding must be cleared even when the block is recycled.
            assert_eq!(*block.add(24).cast::<usize>(), 0);
            std::ptr::write_bytes(block, (i % 251) as u8, 24);
            *address = block as usize;
        }
        collector::collect();
        for (i, &address) in taken.iter().enumerate() {
            let block = address as *mut u8;
            assert_eq!(*block, (i % 251) as u8);
            assert_eq!(*block.add(23), (i % 251) as u8);
            zyntax_free(block);
        }
        assert_eq!(*held, 0x35);
        assert_eq!(*held.add(31), 0x35);
        zyntax_free(held);
        ROOTS = [0; 2];
        taken.fill(0);
        collector::collect();
        // A conservative stack root can retain the previously named unused slot.
        assert!(collector::stats().live <= 32);

        // A different size class must not overlap an old reservation.
        let other = zyntax_alloc(48);
        std::ptr::write_bytes(other, 0x79, 48);
        for address in &mut taken {
            let block = zyntax_alloc(32);
            assert!(block.add(32) <= other || block >= other.add(48));
            std::ptr::write_bytes(block, 0x21, 32);
            *address = block as usize;
        }
        for i in 0..48 {
            assert_eq!(*other.add(i), 0x79);
        }
        for &address in &taken {
            zyntax_free(address as *mut u8);
        }
        zyntax_free(other);
        collector::remove_root_range(taken.as_ptr().cast());
        collector::remove_root_range(std::ptr::addr_of!(ROOTS).cast());
    }
    collector::disable();
}
