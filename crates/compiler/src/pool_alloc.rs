//! A size-class pool behind the allocation intrinsics.
//!
//! `Intrinsic::Malloc` and `Intrinsic::Free` used to lower to calls to
//! libc's `malloc` and `free`. For a program that allocates in a loop
//! that is most of its runtime: the binary-trees kernel spends about
//! 87% of its time inside the allocator, measured by comparing it
//! against the same kernel carving nodes from a bump arena.
//!
//! A general allocator has to serve any size and return memory to the
//! OS. A language runtime allocating one small object at a time does
//! not, so this keeps a free list per size class, carves fresh slots
//! from slabs, and never gives a slab back. Freeing is pushing onto a
//! list and allocating is popping off one, which is what makes it
//! cheaper than the thing it replaces rather than merely different.
//!
//! Anything larger than [`MAX_POOLED`] goes to libc, since a pool that
//! keeps every size forever is a leak wearing a hat.
//!
//! Under `debug_assertions` a freed payload is overwritten with
//! [`POISON`]. A pool hands the same bytes back rather than unmapping
//! them, so without it a read through a stale pointer returns the old
//! contents and looks like a correct answer.
//!
//! ## The contract
//!
//! [`zyntax_free`] must only ever be handed a pointer from
//! [`zyntax_alloc`], which is the contract `free` already has with
//! `malloc`. A header in front of every block records which pool it
//! came from and carries a magic word, so a pointer from somewhere
//! else is passed to libc's `free` rather than corrupting a list. That
//! is a guard against a mistake, not a licence to mix them.
//!
//! ## Threads
//!
//! The lists are thread-local, so the common path takes no lock. A
//! block allocated on one thread and freed on another lands on the
//! freeing thread's list, which is safe because slabs are shared and
//! never unmapped; it migrates capacity between threads rather than
//! losing it. A large block carries its length in its own header for
//! the same reason — keeping it in a thread-local table beside the
//! block would have leaked exactly the cross-thread case this
//! paragraph promises works.

use std::alloc::{alloc as sys_alloc, dealloc as sys_dealloc, Layout};
use std::cell::Cell;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Largest request served from a pool. Above this, libc.
const MAX_POOLED: usize = 1024;

/// Size classes step by this, so class `i` holds `(i + 1) * STEP`
/// payload bytes.
const STEP: usize = 16;

/// Number of pools, covering `STEP..=MAX_POOLED`.
const CLASSES: usize = MAX_POOLED / STEP;

/// Bytes in front of every block. Sixteen rather than eight so the
/// payload keeps the sixteen-byte alignment libc hands out and vector
/// loads still land where they expect to.
const HEADER: usize = 16;

/// Marks a block as this allocator's. Chosen to be implausible as a
/// length, a pointer, or ASCII.
const MAGIC: u64 = 0x5A79_6E50_6F6F_6C01;

/// Written over a freed payload under `debug_assertions`, so a read
/// through a stale pointer is recognisable rather than plausible.
/// `0x55` repeats to `0x5555555555555555`, which is not a small
/// integer, not a mappable address on any target here, and not ASCII.
#[cfg(debug_assertions)]
const POISON: u8 = 0x55;

/// Bytes carved per slab. Large enough that carving is rare and small
/// enough that a program allocating once does not take a megabyte.
const SLAB: usize = 64 * 1024;

/// Set in a header's `class` to mark a block libc owns, with the rest
/// of the word carrying its total length.
///
/// The length has to live in the block, not beside it: a side table
/// would have to be thread-local to stay lock-free, and then a block
/// allocated on one thread and released on another would find nothing
/// in it and leak. A total length is far below this bit, so there is
/// room to keep both in one word.
const LARGE_MARK: usize = 1 << (usize::BITS - 1);

/// Written immediately before every payload.
#[repr(C)]
struct Header {
    magic: u64,
    /// Pool this block returns to, or `LARGE_MARK | total_bytes` when
    /// it came from libc.
    class: usize,
}

thread_local! {
    /// Head of each pool's free list. A block on a list stores the
    /// next pointer in its payload, which is why a class must be at
    /// least a pointer wide.
    ///
    /// `Cell` rather than `RefCell`: this is the hot path, and a
    /// borrow flag checked twice per allocation is a real share of
    /// what the pool exists to save.
    static FREE: [Cell<*mut u8>; CLASSES] =
        const { [const { Cell::new(std::ptr::null_mut()) }; CLASSES] };
    /// The slab being carved, and how much of it is spoken for.
    static SLAB_PTR: Cell<*mut u8> = const { Cell::new(std::ptr::null_mut()) };
    static SLAB_USED: Cell<usize> = const { Cell::new(SLAB) };
}

/// Requests the pools have served, across every thread.
///
/// The behaviour tests below prove what a pool does with a block. They
/// cannot prove a compiled program's allocations arrive here at all,
/// and a pool nothing reaches would pass every one of them. This is
/// what an end-to-end test reads to assert the path rather than the
/// behaviour.
#[cfg(debug_assertions)]
static SERVED: AtomicUsize = AtomicUsize::new(0);

/// How many requests the pools have served. Debug builds only.
#[cfg(debug_assertions)]
pub fn pooled_allocation_count() -> usize {
    SERVED.load(Ordering::Relaxed)
}

/// Size class for a payload, or `None` when libc should take it.
#[inline]
fn class_of(size: usize) -> Option<usize> {
    if size == 0 || size > MAX_POOLED {
        return None;
    }
    Some((size - 1) / STEP)
}

/// Total bytes a block in `class` occupies, header included.
#[inline]
fn slot_bytes(class: usize) -> usize {
    HEADER + (class + 1) * STEP
}

/// Allocate `size` bytes. Never returns null for a request libc could
/// have served.
///
/// # Safety
/// The returned pointer is valid for `size` bytes and must be released
/// with [`zyntax_free`].
#[no_mangle]
pub unsafe extern "C" fn zyntax_alloc(size: usize) -> *mut u8 {
    let Some(class) = class_of(size) else {
        return large_alloc(size);
    };
    #[cfg(debug_assertions)]
    SERVED.fetch_add(1, Ordering::Relaxed);

    // A block already on this pool's list.
    let reused = FREE.with(|lists| {
        let head = lists[class].get();
        if head.is_null() {
            return std::ptr::null_mut();
        }
        // The next pointer lives in the payload of the free block.
        lists[class].set(*(head as *mut *mut u8));
        head
    });
    if !reused.is_null() {
        return payload_of(reused, class);
    }

    // Otherwise carve one, taking a fresh slab if this one cannot fit.
    let want = slot_bytes(class);
    let block = SLAB_PTR.with(|sp| {
        SLAB_USED.with(|su| {
            let mut used = su.get();
            if used + want > SLAB {
                let slab = sys_alloc(Layout::from_size_align_unchecked(SLAB, HEADER));
                if slab.is_null() {
                    return std::ptr::null_mut();
                }
                sp.set(slab);
                used = 0;
            }
            let block = sp.get().add(used);
            su.set(used + want);
            block
        })
    });
    if block.is_null() {
        // Out of memory for a slab; the request itself may still fit.
        return large_alloc(size);
    }
    payload_of(block, class)
}

/// Stamp a block's header and hand back the payload.
#[inline]
unsafe fn payload_of(block: *mut u8, class: usize) -> *mut u8 {
    let head = block as *mut Header;
    (*head).magic = MAGIC;
    (*head).class = class;
    block.add(HEADER)
}

/// Anything a pool will not take.
#[inline]
unsafe fn large_alloc(size: usize) -> *mut u8 {
    let total = HEADER + size.max(1);
    let block = sys_alloc(Layout::from_size_align_unchecked(total, HEADER));
    if block.is_null() {
        return std::ptr::null_mut();
    }
    let head = block as *mut Header;
    (*head).magic = MAGIC;
    (*head).class = LARGE_MARK | total;
    block.add(HEADER)
}

/// Release a pointer from [`zyntax_alloc`].
///
/// # Safety
/// `ptr` must have come from [`zyntax_alloc`] and must not be used
/// afterwards. A null pointer is ignored, as `free`'s is.
#[no_mangle]
pub unsafe extern "C" fn zyntax_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    let block = ptr.sub(HEADER);
    let head = block as *const Header;
    if (*head).magic != MAGIC {
        // Not ours. Hand it to the allocator that most likely owns it
        // rather than threading a foreign block onto a free list.
        libc_free(ptr);
        return;
    }
    let class = (*head).class;
    if class & LARGE_MARK != 0 {
        let total = class & !LARGE_MARK;
        sys_dealloc(block, Layout::from_size_align_unchecked(total, HEADER));
        return;
    }
    // A freed block keeps its bytes, so a read through a stale pointer
    // returns the old contents: plausible, wrong, and silent. Overwrite
    // the payload with a byte that is none of a small integer, a valid
    // pointer, or ASCII, so such a read is recognisable instead. Debug
    // only — the whole point of the pool is that freeing is a push.
    #[cfg(debug_assertions)]
    std::ptr::write_bytes(block.add(HEADER), POISON, (class + 1) * STEP);

    FREE.with(|lists| {
        // Threaded through the header, not the payload: the first word
        // of the block is the magic, which a freed block no longer
        // needs, and leaving the payload alone is what lets it carry
        // the poison above.
        *(block as *mut *mut u8) = lists[class].get();
        lists[class].set(block);
    });
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn libc_free(ptr: *mut u8) {
    extern "C" {
        fn free(p: *mut core::ffi::c_void);
    }
    free(ptr as *mut core::ffi::c_void);
}

/// wasm32 links no libc, so there is nothing to hand a foreign block
/// back to. This path is only reached by a pointer that did not come
/// from here, which is a mistake somewhere else; leaking it is what
/// there is to do, and it beats guessing a layout for
/// `dealloc` and corrupting the allocator that does own it.
#[cfg(target_arch = "wasm32")]
unsafe fn libc_free(_ptr: *mut u8) {}

/// The symbols JIT'd code calls, for registration alongside the other
/// runtime groups.
pub fn alloc_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("zyntax_alloc", zyntax_alloc as *const u8),
        ("zyntax_free", zyntax_free as *const u8),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block comes back usable, and its bytes are its own.
    #[test]
    fn an_allocation_is_writable_and_distinct() {
        unsafe {
            let a = zyntax_alloc(24);
            let b = zyntax_alloc(24);
            assert!(!a.is_null() && !b.is_null());
            assert_ne!(a, b);
            std::ptr::write_bytes(a, 0xAA, 24);
            std::ptr::write_bytes(b, 0xBB, 24);
            assert_eq!(*a, 0xAA);
            assert_eq!(*b, 0xBB);
            zyntax_free(a);
            zyntax_free(b);
        }
    }

    /// A freed block is handed out again, which is the whole point.
    #[test]
    fn a_freed_block_is_reused() {
        unsafe {
            let a = zyntax_alloc(24);
            zyntax_free(a);
            let b = zyntax_alloc(24);
            assert_eq!(a, b, "the pool should hand back the block it just took");
            zyntax_free(b);
        }
    }

    /// Payloads stay sixteen-byte aligned, which vector loads assume.
    #[test]
    fn every_payload_is_aligned() {
        unsafe {
            for size in [1usize, 8, 16, 17, 24, 64, 255, 1024] {
                let p = zyntax_alloc(size);
                assert_eq!(p as usize % 16, 0, "size {size} came back misaligned");
                zyntax_free(p);
            }
        }
    }

    /// Sizes past the pools still work, and do not go on a list.
    #[test]
    fn a_large_request_round_trips() {
        unsafe {
            let big = MAX_POOLED * 4;
            let p = zyntax_alloc(big);
            assert!(!p.is_null());
            std::ptr::write_bytes(p, 0xCD, big);
            assert_eq!(*p.add(big - 1), 0xCD);
            zyntax_free(p);
        }
    }

    /// Distinct sizes land in distinct classes and do not alias.
    #[test]
    fn classes_do_not_collide() {
        unsafe {
            let mut live = vec![];
            for size in 1..=128usize {
                let p = zyntax_alloc(size);
                std::ptr::write_bytes(p, (size & 0xFF) as u8, size);
                live.push((p, size));
            }
            for (p, size) in &live {
                assert_eq!(**p, (*size & 0xFF) as u8, "size {size} was overwritten");
            }
            for (p, _) in live {
                zyntax_free(p);
            }
        }
    }

    /// A read through a pointer that was already freed is
    /// recognisable rather than plausible.
    ///
    /// The pool hands the same bytes back instead of unmapping them,
    /// so before the poison a stale read returned whatever the block
    /// last held — a correct-looking answer with nothing to notice.
    ///
    /// This reads freed memory on purpose. It is safe here because a
    /// slab is never returned, so the page is still mapped; that is
    /// exactly why the bug it guards is invisible without help.
    ///
    /// Reads past the header deliberately: the first word of a freed
    /// block is the free-list link, which follows the address and so
    /// differs every run.
    ///
    /// Every size here is checked to be one a pool actually serves,
    /// including the last one it takes. An instrument that tests a
    /// size the allocator hands to libc reports on the path it did not
    /// change, and passes while seeing none of the memory this owns.
    #[cfg(debug_assertions)]
    #[test]
    fn a_freed_payload_is_poisoned() {
        unsafe {
            for size in [16usize, 24, 64, 512, MAX_POOLED] {
                assert!(
                    class_of(size).is_some(),
                    "size {size} is not pooled, so poisoning it proves nothing"
                );
                let p = zyntax_alloc(size);
                std::ptr::write_bytes(p, 0x11, size);
                assert_eq!(*p, 0x11, "size {size} was not writable before free");
                zyntax_free(p);

                // Every byte, not just the first: a partial overwrite
                // would still leave stale data to read.
                for i in 0..size {
                    assert_eq!(
                        *p.add(i),
                        POISON,
                        "byte {i} of a freed {size}-byte payload still holds \
                         what it held before"
                    );
                }
            }
        }
    }

    /// And the first size past the cap is not pooled, which is what
    /// makes the sizes above meaningful.
    ///
    /// A block libc owns is released rather than kept, so there is
    /// nothing to poison and no stale read to guard. This pins the
    /// boundary so that raising `MAX_POOLED` cannot quietly move the
    /// test above it.
    #[test]
    fn the_cap_is_where_the_pool_stops() {
        assert!(class_of(MAX_POOLED).is_some(), "the cap itself is pooled");
        assert!(
            class_of(MAX_POOLED + 1).is_none(),
            "one byte past the cap must go to libc, or the poison tests \
             above are exercising a path the pool does not own"
        );
    }

    /// A large block released on a different thread than took it is
    /// actually released.
    ///
    /// Its length used to live in a thread-local table beside the
    /// block, so the freeing thread looked it up, found nothing, and
    /// returned without deallocating. Nothing crashed and nothing
    /// reported it; the memory was simply gone. The length lives in
    /// the block's own header now, which is why this passes.
    ///
    /// Measured by resident size because a leak has no other symptom.
    #[test]
    fn a_large_block_freed_on_another_thread_is_released() {
        fn rss_kb() -> i64 {
            let out = std::process::Command::new("ps")
                .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
                .expect("ps");
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .unwrap_or(0)
        }

        const BLOCK: usize = 64 * 1024;
        const PER_ROUND: usize = 512;

        // One round first, so the baseline includes whatever the
        // allocator keeps for itself rather than counting it as growth.
        let round = || {
            let taken: Vec<usize> = (0..PER_ROUND)
                .map(|_| unsafe { zyntax_alloc(BLOCK) } as usize)
                .collect();
            std::thread::spawn(move || {
                for p in taken {
                    unsafe { zyntax_free(p as *mut u8) };
                }
            })
            .join()
            .expect("freeing thread");
        };
        round();
        let base = rss_kb();
        for _ in 0..4 {
            round();
        }
        let grew = rss_kb() - base;

        // Four rounds leak 128 MB if nothing is released. Half a round
        // of slack absorbs allocator bookkeeping and the test harness.
        let budget = (PER_ROUND * BLOCK / 2 / 1024) as i64;
        assert!(
            grew < budget,
            "resident size grew {grew} KiB over four rounds, budget {budget} KiB — \
             large blocks freed off-thread are not coming back"
        );
    }

    /// Churn at one size stays bounded, since blocks come back.
    #[test]
    fn churn_reuses_rather_than_growing() {
        unsafe {
            let first = zyntax_alloc(32);
            zyntax_free(first);
            for _ in 0..10_000 {
                let p = zyntax_alloc(32);
                assert_eq!(p, first, "every round should reuse the one block");
                zyntax_free(p);
            }
        }
    }
}
