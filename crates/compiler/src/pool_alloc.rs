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
//! `malloc`. A block carries no header of its own: a slab serves one
//! size class and says so at its own base, and a block's slab is the
//! address it masks down to. A pointer from somewhere else fails the
//! magic word there, and then again in front of itself where a large
//! block keeps one, and is passed to libc's `free` rather than
//! corrupting a list. That is a guard against a mistake, not a licence
//! to mix them.
//!
//! What the guard does not cover: reading that magic word means reading
//! through an address derived from the pointer, at its slab base and at
//! sixteen bytes in front of it. For a pointer far enough from anything
//! mapped, the read itself faults before any word can be compared.
//! [`could_be_ours`] rules out the one case that is certain rather than
//! unlucky, which is an address below a single slab: those all mask to
//! zero. Ruling out the rest needs slabs carved from one reserved range
//! so that membership is a comparison, and that is a different design
//! rather than another check.
//!
//! Segregating by slab is what makes a small object cheap. A header
//! per block cost sixteen bytes on top of a class that was already
//! rounded up, so a twenty-four byte node moved forty-eight bytes for
//! twenty-four bytes of program data. It now moves thirty-two, and the
//! two words that said which pool it belonged to are not written at
//! all.
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
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

/// Largest request served from a pool. Above this, libc.
const MAX_POOLED: usize = 1024;

/// Size classes step by this, so class `i` holds `(i + 1) * STEP`
/// payload bytes.
const STEP: usize = 16;

/// Number of pools, covering `STEP..=MAX_POOLED`.
const CLASSES: usize = MAX_POOLED / STEP;

/// Bytes at the front of a *slab*, not of a block.
///
/// A block used to carry its own sixteen bytes saying which pool it
/// belonged to, which cost more than it sounds: a twenty-four byte node
/// took a thirty-two byte class plus sixteen, so forty-eight bytes moved
/// for twenty-four bytes of program data. A slab now serves one class
/// and says so once at its own base, and a block is found back to it by
/// masking its address. Sixteen keeps every payload sixteen-aligned,
/// since a slab is aligned to its own size and every class is a
/// multiple of sixteen.
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

/// Bytes carved per slab, and its alignment: a block's slab is found by
/// masking the block's address down to this, so it has to be a power of
/// two and slabs have to be aligned to it.
const SLAB: usize = 64 * 1024;

/// A slab header's second word: the size class in the low half and the
/// bytes carved so far, header included, in the half above.
const CLASS_MASK: usize = 0xFFFF_FFFF;
const USED_SHIFT: u32 = 32;

/// Set in a header's `class` to mark a block libc owns, with the rest
/// of the word carrying its total length.
///
/// The length has to live in the block, not beside it: a side table
/// would have to be thread-local to stay lock-free, and then a block
/// allocated on one thread and released on another would find nothing
/// in it and leak. A total length is far below this bit, so there is
/// room to keep both in one word.
const LARGE_MARK: usize = 1 << (usize::BITS - 1);

/// Written at the base of every slab, once, and read back by masking a
/// block's address. Also written in front of a large block, where
/// `class` carries `LARGE_MARK | total_bytes` instead.
#[repr(C)]
struct Header {
    magic: u64,
    class: usize,
}

impl Header {
    /// A slab's size class and how many bytes of it are carved.
    #[inline]
    fn slab_class_and_used(&self) -> (usize, usize) {
        (self.class & CLASS_MASK, self.class >> USED_SHIFT)
    }
}

/// The slab a block belongs to. Only meaningful for a pooled block.
#[inline]
fn slab_of(ptr: *mut u8) -> *mut Header {
    ((ptr as usize) & !(SLAB - 1)) as *mut Header
}

/// The size class and carved extent of the slab at `slab`, which must
/// be one this pool made.
///
/// # Safety
/// `slab` must be the base of a live slab.
pub(crate) unsafe fn slab_layout(slab: usize) -> (usize, usize) {
    (*(slab as *const Header)).slab_class_and_used()
}

/// Bytes at the front of a slab before its first block.
pub(crate) const SLAB_HEADER: usize = HEADER;
/// Bytes per slab and its alignment.
pub(crate) const SLAB_BYTES: usize = SLAB;
/// Bytes a block in `class` occupies; see [`slot_bytes`].
pub(crate) fn class_slot_bytes(class: usize) -> usize {
    slot_bytes(class)
}

/// Every block on this thread's free lists, by address.
pub(crate) fn for_each_free_block(mut f: impl FnMut(usize)) {
    FREE.with(|lists| {
        for list in lists.iter() {
            let mut p = list.get();
            while !p.is_null() {
                f(p as usize);
                // SAFETY: a block on a free list holds the next block
                // in its first word.
                p = unsafe { *(p as *mut *mut u8) };
            }
        }
    });
}

/// Empty this thread's free lists; the collector fills them again from
/// what its sweep finds unreached.
pub(crate) fn clear_free_lists() {
    FREE.with(|lists| {
        for list in lists.iter() {
            list.set(std::ptr::null_mut());
        }
    });
}

/// Put a chain of blocks of `class`, threaded through their first
/// words from `head` to `tail`, at the front of this thread's free
/// list.
///
/// # Safety
/// Every block on the chain must be one of `class` from this pool
/// that nothing reads any more.
pub(crate) unsafe fn push_free_chain(class: usize, head: usize, tail: usize) {
    FREE.with(|lists| {
        *(tail as *mut *mut u8) = lists[class].get();
        lists[class].set(head as *mut u8);
    });
}

/// Overwrite a block's payload with the debug poison; nothing in a
/// release build.
pub(crate) unsafe fn poison(block: usize, class: usize) {
    #[cfg(debug_assertions)]
    std::ptr::write_bytes(block as *mut u8, POISON, slot_bytes(class));
    #[cfg(not(debug_assertions))]
    let _ = (block, class);
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
    /// The slab being carved for each class, and how much of it is
    /// spoken for. One per class now: a slab serves a single class, so
    /// that masking a block's address back to it says which.
    static SLAB_PTR: [Cell<*mut u8>; CLASSES] =
        const { [const { Cell::new(std::ptr::null_mut()) }; CLASSES] };
    static SLAB_USED: [Cell<usize>; CLASSES] = const { [const { Cell::new(SLAB) }; CLASSES] };
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

/// Blocks past the cap that libc has handed out and not taken back.
///
/// A leaked large block used to be visible only as resident size, which
/// is a property of the whole process: every other test allocating
/// beside this one lands inside the measurement, so the reading said as
/// much about what else was running as about the pool. This counts the
/// thing itself.
#[cfg(debug_assertions)]
static LARGE_LIVE: AtomicIsize = AtomicIsize::new(0);

/// Large blocks taken and not yet released. Debug builds only.
#[cfg(debug_assertions)]
pub fn large_blocks_live() -> isize {
    LARGE_LIVE.load(Ordering::Relaxed)
}

/// Size class for a payload, or `None` when libc should take it.
#[inline]
fn class_of(size: usize) -> Option<usize> {
    if size == 0 || size > MAX_POOLED {
        return None;
    }
    Some((size - 1) / STEP)
}

/// Bytes a block in `class` occupies. No header: the class is a
/// property of the slab it came from.
#[inline]
fn slot_bytes(class: usize) -> usize {
    (class + 1) * STEP
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
        return reused;
    }

    // Nothing to reuse means the heap grows. The collector gets its
    // say first: what it finds unreached goes on the lists, and a
    // list that filled up serves the request after all.
    #[cfg(not(target_arch = "wasm32"))]
    if crate::collector::wants_collection() {
        crate::collector::collect();
        let reused = FREE.with(|lists| {
            let head = lists[class].get();
            if head.is_null() {
                return std::ptr::null_mut();
            }
            lists[class].set(*(head as *mut *mut u8));
            head
        });
        if !reused.is_null() {
            return reused;
        }
    }

    // Otherwise carve one, taking a fresh slab for this class if the
    // current one cannot fit. A slab is aligned to its own size so that
    // masking any block in it lands on its header.
    let want = slot_bytes(class);
    let block = SLAB_PTR.with(|sp| {
        SLAB_USED.with(|su| {
            let mut used = su[class].get();
            if used + want > SLAB {
                let slab = sys_alloc(Layout::from_size_align_unchecked(SLAB, SLAB));
                if slab.is_null() {
                    return std::ptr::null_mut();
                }
                // Said once per slab rather than once per block.
                let head = slab as *mut Header;
                (*head).magic = MAGIC;
                (*head).class = class | (HEADER << USED_SHIFT);
                sp[class].set(slab);
                used = HEADER;
                #[cfg(not(target_arch = "wasm32"))]
                crate::collector::note_slab(slab as usize);
            }
            let slab = sp[class].get();
            let block = slab.add(used);
            su[class].set(used + want);
            // The header keeps the carved extent too, so a reader that
            // only has the slab knows where its blocks end.
            (*(slab as *mut Header)).class = class | ((used + want) << USED_SHIFT);
            block
        })
    });
    #[cfg(not(target_arch = "wasm32"))]
    crate::collector::note_carved(want);
    if block.is_null() {
        // Out of memory for a slab; the request itself may still fit.
        return large_alloc(size);
    }
    block
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
    #[cfg(debug_assertions)]
    LARGE_LIVE.fetch_add(1, Ordering::Relaxed);
    let payload = block.add(HEADER);
    #[cfg(not(target_arch = "wasm32"))]
    crate::collector::note_large(payload as usize, total);
    payload
}

/// Release a large block by its payload address, for the collector.
///
/// # Safety
/// `payload` must have come from [`large_alloc`] and be unreached.
pub(crate) unsafe fn free_large(payload: usize) {
    let block = (payload as *mut u8).sub(HEADER);
    let total = (*(block as *const Header)).class & !LARGE_MARK;
    sys_dealloc(block, Layout::from_size_align_unchecked(total, HEADER));
    #[cfg(debug_assertions)]
    LARGE_LIVE.fetch_sub(1, Ordering::Relaxed);
}

/// Whether an address could have come from here at all.
///
/// A slab is taken from the system allocator aligned to its own size,
/// so it starts at or above [`SLAB`], and every block sits inside one.
/// Nothing this pool hands out is below that, and the bound therefore
/// turns no real block away.
///
/// It matters because of how a block is found back to its slab. Masking
/// an address down to the slab size sends everything in the first slab's
/// worth of address space to zero, so reading the header there is a
/// fault rather than a wrong guess. That is exactly where a small
/// integer mistaken for a pointer lands.
#[inline]
fn could_be_ours(ptr: *mut u8) -> bool {
    (ptr as usize) >= SLAB
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
    // A thread that frees will allocate from its own list without
    // ever carving, which is the other way a second mutator appears.
    #[cfg(not(target_arch = "wasm32"))]
    if crate::collector::is_enabled() {
        crate::collector::note_thread();
    }
    // An address no allocator returns is not passed on to one either:
    // libc would fault on it for its own reasons and the report would
    // name libc rather than whatever produced it. Loud where a
    // developer can see it, and nothing in a release build, because a
    // release that aborts on a bad pointer turns a leak into a crash.
    if !could_be_ours(ptr) {
        debug_assert!(
            false,
            "release of {ptr:p}, which is too low to have come from any \
             allocator: whatever produced it is holding a value that is \
             not a pointer"
        );
        return;
    }
    // The sixteen bytes in front of the payload are read first, and the
    // order is the point rather than a preference. They are always the
    // program's own memory for anything this pool handed out: a large
    // block's payload starts sixteen bytes into its allocation, and a
    // pooled block's slab begins at or before the same place, because a
    // slab spends its own first sixteen bytes on a header. Masking to
    // the slab is what cannot be done first. A large block is taken
    // from the system allocator wherever it likes, so its address masks
    // down to as much as a slab's width in front of it, and that is
    // memory this process may never have asked for.
    let head = ptr.sub(HEADER) as *const Header;
    if (*head).magic == MAGIC && (*head).class & LARGE_MARK != 0 {
        let total = (*head).class & !LARGE_MARK;
        #[cfg(not(target_arch = "wasm32"))]
        crate::collector::forget_large(ptr as usize);
        sys_dealloc(
            ptr.sub(HEADER),
            Layout::from_size_align_unchecked(total, HEADER),
        );
        #[cfg(debug_assertions)]
        LARGE_LIVE.fetch_sub(1, Ordering::Relaxed);
        return;
    }

    // Not a large block, so it is pooled or foreign, and either way the
    // mask now lands on a slab this pool wrote or on nothing. Reading
    // in front of a pooled block can only have found another block's
    // payload or the slab's own header, neither of which carries the
    // large mark, so arriving here says nothing was mistaken above.
    let slab = slab_of(ptr);
    let class = if (*slab).magic == MAGIC && (*slab).class & LARGE_MARK == 0 {
        (*slab).class & CLASS_MASK
    } else {
        // Hand it to the allocator that most likely owns it rather than
        // threading a foreign block onto a free list.
        libc_free(ptr);
        return;
    };
    let block = ptr;
    // A freed block keeps its bytes, so a read through a stale pointer
    // returns the old contents: plausible, wrong, and silent. Overwrite
    // the payload with a byte that is none of a small integer, a valid
    // pointer, or ASCII, so such a read is recognisable instead. Debug
    // only — the whole point of the pool is that freeing is a push.
    #[cfg(debug_assertions)]
    std::ptr::write_bytes(block, POISON, slot_bytes(class));

    FREE.with(|lists| {
        // Threaded through the block's own first word. A freed block
        // holds nothing a live one needed, and the poison above is
        // overwritten here for that word alone, which is why the test
        // for it reads past the first pointer.
        *(block as *mut *mut u8) = lists[class].get();
        lists[class].set(block);
    });
}

/// The usable size of a block from [`zyntax_alloc`], or `None` for a
/// pointer this pool did not hand out.
///
/// # Safety
/// `ptr` must have come from [`zyntax_alloc`].
unsafe fn usable_size(ptr: *mut u8) -> Option<usize> {
    if !could_be_ours(ptr) {
        return None;
    }
    let head = ptr.sub(HEADER) as *const Header;
    if (*head).magic == MAGIC && (*head).class & LARGE_MARK != 0 {
        return Some(((*head).class & !LARGE_MARK) - HEADER);
    }
    let slab = slab_of(ptr);
    if (*slab).magic == MAGIC && (*slab).class & LARGE_MARK == 0 {
        Some(slot_bytes((*slab).class & CLASS_MASK))
    } else {
        None
    }
}

/// Resize a block from [`zyntax_alloc`] to `new_size` bytes, keeping
/// its contents up to the smaller of the two sizes. A block that already
/// has room is returned as it is; a null pointer allocates.
///
/// # Safety
/// `ptr` must be null or have come from [`zyntax_alloc`], and must not
/// be used afterwards except through the returned pointer.
#[no_mangle]
pub unsafe extern "C" fn zyntax_realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
    if ptr.is_null() {
        return zyntax_alloc(new_size);
    }
    let Some(have) = usable_size(ptr) else {
        // Not this pool's: let the allocator that owns it resize it.
        return libc_realloc(ptr, new_size);
    };
    if have >= new_size {
        return ptr;
    }
    let fresh = zyntax_alloc(new_size);
    if fresh.is_null() {
        return fresh;
    }
    std::ptr::copy_nonoverlapping(ptr, fresh, have.min(new_size));
    zyntax_free(ptr);
    fresh
}

#[cfg(not(target_arch = "wasm32"))]
unsafe fn libc_realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
    extern "C" {
        fn realloc(p: *mut core::ffi::c_void, size: usize) -> *mut core::ffi::c_void;
    }
    realloc(ptr as *mut core::ffi::c_void, new_size) as *mut u8
}

#[cfg(target_arch = "wasm32")]
unsafe fn libc_realloc(ptr: *mut u8, new_size: usize) -> *mut u8 {
    let fresh = zyntax_alloc(new_size);
    if !fresh.is_null() {
        std::ptr::copy_nonoverlapping(ptr, fresh, new_size);
    }
    fresh
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
        ("zyntax_realloc", zyntax_realloc as *const u8),
        ("zyntax_free", zyntax_free as *const u8),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Growing keeps the bytes, whether the block stays pooled or
    /// crosses into a large allocation.
    #[test]
    fn realloc_keeps_contents_across_classes() {
        unsafe {
            let p = zyntax_alloc(24);
            for i in 0..24u8 {
                *p.add(i as usize) = i;
            }
            let q = zyntax_realloc(p, 40);
            for i in 0..24u8 {
                assert_eq!(*q.add(i as usize), i);
            }
            let big = zyntax_realloc(q, 100_000);
            for i in 0..24u8 {
                assert_eq!(*big.add(i as usize), i);
            }
            let same = zyntax_realloc(big, 50_000);
            assert_eq!(same, big, "shrinking keeps the block");
            zyntax_free(same);
            assert!(!zyntax_realloc(std::ptr::null_mut(), 8).is_null());
        }
    }

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
    /// Skips the first word deliberately: a freed block is threaded
    /// onto its list through its own first word, which holds an address
    /// and so differs every run. Everything after it is the poison.
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

                // Every byte after the list link, not just one: a
                // partial overwrite would still leave stale data to
                // read.
                for i in std::mem::size_of::<*mut u8>()..size {
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

    /// A block past the cap is not mistaken for one inside it.
    ///
    /// Without a header per block, a pooled block is recognised by
    /// masking its address down to its slab and reading the magic
    /// there. A large block is not in a slab, and the address it masks
    /// down to is whatever libc put there, including, when the block
    /// happens to start on a slab boundary, its own header. That is why
    /// the slab test also refuses a header marked large: reaching the
    /// wrong arm would thread a libc block onto a free list and hand it
    /// back as a small one.
    ///
    /// Interleaved so the two kinds are adjacent in the order they are
    /// taken and released, which is when a confusion between them
    /// shows.
    #[test]
    fn a_large_block_is_not_taken_for_a_pooled_one() {
        unsafe {
            let mut blocks = Vec::new();
            for i in 0..64usize {
                // Alternating, and each written with a byte derived
                // from its index so a block handed out twice is a
                // mismatch rather than a crash.
                let size = if i % 2 == 0 { 48 } else { MAX_POOLED + 64 };
                let p = zyntax_alloc(size);
                assert!(!p.is_null(), "allocation {i} of {size} bytes failed");
                std::ptr::write_bytes(p, i as u8, size);
                blocks.push((p, size, i as u8));
            }
            for (p, size, tag) in &blocks {
                for j in 0..*size {
                    assert_eq!(
                        *p.add(j),
                        *tag,
                        "byte {j} of the block tagged {tag} was written by another"
                    );
                }
            }
            for (p, _, _) in blocks {
                zyntax_free(p);
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

    /// An address too small to be one of these is refused before it is
    /// read through.
    ///
    /// A block is found back to its slab by masking its address down to
    /// the slab size, so every address below one slab masks to zero and
    /// reading a header there faults. That is where a small integer
    /// mistaken for a pointer lands, which is how this was found: a
    /// parameter with no type reached `free` holding something that was
    /// never an address, and the report named a null dereference inside
    /// the allocator rather than the value that was wrong.
    ///
    /// The bound turns no real block away. A slab is taken aligned to
    /// its own size, so it starts at or above `SLAB` and every block
    /// sits inside one.
    #[test]
    fn an_address_below_the_first_slab_is_refused() {
        for addr in [1usize, 8, 4096, SLAB - 1] {
            assert_eq!(
                slab_of(addr as *mut u8) as usize,
                0,
                "address {addr} masks to zero, which is why it must not be read"
            );
            assert!(
                !could_be_ours(addr as *mut u8),
                "address {addr} must be refused before anything reads through it"
            );
        }
        assert_eq!(
            slab_of(SLAB as *mut u8) as usize,
            SLAB,
            "an address at the boundary masks to itself"
        );
        assert!(could_be_ours(SLAB as *mut u8));
    }

    /// And every block a pool actually hands out clears the bound, so
    /// the guard above cannot be refusing real work.
    #[test]
    fn every_block_the_pool_hands_out_clears_the_bound() {
        unsafe {
            let mut taken = Vec::new();
            for size in [1usize, 16, 24, 512, MAX_POOLED, MAX_POOLED + 1, 1 << 20] {
                let p = zyntax_alloc(size);
                assert!(!p.is_null(), "allocation of {size} failed");
                assert!(
                    could_be_ours(p),
                    "a {size}-byte block came back at {p:p}, below the bound \
                     that decides whether to read its header"
                );
                taken.push(p);
            }
            for p in taken {
                zyntax_free(p);
            }
        }
    }

    /// A large block is recognised without reading in front of its
    /// slab-aligned address.
    ///
    /// A block past the cap is taken from the system allocator wherever
    /// it likes, so masking its address down to the slab size points at
    /// as much as a whole slab in front of it, which this process may
    /// never have asked for. Reading there to decide what the block is
    /// would be a fault on a legitimate pointer rather than on a bad
    /// one. The header in front of the payload is always the program's
    /// own memory, so it is what decides first.
    ///
    /// Many at once and each written before release, because the fault
    /// this guards needs a block whose masked address falls outside the
    /// region it was taken from, and which allocation that is depends
    /// on where the allocator happens to be working.
    #[test]
    fn a_large_block_is_released_without_reading_below_it() {
        unsafe {
            let mut taken = Vec::new();
            for i in 0..256usize {
                // Sizes that straddle the cap so both paths are taken,
                // and none of them a multiple of the slab, so the
                // addresses do not line up with slab boundaries.
                let size = MAX_POOLED + 1 + (i * 97) % 4096;
                let p = zyntax_alloc(size);
                assert!(!p.is_null(), "allocation {i} of {size} failed");
                std::ptr::write_bytes(p, i as u8, size);
                taken.push((p, size, i as u8));
            }
            for (p, size, tag) in &taken {
                assert_eq!(*p.add(size - 1), *tag, "block {tag} was overwritten");
            }
            for (p, _, _) in taken {
                zyntax_free(p);
            }
        }
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
    /// Counted rather than weighed. This read resident size before,
    /// which is a property of the process and not of the pool: the
    /// three hundred tests running beside it allocate into the same
    /// number, so the reading moved with whatever else was scheduled
    /// and the test failed for reasons that had nothing to do with it.
    /// A block taken and not given back is exactly what
    /// [`large_blocks_live`] counts.
    #[cfg(debug_assertions)]
    #[test]
    fn a_large_block_freed_on_another_thread_is_released() {
        const BLOCK: usize = 64 * 1024;
        const PER_ROUND: usize = 512;
        assert!(
            class_of(BLOCK).is_none(),
            "the block has to be one libc owns, or this tests the pools instead"
        );

        let before = large_blocks_live();
        for _ in 0..4 {
            let taken: Vec<usize> = (0..PER_ROUND)
                .map(|_| unsafe { zyntax_alloc(BLOCK) } as usize)
                .collect();
            // Freed somewhere else, which is the whole question.
            std::thread::spawn(move || {
                for p in taken {
                    unsafe { zyntax_free(p as *mut u8) };
                }
            })
            .join()
            .expect("freeing thread");
        }
        let after = large_blocks_live();

        // Every block this test took is accounted for. Another test
        // holding its own large block while this reads the counter
        // shifts both readings alike, so the difference is still this
        // test's, and a leak of even one round is 512.
        assert!(
            after - before < PER_ROUND as isize,
            "{} large blocks were taken and not given back, over four \
             rounds of {PER_ROUND}: freeing one off-thread is not \
             releasing it",
            after - before
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
