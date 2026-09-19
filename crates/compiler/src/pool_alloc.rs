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
//! not, so this keeps a free list per size class, carves slabs onto
//! the lists a slab at a time, and never gives a slab back to the
//! system. Freeing is pushing onto a list and allocating is popping
//! off one, which is what makes it cheaper than the thing it replaces
//! rather than merely different. The collector in [`crate::collector`]
//! sweeps what it finds unreached onto a second list per class, and a
//! slab it finds wholly unreached is set aside to be carved again under
//! whatever class next needs one.
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
//! Whether a pointer lies in a slab is answered by [`SLAB_INDEX`], and
//! whether it is a large block by the collector's registry, so a
//! pointer from another allocator is handed back to it without reading
//! through it: the memory in front of it, or where its slab header
//! would be, may not be mapped. [`could_be_ours`] turns away the one
//! address no allocator returns, a small integer mistaken for a
//! pointer.
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

use std::alloc::{Layout, alloc as sys_alloc, dealloc as sys_dealloc};
use std::cell::Cell;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicIsize, AtomicUsize};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// Largest request served from a pool. Above this, libc. Two kilobytes
/// keeps a hash table of 64 slots and a list buffer of 256 words in
/// the pool, where a release is a push and the collector can see them.
const MAX_POOLED: usize = 2048;

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

/// A slab header's second word: the size class in the low bits and the
/// bytes carved so far, header included, above them. Twelve bits of
/// class leaves twenty for the extent on a 32-bit target, enough for a
/// slab sixteen times this size.
const CLASS_BITS: u32 = 12;
const CLASS_MASK: usize = (1 << CLASS_BITS) - 1;
const USED_SHIFT: u32 = CLASS_BITS;

/// The class of a slab the collector found empty and set aside. It is
/// carved again under whatever class next needs a slab; until then no
/// block in it is anyone's, and a cursor still pointing at it sees a
/// class that is not its own and takes another slab.
const RETIRED: usize = CLASS_MASK;

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
    let head = &*(slab as *const Header);
    let (class, used) = head.slab_class_and_used();
    // A header only the pool writes; anything else here was written
    // by a program reaching below its first block. Such a slab is
    // reported and left alone, as if set aside.
    if head.magic != MAGIC || used > SLAB || (class >= CLASSES && class != RETIRED) {
        eprintln!(
            "[pool] slab {slab:#x} has header magic {:#x}, class {class}, used {used}, \
             which the pool did not write; a block's owner wrote below it",
            head.magic
        );
        return (RETIRED, HEADER);
    }
    (class, used)
}

/// Bytes at the front of a slab before its first block.
pub(crate) const SLAB_HEADER: usize = HEADER;
/// Bytes per slab and its alignment.
pub(crate) const SLAB_BYTES: usize = SLAB;
/// Bytes a block in `class` occupies; see [`slot_bytes`].
pub(crate) fn class_slot_bytes(class: usize) -> usize {
    slot_bytes(class)
}

/// Every block on this thread's free and swept lists, by address.
pub(crate) fn for_each_free_block(mut f: impl FnMut(usize)) {
    let mut walk = |lists: &[Cell<*mut u8>; CLASSES]| {
        for (class, list) in lists.iter().enumerate() {
            let mut prev: *mut u8 = std::ptr::null_mut();
            let mut p = list.get();
            while !p.is_null() {
                // A link that leaves the slabs was written through a
                // block after its release. The walk stops there: the
                // sweep rebuilds every list, so what follows the bad
                // link is recovered as unreached rather than followed.
                if !in_a_slab(p) || (p as usize) & (STEP - 1) != 0 {
                    eprintln!(
                        "[pool] free list of class {class} holds {p:p} after {prev:p}, \
                         which is not a block of this pool; a released block was written to"
                    );
                    break;
                }
                f(p as usize);
                prev = p;
                // SAFETY: a block on a free list holds the next block
                // in its first word.
                p = unsafe { *(p as *mut *mut u8) };
            }
        }
    };
    POOL.with(|p| {
        walk(&p.free);
        walk(&p.swept);
    });
}

/// Empty this thread's free and swept lists; the collector fills the
/// swept lists again from what its sweep finds unreached.
pub(crate) fn clear_free_lists() {
    POOL.with(|p| {
        for list in p.free.iter().chain(p.swept.iter()) {
            list.set(std::ptr::null_mut());
        }
    });
}

/// Put a chain of reclaimed blocks of `class`, threaded through their
/// first words from `head` to `tail`, at the front of this thread's
/// swept list.
///
/// # Safety
/// Every block on the chain must be one of `class` from this pool
/// that nothing reads any more.
pub(crate) unsafe fn push_free_chain(class: usize, head: usize, tail: usize) {
    POOL.with(|p| {
        *(tail as *mut *mut u8) = p.swept[class].get();
        p.swept[class].set(head as *mut u8);
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

/// A thread's share of the pool. One thread-local rather than one per
/// field: on some targets every thread-local looked up is a call, and
/// an allocation or release needs all of this at once.
///
/// `Cell` rather than `RefCell`: this is the hot path, and a borrow
/// flag checked twice per allocation is a real share of what the pool
/// exists to save.
struct Lists {
    /// Head of each class's free list. A block on a list stores the
    /// next pointer in its payload, which is why a class must be at
    /// least a pointer wide.
    free: [Cell<*mut u8>; CLASSES],
    /// Blocks the collector's sweep found unreached, by class. Served
    /// after the free list and counted like fresh storage, since a
    /// program that lives on these is one whose garbage the collector
    /// has to keep finding.
    swept: [Cell<*mut u8>; CLASSES],
    /// The slab being carved for each class. One per class: a slab
    /// serves a single class, so that masking a block's address back to
    /// it says which. How much of it is spoken for is in its header.
    slab: [Cell<*mut u8>; CLASSES],
    /// The collector's mark of whether this thread is the one it
    /// belongs to, kept here so a release checks it without a second
    /// lookup; the collector owns its meaning.
    mutator: Cell<usize>,
}

const NO_LISTS: [Cell<*mut u8>; CLASSES] = [const { Cell::new(std::ptr::null_mut()) }; CLASSES];

thread_local! {
    static POOL: Lists = const {
        Lists {
            free: NO_LISTS,
            swept: NO_LISTS,
            slab: NO_LISTS,
            mutator: Cell::new(usize::MAX),
        }
    };
}

/// Take the head of a list, or null.
///
/// # Safety
/// A block on the list holds the next block in its first word.
#[inline]
unsafe fn pop(list: &Cell<*mut u8>) -> *mut u8 {
    let head = list.get();
    if !head.is_null() {
        list.set(*(head as *mut *mut u8));
    }
    head
}

/// The collector's mark for the calling thread; see [`Lists::mutator`].
pub(crate) fn with_mutator_mark<R>(f: impl FnOnce(&Cell<usize>) -> R) -> R {
    POOL.with(|p| f(&p.mutator))
}

/// Slabs the collector found empty, waiting to be carved again.
static EMPTY_SLABS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Set an empty slab aside for reuse under any class.
///
/// # Safety
/// No block in the slab may be reached or on any list any more.
pub(crate) unsafe fn retire_slab(slab: usize) {
    (*(slab as *mut Header)).class = RETIRED | (SLAB << USED_SHIFT);
    EMPTY_SLABS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(slab);
}

/// Whether a slab has been set aside.
pub(crate) fn is_retired_class(class: usize) -> bool {
    class == RETIRED
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zyntax_alloc(size: usize) -> *mut u8 {
    let Some(class) = class_of(size) else {
        return large_alloc(size);
    };
    #[cfg(debug_assertions)]
    SERVED.fetch_add(1, Ordering::Relaxed);

    // A block already on this pool's list, then one the collector
    // reclaimed, which counts as fresh.
    let reused = POOL.with(|p| {
        let head = pop(&p.free[class]);
        #[cfg(not(target_arch = "wasm32"))]
        if head.is_null() {
            let swept = pop(&p.swept[class]);
            if !swept.is_null() {
                crate::collector::note_spent(slot_bytes(class));
            }
            return swept;
        }
        head
    });
    if !reused.is_null() {
        return reused;
    }
    alloc_slow(class, size)
}

/// Nothing to reuse means the heap grows. The collector gets its say
/// first: what it finds unreached serves the request after all.
/// Otherwise carve: the rest of a slab goes on the free list in one
/// go, so the bookkeeping is paid once per slab and the blocks come
/// off the list like any other.
#[inline(never)]
unsafe fn alloc_slow(class: usize, size: usize) -> *mut u8 {
    #[cfg(not(target_arch = "wasm32"))]
    if crate::collector::wants_collection() {
        crate::collector::collect();
        let swept = POOL.with(|p| pop(&p.swept[class]));
        if !swept.is_null() {
            crate::collector::note_spent(slot_bytes(class));
            return swept;
        }
    }
    if !refill(class) {
        // Out of memory for a slab; the request itself may still fit.
        return large_alloc(size);
    }
    POOL.with(|p| pop(&p.free[class]))
}

/// Put the rest of `class`'s slab on the free list, taking another
/// slab if the current one cannot fit a block or has been set aside.
/// A slab is aligned to its own size so that masking any block in it
/// lands on its header, which is where the carved extent lives.
///
/// # Safety
/// Called with the pool's lists consistent; the list gains blocks
/// nothing else names.
unsafe fn refill(class: usize) -> bool {
    let want = slot_bytes(class);
    let (slab, used) = POOL.with(|p| {
        let sp = &p.slab;
        let mut slab = sp[class].get();
        let (slab_class, mut used) = if slab.is_null() {
            (RETIRED, SLAB)
        } else {
            (*(slab as *const Header)).slab_class_and_used()
        };
        if slab_class != class || used + want > SLAB {
            // An empty slab set aside by the collector, or a fresh one.
            let recycled = EMPTY_SLABS.lock().unwrap_or_else(|e| e.into_inner()).pop();
            slab = match recycled {
                Some(s) => s as *mut u8,
                None => {
                    let fresh = sys_alloc(Layout::from_size_align_unchecked(SLAB, SLAB));
                    if fresh.is_null() {
                        return (std::ptr::null_mut(), 0);
                    }
                    (*(fresh as *mut Header)).magic = MAGIC;
                    index_slab(fresh as usize);
                    #[cfg(not(target_arch = "wasm32"))]
                    crate::collector::note_slab(fresh as usize);
                    fresh
                }
            };
            sp[class].set(slab);
            used = HEADER;
        }
        (slab, used)
    });
    if slab.is_null() {
        return false;
    }
    // Every block that fits, threaded lowest first so the lowest
    // address is handed out first, and the header marked as carved to
    // the end.
    let count = (SLAB - used) / want;
    let first = slab.add(used);
    let last = first.add((count - 1) * want);
    let mut p = last;
    POOL.with(|lists| {
        let lists = &lists.free;
        let mut next = lists[class].get();
        loop {
            *(p as *mut *mut u8) = next;
            next = p;
            if p == first {
                break;
            }
            p = p.sub(want);
        }
        lists[class].set(first);
    });
    (*(slab as *mut Header)).class = class | ((used + count * want) << USED_SHIFT);
    #[cfg(not(target_arch = "wasm32"))]
    crate::collector::note_carved(count * want);
    true
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

/// Every slab this pool has taken, as a two-level bitmap over the
/// address space: one bit per slab-sized span, in leaves of a
/// [`INDEX_LEAF_SPAN`] each, made on demand. Asking whether an address
/// lies in a slab is then two loads and no read through the address,
/// which is what lets a pointer from another allocator be handed back
/// to it without first reading where its slab header would be, memory
/// this process may never have mapped.
const INDEX_LEAF_BITS: usize = if usize::BITS > 32 { 16 } else { 15 };
const INDEX_LEAF_SPAN: usize = SLAB << INDEX_LEAF_BITS;
const INDEX_LEAF_WORDS: usize = (1 << INDEX_LEAF_BITS) / 64;
/// Leaves for a 48-bit address space, or the whole of a 32-bit one; an
/// address above that is not this pool's.
const INDEX_LEAVES: usize = if usize::BITS > 32 {
    1 << (48 - INDEX_LEAF_BITS - 16)
} else {
    2
};

type IndexLeaf = [AtomicU64; INDEX_LEAF_WORDS];

static SLAB_INDEX: [AtomicPtr<IndexLeaf>; INDEX_LEAVES] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; INDEX_LEAVES];

/// Record a slab in the index.
///
/// # Safety
/// `slab` must be the base of a slab this pool just took.
unsafe fn index_slab(slab: usize) {
    let leaf_no = slab / INDEX_LEAF_SPAN;
    let Some(slot) = SLAB_INDEX.get(leaf_no) else {
        return;
    };
    let mut leaf = slot.load(Ordering::Acquire);
    if leaf.is_null() {
        let fresh: Box<IndexLeaf> = Box::new([const { AtomicU64::new(0) }; INDEX_LEAF_WORDS]);
        let fresh = Box::into_raw(fresh);
        match slot.compare_exchange(
            std::ptr::null_mut(),
            fresh,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => leaf = fresh,
            Err(other) => {
                drop(Box::from_raw(fresh));
                leaf = other;
            }
        }
    }
    let bit = (slab % INDEX_LEAF_SPAN) / SLAB;
    (*leaf)[bit / 64].fetch_or(1 << (bit % 64), Ordering::Release);
}

/// Whether `ptr` lies in a slab this pool took.
#[inline]
fn in_a_slab(ptr: *mut u8) -> bool {
    in_a_slab_at(ptr as usize)
}

/// [`in_a_slab`] for an address.
#[inline]
pub(crate) fn in_a_slab_at(a: usize) -> bool {
    let Some(slot) = SLAB_INDEX.get(a / INDEX_LEAF_SPAN) else {
        return false;
    };
    let leaf = slot.load(Ordering::Acquire);
    if leaf.is_null() {
        return false;
    }
    let bit = (a % INDEX_LEAF_SPAN) / SLAB;
    // SAFETY: a leaf, once published, lives for the process.
    unsafe { (*leaf)[bit / 64].load(Ordering::Relaxed) & (1 << (bit % 64)) != 0 }
}

/// Whether an address could have come from here at all.
///
/// A slab is taken from the system allocator aligned to its own size,
/// so it starts at or above [`SLAB`], and every block sits inside one.
/// Nothing this pool hands out is below that, and the bound therefore
/// turns no real block away.
///
/// It matters because a large block keeps its header in front of its
/// payload, and reading there for an address in the first slab's worth
/// of address space is a fault rather than a wrong guess. That is
/// exactly where a small integer mistaken for a pointer lands.
#[inline]
fn could_be_ours(ptr: *mut u8) -> bool {
    (ptr as usize) >= SLAB
}

/// Release a pointer from [`zyntax_alloc`].
///
/// # Safety
/// `ptr` must have come from [`zyntax_alloc`] and must not be used
/// afterwards. A null pointer is ignored, as `free`'s is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zyntax_free(ptr: *mut u8) {
    if ptr.is_null() {
        return;
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
    // A pooled block is found by the index, a large one by the
    // collector's registry, and neither read through the pointer, since
    // for a block from another allocator the memory in front of it or
    // where its slab header would be may not be mapped at all.
    if !in_a_slab(ptr) {
        // A box every program shares lives outside the slabs and is
        // never released, so only this branch has to ask.
        #[cfg(not(target_arch = "wasm32"))]
        if crate::interned::is_interned(ptr as usize) {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::collector::note_thread();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(total) = crate::collector::forget_large(ptr as usize) {
            sys_dealloc(
                ptr.sub(HEADER),
                Layout::from_size_align_unchecked(total, HEADER),
            );
            #[cfg(debug_assertions)]
            LARGE_LIVE.fetch_sub(1, Ordering::Relaxed);
            return;
        }
        #[cfg(target_arch = "wasm32")]
        {
            let head = ptr.sub(HEADER) as *const Header;
            if (*head).magic == MAGIC && (*head).class & LARGE_MARK != 0 {
                let total = (*head).class & !LARGE_MARK;
                sys_dealloc(
                    ptr.sub(HEADER),
                    Layout::from_size_align_unchecked(total, HEADER),
                );
                return;
            }
        }
        // Hand it to the allocator that most likely owns it rather than
        // threading a foreign block onto a free list.
        libc_free(ptr);
        return;
    }
    let slab = slab_of(ptr);
    let class = (*slab).class & CLASS_MASK;
    if class >= CLASSES {
        // A block of a slab the collector set aside: nothing held it,
        // so this release is of storage already reclaimed.
        debug_assert!(
            false,
            "release of {ptr:p}, a block the collector already reclaimed"
        );
        return;
    }
    let block = ptr;
    // A freed block keeps its bytes, so a read through a stale pointer
    // returns the old contents: plausible, wrong, and silent. Overwrite
    // the payload with a byte that is none of a small integer, a valid
    // pointer, or ASCII, so such a read is recognisable instead. Debug
    // only — the whole point of the pool is that freeing is a push.
    #[cfg(debug_assertions)]
    std::ptr::write_bytes(block, POISON, slot_bytes(class));

    POOL.with(|p| {
        // A thread that frees will allocate from its own list without
        // ever carving, which is the other way a second mutator
        // appears; the mark is read here, where the lists already are.
        #[cfg(not(target_arch = "wasm32"))]
        crate::collector::note_thread_mark(&p.mutator);
        // Threaded through the block's own first word. A freed block
        // holds nothing a live one needed, and the poison above is
        // overwritten here for that word alone, which is why the test
        // for it reads past the first pointer.
        *(block as *mut *mut u8) = p.free[class].get();
        p.free[class].set(block);
    });
}

/// The usable size of a block from [`zyntax_alloc`], or `None` for a
/// pointer this pool did not hand out.
///
/// # Safety
/// `ptr` must have come from [`zyntax_alloc`].
unsafe fn usable_size(ptr: *mut u8) -> Option<usize> {
    if in_a_slab(ptr) {
        let slab = slab_of(ptr);
        return Some(slot_bytes((*slab).class & CLASS_MASK));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::collector::large_total(ptr as usize).map(|total| total - HEADER)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let head = ptr.sub(HEADER) as *const Header;
        if (*head).magic == MAGIC && (*head).class & LARGE_MARK != 0 {
            Some(((*head).class & !LARGE_MARK) - HEADER)
        } else {
            None
        }
    }
}

/// Resize a block from [`zyntax_alloc`] to `new_size` bytes, keeping
/// its contents up to the smaller of the two sizes. A block that already
/// has room is returned as it is; a null pointer allocates.
///
/// # Safety
/// `ptr` must be null or have come from [`zyntax_alloc`], and must not
/// be used afterwards except through the returned pointer.
#[unsafe(no_mangle)]
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
    unsafe extern "C" {
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
    unsafe extern "C" {
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

    /// Every block the pool hands out is found in the slab index, its
    /// slab is aligned to the slab size, and a large block or a foreign
    /// pointer is not in the index.
    #[test]
    fn the_index_knows_every_pooled_block_and_nothing_else() {
        unsafe {
            let mut blocks = Vec::new();
            for size in [1usize, 16, 24, 32, 100, 512, 1024] {
                for _ in 0..3000 {
                    let p = zyntax_alloc(size);
                    assert!(in_a_slab(p), "{p:p} (size {size}) is not in the index");
                    assert!(in_a_slab(p.add(size - 1)));
                    assert_eq!(
                        slab_of(p) as usize % SLAB,
                        0,
                        "the slab of {p:p} is not aligned"
                    );
                    blocks.push(p);
                }
            }
            let big = zyntax_alloc(MAX_POOLED * 4);
            assert!(!in_a_slab(big));
            let foreign = Box::into_raw(Box::new([0u8; 64])) as *mut u8;
            assert!(!in_a_slab(foreign));
            drop(Box::from_raw(foreign as *mut [u8; 64]));
            zyntax_free(big);
            for p in blocks {
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
