//! A conservative mark-sweep collector over the pool's storage.
//!
//! The drop analysis releases an allocation where it can prove nothing
//! else holds it. A value that is stored into a container, captured, or
//! handed to a callee that may keep it falls outside that proof, and
//! without this it stays allocated for the life of the process. This
//! reclaims those: from time to time, everything the pool has handed
//! out that no root reaches goes back on the free lists.
//!
//! ## What it assumes
//!
//! Roots are found conservatively: every aligned word on the native
//! stacks, in the registered global ranges, and in what the fiber
//! runtime holds is taken for a pointer when it lands inside a pool
//! block, and every word of a reached block is followed the same way.
//! An integer that happens to look like an address keeps a block alive
//! a while longer; nothing is ever freed while a word anywhere still
//! names it. A pointer just past a block's end names the block after
//! it, not the block: compiled code keeps a buffer's base live for as
//! long as it indexes into it, so the end alone never has to.
//!
//! Compiled code keeps every live pointer in a stack slot or a
//! callee-saved register at a call, and a collection only ever starts
//! inside an allocation call, so spilling the callee-saved registers
//! and reading the stack sees them all. Values the interpreter holds in
//! its own registers are not on any stack, so a program that runs
//! interpreted must not turn this on.
//!
//! One mutator thread. Slabs are shared between threads, but the free
//! lists the sweep rebuilds are the collecting thread's, and another
//! thread's stack is never read. The thread that enables the collector
//! is the one it runs on; the first allocation or release from any
//! other thread turns it off. Turning it off also forgets the roots,
//! since they belong to the runtime that registered them; a runtime
//! that enables it again registers its own.
//!
//! ## When it runs
//!
//! Each collection grants a budget of storage: as much as it found
//! live, or a floor below which no collection is worth its time. The
//! pool spends the budget on what it carves from slab space, takes
//! from libc, or hands back out from what the last sweep reclaimed;
//! a block the program itself released and takes again costs nothing.
//! So a program whose drop analysis releases everything it makes
//! never collects at all, and one that releases nothing collects once
//! per live set's worth of allocation, with the heap held near twice
//! the live set. A slab the sweep finds wholly unreached is set aside
//! whole and carved again under any class, so the sweep costs by the
//! live slabs rather than by every dead block, and storage freed in
//! one size class serves another.
//!
//! `ZYNTAX_DISABLE_GC=1` keeps it off however it was enabled; safe, the
//! program merely leaks what it would have collected. `ZYNTAX_TRACE_GC=1`
//! reports each collection. `ZYNTAX_GC_FLOOR_KB=<n>` sets the heap size
//! below which nothing is collected; a small value collects constantly,
//! which is how a missed root is found.
//!
//! The collector is process state, so it is tested from a process of
//! its own: a program that runs under it with the floor lowered, as
//! the Python frontend's pressure test does.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::pool_alloc;

/// Which collector a runtime runs behind the drop analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Collector {
    /// None: what the analysis cannot prove released stays allocated.
    #[default]
    None,
    /// This module's conservative mark-sweep.
    MarkSweep,
}

/// The least budget a collection grants, and what the first one waits
/// for: small enough to keep a short program's memory small, large
/// enough that a collection sweeps a worthwhile amount.
const MIN_HEAP: usize = 16 << 20;

fn heap_floor() -> usize {
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| {
        std::env::var("ZYNTAX_GC_FLOOR_KB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|kb| kb << 10)
            .unwrap_or(MIN_HEAP)
    })
}

/// Bits in a slab's mark and free maps: one per sixteen bytes.
const SLAB_BITS: usize = pool_alloc::SLAB_BYTES / 16;
const SLAB_WORDS: usize = SLAB_BITS / 64;

/// Everything the collector knows across collections.
struct Registry {
    /// Base address of every slab the pool has carved.
    slabs: BTreeSet<usize>,
    /// Large blocks by payload address, with their total length.
    large: BTreeMap<usize, usize>,
    /// Memory outside the heap that may hold pointers into it, by
    /// start address and length.
    roots: BTreeMap<usize, usize>,
    /// Bytes reached by the last collection.
    live: usize,
    collections: usize,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Bumped whenever the owner changes, so a thread's cached answer to
/// "am I the owner" is only trusted for the owner it was made for.
static OWNER_GENERATION: AtomicUsize = AtomicUsize::new(0);
static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    slabs: BTreeSet::new(),
    large: BTreeMap::new(),
    roots: BTreeMap::new(),
    live: 0,
    collections: 0,
});
/// The thread the collector was enabled on, which is the only one it
/// runs on.
static OWNER: Mutex<Option<std::thread::ThreadId>> = Mutex::new(None);

/// What the collector keeps per thread, in one place so a request
/// pays one thread-local access, not one per fact.
#[derive(Clone, Copy)]
struct Local {
    /// Bytes the heap holds: slab space taken and large blocks out.
    heap: usize,
    /// Bytes spent against the budget since the last collection.
    spent: usize,
    /// Bytes the last collection allowed before the next.
    budget: usize,
    collecting: bool,
}

thread_local! {
    static LOCAL: std::cell::Cell<Local> = const {
        std::cell::Cell::new(Local {
            heap: 0,
            spent: 0,
            budget: MIN_HEAP,
            collecting: false,
        })
    };
}

#[inline]
fn local() -> Local {
    LOCAL.with(|l| l.get())
}

#[inline]
fn update(f: impl FnOnce(&mut Local)) {
    LOCAL.with(|l| {
        let mut v = l.get();
        f(&mut v);
        l.set(v);
    });
}

fn registry() -> MutexGuard<'static, Registry> {
    REGISTRY.lock().unwrap_or_else(|e| e.into_inner())
}

fn trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZYNTAX_TRACE_GC").is_some())
}

fn trace_detail() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ZYNTAX_TRACE_GC").as_deref() == Ok("2"))
}

fn disabled_by_env() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("ZYNTAX_DISABLE_GC").is_some())
}

/// Turn the collector on, on this thread. Roots registered before are
/// forgotten: they belonged to whatever enabled it last.
pub fn enable() {
    if disabled_by_env() {
        return;
    }
    let mut owner = OWNER.lock().unwrap_or_else(|e| e.into_inner());
    if *owner != Some(std::thread::current().id()) {
        *owner = Some(std::thread::current().id());
        OWNER_GENERATION.fetch_add(1, Ordering::SeqCst);
    }
    drop(owner);
    registry().roots.clear();
    update(|l| {
        l.spent = 0;
        l.budget = heap_floor();
    });
    ENABLED.store(true, Ordering::SeqCst);
}

/// Turn the collector off and forget its roots; what is allocated
/// stays allocated.
pub fn disable() {
    ENABLED.store(false, Ordering::SeqCst);
    registry().roots.clear();
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Whether the calling thread is the one the collector belongs to.
fn on_owner_thread() -> bool {
    pool_alloc::with_mutator_mark(owner_by_mark)
}

/// Whether the calling thread is the one the collector belongs to,
/// read from the mark the pool keeps for the thread: the owner
/// generation the thread was last confirmed for, or a value that is
/// none until it is. A thread that is not the owner is never marked,
/// so it asks each time; it asks once, since the answer ends the
/// collector.
#[inline]
pub(crate) fn owner_by_mark(mark: &Cell<usize>) -> bool {
    let generation = OWNER_GENERATION.load(Ordering::Relaxed);
    mark.get() == generation || confirm_owner(mark, generation)
}

#[inline(never)]
fn confirm_owner(mark: &Cell<usize>, generation: usize) -> bool {
    let owner =
        *OWNER.lock().unwrap_or_else(|e| e.into_inner()) == Some(std::thread::current().id());
    if owner {
        mark.set(generation);
    }
    owner
}

/// A slab the pool has just taken from the system.
pub(crate) fn note_slab(slab: usize) {
    registry().slabs.insert(slab);
    note_grown(pool_alloc::SLAB_BYTES);
}

/// A large block the pool has just handed out. Recorded whether or
/// not the collector is on: the registry says what is allocated, and
/// an entry a later collection would find stale is a read of memory
/// that may no longer be mapped.
pub(crate) fn note_large(payload: usize, total: usize) {
    registry().large.insert(payload, total);
    if is_enabled() {
        note_carved(total);
        note_grown(total);
    }
}

/// A large block the program has released itself: its total length,
/// or none for an address that is not a large block of the pool's.
pub(crate) fn forget_large(payload: usize) -> Option<usize> {
    let total = registry().large.remove(&payload)?;
    if is_enabled() && on_owner_thread() {
        update(|l| l.heap = l.heap.saturating_sub(total));
    }
    Some(total)
}

/// The total length of the large block at `payload`, if it is one.
pub(crate) fn large_total(payload: usize) -> Option<usize> {
    registry().large.get(&payload).copied()
}

/// The calling thread uses the pool: whether the collector is on and
/// this is its thread. A second mutator thread has a stack the
/// collector would never read, so it ends the collector.
pub(crate) fn note_thread() -> bool {
    if !is_enabled() {
        return false;
    }
    if on_owner_thread() {
        return true;
    }
    second_thread();
    false
}

/// [`note_thread`] for the pool's release path, given the mark it
/// already has in hand.
#[inline]
pub(crate) fn note_thread_mark(mark: &Cell<usize>) {
    if is_enabled() && !owner_by_mark(mark) {
        second_thread();
    }
}

#[inline(never)]
fn second_thread() {
    disable();
    if trace() {
        eprintln!("[gc] off: the pool is used from a second thread");
    }
}

/// Fresh bytes the pool has handed out, spent against the budget.
pub(crate) fn note_carved(bytes: usize) {
    if !note_thread() {
        return;
    }
    update(|l| l.spent += bytes);
}

/// A reclaimed block handed out again, spent against the budget. Only
/// the collecting thread has any, so there is no thread to check.
#[inline]
pub(crate) fn note_spent(bytes: usize) {
    update(|l| l.spent += bytes);
}

/// Fresh bytes the heap has grown by.
pub(crate) fn note_grown(bytes: usize) {
    if !is_enabled() || !on_owner_thread() {
        return;
    }
    update(|l| l.heap += bytes);
}

/// Whether what was spent since the last collection has used up its
/// budget.
pub(crate) fn wants_collection() -> bool {
    if !is_enabled() {
        return false;
    }
    let l = local();
    l.spent >= l.budget && !l.collecting && on_owner_thread()
}

/// Memory outside the heap the collector must read for pointers: a
/// global the program writes, for one.
pub fn add_root_range(ptr: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    registry().roots.insert(ptr as usize, len);
}

/// What a collection found and did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub collections: usize,
    /// Bytes reached by the last collection.
    pub live: usize,
}

pub fn stats() -> Stats {
    let r = registry();
    Stats {
        collections: r.collections,
        live: r.live,
    }
}

/// One slab as a collection sees it: its layout, and a mark bit and a
/// free bit per sixteen bytes.
struct SlabBits {
    class: usize,
    slot: usize,
    /// Blocks carved so far.
    count: usize,
    marks: [u64; SLAB_WORDS],
    free: [u64; SLAB_WORDS],
}

impl SlabBits {
    /// # Safety
    /// `slab` must be the base of a live slab.
    unsafe fn read(slab: usize) -> Box<Self> {
        let (class, used) = pool_alloc::slab_layout(slab);
        // A slab set aside holds nothing until it is carved again.
        let (slot, count) = if pool_alloc::is_retired_class(class) {
            (pool_alloc::SLAB_BYTES, 0)
        } else {
            let slot = pool_alloc::class_slot_bytes(class);
            (slot, (used - pool_alloc::SLAB_HEADER) / slot)
        };
        Box::new(SlabBits {
            class,
            slot,
            count,
            marks: [0; SLAB_WORDS],
            free: [0; SLAB_WORDS],
        })
    }

    /// The block at `offset` from the slab base, as its index, or none
    /// past what is carved.
    #[inline]
    fn block_at(&self, offset: usize) -> Option<usize> {
        if offset < pool_alloc::SLAB_HEADER {
            return None;
        }
        let idx = (offset - pool_alloc::SLAB_HEADER) / self.slot;
        (idx < self.count).then_some(idx)
    }

    #[inline]
    fn base(&self, idx: usize) -> usize {
        pool_alloc::SLAB_HEADER + idx * self.slot
    }
}

/// Hashes a slab address: the bits above the slab alignment, spread by
/// one multiplication. A general hasher costs more than the lookup it
/// serves, on a key that is a multiple of the slab size.
#[derive(Default)]
struct SlabHasher(u64);

impl Hasher for SlabHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("slab keys are written as usize")
    }
    fn write_usize(&mut self, key: usize) {
        self.0 = ((key >> 16) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type SlabMap<V> = HashMap<usize, V, BuildHasherDefault<SlabHasher>>;

#[inline]
fn bit(offset: usize) -> (usize, u64) {
    let i = offset / 16;
    (i / 64, 1u64 << (i % 64))
}

/// One collection's working state.
struct Marker<'a> {
    reg: &'a Registry,
    /// The slabs touched so far, by base address: made on first touch,
    /// so a slab nothing reaches costs nothing to mark and is set
    /// aside whole by the sweep.
    bits: SlabMap<Box<SlabBits>>,
    /// The lowest and highest addresses any block may have, so most
    /// words are turned away without a lookup.
    lo: usize,
    hi: usize,
    large_marked: HashSet<usize>,
    /// Reached blocks whose words are still to be read: base and length.
    work: Vec<(usize, usize)>,
    marked_bytes: usize,
}

impl<'a> Marker<'a> {
    fn new(reg: &'a Registry) -> Self {
        let bits = SlabMap::with_capacity_and_hasher(reg.slabs.len(), Default::default());
        let mut lo = reg.slabs.iter().next().copied().unwrap_or(usize::MAX);
        let mut hi = reg
            .slabs
            .iter()
            .next_back()
            .map(|s| s + pool_alloc::SLAB_BYTES)
            .unwrap_or(0);
        if let Some((&first, _)) = reg.large.iter().next() {
            lo = lo.min(first);
        }
        if let Some((&last, &total)) = reg.large.iter().next_back() {
            hi = hi.max(last + total);
        }
        Marker {
            reg,
            bits,
            lo,
            hi,
            large_marked: HashSet::new(),
            work: Vec::new(),
            marked_bytes: 0,
        }
    }

    /// The bits of the slab at `slab`, made on first touch; none for
    /// an address outside every slab.
    #[inline]
    fn slab_bits(&mut self, slab: usize) -> Option<&mut SlabBits> {
        if !self.bits.contains_key(&slab) {
            if !pool_alloc::in_a_slab_at(slab) {
                return None;
            }
            // SAFETY: a registered slab is live for the life of the process.
            self.bits.insert(slab, unsafe { SlabBits::read(slab) });
        }
        self.bits.get_mut(&slab).map(|b| &mut **b)
    }

    /// Note a block on a free list, which is not storage to follow.
    fn note_free(&mut self, block: usize) {
        let slab = block & !(pool_alloc::SLAB_BYTES - 1);
        if let Some(bits) = self.slab_bits(slab) {
            let (w, m) = bit(block - slab);
            bits.free[w] |= m;
        }
    }

    /// Take `a` for a pointer and mark what it lands in.
    #[inline]
    fn consider(&mut self, a: usize) {
        if a < self.lo || a >= self.hi {
            return;
        }
        let slab = a & !(pool_alloc::SLAB_BYTES - 1);
        if let Some(bits) = self.slab_bits(slab) {
            let Some(idx) = bits.block_at(a - slab) else {
                return;
            };
            let base = bits.base(idx);
            let (w, m) = bit(base);
            if (bits.free[w] | bits.marks[w]) & m != 0 {
                return;
            }
            bits.marks[w] |= m;
            let len = bits.slot;
            self.marked_bytes += len;
            self.work.push((slab + base, len));
            return;
        }
        if let Some((&payload, &total)) = self.reg.large.range(..=a).next_back() {
            let len = total - pool_alloc::SLAB_HEADER;
            if a < payload + len && self.large_marked.insert(payload) {
                if !mapped(payload, payload + len) {
                    eprintln!(
                        "[gc] large block {payload:#x}..{:#x} is not mapped; skipped",
                        payload + len
                    );
                    return;
                }
                self.marked_bytes += total;
                self.work.push((payload, len));
            }
        }
    }

    /// [`Self::scan`] of memory outside the heap, checked to be mapped
    /// first: a range registered by something that has since gone is
    /// the one fault a collection can take, and a skipped range with a
    /// report beats a dead process.
    fn scan_outside(&mut self, what: &str, lo: usize, hi: usize) {
        if !mapped(lo, hi) {
            eprintln!("[gc] {what} {lo:#x}..{hi:#x} is not mapped; skipped");
            return;
        }
        self.scan(lo, hi);
    }

    /// Read every aligned word in `[lo, hi)` as a possible pointer.
    fn scan(&mut self, lo: usize, hi: usize) {
        let mut p = (lo + 7) & !7;
        while p + 8 <= hi {
            // SAFETY: the caller hands over memory it owns and that is
            // mapped for the whole range.
            let w = unsafe { std::ptr::read_volatile(p as *const usize) };
            self.consider(w);
            p += 8;
        }
    }

    /// Follow what has been reached until nothing new is.
    fn drain(&mut self) {
        while let Some((base, len)) = self.work.pop() {
            self.scan(base, base + len);
        }
    }

    /// Put every unreached block back on the free lists, the lists
    /// being rebuilt from nothing so a block is on one exactly once.
    /// Returns the blocks and bytes that were not free before.
    /// Returns the blocks and bytes that were not free before, and the
    /// bytes on the free lists afterwards.
    fn sweep(&self) -> (usize, usize, usize) {
        pool_alloc::clear_free_lists();
        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0usize;
        let mut free_bytes = 0usize;
        // Highest first, so the lowest addresses come off the lists
        // first. A slab's blocks are chained here and joined to the
        // list in one go.
        for &slab in self.reg.slabs.iter().rev() {
            let Some(bits) = self.bits.get(&slab) else {
                // Never touched: nothing in it is reached or on a list,
                // so whatever it holds is garbage.
                // SAFETY: a registered slab is live.
                let (class, used) = unsafe { pool_alloc::slab_layout(slab) };
                if pool_alloc::is_retired_class(class) {
                    continue;
                }
                let slot = pool_alloc::class_slot_bytes(class);
                let count = (used - pool_alloc::SLAB_HEADER) / slot;
                if count == 0 {
                    continue;
                }
                freed_blocks += count;
                freed_bytes += count * slot;
                // SAFETY: nothing reaches any block of the slab.
                unsafe { pool_alloc::retire_slab(slab) };
                continue;
            };
            if bits.count == 0 {
                continue;
            }
            // A slab with nothing reached is set aside whole rather
            // than threaded block by block, and can serve any class.
            if bits.marks.iter().all(|w| *w == 0) {
                for idx in 0..bits.count {
                    let (w, m) = bit(bits.base(idx));
                    if bits.free[w] & m == 0 {
                        freed_blocks += 1;
                        freed_bytes += bits.slot;
                    }
                }
                // SAFETY: nothing reaches any block of the slab, and the
                // lists that held its free blocks were emptied above.
                unsafe { pool_alloc::retire_slab(slab) };
                continue;
            }
            let mut head = 0usize;
            let mut tail = 0usize;
            for idx in (0..bits.count).rev() {
                let base = bits.base(idx);
                let (w, m) = bit(base);
                if bits.marks[w] & m != 0 {
                    continue;
                }
                let block = slab + base;
                free_bytes += bits.slot;
                // SAFETY: nothing reaches the block; the sweep owns it.
                unsafe {
                    if bits.free[w] & m == 0 {
                        freed_blocks += 1;
                        freed_bytes += bits.slot;
                        pool_alloc::poison(block, bits.class);
                    }
                    *(block as *mut usize) = head;
                }
                head = block;
                if tail == 0 {
                    tail = block;
                }
            }
            if head != 0 {
                // SAFETY: the chain holds only unreached blocks of this
                // slab's class.
                unsafe { pool_alloc::push_free_chain(bits.class, head, tail) };
            }
        }
        (freed_blocks, freed_bytes, free_bytes)
    }
}

/// Whether every page of `[lo, hi)` is mapped. `msync` refuses a range
/// with a hole in it.
#[cfg(unix)]
fn mapped(lo: usize, hi: usize) -> bool {
    let page = 16384usize;
    let start = lo & !(page - 1);
    let end = (hi + page - 1) & !(page - 1);
    if end <= start {
        return true;
    }
    // SAFETY: asks the kernel about the range; nothing is read.
    unsafe { libc::msync(start as *mut libc::c_void, end - start, libc::MS_ASYNC) == 0 }
}

#[cfg(not(unix))]
fn mapped(_lo: usize, _hi: usize) -> bool {
    true
}

/// The highest address of the calling thread's stack.
fn host_stack_top() -> usize {
    thread_local! {
        static TOP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    TOP.with(|t| {
        if t.get() == 0 {
            t.set(query_stack_top());
        }
        t.get()
    })
}

#[cfg(target_os = "macos")]
fn query_stack_top() -> usize {
    // SAFETY: querying the calling thread.
    unsafe { libc::pthread_get_stackaddr_np(libc::pthread_self()) as usize }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn query_stack_top() -> usize {
    // SAFETY: the attribute is initialised by getattr and destroyed
    // after the read; every pointer handed in is to a live local.
    unsafe {
        let mut attr: libc::pthread_attr_t = std::mem::zeroed();
        if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
            return 0;
        }
        let mut addr: *mut libc::c_void = std::ptr::null_mut();
        let mut size: libc::size_t = 0;
        let rc = libc::pthread_attr_getstack(&attr, &mut addr, &mut size);
        libc::pthread_attr_destroy(&mut attr);
        if rc != 0 {
            return 0;
        }
        addr as usize + size
    }
}

#[cfg(windows)]
fn query_stack_top() -> usize {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
    }
    let (mut low, mut high) = (0usize, 0usize);
    // SAFETY: both pointers are to live locals.
    unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
    high
}

#[cfg(not(any(unix, windows)))]
fn query_stack_top() -> usize {
    0
}

/// Run a collection now, on the calling thread.
///
/// The callee-saved registers are spilled into this frame first, so a
/// pointer that compiled code kept in one is read off the stack with
/// the rest.
#[inline(never)]
pub fn collect() {
    if !is_enabled() || !on_owner_thread() || local().collecting {
        return;
    }
    let mut regs = [0usize; 12];
    #[cfg(target_arch = "aarch64")]
    // SAFETY: twelve words of the local array are written.
    unsafe {
        std::arch::asm!(
            "stp x19, x20, [{p}]",
            "stp x21, x22, [{p}, #16]",
            "stp x23, x24, [{p}, #32]",
            "stp x25, x26, [{p}, #48]",
            "stp x27, x28, [{p}, #64]",
            "stp x29, x30, [{p}, #80]",
            p = in(reg) regs.as_mut_ptr(),
            options(nostack, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: six words of the local array are written.
    unsafe {
        std::arch::asm!(
            "mov [{p}], rbx",
            "mov [{p} + 8], rbp",
            "mov [{p} + 16], r12",
            "mov [{p} + 24], r13",
            "mov [{p} + 32], r14",
            "mov [{p} + 40], r15",
            p = in(reg) regs.as_mut_ptr(),
            options(nostack, preserves_flags)
        );
    }
    let sp = regs.as_ptr() as usize;
    collect_from(sp);
    std::hint::black_box(&regs);
}

/// A collection whose deepest live stack word is at `sp`.
fn collect_from(sp: usize) {
    let host_top = host_stack_top();
    if host_top <= sp {
        // No way to bound the stack; leave the heap as it is.
        if trace() {
            eprintln!("[gc] skipped: the stack's extent is unknown");
        }
        update(|l| l.budget = usize::MAX);
        return;
    }
    update(|l| l.collecting = true);
    let started = web_time::Instant::now();
    let mut reg = registry();
    let carved = local().spent;
    update(|l| l.spent = 0);

    let (live, freed_blocks, freed_bytes, free_bytes, large_freed) = {
        let mut marker = Marker::new(&reg);
        let built_at = started.elapsed();
        pool_alloc::for_each_free_block(|b| marker.note_free(b));
        let free_walked_at = started.elapsed();

        // Roots: the stacks, the globals, what the fiber runtime holds.
        let mut windows: Vec<(usize, usize)> = Vec::new();
        match crate::fiber_backend::installed_fiber_backend() {
            Some(backend) => {
                backend.stack_windows(sp, host_top, &mut |lo, hi| windows.push((lo, hi)));
                backend.held_addresses(&mut |a| marker.consider(a));
            }
            None => windows.push((sp, host_top)),
        }
        if trace_detail() {
            eprintln!(
                "[gc]   {} slabs, {} large blocks, {} root ranges",
                marker.reg.slabs.len(),
                marker.reg.large.len(),
                marker.reg.roots.len()
            );
        }
        for (lo, hi) in windows {
            if lo < hi {
                if trace_detail() {
                    eprintln!("[gc]   stack {lo:#x}..{hi:#x} ({} KB)", (hi - lo) >> 10);
                }
                let before = marker.marked_bytes;
                marker.scan_outside("stack", lo, hi);
                if trace_detail() {
                    eprintln!(
                        "[gc]     {} KB reached directly",
                        (marker.marked_bytes - before) >> 10
                    );
                }
            }
        }
        let roots: Vec<(usize, usize)> = marker.reg.roots.iter().map(|(a, l)| (*a, *l)).collect();
        for (a, l) in roots {
            if trace_detail() {
                eprintln!("[gc]   global {a:#x} ({l} bytes)");
            }
            marker.scan_outside("global", a, a + l);
        }
        let direct = marker.marked_bytes;
        let roots_at = started.elapsed();
        marker.drain();
        let marked_at = started.elapsed();
        let (freed_blocks, freed_bytes, free_bytes) = marker.sweep();
        if trace() {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "[gc]   {} KB reached directly, {} KB through what they hold; tables {:.2} ms, free lists {:.2} ms, roots {:.2} ms, mark {:.2} ms, sweep {:.2} ms",
                direct >> 10,
                (marker.marked_bytes - direct) >> 10,
                ms(built_at),
                ms(free_walked_at - built_at),
                ms(roots_at - free_walked_at),
                ms(marked_at - roots_at),
                ms(started.elapsed() - marked_at)
            );
        }
        let dead_large: Vec<usize> = marker
            .reg
            .large
            .keys()
            .copied()
            .filter(|p| !marker.large_marked.contains(p))
            .collect();
        (
            marker.marked_bytes,
            freed_blocks,
            freed_bytes,
            free_bytes,
            dead_large,
        )
    };
    let large_count = large_freed.len();
    let mut freed_bytes = freed_bytes;
    for payload in large_freed {
        if let Some(total) = reg.large.remove(&payload) {
            freed_bytes += total;
            update(|l| l.heap = l.heap.saturating_sub(total));
            // SAFETY: unreached, and taken out of the registry first.
            unsafe { pool_alloc::free_large(payload) };
        }
    }
    reg.live = live;
    reg.collections += 1;
    // As much again as is live before the next one.
    update(|l| l.budget = live.max(heap_floor()));
    if trace() {
        eprintln!(
            "[gc] #{}: heap {} KB, {} KB carved since last, {} KB reached, {} blocks / {} KB and {} large freed, {} KB free, {:.2} ms",
            reg.collections,
            local().heap >> 10,
            carved >> 10,
            live >> 10,
            freed_blocks,
            freed_bytes >> 10,
            large_count,
            free_bytes >> 10,
            started.elapsed().as_secs_f64() * 1e3
        );
    }
    drop(reg);
    update(|l| l.collecting = false);
}
