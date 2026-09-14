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
//! other thread turns it off for good.
//!
//! ## When it runs
//!
//! Only when the heap grows: the pool counts the bytes it has carved
//! from slab space or taken from libc, and a collection starts when
//! that heap would pass twice what the last collection found live, or
//! a floor below which no collection is worth its time. Between two
//! collections the free lists are used up before anything is carved,
//! so a program whose drop analysis releases everything it makes never
//! collects at all, and one that releases nothing collects once per
//! heap's worth of allocation with the heap held at the bound.
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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// The heap below which nothing is collected: small enough to keep a
/// short program's memory small, large enough that a collection sweeps
/// a worthwhile amount.
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
static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    slabs: BTreeSet::new(),
    large: BTreeMap::new(),
    roots: BTreeMap::new(),
    live: 0,
    collections: 0,
});
/// The thread the collector was enabled on, which is the only one it
/// runs on.
static OWNER: OnceLock<std::thread::ThreadId> = OnceLock::new();

thread_local! {
    /// Bytes the heap holds: slab space carved and large blocks out.
    static HEAP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Bytes carved since the last collection, for the trace.
    static CARVED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The heap size that starts the next collection.
    static THRESHOLD: std::cell::Cell<usize> = const { std::cell::Cell::new(MIN_HEAP) };
    static COLLECTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether this thread is the owner; decided once per thread.
    static IS_OWNER: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
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

/// Turn the collector on, on this thread.
pub fn enable() {
    if disabled_by_env() {
        return;
    }
    let _ = OWNER.set(std::thread::current().id());
    IS_OWNER.with(|c| c.set(None));
    THRESHOLD.with(|t| t.set(heap_floor()));
    ENABLED.store(true, Ordering::SeqCst);
}

/// Turn the collector off; what is allocated stays allocated.
pub fn disable() {
    ENABLED.store(false, Ordering::SeqCst);
}

pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Whether the calling thread is the one the collector belongs to.
fn on_owner_thread() -> bool {
    IS_OWNER.with(|c| match c.get() {
        Some(b) => b,
        None => {
            let b = OWNER.get() == Some(&std::thread::current().id());
            c.set(Some(b));
            b
        }
    })
}

/// A slab the pool has just carved.
pub(crate) fn note_slab(slab: usize) {
    registry().slabs.insert(slab);
}

/// A large block the pool has just handed out.
pub(crate) fn note_large(payload: usize, total: usize) {
    if !is_enabled() {
        return;
    }
    registry().large.insert(payload, total);
    note_carved(total);
}

/// A large block the program has released itself.
pub(crate) fn forget_large(payload: usize) {
    if !is_enabled() {
        return;
    }
    if let Some(total) = registry().large.remove(&payload) {
        if on_owner_thread() {
            HEAP.with(|h| h.set(h.get().saturating_sub(total)));
        }
    }
}

/// The calling thread uses the pool. A second mutator thread has a
/// stack the collector would never read, so it ends the collector.
pub(crate) fn note_thread() -> bool {
    if on_owner_thread() {
        return true;
    }
    disable();
    if trace() {
        eprintln!("[gc] off: the pool is used from a second thread");
    }
    false
}

/// Fresh bytes the pool has handed out.
pub(crate) fn note_carved(bytes: usize) {
    if !is_enabled() || !note_thread() {
        return;
    }
    CARVED.with(|c| c.set(c.get() + bytes));
    HEAP.with(|h| h.set(h.get() + bytes));
}

/// Whether the heap has reached the size that starts a collection.
pub(crate) fn wants_collection() -> bool {
    is_enabled()
        && !COLLECTING.with(|c| c.get())
        && on_owner_thread()
        && HEAP.with(|h| h.get()) >= THRESHOLD.with(|t| t.get())
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
        let slot = pool_alloc::class_slot_bytes(class);
        Box::new(SlabBits {
            class,
            slot,
            count: (used - pool_alloc::SLAB_HEADER) / slot,
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
    /// Every slab, by base address.
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
        let mut bits = SlabMap::with_capacity_and_hasher(reg.slabs.len(), Default::default());
        for &slab in &reg.slabs {
            // SAFETY: a registered slab is live for the life of the process.
            bits.insert(slab, unsafe { SlabBits::read(slab) });
        }
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

    /// Note a block on a free list, which is not storage to follow.
    fn note_free(&mut self, block: usize) {
        let slab = block & !(pool_alloc::SLAB_BYTES - 1);
        if let Some(bits) = self.bits.get_mut(&slab) {
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
        if let Some(bits) = self.bits.get_mut(&slab) {
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
            if a < payload + total - pool_alloc::SLAB_HEADER && self.large_marked.insert(payload) {
                self.marked_bytes += total;
                self.work.push((payload, total - pool_alloc::SLAB_HEADER));
            }
        }
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
    fn sweep(&self) -> (usize, usize) {
        pool_alloc::clear_free_lists();
        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0usize;
        // Highest first, so the lowest addresses come off the lists
        // first. A slab's blocks are chained here and joined to the
        // list in one go.
        let mut slabs: Vec<usize> = self.bits.keys().copied().collect();
        slabs.sort_unstable_by(|a, b| b.cmp(a));
        for slab in slabs {
            let bits = &self.bits[&slab];
            let mut head = 0usize;
            let mut tail = 0usize;
            for idx in (0..bits.count).rev() {
                let base = bits.base(idx);
                let (w, m) = bit(base);
                if bits.marks[w] & m != 0 {
                    continue;
                }
                let block = slab + base;
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
        (freed_blocks, freed_bytes)
    }
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
    if !is_enabled() || !on_owner_thread() || COLLECTING.with(|c| c.get()) {
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
        THRESHOLD.with(|t| t.set(usize::MAX));
        return;
    }
    COLLECTING.with(|c| c.set(true));
    let started = web_time::Instant::now();
    let mut reg = registry();
    let carved = CARVED.with(|c| c.replace(0));

    let (live, freed_blocks, freed_bytes, large_freed) = {
        let mut marker = Marker::new(&reg);
        pool_alloc::for_each_free_block(|b| marker.note_free(b));

        // Roots: the stacks, the globals, what the fiber runtime holds.
        let mut windows: Vec<(usize, usize)> = Vec::new();
        match crate::fiber_backend::installed_fiber_backend() {
            Some(backend) => {
                backend.stack_windows(sp, host_top, &mut |lo, hi| windows.push((lo, hi)));
                backend.held_addresses(&mut |a| marker.consider(a));
            }
            None => windows.push((sp, host_top)),
        }
        for (lo, hi) in windows {
            if lo < hi {
                let before = marker.marked_bytes;
                marker.scan(lo, hi);
                if trace_detail() {
                    eprintln!(
                        "[gc]   stack {lo:#x}..{hi:#x} ({} KB): {} KB reached directly",
                        (hi - lo) >> 10,
                        (marker.marked_bytes - before) >> 10
                    );
                }
            }
        }
        let roots: Vec<(usize, usize)> = marker.reg.roots.iter().map(|(a, l)| (*a, *l)).collect();
        for (a, l) in roots {
            let before = marker.marked_bytes;
            marker.scan(a, a + l);
            if trace_detail() && marker.marked_bytes > before {
                eprintln!(
                    "[gc]   global {a:#x} ({l} bytes): {} KB reached directly",
                    (marker.marked_bytes - before) >> 10
                );
            }
        }
        let direct = marker.marked_bytes;
        marker.drain();
        let marked_at = started.elapsed();
        let (freed_blocks, freed_bytes) = marker.sweep();
        if trace_detail() {
            eprintln!(
                "[gc]   {} KB reached directly, {} KB through what they hold; setup+mark {:.2} ms, sweep {:.2} ms",
                direct >> 10,
                (marker.marked_bytes - direct) >> 10,
                marked_at.as_secs_f64() * 1e3,
                (started.elapsed() - marked_at).as_secs_f64() * 1e3
            );
        }
        let dead_large: Vec<usize> = marker
            .reg
            .large
            .keys()
            .copied()
            .filter(|p| !marker.large_marked.contains(p))
            .collect();
        (marker.marked_bytes, freed_blocks, freed_bytes, dead_large)
    };
    let large_count = large_freed.len();
    let mut freed_bytes = freed_bytes;
    for payload in large_freed {
        if let Some(total) = reg.large.remove(&payload) {
            freed_bytes += total;
            HEAP.with(|h| h.set(h.get().saturating_sub(total)));
            // SAFETY: unreached, and taken out of the registry first.
            unsafe { pool_alloc::free_large(payload) };
        }
    }
    reg.live = live;
    reg.collections += 1;
    // The heap may reach twice what is live before the next one, and
    // never less than the floor.
    THRESHOLD.with(|t| t.set((live * 2).max(heap_floor())));
    if trace() {
        eprintln!(
            "[gc] #{}: heap {} KB, {} KB carved since last, {} KB reached, {} blocks / {} KB and {} large freed, {:.2} ms",
            reg.collections,
            HEAP.with(|h| h.get()) >> 10,
            carved >> 10,
            live >> 10,
            freed_blocks,
            freed_bytes >> 10,
            large_count,
            started.elapsed().as_secs_f64() * 1e3
        );
    }
    drop(reg);
    COLLECTING.with(|c| c.set(false));
}
