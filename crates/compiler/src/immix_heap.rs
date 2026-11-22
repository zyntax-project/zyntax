//! An Immix-style heap behind the host-heap seam.
//!
//! `ZYNTAX_HOST_HEAP=immix` runs a program on it: every allocation comes
//! from one reservation of address space carved into 32 KiB blocks of
//! 128-byte lines, and a conservative mark-sweep reclaims it. Nothing
//! moves, and nothing is released but by a collection, except that a
//! `free` of the last small object a thread allocated gives its bytes
//! back to the thread's region. Any other `free` is a hint the heap
//! ignores.
//!
//! An object of up to a line is bump-allocated and packed at 16 bytes,
//! straddling a line boundary where it falls; a larger one owns whole
//! lines up to [`MEDIUM_MAX`]; anything
//! bigger is a block of its own from the system allocator. Object starts,
//! span lengths and line marks live in side tables, so any address inside
//! an allocation resolves to its start. A sweep frees blocks with no
//! marked line and hands runs of unmarked lines back as bump regions.
//!
//! Collections run on the thread that claimed the heap, from an
//! allocation, and only while no other thread has allocated: another
//! thread's stack is never read.
//!
//! With `ZYNTAX_IMMIX_THREADS=1` any thread that allocates can collect.
//! It stops the others at their next slow-path allocation, inside a
//! [`blocking`] region or at a [`safepoint`], each of which publishes
//! its own stack windows first, and it gives up after a short wait if
//! one does not answer. A thread that runs program code calls [`attach`]
//! before it takes a pointer, and a pointer kept where no collection
//! reads is held with `host_heap::hold`. Plugins reach all of these
//! through `zrtl::heap`. Off by default.

use std::alloc::Layout;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use crate::host_heap::{HeapKind, HostHeap, SpanReader};

const BLOCK: usize = 32 * 1024;
const LINE: usize = 128;
const QUANTUM: usize = 16;
const LINES: usize = BLOCK / LINE;
const QUANTA_PER_LINE: usize = LINE / QUANTUM;
/// The most quanta a small object takes.
const SMALL_QUANTA: usize = LINE / QUANTUM;
const QUANTA: usize = BLOCK / QUANTUM;
const MARK_WORDS: usize = LINES / 64;

/// Largest request served from lines; above it the system allocator.
const MEDIUM_MAX: usize = 16 * 1024;

/// Start byte of a span; 1..=8 is a small object's size in quanta.
const SPAN: u8 = 9;
/// Start-byte bit of an object the open collection reached.
const MARK: u8 = 0x80;
const CODE: u8 = 0x7f;
const MARKS_OF_EIGHT: u64 = 0x8080_8080_8080_8080;

const FREE: u8 = 0;
const USED: u8 = 1;

const DEFAULT_HEAP_MB: usize = 8 * 1024;
/// Allocation volume before the first collection, as the pool's
/// collector waits.
const DEFAULT_FLOOR_KB: usize = 32 * 1024;
const MAX_GROWTH: usize = 4;
const DEFAULT_GROWTH: usize = 2;

fn tracing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZYNTAX_IMMIX_TRACE").is_some())
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// A bump region: allocation proceeds from `cur` to `limit`.
#[derive(Clone, Copy)]
struct Region {
    start: usize,
    cur: usize,
    limit: usize,
}

impl Region {
    const EMPTY: Region = Region {
        start: 0,
        cur: 0,
        limit: 0,
    };
}

thread_local! {
    static SMALL: Cell<Region> = const { Cell::new(Region::EMPTY) };
    static MEDIUM: Cell<Region> = const { Cell::new(Region::EMPTY) };
    /// This thread claimed the heap and runs its collections.
    static OWNER: Cell<bool> = const { Cell::new(false) };
    /// The calling thread's place among the threads that can collect.
    static MINE: RefCell<Option<Mine>> = const { RefCell::new(None) };
    /// Depth of the calling thread's [`blocking`] regions.
    static NATIVE: Cell<u32> = const { Cell::new(0) };
}

/// A thread claimed the heap.
static CLAIMED: AtomicBool = AtomicBool::new(false);
/// A thread other than the owner has allocated.
static FOREIGN: AtomicBool = AtomicBool::new(false);
/// Whether a spent budget starts a collection.
static AUTOMATIC: AtomicBool = AtomicBool::new(true);
/// A collection ran on a thread that may not run the after-collection
/// functions; the owner runs them at its next refill.
static AFTER_PENDING: AtomicBool = AtomicBool::new(false);

struct Pool {
    /// Blocks handed out so far; the rest of the reservation was never
    /// touched and reads as zero.
    cursor: usize,
    /// Blocks every line of which died; their bytes are stale, and
    /// allocation does not clear them.
    free_blocks: Vec<u32>,
    /// Runs of dead lines in blocks with live ones: block, first line,
    /// line count.
    holes: Vec<(u32, u16, u16)>,
}

struct Large {
    size: usize,
    held: bool,
}

enum Root {
    Range(usize),
    Spans { word: usize, reader: SpanReader },
}

/// How long a collection waits for the other threads to stop. A thread
/// that computes without allocating answers late, so the wait is short.
const STOP_WAIT: Duration = Duration::from_millis(20);
/// The wait of a collection the program asked for.
const EXPLICIT_WAIT: Duration = Duration::from_secs(1);
/// A cancelled stop backs the next attempt off by one floor of
/// allocation, doubled per failure up to `1 << MAX_BACKOFF`.
const MAX_BACKOFF: u32 = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    /// Waiting inside a slow-path allocation for a collection to end.
    Parked,
    /// Inside a [`blocking`] region: it touches no heap object.
    Native,
}

/// One thread that can take part in a collection.
struct Thread {
    id: u64,
    name: String,
    state: State,
    /// What the thread published before it stopped: its stack windows and
    /// the addresses its runtime holds. Read only while it is stopped.
    windows: Vec<(usize, usize)>,
    held: Vec<usize>,
}

struct Threads {
    next_id: u64,
    list: Vec<Thread>,
    /// A thread has begun a collection and not yet ended it.
    collecting: bool,
    /// Collections completed.
    epoch: u64,
    /// Stops cancelled since the last collection.
    failures: u32,
    /// Stops cancelled in all.
    cancelled: u64,
}

struct World {
    /// A collection wants the other threads to stop. Read on slow paths.
    stop: AtomicBool,
    inner: Mutex<Threads>,
    /// A thread stopped, or left.
    quiet: Condvar,
    /// A collection ended or was cancelled.
    resume: Condvar,
}

impl World {
    fn new() -> World {
        World {
            stop: AtomicBool::new(false),
            inner: Mutex::new(Threads {
                next_id: 1,
                list: Vec::new(),
                collecting: false,
                epoch: 0,
                failures: 0,
                cancelled: 0,
            }),
            quiet: Condvar::new(),
            resume: Condvar::new(),
        }
    }
}

/// The calling thread's registration; dropping it, when the thread
/// leaves or exits, removes the thread from the world.
struct Mine {
    world: Arc<World>,
    id: u64,
    /// The collection count this thread's open regions belong to.
    epoch: u64,
}

impl Drop for Mine {
    fn drop(&mut self) {
        let mut g = lock(&self.world.inner);
        g.list.retain(|t| t.id != self.id);
        drop(g);
        self.world.quiet.notify_all();
    }
}

pub struct Heap {
    base: usize,
    blocks: usize,
    /// Bytes of the reservation that hold blocks.
    span: usize,
    /// One byte per quantum: 0, a small size in quanta, or [`SPAN`],
    /// with [`MARK`] set once reached.
    starts: *mut u8,
    /// The line count of the span starting at each line.
    spans: *mut u16,
    /// A bit per line, per block, set when a reached object covers it.
    marks: *mut u64,
    state: *mut u8,
    has_span: *mut u8,
    pool: Mutex<Pool>,
    large: Mutex<BTreeMap<usize, Large>>,
    roots: Mutex<BTreeMap<usize, Root>>,
    since_gc: AtomicUsize,
    threshold: AtomicUsize,
    floor: usize,
    growth: usize,
    stress: bool,
    /// Several threads may allocate and collect.
    threads: bool,
    world: Arc<World>,
    live: AtomicUsize,
    collections: AtomicUsize,
}

// SAFETY: the tables are plain memory indexed by block; the pool, the
// large set and the roots sit behind locks, and a collection runs on one
// thread with the others kept out of every shared structure.
unsafe impl Send for Heap {}
unsafe impl Sync for Heap {}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The process's heap, once installed.
static HEAP: OnceLock<&'static Heap> = OnceLock::new();

pub fn active() -> bool {
    HEAP.get().is_some()
}

/// Make the calling thread the one that collects. A second thread
/// claiming it ends collection for the process: two runtimes' stacks
/// cannot both be read.
pub fn claim_owner() {
    let Some(h) = HEAP.get() else {
        return;
    };
    if h.threads {
        // Every thread can collect; the first claimer owns the work that
        // must run on the program's own thread.
        if !CLAIMED.swap(true, Relaxed) {
            OWNER.with(|o| o.set(true));
        }
        h.register();
        return;
    }
    if CLAIMED.swap(true, Relaxed) && !OWNER.with(|o| o.get()) {
        FOREIGN.store(true, Relaxed);
        return;
    }
    OWNER.with(|o| o.set(true));
}

/// Note an allocation by a thread that does not own the heap.
#[inline]
fn note_foreign() {
    if CLAIMED.load(Relaxed) && !OWNER.with(|o| o.get()) {
        FOREIGN.store(true, Relaxed);
    }
}

pub fn set_automatic(on: bool) {
    AUTOMATIC.store(on, Relaxed);
}

/// Stops cancelled so far because a thread did not answer in time.
pub fn cancelled_stops() -> u64 {
    HEAP.get().map_or(0, |h| lock(&h.world.inner).cancelled)
}

pub fn collecting_here() -> bool {
    active() && OWNER.with(|o| o.get())
}

/// A collection, if the calling thread may run one, whose deepest live
/// stack word is at `sp`.
pub fn collect_from_sp(sp: usize) {
    let Some(h) = HEAP.get() else {
        return;
    };
    if h.threads {
        h.collect_threads(sp, EXPLICIT_WAIT, true);
    } else if OWNER.with(|o| o.get()) && !FOREIGN.load(Relaxed) {
        h.collect_from(sp);
    }
}

/// Run `f`, which blocks or waits and touches no heap object, with the
/// calling thread counted as stopped: a collection does not wait for it.
/// What the caller holds is still read, but a heap pointer kept only in
/// `f`'s own frames is not. A plain call to `f` when the heap does not
/// collect across threads.
#[inline]
pub fn blocking<R>(f: impl FnOnce() -> R) -> R {
    match HEAP.get() {
        Some(h) if h.threads => h.blocking(f),
        _ => f(),
    }
}

/// Count the calling thread among those a collection reads from now on,
/// where it would otherwise join at its first allocation. A thread that
/// runs program code calls it before it takes a pointer another thread
/// made, because a thread that has not joined holds pointers no
/// collection sees. A collection then waits for it at its next
/// allocation, wait or safepoint. Does nothing unless the heap collects
/// across threads.
pub fn attach() {
    if let Some(h) = HEAP.get()
        && h.threads
    {
        h.register();
    }
}

/// Stop here while a collection waits for this thread. Called by a loop
/// that neither allocates nor blocks, so that it does not hold one up.
/// One relaxed load when no collection is waiting.
#[inline]
pub fn safepoint() {
    if let Some(h) = HEAP.get() {
        h.safepoint();
    }
}

/// Bytes the calling thread has taken from its open regions.
fn open_bytes() -> usize {
    let used = |c: &Cell<Region>| {
        let r = c.get();
        r.cur - r.start
    };
    SMALL.with(used) + MEDIUM.with(used)
}

pub fn stats() -> crate::collector::Stats {
    match HEAP.get() {
        Some(h) => crate::collector::Stats {
            collections: h.collections.load(Relaxed),
            live: h.live.load(Relaxed),
            // Regions are charged when they are retired; the calling
            // thread's open ones count as far as they are used.
            allocated: h.since_gc.load(Relaxed) + open_bytes(),
        },
        None => crate::collector::Stats::default(),
    }
}

/// Reserve address space that reads as zero and costs nothing until
/// touched.
fn reserve(bytes: usize) -> *mut u8 {
    // SAFETY: an anonymous private mapping of no particular address.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        std::ptr::null_mut()
    } else {
        p as *mut u8
    }
}

impl Heap {
    pub fn new(heap_bytes: usize) -> Option<Heap> {
        let floor = env_usize("ZYNTAX_IMMIX_FLOOR_KB").unwrap_or(DEFAULT_FLOOR_KB) << 10;
        let threads = std::env::var_os("ZYNTAX_IMMIX_THREADS").is_some_and(|v| v == "1");
        Heap::with(heap_bytes, floor, threads)
    }

    fn with(heap_bytes: usize, floor: usize, threads: bool) -> Option<Heap> {
        let blocks = heap_bytes / BLOCK;
        let raw = reserve(blocks * BLOCK + BLOCK);
        let starts = reserve(blocks * QUANTA);
        let spans = reserve(blocks * LINES * 2);
        let marks = reserve(blocks * MARK_WORDS * 8);
        let state = reserve(blocks);
        let has_span = reserve(blocks);
        if [raw, starts, spans, marks, state, has_span]
            .iter()
            .any(|p| p.is_null())
        {
            return None;
        }
        let stress = std::env::var_os("ZYNTAX_IMMIX_STRESS").is_some();
        Some(Heap {
            base: (raw as usize).next_multiple_of(BLOCK),
            blocks,
            span: blocks * BLOCK,
            starts,
            spans: spans as *mut u16,
            marks: marks as *mut u64,
            state,
            has_span,
            pool: Mutex::new(Pool {
                cursor: 0,
                free_blocks: Vec::new(),
                holes: Vec::new(),
            }),
            large: Mutex::new(BTreeMap::new()),
            roots: Mutex::new(BTreeMap::new()),
            since_gc: AtomicUsize::new(0),
            threshold: AtomicUsize::new(if stress { 0 } else { floor }),
            floor,
            growth: env_usize("ZYNTAX_IMMIX_GROWTH").unwrap_or(0),
            stress,
            threads,
            world: Arc::new(World::new()),
            live: AtomicUsize::new(0),
            collections: AtomicUsize::new(0),
        })
    }

    #[inline]
    fn in_range(&self, a: usize) -> bool {
        a.wrapping_sub(self.base) < self.span
    }

    #[inline]
    fn block_addr(&self, b: usize) -> usize {
        self.base + b * BLOCK
    }

    #[inline]
    fn start_at(&self, quantum: usize) -> *mut u8 {
        // SAFETY: the table covers every quantum of the reservation.
        unsafe { self.starts.add(quantum) }
    }

    // -- Allocation ---------------------------------------------------

    /// Takes `a` back when it is the last small object this thread
    /// bumped from its open region: the next allocation reuses its bytes.
    /// Any other block is left to a collection.
    #[inline]
    fn release_top(&self, a: usize) {
        SMALL.with(|cell| {
            // SAFETY: the cell belongs to this thread and nothing else
            // holds a reference into it.
            let r = unsafe { &mut *cell.as_ptr() };
            if a.wrapping_sub(r.start) >= r.cur - r.start {
                return;
            }
            let q = (a - self.base) / QUANTUM;
            // SAFETY: `a` lies in the open region of this heap.
            let code = unsafe { *self.start_at(q) } & CODE;
            if (1..=SMALL_QUANTA as u8).contains(&code) && a + code as usize * QUANTUM == r.cur {
                // SAFETY: as above.
                unsafe { *self.start_at(q) = 0 };
                r.cur = a;
            }
        })
    }

    /// A small object from this thread's region, or null once the
    /// region is spent.
    #[inline(always)]
    fn bump_small(&self, size: usize) -> *mut u8 {
        let quanta = size.max(1).div_ceil(QUANTUM);
        let bytes = quanta * QUANTUM;
        SMALL.with(|cell| {
            // SAFETY: the cell belongs to this thread and nothing else
            // holds a reference into it.
            let r = unsafe { &mut *cell.as_ptr() };
            let p = r.cur;
            if p + bytes > r.limit {
                return std::ptr::null_mut();
            }
            r.cur = p + bytes;
            // SAFETY: `p` lies in a region of this heap.
            unsafe {
                *self.start_at((p - self.base) / QUANTUM) = quanta as u8;
                // Recycled memory is not cleared; the bytes past the
                // request are, as the marker reads to the end.
                std::ptr::write_bytes((p + bytes - QUANTUM) as *mut u8, 0, QUANTUM);
            }
            p as *mut u8
        })
    }

    #[inline(never)]
    fn alloc_small(&self, size: usize) -> *mut u8 {
        loop {
            let p = self.bump_small(size);
            if !p.is_null() {
                return p;
            }
            let r = self.next_region(SMALL.with(|c| c.get()), 1);
            SMALL.with(|c| c.set(r));
            if r.limit == 0 {
                return std::ptr::null_mut();
            }
        }
    }

    fn alloc_medium(&self, size: usize) -> *mut u8 {
        let lines = size.div_ceil(LINE);
        MEDIUM.with(|cell| {
            let mut r = cell.get();
            loop {
                let p = (r.cur + LINE - 1) & !(LINE - 1);
                if p + lines * LINE <= r.limit {
                    r.cur = p + lines * LINE;
                    cell.set(r);
                    let q = (p - self.base) / QUANTUM;
                    let line = (p - self.base) / LINE;
                    let block = (p - self.base) / BLOCK;
                    // SAFETY: `p` lies in a region of this heap.
                    unsafe {
                        *self.start_at(q) = SPAN;
                        *self.spans.add(line) = lines as u16;
                        *self.has_span.add(block) = 1;
                        let from = p + size.next_multiple_of(8);
                        std::ptr::write_bytes(from as *mut u8, 0, p + lines * LINE - from);
                    }
                    return p as *mut u8;
                }
                r = self.next_region(r, lines);
                cell.set(r);
                if r.limit == 0 {
                    return std::ptr::null_mut();
                }
            }
        })
    }

    fn due(&self) -> bool {
        self.since_gc.load(Relaxed) >= self.threshold.load(Relaxed)
    }

    /// A region of at least `lines` lines in place of `old`, which is
    /// charged for what it handed out: a hole a sweep left, else a
    /// block. Empty once the reservation is spent.
    #[inline(never)]
    fn next_region(&self, old: Region, lines: usize) -> Region {
        self.enter_slow_path();
        self.since_gc.fetch_add(old.cur - old.start, Relaxed);
        if self.due() {
            self.collect_now(STOP_WAIT);
        }
        for attempt in 0..2 {
            if let Some(r) = self.take_region(lines) {
                return r;
            }
            if attempt == 0 {
                self.collect_now(EXPLICIT_WAIT);
            }
        }
        Region::EMPTY
    }

    fn take_region(&self, lines: usize) -> Option<Region> {
        let mut pool = lock(&self.pool);
        let (block, first, count) = if let Some(i) = pool
            .holes
            .iter()
            .rposition(|&(_, _, n)| n as usize >= lines)
        {
            let (b, f, n) = pool.holes.swap_remove(i);
            (b as usize, f as usize, n as usize)
        } else if let Some(b) = pool.free_blocks.pop() {
            self.state_set(b as usize, USED);
            (b as usize, 0, LINES)
        } else if pool.cursor < self.blocks {
            let b = pool.cursor;
            pool.cursor += 1;
            self.state_set(b, USED);
            (b, 0, LINES)
        } else {
            return None;
        };
        drop(pool);
        let lo = self.block_addr(block) + first * LINE;
        Some(Region {
            start: lo,
            cur: lo,
            limit: lo + count * LINE,
        })
    }

    fn state_set(&self, block: usize, v: u8) {
        // SAFETY: one byte per block.
        unsafe { *self.state.add(block) = v };
    }

    fn state_of(&self, block: usize) -> u8 {
        // SAFETY: one byte per block.
        unsafe { *self.state.add(block) }
    }

    fn alloc_large(&self, size: usize, held: bool) -> *mut u8 {
        if !held {
            self.enter_slow_path();
            self.since_gc.fetch_add(size, Relaxed);
            if self.due() {
                self.collect_now(STOP_WAIT);
            }
        }
        let Ok(layout) = Layout::from_size_align(size, 16) else {
            return std::ptr::null_mut();
        };
        // SAFETY: a non-zero layout.
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        if !p.is_null() {
            lock(&self.large).insert(p as usize, Large { size, held });
        }
        p
    }

    /// A system-allocated block resized by the system, which can move
    /// its pages in place of copying them. `None` when `ptr` is not one.
    ///
    /// # Safety
    /// `ptr` must not be used again if this returns a block.
    unsafe fn resize_large(&self, ptr: *mut u8, size: usize) -> Option<*mut u8> {
        let mut large = lock(&self.large);
        let &Large { size: old, held } = large.get(&(ptr as usize))?;
        let old_layout = Layout::from_size_align(old, 16).ok()?;
        Layout::from_size_align(size, 16).ok()?;
        // SAFETY: `ptr` was allocated here with `old_layout`; `size` is
        // not zero and makes a valid layout.
        let p = unsafe { std::alloc::realloc(ptr, old_layout, size) };
        if p.is_null() {
            return Some(p);
        }
        large.remove(&(ptr as usize));
        large.insert(p as usize, Large { size, held });
        if held {
            if size > old {
                // SAFETY: the new tail lies inside the block.
                unsafe { std::ptr::write_bytes(p.add(old), 0, size - old) };
            }
        } else {
            self.since_gc.fetch_add(size, Relaxed);
        }
        Some(p)
    }

    // -- Address resolution -------------------------------------------

    /// The size in bytes of the allocation starting at quantum `q`.
    #[inline]
    fn bytes_at(&self, q: usize, code: u8) -> usize {
        if code == SPAN {
            // SAFETY: the span table has an entry per line.
            unsafe { *self.spans.add(q / QUANTA_PER_LINE) as usize * LINE }
        } else {
            code as usize * QUANTUM
        }
    }

    /// The start and size of the heap allocation holding `a`.
    fn containing(&self, a: usize) -> Option<(usize, usize)> {
        if !self.in_range(a) {
            return None;
        }
        let off = a - self.base;
        let block = off / BLOCK;
        let q = off / QUANTUM;
        let floor = q.saturating_sub(SMALL_QUANTA - 1);
        let mut i = q;
        loop {
            // SAFETY: `i` is a quantum of this heap.
            let code = unsafe { *self.start_at(i) } & CODE;
            if code != 0 {
                let bytes = self.bytes_at(i, code);
                return (q < i + bytes / QUANTUM).then(|| (self.base + i * QUANTUM, bytes));
            }
            if i == floor {
                break;
            }
            i -= 1;
        }
        // SAFETY: one flag per block.
        if unsafe { *self.has_span.add(block) } == 0 {
            return None;
        }
        let mut line = off / LINE;
        let first_line = block * LINES;
        loop {
            // SAFETY: the span table has an entry per line.
            let n = unsafe { *self.spans.add(line) } as usize;
            if n != 0 {
                let sq = line * QUANTA_PER_LINE;
                // SAFETY: `sq` is a quantum of this heap.
                let is_span = unsafe { *self.start_at(sq) } & CODE == SPAN;
                return (is_span && off / LINE < line + n)
                    .then(|| (self.base + line * LINE, n * LINE));
            }
            if line == first_line {
                return None;
            }
            line -= 1;
        }
    }

    // -- Collection ---------------------------------------------------

    /// An automatic collection, when one is allowed. Other threads get
    /// `wait` to stop.
    fn collect_now(&self, wait: Duration) {
        if !AUTOMATIC.load(Relaxed) {
            return;
        }
        let mut regs = [0usize; 12];
        crate::collector::spill_callee_saved(&mut regs);
        let sp = regs.as_ptr() as usize;
        if self.threads {
            self.collect_threads(sp, wait, false);
        } else if OWNER.with(|o| o.get()) && !FOREIGN.load(Relaxed) {
            self.collect_from(sp);
        }
        std::hint::black_box(&regs);
    }

    /// A collection on the calling thread alone, whose deepest live
    /// stack word is at `sp`.
    #[inline(never)]
    fn collect_from(&self, sp: usize) {
        let Some((windows, held)) = self.own_roots(sp) else {
            return;
        };
        self.collect_windows(&windows, &held);
        crate::collector::run_after_collection();
    }

    /// The calling thread's roots: its stack windows, with the fiber
    /// stacks it runs and without the hidden ranges, and the addresses
    /// its runtime holds. `None` when its stack's extent is unknown.
    fn own_roots(&self, sp: usize) -> Option<(Vec<(usize, usize)>, Vec<usize>)> {
        let host_top = crate::collector::host_stack_top();
        let backend = crate::fiber_backend::installed_fiber_backend();
        let mut windows: Vec<(usize, usize)> = Vec::new();
        match backend {
            Some(b) => b.stack_windows(sp, host_top, &mut |lo, hi| windows.push((lo, hi))),
            None => windows.push((sp, host_top)),
        }
        if !windows.iter().any(|&(lo, hi)| hi == host_top && lo < hi) {
            return None;
        }
        let windows = crate::collector::without_hidden(windows);
        let mut held = Vec::new();
        if let Some(b) = backend {
            b.held_addresses(&mut |a| held.push(a));
        }
        Some((windows, held))
    }

    // -- Threads ------------------------------------------------------

    /// What every slow-path allocation does first: count the thread among
    /// those that collect, or stop for a collection under way.
    #[inline]
    fn enter_slow_path(&self) {
        if !self.threads {
            note_foreign();
            return;
        }
        debug_assert_eq!(
            NATIVE.with(|n| n.get()),
            0,
            "a heap allocation inside a blocking region"
        );
        self.register();
        if self.world.stop.load(Relaxed) {
            self.park_here();
        }
        if AFTER_PENDING.load(Relaxed)
            && OWNER.with(|o| o.get())
            && AFTER_PENDING.swap(false, Relaxed)
        {
            crate::collector::run_after_collection();
        }
    }

    /// The calling thread's id in this heap's world; it joins on first
    /// use.
    fn register(&self) -> u64 {
        MINE.with(|m| {
            let mut m = m.borrow_mut();
            if let Some(me) = m.as_ref()
                && Arc::ptr_eq(&me.world, &self.world)
            {
                return me.id;
            }
            *m = None;
            if crate::collector::host_stack_top() == 0 {
                // Its stack cannot be bounded, so no collection can read it.
                FOREIGN.store(true, Relaxed);
            }
            let name = std::thread::current()
                .name()
                .unwrap_or("unnamed")
                .to_string();
            let mut g = lock(&self.world.inner);
            let id = g.next_id;
            g.next_id += 1;
            g.list.push(Thread {
                id,
                name,
                state: State::Running,
                windows: Vec::new(),
                held: Vec::new(),
            });
            let epoch = g.epoch;
            drop(g);
            *m = Some(Mine {
                world: self.world.clone(),
                id,
                epoch,
            });
            id
        })
    }

    /// The calling thread leaves the world; its open regions go back to
    /// the next collection.
    fn leave_thread(&self) {
        self.since_gc.fetch_add(open_bytes(), Relaxed);
        SMALL.with(|c| c.set(Region::EMPTY));
        MEDIUM.with(|c| c.set(Region::EMPTY));
        MINE.with(|m| m.borrow_mut().take());
    }

    /// After a wait that may have spanned a collection: the thread's open
    /// regions are stale if one ran, as its sweep treats every line they
    /// have not filled as free.
    fn resync(&self, epoch: u64) {
        let stale = MINE.with(|m| match m.borrow_mut().as_mut() {
            Some(me) if me.epoch != epoch => {
                me.epoch = epoch;
                true
            }
            _ => false,
        });
        if stale {
            SMALL.with(|c| c.set(Region::EMPTY));
            MEDIUM.with(|c| c.set(Region::EMPTY));
        }
    }

    fn set_state(
        g: &mut Threads,
        id: u64,
        state: State,
        windows: Vec<(usize, usize)>,
        held: Vec<usize>,
    ) {
        if let Some(t) = g.list.iter_mut().find(|t| t.id == id) {
            t.state = state;
            t.windows = windows;
            t.held = held;
        }
    }

    #[inline]
    fn safepoint(&self) {
        if self.threads && self.world.stop.load(Relaxed) {
            self.park_here();
        }
    }

    /// Stop here until the collection in progress ends.
    #[inline(never)]
    fn park_here(&self) {
        let mut regs = [0usize; 12];
        crate::collector::spill_callee_saved(&mut regs);
        self.park(regs.as_ptr() as usize);
        std::hint::black_box(&regs);
    }

    /// Publish this thread's roots, whose deepest live word is at `sp`,
    /// and wait for the collection in progress to end.
    fn park(&self, sp: usize) {
        let me = self.register();
        let Some((windows, held)) = self.own_roots(sp) else {
            FOREIGN.store(true, Relaxed);
            return;
        };
        let mut g = lock(&self.world.inner);
        if !g.collecting {
            return;
        }
        Self::set_state(&mut g, me, State::Parked, windows, held);
        self.world.quiet.notify_all();
        while g.collecting {
            g = self.world.resume.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        Self::set_state(&mut g, me, State::Running, Vec::new(), Vec::new());
        let epoch = g.epoch;
        drop(g);
        self.resync(epoch);
    }

    #[inline(never)]
    fn blocking<R>(&self, f: impl FnOnce() -> R) -> R {
        struct Leave<'a>(&'a Heap);
        impl Drop for Leave<'_> {
            fn drop(&mut self) {
                self.0.native_leave();
            }
        }
        let mut regs = [0usize; 12];
        crate::collector::spill_callee_saved(&mut regs);
        self.native_enter(regs.as_ptr() as usize);
        let leave = Leave(self);
        let r = f();
        std::hint::black_box(&regs);
        drop(leave);
        r
    }

    /// Count this thread as stopped, its roots published from `sp` up.
    fn native_enter(&self, sp: usize) {
        let depth = NATIVE.with(|n| n.get());
        NATIVE.with(|n| n.set(depth + 1));
        if depth > 0 {
            return;
        }
        let me = self.register();
        let Some((windows, held)) = self.own_roots(sp) else {
            FOREIGN.store(true, Relaxed);
            return;
        };
        let mut g = lock(&self.world.inner);
        Self::set_state(&mut g, me, State::Native, windows, held);
        drop(g);
        self.world.quiet.notify_all();
    }

    /// The thread runs again, after any collection in progress.
    fn native_leave(&self) {
        let depth = NATIVE.with(|n| n.get());
        if depth == 0 {
            return;
        }
        NATIVE.with(|n| n.set(depth - 1));
        if depth > 1 {
            return;
        }
        let me = self.register();
        let mut g = lock(&self.world.inner);
        while g.collecting
            && g.list
                .iter()
                .any(|t| t.id == me && t.state == State::Native)
        {
            g = self.world.resume.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        Self::set_state(&mut g, me, State::Running, Vec::new(), Vec::new());
        let epoch = g.epoch;
        drop(g);
        self.resync(epoch);
    }

    /// A collection that stops every other thread first. `wait` bounds
    /// the stop. A program's own request waits out another thread's
    /// collection and then runs one, and does not back later attempts
    /// off when it fails.
    #[inline(never)]
    fn collect_threads(&self, sp: usize, wait: Duration, explicit: bool) {
        if FOREIGN.load(Relaxed) {
            return;
        }
        let me = self.register();
        let world = &*self.world;
        let mut g = lock(&world.inner);
        while g.collecting {
            drop(g);
            self.park(sp);
            if !explicit {
                return;
            }
            g = lock(&world.inner);
        }
        g.collecting = true;
        world.stop.store(true, Relaxed);
        let deadline = Instant::now() + wait;
        while g
            .list
            .iter()
            .any(|t| t.id != me && t.state == State::Running)
        {
            let now = Instant::now();
            if now >= deadline {
                self.cancel_stop(g, me, !explicit);
                return;
            }
            g = world
                .quiet
                .wait_timeout(g, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        let Some((mut windows, mut held)) = self.own_roots(sp) else {
            self.end_stop(g);
            return;
        };
        for t in g.list.iter().filter(|t| t.id != me) {
            windows.extend_from_slice(&t.windows);
            held.extend_from_slice(&t.held);
        }
        self.collect_windows(&windows, &held);
        g.epoch += 1;
        g.failures = 0;
        let epoch = g.epoch;
        self.end_stop(g);
        self.resync(epoch);
        if OWNER.with(|o| o.get()) || !CLAIMED.load(Relaxed) {
            crate::collector::run_after_collection();
        } else {
            AFTER_PENDING.store(true, Relaxed);
        }
    }

    /// Give up a stop that was not answered in time, and wait longer
    /// before the next.
    fn cancel_stop(&self, mut g: MutexGuard<'_, Threads>, me: u64, back_off: bool) {
        let late: Vec<String> = g
            .list
            .iter()
            .filter(|t| t.id != me && t.state == State::Running)
            .map(|t| format!("{} ({})", t.id, t.name))
            .collect();
        g.cancelled += 1;
        if back_off {
            g.failures = (g.failures + 1).min(MAX_BACKOFF);
            let more = self.floor << g.failures;
            self.threshold
                .store(self.since_gc.load(Relaxed) + more, Relaxed);
        }
        if tracing() {
            eprintln!(
                "[immix] stop cancelled: no answer from thread {}",
                late.join(", ")
            );
        }
        self.end_stop(g);
    }

    /// End the stop and wake the threads waiting on it.
    fn end_stop(&self, mut g: MutexGuard<'_, Threads>) {
        g.collecting = false;
        self.world.stop.store(false, Relaxed);
        drop(g);
        self.world.resume.notify_all();
    }

    /// Mark from the stack `windows`, the registered roots and the
    /// `held` addresses, then sweep.
    fn collect_windows(&self, windows: &[(usize, usize)], held: &[usize]) {
        let began = std::time::Instant::now();
        SMALL.with(|c| c.set(Region::EMPTY));
        MEDIUM.with(|c| c.set(Region::EMPTY));

        let mut pool = lock(&self.pool);
        let mut large = lock(&self.large);
        let roots = lock(&self.roots);

        let large_lo = large.keys().next().copied().unwrap_or(0);
        let large_hi = large
            .iter()
            .map(|(&a, l)| a + l.size.max(1))
            .max()
            .unwrap_or(0);
        let mut m = Marker {
            h: self,
            stack: Vec::new(),
            marked: 0,
            large: large.iter().map(|(&a, l)| (a, l.size, l.held)).collect(),
            large_marked: vec![false; large.len()],
            large_lo,
            large_span: large_hi.wrapping_sub(large_lo),
        };
        let (hooks, sweepers) = crate::collector::weak_registry();
        for h in &hooks {
            (h.hold)(&mut |a| m.hold(a));
        }
        for &(lo, hi) in windows {
            m.scan(lo, hi);
        }
        for (&a, root) in roots.iter() {
            match root {
                Root::Range(len) => {
                    if crate::collector::mapped(a, a + len) {
                        m.scan(a, a + len);
                    }
                }
                Root::Spans { word, reader } => {
                    // SAFETY: the reader is the runtime's, called with
                    // the word it was registered with.
                    unsafe {
                        reader(
                            a as *const u8,
                            *word,
                            visit_span,
                            &mut m as *mut _ as *mut c_void,
                        )
                    };
                }
            }
        }
        for i in 0..m.large.len() {
            let (a, size, held) = m.large[i];
            if held {
                m.large_marked[i] = true;
                m.scan(a, a + size);
            }
        }
        for &a in held {
            m.consider(a);
        }
        m.drain();
        if !hooks.is_empty() {
            crate::collector::settle_weak(&mut m, &hooks);
        }
        if !sweepers.is_empty() {
            let reached = |a: usize| m.is_marked(a);
            for f in &sweepers {
                f(&reached);
            }
        }
        let marked_at = began.elapsed();

        let live = m.marked;
        let large_marked = std::mem::take(&mut m.large_marked);
        let large_snapshot = std::mem::take(&mut m.large);
        drop(m);
        let mut freed = self.sweep(&mut pool);
        for (i, &(a, size, _)) in large_snapshot.iter().enumerate() {
            if !large_marked[i] {
                if let Ok(layout) = Layout::from_size_align(size, 16) {
                    // SAFETY: allocated with this layout, unreached.
                    unsafe { std::alloc::dealloc(a as *mut u8, layout) };
                }
                large.remove(&a);
                freed += size;
            }
        }
        drop(roots);
        drop(large);
        drop(pool);

        self.live.store(live, Relaxed);
        self.collections.fetch_add(1, Relaxed);
        let growth = if self.growth != 0 {
            self.growth
        } else if freed * 8 < live {
            MAX_GROWTH
        } else {
            DEFAULT_GROWTH
        };
        self.since_gc.store(0, Relaxed);
        self.threshold.store(
            if self.stress {
                0
            } else {
                (live * growth).max(self.floor)
            },
            Relaxed,
        );
        if tracing() {
            eprintln!(
                "[immix] #{}: {} KB reached, {} KB freed, next after {} KB, mark {:.2} ms, total {:.2} ms",
                self.collections.load(Relaxed),
                live >> 10,
                freed >> 10,
                self.threshold.load(Relaxed) >> 10,
                marked_at.as_secs_f64() * 1e3,
                began.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// Free what no mark reached: blocks wholly dead return to the pool,
    /// the dead line runs of the rest become regions. Returns the bytes
    /// freed.
    fn sweep(&self, pool: &mut Pool) -> usize {
        pool.holes.clear();
        let mut freed = 0usize;
        for b in 0..pool.cursor {
            if self.state_of(b) != USED {
                continue;
            }
            // SAFETY: a block of this heap's tables.
            let marks = unsafe {
                std::slice::from_raw_parts_mut(self.marks.add(b * MARK_WORDS), MARK_WORDS)
            };
            let starts = unsafe { self.starts.add(b * QUANTA) } as *mut u64;
            if marks.iter().all(|w| *w == 0) {
                // SAFETY: the block's own table entries.
                unsafe {
                    std::ptr::write_bytes(starts as *mut u8, 0, QUANTA);
                    std::ptr::write_bytes(self.spans.add(b * LINES), 0, LINES);
                    *self.has_span.add(b) = 0;
                }
                self.state_set(b, FREE);
                pool.free_blocks.push(b as u32);
                freed += BLOCK;
                continue;
            }
            let mut run_start = 0usize;
            let mut in_run = false;
            for l in 0..LINES {
                let marked = marks[l / 64] & (1u64 << (l % 64)) != 0;
                // SAFETY: eight start bytes of line `l`.
                unsafe {
                    let w = starts.add(l);
                    if marked {
                        *w &= !MARKS_OF_EIGHT;
                    } else {
                        *w = 0;
                        *self.spans.add(b * LINES + l) = 0;
                    }
                }
                if marked {
                    if in_run {
                        pool.holes
                            .push((b as u32, run_start as u16, (l - run_start) as u16));
                        freed += (l - run_start) * LINE;
                        in_run = false;
                    }
                } else if !in_run {
                    in_run = true;
                    run_start = l;
                }
            }
            if in_run {
                pool.holes
                    .push((b as u32, run_start as u16, (LINES - run_start) as u16));
                freed += (LINES - run_start) * LINE;
            }
            marks.fill(0);
        }
        freed
    }
}

impl crate::collector::Marking for Marker<'_> {
    fn is_marked(&self, a: usize) -> bool {
        Marker::is_marked(self, a)
    }

    fn mark(&mut self, a: usize) {
        self.consider(a);
        self.drain();
    }
}

impl crate::collector::Tracing for Marker<'_> {
    fn reached_bytes(&self) -> usize {
        self.marked
    }

    fn follow(&mut self) {
        self.drain();
    }
}

unsafe extern "C" fn visit_span(cx: *mut c_void, lo: usize, hi: usize) {
    // SAFETY: `cx` is the marker that called the reader.
    let m = unsafe { &mut *(cx as *mut Marker<'_>) };
    m.scan(lo, hi);
}

struct Marker<'h> {
    h: &'h Heap,
    stack: Vec<(usize, usize)>,
    marked: usize,
    /// The system-allocated blocks, in address order.
    large: Vec<(usize, usize, bool)>,
    large_marked: Vec<bool>,
    /// The lowest address of a system-allocated block and the distance
    /// to the end of the highest: a word outside it names none.
    large_lo: usize,
    large_span: usize,
}

impl Marker<'_> {
    fn scan(&mut self, lo: usize, hi: usize) {
        let (base, span) = (self.h.base, self.h.span);
        let (large_lo, large_span) = (self.large_lo, self.large_span);
        let mut p = lo.next_multiple_of(8);
        while p + 8 <= hi {
            // SAFETY: the caller names memory it holds; a racing write
            // can only make a word stale.
            let w = unsafe { std::ptr::read_volatile(p as *const usize) };
            if w.wrapping_sub(base) < span {
                self.consider_heap(w);
            } else if w.wrapping_sub(large_lo) < large_span {
                self.consider_large(w);
            }
            p += 8;
        }
    }

    fn consider(&mut self, a: usize) {
        if a.wrapping_sub(self.h.base) < self.h.span {
            self.consider_heap(a);
        } else if a.wrapping_sub(self.large_lo) < self.large_span {
            self.consider_large(a);
        }
    }

    /// The start nearest at or below quantum `q`, within the longest
    /// small object: its quantum and size byte.
    #[inline(always)]
    fn start_below(&self, q: usize) -> Option<(usize, u8)> {
        let starts = self.h.starts;
        if q >= SMALL_QUANTA - 1 {
            // SAFETY: the eight bytes ending at quantum `q` are in the
            // table, which has a byte per quantum.
            let w =
                unsafe { std::ptr::read_unaligned(starts.add(q + 1 - SMALL_QUANTA) as *const u64) };
            // Bit 7 of each byte whose size code is not zero.
            let set = ((w & !MARKS_OF_EIGHT).wrapping_add(!MARKS_OF_EIGHT)) & MARKS_OF_EIGHT;
            if set == 0 {
                return None;
            }
            let back = (set.leading_zeros() / 8) as usize;
            let i = q - back;
            return Some((i, (w >> (8 * (SMALL_QUANTA - 1 - back))) as u8));
        }
        let mut i = q;
        loop {
            // SAFETY: `i` is a quantum of the heap.
            let b = unsafe { *starts.add(i) };
            if b & CODE != 0 {
                return Some((i, b));
            }
            if i == 0 {
                return None;
            }
            i -= 1;
        }
    }

    fn consider_heap(&mut self, a: usize) {
        let h = self.h;
        let q = (a - h.base) / QUANTUM;
        if let Some((i, b)) = self.start_below(q) {
            if b & MARK == 0 {
                let bytes = h.bytes_at(i, b & CODE);
                if q < i + bytes / QUANTUM {
                    self.mark_at(i, bytes);
                }
            }
            return;
        }
        // An address inside a span, past its first line.
        if let Some((start, bytes)) = h.containing(a) {
            let q = (start - h.base) / QUANTUM;
            // SAFETY: `q` is a quantum of the heap.
            if unsafe { *h.starts.add(q) } & MARK == 0 {
                self.mark_at(q, bytes);
            }
        }
    }

    fn consider_large(&mut self, a: usize) {
        let i = self.large.partition_point(|&(s, _, _)| s <= a);
        if i == 0 {
            return;
        }
        let (s, size, _) = self.large[i - 1];
        if a < s + size.max(1) && !self.large_marked[i - 1] {
            self.large_marked[i - 1] = true;
            self.marked += size;
            self.stack.push((s, size));
        }
    }

    /// Claim the allocation of `bytes` starting at quantum `q`, to be
    /// scanned.
    fn mark_at(&mut self, q: usize, bytes: usize) {
        self.claim(q, bytes);
        self.stack.push((self.h.base + q * QUANTUM, bytes));
    }

    /// Take the allocation of `bytes` starting at quantum `q` as
    /// reached, without scanning it.
    fn claim(&mut self, q: usize, bytes: usize) {
        let h = self.h;
        // SAFETY: `q` is a quantum of the heap.
        unsafe { *h.starts.add(q) |= MARK };
        let block = q / QUANTA;
        let in_block = (q % QUANTA) * QUANTUM;
        let first = in_block / LINE;
        let last = (in_block + bytes - 1) / LINE;
        for l in first..=last {
            // SAFETY: a mark word of the block.
            unsafe {
                *h.marks.add(block * MARK_WORDS + l / 64) |= 1u64 << (l % 64);
            }
        }
        self.marked += bytes;
    }

    /// Take the allocation `a` lands in as reached without reading it:
    /// its words are a weak-reference hook's to follow.
    fn hold(&mut self, a: usize) {
        let h = self.h;
        if let Some((start, bytes)) = h.containing(a) {
            let q = (start - h.base) / QUANTUM;
            // SAFETY: `q` is a quantum of the heap.
            if unsafe { *h.starts.add(q) } & MARK == 0 {
                self.claim(q, bytes);
            }
            return;
        }
        if let Some(i) = self.large_at(a)
            && !self.large_marked[i]
        {
            self.large_marked[i] = true;
            self.marked += self.large[i].1;
        }
    }

    /// The system-allocated block `a` lands in.
    fn large_at(&self, a: usize) -> Option<usize> {
        let i = self.large.partition_point(|&(s, _, _)| s <= a);
        let (s, size, _) = *self.large.get(i.checked_sub(1)?)?;
        (a < s + size.max(1)).then_some(i - 1)
    }

    /// Whether what `a` lands in is reached. What the heap does not
    /// manage counts as reached, so it is never taken for dead.
    fn is_marked(&self, a: usize) -> bool {
        let h = self.h;
        if h.in_range(a) {
            return match h.containing(a) {
                Some((start, _)) => {
                    // SAFETY: the start of an allocation of this heap.
                    let byte = unsafe { *h.starts.add((start - h.base) / QUANTUM) };
                    byte & MARK != 0
                }
                None => true,
            };
        }
        match self.large_at(a) {
            Some(i) => self.large_marked[i],
            None => true,
        }
    }

    fn drain(&mut self) {
        while let Some((s, bytes)) = self.stack.pop() {
            self.scan(s, s + bytes);
        }
    }
}

// -- The host-heap table ------------------------------------------------

unsafe extern "C" fn alloc(cx: *mut c_void, size: usize, kind: HeapKind) -> *mut u8 {
    // SAFETY: the context is the heap this table was made for.
    let h = unsafe { &*(cx as *const Heap) };
    if kind == HeapKind::Collected && size <= LINE {
        let p = h.bump_small(size);
        if !p.is_null() {
            return p;
        }
    }
    alloc_general(h, size, kind)
}

/// Every allocation the bump of a small object does not serve.
#[inline(never)]
fn alloc_general(h: &Heap, size: usize, kind: HeapKind) -> *mut u8 {
    let size = size.max(1);
    if kind == HeapKind::Held || size > MEDIUM_MAX {
        h.alloc_large(size, kind == HeapKind::Held)
    } else if size <= LINE {
        h.alloc_small(size)
    } else {
        h.alloc_medium(size)
    }
}

unsafe extern "C" fn free(cx: *mut c_void, ptr: *mut u8) {
    // SAFETY: as in `alloc`.
    let h = unsafe { &*(cx as *const Heap) };
    if h.in_range(ptr as usize) {
        h.release_top(ptr as usize);
        return;
    }
    let mut large = lock(&h.large);
    if let Some(l) = large.remove(&(ptr as usize))
        && let Ok(layout) = Layout::from_size_align(l.size, 16)
    {
        // SAFETY: allocated with this layout and just forgotten.
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

unsafe extern "C" fn realloc(cx: *mut c_void, ptr: *mut u8, size: usize) -> *mut u8 {
    // SAFETY: as in `alloc`.
    let h = unsafe { &*(cx as *const Heap) };
    if h.threads {
        // The size lookup below reads tables a collection rewrites; a
        // registered thread is stopped before one runs.
        h.register();
    }
    if size > MEDIUM_MAX && !h.in_range(ptr as usize) {
        // SAFETY: the block is the caller's, from this heap.
        if let Some(p) = unsafe { h.resize_large(ptr, size) } {
            return p;
        }
    }
    let (old, held) = if let Some((s, bytes)) = h.containing(ptr as usize) {
        if s == ptr as usize && size <= bytes {
            return ptr;
        }
        (bytes, false)
    } else {
        match lock(&h.large).get(&(ptr as usize)) {
            Some(l) => (l.size, l.held),
            None => return std::ptr::null_mut(),
        }
    };
    let kind = if held {
        HeapKind::Held
    } else {
        HeapKind::Collected
    };
    // SAFETY: as in `alloc`.
    let fresh = unsafe { alloc(cx, size, kind) };
    if !fresh.is_null() {
        // SAFETY: both blocks are at least `old.min(size)` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, fresh, old.min(size));
            free(cx, ptr);
        }
    }
    fresh
}

unsafe extern "C" fn owns(cx: *mut c_void, ptr: *const u8) -> bool {
    // SAFETY: as in `alloc`.
    let h = unsafe { &*(cx as *const Heap) };
    let a = ptr as usize;
    if h.in_range(a) {
        return true;
    }
    lock(&h.large)
        .range(..=a)
        .next_back()
        .is_some_and(|(s, l)| a < s + l.size.max(1))
}

unsafe extern "C" fn add_root_range(cx: *mut c_void, start: *const u8, len: usize) {
    // SAFETY: as in `alloc`.
    let h = unsafe { &*(cx as *const Heap) };
    lock(&h.roots).insert(start as usize, Root::Range(len));
}

unsafe extern "C" fn add_root_spans(
    cx: *mut c_void,
    start: *const u8,
    word: usize,
    reader: SpanReader,
) {
    // SAFETY: as in `alloc`.
    let h = unsafe { &*(cx as *const Heap) };
    lock(&h.roots).insert(start as usize, Root::Spans { word, reader });
}

unsafe extern "C" fn remove_root_range(cx: *mut c_void, start: *const u8) {
    // SAFETY: as in `alloc`.
    let h = unsafe { &*(cx as *const Heap) };
    lock(&h.roots).remove(&(start as usize));
}

/// A thread joins the world at its first slow-path allocation, so a
/// thread that never allocates is never waited for.
unsafe extern "C" fn thread_enter(_cx: *mut c_void, _top: *const u8) {}

unsafe extern "C" fn thread_leave(cx: *mut c_void) {
    // SAFETY: as in `alloc`.
    let h = unsafe { &*(cx as *const Heap) };
    if h.threads {
        h.leave_thread();
    }
}

/// Reserve the heap and make it the process's. `None` when the address
/// space is not available or one is already installed.
pub fn table() -> Option<HostHeap> {
    let mb = env_usize("ZYNTAX_IMMIX_HEAP_MB").unwrap_or(DEFAULT_HEAP_MB);
    let heap: &'static Heap = Box::leak(Box::new(Heap::new(mb << 20)?));
    HEAP.set(heap).ok()?;
    Some(HostHeap {
        version: crate::host_heap::HOST_HEAP_VERSION,
        size: std::mem::size_of::<HostHeap>() as u32,
        context: heap as *const Heap as *mut c_void,
        alloc,
        free,
        realloc,
        owns,
        add_root_range,
        add_root_spans,
        remove_root_range,
        thread_enter,
        thread_leave,
        type_header: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn heap() -> Heap {
        OWNER.with(|o| o.set(true));
        Heap::new(64 << 20).expect("a reservation")
    }

    fn make(h: &Heap, size: usize) -> usize {
        // SAFETY: the context is this heap.
        unsafe { alloc(h as *const Heap as *mut c_void, size, HeapKind::Collected) as usize }
    }

    /// Roots as a boxed array of addresses the heap reads.
    fn root(h: &Heap, words: &mut Vec<usize>) {
        lock(&h.roots).insert(words.as_ptr() as usize, Root::Range(words.len() * 8));
    }

    fn collect(h: &Heap) {
        h.collect_windows(&[], &[]);
    }

    #[test]
    fn the_nearest_start_below_matches_a_byte_walk() {
        let h = heap();
        let mut seed = 0x1234_5678_9abc_def1u64;
        let n = 4096;
        for q in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let code = match seed % 7 {
                0 => (seed >> 8) % 8 + 1,
                1 => SPAN as u64,
                _ => 0,
            } as u8;
            let mark = if code != 0 && (seed >> 20) % 3 == 0 {
                MARK
            } else {
                0
            };
            // SAFETY: a quantum of the heap.
            unsafe { *h.starts.add(q) = code | mark };
        }
        let m = Marker {
            h: &h,
            stack: Vec::new(),
            marked: 0,
            large: Vec::new(),
            large_marked: Vec::new(),
            large_lo: 0,
            large_span: 0,
        };
        for q in 0..n {
            let floor = q.saturating_sub(SMALL_QUANTA - 1);
            let mut i = q;
            let want = loop {
                // SAFETY: a quantum of the heap.
                let b = unsafe { *h.starts.add(i) };
                if b & CODE != 0 {
                    break Some((i, b));
                }
                if i == floor {
                    break None;
                }
                i -= 1;
            };
            assert_eq!(m.start_below(q), want, "quantum {q}");
        }
    }

    #[test]
    fn small_objects_are_aligned_slack_cleared_and_packed() {
        let h = heap();
        let mut last = 0;
        for size in [1, 16, 24, 40, 72, 100, 128] {
            for _ in 0..200 {
                let p = make(&h, size);
                assert_eq!(p % 16, 0);
                let bytes = size.div_ceil(16) * 16;
                // SAFETY: the allocation is `bytes` long.
                assert!(
                    unsafe { std::slice::from_raw_parts((p + size) as *const u8, bytes - size) }
                        .iter()
                        .all(|b| *b == 0)
                );
                if last != 0 && (p - h.base) / BLOCK == (last - h.base) / BLOCK {
                    assert!(p - last <= 128, "a gap of {} between objects", p - last);
                }
                last = p;
            }
        }
    }

    #[test]
    fn an_object_across_a_line_boundary_is_found_from_every_word() {
        let h = heap();
        let a = make(&h, 72);
        let b = make(&h, 72);
        assert_eq!(b, a + 80);
        for off in (0..72).step_by(8) {
            assert_eq!(h.containing(b + off).map(|x| x.0), Some(b), "word {off}");
        }
    }

    #[test]
    fn an_interior_address_finds_its_allocation_and_one_past_does_not() {
        let h = heap();
        let a = make(&h, 48);
        let b = make(&h, 1000);
        assert_eq!(h.containing(a + 47).map(|x| x.0), Some(a));
        assert_eq!(h.containing(b + 999).map(|x| x.0), Some(b));
        assert_eq!(h.containing(b + 1023).map(|x| x.0), Some(b));
        assert_ne!(h.containing(a + 48).map(|x| x.0), Some(a));
        assert_eq!(h.containing(b + 1024), None);
    }

    #[test]
    fn what_a_root_reaches_survives_and_the_rest_is_reclaimed() {
        let h = heap();
        let kept = make(&h, 64);
        let behind = make(&h, 64);
        let dead = make(&h, 64);
        let medium = make(&h, 3000);
        let dead_medium = make(&h, 3000);
        // SAFETY: `kept` is 64 bytes.
        unsafe { *(kept as *mut usize) = behind };
        let mut roots = vec![kept, medium + 100];
        root(&h, &mut roots);
        collect(&h);
        assert!(h.containing(kept).is_some());
        assert!(h.containing(behind).is_some(), "reached through an object");
        assert!(
            h.containing(medium).is_some(),
            "reached through an interior word"
        );
        assert!(h.containing(dead_medium).is_none());
        assert!(h.containing(dead).is_none() || h.containing(dead).map(|x| x.0) != Some(dead));
    }

    #[test]
    fn a_dead_line_is_reused_with_its_slack_cleared() {
        let h = heap();
        let mut keep = Vec::new();
        for i in 0..4000 {
            let p = make(&h, 60);
            let m = make(&h, 200);
            // SAFETY: the allocations are 64 and 256 bytes long.
            unsafe {
                std::ptr::write_bytes(p as *mut u8, 0xAB, 64);
                std::ptr::write_bytes(m as *mut u8, 0xAB, 256);
            }
            if i % 50 == 0 {
                keep.push(p);
            }
        }
        root(&h, &mut keep);
        collect(&h);
        let before = h.pool.lock().unwrap().cursor;
        for _ in 0..1500 {
            let p = make(&h, 60);
            let m = make(&h, 200);
            // SAFETY: the allocations are 64 and 256 bytes long.
            let (ps, ms) = unsafe {
                (
                    std::slice::from_raw_parts((p + 60) as *const u8, 4),
                    std::slice::from_raw_parts((m + 200) as *const u8, 56),
                )
            };
            assert!(ps.iter().all(|b| *b == 0), "slack of a small object");
            assert!(ms.iter().all(|b| *b == 0), "slack of a medium object");
            assert!(!keep.contains(&p), "a live object was handed out again");
        }
        assert_eq!(h.pool.lock().unwrap().cursor, before, "took fresh blocks");
    }

    #[test]
    fn a_freed_last_small_object_is_reused_and_others_are_not() {
        let h = heap();
        let cx = &h as *const Heap as *mut c_void;
        let a = make(&h, 48);
        let b = make(&h, 48);
        // SAFETY: the context is this heap and the blocks are its own.
        unsafe { free(cx, a as *mut u8) };
        let c = make(&h, 48);
        assert_ne!(c, a, "a block under a later one stays allocated");
        // SAFETY: as above.
        unsafe { free(cx, c as *mut u8) };
        let d = make(&h, 48);
        assert_eq!(d, c, "the last block is handed out again");
        assert_ne!(d, b);
        // A second free of the same block, and one of an interior
        // address, change nothing.
        unsafe {
            free(cx, d as *mut u8);
            free(cx, d as *mut u8);
            free(cx, (b + 16) as *mut u8);
        }
        let e = make(&h, 16);
        assert_eq!(e, d);
        let f = make(&h, 16);
        assert_eq!(f, e + 16, "a repeated free took back one block only");
    }

    #[test]
    fn a_reused_small_block_has_its_slack_cleared_and_its_own_size() {
        let h = heap();
        let cx = &h as *const Heap as *mut c_void;
        let a = make(&h, 64);
        // SAFETY: the block is 64 bytes.
        unsafe { std::ptr::write_bytes(a as *mut u8, 0xff, 64) };
        // SAFETY: the context is this heap.
        unsafe { free(cx, a as *mut u8) };
        let b = make(&h, 40);
        assert_eq!(b, a);
        // SAFETY: the last quantum of a 48-byte block.
        let slack = unsafe { std::slice::from_raw_parts((b + 32) as *const u8, 16) };
        assert!(slack.iter().all(|x| *x == 0));
        assert_eq!(h.containing(b), Some((b, 48)));
        let c = make(&h, 16);
        assert_eq!(c, b + 48);
    }

    #[test]
    fn a_free_with_no_open_region_changes_nothing() {
        let h = heap();
        let cx = &h as *const Heap as *mut c_void;
        let a = make(&h, 32);
        collect(&h);
        // The collection closed the region; `a` died with it.
        // SAFETY: the context is this heap.
        unsafe { free(cx, a as *mut u8) };
        let b = make(&h, 32);
        let c = make(&h, 32);
        assert_eq!(c, b + 32);
        assert_eq!(h.containing(b), Some((b, 32)));
    }

    #[test]
    fn large_and_held_blocks_follow_their_rules() {
        let h = heap();
        let cx = &h as *const Heap as *mut c_void;
        let big = make(&h, 100_000);
        let dead_big = make(&h, 100_000);
        // SAFETY: the context is this heap.
        let held = unsafe { alloc(cx, 64, HeapKind::Held) } as usize;
        let behind = make(&h, 64);
        // SAFETY: 64 bytes.
        unsafe { *(held as *mut usize) = behind };
        let mut roots = vec![big + 5000];
        root(&h, &mut roots);
        collect(&h);
        // SAFETY: as above.
        assert!(unsafe { owns(cx, big as *const u8) });
        assert!(!unsafe { owns(cx, dead_big as *const u8) });
        assert!(h.containing(behind).is_some(), "a held block is a root");
        unsafe { free(cx, held as *mut u8) };
        collect(&h);
        assert!(h.containing(behind).is_none());
    }

    #[test]
    fn a_large_block_resized_keeps_its_contents_and_stays_owned() {
        let h = heap();
        let cx = &h as *const Heap as *mut c_void;
        let a = make(&h, 40_000);
        for i in 0..40_000 / 8 {
            // SAFETY: inside the block.
            unsafe { *(a as *mut usize).add(i) = i };
        }
        // SAFETY: the context is this heap, and `a` is its block.
        let b = unsafe { realloc(cx, a as *mut u8, 300_000) } as usize;
        assert_ne!(b, 0);
        for i in 0..40_000 / 8 {
            // SAFETY: inside the block.
            assert_eq!(unsafe { *(b as *const usize).add(i) }, i);
        }
        assert!(unsafe { owns(cx, b as *const u8) });
        assert!(unsafe { owns(cx, (b + 299_999) as *const u8) });
        // SAFETY: as above.
        let c = unsafe { realloc(cx, b as *mut u8, 20_000) } as usize;
        assert_ne!(c, 0);
        assert_eq!(unsafe { *(c as *const usize).add(7) }, 7);
        assert!(unsafe { owns(cx, c as *const u8) });
        let mut roots = vec![c];
        root(&h, &mut roots);
        collect(&h);
        assert!(unsafe { owns(cx, c as *const u8) });
    }

    #[test]
    fn a_collection_with_nothing_left_frees_every_block() {
        let h = heap();
        for _ in 0..2000 {
            make(&h, 100);
        }
        collect(&h);
        let pool = h.pool.lock().unwrap();
        assert_eq!(pool.free_blocks.len(), pool.cursor);
        assert!(pool.holes.is_empty());
    }

    // -- Several threads -----------------------------------------------

    const MAGIC: usize = 0x5eed_0000;

    fn threaded(floor_kb: usize) -> &'static Heap {
        watchdog();
        Box::leak(Box::new(
            Heap::with(256 << 20, floor_kb << 10, true).expect("a reservation"),
        ))
    }

    /// A test that hangs ends the process instead of the run.
    fn watchdog() {
        std::thread::spawn(|| {
            std::thread::sleep(Duration::from_secs(120));
            eprintln!("a threaded heap test hung");
            std::process::abort();
        });
    }

    fn fill(p: usize, size: usize) {
        // SAFETY: the allocation is `size` bytes.
        unsafe { std::ptr::write_bytes(p as *mut u8, 0xAA, size) };
    }

    /// A list of `n` nodes in the heap; the caller's stack holds the head.
    fn build(h: &Heap, n: usize) -> usize {
        let mut head = 0usize;
        for i in 0..n {
            let node = make(h, 32);
            // SAFETY: the node is 32 bytes.
            unsafe {
                *(node as *mut usize) = head;
                *(node as *mut usize).add(1) = MAGIC + i;
            }
            head = node;
        }
        head
    }

    /// The length of the list at `head`, which stops at the first node
    /// that does not carry the tag.
    fn walk(mut head: usize) -> usize {
        let mut n = 0;
        while head != 0 {
            // SAFETY: a node of the list, or memory the heap owns.
            let (next, tag) = unsafe { (*(head as *const usize), *(head as *const usize).add(1)) };
            if tag & !0xffff != MAGIC {
                break;
            }
            n += 1;
            head = next;
        }
        n
    }

    /// How many nodes of the list at `head` the heap still holds.
    fn held_nodes(h: &Heap, mut head: usize) -> usize {
        let mut n = 0;
        while head != 0 && h.containing(head) == Some((head, 32)) {
            n += 1;
            // SAFETY: a node the heap holds.
            head = unsafe { *(head as *const usize) };
        }
        n
    }

    #[inline(never)]
    fn collect_all(h: &Heap) {
        let mut regs = [0usize; 12];
        crate::collector::spill_callee_saved(&mut regs);
        h.collect_threads(regs.as_ptr() as usize, EXPLICIT_WAIT, true);
        std::hint::black_box(&regs);
    }

    fn cancelled(h: &Heap) -> u64 {
        lock(&h.world.inner).cancelled
    }

    fn registered(h: &Heap) -> usize {
        lock(&h.world.inner).list.len()
    }

    #[test]
    fn a_collection_reads_the_stack_of_a_thread_it_stopped() {
        let h = threaded(1 << 20);
        let ready = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = {
            let (ready, stop) = (ready.clone(), stop.clone());
            std::thread::spawn(move || {
                let head = build(h, 100);
                ready.store(true, Relaxed);
                while !stop.load(Relaxed) {
                    fill(make(h, 3000), 3000);
                    for _ in 0..40 {
                        fill(make(h, 32), 32);
                    }
                }
                assert_eq!(walk(std::hint::black_box(head)), 100);
                assert_eq!(held_nodes(h, head), 100, "the heap dropped a node");
                h.leave_thread();
            })
        };
        while !ready.load(Relaxed) {
            std::thread::yield_now();
        }
        for _ in 0..20 {
            collect_all(h);
        }
        stop.store(true, Relaxed);
        worker.join().unwrap();
        assert!(h.collections.load(Relaxed) >= 20);
        assert_eq!(cancelled(h), 0);
    }

    #[test]
    fn eight_threads_keep_their_lists_through_automatic_collections() {
        let h = threaded(256);
        let workers: Vec<_> = (0..8)
            .map(|t| {
                std::thread::spawn(move || {
                    let head = build(h, 3000);
                    for i in 0..20_000 {
                        fill(make(h, 200), 200);
                        if i % 50 == 0 {
                            fill(make(h, 4000), 4000);
                        }
                        if i % 2000 == 0 {
                            assert_eq!(walk(std::hint::black_box(head)), 3000, "thread {t}");
                        }
                        if t < 2 && i % 4000 == 0 {
                            collect_all(h);
                        }
                    }
                    assert_eq!(walk(std::hint::black_box(head)), 3000, "thread {t}");
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        assert!(h.collections.load(Relaxed) > 4);
        let live = h.live.load(Relaxed);
        assert!(live < 8 << 20, "{live} bytes stayed live");
    }

    #[test]
    fn a_thread_in_a_blocking_region_does_not_hold_up_a_collection() {
        let h = threaded(1 << 20);
        let entered = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let worker = {
            let entered = entered.clone();
            std::thread::spawn(move || {
                let head = build(h, 100);
                h.blocking(|| {
                    entered.store(true, Relaxed);
                    rx.recv().unwrap();
                });
                assert_eq!(held_nodes(h, std::hint::black_box(head)), 100);
                for _ in 0..2000 {
                    fill(make(h, 100), 100);
                }
                assert_eq!(walk(std::hint::black_box(head)), 100);
                assert_eq!(held_nodes(h, head), 100);
                h.leave_thread();
            })
        };
        while !entered.load(Relaxed) {
            std::thread::yield_now();
        }
        for _ in 0..10 {
            collect_all(h);
            for _ in 0..200 {
                fill(make(h, 500), 500);
            }
        }
        assert_eq!(cancelled(h), 0, "a native thread was waited for");
        tx.send(()).unwrap();
        worker.join().unwrap();
        collect_all(h);
    }

    #[test]
    fn a_loop_that_polls_the_safepoint_does_not_hold_up_a_collection() {
        let h = threaded(1 << 20);
        let ready = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let spinner = {
            let (ready, stop) = (ready.clone(), stop.clone());
            std::thread::spawn(move || {
                let head = build(h, 100);
                ready.store(true, Relaxed);
                while !stop.load(Relaxed) {
                    h.safepoint();
                    std::hint::spin_loop();
                }
                assert_eq!(held_nodes(h, std::hint::black_box(head)), 100);
                assert_eq!(walk(std::hint::black_box(head)), 100);
                h.leave_thread();
            })
        };
        while !ready.load(Relaxed) {
            std::thread::yield_now();
        }
        for _ in 0..10 {
            collect_all(h);
            for _ in 0..200 {
                fill(make(h, 500), 500);
            }
        }
        assert_eq!(cancelled(h), 0, "a polling thread was waited for");
        stop.store(true, Relaxed);
        spinner.join().unwrap();
        collect_all(h);
    }

    #[test]
    fn a_thread_that_never_answers_cancels_the_stop_and_backs_off() {
        let h = threaded(1 << 10);
        let ready = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let spinner = {
            let (ready, stop) = (ready.clone(), stop.clone());
            std::thread::spawn(move || {
                fill(make(h, 32), 32);
                ready.store(true, Relaxed);
                while !stop.load(Relaxed) {
                    std::hint::spin_loop();
                }
                h.leave_thread();
            })
        };
        while !ready.load(Relaxed) {
            std::thread::yield_now();
        }
        make(h, 64);
        let began = Instant::now();
        let mut regs = [0usize; 12];
        crate::collector::spill_callee_saved(&mut regs);
        h.collect_threads(regs.as_ptr() as usize, STOP_WAIT, false);
        std::hint::black_box(&regs);
        let took = began.elapsed();
        assert!(
            took >= STOP_WAIT && took < Duration::from_secs(2),
            "{took:?}"
        );
        assert_eq!(cancelled(h), 1);
        assert_eq!(h.collections.load(Relaxed), 0);
        assert!(!h.world.stop.load(Relaxed), "the stop was left on");
        assert!(h.threshold.load(Relaxed) >= h.floor << 1, "no back-off");
        stop.store(true, Relaxed);
        spinner.join().unwrap();
        collect_all(h);
        assert_eq!(h.collections.load(Relaxed), 1);
    }

    #[test]
    fn threads_that_start_and_exit_during_collections_do_not_stall_them() {
        let h = threaded(64);
        let done = Arc::new(AtomicBool::new(false));
        let collector = {
            let done = done.clone();
            std::thread::spawn(move || {
                while !done.load(Relaxed) {
                    collect_all(h);
                    fill(make(h, 300), 300);
                }
                h.leave_thread();
            })
        };
        let workers: Vec<_> = (0..200)
            .map(|_| {
                std::thread::spawn(move || {
                    let head = build(h, 50);
                    for _ in 0..500 {
                        fill(make(h, 100), 100);
                    }
                    assert_eq!(walk(std::hint::black_box(head)), 50);
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        done.store(true, Relaxed);
        collector.join().unwrap();
        assert_eq!(registered(h), 0, "an exited thread stayed registered");
        // A thread descheduled for the stop's wait costs one cancelled
        // collection, so only progress is asserted.
        assert!(h.collections.load(Relaxed) > 10);
    }
}
