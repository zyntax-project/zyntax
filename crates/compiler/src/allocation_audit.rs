//! Rust allocator requests grouped by the innermost compiler phase.
//! Enabled only by `allocation-audit`; excludes direct libc, GC slab,
//! and native-library allocations. Profiled binaries are diagnostic tools.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    Other,
    Parse,
    Runtime,
    Register,
    Lower,
    Execute,
    Optimize,
    Scratch,
    HirClone,
    Decode,
    Cranelift,
    Llvm,
    Osr,
    InterpBody,
}

const NAMES: [&str; 14] = [
    "other",
    "parse",
    "runtime",
    "register",
    "lower",
    "execute",
    "optimize",
    "scratch",
    "hir_clone",
    "decode",
    "cranelift",
    "llvm",
    "osr",
    "interp_body",
];

thread_local! {
    static PHASE: Cell<Phase> = const { Cell::new(Phase::Other) };
}

struct Counters {
    entries: AtomicU64,
    allocations: AtomicU64,
    reallocations: AtomicU64,
    requested_bytes: AtomicU64,
}

static COUNTS: [Counters; NAMES.len()] = [const {
    Counters {
        entries: AtomicU64::new(0),
        allocations: AtomicU64::new(0),
        reallocations: AtomicU64::new(0),
        requested_bytes: AtomicU64::new(0),
    }
}; NAMES.len()];
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

/// Restore the previous phase when a nested operation returns or unwinds.
pub struct Scope(Phase);

/// Set the foreground phase between frontend operations.
pub fn set_phase(phase: Phase) {
    COUNTS[phase as usize].entries.fetch_add(1, Relaxed);
    PHASE.with(|current| current.set(phase));
}

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub entries: u64,
    pub allocations: u64,
    pub reallocations: u64,
    pub requested_bytes: u64,
}

pub fn for_phase(phase: Phase) -> Stats {
    let c = &COUNTS[phase as usize];
    Stats {
        entries: c.entries.load(Relaxed),
        allocations: c.allocations.load(Relaxed),
        reallocations: c.reallocations.load(Relaxed),
        requested_bytes: c.requested_bytes.load(Relaxed),
    }
}

impl Scope {
    pub fn enter(phase: Phase) -> Self {
        COUNTS[phase as usize].entries.fetch_add(1, Relaxed);
        Self(PHASE.with(|current| current.replace(phase)))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        let _ = PHASE.try_with(|current| current.set(self.0));
    }
}

fn current() -> &'static Counters {
    let phase = PHASE.try_with(Cell::get).unwrap_or(Phase::Other);
    &COUNTS[phase as usize]
}

fn reserve(bytes: usize) {
    let live = LIVE.fetch_add(bytes as u64, Relaxed) + bytes as u64;
    PEAK.fetch_max(live, Relaxed);
}

/// Install as a binary's global allocator to record Rust heap requests.
pub struct CountingAllocator;

// SAFETY: allocation, zeroing, resizing and deallocation retain System's
// contracts. Accounting uses allocation-free thread-local cells and atomics.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            current().allocations.fetch_add(1, Relaxed);
            current()
                .requested_bytes
                .fetch_add(layout.size() as u64, Relaxed);
            reserve(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            current().allocations.fetch_add(1, Relaxed);
            current()
                .requested_bytes
                .fetch_add(layout.size() as u64, Relaxed);
            reserve(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let ptr = unsafe { System.realloc(ptr, layout, size) };
        if !ptr.is_null() {
            current().reallocations.fetch_add(1, Relaxed);
            current().requested_bytes.fetch_add(size as u64, Relaxed);
            if size >= layout.size() {
                reserve(size - layout.size());
            } else {
                LIVE.fetch_sub((layout.size() - size) as u64, Relaxed);
            }
        }
        ptr
    }
}

/// Emit one cumulative snapshot. Requested bytes include the full size of
/// each successful realloc; peak live bytes cover Rust storage only, not RSS.
pub fn report() {
    use std::io::Write;
    let peak = PEAK.load(Relaxed);
    let live = LIVE.load(Relaxed);
    let counts: [_; NAMES.len()] = std::array::from_fn(|i| {
        let c = &COUNTS[i];
        (
            c.entries.load(Relaxed),
            c.allocations.load(Relaxed),
            c.reallocations.load(Relaxed),
            c.requested_bytes.load(Relaxed),
        )
    });
    let mut stderr = std::io::stderr().lock();
    let _ = write!(
        stderr,
        "[ALLOC] {{\"rust_peak_live_bytes\":{peak},\"rust_live_bytes\":{live},\"phases\":{{"
    );
    for (i, (entries, allocations, reallocations, requested_bytes)) in
        counts.into_iter().enumerate()
    {
        if i != 0 {
            let _ = write!(stderr, ",");
        }
        let _ = write!(
            stderr,
            "\"{}\":{{\"entries\":{entries},\"allocations\":{allocations},\"reallocations\":{reallocations},\"requested_bytes\":{requested_bytes}}}",
            NAMES[i]
        );
    }
    let _ = writeln!(stderr, "}}}}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scopes_restore_the_thread_phase_after_unwind() {
        let _outer = Scope::enter(Phase::Optimize);
        let _ = std::panic::catch_unwind(|| {
            let _inner = Scope::enter(Phase::HirClone);
            assert!(matches!(PHASE.with(Cell::get), Phase::HirClone));
            panic!("unwind");
        });
        assert!(matches!(PHASE.with(Cell::get), Phase::Optimize));
        let phase = std::thread::spawn(|| PHASE.with(Cell::get)).join().unwrap();
        assert!(matches!(phase, Phase::Other));
    }

    #[test]
    fn records_zeroing_resizing_and_release() {
        let _scope = Scope::enter(Phase::Other);
        let before = for_phase(Phase::Other);
        let live = LIVE.load(Relaxed);
        let small = Layout::from_size_align(16, 16).unwrap();
        let large = Layout::from_size_align(32, 16).unwrap();
        // SAFETY: every successful allocation is used and released with
        // its current layout; realloc's alignment remains unchanged.
        unsafe {
            let p = CountingAllocator.alloc_zeroed(small);
            assert!(!p.is_null());
            assert!(std::slice::from_raw_parts(p, 16).iter().all(|v| *v == 0));
            let p = CountingAllocator.realloc(p, small, 32);
            assert!(!p.is_null());
            assert_eq!(LIVE.load(Relaxed), live + 32);
            let p = CountingAllocator.realloc(p, large, 16);
            assert!(!p.is_null());
            assert_eq!(LIVE.load(Relaxed), live + 16);
            CountingAllocator.dealloc(p, small);
        }
        let after = for_phase(Phase::Other);
        assert_eq!(after.allocations - before.allocations, 1);
        assert_eq!(after.reallocations - before.reallocations, 2);
        assert_eq!(after.requested_bytes - before.requested_bytes, 64);
        assert_eq!(LIVE.load(Relaxed), live);
    }
}
