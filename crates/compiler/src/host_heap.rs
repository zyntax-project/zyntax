//! A host's heap as the runtime's: the seam an embedder fills so that
//! what a program allocates comes from the host's allocator, and what the
//! runtime roots is reported to the host's collector.
//!
//! The host installs a [`HostHeap`] table once. From then on
//! [`crate::pool_alloc`]'s allocation entry points delegate to it, root
//! ranges are registered with it rather than with this crate's collector
//! (which stays off), and every thread the runtime starts announces itself
//! before it runs program code. Without a table nothing changes.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

/// The table layout this runtime reads. A host fills [`HostHeap::version`]
/// with it.
pub const HOST_HEAP_VERSION: u32 = 1;

/// What an allocation is for.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapKind {
    /// A program's allocation: the host's collector may reclaim it once
    /// nothing reaches it, and a free is a hint that it is dead.
    Collected = 0,
    /// Storage the runtime keeps and frees itself: the host keeps it
    /// until it is freed, and scans it for pointers.
    Held = 1,
}

/// Called by a span reader for each span `[lo, hi)` the host should scan.
pub type SpanVisit = unsafe extern "C" fn(cx: *mut c_void, lo: usize, hi: usize);

/// Hands `visit` the spans to scan for the root registered at `start`
/// with `word`, as they are when the host's collector runs. Called with
/// every mutator stopped: it neither allocates nor takes a lock a stopped
/// thread can hold, and `visit` reads each span before it returns.
pub type SpanReader =
    unsafe extern "C" fn(start: *const u8, word: usize, visit: SpanVisit, cx: *mut c_void);

/// The host's heap, as the runtime calls it. Every slot may be called
/// from any thread.
///
/// `alloc` returns zeroed memory aligned to 16 bytes, or null when it has
/// none; `realloc` keeps the contents and zeroes what it gains. `free` and
/// `realloc` are only passed blocks `owns` answers for, and `free` of a
/// block the host already reclaimed does nothing.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HostHeap {
    /// [`HOST_HEAP_VERSION`].
    pub version: u32,
    /// `size_of::<HostHeap>()` as the host built it; slots appended by a
    /// later version lie past a size an older host gives.
    pub size: u32,
    /// Passed back to every slot.
    pub context: *mut c_void,
    pub alloc: unsafe extern "C" fn(cx: *mut c_void, size: usize, kind: HeapKind) -> *mut u8,
    pub free: unsafe extern "C" fn(cx: *mut c_void, ptr: *mut u8),
    pub realloc: unsafe extern "C" fn(cx: *mut c_void, ptr: *mut u8, size: usize) -> *mut u8,
    /// Whether `ptr` is a block of the host's heap, interior addresses
    /// included.
    pub owns: unsafe extern "C" fn(cx: *mut c_void, ptr: *const u8) -> bool,
    /// Memory outside the heap to scan word by word for as long as it is
    /// registered.
    pub add_root_range: unsafe extern "C" fn(cx: *mut c_void, start: *const u8, len: usize),
    /// A root whose spans `reader` gives at each collection, passed
    /// `start` and `word` as registered.
    pub add_root_spans:
        unsafe extern "C" fn(cx: *mut c_void, start: *const u8, word: usize, reader: SpanReader),
    /// Forget the root registered at `start`, either kind.
    pub remove_root_range: unsafe extern "C" fn(cx: *mut c_void, start: *const u8),
    /// The calling thread runs program code from here until
    /// `thread_leave`; its stack ends at `stack_top`.
    pub thread_enter: unsafe extern "C" fn(cx: *mut c_void, stack_top: *const u8),
    pub thread_leave: unsafe extern "C" fn(cx: *mut c_void),
}

struct Installed(HostHeap);

// SAFETY: the host promises every slot may be called from any thread
// with its context.
unsafe impl Send for Installed {}
unsafe impl Sync for Installed {}

static HOST: OnceLock<Installed> = OnceLock::new();

/// Why [`install`] refused a table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallError {
    /// A table already went in; the first stays.
    AlreadyInstalled,
    /// The table is of a layout this runtime does not read.
    Version { found: u32, expected: u32 },
    /// The table is shorter than this version's layout.
    TooShort { found: u32, expected: u32 },
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstallError::AlreadyInstalled => write!(f, "a host heap is already installed"),
            InstallError::Version { found, expected } => write!(
                f,
                "the host heap table is version {found}; this runtime reads version {expected}"
            ),
            InstallError::TooShort { found, expected } => write!(
                f,
                "the host heap table is {found} bytes; version {HOST_HEAP_VERSION} needs {expected}"
            ),
        }
    }
}

impl std::error::Error for InstallError {}

/// Make `table` the runtime's heap. Blocks allocated before keep their
/// own allocator; everything after comes from the host.
///
/// # Safety
/// Every slot must behave as [`HostHeap`] describes for the rest of the
/// process, and `table.context` must stay valid that long.
pub unsafe fn install(table: &HostHeap) -> Result<(), InstallError> {
    if table.version != HOST_HEAP_VERSION {
        return Err(InstallError::Version {
            found: table.version,
            expected: HOST_HEAP_VERSION,
        });
    }
    let expected = std::mem::size_of::<HostHeap>() as u32;
    if table.size < expected {
        return Err(InstallError::TooShort {
            found: table.size,
            expected,
        });
    }
    if HOST.set(Installed(*table)).is_err() {
        return Err(InstallError::AlreadyInstalled);
    }
    // This crate's collector stays off from here, and what it rooted is
    // the host's to root now.
    crate::collector::disable();
    root_holds_again();
    Ok(())
}

#[inline]
pub(crate) fn installed() -> Option<&'static HostHeap> {
    HOST.get().map(|h| &h.0)
}

/// Whether a host heap is installed.
#[inline]
pub fn is_installed() -> bool {
    HOST.get().is_some()
}

/// Storage the runtime keeps and frees itself, which holds pointers into
/// the program's heap: from the host as [`HeapKind::Held`], so its
/// collector scans it, or from the pool without one.
///
/// # Safety
/// As [`crate::pool_alloc::zyntax_alloc`]; release it with
/// [`crate::pool_alloc::zyntax_free`].
pub unsafe fn alloc_held(size: usize) -> *mut u8 {
    match installed() {
        Some(h) => (h.alloc)(h.context, size.max(1), HeapKind::Held),
        None => crate::pool_alloc::zyntax_alloc(size),
    }
}

/// A span root as registered with the host: the reader and length the
/// runtime registered it with. Its address is the word the host passes
/// back, so a reader finds it without a lookup a stopped thread could
/// hold a lock on.
struct SpanRoot {
    reader: crate::collector::RootSpans,
    len: usize,
}

/// Each span root's [`SpanRoot`], by start, for its removal. Readers never
/// take this lock.
static SPAN_ROOTS: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);

unsafe extern "C" fn read_spans(start: *const u8, word: usize, visit: SpanVisit, cx: *mut c_void) {
    // SAFETY: `word` is a `SpanRoot` that stays until its root is removed.
    let root = unsafe { &*(word as *const SpanRoot) };
    (root.reader)(start, root.len, &mut |lo, hi| unsafe { visit(cx, lo, hi) });
}

pub(crate) fn add_root_range(h: &HostHeap, start: *const u8, len: usize) {
    unsafe { (h.add_root_range)(h.context, start, len) }
}

pub(crate) fn add_root_spans(
    h: &HostHeap,
    start: *const u8,
    len: usize,
    reader: crate::collector::RootSpans,
) {
    let root = Box::into_raw(Box::new(SpanRoot { reader, len })) as usize;
    let replaced = SPAN_ROOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(start as usize, root);
    unsafe { (h.add_root_spans)(h.context, start, root, read_spans) }
    if let Some(old) = replaced {
        // SAFETY: made above for the root this registration replaced.
        drop(unsafe { Box::from_raw(old as *mut SpanRoot) });
    }
}

pub(crate) fn remove_root(h: &HostHeap, start: *const u8) {
    unsafe { (h.remove_root_range)(h.context, start) }
    let root = SPAN_ROOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|m| m.remove(&(start as usize)));
    if let Some(root) = root {
        // SAFETY: the host no longer calls the reader with it.
        drop(unsafe { Box::from_raw(root as *mut SpanRoot) });
    }
}

/// A thread the runtime started, announced to the host's heap for as
/// long as the guard lives. Made first thing on the thread, so its
/// stack top is above anything the thread later holds.
pub struct ThreadGuard(bool);

impl ThreadGuard {
    #[inline(never)]
    pub fn enter() -> ThreadGuard {
        let Some(h) = installed() else {
            return ThreadGuard(false);
        };
        let top = 0u8;
        let top = std::hint::black_box(&top as *const u8);
        unsafe { (h.thread_enter)(h.context, top) };
        ThreadGuard(true)
    }
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        if self.0
            && let Some(h) = installed()
        {
            unsafe { (h.thread_leave)(h.context) };
        }
    }
}

/// Addresses the runtime names from its own tables, which no collector
/// can read: each kept alive while held.
struct Holds {
    /// One entry per hold, so a block held twice stays until both go.
    addresses: Vec<usize>,
    registered: bool,
}

static HOLDS: Mutex<Holds> = Mutex::new(Holds {
    addresses: Vec::new(),
    registered: false,
});

/// The root [`HOLDS`] is registered under: its storage moves as it
/// grows, so the root is a fixed address read through [`hold_spans`].
static HOLDS_ROOT: u8 = 0;

fn hold_spans(_start: *const u8, _len: usize, visit: &mut dyn FnMut(usize, usize)) {
    // No thread allocates while it holds the lock, so none is stopped
    // holding it.
    let holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
    let lo = holds.addresses.as_ptr() as usize;
    visit(
        lo,
        lo + holds.addresses.len() * std::mem::size_of::<usize>(),
    );
}

/// Keep `address`, a program object the runtime is about to name from a
/// table of its own, alive until [`unhold`].
pub fn hold(address: *const u8) {
    // With no collector nothing reads the holds, and none is turned on
    // once a program has run.
    if address.is_null() || !crate::collector::roots_wanted() {
        return;
    }
    // The set's storage grows outside its lock: a collection reads it
    // with every mutator stopped at an allocation, so none may stop there
    // holding the lock.
    let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
    while holds.addresses.len() == holds.addresses.capacity() {
        let want = (holds.addresses.capacity() * 2).max(64);
        drop(holds);
        let mut grown: Vec<usize> = Vec::with_capacity(want);
        holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
        if grown.capacity() > holds.addresses.capacity() {
            grown.extend_from_slice(&holds.addresses);
            std::mem::swap(&mut holds.addresses, &mut grown);
        }
        drop(holds);
        drop(grown);
        holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
    }
    holds.addresses.push(address as usize);
    if !holds.registered && crate::collector::roots_wanted() {
        holds.registered = true;
        drop(holds);
        crate::collector::add_root_range_with(&HOLDS_ROOT as *const u8, 1, hold_spans);
    }
}

/// Register the holds as a root again once a collector forgot its roots
/// on being turned on.
pub(crate) fn root_holds_again() {
    let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
    holds.registered = !holds.addresses.is_empty();
    if holds.registered {
        drop(holds);
        crate::collector::add_root_range_with(&HOLDS_ROOT as *const u8, 1, hold_spans);
    }
}

/// Give up one [`hold`] of `address`.
pub fn unhold(address: *const u8) {
    if address.is_null() {
        return;
    }
    let mut holds = HOLDS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = holds.addresses.iter().rposition(|a| *a == address as usize) {
        holds.addresses.swap_remove(i);
    }
}

/// How many holds are outstanding (diagnostics and tests).
pub fn hold_count() -> usize {
    HOLDS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .addresses
        .len()
}

/// One [`hold`] of an address, given up when the guard drops: for an
/// entry of a runtime table that names a program object, kept with it.
#[derive(Debug)]
pub struct Hold(usize);

impl Hold {
    pub fn new(address: *const u8) -> Hold {
        hold(address);
        Hold(address as usize)
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        unhold(self.0 as *const u8);
    }
}

/// A table over the system allocator, which collects nothing: what a
/// host's table must do, in its simplest form, and a way to run any
/// program through the seam.
pub mod reference {
    use super::{HeapKind, HostHeap, SpanReader};
    use std::collections::BTreeMap;
    use std::ffi::c_void;
    use std::sync::Mutex;

    static BLOCKS: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

    fn layout(size: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(size.max(16), 16).expect("a block layout")
    }

    unsafe extern "C" fn alloc(_cx: *mut c_void, size: usize, _kind: HeapKind) -> *mut u8 {
        let p = unsafe { std::alloc::alloc_zeroed(layout(size)) };
        if !p.is_null() {
            BLOCKS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(p as usize, size);
        }
        p
    }

    unsafe extern "C" fn free(_cx: *mut c_void, ptr: *mut u8) {
        let size = BLOCKS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(ptr as usize));
        if let Some(size) = size {
            unsafe { std::alloc::dealloc(ptr, layout(size)) };
        }
    }

    unsafe extern "C" fn realloc(cx: *mut c_void, ptr: *mut u8, size: usize) -> *mut u8 {
        let old = BLOCKS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(ptr as usize))
            .copied();
        let Some(old) = old else {
            return std::ptr::null_mut();
        };
        let fresh = unsafe { alloc(cx, size, HeapKind::Collected) };
        if !fresh.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, fresh, old.min(size));
                free(cx, ptr);
            }
        }
        fresh
    }

    unsafe extern "C" fn owns(_cx: *mut c_void, ptr: *const u8) -> bool {
        let a = ptr as usize;
        BLOCKS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .range(..=a)
            .next_back()
            .is_some_and(|(start, size)| a < start + (*size).max(1))
    }

    unsafe extern "C" fn add_root_range(_cx: *mut c_void, _start: *const u8, _len: usize) {}
    unsafe extern "C" fn add_root_spans(
        _cx: *mut c_void,
        _start: *const u8,
        _len: usize,
        _reader: SpanReader,
    ) {
    }
    unsafe extern "C" fn remove_root_range(_cx: *mut c_void, _start: *const u8) {}
    unsafe extern "C" fn thread_enter(_cx: *mut c_void, _top: *const u8) {}
    unsafe extern "C" fn thread_leave(_cx: *mut c_void) {}

    /// The table.
    pub fn table() -> HostHeap {
        HostHeap {
            version: super::HOST_HEAP_VERSION,
            size: std::mem::size_of::<HostHeap>() as u32,
            context: std::ptr::null_mut(),
            alloc,
            free,
            realloc,
            owns,
            add_root_range,
            add_root_spans,
            remove_root_range,
            thread_enter,
            thread_leave,
        }
    }
}

/// `ZYNTAX_HOST_HEAP=reference` runs the program on [`reference`]'s
/// table: everything through the seam, nothing collected. Safe; slower.
pub fn install_from_env() {
    if std::env::var("ZYNTAX_HOST_HEAP").as_deref() == Ok("reference") {
        let _ = unsafe { install(&reference::table()) };
    }
}
