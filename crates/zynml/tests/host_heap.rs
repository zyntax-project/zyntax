//! A program's heap is its host's once the host installs a heap table:
//! every allocation comes from the host, every release of one goes back,
//! and the runtime's roots and threads are reported to it.
//!
//! The table is process-wide, so this is a test binary of its own.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use zynml::{Grammar2, ZYNML_GRAMMAR};
use zyntax_compiler::host_heap::{self, HeapKind, HostHeap, SpanReader, SpanVisit};
use zyntax_embed::{TieredConfig, TieredRuntime};

/// The fake host's blocks, by address, with their sizes.
static BLOCKS: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static HELD: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static ROOTS: Mutex<BTreeMap<usize, Option<(SpanReader, usize)>>> = Mutex::new(BTreeMap::new());
static ROOTS_ADDED: AtomicUsize = AtomicUsize::new(0);
static THREADS: AtomicUsize = AtomicUsize::new(0);
static THREADS_LEFT: AtomicUsize = AtomicUsize::new(0);

fn layout(size: usize) -> std::alloc::Layout {
    std::alloc::Layout::from_size_align(size.max(16), 16).unwrap()
}

unsafe extern "C" fn alloc(_cx: *mut c_void, size: usize, kind: HeapKind) -> *mut u8 {
    let p = unsafe { std::alloc::alloc_zeroed(layout(size)) };
    BLOCKS.lock().unwrap().insert(p as usize, size);
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    if kind == HeapKind::Held {
        HELD.fetch_add(1, Ordering::Relaxed);
    }
    p
}

unsafe extern "C" fn free(_cx: *mut c_void, ptr: *mut u8) {
    let size = BLOCKS.lock().unwrap().remove(&(ptr as usize));
    if let Some(size) = size {
        FREES.fetch_add(1, Ordering::Relaxed);
        unsafe { std::alloc::dealloc(ptr, layout(size)) };
    }
}

unsafe extern "C" fn realloc(cx: *mut c_void, ptr: *mut u8, size: usize) -> *mut u8 {
    let old = BLOCKS.lock().unwrap().get(&(ptr as usize)).copied();
    let Some(old) = old else {
        return std::ptr::null_mut();
    };
    let fresh = unsafe { alloc(cx, size, HeapKind::Collected) };
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, fresh, old.min(size));
        free(cx, ptr);
    }
    fresh
}

unsafe extern "C" fn owns(_cx: *mut c_void, ptr: *const u8) -> bool {
    let a = ptr as usize;
    BLOCKS
        .lock()
        .unwrap()
        .range(..=a)
        .next_back()
        .is_some_and(|(start, size)| a < start + size.max(&1))
}

unsafe extern "C" fn add_root_range(_cx: *mut c_void, start: *const u8, _len: usize) {
    ROOTS.lock().unwrap().insert(start as usize, None);
    ROOTS_ADDED.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn add_root_spans(
    _cx: *mut c_void,
    start: *const u8,
    word: usize,
    reader: SpanReader,
) {
    ROOTS
        .lock()
        .unwrap()
        .insert(start as usize, Some((reader, word)));
    ROOTS_ADDED.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn remove_root_range(_cx: *mut c_void, start: *const u8) {
    ROOTS.lock().unwrap().remove(&(start as usize));
}

unsafe extern "C" fn thread_enter(_cx: *mut c_void, _top: *const u8) {
    THREADS.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn thread_leave(_cx: *mut c_void) {
    THREADS_LEFT.fetch_add(1, Ordering::Relaxed);
}

fn table() -> HostHeap {
    HostHeap {
        version: host_heap::HOST_HEAP_VERSION,
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

/// Install the fake host's table, once for the process.
fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe { host_heap::install(&table()) }.expect("installs"));
}

unsafe extern "C" fn count_span(cx: *mut c_void, lo: usize, hi: usize) {
    let spans = unsafe { &mut *(cx as *mut Vec<(usize, usize)>) };
    spans.push((lo, hi));
}

/// Read every root with a reader as a collection would, returning the
/// spans it named.
fn read_roots() -> Vec<(usize, usize)> {
    let roots: Vec<(usize, Option<(SpanReader, usize)>)> = ROOTS
        .lock()
        .unwrap()
        .iter()
        .map(|(s, r)| (*s, *r))
        .collect();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for (start, reader) in roots {
        if let Some((reader, word)) = reader {
            let visit: SpanVisit = count_span;
            unsafe {
                reader(
                    start as *const u8,
                    word,
                    visit,
                    &mut spans as *mut Vec<(usize, usize)> as *mut c_void,
                )
            };
        }
    }
    spans
}

const SRC: &str = r#"
struct Pair {
    a: i64,
    b: i64
}

def build(n: i64): i64 {
    let mut xs: List<i64> = []
    let mut i = 0
    while i < n {
        xs.push(i)
        i = i + 1
    }
    let mut total = 0
    for j in 0..n {
        total = total + xs[j]
    }
    return total
}

def text(n: i64): i64 {
    let mut s = ""
    let mut i = 0
    while i < n {
        s = s + "ab"
        i = i + 1
    }
    if s == "" {
        return 0
    }
    return i * 2
}

def pairs(n: i64): i64 {
    let mut total = 0
    let mut i = 0
    while i < n {
        let p = Pair { a: i, b: i * 2 }
        total = total + p.a + p.b
        i = i + 1
    }
    return total
}
"#;

#[test]
fn a_program_allocates_from_the_host_heap() {
    // Refused before it is installed: a table of another layout.
    let mut wrong = table();
    wrong.version = host_heap::HOST_HEAP_VERSION + 1;
    assert!(unsafe { host_heap::install(&wrong) }.is_err());
    let mut short = table();
    short.size -= 8;
    assert!(unsafe { host_heap::install(&short) }.is_err());

    install();
    assert!(
        unsafe { host_heap::install(&table()) }.is_err(),
        "once only"
    );

    let mut rt = TieredRuntime::new(TieredConfig::development()).expect("runtime should start");
    rt.register_static_plugins([zrtl_io::static_plugin()])
        .expect("plugins");
    rt.set_collector(zyntax_embed::Collector::MarkSweep);
    assert!(
        !zyntax_compiler::collector::is_enabled(),
        "the runtime's own collector stays off under a host heap"
    );
    let grammar = Grammar2::from_source(ZYNML_GRAMMAR).expect("grammar");
    let program = grammar
        .parse_with_filename(SRC, "<host_heap>")
        .expect("parse");
    rt.compile_typed_program(program).expect("compile");

    let before = ALLOCS.load(Ordering::Relaxed);
    let call = |f: &str, n: i64| {
        rt.call::<i64>(f, &[zyntax_embed::ZyntaxValue::Int(n)])
            .map_err(|e| e.to_string())
    };
    assert_eq!(call("build", 1000), Ok(499_500));
    assert_eq!(call("text", 300), Ok(600));
    assert_eq!(call("pairs", 100), Ok(14_850));
    let allocated = ALLOCS.load(Ordering::Relaxed) - before;
    assert!(allocated > 0, "the program allocated nothing from the host");
    assert!(
        FREES.load(Ordering::Relaxed) > 0,
        "nothing the program released went back to the host"
    );
    assert!(
        ROOTS_ADDED.load(Ordering::Relaxed) > 0,
        "the runtime reported no roots"
    );
    // Every root a reader describes reads cleanly, as a collection on
    // another thread would read it.
    for (lo, hi) in read_roots() {
        assert!(lo <= hi, "a span runs backwards: {lo:#x}..{hi:#x}");
    }
    assert!(
        THREADS_LEFT.load(Ordering::Relaxed) <= THREADS.load(Ordering::Relaxed),
        "a thread left that never entered"
    );
    std::mem::forget(rt);
}

/// What the runtime keeps from its own tables stays rooted while held,
/// and the root reads through to it.
#[test]
fn a_hold_is_read_through_its_root() {
    install();
    let block = unsafe { zyntax_compiler::pool_alloc::zyntax_alloc(32) };
    assert!(
        unsafe { owns(std::ptr::null_mut(), block) },
        "from the host"
    );
    let held = host_heap::Hold::new(block);
    let words: Vec<usize> = read_roots()
        .into_iter()
        .flat_map(|(lo, hi)| (lo..hi).step_by(8).map(|a| unsafe { *(a as *const usize) }))
        .collect();
    assert!(words.contains(&(block as usize)), "the hold is not read");
    drop(held);
    let words: Vec<usize> = read_roots()
        .into_iter()
        .flat_map(|(lo, hi)| (lo..hi).step_by(8).map(|a| unsafe { *(a as *const usize) }))
        .collect();
    assert!(
        !words.contains(&(block as usize)),
        "a released hold is still read"
    );
    unsafe { zyntax_compiler::pool_alloc::zyntax_free(block) };
    assert!(
        !unsafe { owns(std::ptr::null_mut(), block) },
        "freed to the host"
    );
}
