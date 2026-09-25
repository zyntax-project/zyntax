//! Lua's garbage-collection semantics over the compiler's collector:
//! weak tables, `__gc` finalizers, `collectgarbage` and `warn`.
//!
//! `setmetatable` reports a table whose metatable holds `__mode` or
//! `__gc` at that moment. A weak table's entry storage is held from
//! the marker, and this module's trace marks what it keeps strongly,
//! as the reference's traversals do: nothing of an entry whose value
//! is nil; of any other, every key and value that is not a
//! collectable object, a strong key, a weak key's value once its key
//! is reached (an ephemeron), a weak value once it is reached
//! elsewhere. When tracing settles, weak values that died are cleared,
//! then every unreached object marked for finalization is resurrected
//! and queued; once tracing settles again, collectable keys nothing
//! reached become tombstones. So a strong key whose value died stays
//! for that collection and goes in the next, as in the reference. The
//! finalizers run after the collection, outside it, in the reverse of
//! the order their objects were marked.
//!
//! A dead key cannot leave its dict while the world is stopped, since
//! reindexing hashes through the library. Its box is rewritten in
//! place as a tombstone: an instance of [`library::DEAD_KEY_KIND`]
//! naming an address outside the heap, which no lookup matches, with
//! a nil value, which no traversal shows. Tombstones keep every
//! entry's position, so a traversal a collection interrupts goes on
//! where it was. A table the collection changed is dirty: the library
//! compacts it, dropping its tombstones and nil-valued entries, when a
//! new key is next stored in it, which Lua already makes the end of
//! any traversal of that table.
//!
//! A shaped table is never weak: its slots are words of the table
//! itself, which the marker reads, so the types keep every table that
//! may get a weak metatable out of slots.
//!
//! After `collectgarbage` the finalizers run before it returns; after
//! a collection the heap's budget started, they run inside the
//! allocation that started it, once the collection is over.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Mutex;

use zyntax_compiler::collector::{self, Marking, WeakHooks};

use crate::library;

/// `__mode` holds `k`.
const WEAK_KEYS: i64 = 1;
/// `__mode` holds `v`.
const WEAK_VALUES: i64 = 2;
/// The metatable held `__gc` when it was set.
const FINALIZE: i64 = 4;

/// A table as the library lays it out: the header fields in the order
/// of `library::TABLE_FIELDS`.
#[repr(C)]
struct TableHeader {
    arr: *mut BoxHeader,
    hash: *mut BoxHeader,
    meta: usize,
    high: i64,
    shape: i64,
    present: i64,
}

/// The fields of a dynamic box this module reads and rewrites.
#[repr(C)]
struct BoxHeader {
    tag: u32,
    size: u32,
    data: usize,
}

/// A list as the compiler lays it out.
#[repr(C)]
struct ListHeader {
    data: *mut *mut BoxHeader,
    len: i64,
    capacity: i64,
}

/// A weak table's mode, and whether it holds tombstones to compact.
#[derive(Clone, Copy)]
struct Weak {
    mode: i64,
    dirty: bool,
}

/// Where a collection is between its settle steps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Weak values not yet cleared, nothing resurrected.
    Marking,
    /// Resurrection done; dead keys still to turn into tombstones.
    Resurrected,
    /// Nothing left for this collection.
    Done,
}

struct State {
    weak: BTreeMap<usize, Weak>,
    /// How many tables of `weak` are dirty.
    dirty: usize,
    /// The count of dirty tables the program last read.
    published: usize,
    /// Objects marked for finalization, in the order they were marked.
    finalizable: Vec<usize>,
    marked: BTreeSet<usize>,
    /// Resurrected objects whose finalizers are due, in calling order.
    pending: VecDeque<usize>,
    phase: Phase,
    /// The library's finalizer runner, `extern "C" fn()`.
    runner: usize,
    /// A finalizer is running: collections are refused meanwhile.
    finalizing: bool,
    /// `collectgarbage("stop")` is in force.
    stopped: bool,
    generational: bool,
    /// `setpause` and `setstepmul`, stored in quarters as the
    /// reference stores them.
    pause: i64,
    stepmul: i64,
    warn: Warn,
}

/// The reference's warning function states: off, on at the start of a
/// message, or in the middle of one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Warn {
    Off,
    On,
    Continued,
}

static STATE: Mutex<State> = Mutex::new(State {
    weak: BTreeMap::new(),
    dirty: 0,
    published: 0,
    finalizable: Vec::new(),
    marked: BTreeSet::new(),
    pending: VecDeque::new(),
    phase: Phase::Done,
    runner: 0,
    finalizing: false,
    stopped: false,
    generational: true,
    pause: 200 / 4,
    stepmul: 100 / 4,
    warn: Warn::Off,
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn table_tag() -> u32 {
    library::table_tag() as u32
}

fn dead_key_tag() -> u32 {
    zyntax_builtins::instance_tag(library::DEAD_KEY_KIND) as u32
}

/// The object a box names when it is one the collector may take from
/// a weak table: a table, a function or a coroutine. Strings, numbers
/// and booleans are values and never leave one.
///
/// # Safety
/// `b` is null or a live box.
unsafe fn referent(b: *const BoxHeader) -> Option<usize> {
    if b.is_null() {
        return None;
    }
    let tag = unsafe { (*b).tag };
    let collectable = tag == table_tag()
        || tag == zyntax_builtins::FUNC_TAG as u32
        || tag == library::thread_tag() as u32;
    collectable.then(|| unsafe { (*b).data })
}

/// The list a box holds, or none.
///
/// # Safety
/// `b` is null or a live list box.
unsafe fn list_of(b: *const BoxHeader) -> Option<&'static mut ListHeader> {
    if b.is_null() {
        return None;
    }
    let list = unsafe { (*b).data } as *mut ListHeader;
    (!list.is_null()).then(|| unsafe { &mut *list })
}

/// A list's elements.
///
/// # Safety
/// The list is live and nothing else touches it meanwhile.
unsafe fn elements(list: &mut ListHeader) -> &'static mut [*mut BoxHeader] {
    if list.data.is_null() || list.len <= 0 {
        return &mut [];
    }
    unsafe { std::slice::from_raw_parts_mut(list.data, list.len as usize) }
}

// ─── the collector's hooks ──────────────────────────────────────────

fn hold(f: &mut dyn FnMut(usize)) {
    let mut s = state();
    s.phase = Phase::Marking;
    for (&t, w) in &s.weak {
        // SAFETY: a registered table is live: one the last collection
        // did not reach left the registry then.
        let t = unsafe { &*(t as *const TableHeader) };
        if let Some(list) = unsafe { list_of(t.hash) }
            && !list.data.is_null()
        {
            f(list.data as usize);
        }
        if w.mode & WEAK_VALUES != 0
            && let Some(list) = unsafe { list_of(t.arr) }
            && !list.data.is_null()
        {
            f(list.data as usize);
        }
    }
}

fn trace(m: &mut dyn Marking) {
    let s = state();
    for &o in &s.pending {
        m.mark(o);
    }
    for (&t, w) in &s.weak {
        if !m.is_marked(t) {
            continue;
        }
        // SAFETY: reached, so live; the world is stopped.
        let t = unsafe { &*(t as *const TableHeader) };
        let alive = |m: &dyn Marking, b: *mut BoxHeader, weak: bool| {
            // SAFETY: an element of a live table is null or a live box.
            match unsafe { referent(b) } {
                Some(r) if weak => m.is_marked(r),
                _ => true,
            }
        };
        if let Some(list) = unsafe { list_of(t.hash) } {
            let items = unsafe { elements(list) };
            if let Some((&index, pairs)) = items.split_first() {
                m.mark(index as usize);
                let (whole, rest) = pairs.as_chunks::<2>();
                for &[k, v] in whole {
                    if v.is_null() || !alive(m, k, w.mode & WEAK_KEYS != 0) {
                        continue;
                    }
                    m.mark(k as usize);
                    if alive(m, v, w.mode & WEAK_VALUES != 0) {
                        m.mark(v as usize);
                    }
                }
                // A key pushed without its value yet.
                for &k in rest {
                    m.mark(k as usize);
                }
            }
        }
        if w.mode & WEAK_VALUES != 0
            && let Some(list) = unsafe { list_of(t.arr) }
        {
            for &v in unsafe { elements(list) }.iter() {
                if alive(m, v, true) {
                    m.mark(v as usize);
                }
            }
        }
    }
}

fn settle(m: &mut dyn Marking) {
    let mut s = state();
    match s.phase {
        Phase::Marking => {
            s.phase = Phase::Resurrected;
            clear_dead_values(&mut s.weak, m);
            // Every unreached finalizable object comes back for one
            // cycle, its finalizer due; the last marked runs first.
            let mut due = Vec::new();
            let State {
                finalizable,
                marked,
                ..
            } = &mut *s;
            finalizable.retain(|&o| {
                if m.is_marked(o) {
                    return true;
                }
                marked.remove(&o);
                due.push(o);
                false
            });
            for &o in due.iter().rev() {
                s.pending.push_back(o);
            }
            for &o in &due {
                m.mark(o);
            }
            // Resurrecting nothing reached nothing new, so tracing is
            // settled already and the keys can go now; otherwise they
            // go on the call after tracing settles again.
            if due.is_empty() {
                clear_dead_keys(&mut s, m);
            }
        }
        Phase::Resurrected => clear_dead_keys(&mut s, m),
        Phase::Done => {}
    }
}

/// The last step: dead keys become tombstones, and weak values of
/// tables reached only through what was resurrected are cleared.
fn clear_dead_keys(s: &mut State, m: &mut dyn Marking) {
    s.phase = Phase::Done;
    let keys = tombstone_dead_keys(&mut s.weak, m);
    clear_dead_values(&mut s.weak, m);
    s.dirty = s.weak.values().filter(|w| w.dirty).count();
    for k in keys {
        m.mark(k);
    }
}

/// Clear every weak value of a reached table whose object is not.
fn clear_dead_values(weak: &mut BTreeMap<usize, Weak>, m: &dyn Marking) {
    let dead = |b: *mut BoxHeader| {
        // SAFETY: an element of a reached table is null or a live box.
        unsafe { referent(b) }.is_some_and(|r| !m.is_marked(r))
    };
    for (&t, w) in weak.iter_mut() {
        if w.mode & WEAK_VALUES == 0 || !m.is_marked(t) {
            continue;
        }
        // SAFETY: reached, so live; the world is stopped.
        let t = unsafe { &mut *(t as *mut TableHeader) };
        if let Some(list) = unsafe { list_of(t.hash) } {
            let items = unsafe { elements(list) };
            for pair in items
                .get_mut(1..)
                .unwrap_or_default()
                .as_chunks_mut::<2>()
                .0
            {
                if dead(pair[1]) {
                    pair[1] = std::ptr::null_mut();
                    w.dirty = true;
                }
            }
        }
        if let Some(list) = unsafe { list_of(t.arr) } {
            let n = list.len;
            for v in unsafe { elements(list) }.iter_mut() {
                if dead(*v) {
                    *v = std::ptr::null_mut();
                }
            }
            // The array part never ends in nil: its length stays a
            // border, and `high` remembers how long it was.
            let items = unsafe { elements(list) };
            let mut len = items.len();
            while len > 0 && items[len - 1].is_null() {
                len -= 1;
            }
            if (len as i64) < n {
                list.len = len as i64;
                t.high = t.high.max(n);
            }
        }
    }
}

/// Turn every collectable key of a reached table that nothing reached
/// (a dead weak key, or a key of an entry whose value is nil) into a
/// tombstone with a nil value. Returns the key boxes the tables still
/// name that tracing left unmarked, the tombstones among them.
fn tombstone_dead_keys(weak: &mut BTreeMap<usize, Weak>, m: &dyn Marking) -> Vec<usize> {
    let mut boxes = Vec::new();
    for (&t, w) in weak.iter_mut() {
        if !m.is_marked(t) {
            continue;
        }
        // SAFETY: reached, so live; the world is stopped.
        let t = unsafe { &*(t as *const TableHeader) };
        let Some(list) = (unsafe { list_of(t.hash) }) else {
            continue;
        };
        let items = unsafe { elements(list) };
        for pair in items
            .get_mut(1..)
            .unwrap_or_default()
            .as_chunks_mut::<2>()
            .0
        {
            let k = pair[0];
            // SAFETY: a key of a reached table is a live box.
            match unsafe { referent(k) } {
                Some(r) if !m.is_marked(r) => {
                    // The complement of an address lies outside the
                    // heap, so the tombstone keeps nothing alive, and
                    // is distinct per dead object, so tombstones hash
                    // apart.
                    unsafe {
                        (*k).tag = dead_key_tag();
                        (*k).data = !r;
                    }
                    pair[1] = std::ptr::null_mut();
                    boxes.push(k as usize);
                    w.dirty = true;
                }
                // A number or string key of a nil value, or a key whose
                // object lives on: kept until the table is compacted.
                _ if !k.is_null() && !m.is_marked(k as usize) => boxes.push(k as usize),
                _ => {}
            }
        }
    }
    boxes
}

/// Drop what the collection did not reach from the registries.
fn sweep(reached: &dyn Fn(usize) -> bool) {
    let mut s = state();
    s.weak.retain(|&t, _| reached(t));
    s.dirty = s.weak.values().filter(|w| w.dirty).count();
}

/// Run the library's runner after a collection that queued finalizers
/// or changed how many tables are dirty: it runs the finalizers, unless
/// one is running, and tells the program that count.
fn after_collection() {
    let runner = {
        let s = state();
        if (s.pending.is_empty() && s.dirty == s.published) || s.runner == 0 {
            return;
        }
        s.runner
    };
    // SAFETY: the library's runner, compiled with this signature.
    let run: extern "C" fn() = unsafe { std::mem::transmute(runner) };
    run();
}

fn install() {
    collector::add_weak_hooks(WeakHooks {
        hold,
        trace,
        settle,
    });
    collector::add_weak_sweeper(sweep);
    collector::add_after_collection(after_collection);
}

// ─── what the library calls ─────────────────────────────────────────

/// `setmetatable(t, mt)` has set a metatable whose fields gave `flags`
/// (`WEAK_KEYS`, `WEAK_VALUES`, `FINALIZE`), or removed one: weakness
/// follows the metatable set last, and a table once marked for
/// finalization stays marked until it is finalized. `runner` is the
/// library's finalizer runner.
pub(crate) extern "C" fn host_gc_note(t: i64, flags: i64, runner: i64) {
    let t = t as usize;
    let mut s = state();
    let weak = flags & (WEAK_KEYS | WEAK_VALUES);
    // SAFETY: the library hands over a live table.
    let shaped = unsafe { (*(t as *const TableHeader)).shape } != 0;
    if weak != 0 && !shaped {
        let dirty = s.weak.get(&t).is_some_and(|w| w.dirty);
        s.weak.insert(t, Weak { mode: weak, dirty });
    } else if s.weak.remove(&t).is_some_and(|w| w.dirty) {
        s.dirty -= 1;
    }
    if flags & FINALIZE != 0 && s.marked.insert(t) {
        s.finalizable.push(t);
    }
    if flags != 0 && s.runner == 0 {
        s.runner = runner as usize;
        drop(s);
        install();
    }
}

/// The next object whose finalizer is due, or 0.
pub(crate) extern "C" fn host_gc_next_finalizer() -> i64 {
    state().pending.pop_front().unwrap_or(0) as i64
}

/// Enter (`on` 1) or leave (0) a finalizer run; whether one was
/// running already. Collections are refused while one runs, as the
/// reference refuses them.
pub(crate) extern "C" fn host_gc_finalizing(on: i64) -> i64 {
    let mut s = state();
    let was = s.finalizing;
    s.finalizing = on != 0;
    collector::set_automatic(!s.finalizing && !s.stopped);
    was as i64
}

/// At the end of the program: every object still marked for
/// finalization is due, the last marked first, after those queued.
pub(crate) extern "C" fn host_gc_close() -> i64 {
    let mut s = state();
    let rest = std::mem::take(&mut s.finalizable);
    s.marked.clear();
    for &o in rest.iter().rev() {
        s.pending.push_back(o);
    }
    s.pending.len() as i64
}

/// How many weak tables are dirty, which the program now knows.
pub(crate) extern "C" fn host_gc_dirty_count() -> i64 {
    let mut s = state();
    s.published = s.dirty;
    s.dirty as i64
}

/// Whether `t` is dirty, which the caller now compacts.
pub(crate) extern "C" fn host_gc_take_dirty(t: i64) -> i64 {
    let mut s = state();
    let Some(w) = s.weak.get_mut(&(t as usize)) else {
        return 0;
    };
    if !std::mem::take(&mut w.dirty) {
        return 0;
    }
    s.dirty -= 1;
    1
}

/// `collectgarbage(opt, arg)` by the option's number, in the order
/// `library::gc::GC_OPTIONS` lists them: what the option answers, as
/// an integer (a mode as 1 for generational, a boolean as 1 or 0), or
/// -1 while a finalizer runs, when every option fails.
pub(crate) extern "C" fn host_gc(op: i64, arg: i64) -> i64 {
    let mut s = state();
    if s.finalizing {
        return -1;
    }
    match library::gc::GC_OPTIONS
        .get(op as usize)
        .copied()
        .unwrap_or("")
    {
        // A step is a whole cycle: whether it ended one, which the
        // reference's generational mode never reports.
        "collect" | "step" => {
            let ended = !s.generational;
            drop(s);
            collector::collect();
            ended as i64
        }
        "count" => {
            let stats = collector::stats();
            (stats.live + stats.allocated) as i64
        }
        "isrunning" => !s.stopped as i64,
        "stop" | "restart" => {
            s.stopped = op == library::gc::option("stop");
            collector::set_automatic(!s.stopped);
            0
        }
        "incremental" | "generational" => {
            let was = s.generational;
            s.generational = op == library::gc::option("generational");
            was as i64
        }
        "setpause" => std::mem::replace(&mut s.pause, arg / 4) * 4,
        "setstepmul" => std::mem::replace(&mut s.stepmul, arg / 4) * 4,
        _ => 0,
    }
}

/// One piece of a warning, `tocont` when more of the message follows,
/// as the reference's standalone interpreter handles it: off until
/// `@on`, a message of one piece starting with `@` a control message,
/// and each message written as `Lua warning: ` and its pieces.
pub(crate) extern "C" fn host_warn(message: zrtl::StringConstPtr, tocont: bool) {
    // SAFETY: the library hands over a live string.
    let text = unsafe { crate::host::bytes_of(message) };
    let mut s = state();
    if s.warn != Warn::Continued && !tocont && text.first() == Some(&b'@') {
        match &text[1..] {
            b"on" => s.warn = Warn::On,
            b"off" => s.warn = Warn::Off,
            _ => {}
        }
        return;
    }
    use std::io::Write;
    let mut err = std::io::stderr().lock();
    match s.warn {
        Warn::Off => return,
        Warn::On => {
            let _ = err.write_all(b"Lua warning: ");
        }
        Warn::Continued => {}
    }
    let _ = err.write_all(text);
    if tocont {
        s.warn = Warn::Continued;
    } else {
        let _ = err.write_all(b"\n");
        s.warn = Warn::On;
    }
}

pub(crate) static SYMBOLS: [zrtl::ZrtlSymbol; 8] = [
    zrtl::ZrtlSymbol::new(c"$Lua$gc".as_ptr(), host_gc as *const u8),
    zrtl::ZrtlSymbol::new(c"$Lua$gc_note".as_ptr(), host_gc_note as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$gc_next_finalizer".as_ptr(),
        host_gc_next_finalizer as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$gc_finalizing".as_ptr(),
        host_gc_finalizing as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$gc_close".as_ptr(), host_gc_close as *const u8),
    zrtl::ZrtlSymbol::new(
        c"$Lua$gc_dirty_count".as_ptr(),
        host_gc_dirty_count as *const u8,
    ),
    zrtl::ZrtlSymbol::new(
        c"$Lua$gc_take_dirty".as_ptr(),
        host_gc_take_dirty as *const u8,
    ),
    zrtl::ZrtlSymbol::new(c"$Lua$warn".as_ptr(), host_warn as *const u8),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_header_is_the_library_s() {
        let fields: Vec<String> = library::table_header_fields(zyntax_typed_ast::TypeId::next())
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            fields,
            ["arr", "hash", "meta", "high", "shape", "present"],
            "TableHeader must follow the library's field order"
        );
        assert_eq!(std::mem::size_of::<TableHeader>(), 6 * 8);
    }
}
