//! krio-fiber backed `FiberCfg` implementation.
//!
//! `KrioFiberBackend` is the default native impl of the fiber
//! contract defined in `zyntax_compiler::fiber_backend::FiberCfg`. It
//! wraps krio-fiber's stackful coroutines behind the C-ABI surface the
//! `krio_fiber_*` runtime symbols expose; the HIR's fiber ops route
//! through here once [`install`] runs during runtime construction.
//!
//! Closure handles today are `extern "C" fn()` pointers — no captured
//! environment. When the frontend gains real closures-with-captures,
//! the handle becomes a richer struct pointer and the body trampoline
//! gains a marshalling layer; the `FiberCfg` contract itself doesn't
//! change.
//!
//! Returned `*mut FiberRepr` handles point to a [`Shell`] naming a
//! `Box<krio_fiber::Fiber>`: a `Box` for most fibers, and a small block
//! of the collector's heap for a fiber that `make_fiber` allows. The contract requires every
//! handle stays on the thread that created it (matches krio-fiber's
//! `!Send` `Fiber`), and the caller owns the lifetime. `fiber_free`
//! is the symmetric `Box::from_raw` drop (unmaps the stack + guard
//! page + clears the per-fiber error slot); the frontend emits it at
//! the scope exit of a fiber that isn't moved out. Fibers created in
//! conditional/inner scopes, or received via a call return, aren't
//! dropped yet (a later liveness-based pass) — they leak but never
//! double-free.
//!
//! A collectable fiber is reclaimed by the collector once nothing
//! refers to its handle: its stack, environment and error payload are
//! read only while the handle is reached, and a fiber found unreached
//! is dropped after the collection.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use zyntax_compiler::host_heap::Hold;

use krio_fiber::{Fiber, FiberStep};
use zyntax_compiler::collector::{self, Marking, WeakHooks};
use zyntax_compiler::fiber_backend::{
    self, FIBER_STEP_DONE, FIBER_STEP_ERRORED, FIBER_STEP_YIELDED, FiberCfg,
};
use zyntax_compiler::pool_alloc::{zyntax_alloc, zyntax_free};
use zyntax_compiler::zrtl::FiberRepr;

/// `(variant_tag, payload)` stashed by [`Self::fiber_abort_with`].
/// The tag matches the prelude's `FiberError` enum (1 = Message,
/// 2 = Custom). Payload is a pointer for both — `*const u8` to a
/// `String` for Message, `*const u8` to a `DynamicBox` for Custom.
type AbortInfo = (i64, i64);

thread_local! {
    /// Payload latched by `fiber_abort_with` before yielding the
    /// active fiber. Promoted to the per-fiber error map by
    /// `encode_step` when it surfaces an `Errored` step.
    ///
    /// We can't unwind through the Cranelift / LLVM-generated frame
    /// above the abort call cleanly (DWARF info varies by platform
    /// / opt level), so the abort is implemented as a *deferred*
    /// one: stash payload, yield, then have the resume-side encoder
    /// swap `Yielded` for `Errored`. The fiber stays suspended at
    /// the abort point.
    static ABORT_PAYLOAD: Cell<Option<AbortInfo>> = const { Cell::new(None) };

    /// Scalar returned by `yield_u64` when the host resumes the active fiber.
    /// The yield runtime call must return `void` to generated HIR today, so the
    /// backend retains the value until `krio_fiber_take_input` consumes it.
    static RESUME_INPUT: Cell<Option<u64>> = const { Cell::new(None) };

    /// Per-fiber error slot. Populated by `encode_step` when it
    /// surfaces Errored, read+cleared by `fiber_take_error` from
    /// the caller's side after iteration ends. Keyed by the fiber
    /// handle's raw pointer cast to `usize` (the handle stays
    /// thread-local so this map is safely accessible without
    /// synchronisation).
    static ERROR_MAP: RefCell<HashMap<usize, AbortInfo>> = RefCell::new(HashMap::new());

    /// Each fiber's environment address, by handle; see `fiber_env`.
    static ENV_MAP: RefCell<HashMap<usize, usize>> = RefCell::new(HashMap::new());

    /// What a fiber's environment and error payload name, kept for a
    /// collector on any thread, by handle, until the fiber is freed.
    static HELD: RefCell<HashMap<usize, Vec<Hold>>> = RefCell::new(HashMap::new());

    /// Each fiber's body address, by handle, read when its first resume
    /// starts it.
    static BODIES: RefCell<HashMap<usize, usize>> = RefCell::new(HashMap::new());

    /// The fibers being resumed right now, innermost last, so a body
    /// can find its own handle.
    static RUNNING: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };

    /// Task that owns fibers created right now. The cooperative executor
    /// stamps this (via `set_fiber_task_id`) before driving a task, so a
    /// fiber the task creates can be freed if that task is later cancelled
    /// — a cancelled task never reaches its normal scope-exit `FiberDrop`.
    /// 0 = no owning task (plain, non-async fibers).
    static FIBER_TASK_ID: Cell<i64> = const { Cell::new(0) };

    /// Live fibers grouped by owning task, for cancel-time cleanup. Added
    /// on `fiber_new` under the current `FIBER_TASK_ID`, removed on
    /// `fiber_free` (the normal scope-exit drop). `free_task_fibers` frees
    /// whatever a cancelled task left behind. A fiber that a task RETURNS
    /// only escapes on a completing (non-cancelled) path, so this never
    /// frees a live escapee.
    static TASK_FIBERS: RefCell<HashMap<i64, HashSet<usize>>> =
        RefCell::new(HashMap::new());

    /// Fibers kept in the collector's heap, by handle; see `make_fiber`.
    static COLLECTABLE: RefCell<HashMap<usize, Reachability>> = RefCell::new(HashMap::new());

    /// Collectable fibers the collection in progress found unreached,
    /// dropped once it ends. Capacity is reserved when a fiber is made,
    /// because the hook that fills it may not allocate.
    static DOOMED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };

    /// Which collection is running, so a fiber's stack is read once.
    static GC_EPOCH: Cell<u64> = const { Cell::new(0) };
}

/// What the collector has found of a collectable fiber.
#[derive(Clone, Copy, Default)]
struct Reachability {
    /// The collection whose marking last read this fiber's stack.
    scanned: u64,
    /// Found unreached; dropped when the collection ends.
    doomed: bool,
}

/// What a handle points to. The fiber stays out of the collector's
/// heap: a word left in it would hold other objects up, and fibers
/// made one after another would hold each other up. Only these two
/// words are in the heap, and both are written whole.
#[repr(C)]
struct Shell {
    fiber: *mut Fiber,
    _reserved: usize,
}

/// The fiber a handle names.
///
/// # Safety
/// `handle` is a live handle from `make_fiber`.
unsafe fn fiber_ptr(handle: usize) -> *mut Fiber {
    // SAFETY: a handle points to its shell until released.
    unsafe { (*(handle as *const Shell)).fiber }
}

/// Resident bytes a fiber's stack and bookkeeping hold outside the heap,
/// spent against the collector's budget when one is made.
const FIBER_PRESSURE: usize = 16 << 10;

/// Stamp the task that owns fibers created from now on (see `TASK_FIBERS`).
pub fn set_fiber_task_id(id: i64) {
    FIBER_TASK_ID.with(|c| c.set(id));
}

/// Number of live fibers currently attributed to `task_id`. For tests /
/// diagnostics — after a task completes or is cancelled this should be 0.
pub fn task_fiber_count(task_id: i64) -> usize {
    TASK_FIBERS.with(|m| m.borrow().get(&task_id).map(|s| s.len()).unwrap_or(0))
}

/// Free every fiber still owned by `task_id`. Used when the task is
/// cancelled and won't run its own scope-exit `FiberDrop`s: unmaps each
/// fiber's stack + guard page and clears its per-fiber error slot.
pub fn free_task_fibers(task_id: i64) {
    let ptrs: Vec<usize> = TASK_FIBERS.with(|m| {
        m.borrow_mut()
            .remove(&task_id)
            .map(|s| s.into_iter().collect())
            .unwrap_or_default()
    });
    let backend = KrioFiberBackend;
    for ptr in ptrs {
        unsafe {
            backend.fiber_free(ptr as *mut FiberRepr);
        }
    }
}

/// Default fiber backend backed by krio-fiber. Stateless — the
/// per-fiber state lives inside each `Fiber` instance; this struct is
/// only the vtable carrier.
pub struct KrioFiberBackend;

impl KrioFiberBackend {
    /// Construct a backend handle. Cheap — there is no setup; the
    /// trait methods reach into krio-fiber statics directly.
    pub fn new() -> Self {
        Self
    }

    /// Make a fiber and hand its ownership to the caller as a handle.
    /// With `may_collect`, a fiber nothing owns on the collector's own
    /// thread lives in the collector's heap, to be reclaimed once
    /// nothing refers to it; any other fiber is a `Box` the caller
    /// frees.
    ///
    /// # Safety
    /// `closure` must satisfy the contract of `fiber_new`.
    unsafe fn make_fiber(
        &self,
        closure: *mut u8,
        stack_size: i64,
        may_collect: bool,
    ) -> *mut FiberRepr {
        let body = make_body();
        let fiber = if stack_size > 0 {
            Fiber::with_stack_size(stack_size as usize, body)
        } else {
            Fiber::new(body)
        };
        // Record the fiber under the task driving right now so a cancel of
        // that task can free it.
        let task = FIBER_TASK_ID.with(|c| c.get());
        let in_heap = may_collect
            && task == 0
            && !zyntax_compiler::host_heap::is_installed()
            && collector::collecting_here();
        let shell = Shell {
            fiber: Box::into_raw(Box::new(fiber)),
            _reserved: 0,
        };
        let ptr = if in_heap {
            // SAFETY: the block is sized for a `Shell`, written whole
            // before anything reads it, and aligned to 16 by the pool.
            unsafe {
                let block = zyntax_alloc(std::mem::size_of::<Shell>()) as *mut Shell;
                assert!(!block.is_null(), "fiber: the pool could not allocate");
                block.write(shell);
                block
            }
        } else {
            Box::into_raw(Box::new(shell))
        } as *mut FiberRepr;
        BODIES.with(|b| b.borrow_mut().insert(ptr as usize, closure as usize));
        if in_heap {
            watch_for_collection();
            COLLECTABLE.with(|c| {
                let mut c = c.borrow_mut();
                c.insert(ptr as usize, Reachability::default());
                DOOMED.with(|d| {
                    let mut d = d.borrow_mut();
                    let want = c.len();
                    if d.capacity() < want {
                        let extra = want - d.len();
                        d.reserve(extra);
                    }
                });
            });
            collector::note_external(FIBER_PRESSURE);
        }
        TASK_FIBERS.with(|m| {
            m.borrow_mut().entry(task).or_default().insert(ptr as usize);
        });
        // A host's collector reads the stack from whatever thread it
        // runs on; this one's own windows are read in `stack_windows`.
        if zyntax_compiler::host_heap::is_installed() {
            zyntax_compiler::collector::add_root_range_with(ptr as *const u8, 1, fiber_spans);
        }
        ptr
    }

    /// Drop a fiber and everything kept under its handle, and give its
    /// storage back: to the pool for a fiber in the collector's heap, to
    /// the allocator for a `Box`.
    ///
    /// # Safety
    /// `fiber` is a live handle from `make_fiber`, released once.
    unsafe fn release(&self, fiber: *mut FiberRepr) {
        let key = fiber as usize;
        // Clear any per-fiber error latched under this handle before
        // the address is reclaimed — otherwise a future fiber reusing
        // the same address could inherit a stale error.
        ERROR_MAP.with(|m| {
            m.borrow_mut().remove(&key);
        });
        ENV_MAP.with(|m| {
            m.borrow_mut().remove(&key);
        });
        HELD.with(|m| m.borrow_mut().remove(&key));
        BODIES.with(|m| m.borrow_mut().remove(&key));
        if zyntax_compiler::host_heap::is_installed() {
            zyntax_compiler::collector::remove_root_range(fiber as *const u8);
        }
        // Drop the fiber from its owning task's set so a later
        // `free_task_fibers` can't double-free it.
        TASK_FIBERS.with(|m| {
            for set in m.borrow_mut().values_mut() {
                set.remove(&key);
            }
        });
        let in_heap = COLLECTABLE.with(|c| c.borrow_mut().remove(&key).is_some());
        // Reclaim ownership handed out by `make_fiber`'s `Box::into_raw`.
        // Dropping the fiber returns its stack.
        // SAFETY: the task maps above ensure this handle is reclaimed once.
        unsafe {
            drop(Box::from_raw(fiber_ptr(key)));
            if in_heap {
                zyntax_free(fiber as *mut u8);
            } else {
                drop(Box::from_raw(fiber as *mut Shell));
            }
        }
    }
}

/// Whether the collector reclaims this fiber once nothing refers to it.
fn is_collectable(handle: usize) -> bool {
    COLLECTABLE.with(|c| c.borrow().contains_key(&handle))
}

/// Register the collector hooks once, when the first collectable fiber is
/// made.
fn watch_for_collection() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        collector::add_weak_hooks(WeakHooks {
            hold: begin_collection,
            trace: read_reached_fibers,
            settle: doom_unreached_fibers,
        });
        collector::add_after_collection(drop_doomed_fibers);
    });
}

/// A collection starts: fibers' stacks are read afresh.
fn begin_collection(_hold: &mut dyn FnMut(usize)) {
    GC_EPOCH.with(|e| e.set(e.get() + 1));
}

/// Marking has drained: what a reached collectable fiber keeps alive is
/// marked, which may reach more fibers, and this runs again until none
/// does. A running fiber is reached by definition.
fn read_reached_fibers(m: &mut dyn Marking) {
    let epoch = GC_EPOCH.with(Cell::get);
    RUNNING.with(|r| {
        for &h in r.borrow().iter() {
            if COLLECTABLE.with(|c| c.borrow().contains_key(&h)) {
                m.mark(h);
            }
        }
    });
    COLLECTABLE.with(|c| {
        for (&h, slot) in c.borrow_mut().iter_mut() {
            if slot.doomed || slot.scanned == epoch || !m.is_marked(h) {
                continue;
            }
            slot.scanned = epoch;
            // SAFETY: a key of COLLECTABLE is a live handle until released.
            let fiber = unsafe { &*fiber_ptr(h) };
            if let Some(env) = ENV_MAP.with(|e| e.borrow().get(&h).copied()) {
                m.mark(env);
            }
            if let Some((_, payload)) = ERROR_MAP.with(|e| e.borrow().get(&h).copied()) {
                m.mark(payload as usize);
            }
            if matches!(fiber.state(), krio_fiber::FiberState::Suspended) {
                let (start, len) = fiber.stack_range();
                let top = start as usize + len;
                // A suspended fiber's window starts at its saved stack
                // pointer, below which the switch spilled the registers.
                let mut at = fiber.saved_sp() as usize & !7;
                while at + 8 <= top {
                    // SAFETY: the window lies inside the fiber's stack.
                    m.mark(unsafe { *(at as *const usize) });
                    at += 8;
                }
            }
        }
    });
}

/// Tracing has settled: a collectable fiber still unreached is dead. It
/// is marked once more so its storage outlives the collection, and
/// queued to be dropped when the collection ends.
fn doom_unreached_fibers(m: &mut dyn Marking) {
    COLLECTABLE.with(|c| {
        for (&h, slot) in c.borrow_mut().iter_mut() {
            if slot.doomed || m.is_marked(h) {
                continue;
            }
            slot.doomed = true;
            m.mark(h);
            DOOMED.with(|d| d.borrow_mut().push(h));
        }
    });
}

/// A collection has ended: drop the fibers it found unreached.
fn drop_doomed_fibers() {
    let doomed = DOOMED.with(|d| std::mem::take(&mut *d.borrow_mut()));
    if std::env::var_os("ZYNTAX_TRACE_GC").is_some() {
        let alive = COLLECTABLE.with(|c| c.borrow().len()) - doomed.len();
        eprintln!("[gc]   fibers: {} dropped, {alive} kept", doomed.len());
    }
    let backend = KrioFiberBackend;
    for h in doomed {
        // SAFETY: each is a live collectable handle, queued once.
        unsafe { backend.release(h as *mut FiberRepr) };
    }
}

impl Default for KrioFiberBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a krio-fiber `FiberStep` to the packed (tag, payload) i64
/// the C-ABI surface returns.
///
/// On Errored, the `(variant, payload)` from `ABORT_PAYLOAD` (or
/// zeros for a non-abort panic) gets stashed in `ERROR_MAP` keyed
/// by the fiber handle so the caller's later `fiber_take_error`
/// can return a typed `FiberError`.
///
/// * `Yielded` — if `ABORT_PAYLOAD` is latched, the yield was
///   actually a `Fiber.abort(err)`; surface `Errored` and write
///   the variant/payload to `ERROR_MAP`. Otherwise the payload
///   is the u64 the body passed to `yield_u64`.
/// * `Done` — fiber returned normally; payload 0.
/// * `Errored` — krio-fiber-side panic catch fired (user-code
///   panic, not `abort_with`); payload 0; map gets a `(0, 0)`
///   placeholder so `f.error()` returns `None` (variant=0 is the
///   absent marker).
fn encode_step(step: FiberStep, key: usize, fiber: &Fiber) -> i64 {
    if let Some(info) = ABORT_PAYLOAD.with(|cell| cell.take()) {
        // Drain whatever yield value the body pushed (the yield
        // we used to suspend the body — not the payload of record).
        let _ = fiber.take_yield_u64();
        ERROR_MAP.with(|m| m.borrow_mut().insert(key, info));
        hold_for(key, info.1 as usize as *const u8);
        // The packed step's i64 payload carries the variant tag in
        // its low bits so callers that don't bother with
        // `take_error` still see *something* indicative; the
        // canonical retrieval is `take_error` though.
        return fiber_backend::pack_fiber_step(FIBER_STEP_ERRORED, info.0);
    }
    match step {
        FiberStep::Yielded => {
            let payload = fiber.take_yield_u64().unwrap_or(0) as i64;
            fiber_backend::pack_fiber_step(FIBER_STEP_YIELDED, payload)
        }
        FiberStep::Done => fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0),
        FiberStep::Errored => {
            // User-code panic, not abort_with — no payload to
            // record. Still mark the slot so a later
            // `fiber_take_error` doesn't return stale data from
            // an old fiber that reused the address.
            ERROR_MAP.with(|m| m.borrow_mut().insert(key, (0, 0)));
            fiber_backend::pack_fiber_step(FIBER_STEP_ERRORED, 0)
        }
    }
}

/// Build a closure for `Fiber::new` from a raw `extern "C" fn()`
/// pointer. Today's contract: the body takes no arguments and returns
/// nothing — it interacts with the host via `krio_fiber_yield` /
/// `take_input_u64` while running.
fn make_body() -> impl FnOnce() {
    // The closure captures nothing, so its box is zero-sized and owns no
    // allocation: a suspended fiber that is dropped never returns from
    // the call that would free one. The body's address is looked up by
    // the handle of the fiber that is running, which the first resume
    // pushed. The address is a function pointer; treating it as
    // `extern "C" fn()` is sound when the caller honours the
    // closure-handle contract.
    || {
        let addr = RUNNING
            .with(|r| r.borrow().last().copied())
            .and_then(|h| BODIES.with(|b| b.borrow().get(&h).copied()))
            .unwrap_or(0);
        if addr == 0 {
            // Empty body — yields nothing, returns immediately. Useful
            // for "smoke test that fiber lifecycle works at all"
            // checks before any real body is wired.
            return;
        }
        let body: extern "C" fn() = unsafe { std::mem::transmute(addr) };
        body();
    }
}

fn hold_for(fiber: usize, address: *const u8) {
    // A collectable fiber's are read by the hook while its handle is
    // reached; holding one would keep the fiber alive through an
    // environment that refers back to it.
    if address.is_null() || is_collectable(fiber) {
        return;
    }
    HELD.with(|m| {
        m.borrow_mut()
            .entry(fiber)
            .or_default()
            .push(Hold::new(address))
    });
}

/// The stack a live fiber may hold pointers on, read from any thread: the
/// whole of it, since whether it is running is known only on its own.
fn fiber_spans(start: *const u8, _len: usize, visit: &mut dyn FnMut(usize, usize)) {
    // SAFETY: registered by `make_fiber` for a live handle and removed
    // by `fiber_free` before it goes.
    let fiber = unsafe { &*fiber_ptr(start as usize) };
    if matches!(
        fiber.state(),
        krio_fiber::FiberState::New | krio_fiber::FiberState::Suspended
    ) {
        let (low, len) = fiber.stack_range();
        visit(low as usize, low as usize + len);
    }
}

impl FiberCfg for KrioFiberBackend {
    unsafe fn fiber_new(&self, closure: *mut u8, stack_size: i64) -> *mut FiberRepr {
        // SAFETY: forwards the caller's closure contract.
        unsafe { self.make_fiber(closure, stack_size, false) }
    }

    unsafe fn fiber_new_with_env(
        &self,
        closure: *mut u8,
        env: *mut u8,
        stack_size: i64,
    ) -> *mut FiberRepr {
        // SAFETY: this method forwards the caller's valid closure contract.
        let ptr = unsafe { self.make_fiber(closure, stack_size, true) };
        ENV_MAP.with(|m| {
            m.borrow_mut().insert(ptr as usize, env as usize);
        });
        hold_for(ptr as usize, env);
        ptr
    }

    fn fiber_env(&self) -> *mut u8 {
        let current = RUNNING.with(|r| r.borrow().last().copied());
        match current {
            Some(handle) => ENV_MAP
                .with(|m| m.borrow().get(&handle).copied())
                .unwrap_or(0) as *mut u8,
            None => std::ptr::null_mut(),
        }
    }

    unsafe fn fiber_resume(&self, fiber: *mut FiberRepr) -> i64 {
        let handle = fiber as usize;
        // SAFETY: callers pass a live handle returned by `fiber_new`.
        let fiber = unsafe { &mut *fiber_ptr(handle) };
        // Honour cooperative cancel at the resume boundary: an
        // auto-generated `fiber def` body has no chance to call
        // `is_cancelled()` itself between yields, so the cancel flag
        // would otherwise sit unobserved until the body finished
        // naturally. Surfacing `Done` here matches the user-visible
        // semantics of `Fiber::cancel()` ("subsequent `next()` ends
        // the iteration").
        if fiber.is_cancelled() {
            return fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0);
        }
        // A finished fiber stays finished: resuming past exhaustion is an
        // ordinary thing for a consumer loop to do, and it means Done —
        // matching `next()` returning None forever, not a panic.
        if matches!(
            fiber.state(),
            krio_fiber::FiberState::Done | krio_fiber::FiberState::Errored
        ) {
            return fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0);
        }
        RUNNING.with(|r| r.borrow_mut().push(handle));
        let step = fiber.resume();
        RUNNING.with(|r| {
            r.borrow_mut().pop();
        });
        encode_step(step, handle, fiber)
    }

    unsafe fn fiber_resume_with(&self, fiber: *mut FiberRepr, value: i64) -> i64 {
        let handle = fiber as usize;
        // SAFETY: callers pass a live handle returned by `fiber_new`.
        let fiber = unsafe { &mut *fiber_ptr(handle) };
        if fiber.is_cancelled() {
            return fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0);
        }
        if matches!(
            fiber.state(),
            krio_fiber::FiberState::Done | krio_fiber::FiberState::Errored
        ) {
            return fiber_backend::pack_fiber_step(FIBER_STEP_DONE, 0);
        }
        RUNNING.with(|r| r.borrow_mut().push(handle));
        let step = fiber.resume_with_u64(value as u64);
        RUNNING.with(|r| {
            r.borrow_mut().pop();
        });
        encode_step(step, handle, fiber)
    }

    fn fiber_yield(&self, value: i64) {
        // `yield_u64` is a free function on krio-fiber — it consults
        // the thread-local "currently active fiber" set by `resume`.
        // Its `Option<u64>` return is what the matching `resume_with`
        // delivered. Generated HIR currently models FiberYield as a void
        // instruction, so retain that return until the explicit receiving
        // primitive reads it.
        let input = krio_fiber::yield_u64(value as u64);
        RESUME_INPUT.with(|slot| slot.set(input));
    }

    fn fiber_take_input(&self) -> i64 {
        RESUME_INPUT.with(|slot| slot.take().map_or(0, |value| value as i64))
    }

    unsafe fn fiber_transfer(&self, _target: *mut FiberRepr, _value: i64) -> i64 {
        // Symmetric transfer isn't in krio-fiber upstream yet. Two
        // options when the HIR op fires: panic to make the gap loud,
        // or emulate via resume + yield through the caller chain.
        // Emulation changes timing semantics in subtle ways, so panic
        // here keeps the gap honest and drives the upstream PR.
        unimplemented!(
            "FiberTransfer: symmetric switch not yet supported (krio-fiber upstream gap)"
        )
    }

    unsafe fn fiber_cancel(&self, fiber: *mut FiberRepr) {
        // SAFETY: callers pass a live handle returned by `fiber_new`.
        let fiber = unsafe { &*fiber_ptr(fiber as usize) };
        fiber.cancel();
    }

    fn stack_windows(&self, sp: usize, host_top: usize, out: &mut dyn FnMut(usize, usize)) {
        let running: Vec<usize> = RUNNING.with(|r| r.borrow().clone());
        // SAFETY: every handle in RUNNING and TASK_FIBERS is a live
        // Box<Fiber> until fiber_free removes it.
        let fiber = |h: usize| unsafe { &*fiber_ptr(h) };
        let top_of = |f: &Fiber| {
            let (start, len) = f.stack_range();
            start as usize + len
        };
        // The host stack was left where the outermost running fiber
        // was entered; each running fiber's stack where it entered the
        // next; the innermost is the one the collector runs on.
        let host_low = match running.first() {
            Some(&h) => fiber(h).caller_sp() as usize,
            None => sp,
        };
        out(host_low, host_top);
        for (i, &h) in running.iter().enumerate() {
            let low = match running.get(i + 1) {
                Some(&inner) => fiber(inner).caller_sp() as usize,
                None => sp,
            };
            out(low, top_of(fiber(h)));
        }
        // A suspended fiber's window starts at the saved stack pointer,
        // below which its context switch spilled the registers.
        TASK_FIBERS.with(|m| {
            for set in m.borrow().values() {
                for &h in set {
                    if running.contains(&h) || is_collectable(h) {
                        continue;
                    }
                    let f = fiber(h);
                    if matches!(f.state(), krio_fiber::FiberState::Suspended) {
                        out(f.saved_sp() as usize, top_of(f));
                    }
                }
            }
        });
    }

    fn held_addresses(&self, out: &mut dyn FnMut(usize)) {
        // A collectable fiber's are read by the hook while it is reached.
        ENV_MAP.with(|m| {
            for (&h, &env) in m.borrow().iter() {
                if !is_collectable(h) {
                    out(env);
                }
            }
        });
        // An error payload waits here until the caller takes it.
        ERROR_MAP.with(|m| {
            for (&h, &(_, payload)) in m.borrow().iter() {
                if !is_collectable(h) {
                    out(payload as usize);
                }
            }
        });
    }

    unsafe fn fiber_free(&self, fiber: *mut FiberRepr) {
        if fiber.is_null() {
            return;
        }
        // SAFETY: the caller passes a live handle, freed once.
        unsafe { self.release(fiber) };
    }

    fn fiber_abort_with(&self, variant: i64, payload: i64) {
        // Latch the (variant, payload), then yield. The caller-
        // side resume (via `encode_step`) sees the latch, writes
        // it to the per-fiber `ERROR_MAP`, and surfaces an
        // `Errored` step. We can't unwind through Cranelift /
        // LLVM-emitted frames cleanly enough across platforms to
        // do a panic-based immediate abort, so the abort is
        // *deferred*: the fiber body suspends at the abort point.
        // The typical `while let Some(x) = f.next()` pattern then
        // ends because Errored decodes to None.
        ABORT_PAYLOAD.with(|cell| cell.set(Some((variant, payload))));
        let _ = krio_fiber::yield_u64(payload as u64);
    }

    unsafe fn fiber_take_error(&self, fiber: *mut FiberRepr) -> i64 {
        let key = fiber as usize;
        let info = ERROR_MAP.with(|m| m.borrow_mut().remove(&key));
        match info {
            // No entry, or the sentinel for "panicked but no
            // `abort_with` was called" — caller sees `None`.
            None | Some((0, _)) => 0,
            Some((variant, payload)) => {
                // Packed shape: bit 0 = present, bits 1-2 = variant,
                // bits 3-63 = payload. The variant fits in 2 bits
                // (we only ever use 1 or 2). The payload's high
                // bits get sign-extended on the SSA-side shr.
                let present = 1i64;
                let v_bits = (variant & 0b11) << 1;
                let p_bits = payload << 3;
                present | v_bits | p_bits
            }
        }
    }
}

/// Install [`KrioFiberBackend`] as the global fiber backend. Returns
/// `true` on first install, `false` if a backend was already wired
/// (set-once semantics — runtimes that share a process don't fight
/// over the slot).
pub fn install() -> bool {
    fiber_backend::install_fiber_backend(Box::new(KrioFiberBackend::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;
    use zyntax_compiler::fiber_backend::{FIBER_STEP_DONE, FIBER_STEP_YIELDED, unpack_fiber_step};
    use zyntax_compiler::zrtl;

    /// Install the backend exactly once per test binary. Multiple
    /// tests share the global slot, and `install_fiber_backend` is
    /// set-once — call directly per-test would race; this gate makes
    /// it deterministic.
    fn ensure_installed() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            install();
        });
    }

    /// Module-scoped sink so the empty-body fiber test has a visible
    /// side effect to assert against. The fiber body writes a marker
    /// here; the test reads it after resume.
    static EMPTY_BODY_RAN: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    extern "C" fn empty_body() {
        EMPTY_BODY_RAN.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    extern "C" fn yields_42_then_done() {
        let _ = krio_fiber::yield_u64(42);
    }

    extern "C" fn yields_twice() {
        let _ = krio_fiber::yield_u64(10);
        let _ = krio_fiber::yield_u64(20);
    }

    extern "C" fn receives_resume_value() {
        unsafe { zrtl::krio_fiber_yield(7) };
        let input = unsafe { zrtl::krio_fiber_take_input() };
        unsafe { zrtl::krio_fiber_yield(input) };
    }

    #[test]
    fn empty_body_runs_to_done() {
        ensure_installed();
        EMPTY_BODY_RAN.store(false, std::sync::atomic::Ordering::SeqCst);
        unsafe {
            let f = zrtl::krio_fiber_new(empty_body as *mut u8, 0);
            assert!(!f.is_null(), "fiber allocation failed");
            let step = zrtl::krio_fiber_resume(f);
            let (tag, payload) = unpack_fiber_step(step);
            assert_eq!(tag, FIBER_STEP_DONE);
            assert_eq!(payload, 0);
        }
        assert!(EMPTY_BODY_RAN.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn yields_then_completes() {
        ensure_installed();
        unsafe {
            let f = zrtl::krio_fiber_new(yields_42_then_done as *mut u8, 0);
            // First resume: body runs until it yields 42.
            let step = zrtl::krio_fiber_resume(f);
            let (tag, payload) = unpack_fiber_step(step);
            assert_eq!(tag, FIBER_STEP_YIELDED);
            assert_eq!(payload, 42);
            // Second resume: body returns; tag becomes Done.
            let step = zrtl::krio_fiber_resume(f);
            let (tag, payload) = unpack_fiber_step(step);
            assert_eq!(tag, FIBER_STEP_DONE);
            assert_eq!(payload, 0);
        }
    }

    #[test]
    fn multi_yield_sequence() {
        ensure_installed();
        unsafe {
            let f = zrtl::krio_fiber_new(yields_twice as *mut u8, 0);
            let (t1, p1) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            let (t2, p2) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            let (t3, p3) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            assert_eq!((t1, p1), (FIBER_STEP_YIELDED, 10));
            assert_eq!((t2, p2), (FIBER_STEP_YIELDED, 20));
            assert_eq!((t3, p3), (FIBER_STEP_DONE, 0));
        }
    }

    /// Make `n` fibers, run each to its first yield, and keep no handle.
    #[inline(never)]
    fn abandon(n: usize) {
        for _ in 0..n {
            // SAFETY: the body is a plain `extern "C" fn()`.
            unsafe {
                let f =
                    zrtl::krio_fiber_new_with_env(yields_twice as *mut u8, std::ptr::null_mut(), 0);
                zrtl::krio_fiber_resume(f);
            }
        }
    }

    /// Where the test keeps a handle the collector can find.
    static KEPT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    #[test]
    fn a_collection_reclaims_abandoned_fibers_and_keeps_referenced_ones() {
        ensure_installed();
        collector::enable();
        // SAFETY: `KEPT` is a static of one word.
        collector::add_root_range(&KEPT as *const _ as *const u8, 8);
        let count = || COLLECTABLE.with(|c| c.borrow().len());
        let before = count();

        // SAFETY: the body is a plain `extern "C" fn()`.
        let kept = unsafe {
            let f = zrtl::krio_fiber_new_with_env(yields_twice as *mut u8, std::ptr::null_mut(), 0);
            zrtl::krio_fiber_resume(f);
            f
        };
        KEPT.store(kept as usize, std::sync::atomic::Ordering::SeqCst);
        abandon(64);
        assert_eq!(count() - before, 65, "each fiber is made in the heap");

        collector::collect();
        // A handle can linger in a stale stack slot and keep a fiber or
        // two alive; the rest must be gone.
        let left = count() - before;
        assert!(
            (1..=4).contains(&left),
            "the kept fiber stays and the abandoned ones go, {left} left"
        );

        // The kept fiber still runs from where it suspended.
        let kept = KEPT.load(std::sync::atomic::Ordering::SeqCst) as *mut FiberRepr;
        // SAFETY: a live handle from `krio_fiber_new_with_env`.
        let (tag, payload) = unsafe { unpack_fiber_step(zrtl::krio_fiber_resume(kept)) };
        assert_eq!((tag, payload), (FIBER_STEP_YIELDED, 20));
        // SAFETY: freed once.
        unsafe { zrtl::krio_fiber_free(kept) };
        KEPT.store(0, std::sync::atomic::Ordering::SeqCst);
    }

    #[test]
    fn bidirectional_resume_value_is_visible_inside_the_fiber() {
        ensure_installed();
        unsafe {
            let f = zrtl::krio_fiber_new(receives_resume_value as *mut u8, 0);
            let (first_tag, first_payload) = unpack_fiber_step(zrtl::krio_fiber_resume(f));
            assert_eq!((first_tag, first_payload), (FIBER_STEP_YIELDED, 7));
            let (second_tag, second_payload) =
                unpack_fiber_step(zrtl::krio_fiber_resume_with(f, 99));
            assert_eq!((second_tag, second_payload), (FIBER_STEP_YIELDED, 99));
            zrtl::krio_fiber_free(f);
        }
    }
}
