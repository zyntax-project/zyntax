//! Algebraic-effects runtime — handler stack + 5 `__zyntax_effect_*`
//! symbols compiled code can call.
//!
//! ## Overview
//!
//! For Tier 3 (resumable) algebraic effects, the krio-transformed
//! caller function needs three runtime services that the Cranelift
//! backend can't synthesise by itself:
//!
//!  1. A per-thread *handler stack* — a `Vec<HandlerFrame>` describing
//!     which handlers are in scope when a `perform` instruction runs.
//!     `@with(H)` annotations push frames at function entry; the
//!     matching `__zyntax_effect_pop_handler` call pops them at exit.
//!  2. A *lookup* that walks the stack to find the topmost frame
//!     handling a given effect — needed because handler scope is
//!     dynamic (you can `@with(H1)` an outer block and `@with(H2)` an
//!     inner one calling the same effect).
//!  3. A pair of helpers (`resume`, `abort`) that the resumable
//!     handler body calls instead of `return` — these talk to the
//!     caller's state machine (set state, store result, return
//!     control).
//!
//! ## Five symbols this module exports
//!
//! All are `extern "C"` and registered with the JIT under their
//! `__zyntax_effect_*` names. Their typed signatures are filled in by
//! [`register_effect_runtime_symbols`] so the Cranelift backend's
//! call-site lowering agrees with the actual Rust ABI.
//!
//! | Symbol                              | Purpose                                  |
//! |-------------------------------------|------------------------------------------|
//! | `__zyntax_effect_push_handler`      | push a `HandlerFrame` onto the stack     |
//! | `__zyntax_effect_pop_handler`       | pop the matching frame (FIFO check)      |
//! | `__zyntax_effect_lookup_handler`    | find topmost frame for an effect id      |
//! | `__zyntax_effect_resume`            | store value + advance caller state       |
//! | `__zyntax_effect_abort`             | bail out of the handler without resuming |
//!
//! ## Threading
//!
//! The handler stack is `thread_local!`. Each ZynML poll runs on the
//! caller's thread; the runtime doesn't move handler frames between
//! threads. If a future revision introduces a per-fiber stack, switch
//! to a fiber-local cell.

use std::cell::RefCell;

use zyntax_compiler::zrtl::{
    PrimitiveSize, TypeCategory, TypeFlags, TypeTag, ZrtlSigFlags, ZrtlSymbolSig, MAX_PARAMS,
};

// C `free` for releasing state-machine allocations. The JIT-side
// allocation uses `Intrinsic::Malloc` which lowers to libc's
// `malloc`, so the matching deallocator is `free`. Declared as a raw
// extern rather than via the `libc` crate to avoid pulling that
// dependency in for one symbol.
//
// Native targets link against libc directly. On wasm32-unknown-unknown
// there's no libc; the WasmBackend doesn't yet emit `Intrinsic::Malloc`
// either (debt #2 in the wasm work) so the SM-release path is never
// invoked there. A no-op stub keeps the link clean until wasm32 grows
// its own allocator pair.
#[cfg(not(target_arch = "wasm32"))]
extern "C" {
    #[link_name = "free"]
    fn c_free(ptr: *mut core::ffi::c_void);
}

#[cfg(target_arch = "wasm32")]
unsafe fn c_free(ptr: *mut core::ffi::c_void) {
    // wasm32 doesn't have libc free. Pair with the size-header
    // allocation convention `zyntax_wasm::host_alloc_impl` uses:
    // the caller-visible pointer is offset 8 past the underlying
    // allocation; the 8 bytes before it carry the total allocation
    // size used by `Layout::from_size_align`. Recover the layout and
    // hand back to Rust's global allocator.
    //
    // This is safe IFF the SM was allocated with the same convention.
    // Today the only wasm32 path that reaches here is the
    // algebraic-effects refcount drop in `__zyntax_runtime_release_sm`
    // / `release_sm_by_offset`; SM allocation on wasm32 ultimately
    // routes through `host.malloc@1` (WasmBackend emit) or
    // `host_alloc_impl` directly. Both prepend the size header.
    use core::alloc::Layout;
    if ptr.is_null() {
        return;
    }
    let user_ptr = ptr as *mut u8;
    let raw_ptr = user_ptr.sub(8);
    let total = *(raw_ptr as *mut usize);
    if let Ok(layout) = Layout::from_size_align(total, 8) {
        std::alloc::dealloc(raw_ptr, layout);
    }
}

/// One handler in scope: the effect it handles, plus opaque pointers
/// into its closed-over state and operation-dispatch table.
///
/// Both pointers are opaque from the runtime's perspective — the
/// compiled handler code knows the actual layout (Cranelift emitted
/// the loads/stores).
#[derive(Debug, Clone, Copy)]
pub struct HandlerFrame {
    /// `HirId` of the effect this handler implements. Compared against
    /// the `effect_id` arg of `__zyntax_effect_lookup_handler` to find
    /// matches during a perform.
    pub effect_id: u64,
    /// Pointer to the handler's captured state (its fields). Opaque to
    /// the runtime; the handler op fn reinterprets it.
    pub handler_state: *mut u8,
    /// Pointer to the handler's op-table (vtable). The lookup helper
    /// returns this; the perform-site indexes into it to find the
    /// concrete op function.
    pub op_table: *mut u8,
    /// Bit `i` set = this handler's op at index `i` is async (its op fn is
    /// a promise entry the perform site must drive), clear = synchronous.
    /// A compile-time constant passed at the `with`-scope push site, so a
    /// perform under this handler knows whether to drive or call directly
    /// even when different `with` blocks install async and sync handlers
    /// for the same operation.
    pub async_mask: u64,
}

// HandlerFrame is !Send because of the raw pointers, which is fine
// since the stack is thread-local.

thread_local! {
    /// The per-thread handler stack. Pushed by `@with(H)` blocks,
    /// popped at scope exit, walked by `perform`.
    static HANDLER_STACK: RefCell<Vec<HandlerFrame>> = const { RefCell::new(Vec::new()) };
}

/// Push a handler frame onto the per-thread stack.
///
/// Returns the frame ID (= stack index when pushed) so the matching
/// `__zyntax_effect_pop_handler` call can verify it's popping the
/// expected frame. Stack depths in well-formed code are tiny (handlers
/// rarely nest beyond 2–3), so we don't worry about overflow.
#[no_mangle]
pub extern "C" fn __zyntax_effect_push_handler(
    effect_id: u64,
    handler_state: *mut u8,
    op_table: *mut u8,
    async_mask: u64,
) -> u64 {
    HANDLER_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        let frame_id = s.len() as u64;
        s.push(HandlerFrame {
            effect_id,
            handler_state,
            op_table,
            async_mask,
        });
        frame_id
    })
}

/// Whether the active handler for `effect_id`'s op at `op_index` is async.
/// Returns 1 if async, 0 if sync, and -1 if no handler is in scope (the
/// perform site then falls back to its compile-time static default). The
/// perform site uses this to pick the drive-vs-call convention at runtime
/// for an operation with mixed async/sync handlers.
#[no_mangle]
pub extern "C" fn __zyntax_effect_lookup_op_is_async(effect_id: u64, op_index: u64) -> i64 {
    HANDLER_STACK.with(|stack| {
        let s = stack.borrow();
        for frame in s.iter().rev() {
            if frame.effect_id == effect_id {
                return ((frame.async_mask >> op_index) & 1) as i64;
            }
        }
        -1
    })
}

/// Finish a resumable perform whose handler op was dispatched at runtime.
/// `raw` is the value the op fn returned: for an async handler it is a
/// `*Promise` (drive it via `__zyntax_effect_launch_handler` — the perform's
/// result is the handler's return / park value); for a sync handler it is
/// the op result directly. `is_async` (0/1) selects. This keeps the perform
/// site a single straight-line block even for mixed async/sync operations.
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn __zyntax_effect_finish_op(raw: i64, is_async: i64, resume_ptr: *mut u8) -> i64 {
    if is_async != 0 {
        __zyntax_effect_launch_handler(raw as *mut u8, resume_ptr)
    } else {
        raw
    }
}

/// Pop a handler frame.
///
/// `frame_id` is the value the matching `push` returned. If the stack
/// top doesn't match (e.g., the body unwound through a foreign
/// boundary that didn't run `pop`), this still pops whatever is on top
/// — the safer wrong behaviour, since leaving stale frames around
/// breaks subsequent lookups.
///
/// Returns `0` on success, non-zero on detected stack corruption (for
/// future diagnostics; current callers ignore the value).
#[no_mangle]
pub extern "C" fn __zyntax_effect_pop_handler(frame_id: u64) -> u64 {
    HANDLER_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        if s.is_empty() {
            return 1; // stack underflow
        }
        let actual_top = s.len() as u64 - 1;
        s.pop();
        if actual_top == frame_id {
            0
        } else {
            // Frame ID mismatch — non-LIFO pop. Compiler should never
            // emit this for well-formed code; surfacing it as a
            // non-zero return lets future diagnostics catch it.
            2
        }
    })
}

/// Walk the handler stack from the top down, return a pointer to the
/// op_table of the first frame handling `effect_id`. Returns null if
/// no frame matches.
///
/// The returned pointer is borrowed from the live stack — caller must
/// use it before the next `push` or `pop` for the same effect_id
/// scope. In practice the call sequence is:
///
///   1. The compiled poll fn calls `__zyntax_effect_lookup_handler`.
///   2. Loads the op pointer from the returned op_table.
///   3. Calls the op pointer (passing handler_state, args, resume).
///
/// All three steps happen in the same call frame; the stack isn't
/// mutated in between. Safe.
#[no_mangle]
pub extern "C" fn __zyntax_effect_lookup_handler(effect_id: u64) -> *mut u8 {
    HANDLER_STACK.with(|stack| {
        let s = stack.borrow();
        for frame in s.iter().rev() {
            if frame.effect_id == effect_id {
                return frame.op_table;
            }
        }
        core::ptr::null_mut()
    })
}

/// Resolve a single effect operation to its handler function pointer.
/// Walks the handler stack top-down for the innermost frame handling
/// `effect_id`, then reads `op_table[op_index]` (the op-table is a
/// fn-pointer array laid out like a vtable). Returns the raw function
/// address, or null if no handler for this effect is in scope.
///
/// The perform-site codegen uses this to build the dynamic dispatch:
///   `fn_ptr = select(lookup_op(...) != 0, lookup_op(...), static_fn)`
/// so a `with H { }` in scope wins, and otherwise the compile-time
/// static handler is used — no branch, one `call_indirect`.
///
/// # Safety
/// The returned pointer is a code address into JIT-compiled memory
/// held live by the runtime for the program's duration; the caller
/// invokes it as the handler op's function type.
#[no_mangle]
pub extern "C" fn __zyntax_effect_lookup_op(effect_id: u64, op_index: u64) -> *mut u8 {
    HANDLER_STACK.with(|stack| {
        let s = stack.borrow();
        for frame in s.iter().rev() {
            if frame.effect_id == effect_id {
                if frame.op_table.is_null() {
                    return core::ptr::null_mut();
                }
                // op_table is `*mut u8` to an array of pointer-sized
                // function addresses; index by op position.
                let slot = unsafe { (frame.op_table as *const *mut u8).add(op_index as usize) };
                return unsafe { *slot };
            }
        }
        core::ptr::null_mut()
    })
}

/// Resolve the innermost in-scope handler state for `effect_id`. Walks the
/// handler stack top-down and returns the first matching frame's
/// `handler_state` pointer (the `self` region a `with H { }` allocated), or
/// null if no handler for this effect is in scope. The perform-site codegen
/// passes this as the implicit first argument to a stateful handler op.
///
/// # Safety
/// The returned pointer aliases a heap region owned by the enclosing
/// `with` scope; it stays valid until that scope's `pop_handler`.
#[no_mangle]
pub extern "C" fn __zyntax_effect_lookup_state(effect_id: u64) -> *mut u8 {
    HANDLER_STACK.with(|stack| {
        let s = stack.borrow();
        for frame in s.iter().rev() {
            if frame.effect_id == effect_id {
                return frame.handler_state;
            }
        }
        core::ptr::null_mut()
    })
}

thread_local! {
    /// Per-fiber saved handler-stack segments. A fiber shares the one
    /// thread-local `HANDLER_STACK` with its caller (same OS thread), so
    /// any handler it pushes and doesn't pop before yielding would leak
    /// into the caller's view — and truncating the shared stack after a
    /// resume would strand the fiber's own open handlers before it pops
    /// them on the next resume. Instead, each fiber's frames above the
    /// caller's baseline are lifted out on yield/return and re-pushed on
    /// the next resume, giving every fiber its own handler-stack segment
    /// layered on top of whatever was active when it was resumed. Keyed
    /// by fiber pointer (as `usize`).
    static HANDLER_SEGMENTS: RefCell<std::collections::HashMap<usize, Vec<HandlerFrame>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Enter a fiber's handler context just before resuming it. Records the
/// current stack depth as the fiber's baseline, then re-pushes any frames
/// the fiber had open when it last yielded (empty on the first resume).
/// Returns the baseline so the matching `leave` can restore it.
#[no_mangle]
pub extern "C" fn __zyntax_effect_fiber_enter(fiber: *mut u8) -> i64 {
    let baseline = HANDLER_STACK.with(|stack| stack.borrow().len());
    let saved = HANDLER_SEGMENTS.with(|segs| segs.borrow_mut().remove(&(fiber as usize)));
    if let Some(frames) = saved {
        HANDLER_STACK.with(|stack| stack.borrow_mut().extend(frames));
    }
    baseline as i64
}

/// Leave a fiber's handler context just after a resume returns. Lifts the
/// frames the fiber left open (everything above `baseline`) back into its
/// saved segment and truncates the shared stack to `baseline`, so the
/// caller sees exactly the stack it had before the resume.
#[no_mangle]
pub extern "C" fn __zyntax_effect_fiber_leave(fiber: *mut u8, baseline: i64) {
    let base = baseline.max(0) as usize;
    HANDLER_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        if s.len() > base {
            let frames: Vec<HandlerFrame> = s.split_off(base);
            HANDLER_SEGMENTS.with(|segs| {
                segs.borrow_mut().insert(fiber as usize, frames);
            });
        }
    });
}

/// Drop a fiber's saved handler segment. Called when the fiber is freed so
/// a later fiber that reuses the same address can't inherit stale frames.
#[no_mangle]
pub extern "C" fn __zyntax_effect_fiber_forget(fiber: *mut u8) {
    HANDLER_SEGMENTS.with(|segs| {
        segs.borrow_mut().remove(&(fiber as usize));
    });
}

/// Replace the state pointer of every live frame handling `effect_id`
/// — on the current stack, in a fiber's saved segment, or in a parked
/// task's — with whatever `migrate` returns for it. Each distinct
/// region is migrated once, so two frames sharing a region keep
/// sharing it.
///
/// This is how a reload moves handler state into an edited layout: the
/// regions themselves are unreachable from the compiler, but every
/// live one is named by a frame here.
pub(crate) fn migrate_handler_states(
    effect_id: u64,
    mut migrate: impl FnMut(*mut u8) -> Option<*mut u8>,
) -> usize {
    let mut done: std::collections::HashMap<usize, *mut u8> = std::collections::HashMap::new();
    let mut migrated = 0usize;

    let mut visit = |frames: &mut Vec<HandlerFrame>| {
        for frame in frames.iter_mut() {
            if frame.effect_id != effect_id || frame.handler_state.is_null() {
                continue;
            }
            let key = frame.handler_state as usize;
            let replacement = match done.get(&key) {
                Some(p) => Some(*p),
                None => match migrate(frame.handler_state) {
                    Some(p) => {
                        done.insert(key, p);
                        migrated += 1;
                        Some(p)
                    }
                    None => None,
                },
            };
            if let Some(p) = replacement {
                frame.handler_state = p;
            }
        }
    };

    HANDLER_STACK.with(|stack| visit(&mut stack.borrow_mut()));
    HANDLER_SEGMENTS.with(|segs| {
        for frames in segs.borrow_mut().values_mut() {
            visit(frames);
        }
    });
    TASK_HANDLER_SEGMENTS.with(|segs| {
        for frames in segs.borrow_mut().values_mut() {
            visit(frames);
        }
    });
    migrated
}

/// Bind a handler frame into a fiber's saved segment so every
/// enter/resume installs it and every leave saves it back — the
/// persistent counterpart of a per-resume push. Inserted at the
/// segment's bottom: the fiber's own `with` scopes (and earlier
/// binds) sit above it and keep precedence during lookup.
pub(crate) fn fiber_bind_handler(
    fiber: *mut u8,
    effect_id: u64,
    handler_state: *mut u8,
    op_table: *mut u8,
    async_mask: u64,
) {
    HANDLER_SEGMENTS.with(|segs| {
        segs.borrow_mut().entry(fiber as usize).or_default().insert(
            0,
            HandlerFrame {
                effect_id,
                handler_state,
                op_table,
                async_mask,
            },
        );
    });
}

thread_local! {
    /// Per-async-task saved handler-stack segments — the async analogue of
    /// `HANDLER_SEGMENTS` (per fiber). Cooperative tasks share the one
    /// thread-local `HANDLER_STACK`, so a task that opens a `with H { }`
    /// spanning an `await` would leave `H` on the stack while parked,
    /// visible to any other task the executor drives next. The cooperative
    /// executor brackets every task drive with `task_handler_enter` /
    /// `task_handler_leave`, giving each task its own segment layered on the
    /// executor's baseline: a parked task's open handlers live in its
    /// segment, not on the shared stack. Keyed by task id.
    static TASK_HANDLER_SEGMENTS: RefCell<std::collections::HashMap<i64, Vec<HandlerFrame>>> =
        RefCell::new(std::collections::HashMap::new());
}

/// Enter a task's handler context just before driving it: record the
/// current stack depth as the baseline and re-push any frames the task had
/// open when it last parked (empty on the first drive). Returns the
/// baseline for the matching `task_handler_leave`.
pub fn task_handler_enter(task_id: i64) -> usize {
    let baseline = HANDLER_STACK.with(|stack| stack.borrow().len());
    let saved = TASK_HANDLER_SEGMENTS.with(|segs| segs.borrow_mut().remove(&task_id));
    if let Some(frames) = saved {
        HANDLER_STACK.with(|stack| stack.borrow_mut().extend(frames));
    }
    baseline
}

/// Leave a task's handler context just after a drive returns: lift the
/// frames it left open (everything above `baseline`) into its saved
/// segment and truncate the shared stack to `baseline`, so the next task
/// the executor drives can't see this one's open handlers.
pub fn task_handler_leave(task_id: i64, baseline: usize) {
    HANDLER_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        if s.len() > baseline {
            let frames: Vec<HandlerFrame> = s.split_off(baseline);
            TASK_HANDLER_SEGMENTS.with(|segs| {
                segs.borrow_mut().insert(task_id, frames);
            });
        }
    });
}

/// Drop a task's saved handler segment. Called when the task completes or
/// is cancelled — a cancelled task's open `with`-block frames are freed
/// (the `pop_handler` it never reached) rather than lingering.
pub fn task_handler_forget(task_id: i64) {
    TASK_HANDLER_SEGMENTS.with(|segs| {
        segs.borrow_mut().remove(&task_id);
    });
}

/// Current depth of the shared handler stack (diagnostics/tests). Should
/// return to its baseline (0 at top level) between task drives.
pub fn handler_stack_depth() -> usize {
    HANDLER_STACK.with(|stack| stack.borrow().len())
}

/// Whether any task has a saved handler segment (diagnostics/tests). 0 in
/// the borrowed count means every task's `with`-block frames were popped
/// on completion or forgotten on cancel.
pub fn task_handler_segment_count() -> usize {
    TASK_HANDLER_SEGMENTS.with(|segs| segs.borrow().len())
}

/// Layout of the Resume<T> struct, as compiled code constructs it at
/// each perform site (see
/// `krio_adapter::abi_emit::upgrade_resume_struct_at_perform_sites`).
/// 40 bytes, 5 i64-sized fields, `#[repr(C)]` so the field offsets
/// match what the HIR `Store` instructions emit (0, 8, 16, 24, 32).
///
/// `result_slot_offset` is in bytes (already pre-multiplied by 8 in
/// the compiled code), so we can use it directly as a byte offset
/// into `state_machine_ptr`. `next_state` is the dispatcher state-id
/// the caller's Switch routes to on the next poll — `Switch` reads
/// a `u32` from offset 0, so we store the low 32 bits there.
///
/// `refcount_offset` is the byte offset from `state_machine_ptr` to
/// the SM's atomic refcount slot. The host's
/// `__zyntax_runtime_retain_sm` / `release_sm` helpers use it to
/// navigate from a Resume pointer back to the SM's refcount without
/// needing to know the SM's layout. Stored per perform site since
/// the refcount slot's offset depends on the SM's total size, which
/// varies per function.
#[repr(C)]
struct Resume {
    poll_fn_ptr: *const u8,
    state_machine_ptr: *mut u8,
    result_slot_offset: i64,
    next_state: i64,
    refcount_offset: i64,
}

/// Resume the suspended caller with `value`.
///
/// Walks the `Resume<T>` struct passed by the handler:
///   1. Stores `value` at `state_machine_ptr + result_slot_offset`
///      — this is the slot the caller's resume_entry block will
///      `AsyncLoadSlot` from.
///   2. Writes `next_state` (as u32) to the state slot at offset 0
///      — the dispatcher Switch reads this on the next poll.
///   3. Repeatedly calls `poll_fn(state_machine_ptr)` until it
///      returns non-zero (Ready). That non-zero is the value the
///      caller's @effect fn produces — the handler's `k(v)`
///      "returns" this back to the handler body. Synchronous-resume
///      only: an async-out-of-line handler would loop forever here
///      (the case Phase J.3 addresses with a different ABI).
///
/// Returns whatever the recursive poll produces — for single-shot
/// synchronous resume this is the caller's full computation result.
#[no_mangle]
pub extern "C" fn __zyntax_effect_resume(resume_struct: *mut u8, value: i64) -> i64 {
    if resume_struct.is_null() {
        // Defensive: null resume struct means placeholder-ABI Tier 1
        // call still in the codepath. Preserve the old passthrough
        // semantics so any non-resumable handler that happens to call
        // through this symbol gets its value back.
        return value;
    }
    unsafe {
        let r = &*(resume_struct as *const Resume);
        // Store `value` at result_slot_offset within the state machine.
        let result_slot = r.state_machine_ptr.add(r.result_slot_offset as usize) as *mut i64;
        *result_slot = value;
        // Set the dispatcher state slot (offset 0) to next_state. The
        // dispatcher's `AsyncLoadSlot` reads the slot as **i64** (see
        // `krio_adapter::emit::emit_dispatcher` — `ty: HirType::I64`),
        // so we must write the full 8 bytes, not just the low u32.
        // The previous u32-only store left the upper 4 bytes as
        // whatever happened to be in stack memory before the slot was
        // first used; on macOS / Linux those bytes are reliably zero
        // (OS pre-zeros stack pages + the layout doesn't touch them
        // before the first write), so the loaded i64 equalled
        // next_state and Switch matched the expected case. On
        // x86_64-pc-windows-msvc the stack region carries garbage
        // from prior frames — the loaded i64 was
        // `(garbage_high << 32) | next_state`, which matched no
        // Switch case, fell through to `default = resume_entries[0]`,
        // re-ran the perform path, the handler called `k(v)` again,
        // resume → re-poll → re-perform → real stack-growing
        // recursion → STATUS_STACK_OVERFLOW (0xc00000fd) in
        // `phase_i5_breakthrough_real_resume_continuation` on Windows
        // CI. Writing the full i64 zeroes the upper bytes and the
        // dispatcher matches correctly on every platform.
        *(r.state_machine_ptr as *mut i64) = r.next_state;
        // Re-poll the caller's poll fn synchronously until Ready.
        //
        // Guarded by two diagnostic counters so a runaway resume on a
        // platform we haven't manually verified (Windows multi-shot,
        // abort, async-out-of-line) panics with actionable context
        // rather than spinning forever or growing the call stack
        // until SEH stops it. The thresholds are intentionally loose
        // enough that legitimate multi-shot work (J.1: handler calls
        // k three times; J.3: async out-of-line poll-Pending loops
        // until external signal) doesn't trip them.
        //
        //   - LOOP_BUDGET caps the spin inside this one resume call.
        //     A correct poll_fn returns Ready (rc != 0) on the first
        //     iteration for every synchronous resume path; Pending
        //     in a row beyond this budget means the state machine is
        //     stuck in a state that doesn't make progress.
        //
        //   - REENTRY_BUDGET (thread-local) caps total resume
        //     re-entry depth across the chain. Multi-shot legitimately
        //     re-enters once per `k(v)` call; runaway re-entry
        //     (perform → handler → k → resume → re-perform → handler
        //     → …) shows up as depth climbing without bound.
        //
        // Both guards convert "hang" into "panic with state slot,
        // next_state, value, and iteration count" — far more useful
        // CI signal than a 60-second timeout.
        let poll_fn: extern "C" fn(*mut u8) -> i64 = core::mem::transmute(r.poll_fn_ptr);

        const LOOP_BUDGET: u32 = 100_000;
        const REENTRY_BUDGET: u32 = 4096;
        thread_local! {
            static REENTRY_DEPTH: core::cell::Cell<u32> = const { core::cell::Cell::new(0) };
        }

        // RAII guard: decrement depth on every exit path, including
        // panic-unwind, so a panicked test thread that libtest catches
        // doesn't leave a stale counter for the next test on the same
        // thread.
        struct DepthGuard;
        impl Drop for DepthGuard {
            fn drop(&mut self) {
                REENTRY_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            }
        }

        let depth = REENTRY_DEPTH.with(|d| {
            let n = d.get().saturating_add(1);
            d.set(n);
            n
        });
        let _guard = DepthGuard;
        if depth > REENTRY_BUDGET {
            panic!(
                "__zyntax_effect_resume re-entry depth exceeded {} \
                 (state_machine_ptr={:p}, next_state={}, value={}). \
                 Likely an infinite perform→handler→resume→re-perform \
                 cycle — the state machine's dispatcher is not landing \
                 on the resume entry after the resume runtime sets the \
                 state slot. Check (a) the i64 width on every state-slot \
                 read/write across the boundary, (b) the calling \
                 convention used to call poll_fn (must match the JIT'd \
                 function's emitted convention), (c) whether next_state \
                 is a valid Switch case in the dispatcher's layout.",
                REENTRY_BUDGET, r.state_machine_ptr, r.next_state, value
            );
        }

        let mut iter = 0u32;
        loop {
            let rc = poll_fn(r.state_machine_ptr);
            if rc != 0 {
                return rc;
            }
            // Pending. If the resumed caller reached a real suspension
            // point after this resume — an `await` inside the `@effect`
            // fn continuation — its poll fn parked on a future in the
            // timer queue. Rather than busy-spin the poll fn (which never
            // makes progress on its own; the timer queue is what advances
            // the awaited future), cooperatively drive the nearest timer
            // and re-poll. This is the parking-handler substrate: a
            // continuation that awaits no longer burns LOOP_BUDGET and
            // panics — it waits out its timer and resumes. Driving the
            // timer may advance a different task's SM (interleaving); the
            // handler-segment bracket inside `drive_next_timer_with_task`
            // keeps each task's `with`-block state isolated across that.
            #[cfg(not(target_arch = "wasm32"))]
            {
                if crate::host_futures::has_pending_timers() {
                    let _ = crate::host_futures::drive_next_timer_with_task();
                    // The drive may have resolved THIS SM's own awaited future,
                    // polling it to Ready from inside `resolve_future`. If so,
                    // that completion is recorded by SM ptr — take it and
                    // return WITHOUT re-polling. Re-polling a finished SM
                    // re-enters its post-await region and, e.g., re-resumes a
                    // fiber the scope-exit already freed (a UAF abort on the
                    // `perform -> resume -> await -> fiber.next()` composition).
                    if let Some(done) = crate::host_futures::take_sm_completion(r.state_machine_ptr)
                    {
                        return done;
                    }
                    continue;
                }
            }
            // No timer to drive and still Pending: genuine no-progress.
            // Count these toward the budget so a stuck state machine still
            // panics with actionable context (a legitimate await always
            // has a pending timer, so it never reaches this arm).
            iter = iter.saturating_add(1);
            if iter > LOOP_BUDGET {
                panic!(
                    "__zyntax_effect_resume inner poll loop exceeded {} \
                     iterations without Ready and no pending timer to drive \
                     (state_machine_ptr={:p}, next_state={}, value={}, \
                     depth={}). poll_fn keeps returning Pending (rc=0); the \
                     state machine is stuck in a state that never reaches a \
                     Return.",
                    LOOP_BUDGET, r.state_machine_ptr, r.next_state, value, depth
                );
            }
        }
    }
}

/// LAUNCH an ASYNC handler operation's promise so the executor drives it,
/// then return (the performer parks). Called at a resumable perform site
/// when the resolved handler op is async: the handler entry was invoked and
/// returned a `*Promise` `{ state_machine_ptr @ 0, poll_fn_ptr @ 8 }` WITHOUT
/// running the body.
///
/// Polls the handler ONCE under a fresh negative driver task id (so its
/// awaits register timers tagged with that id, which the executor's global
/// timer draining then advances — INTERLEAVED with other tasks). Two paths:
///   * No-await handler: runs `k(v)` inline on this poll → `__zyntax_effect_resume`
///     drives the performer to completion and records it by SM ptr; returns
///     non-zero. The performer's poll then parks (Pending) and the executor
///     harvests the recorded completion instead of re-polling.
///   * Awaiting handler: parks on its own timer and returns 0. The executor
///     drives that timer to `k(v)`, which resumes the performer; the performer
///     completion is likewise recorded by SM ptr.
///
/// Because the executor (not this call) advances the handler's awaits, a
/// handler CAN await a signal only another task satisfies — the tasks
/// interleave. `CURRENT_TASK_ID` is saved/restored and polls are
/// handler-segment-bracketed so a `with`-block across the await stays isolated.
///
/// # Safety
/// `promise_ptr` must be a `*Promise` from an async handler entry; its
/// `state_machine_ptr`/`poll_fn_ptr` must stay live until the handler
/// completes (the entry mallocs them; nothing frees them before Ready).
#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn __zyntax_effect_launch_handler(promise_ptr: *mut u8, resume_ptr: *mut u8) -> i64 {
    if promise_ptr.is_null() {
        return 0;
    }
    unsafe {
        let sm = *(promise_ptr as *const *mut u8); // offset 0
        let poll_fn_ptr = *((promise_ptr as *const u8).add(8) as *const *const u8); // offset 8
        if sm.is_null() || poll_fn_ptr.is_null() {
            return 0;
        }
        let poll_fn: extern "C" fn(*mut u8) -> i64 = core::mem::transmute(poll_fn_ptr);

        // The Resume struct's field 8 is the performer's state_machine_ptr.
        // The handler's RETURN value is the perform's result (matters for
        // multi-shot), so map handler SM → performer SM; whoever completes the
        // handler (inline below, or the executor via resolve_future) records
        // the handler's return as the performer's completion.
        let performer_sm = if resume_ptr.is_null() {
            core::ptr::null_mut()
        } else {
            *((resume_ptr as *const u8).add(8) as *const *mut u8)
        };
        if !performer_sm.is_null() {
            crate::host_futures::register_handler_performer(sm, performer_sm);
        }

        let handler_task = crate::host_futures::alloc_driver_task_id();
        let prev_task = crate::host_futures::current_task_id();
        crate::host_futures::set_current_task_id(handler_task);
        let baseline = task_handler_enter(handler_task);

        // Kick off the handler. If it awaits it parks its own timer (tagged
        // with handler_task) and returns 0 — the executor drives it from here.
        // If it completes inline (no await), route its return to the performer.
        let rc = poll_fn(sm);
        if rc != 0 {
            crate::host_futures::route_handler_completion(sm, rc);
        }

        task_handler_leave(handler_task, baseline);
        crate::host_futures::set_current_task_id(prev_task);
        // NOTE: handler SM + 16-byte promise intentionally not freed yet
        // (memory-safe leak); reclaiming them is a hardening step.
        rc
    }
}

/// Abort the current handler without resuming. The handler's caller
/// observes this as an early return with `value`.
///
/// Same caveat as `__zyntax_effect_resume` — in the placeholder
/// implementation we just return `value`. Future Tier 3 versions will
/// unwind the caller's state machine into a terminal "aborted" state.
#[no_mangle]
pub extern "C" fn __zyntax_effect_abort(value: i64) -> i64 {
    value
}

/// Atomically increment the state machine's refcount, given a `Resume<T>`
/// pointer that lives inside the SM. Hosts that capture a Resume
/// pointer for use AFTER the original `runtime.call_function` returns
/// (the async-out-of-line pattern — see `phase_j3_async_out_of_line_resume`)
/// must call this BEFORE the call returns, otherwise `generate_sync_entry`
/// auto-frees the SM on its return path and the captured pointer
/// dangles. Pair every retain with a `__zyntax_runtime_release_sm`
/// when the host is done with the pointer.
///
/// The `Resume<T>` struct (laid out by
/// `upgrade_resume_struct_at_perform_sites`) carries `state_machine_ptr`
/// at offset 8 and `refcount_offset` at offset 32, so this helper can
/// navigate without needing to know the SM's full layout.
#[no_mangle]
pub unsafe extern "C" fn __zyntax_runtime_retain_sm(resume_struct: *mut u8) {
    if resume_struct.is_null() {
        return;
    }
    use core::sync::atomic::{AtomicI64, Ordering};
    let r = &*(resume_struct as *const Resume);
    if r.state_machine_ptr.is_null() {
        return;
    }
    let refcount_ptr = r.state_machine_ptr.add(r.refcount_offset as usize) as *mut AtomicI64;
    (*refcount_ptr).fetch_add(1, Ordering::AcqRel);
}

/// Atomically decrement the state machine's refcount via a `Resume<T>`
/// pointer; free the SM if the count reaches zero. The complement of
/// `__zyntax_runtime_retain_sm`. Hosts call this when they're done
/// with a previously-retained Resume pointer.
///
/// Safe to call multiple times only if matched by retains — a release
/// past zero would corrupt the refcount and lead to use-after-free.
#[no_mangle]
pub unsafe extern "C" fn __zyntax_runtime_release_sm(resume_struct: *mut u8) {
    if resume_struct.is_null() {
        return;
    }
    use core::sync::atomic::{AtomicI64, Ordering};
    let r = &*(resume_struct as *const Resume);
    if r.state_machine_ptr.is_null() {
        return;
    }
    let refcount_ptr = r.state_machine_ptr.add(r.refcount_offset as usize) as *mut AtomicI64;
    let prev = (*refcount_ptr).fetch_sub(1, Ordering::AcqRel);
    if prev == 1 {
        // Last reference — free the SM. The allocator is whatever
        // `Intrinsic::Malloc` lowers to in the Cranelift backend.
        // Today that's the platform `libc::malloc`; the matching free
        // is `libc::free`.
        c_free(r.state_machine_ptr as *mut core::ffi::c_void);
    }
}

/// JIT-side release: decrement the SM's refcount, free if zero. Same
/// behaviour as `__zyntax_runtime_release_sm` but takes the SM
/// pointer + refcount byte-offset directly rather than navigating
/// through a `Resume<T>` struct. Used by `generate_sync_entry`'s
/// return path where there's no Resume pointer in scope but the
/// JIT has both pieces of info as constants.
#[no_mangle]
pub unsafe extern "C" fn __zyntax_runtime_release_sm_by_offset(
    sm_ptr: *mut u8,
    refcount_offset: i64,
) {
    if sm_ptr.is_null() {
        return;
    }
    use core::sync::atomic::{AtomicI64, Ordering};
    let refcount_ptr = sm_ptr.add(refcount_offset as usize) as *mut AtomicI64;
    let prev = (*refcount_ptr).fetch_sub(1, Ordering::AcqRel);
    if prev == 1 {
        c_free(sm_ptr as *mut core::ffi::c_void);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Cooperative-async host bridges (Phase I)
// ─────────────────────────────────────────────────────────────────────
//
// `__zyntax_async_set_timeout(handle, ms)` is the simplest host
// bridge — kicks off an async wait of `ms` milliseconds and calls
// `host_futures::resolve_future(handle, 0)` when it elapses.
//
// On native: spawns a worker thread that sleeps and resolves.
// `FutureTable` is currently thread-local, so the resolve fires on
// the worker thread; the host_futures `set_complete_task_callback`
// invocation (if any) also fires there. For Phase I.0 the native
// shape exists primarily to register the symbol so the JIT can
// resolve the name; meaningful native async sleep still uses the
// existing tokio-backed `ZyntaxPromise` path.
//
// On wasm32: declared as a Rust-callable wrapper around the
// `host.async_set_timeout` wasm import the JS side provides. The
// JS shim (`web/zynml.mjs::makeHostDispatcher`) wires it to
// `setTimeout(ms, () => _zyntax_resolve_future(handle, 0))`.
//
// Phase I.1 (separate commit) wires the krio_adapter sentinel
// that emits `register_future(...)` + `__zyntax_async_set_timeout`
// at `await __builtin_sleep(...)` sites; Phase I.0 is the bridge
// plumbing only — tests invoke `register_future` + the symbol
// directly without ZynML-level integration.

#[cfg(not(target_arch = "wasm32"))]
#[no_mangle]
pub extern "C" fn __zyntax_async_set_timeout(handle: i64, ms: i64) {
    // Cooperative native path: record a timer for `handle` and return
    // immediately. The state machine's poll then returns Pending (parks)
    // instead of the thread blocking inline. The cooperative executor
    // (the promise driver) sleeps until the nearest deadline and resolves
    // the future, re-driving the parked SM one step. This is the native
    // analogue of the wasm `setTimeout(ms, () => resolve_future(handle))`
    // event-loop bridge and lets multiple tasks interleave: while one task
    // is parked on its timer, the executor can advance another.
    crate::host_futures::schedule_timer(handle, ms);
}

// Wasm32: Rust-side stub. The actual bridge call gets emitted by
// the WasmBackend as a `host.async_set_timeout@2` import in
// JIT-compiled SM modules (Phase I.1 krio sentinel work); the
// Rust-side function exists solely so the symbol can be
// registered with the InterpRuntime's symbol table for JIT
// extern-name resolution.
//
// Today's wasm path doesn't actually call this function — krio
// emits the host import directly at await-site rewrites — so
// this body never runs in production wasm builds. We avoid
// declaring a wasm extern block here so the host import doesn't
// leak into the wasm-bindgen-test runner (which tries to
// `require()` every import module name at test setup time).
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn __zyntax_async_set_timeout(_handle: i64, _ms: i64) {
    // Intentional no-op. The wasm-target call path goes through
    // JIT-emitted `(import "host" "async_set_timeout@2" ...)` —
    // never through this Rust entry point.
}

// ─────────────────────────────────────────────────────────────────────
// Signature constants for register_effect_runtime_symbols
// ─────────────────────────────────────────────────────────────────────

const fn ptr_tag() -> TypeTag {
    TypeTag::new(TypeCategory::Pointer, 0, TypeFlags::NONE)
}

const fn u64_tag() -> TypeTag {
    TypeTag::new(
        TypeCategory::UInt,
        PrimitiveSize::Bits64 as u16,
        TypeFlags::NONE,
    )
}

const fn i32_tag() -> TypeTag {
    TypeTag::new(
        TypeCategory::Int,
        PrimitiveSize::Bits32 as u16,
        TypeFlags::NONE,
    )
}

const fn i64_tag() -> TypeTag {
    TypeTag::new(
        TypeCategory::Int,
        PrimitiveSize::Bits64 as u16,
        TypeFlags::NONE,
    )
}

const fn f32_tag() -> TypeTag {
    TypeTag::new(
        TypeCategory::Float,
        PrimitiveSize::Bits32 as u16,
        TypeFlags::NONE,
    )
}

const fn f64_tag() -> TypeTag {
    TypeTag::new(
        TypeCategory::Float,
        PrimitiveSize::Bits64 as u16,
        TypeFlags::NONE,
    )
}

const fn void_tag() -> TypeTag {
    TypeTag::VOID
}

const fn empty_params() -> [TypeTag; MAX_PARAMS] {
    [TypeTag::VOID; MAX_PARAMS]
}

const fn params1(a: TypeTag) -> [TypeTag; MAX_PARAMS] {
    let mut p = empty_params();
    p[0] = a;
    p
}

const fn params2(a: TypeTag, b: TypeTag) -> [TypeTag; MAX_PARAMS] {
    let mut p = empty_params();
    p[0] = a;
    p[1] = b;
    p
}

const fn params3(a: TypeTag, b: TypeTag, c: TypeTag) -> [TypeTag; MAX_PARAMS] {
    let mut p = empty_params();
    p[0] = a;
    p[1] = b;
    p[2] = c;
    p
}

const fn params4(a: TypeTag, b: TypeTag, c: TypeTag, d: TypeTag) -> [TypeTag; MAX_PARAMS] {
    let mut p = empty_params();
    p[0] = a;
    p[1] = b;
    p[2] = c;
    p[3] = d;
    p
}

#[allow(clippy::too_many_arguments)]
const fn params6(
    a: TypeTag,
    b: TypeTag,
    c: TypeTag,
    d: TypeTag,
    e: TypeTag,
    f: TypeTag,
) -> [TypeTag; MAX_PARAMS] {
    let mut p = empty_params();
    p[0] = a;
    p[1] = b;
    p[2] = c;
    p[3] = d;
    p[4] = e;
    p[5] = f;
    p
}

/// Register the 5 `__zyntax_effect_*` runtime symbols with the
/// runtime's backend.
///
/// Call once at runtime construction (or before the first
/// `compile_typed_program` for a module that uses resumable effects).
/// Idempotent — registering the same name twice is a no-op on the
/// underlying `register_runtime_symbol`.
///
/// Gated on `feature = "native"` because it takes `&mut ZyntaxRuntime`,
/// which only exists in the native build. Parse-only / wasm builds
/// will introduce an equivalent for their interpreter-backed runtime
/// in Phase B.
#[cfg(feature = "native")]
/// Where typed runtime symbols land — both runtimes expose the same
/// registration method.
pub trait TypedSymbolSink {
    fn register_function_typed(&mut self, name: &'static str, ptr: *const u8, sig: ZrtlSymbolSig);

    /// Hand the same table to an interpreter tier, where one exists.
    /// Runtimes that dispatch natively from the start have nothing to
    /// feed and keep the default.
    fn register_zrtl_symbols(&mut self, _symbols: &[zyntax_compiler::zrtl::RuntimeSymbolInfo]) {}
}

impl TypedSymbolSink for crate::runtime::ZyntaxRuntime {
    fn register_function_typed(&mut self, name: &'static str, ptr: *const u8, sig: ZrtlSymbolSig) {
        crate::runtime::ZyntaxRuntime::register_function_typed(self, name, ptr, sig);
    }

    fn register_zrtl_symbols(&mut self, symbols: &[zyntax_compiler::zrtl::RuntimeSymbolInfo]) {
        crate::runtime::ZyntaxRuntime::register_zrtl_symbols(self, symbols);
    }
}

impl TypedSymbolSink for crate::runtime::TieredRuntime {
    fn register_function_typed(&mut self, name: &'static str, ptr: *const u8, sig: ZrtlSymbolSig) {
        crate::runtime::TieredRuntime::register_function_typed(self, name, ptr, sig);
    }
}

pub fn register_effect_runtime_symbols(runtime: &mut impl TypedSymbolSink) {
    // push_handler(effect_id: u64, handler_state: *u8, op_table: *u8, async_mask: u64) -> u64
    runtime.register_function_typed(
        "__zyntax_effect_push_handler",
        __zyntax_effect_push_handler as *const u8,
        ZrtlSymbolSig {
            param_count: 4,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: u64_tag(),
            params: params4(u64_tag(), ptr_tag(), ptr_tag(), u64_tag()),
        },
    );

    // lookup_op_is_async(effect_id: u64, op_index: u64) -> i64 (1/0/-1)
    runtime.register_function_typed(
        "__zyntax_effect_lookup_op_is_async",
        __zyntax_effect_lookup_op_is_async as *const u8,
        ZrtlSymbolSig {
            param_count: 2,
            flags: ZrtlSigFlags::NONE,
            return_type: TypeTag::I64,
            params: params2(u64_tag(), u64_tag()),
        },
    );

    // finish_op(raw: i64, is_async: i64, resume: *u8) -> i64
    #[cfg(not(target_arch = "wasm32"))]
    runtime.register_function_typed(
        "__zyntax_effect_finish_op",
        __zyntax_effect_finish_op as *const u8,
        ZrtlSymbolSig {
            param_count: 3,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::I64,
            params: params3(TypeTag::I64, TypeTag::I64, ptr_tag()),
        },
    );

    // pop_handler(frame_id: u64) -> u64
    runtime.register_function_typed(
        "__zyntax_effect_pop_handler",
        __zyntax_effect_pop_handler as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: u64_tag(),
            params: params1(u64_tag()),
        },
    );

    // lookup_handler(effect_id: u64) -> *u8
    runtime.register_function_typed(
        "__zyntax_effect_lookup_handler",
        __zyntax_effect_lookup_handler as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::NONE,
            return_type: ptr_tag(),
            params: params1(u64_tag()),
        },
    );

    // lookup_op(effect_id: u64, op_index: u64) -> *u8 (fn ptr or null)
    runtime.register_function_typed(
        "__zyntax_effect_lookup_op",
        __zyntax_effect_lookup_op as *const u8,
        ZrtlSymbolSig {
            param_count: 2,
            flags: ZrtlSigFlags::NONE,
            return_type: ptr_tag(),
            params: params2(u64_tag(), u64_tag()),
        },
    );

    // lookup_state(effect_id: u64) -> *u8 (handler `self` region or null)
    runtime.register_function_typed(
        "__zyntax_effect_lookup_state",
        __zyntax_effect_lookup_state as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::NONE,
            return_type: ptr_tag(),
            params: params1(u64_tag()),
        },
    );

    // fiber_enter(fiber: *u8) -> i64 baseline (re-push fiber's saved frames)
    runtime.register_function_typed(
        "__zyntax_effect_fiber_enter",
        __zyntax_effect_fiber_enter as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: i64_tag(),
            params: params1(ptr_tag()),
        },
    );

    // fiber_leave(fiber: *u8, baseline: i64) (lift fiber's frames out)
    runtime.register_function_typed(
        "__zyntax_effect_fiber_leave",
        __zyntax_effect_fiber_leave as *const u8,
        ZrtlSymbolSig {
            param_count: 2,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: void_tag(),
            params: params2(ptr_tag(), i64_tag()),
        },
    );

    // fiber_forget(fiber: *u8) (drop saved segment on free)
    runtime.register_function_typed(
        "__zyntax_effect_fiber_forget",
        __zyntax_effect_fiber_forget as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: void_tag(),
            params: params1(ptr_tag()),
        },
    );

    // resume(resume_struct: *u8, value: i64) -> i64
    runtime.register_function_typed(
        "__zyntax_effect_resume",
        __zyntax_effect_resume as *const u8,
        ZrtlSymbolSig {
            param_count: 2,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::I64,
            params: params2(ptr_tag(), TypeTag::I64),
        },
    );

    // launch_handler(promise: *u8) -> i64 — kick off an async handler op's
    // promise so the executor drives it; the performer parks. Native-only.
    #[cfg(not(target_arch = "wasm32"))]
    runtime.register_function_typed(
        "__zyntax_effect_launch_handler",
        __zyntax_effect_launch_handler as *const u8,
        ZrtlSymbolSig {
            param_count: 2,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::I64,
            params: params2(ptr_tag(), ptr_tag()),
        },
    );

    // abort(value: i64) -> i64
    runtime.register_function_typed(
        "__zyntax_effect_abort",
        __zyntax_effect_abort as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::I64,
            params: params1(TypeTag::I64),
        },
    );

    // retain_sm(resume_struct: *u8) -> void
    runtime.register_function_typed(
        "__zyntax_runtime_retain_sm",
        __zyntax_runtime_retain_sm as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::VOID,
            params: params1(ptr_tag()),
        },
    );

    // release_sm(resume_struct: *u8) -> void
    runtime.register_function_typed(
        "__zyntax_runtime_release_sm",
        __zyntax_runtime_release_sm as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::VOID,
            params: params1(ptr_tag()),
        },
    );

    // release_sm_by_offset(sm_ptr: *u8, refcount_offset: i64) -> void
    // JIT-side variant; sync_entry's return path calls this.
    runtime.register_function_typed(
        "__zyntax_runtime_release_sm_by_offset",
        __zyntax_runtime_release_sm_by_offset as *const u8,
        ZrtlSymbolSig {
            param_count: 2,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::VOID,
            params: params2(ptr_tag(), TypeTag::I64),
        },
    );

    // async_set_timeout(handle: i64, ms: i64) -> void
    // Phase I.0 — native side spawns a worker thread that sleeps
    // and calls resolve_future. The JIT side emits a call to this
    // symbol; the krio_adapter sentinel work (Phase I.1) handles
    // the SM context for the matching register_future call.
    runtime.register_function_typed(
        "__zyntax_async_set_timeout",
        __zyntax_async_set_timeout as *const u8,
        ZrtlSymbolSig {
            param_count: 2,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::VOID,
            params: params2(TypeTag::I64, TypeTag::I64),
        },
    );

    // register_future(poll_fn_ptr, sm_ptr, result_offset,
    //                 next_state, refcount_offset, task_id) -> i64
    // The krio_adapter sentinel emit (Phase I.1) calls this
    // immediately before issuing the host bridge call. Returns
    // the i64 future handle the bridge passes to JS so JS can
    // call _zyntax_resolve_future when its Promise settles.
    runtime.register_function_typed(
        "__zyntax_register_future",
        crate::host_futures::__zyntax_register_future as *const u8,
        ZrtlSymbolSig {
            param_count: 6,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: TypeTag::I64,
            params: params6(
                ptr_tag(),
                ptr_tag(),
                TypeTag::I64,
                TypeTag::I64,
                TypeTag::I64,
                TypeTag::I64,
            ),
        },
    );
}

/// Return the `(name, ptr, sig)` table for every `zyntax_box_*`
/// helper the SSA-lowering autobox / autounbox path can emit. The
/// list is consumed by `register_box_runtime_symbols` (mirrors to
/// Cranelift backend + BC interp) and again at LLVM JIT install
/// time so the LLVM backend sees the same per-symbol return / param
/// shapes. Without LLVM-side signatures the backend defaults each
/// `Call::Symbol` to `void(args)` returning `i32 0` and crashes the
/// first time an unboxed value is consumed as anything other than
/// `i32`.
#[cfg(feature = "native")]
pub fn box_runtime_symbol_infos() -> Vec<zyntax_compiler::zrtl::RuntimeSymbolInfo> {
    use zyntax_compiler::zrtl::{RuntimeSymbolInfo, ZrtlSigFlags, ZrtlSymbolSig};
    let no_flags = ZrtlSigFlags::NONE;
    let entries: &[(&'static str, *const u8, ZrtlSymbolSig)] = &[
        (
            "zyntax_box_i32",
            zyntax_compiler::zrtl::zyntax_box_i32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(i32_tag()),
            },
        ),
        (
            "zyntax_box_i64",
            zyntax_compiler::zrtl::zyntax_box_i64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(i64_tag()),
            },
        ),
        (
            "zyntax_box_f32",
            zyntax_compiler::zrtl::zyntax_box_f32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(f32_tag()),
            },
        ),
        (
            "zyntax_box_f64",
            zyntax_compiler::zrtl::zyntax_box_f64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(f64_tag()),
            },
        ),
        (
            "zyntax_box_bool",
            zyntax_compiler::zrtl::zyntax_box_bool as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(i32_tag()),
            },
        ),
        (
            "zyntax_box_free",
            zyntax_compiler::zrtl::zyntax_box_free as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: void_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_i32",
            zyntax_compiler::zrtl::zyntax_box_get_i32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i32_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_i64",
            zyntax_compiler::zrtl::zyntax_box_get_i64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i64_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_f32",
            zyntax_compiler::zrtl::zyntax_box_get_f32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: f32_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_f64",
            zyntax_compiler::zrtl::zyntax_box_get_f64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: f64_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_bool",
            zyntax_compiler::zrtl::zyntax_box_get_bool as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i32_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_tag",
            zyntax_compiler::zrtl::zyntax_box_get_tag as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i32_tag(),
                params: params1(ptr_tag()),
            },
        ),
    ];
    entries
        .iter()
        .map(|(name, ptr, sig)| RuntimeSymbolInfo {
            name,
            ptr: *ptr,
            sig: Some(*sig),
        })
        .collect()
}

/// Register the `zyntax_box_*` runtime helpers used by the SSA
/// lowering's autobox / autounbox path for `Type::Any` field stores
/// and reads. Without this, modules that touch the autobox code path
/// hit "unknown function 'zyntax_box_X'" at the BC interp tier and
/// type-mismatch panics at the LLVM tier (the JIT defaults to
/// `i32 0` for unresolved-signature symbols, then crashes when an
/// unboxed value is consumed at a different type).
///
/// Mirrors `register_effect_runtime_symbols` in shape; called from
/// `ZyntaxRuntime::with_config` and `with_symbols`.
#[cfg(feature = "native")]
pub fn register_box_runtime_symbols(runtime: &mut crate::runtime::ZyntaxRuntime) {
    use zyntax_compiler::zrtl::{RuntimeSymbolInfo, ZrtlSigFlags, ZrtlSymbolSig};
    let no_flags = ZrtlSigFlags::NONE;

    // Collect each (name, ptr, sig) tuple so we can hand the same
    // set to the BC interp via `register_zrtl_symbols` after wiring
    // the backend. `register_function_typed` registers with the
    // backend + plugin signature table; it does NOT mirror into the
    // interp's FFI table, so the interp would otherwise miss them.
    let mut infos: Vec<RuntimeSymbolInfo> = Vec::new();

    // Box constructors: take a primitive, return a *mut DynamicBox
    // (pointer-sized — `ptr_tag()` in the runtime FFI surface).
    runtime.register_function_typed(
        "zyntax_box_i32",
        zyntax_compiler::zrtl::zyntax_box_i32 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: ptr_tag(),
            params: params1(i32_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_i64",
        zyntax_compiler::zrtl::zyntax_box_i64 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: ptr_tag(),
            params: params1(i64_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_f32",
        zyntax_compiler::zrtl::zyntax_box_f32 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: ptr_tag(),
            params: params1(f32_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_f64",
        zyntax_compiler::zrtl::zyntax_box_f64 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: ptr_tag(),
            params: params1(f64_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_bool",
        zyntax_compiler::zrtl::zyntax_box_bool as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: ptr_tag(),
            params: params1(i32_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_free",
        zyntax_compiler::zrtl::zyntax_box_free as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: void_tag(),
            params: params1(ptr_tag()),
        },
    );

    // Box getters: take a *mut DynamicBox, return primitive.
    runtime.register_function_typed(
        "zyntax_box_get_i32",
        zyntax_compiler::zrtl::zyntax_box_get_i32 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: i32_tag(),
            params: params1(ptr_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_get_i64",
        zyntax_compiler::zrtl::zyntax_box_get_i64 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: i64_tag(),
            params: params1(ptr_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_get_f32",
        zyntax_compiler::zrtl::zyntax_box_get_f32 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: f32_tag(),
            params: params1(ptr_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_get_f64",
        zyntax_compiler::zrtl::zyntax_box_get_f64 as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: f64_tag(),
            params: params1(ptr_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_get_bool",
        zyntax_compiler::zrtl::zyntax_box_get_bool as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: i32_tag(),
            params: params1(ptr_tag()),
        },
    );
    runtime.register_function_typed(
        "zyntax_box_get_tag",
        zyntax_compiler::zrtl::zyntax_box_get_tag as *const u8,
        ZrtlSymbolSig {
            param_count: 1,
            flags: no_flags,
            return_type: i32_tag(),
            params: params1(ptr_tag()),
        },
    );

    // Mirror to the BC interp's FFI table — `register_function_typed`
    // only touches the backend + plugin-signature side. Without this
    // the interp dispatch hits "unknown function 'zyntax_box_X'"
    // the moment a hot path resolves through it.
    let names_and_sigs: &[(&'static str, *const u8, ZrtlSymbolSig)] = &[
        (
            "zyntax_box_i32",
            zyntax_compiler::zrtl::zyntax_box_i32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(i32_tag()),
            },
        ),
        (
            "zyntax_box_i64",
            zyntax_compiler::zrtl::zyntax_box_i64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(i64_tag()),
            },
        ),
        (
            "zyntax_box_f32",
            zyntax_compiler::zrtl::zyntax_box_f32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(f32_tag()),
            },
        ),
        (
            "zyntax_box_f64",
            zyntax_compiler::zrtl::zyntax_box_f64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(f64_tag()),
            },
        ),
        (
            "zyntax_box_bool",
            zyntax_compiler::zrtl::zyntax_box_bool as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params1(i32_tag()),
            },
        ),
        (
            "zyntax_box_free",
            zyntax_compiler::zrtl::zyntax_box_free as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: void_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_i32",
            zyntax_compiler::zrtl::zyntax_box_get_i32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i32_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_i64",
            zyntax_compiler::zrtl::zyntax_box_get_i64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i64_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_f32",
            zyntax_compiler::zrtl::zyntax_box_get_f32 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: f32_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_f64",
            zyntax_compiler::zrtl::zyntax_box_get_f64 as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: f64_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_bool",
            zyntax_compiler::zrtl::zyntax_box_get_bool as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i32_tag(),
                params: params1(ptr_tag()),
            },
        ),
        (
            "zyntax_box_get_tag",
            zyntax_compiler::zrtl::zyntax_box_get_tag as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i32_tag(),
                params: params1(ptr_tag()),
            },
        ),
    ];
    for (name, ptr, sig) in names_and_sigs {
        infos.push(RuntimeSymbolInfo {
            name,
            ptr: *ptr,
            sig: Some(*sig),
        });
    }
    runtime.register_zrtl_symbols(&infos);
}

/// Build the typed-FFI signatures for the `krio_fiber_*` symbol
/// family. Returned in the same `RuntimeSymbolInfo` shape as the box
/// helpers; callers feed it to both the JIT backend and the BC
/// interp's `register_zrtl_symbols`.
///
/// Scaffold: the function pointers are the panic-on-call stubs in
/// `zyntax_compiler::zrtl::krio_fiber_*`. A future commit swaps them
/// for the real krio-fiber backend impls behind the `FiberCfg`
/// trait — same symbol names, same signatures, working bodies.
pub fn fiber_runtime_symbol_infos() -> Vec<zyntax_compiler::zrtl::RuntimeSymbolInfo> {
    use zyntax_compiler::zrtl::{RuntimeSymbolInfo, ZrtlSigFlags, ZrtlSymbolSig};
    let no_flags = ZrtlSigFlags::NONE;

    let entries: &[(&'static str, *const u8, ZrtlSymbolSig)] = &[
        // krio_fiber_new(closure: ptr, stack_size: i64) -> ptr
        (
            "krio_fiber_new",
            zyntax_compiler::zrtl::krio_fiber_new as *const u8,
            ZrtlSymbolSig {
                param_count: 2,
                flags: no_flags,
                return_type: ptr_tag(),
                params: params2(ptr_tag(), i64_tag()),
            },
        ),
        // krio_fiber_resume(fiber: ptr) -> i64 (FiberStep tag)
        (
            "krio_fiber_resume",
            zyntax_compiler::zrtl::krio_fiber_resume as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i64_tag(),
                params: params1(ptr_tag()),
            },
        ),
        // krio_fiber_resume_with(fiber: ptr, value: i64) -> i64
        (
            "krio_fiber_resume_with",
            zyntax_compiler::zrtl::krio_fiber_resume_with as *const u8,
            ZrtlSymbolSig {
                param_count: 2,
                flags: no_flags,
                return_type: i64_tag(),
                params: params2(ptr_tag(), i64_tag()),
            },
        ),
        // krio_fiber_yield(value: i64) -> void
        (
            "krio_fiber_yield",
            zyntax_compiler::zrtl::krio_fiber_yield as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: void_tag(),
                params: params1(i64_tag()),
            },
        ),
        // krio_fiber_take_input() -> i64
        (
            "krio_fiber_take_input",
            zyntax_compiler::zrtl::krio_fiber_take_input as *const u8,
            ZrtlSymbolSig {
                param_count: 0,
                flags: no_flags,
                return_type: i64_tag(),
                params: [zyntax_compiler::zrtl::TypeTag::VOID; 16],
            },
        ),
        // krio_fiber_transfer(target: ptr, value: i64) -> i64
        (
            "krio_fiber_transfer",
            zyntax_compiler::zrtl::krio_fiber_transfer as *const u8,
            ZrtlSymbolSig {
                param_count: 2,
                flags: no_flags,
                return_type: i64_tag(),
                params: params2(ptr_tag(), i64_tag()),
            },
        ),
        // krio_fiber_cancel(fiber: ptr) -> void
        (
            "krio_fiber_cancel",
            zyntax_compiler::zrtl::krio_fiber_cancel as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: void_tag(),
                params: params1(ptr_tag()),
            },
        ),
        // krio_fiber_free(fiber: ptr) -> void
        (
            "krio_fiber_free",
            zyntax_compiler::zrtl::krio_fiber_free as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: void_tag(),
                params: params1(ptr_tag()),
            },
        ),
        // krio_fiber_abort_with(variant: i64, payload: i64) -> void
        (
            "krio_fiber_abort_with",
            zyntax_compiler::zrtl::krio_fiber_abort_with as *const u8,
            ZrtlSymbolSig {
                param_count: 2,
                flags: no_flags,
                return_type: void_tag(),
                params: params2(i64_tag(), i64_tag()),
            },
        ),
        // krio_fiber_take_error(fiber: ptr) -> i64 (packed)
        (
            "krio_fiber_take_error",
            zyntax_compiler::zrtl::krio_fiber_take_error as *const u8,
            ZrtlSymbolSig {
                param_count: 1,
                flags: no_flags,
                return_type: i64_tag(),
                params: params1(ptr_tag()),
            },
        ),
    ];

    entries
        .iter()
        .map(|(name, ptr, sig)| RuntimeSymbolInfo {
            name,
            ptr: *ptr,
            sig: Some(*sig),
        })
        .collect()
}

/// Register the fiber runtime symbols on `runtime`'s Cranelift /
/// LLVM backends + BC interp. Mirror of
/// [`register_box_runtime_symbols`].
///
/// Scaffold: the registered fn pointers panic on call (see
/// [`fiber_runtime_symbol_infos`] notes). Lowering passes can still
/// emit `Call::Symbol("krio_fiber_*")` and link cleanly; the panic
/// fires only if a fiber op actually executes before the real
/// backend lands.
///
/// Native-only: it registers symbols on the native `ZyntaxRuntime`
/// (`crate::runtime`, gated behind `native`); the wasm/parse-only build
/// never constructs that runtime.
#[cfg(feature = "native")]
pub fn register_fiber_runtime_symbols(runtime: &mut impl TypedSymbolSink) {
    let infos = fiber_runtime_symbol_infos();
    for info in &infos {
        runtime.register_function_typed(info.name, info.ptr, info.sig.unwrap());
    }
    runtime.register_zrtl_symbols(&infos);
}

/// Test-only helper: reset the per-thread handler stack to empty.
/// Tests that exercise push/pop/lookup should call this at the start
/// to isolate from leftover state when running in a shared test
/// thread.
#[cfg(test)]
pub fn reset_handler_stack_for_test() {
    HANDLER_STACK.with(|stack| stack.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ptr(value: usize) -> *mut u8 {
        value as *mut u8
    }

    #[test]
    fn push_pop_round_trip_returns_frame_ids() {
        reset_handler_stack_for_test();
        let id_a = __zyntax_effect_push_handler(1, dummy_ptr(0xaa), dummy_ptr(0xab), 0);
        let id_b = __zyntax_effect_push_handler(2, dummy_ptr(0xba), dummy_ptr(0xbb), 0);
        assert_eq!(id_a, 0, "first push gets frame id 0");
        assert_eq!(id_b, 1, "second push gets frame id 1");
        // LIFO pop: B first, then A.
        assert_eq!(__zyntax_effect_pop_handler(id_b), 0);
        assert_eq!(__zyntax_effect_pop_handler(id_a), 0);
    }

    #[test]
    fn pop_on_empty_stack_returns_error_sentinel() {
        reset_handler_stack_for_test();
        assert_eq!(
            __zyntax_effect_pop_handler(0),
            1,
            "underflow surfaced as return 1"
        );
    }

    #[test]
    fn lookup_returns_topmost_matching_op_table() {
        reset_handler_stack_for_test();
        let outer_op_table = dummy_ptr(0x100);
        let inner_op_table = dummy_ptr(0x200);
        // Both handlers serve effect_id = 42.
        let outer = __zyntax_effect_push_handler(42, dummy_ptr(0xaa), outer_op_table, 0);
        let inner = __zyntax_effect_push_handler(42, dummy_ptr(0xbb), inner_op_table, 0);

        // Walks top-down → inner wins.
        let found = __zyntax_effect_lookup_handler(42);
        assert_eq!(
            found, inner_op_table,
            "innermost @with(H) for the effect must win the dynamic dispatch"
        );

        // Pop the inner — outer is now topmost.
        let rc = __zyntax_effect_pop_handler(inner);
        assert_eq!(rc, 0);
        let found = __zyntax_effect_lookup_handler(42);
        assert_eq!(
            found, outer_op_table,
            "after popping inner, lookup falls through to outer"
        );

        let rc = __zyntax_effect_pop_handler(outer);
        assert_eq!(rc, 0);
    }

    #[test]
    fn lookup_returns_null_when_no_handler_in_scope() {
        reset_handler_stack_for_test();
        // No frame for effect 99.
        __zyntax_effect_push_handler(1, dummy_ptr(0), dummy_ptr(0), 0);
        let found = __zyntax_effect_lookup_handler(99);
        assert!(
            found.is_null(),
            "unhandled effect yields null op_table (runtime trap is the caller's responsibility)"
        );
        __zyntax_effect_pop_handler(0);
    }

    #[test]
    fn resume_and_abort_passthrough_value() {
        // Placeholder behaviour for the initial milestone — the real
        // Resume<T> machinery is a follow-up. Verify the symbols at
        // least flow the value through so calls don't lose data.
        assert_eq!(__zyntax_effect_resume(core::ptr::null_mut(), 42), 42);
        assert_eq!(__zyntax_effect_abort(-7), -7);
    }
}
