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
) -> u64 {
    HANDLER_STACK.with(|stack| {
        let mut s = stack.borrow_mut();
        let frame_id = s.len() as u64;
        s.push(HandlerFrame {
            effect_id,
            handler_state,
            op_table,
        });
        frame_id
    })
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
/// Walks the Resume<T> struct passed by the handler:
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
            iter = iter.saturating_add(1);
            if iter > LOOP_BUDGET {
                panic!(
                    "__zyntax_effect_resume inner poll loop exceeded {} \
                     iterations without Ready (state_machine_ptr={:p}, \
                     next_state={}, value={}, depth={}). poll_fn keeps \
                     returning Pending (rc=0); the state machine is stuck \
                     in a state that never reaches a Return.",
                    LOOP_BUDGET, r.state_machine_ptr, r.next_state, value, depth
                );
            }
            let rc = poll_fn(r.state_machine_ptr);
            if rc != 0 {
                return rc;
            }
        }
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

/// Atomically increment the state machine's refcount, given a Resume<T>
/// pointer that lives inside the SM. Hosts that capture a Resume
/// pointer for use AFTER the original `runtime.call_function` returns
/// (the async-out-of-line pattern — see `phase_j3_async_out_of_line_resume`)
/// must call this BEFORE the call returns, otherwise `generate_sync_entry`
/// auto-frees the SM on its return path and the captured pointer
/// dangles. Pair every retain with a `__zyntax_runtime_release_sm`
/// when the host is done with the pointer.
///
/// The Resume<T> struct (laid out by
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

/// Atomically decrement the state machine's refcount via a Resume<T>
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
/// through a Resume<T> struct. Used by `generate_sync_entry`'s
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
    // Worker thread does the sleep; resolve_future fires on the
    // worker thread. FutureTable is thread-local so the resolve
    // only takes effect if the caller's thread is the one that
    // also called register_future. For Phase I.0 native tests
    // we don't actually invoke this — we'd need cross-thread
    // FutureTable visibility for that to make sense. The
    // worker-thread path here exists for symmetry with the wasm
    // side: callers that own a single-thread runtime (most
    // embedders) can use it as a real async timer; multi-thread
    // callers should route through the tokio-backed runtime
    // instead.
    let ms = ms.max(0) as u64;
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(ms));
        // Best-effort resolve. If the FutureTable doesn't have
        // the handle (different thread), this returns
        // UnknownHandle and the SM never wakes up — caller's
        // responsibility to ensure the resolve thread is the
        // same as the register thread.
        crate::host_futures::resolve_future(handle, 0);
    });
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
pub fn register_effect_runtime_symbols(runtime: &mut crate::runtime::ZyntaxRuntime) {
    // push_handler(effect_id: u64, handler_state: *u8, op_table: *u8) -> u64
    runtime.register_function_typed(
        "__zyntax_effect_push_handler",
        __zyntax_effect_push_handler as *const u8,
        ZrtlSymbolSig {
            param_count: 3,
            flags: ZrtlSigFlags::EFFECTFUL,
            return_type: u64_tag(),
            params: params3(u64_tag(), ptr_tag(), ptr_tag()),
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
        let id_a = __zyntax_effect_push_handler(1, dummy_ptr(0xaa), dummy_ptr(0xab));
        let id_b = __zyntax_effect_push_handler(2, dummy_ptr(0xba), dummy_ptr(0xbb));
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
        let outer = __zyntax_effect_push_handler(42, dummy_ptr(0xaa), outer_op_table);
        let inner = __zyntax_effect_push_handler(42, dummy_ptr(0xbb), inner_op_table);

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
        __zyntax_effect_push_handler(1, dummy_ptr(0), dummy_ptr(0));
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
