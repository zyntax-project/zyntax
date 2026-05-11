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

/// Resume the suspended caller with `value`.
///
/// In the full Tier 3 design, this would:
///   1. Decode `resume_struct` to find the caller's state machine,
///      next-state, and result-slot offsets.
///   2. Store `value` at the result slot.
///   3. Set the state slot to next-state.
///   4. Return to the caller's poll loop.
///
/// For the initial runtime-symbols milestone, `resume_struct` is
/// opaque and this is a passthrough that returns the value — the
/// Cranelift backend's M4 lowering keeps the PerformEffect direct
/// dispatch path for now. Wired up in a future iteration when
/// `lower_perform_effect_calls` emits real Resume<T> structs.
#[no_mangle]
pub extern "C" fn __zyntax_effect_resume(resume_struct: *mut u8, value: i64) -> i64 {
    let _ = resume_struct;
    value
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

/// Register the 5 `__zyntax_effect_*` runtime symbols with the
/// runtime's backend.
///
/// Call once at runtime construction (or before the first
/// `compile_typed_program` for a module that uses resumable effects).
/// Idempotent — registering the same name twice is a no-op on the
/// underlying `register_runtime_symbol`.
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
