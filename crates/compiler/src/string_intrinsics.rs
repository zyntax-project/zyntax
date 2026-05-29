//! Runtime intrinsics for string operations called by JIT'd code.
//!
//! Strings in the Zyntax ABI use the inline length-prefixed layout
//! `[i32 length][utf8_bytes...]`. Pointer-equality on string operands
//! only succeeds when both refer to the same allocation, which would
//! make `"a" == "a"` return false whenever the two literals are
//! distinct data symbols. The Cranelift backend's `BinaryOp::Eq` /
//! `Ne` paths detect `Ptr(I8)` operands and emit a call to
//! [`zrtl_string_equals`] instead — that walks the length headers and
//! compares the byte payload.
//!
//! Registered as a JIT runtime symbol via [`string_runtime_symbols`]
//! alongside [`crate::osr::osr_runtime_symbols`].

/// Compare two ZRTL strings for equality. Returns `1` for equal, `0`
/// otherwise. `i32` return type (not bool) so the Cranelift backend
/// can keep the comparison result in the integer value-map without an
/// extra widen.
///
/// # Safety
///
/// Both pointers must be either null or point at a valid ZRTL string
/// header (`[i32 length][utf8_bytes...]`). Passing a non-string
/// pointer triggers undefined behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zrtl_string_equals(a: *const i32, b: *const i32) -> i32 {
    if a == b {
        return 1;
    }
    if a.is_null() || b.is_null() {
        return 0;
    }
    let len_a = unsafe { *a };
    let len_b = unsafe { *b };
    if len_a != len_b {
        return 0;
    }
    if len_a == 0 {
        return 1;
    }
    let data_a = unsafe { (a as *const u8).add(std::mem::size_of::<i32>()) };
    let data_b = unsafe { (b as *const u8).add(std::mem::size_of::<i32>()) };
    let slice_a = unsafe { std::slice::from_raw_parts(data_a, len_a as usize) };
    let slice_b = unsafe { std::slice::from_raw_parts(data_b, len_b as usize) };
    if slice_a == slice_b {
        1
    } else {
        0
    }
}

/// `(name, function_pointer)` pairs to feed
/// `CraneliftBackend::with_runtime_symbols` so generated code can
/// resolve string runtime intrinsics at JIT link time. Mirror of
/// [`crate::osr::osr_runtime_symbols`] for string ops.
pub fn string_runtime_symbols() -> [(&'static str, *const u8); 1] {
    [("zrtl_string_equals", zrtl_string_equals as *const u8)]
}
