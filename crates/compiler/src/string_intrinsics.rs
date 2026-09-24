//! Runtime intrinsics for string operations called by JIT'd code.
//!
//! Strings in the Zyntax ABI use the ZRTL SDK layout (see
//! `zrtl::string`). Pointer-equality on string operands only
//! succeeds when both refer to the same allocation, which would make
//! `"a" == "a"` return false whenever the two literals are distinct
//! data symbols. The Cranelift backend's `BinaryOp::Eq` / `Ne` paths
//! detect `Ptr(I8)` operands and emit a call to
//! [`zrtl_string_equals`] instead, which compares the bytes.
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
/// Both pointers must be either null or point at a valid ZRTL string.
/// Passing a non-string pointer triggers undefined behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn zrtl_string_equals(a: *const i32, b: *const i32) -> i32 {
    unsafe { ::zrtl::string::string_equals(a, b) as i32 }
}

/// `(name, function_pointer)` pairs to feed
/// `CraneliftBackend::with_runtime_symbols` so generated code can
/// resolve string runtime intrinsics at JIT link time. Mirror of
/// [`crate::osr::osr_runtime_symbols`] for string ops.
pub fn string_runtime_symbols() -> [(&'static str, *const u8); 1] {
    [("zrtl_string_equals", zrtl_string_equals as *const u8)]
}
